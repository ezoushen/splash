// Weight images shared between processes: engine processes of this binary
// load synthetic models through loadModel, with or without a SharedImages
// source, and report where each image came from and what it holds. Each
// loader's images are tested so: the target's from MLX, Model Optimizer and
// compressed-tensors checkpoints and from GGUFs, the draft's and the vision
// tower's from MLX checkpoints and a GGUF mmproj. So is the lifecycle of
// the images in memory: release and restore under an admission, a holder that
// is killed or stopped, and a writer that is refused memory.
//
//   shared-weights PRODUCTION_METALLIB OTHER_METALLIB [TEST]
//
// With a TEST, such as "gguf target keys", only the tests whose name contains
// it, which each test names as it begins.
#include "GgufFixtures.hpp"
#include "TestAdmission.hpp"
#include "TestChecks.hpp"
#include "TestCheckpoint.hpp"
#include "TestFiles.hpp"
#include "metal/MetalBackend.hpp"
#include "model/DraftCheckpoint.hpp"
#include "model/GgufPreparation.hpp"
#include "model/GgufTarget.hpp"
#include "model/ModelFactory.hpp"
#include "model/SafetensorsImage.hpp"
#include "model/SafetensorsTarget.hpp"
#include "model/SharedImages.hpp"
#include "model/VisionLoader.hpp"
#include "model/WeightImages.hpp"
#include "model/WeightStore.hpp"

#include <fcntl.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <signal.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

extern char **environ;

namespace {

using splash::test::require;
using splash::test::SyntheticTensor;
namespace model = splash::model;
namespace metal = splash::metal;

// ---- The synthetic model ----

// A MoE target of small layouts, as model_loading_test's: every projection in
// whole 256-row tiles, 16 experts rather than the family's 256.
model::Qwen3_6MoeLayout syntheticTarget() {
  model::Qwen3_6MoeLayout target;
  target.layers = 4;
  target.hiddenSize = 256;
  target.vocabularySize = 256;
  target.convolutionDimension = 512;
  target.gdnKeyHeads = 2;
  target.gdnValueHeads = 4;
  target.gdnHeadDimension = 64;
  target.attentionWidth = 256;
  target.packedGdnWidth = 1024;
  target.attentionQueryHeads = 4;
  target.attentionKvHeads = 4;
  target.attentionHeadDimension = 64;
  target.packedFullWidth = 1024;
  target.experts = 16;
  target.expertIntermediateSize = 256;
  target.fullAttentionPeriod = 4;
  return target;
}

model::DFlashDraftLayout syntheticDraft(const model::Qwen3_6MoeLayout &target) {
  model::DFlashDraftLayout draft;
  draft.layers = 2;
  // Its hidden size and vocabulary are its target's.
  draft.hiddenSize = target.hiddenSize;
  draft.vocabularySize = target.vocabularySize;
  draft.dynamicSize = 256;
  draft.qkvSize = 512;
  draft.attentionSize = 256;
  draft.intermediateSize = 256;
  draft.attentionHeadDimension = 64;
  draft.rotaryTheta = 10'000'000.0F;
  draft.selectorRank = 256;
  draft.kvHeads = 2;
  // It reads a GDN layer and the attention layer.
  draft.targetLayers = {1, 3};
  draft.targetLayerCount = 2;
  return draft;
}

// A descriptor states a vision layout even for a model without a vision tower.
splash::ops::VisionLayout syntheticVision() {
  splash::ops::VisionLayout vision;
  vision.depth = 2;
  vision.hiddenSize = 128;
  vision.patchDimension = 1536;
  vision.intermediateSize = 200;
  vision.paddedIntermediateSize = 256;
  vision.mergedHiddenSize = 512;
  vision.outputHiddenSize = 256;
  vision.heads = 2;
  vision.headDimension = 64;
  vision.positionGridSide = 4;
  return vision;
}

// count bytes of a pattern seeded by seed.
std::vector<uint8_t> pattern(size_t count, uint32_t seed) {
  std::vector<uint8_t> bytes(count);
  for (size_t i = 0; i < count; ++i) bytes[i] = static_cast<uint8_t>((i * 2654435761u + seed) >> 13);
  return bytes;
}

size_t widthOf(const std::string &dtype) {
  return dtype == "U32" || dtype == "F32" ? 4 : dtype == "U8" || dtype == "F8_E4M3" ? 1 : 2;
}

// count values of dtype in a pattern seeded by seed. Every float is finite
// and exactly a BF16, as a tower's F16 and F32 values must be, and the
// values of a scale are positive.
std::vector<uint8_t> valuesOf(const std::string &dtype, uint64_t count, uint32_t seed, bool scales = false) {
  const size_t width = widthOf(dtype);
  std::vector<uint8_t> bytes = pattern(count * width, seed);
  for (size_t at = 0; at < bytes.size(); at += width) {
    if (dtype == "BF16") {
      bytes[at + 1] = 0x3c | (bytes[at + 1] & 3);
    } else if (dtype == "F16") {
      bytes[at] &= 0xf8;
      bytes[at + 1] = 0x38 | (bytes[at + 1] & 3);
    } else if (dtype == "F32") {
      bytes[at] = bytes[at + 1] = 0;
      bytes[at + 3] = 0x3c | (bytes[at + 3] & 3);
    } else if (dtype == "F8_E4M3") {
      bytes[at] = 0x20 | (bytes[at] & 0x3f);
    } else if (dtype == "U8" && scales) {
      bytes[at] = 0x38 | (bytes[at] & 7);
    }
  }
  return bytes;
}

// Gives each tensor values of a pattern of its own: a tensor of codes holds
// them as they come, and the others as valuesOf says.
void fill(std::vector<SyntheticTensor> &tensors, uint32_t seed) {
  for (SyntheticTensor &tensor : tensors) {
    uint64_t count = 1;
    for (uint64_t dimension : tensor.shape) count *= dimension;
    const bool scales = tensor.name.ends_with(".scales") || tensor.name.ends_with(".weight_scale");
    tensor.data = valuesOf(tensor.dtype, count, seed++, scales);
  }
}

// The checkpoint's tensors: every quantized module's codes hold a pattern of
// their own, so that no two images hold alike, and its scales and biases are
// the BF16 values 0.0625 and 0.03125, so that its codes decode to nonzero values.
std::vector<SyntheticTensor> targetTensors() {
  std::vector<SyntheticTensor> tensors = splash::test::mlxTargetTensors(
      syntheticTarget(), [](const std::string &module) {
        return module.ends_with(".gate") || module.ends_with("shared_expert_gate") ? std::pair{8u, 64u}
                                                                                    : std::pair{4u, 64u};
      });
  uint32_t seed = 1;
  for (SyntheticTensor &tensor : tensors) {
    const bool codes = tensor.dtype == "U32", scales = tensor.name.ends_with(".scales"),
               biases = tensor.name.ends_with(".biases");
    if (!codes && !scales && !biases) continue;
    uint64_t bytes = codes ? 4 : 2;
    for (uint64_t dimension : tensor.shape) bytes *= dimension;
    if (codes) {
      tensor.data = pattern(bytes, seed++);
    } else {
      tensor.data.resize(bytes);
      for (uint64_t i = 0; i < bytes; i += 2) {
        tensor.data[i] = 0x80;
        tensor.data[i + 1] = scales ? 0x3d : 0x3c;
      }
    }
  }
  return tensors;
}

// An MLX target in every format MLX writes: affine in 2 to 8 bits and in
// groups of 32, 64 and 128, mxfp4 and nvfp4, a router that is 8-bit affine,
// a shared-expert gate in F32 and a token table in BF16.
std::vector<SyntheticTensor> formatTensors() {
  const std::map<std::string_view, std::pair<uint32_t, uint32_t>> formats{
      {".linear_attn.in_proj_qkv", {3, 32}},  {".linear_attn.in_proj_z", {5, 128}},
      {".linear_attn.in_proj_b", {4, 64}},    {".linear_attn.in_proj_a", {8, 64}},
      {".linear_attn.out_proj", {6, 64}},     {".self_attn.q_proj", {2, 64}},
      {".self_attn.k_proj", {8, 32}},         {".self_attn.v_proj", {4, 128}},
      {".self_attn.o_proj", {4, 32}},         {".mlp.gate", {8, 64}},
      {".mlp.shared_expert_gate", {8, 64}},   {".switch_mlp.gate_proj", {5, 32}},
      {".switch_mlp.up_proj", {2, 128}},      {".switch_mlp.down_proj", {4, 32}},
      {".shared_expert.gate_proj", {4, 16}},  {".shared_expert.up_proj", {8, 32}},
      {".shared_expert.down_proj", {3, 128}}, {"language_model.lm_head", {4, 16}},
      {"language_model.model.embed_tokens", {6, 32}}};
  const auto format = [&](const std::string &module) {
    for (const auto &[suffix, bitsAndGroup] : formats)
      if (module.ends_with(suffix)) return bitsAndGroup;
    throw std::runtime_error("no format for " + module);
  };
  std::vector<SyntheticTensor> tensors = splash::test::mlxTargetTensors(syntheticTarget(), format);
  // mxfp4 (o_proj and the routed down projection) and nvfp4 (the shared
  // expert's gate and the head) have U8 scales and no biases; the shared-expert
  // gate and the token table are unquantized.
  const auto is = [](const SyntheticTensor &tensor, std::string_view module, std::string_view tensorName) {
    return tensor.name.ends_with(std::string(module) + "." + std::string(tensorName));
  };
  std::erase_if(tensors, [&](const SyntheticTensor &tensor) {
    for (const std::string_view module : {".self_attn.o_proj", ".switch_mlp.down_proj", ".shared_expert.gate_proj",
                                          "language_model.lm_head"})
      if (is(tensor, module, "biases")) return true;
    for (const std::string_view module : {".mlp.shared_expert_gate", "language_model.model.embed_tokens"})
      if (is(tensor, module, "scales") || is(tensor, module, "biases")) return true;
    return false;
  });
  for (SyntheticTensor &tensor : tensors) {
    for (const std::string_view module : {".self_attn.o_proj", ".switch_mlp.down_proj", ".shared_expert.gate_proj",
                                          "language_model.lm_head"})
      if (is(tensor, module, "scales")) tensor.dtype = "U8";
    if (is(tensor, ".mlp.shared_expert_gate", "weight")) tensor = {tensor.name, "F32", {1, 256}};
    if (is(tensor, "language_model.model.embed_tokens", "weight")) tensor = {tensor.name, "BF16", {256, 256}};
  }
  fill(tensors, 1);
  return tensors;
}

// A target of NVFP4 and FP8 modules, as Model Optimizer or compressed-tensors
// stores them.
std::vector<SyntheticTensor> floatTensors(splash::test::FloatCheckpoint checkpoint) {
  const auto format = [](const std::string &module) -> uint32_t {
    return module.ends_with("o_proj") || module.ends_with("out_proj") || module == "lm_head" ||
                   module.find(".shared_expert.") != std::string::npos
               ? GGUF_FMT_FP8
               : GGUF_FMT_NVFP4;
  };
  std::vector<SyntheticTensor> tensors = splash::test::floatTargetTensors(syntheticTarget(), format, checkpoint);
  fill(tensors, 1);
  return tensors;
}

// ---- The GGUF targets ----

namespace ggml = splash::model::ggml;

// A GGUF target of every projection in the types, which name a tensor with or
// without its "blk.N." prefix (gguf_fixtures::TensorTypes). The routed
// experts' gate and up tensors of the layers in fused are one tensor, and
// every NVFP4 tensor of a layer in scaled, and a fused one, has a .scale
// tensor: one F32 value, or one per expert for the routed experts'.
std::vector<gguf_fixtures::Tensor> ggufTensors(const gguf_fixtures::TensorTypes &types, std::vector<uint32_t> fused,
                                               std::vector<uint32_t> scaled) {
  const model::Qwen3_6MoeLayout target = syntheticTarget();
  std::vector<gguf_fixtures::Tensor> tensors = gguf_fixtures::targetTensors(target, types);
  uint32_t seed = 100;
  for (gguf_fixtures::Tensor &tensor : tensors) {
    uint64_t rows = 1;
    for (size_t i = 1; i < tensor.dims.size(); ++i) rows *= tensor.dims[i];
    const uint64_t count = rows * tensor.dims[0];
    const uint32_t format = gguf_format_of(tensor.type);
    ++seed;
    if (tensor.type == ggml::kNVFP4)
      tensor.data = gguf_fixtures::blockNvfp4Fixture(uint32_t(rows), uint32_t(tensor.dims[0]), seed);
    else if (tensor.type == ggml::kF16)
      tensor.data = valuesOf("F16", count, seed);
    else if (tensor.type == ggml::kBF16)
      tensor.data = valuesOf("BF16", count, seed);
    else if (format != GGUF_FMT_COUNT)
      tensor.data = gguf_fixtures::fixture(gguf_reference::Fmt(format), uint32_t(rows), uint32_t(tensor.dims[0]), seed);
    else
      tensor.data = gguf_fixtures::floatValues(count, seed, gguf_fixtures::readAsBfloat16(tensor));
  }
  for (const uint32_t layer : fused)
    tensors = gguf_fixtures::withFusedGateUp(std::move(tensors), layer, target.experts);
  std::vector<gguf_fixtures::Tensor> scales;
  for (const gguf_fixtures::Tensor &tensor : tensors) {
    const size_t dot = tensor.name.find('.', 4);
    if (tensor.type != ggml::kNVFP4 || !tensor.name.starts_with("blk.") || !tensor.name.ends_with(".weight")) continue;
    const uint32_t layer = uint32_t(std::stoul(tensor.name.substr(4, dot - 4)));
    const bool fusedHere = tensor.name.find("ffn_gate_up_exps") != std::string::npos;
    if (!fusedHere && std::find(scaled.begin(), scaled.end(), layer) == scaled.end()) continue;
    const uint64_t values = tensor.name.find("_exps.") != std::string::npos ? target.experts : 1;
    scales.push_back({tensor.name.substr(0, tensor.name.size() - 7) + ".scale", {values}, ggml::kF32,
                      valuesOf("F32", values, ++seed, true)});
  }
  tensors.insert(tensors.end(), std::make_move_iterator(scales.begin()), std::make_move_iterator(scales.end()));
  return tensors;
}

// The GGUF target of K-quants, IQ-quants and Q8_0 and 16-bit tensors, in
// layers of every alpha/beta (one type, two types, F16) and routed experts
// that are fused (layers 1 and 3) or not.
std::vector<gguf_fixtures::Tensor> formatGgufTensors() {
  using namespace gguf_fixtures;
  return ggufTensors({{"attn_qkv.weight", kQ4_K},
                      {"attn_gate.weight", kQ5_K},
                      {"ssm_beta.weight", kQ8_0},
                      {"ssm_alpha.weight", kQ8_0},
                      {"ssm_out.weight", kIQ4_XS},
                      {"attn_q.weight", kQ6_K},
                      {"attn_k.weight", kQ8_0},
                      {"attn_v.weight", kQ4_K},
                      {"attn_output.weight", kQ5_K},
                      {"ffn_gate_exps.weight", kQ4_K},
                      {"ffn_up_exps.weight", kQ4_K},
                      {"ffn_down_exps.weight", kQ5_K},
                      {"ffn_gate_shexp.weight", kQ8_0},
                      {"ffn_up_shexp.weight", kQ8_0},
                      {"ffn_down_shexp.weight", kQ8_0},
                      {"output.weight", kQ6_K},
                      {"token_embd.weight", ggml::kBF16},
                      {"blk.1.attn_qkv.weight", kQ6_K},
                      {"blk.1.ssm_alpha.weight", kQ4_K},
                      {"blk.2.ssm_beta.weight", ggml::kF16},
                      {"blk.2.ssm_alpha.weight", ggml::kF16},
                      {"blk.1.ffn_gate_exps.weight", kQ3_K},
                      {"blk.1.ffn_up_exps.weight", kQ3_K},
                      {"blk.2.ffn_up_exps.weight", kQ6_K},
                      {"blk.0.ffn_down_shexp.weight", kIQ4_NL}},
                     {1, 3}, {});
}

// The GGUF target of llama.cpp's NVFP4 in every projection, with a token table
// in F16, the tensor scales of layers 0 and 2 and of the fused experts of layers 1 and 3.
std::vector<gguf_fixtures::Tensor> nvfp4GgufTensors() {
  gguf_fixtures::TensorTypes types{{"token_embd.weight", ggml::kF16}};
  for (const char *name :
       {"attn_qkv", "attn_gate", "ssm_beta", "ssm_alpha", "ssm_out", "attn_q", "attn_k", "attn_v", "attn_output",
        "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"})
    types[std::string(name) + ".weight"] = ggml::kNVFP4;
  types["output.weight"] = ggml::kNVFP4;
  return ggufTensors(types, {1, 3}, {0, 2});
}

void writeGgufTarget(const std::filesystem::path &root, std::vector<gguf_fixtures::Tensor> tensors, bool reversed) {
  if (reversed) std::reverse(tensors.begin(), tensors.end());
  std::filesystem::create_directories(root / "target");
  gguf_fixtures::writeGguf(root / "target" / "model.gguf", tensors, syntheticTarget());
}

// ---- The vision towers ----

// The dtype of the vision tensor at index: BF16, F16 and F32 in turn.
std::string visionDtype(size_t index) { return index % 3 == 0 ? "BF16" : index % 3 == 1 ? "F16" : "F32"; }

std::vector<SyntheticTensor> visionMlxTensors() {
  std::vector<SyntheticTensor> tensors = splash::test::visionTensors(syntheticVision());
  for (size_t index = 0; index < tensors.size(); ++index) tensors[index].dtype = visionDtype(index);
  fill(tensors, 700);
  return tensors;
}

// The tower of visionMlxTensors as a clip qwen3vl_merger mmproj holds it: the
// patch embedding's weight one tensor per frame, and every matrix's dimensions
// the other way round.
void writeMmproj(const std::filesystem::path &path, bool reversed) {
  const splash::ops::VisionLayout layout = syntheticVision();
  std::vector<gguf_fixtures::Tensor> tensors;
  uint32_t seed = 900;
  const auto add = [&](const std::string &name, std::vector<uint64_t> dims) {
    const std::string dtype = visionDtype(tensors.size());
    uint64_t count = 1;
    for (uint64_t dimension : dims) count *= dimension;
    tensors.push_back({name, std::move(dims), dtype == "BF16" ? ggml::kBF16 : dtype == "F16" ? ggml::kF16 : ggml::kF32,
                       valuesOf(dtype, count, ++seed)});
  };
  const uint64_t hidden = layout.hiddenSize, patch = layout.patchSize;
  const auto affine = [&](const std::string &name, uint64_t rows, uint64_t columns) {
    add(name + ".weight", {columns, rows});
    add(name + ".bias", {rows});
  };
  const auto norm = [&](const std::string &name) {
    add(name + ".weight", {hidden});
    add(name + ".bias", {hidden});
  };
  add("v.patch_embd.weight", {patch, patch, 3, hidden});
  add("v.patch_embd.weight.1", {patch, patch, 3, hidden});
  add("v.patch_embd.bias", {hidden});
  add("v.position_embd.weight", {hidden, uint64_t{layout.positionGridSide} * layout.positionGridSide});
  for (uint32_t block = 0; block < layout.depth; ++block) {
    const std::string at = "v.blk." + std::to_string(block) + ".";
    norm(at + "ln1");
    affine(at + "attn_qkv", 3 * hidden, hidden);
    affine(at + "attn_out", hidden, hidden);
    norm(at + "ln2");
    affine(at + "ffn_up", layout.intermediateSize, hidden);
    affine(at + "ffn_down", hidden, layout.intermediateSize);
  }
  norm("v.post_ln");
  affine("mm.0", layout.mergedHiddenSize, layout.mergedHiddenSize);
  affine("mm.2", layout.outputHiddenSize, layout.mergedHiddenSize);
  if (reversed) std::reverse(tensors.begin(), tensors.end());
  namespace gguf = splash::test::gguf;
  const std::vector<gguf::Key> keys{
      gguf::stringKey("general.architecture", "clip"),
      gguf::stringKey("clip.projector_type", "qwen3vl_merger"),
      gguf::uint32Key("clip.vision.projection_dim", layout.outputHiddenSize),
      gguf::uint32Key("clip.vision.patch_size", layout.patchSize),
      gguf::uint32Key("clip.vision.embedding_length", layout.hiddenSize),
      gguf::uint32Key("clip.vision.feed_forward_length", layout.intermediateSize),
      gguf::uint32Key("clip.vision.block_count", layout.depth),
      gguf::uint32Key("clip.vision.attention.head_count", layout.heads),
      gguf::uint32Key("clip.vision.spatial_merge_size", layout.spatialMerge),
      gguf::boolKey("clip.use_gelu", true),
      gguf::float32Key("clip.vision.attention.layer_norm_epsilon", 1e-6F),
      gguf::float32ArrayKey("clip.vision.image_mean", {0.5F, 0.5F, 0.5F}),
      gguf::float32ArrayKey("clip.vision.image_std", {0.5F, 0.5F, 0.5F}),
      gguf::int32ArrayKey("clip.vision.is_deepstack_layers", std::vector<int32_t>(layout.depth, 0))};
  std::filesystem::create_directories(path.parent_path());
  splash::test::writeFile(path, gguf::file(keys, tensors));
}

// ---- The kinds of model ----

// What an installed model holds, a directory for each.
enum class Role { Target, Draft, Vision };

const char *directoryOf(Role role) { return role == Role::Target ? "target" : role == Role::Draft ? "draft" : "vision"; }

// The images of each role in a model: the target's each layer's and its
// experts', the head's and the embedding's; the draft's each layer's and
// model.bin; the tower's one.
size_t imageCount(Role role) {
  return role == Role::Target  ? 2 * syntheticTarget().layers + 2
         : role == Role::Draft ? syntheticDraft(syntheticTarget()).layers + 1
                               : 1;
}

void writeShard(const std::filesystem::path &path, std::vector<SyntheticTensor> tensors, bool reversed) {
  if (reversed) std::reverse(tensors.begin(), tensors.end());
  splash::test::writeSyntheticShard(path, tensors);
}

void writeDraft(const std::filesystem::path &root, bool reversed) {
  std::vector<SyntheticTensor> tensors = splash::test::draftTensors(syntheticDraft(syntheticTarget()));
  fill(tensors, 3000);
  writeShard(root / "draft" / "model.safetensors", std::move(tensors), reversed);
}

// A kind of installed model: where its target and its vision tower come from,
// and a writer of one, whose tensors lie in the other order when `reversed`.
struct Kind final {
  std::string name;
  model::TargetSource target = model::TargetSource::Safetensors;
  model::VisionSource vision = model::VisionSource::None;
  std::function<void(const std::filesystem::path &, bool)> write;
};

const std::vector<Kind> &kinds() {
  static const std::vector<Kind> all = [] {
    const auto safetensors = [](std::function<std::vector<SyntheticTensor>()> tensors) {
      return [tensors](const std::filesystem::path &root, bool reversed) {
        writeShard(root / "target" / "model.safetensors", tensors(), reversed);
        writeDraft(root, reversed);
      };
    };
    const auto gguf = [](std::function<std::vector<gguf_fixtures::Tensor>()> tensors) {
      return [tensors](const std::filesystem::path &root, bool reversed) {
        writeGgufTarget(root, tensors(), reversed);
        writeDraft(root, reversed);
      };
    };
    using splash::test::FloatCheckpoint;
    std::vector<Kind> kinds;
    kinds.push_back({"mlx", model::TargetSource::Safetensors, model::VisionSource::None, safetensors(targetTensors)});
    kinds.push_back({"mlx-formats", model::TargetSource::Safetensors, model::VisionSource::None,
                     safetensors(formatTensors)});
    kinds.push_back({"model-optimizer", model::TargetSource::Safetensors, model::VisionSource::None,
                     safetensors([] { return floatTensors(FloatCheckpoint::ModelOptimizer); })});
    kinds.push_back({"compressed-tensors", model::TargetSource::Safetensors, model::VisionSource::None,
                     safetensors([] { return floatTensors(FloatCheckpoint::CompressedTensors); })});
    kinds.push_back({"gguf", model::TargetSource::Gguf, model::VisionSource::None, gguf(formatGgufTensors)});
    kinds.push_back({"gguf-nvfp4", model::TargetSource::Gguf, model::VisionSource::None, gguf(nvfp4GgufTensors)});
    kinds.push_back({"vision-mlx", model::TargetSource::Safetensors, model::VisionSource::Safetensors,
                     [](const std::filesystem::path &root, bool reversed) {
                       writeShard(root / "target" / "model.safetensors", targetTensors(), reversed);
                       writeDraft(root, reversed);
                       writeShard(root / "vision" / "model.safetensors", visionMlxTensors(), reversed);
                     }});
    kinds.push_back({"vision-mmproj", model::TargetSource::Safetensors, model::VisionSource::Gguf,
                     [](const std::filesystem::path &root, bool reversed) {
                       writeShard(root / "target" / "model.safetensors", targetTensors(), reversed);
                       writeDraft(root, reversed);
                       writeMmproj(root / "vision" / "mmproj.gguf", reversed);
                     }});
    return kinds;
  }();
  return all;
}

const Kind &kindNamed(const std::string &name) {
  for (const Kind &kind : kinds())
    if (kind.name == name) return kind;
  throw std::runtime_error("no kind of model named " + name);
}

model::ModelDescriptor descriptorOf(const Kind &kind) {
  const model::Qwen3_6MoeLayout target = syntheticTarget();
  model::ModelDescriptor descriptor = model::makeModelDescriptor(
      "Qwen MoE sharing oracle", target, syntheticDraft(target), syntheticVision(), kind.target, kind.vision);
  descriptor.sourceIdentity = "sources";
  return descriptor;
}

// Changes the codes of the module `name`.
void retune(std::vector<SyntheticTensor> &tensors, const std::string &name) {
  for (SyntheticTensor &tensor : tensors)
    if (tensor.name == name + ".weight") {
      for (uint8_t &byte : tensor.data) byte = static_cast<uint8_t>(~byte);
      return;
    }
  require(false, "a synthetic checkpoint has no module " + name);
}

void writeCheckpoint(const std::filesystem::path &root, const std::vector<SyntheticTensor> &tensors) {
  splash::test::writeSyntheticShard(root / "target" / "model.safetensors", tensors);
  writeDraft(root, false);
}

// ---- The engine process ----

// Whether this process may write the memory at address.
bool writable(const void *address) {
  mach_vm_address_t at = reinterpret_cast<mach_vm_address_t>(address);
  mach_vm_size_t size = 0;
  vm_region_basic_info_data_64_t info{};
  mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
  mach_port_t object = MACH_PORT_NULL;
  require(mach_vm_region(mach_task_self(), &at, &size, VM_REGION_BASIC_INFO_64,
                         reinterpret_cast<vm_region_info_t>(&info), &count, &object) == KERN_SUCCESS,
          "cannot inspect an image's memory");
  return info.protection & VM_PROT_WRITE;
}

const char *originName(model::ImageOrigin origin) {
  switch (origin) {
  case model::ImageOrigin::Private: return "private";
  case model::ImageOrigin::Received: return "received";
  case model::ImageOrigin::Written: return "written";
  }
  return "unknown";
}

// The images this process wrote itself through copyImage, and what a writer
// does as it begins its next image: nothing, stops (SIGSTOP), or is refused
// memory, as a GPU writer is when the driver runs short.
unsigned gWrites = 0;
enum class NextWrite { Normal, Stop, OutOfMemory };
NextWrite gNextWrite = NextWrite::Normal;

constexpr std::string_view kCopyMagic = "TEST0001";

// An image of a 16 KiB header block, then bytes [offset, offset + bytes) of
// file, padded to 16 KiB: what a loader's writer would plan, with no model.
model::ImagePlan copyImage(std::shared_ptr<model::WeightSource> file, std::string component, uint64_t offset,
                           uint64_t bytes) {
  const uint64_t imageBytes = model::kWeightFileAlignment + model::alignWeightOffset(bytes);
  model::ImagePlan plan{std::move(component), std::string(kCopyMagic), 0, 0, imageBytes,
                        [file, offset, bytes](std::span<uint8_t> image, const metal::MetalBuffer &) {
                          ++gWrites;
                          const NextWrite next = std::exchange(gNextWrite, NextWrite::Normal);
                          if (next == NextWrite::Stop) raise(SIGSTOP);
                          if (next == NextWrite::OutOfMemory) throw metal::MetalAllocationError("writer out of memory");
                          std::memset(image.data(), 0, image.size());
                          const auto header = model::weightFileHeader(kCopyMagic, 0, 0);
                          std::memcpy(image.data(), header.data(), header.size());
                          file->readData(offset, image.subspan(model::kWeightFileAlignment, bytes));
                          file->checkUnchanged();
                        }};
  plan.description = "copy " + std::to_string(bytes);
  plan.sources = {{file.get(), offset, bytes}};
  return plan;
}

// The reply that lists each image's origin and the SHA-256 of its bytes.
std::string imagesReply(const model::WeightImages &images) {
  std::string reply = "images";
  for (const auto &image : images.contents())
    reply += std::string(" ") + std::string(image.component) + "=" + originName(image.origin) + ":" +
             (image.bytes.empty() ? "released" : model::weightDigest(image.bytes));
  return reply;
}

// An admission as the memory governor's: it admits `limit` images, then
// refuses, and gives the cause of an allocation the driver refused. The sizes
// it was asked for are in admitted.
struct Admission final {
  unsigned limit = std::numeric_limits<unsigned>::max();
  std::vector<uint64_t> admitted;
  metal::AllocationResult operator()(uint64_t bytes, const std::function<void()> &allocate) {
    if (admitted.size() >= limit) return metal::AllocationFailure::HostPressure;
    admitted.push_back(bytes);
    try {
      allocate();
    } catch (const metal::MetalAllocationError &error) {
      return error.failure();
    }
    return {};
  }
};

// Restores images, as the engine does one image a tick, until they are all
// back or the admission refuses one: then "refused" and the cause.
std::string restoreReply(model::WeightImages &images, Admission &admit) {
  try {
    while (!images.restore(std::ref(admit))) {
    }
  } catch (const metal::MetalAllocationError &refusal) {
    return std::string("refused ") + refusal.what();
  }
  return imagesReply(images);
}

// Arguments: METALLIB ROOT REGISTRY DIGESTS on|off|unrecorded. Every file in
// ROOT is a source, recorded by its SHA-256 as the installer records a file,
// but with unrecorded. "load NAME KIND" loads the model NAME under ROOT of
// the kind of model KIND; "release", "restore [IMAGES]" (an admission that
// refuses after IMAGES images), "contents", "sizes", "admitted" and
// "allocated" are about its images. "copy ..." does the same with images that
// copy bytes of a file, whose next write can stop or run out of memory.
int runEngine(char **argv) {
  metal::MetalBackend backend(argv[2]);
  std::cout << "ready" << std::endl;
  const std::filesystem::path root = argv[3];
  // Sharing begins with the first command, as a server's begins when it
  // loads its model, some time after it started.
  std::string line;
  if (!std::getline(std::cin, line)) return 0;
  std::shared_ptr<model::SharedImages> shared;
  if (std::string_view(argv[6]) != "off") {
    const bool recorded = std::string_view(argv[6]) == "on";
    model::SharedImagesConfig config;
    config.registry = argv[4];
    config.digestTables = argv[5];
    config.writerIdentity = model::writerIdentity(backend);
    config.writeWait = std::chrono::seconds(1);
    config.modelRoot = root;
    for (const auto &entry : std::filesystem::recursive_directory_iterator(root))
      if (recorded && entry.is_regular_file())
        config.sourceDigests[entry.path().lexically_relative(root).generic_string()] =
            model::weightDigest(splash::test::readFile(entry.path()));
    shared = std::make_shared<model::SharedImages>(std::move(config));
  }
  std::vector<model::LoadedModel> models;
  // Images of copyImage's, which a test makes fail as it likes.
  model::WeightImages copies(backend, "sources", shared);
  std::map<std::string, std::shared_ptr<model::WeightSource>> sources;
  Admission admission;
  do {
    std::istringstream command(line);
    std::string verb, name, kind;
    command >> verb >> name >> kind;
    if (verb == "load" || verb == "release" || verb == "restore" || verb == "contents") {
      if (verb == "load")
        models.push_back(model::loadModel(backend, root / name, descriptorOf(kindNamed(kind)), {}, shared));
      else if (verb == "release") models.back().images->release();
      else if (verb == "restore") {
        // After name images (all, with none) the admission refuses.
        admission = Admission{};
        if (!name.empty()) admission.limit = unsigned(std::stoul(name));
        const std::string reply = restoreReply(*models.back().images, admission);
        std::cout << reply << std::endl;
        continue;
      }
      // The images' origin and the SHA-256 of their bytes, empty while
      // released, by component.
      std::cout << imagesReply(*models.back().images) << std::endl;
    } else if (verb == "admitted") {
      // The sizes the last restore asked its admission for, and the sizes
      // of the images, in load order, as they are held.
      std::cout << "admitted";
      for (uint64_t bytes : admission.admitted) std::cout << ' ' << bytes;
      std::cout << std::endl;
    } else if (verb == "sizes") {
      std::cout << "sizes";
      for (const auto &image : models.back().images->contents()) std::cout << ' ' << image.bytes.size();
      std::cout << std::endl;
    } else if (verb == "allocated") {
      std::cout << "allocated " << backend.memoryStats().allocatedBytes << std::endl;
    } else if (verb == "copy") {
      // copy load FILE OFFSET BYTES, copy release, copy restore, copy
      // contents, and copy fail or copy stop for the next write.
      if (name == "load") {
        uint64_t offset = 0, bytes = 0;
        command >> offset >> bytes;
        auto &source = sources[kind];
        if (!source) source = std::make_shared<model::WeightSource>(root / kind);
        const unsigned before = gWrites;
        const model::WeightFile loaded =
            copies.load(copyImage(source, "test/copy-" + std::to_string(copies.contents().size()) + ".bin", offset, bytes));
        static_cast<void>(loaded);
        std::cout << "loaded " << gWrites - before << ' ' << imagesReply(copies) << std::endl;
      } else if (name == "release") {
        copies.release();
        std::cout << imagesReply(copies) << std::endl;
      } else if (name == "restore") {
        admission = Admission{};
        std::cout << restoreReply(copies, admission) << std::endl;
      } else if (name == "contents") {
        std::cout << imagesReply(copies) << std::endl;
      } else if (name == "writes") {
        std::cout << "writes " << gWrites << std::endl;
      } else if (name == "fail" || name == "stop") {
        gNextWrite = name == "fail" ? NextWrite::OutOfMemory : NextWrite::Stop;
        std::cout << "next write " << name << std::endl;
      } else {
        std::cout << "unknown command " << verb << ' ' << name << std::endl;
      }
    } else if (verb == "identity") {
      // The writers' identity as this process runs them, or as its executable
      // file gives it.
      std::cout << "identity "
                << (name == "file" ? model::writerIdentity(argv[0], backend) : model::writerIdentity(backend))
                << std::endl;
    } else if (verb == "writable") {
      unsigned count = 0;
      for (const auto &image : models.back().images->contents()) count += writable(image.bytes.data());
      std::cout << "writable " << count << std::endl;
    } else {
      std::cout << "unknown command " << verb << std::endl;
    }
  } while (std::getline(std::cin, line));
  return 0;
}

// ---- The tests ----

// The engine processes running, which a test that waits for ever kills as it
// fails (watchdog), so that none is left behind.
pid_t gEngines[64];
std::atomic<size_t> gEngineCount = 0;

void watchdog(int) {
  for (size_t i = 0; i < gEngineCount; ++i)
    if (gEngines[i] > 0) kill(gEngines[i], SIGKILL);
  const char message[] = "FAIL: a test waited for ever\n";
  (void)write(STDERR_FILENO, message, sizeof(message) - 1);
  _exit(1);
}

// An engine process of this binary, which answers each command with a line.
class Engine final {
public:
  Engine(const std::string &self, const std::vector<std::string> &arguments) {
    int input[2], output[2];
    require(pipe(input) == 0 && pipe(output) == 0, "cannot create a pipe");
    // No engine process holds another's pipes, which would keep it from
    // seeing the end of its commands.
    for (int descriptor : {input[0], input[1], output[0], output[1]})
      require(fcntl(descriptor, F_SETFD, FD_CLOEXEC) == 0, "cannot set close-on-exec on a pipe");
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, input[0], STDIN_FILENO);
    posix_spawn_file_actions_adddup2(&actions, output[1], STDOUT_FILENO);
    for (int descriptor : {input[0], input[1], output[0], output[1]})
      posix_spawn_file_actions_addclose(&actions, descriptor);
    std::vector<std::string> all{self, "engine"};
    all.insert(all.end(), arguments.begin(), arguments.end());
    std::vector<char *> argv;
    for (std::string &argument : all) argv.push_back(argument.data());
    argv.push_back(nullptr);
    const int spawned = posix_spawn(&pid_, self.c_str(), &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    close(input[0]);
    close(output[1]);
    require(spawned == 0, "cannot start an engine process");
    if (gEngineCount < std::size(gEngines)) {
      slot_ = gEngineCount;
      gEngines[gEngineCount++] = pid_;
    }
    commands_ = fdopen(input[1], "w");
    replies_ = fdopen(output[0], "r");
    require(reply() == "ready", "an engine process did not start");
  }
  ~Engine() {
    // A stopped process cannot see the end of its commands.
    if (stopped_) ::kill(pid_, SIGCONT);
    if (commands_) fclose(commands_);
    if (pid_ > 0) {
      int status = 0;
      waitpid(pid_, &status, 0);
    }
    if (slot_ < std::size(gEngines)) gEngines[slot_] = 0;
    if (replies_) fclose(replies_);
  }
  Engine(const Engine &) = delete;
  Engine &operator=(const Engine &) = delete;

  // The engine's reply to command.
  std::string ask(const std::string &command) {
    send(command);
    return reply();
  }
  void send(const std::string &command) {
    std::fprintf(commands_, "%s\n", command.c_str());
    std::fflush(commands_);
  }
  // The engine's reply to the oldest command it has not answered.
  std::string reply() {
    char *line = nullptr;
    size_t capacity = 0;
    const ssize_t length = getline(&line, &capacity, replies_);
    std::string reply(length > 0 ? line : "", length > 0 ? size_t(length) : 0);
    free(line);
    require(length > 0, "an engine process ended");
    if (reply.back() == '\n') reply.pop_back();
    return reply;
  }

  [[nodiscard]] pid_t pid() const { return pid_; }
  // Ends the process at once, as the system does when it kills it for memory.
  void kill() {
    ::kill(pid_, SIGKILL);
    int status = 0;
    waitpid(pid_, &status, 0);
    if (slot_ < std::size(gEngines)) gEngines[slot_] = 0;
    pid_ = -1;
  }
  // Stops the process, as SIGSTOP or a debugger does, until resume().
  void stop() {
    ::kill(pid_, SIGSTOP);
    stopped();
  }
  // Waits until the process stopped, itself or by stop().
  void stopped() {
    int status = 0;
    require(waitpid(pid_, &status, WUNTRACED) == pid_ && WIFSTOPPED(status), "an engine process did not stop");
    stopped_ = true;
  }
  void resume() {
    ::kill(pid_, SIGCONT);
    stopped_ = false;
  }

private:
  pid_t pid_ = -1;
  size_t slot_ = std::size(gEngines);
  bool stopped_ = false;
  FILE *commands_ = nullptr;
  FILE *replies_ = nullptr;
};

// What an engine reports of one image.
struct Image final {
  std::string origin;
  std::string digest;
};
using Images = std::map<std::string, Image>;

Images parseImages(const std::string &line) {
  std::istringstream reply(line);
  std::string word;
  reply >> word;
  require(word == "images", "an engine process answered " + line);
  Images images;
  for (std::string entry; reply >> entry;) {
    const size_t equals = entry.find('='), colon = entry.find(':', equals);
    require(equals != std::string::npos && colon != std::string::npos, "an engine process answered " + entry);
    images[entry.substr(0, equals)] = {entry.substr(equals + 1, colon - equals - 1), entry.substr(colon + 1)};
  }
  return images;
}

// A registry, digest tables and sources of this test alone. The registry is
// short, under /tmp, as socket paths are.
class Setup final {
public:
  Setup(std::string self, std::string metallib)
      : self_(std::move(self)), metallib_(std::move(metallib)), tables_("shared-weights-digests"),
        root_("shared-weights-sources") {
    char pattern[] = "/tmp/splash-sw-XXXXXX";
    require(mkdtemp(pattern) != nullptr, "cannot create a registry directory");
    registry_ = std::filesystem::path(pattern) / "registry";
  }
  ~Setup() {
    std::error_code ignored;
    std::filesystem::remove_all(registry_.parent_path(), ignored);
  }
  std::unique_ptr<Engine> engine(bool shared = true) const { return engine(self_, metallib_, shared); }
  // An engine process of the executable with the metallib. Without recording,
  // none of its sources has a recorded digest.
  std::unique_ptr<Engine> engine(const std::string &executable, const std::string &metallib, bool shared = true,
                                 bool recorded = true) const {
    return std::make_unique<Engine>(executable,
                                    std::vector<std::string>{metallib, root_.path(), registry_, tables_.path(),
                                                             !shared ? "off" : recorded ? "on" : "unrecorded"});
  }
  // Writes the checkpoint called name.
  void checkpoint(const std::string &name, const std::vector<SyntheticTensor> &tensors) const {
    writeCheckpoint(root_.path() / name, tensors);
  }
  // Writes the model called name of kind, its tensors in the other order when
  // reversed.
  void model(const Kind &kind, const std::string &name, bool reversed = false) const {
    kind.write(root_.path() / name, reversed);
  }
  // Writes the source file called name, which copy images are read from.
  void source(const std::string &name, const std::vector<uint8_t> &bytes) const {
    splash::test::writeFile(root_.path() / name, bytes);
  }
  const std::filesystem::path &registry() const { return registry_; }
  const std::filesystem::path &tables() const { return tables_.path(); }
  const std::filesystem::path &root() const { return root_.path(); }
  const std::string &self() const { return self_; }
  const std::string &metallib() const { return metallib_; }

private:
  std::string self_, metallib_;
  splash::test::TemporaryDirectory tables_, root_;
  std::filesystem::path registry_;
};

// The images of role in loaded, which a test names by component.
Images roleImages(const Images &loaded, Role role) {
  const std::string prefix = std::string(directoryOf(role)) + "/";
  Images images;
  for (const auto &[component, image] : loaded)
    if (component.starts_with(prefix)) images.insert({component, image});
  require(images.size() == imageCount(role), "an engine process loaded " + std::to_string(images.size()) + " " +
                                                 directoryOf(role) + " images, not " +
                                                 std::to_string(imageCount(role)));
  return images;
}

Images targetImages(const Images &loaded) { return roleImages(loaded, Role::Target); }

// A tensor of a source file: where its bytes are in the file.
struct Extent final {
  std::filesystem::path file;
  uint64_t offset, bytes;
  std::string name, dtype;
};

// Where in a tensor to change it.
enum class Where { First, Middle, Last };

// Flips what a tensor holds in place, which a second call restores. A float
// tower's tensor changes its sign and an F32 tensor the upper half of a value
// (both keep the exact BF16 values that writers of F32 and F16 sources need),
// a draft's BF16 weight its exponent, so that quantizing its group cannot
// absorb the change, and any other tensor two bytes.
void flip(const Extent &tensor, Role role, Where where = Where::Middle) {
  const size_t width = widthOf(tensor.dtype);
  const bool floats = tensor.dtype == "BF16" || tensor.dtype == "F16" || tensor.dtype == "F32";
  const uint64_t values = tensor.bytes / width;
  const uint64_t index = where == Where::First ? 0 : where == Where::Last ? values - 1 : values / 2;
  const uint64_t value = tensor.offset + index * width;
  uint64_t at = where == Where::First ? tensor.offset
                : where == Where::Last ? tensor.offset + tensor.bytes - 2
                                       : tensor.offset + tensor.bytes / 2;
  size_t count = 2;
  uint8_t mask = 0x3f;
  if (role == Role::Vision && floats) {
    at = value + width - 1;
    count = 1;
    mask = 0x80;
  } else if (role == Role::Draft && tensor.dtype == "BF16") {
    at = value + 1;
    count = 1;
    mask = 0x7f;
  } else if (tensor.dtype == "F32") {
    at = value + 2;
  }
  std::fstream file(tensor.file, std::ios::binary | std::ios::in | std::ios::out);
  std::array<char, 2> bytes{};
  file.seekg(std::streamoff(at));
  file.read(bytes.data(), std::streamsize(count));
  for (size_t i = 0; i < count; ++i) bytes[i] = char(bytes[i] ^ mask);
  file.seekp(std::streamoff(at));
  file.write(bytes.data(), std::streamsize(count));
  require(bool(file.flush()), "cannot change " + tensor.file.string());
}

std::vector<Extent> safetensorsExtents(const std::filesystem::path &file) {
  std::ifstream input(file, std::ios::binary);
  uint64_t length = 0;
  input.read(reinterpret_cast<char *>(&length), sizeof length);
  std::string header(length, '\0');
  input.read(header.data(), std::streamsize(length));
  require(bool(input), "cannot read the header of " + file.string());
  const std::regex entry(
      R"re("([^"]+)":\{"dtype":"([A-Za-z0-9_]+)","shape":\[[0-9,]*\],"data_offsets":\[([0-9]+),([0-9]+)\]\})re");
  std::vector<Extent> extents;
  for (std::sregex_iterator found(header.begin(), header.end(), entry), end; found != end; ++found) {
    const uint64_t first = std::stoull((*found)[3]), last = std::stoull((*found)[4]);
    extents.push_back({file, sizeof(uint64_t) + length + first, last - first, (*found)[1], (*found)[2]});
  }
  return extents;
}

std::vector<Extent> ggufExtents(const std::filesystem::path &file) {
  model::WeightSource source(file);
  const model::GgufFile gguf(source);
  std::vector<Extent> extents;
  for (const model::GgufTensor &tensor : gguf.tensors())
    extents.push_back({file, source.dataOffset() + tensor.offset, tensor.bytes, tensor.name,
                       model::ggmlTypeName(tensor.type)});
  return extents;
}

// The tensors of the model at directory that the loader of role reads.
std::vector<Extent> tensorsOf(const Kind &kind, Role role, const std::filesystem::path &directory) {
  const std::filesystem::path at = directory / directoryOf(role);
  std::vector<Extent> extents;
  if (role == Role::Target && kind.target == model::TargetSource::Gguf) extents = ggufExtents(model::findTargetGgufs(at).front());
  else if (role == Role::Vision && kind.vision == model::VisionSource::Gguf) extents = ggufExtents(at / "mmproj.gguf");
  else extents = safetensorsExtents(at / "model.safetensors");
  std::erase_if(extents, [](const Extent &tensor) { return tensor.bytes < 2; });
  return extents;
}

// A second process that loads the model a first holds receives every image of
// the role: it writes none, and each holds the bytes an image written alone
// holds. A third that loads the model with some tensors changed writes the
// images they change, as the bytes they hold say, and receives the others.
void testProcessesShare(const Setup &setup, const Kind &kind, Role role) {
  setup.model(kind, "base");
  const auto load = [&](Engine &engine, const std::string &name) {
    return roleImages(parseImages(engine.ask("load " + name + " " + kind.name)), role);
  };
  const Images alone = load(*setup.engine(false), "base");
  auto first = setup.engine(), second = setup.engine();
  const Images written = load(*first, "base");
  const Images received = load(*second, "base");
  for (const auto &[component, image] : alone) {
    require(written.at(component).origin == "written",
            "the first process did not write " + component + ": it " + written.at(component).origin + " it");
    require(received.at(component).origin == "received",
            "the second process did not receive " + component + ": it " + received.at(component).origin + " it");
    require(written.at(component).digest == image.digest, component + " written beside others differs from alone");
    require(received.at(component).digest == image.digest, "a received " + component + " differs from alone");
  }
  // Tensors spread over the model, few enough that images stay unchanged.
  std::filesystem::copy(setup.root() / "base", setup.root() / "tuned", std::filesystem::copy_options::recursive);
  const std::vector<Extent> tensors = tensorsOf(kind, role, setup.root() / "tuned");
  const size_t picks = imageCount(role) >= 8 ? 3 : 1;
  for (size_t pick = 0; pick < picks; ++pick) flip(tensors.at(tensors.size() * (2 * pick + 1) / (2 * picks)), role);
  const Images tunedAlone = load(*setup.engine(false), "tuned");
  const Images tuned = load(*setup.engine(), "tuned");
  size_t changed = 0;
  for (const auto &[component, image] : tunedAlone) {
    const bool differs = image.digest != alone.at(component).digest;
    changed += differs;
    require(tuned.at(component).digest == image.digest, "the tuned model's " + component + " differs from alone");
    require(tuned.at(component).origin == (differs ? "written" : "received"),
            "the tuned model " + tuned.at(component).origin + " " + component + ", which it " +
                (differs ? "changed" : "left unchanged"));
  }
  require(changed > 0 && (changed < imageCount(role) || imageCount(role) == 1),
          "the changes to the tuned model changed " + std::to_string(changed) + " of its images");
}

// Two processes that load one checkpoint at the same time hold one copy of
// each image: one writes it and the other receives it.
void testConcurrentLoadsWriteEachImageOnce(const Setup &setup) {
  setup.checkpoint("base", targetTensors());
  auto first = setup.engine(), second = setup.engine();
  first->send("load base mlx");
  second->send("load base mlx");
  const Images one = targetImages(parseImages(first->reply())), other = targetImages(parseImages(second->reply()));
  for (const auto &[component, image] : one)
    require((image.origin == "written") != (other.at(component).origin == "written") &&
                image.digest == other.at(component).digest,
            "two processes that loaded " + component + " at once did not write it once: " + image.origin + " and " +
                other.at(component).origin);
}

// A fine-tune that changes attention only receives the base's routed experts
// and the embedding and head, and the layers it did not change, and writes the
// layers it changed itself. Its tensors lie in another order, so at other
// offsets of the file.
void testFineTuneSharesWhatMatches(const Setup &setup) {
  std::vector<SyntheticTensor> base = targetTensors(), tuned = base;
  setup.checkpoint("base", base);
  // A full-attention layer's projection, and a GDN layer's.
  retune(tuned, "language_model.model.layers.3.self_attn.q_proj");
  retune(tuned, "language_model.model.layers.1.linear_attn.in_proj_qkv");
  std::reverse(tuned.begin(), tuned.end());
  setup.checkpoint("tuned", tuned);
  const Images alone = targetImages(parseImages(setup.engine(false)->ask("load tuned mlx")));
  auto first = setup.engine(), second = setup.engine();
  const Images held = targetImages(parseImages(first->ask("load base mlx")));
  const Images fineTune = targetImages(parseImages(second->ask("load tuned mlx")));
  for (const auto &[component, image] : alone) {
    const bool changed = component == "target/layer-1.bin" || component == "target/layer-3.bin";
    require(fineTune.at(component).origin == (changed ? "written" : "received"),
            "the fine-tune " + fineTune.at(component).origin + " " + component);
    require(fineTune.at(component).digest == image.digest, "the fine-tune's " + component + " differs from alone");
    require((held.at(component).digest != image.digest) == changed,
            component + (changed ? " did not change" : " changed") + " with the fine-tune");
  }
}

// A process that releases its memory lets go of its view only: another
// process that holds the image keeps its bytes, and the restore maps the image
// that process holds, writing nothing. The memory goes back to macOS when no
// process holds it any more, so that a process that loads the model after
// every holder released writes the images again.
void testReleaseAndRestoreKeepOtherHolders(const Setup &setup) {
  setup.checkpoint("base", targetTensors());
  auto first = setup.engine(), second = setup.engine();
  const Images written = targetImages(parseImages(first->ask("load base mlx")));
  const Images received = targetImages(parseImages(second->ask("load base mlx")));
  const Images released = targetImages(parseImages(second->ask("release")));
  for (const auto &[component, image] : released)
    require(image.digest == "released", "a released process still held " + component);
  const Images kept = targetImages(parseImages(first->ask("contents")));
  require(kept.size() == written.size(), "the images of a process changed with another's release");
  for (const auto &[component, image] : written)
    require(kept.at(component).digest == image.digest && kept.at(component).origin == "written",
            "another process's release changed " + component + " in the process that holds it");
  const Images restored = targetImages(parseImages(second->ask("restore")));
  for (const auto &[component, image] : received)
    require(restored.at(component).digest == image.digest && restored.at(component).origin == "received",
            "a restore did not map " + component + " that another process holds, or it holds other bytes");
  const Images after = targetImages(parseImages(first->ask("contents")));
  for (const auto &[component, image] : written)
    require(after.at(component).digest == image.digest, "a restore in one process changed " + component + " in another");
  // Every holder let go: the memory is gone, and the next load writes it.
  require(!targetImages(parseImages(first->ask("release"))).empty() &&
              !targetImages(parseImages(second->ask("release"))).empty(),
          "a process did not release its images");
  const Images last = targetImages(parseImages(setup.engine()->ask("load base mlx")));
  for (const auto &[component, image] : written)
    require(last.at(component).origin == "written" && last.at(component).digest == image.digest,
            component + " was still held after every process released it, or differs");
}

// The numbers after a reply's first word, which is word.
std::vector<uint64_t> numbersOf(const std::string &reply, const std::string &word) {
  std::istringstream words(reply);
  std::string first;
  words >> first;
  require(first == word, "an engine process answered " + reply);
  std::vector<uint64_t> numbers;
  for (uint64_t number; words >> number;) numbers.push_back(number);
  return numbers;
}

uint64_t allocatedBy(Engine &engine) { return numbersOf(engine.ask("allocated"), "allocated").at(0); }

size_t countOrigin(const Images &images, const std::string &origin) {
  return size_t(std::count_if(images.begin(), images.end(), [&](const auto &entry) {
    return entry.second.origin == origin && entry.second.digest != "released";
  }));
}

double secondsSince(std::chrono::steady_clock::time_point started) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
}

// A restore asks its admission for each image's full size, an image another
// process holds as one it writes: the memory is this process's own to account
// for, whoever else maps it.
void testRestoreAdmitsReceivedImagesInFull(const Setup &setup) {
  setup.checkpoint("base", targetTensors());
  auto first = setup.engine(), second = setup.engine();
  static_cast<void>(parseImages(first->ask("load base mlx")));
  const Images received = parseImages(second->ask("load base mlx"));
  require(countOrigin(received, "received") == received.size(), "a second process did not receive every image");
  const std::vector<uint64_t> sizes = numbersOf(second->ask("sizes"), "sizes");
  static_cast<void>(second->ask("release"));
  const Images restored = parseImages(second->ask("restore"));
  require(countOrigin(restored, "received") == restored.size(), "a restore wrote an image another process holds");
  require(numbersOf(second->ask("admitted"), "admitted") == sizes,
          "a restore did not ask its admission for each image at its full size");
}

// An admission that refuses part-way through a restore leaves the images
// before the refused one mapped and the others released, with no memory held
// for the refused one; a release then gives back only what came back, and a
// restore that starts over maps every image again. Another process that loads
// meanwhile receives what is held and writes the rest.
void testRestoreRefusedPartWay(const Setup &setup) {
  setup.checkpoint("base", targetTensors());
  auto first = setup.engine(), second = setup.engine();
  const Images written = parseImages(first->ask("load base mlx"));
  static_cast<void>(parseImages(second->ask("load base mlx")));
  const uint64_t loaded = allocatedBy(*second);
  static_cast<void>(second->ask("release"));
  const uint64_t released = allocatedBy(*second);
  require(released < loaded, "a released process still holds its images");

  constexpr unsigned back = 3;
  const std::string refused = second->ask("restore " + std::to_string(back));
  require(refused.starts_with("refused ") && refused.find("host memory reserve protected") != std::string::npos,
          "a restore refused by admission answered " + refused);
  const Images partial = parseImages(second->ask("contents"));
  size_t held = 0;
  for (const auto &[component, image] : partial) {
    if (image.digest == "released") continue;
    ++held;
    require(image.origin == "received" && image.digest == written.at(component).digest,
            "the image " + component + " before the refused one did not come back whole");
  }
  require(held == back, std::to_string(held) + " images came back before the refusal, not " + std::to_string(back));
  const uint64_t partway = allocatedBy(*second);
  require(partway > released && partway < loaded, "the images before the refusal were not the memory held");
  require(second->ask("restore 0").starts_with("refused ") && allocatedBy(*second) == partway,
          "an image that admission refused holds memory");

  // Meanwhile nothing waits for a lock: a process that loads receives the
  // images held and writes the others.
  static_cast<void>(first->ask("release"));
  const auto started = std::chrono::steady_clock::now();
  auto third = setup.engine();
  const Images beside = parseImages(third->ask("load base mlx"));
  require(secondsSince(started) < 5, "a restore refused part-way held up another process's load");
  require(countOrigin(beside, "received") == back && countOrigin(beside, "written") == beside.size() - back,
          "another process did not receive the images a partial restore holds and write the others");

  // A release in the middle of a restore gives back what came back, no more.
  static_cast<void>(second->ask("release"));
  require(allocatedBy(*second) == released, "a release during a restore kept memory");
  const Images again = parseImages(second->ask("restore"));
  for (const auto &[component, image] : written)
    require(again.at(component).digest == image.digest && again.at(component).origin == "received",
            "a restore that started over did not map " + component + " as it was");
  require(allocatedBy(*second) == loaded, "a restore that started over did not take back the memory");
}

// What a process answers to a copy image's load: how many images it wrote,
// then its images.
struct Copied final {
  unsigned writes = 0;
  Images images;
};

Copied parseCopied(const std::string &reply) {
  std::istringstream words(reply);
  std::string loaded;
  Copied copied;
  words >> loaded >> copied.writes;
  require(loaded == "loaded", "an engine process answered " + reply);
  std::string images;
  std::getline(words, images);
  copied.images = parseImages(images);
  return copied;
}

// A writer that runs out of memory as a restore writes its image leaves no
// lock and no mapping: the image stays released, holds no memory, and another
// process that needs the same image is not held up by the lock, writes it and
// serves it to the next.
void testWriterOutOfMemoryLeavesNoLockOrMapping(const Setup &setup) {
  setup.source("o.bin", pattern(100000, 31));
  auto writer = setup.engine(), other = setup.engine();
  const Copied written = parseCopied(writer->ask("copy load o.bin 0 65536"));
  require(written.writes == 1, "a process did not write its image");
  static_cast<void>(writer->ask("copy release"));
  const uint64_t released = allocatedBy(*writer);
  const uint64_t writes = numbersOf(writer->ask("copy writes"), "writes").at(0);
  require(writer->ask("copy fail") == "next write fail", "a process did not take a command");
  const std::string refused = writer->ask("copy restore");
  require(refused.starts_with("refused "), "a restore whose writer ran out of memory answered " + refused);
  require(numbersOf(writer->ask("copy writes"), "writes").at(0) == writes + 1,
          "the restore did not reach its writer, so that it refused nothing");
  const Images held = parseImages(writer->ask("copy contents"));
  require(held.begin()->second.digest == "released" && allocatedBy(*writer) == released,
          "an image whose writer ran out of memory holds memory");

  // Another process asks for the key at once: the lock is free and it writes
  // under it, so that the next process receives the image from it.
  const auto started = std::chrono::steady_clock::now();
  const Copied wrote = parseCopied(other->ask("copy load o.bin 0 65536"));
  const double waited = secondsSince(started);
  require(wrote.writes == 1 && waited < 0.8,
          "a process that asked for the key of a writer that ran out of memory waited " + std::to_string(waited) + " s");
  auto third = setup.engine();
  const Copied received = parseCopied(third->ask("copy load o.bin 0 65536"));
  require(received.writes == 0 && received.images.begin()->second.digest == wrote.images.begin()->second.digest &&
              received.images.begin()->second.digest == written.images.begin()->second.digest,
          "an image written after a writer ran out of memory was not served, or differs");

  // The writer that failed restores it by receiving it.
  const Images restored = parseImages(writer->ask("copy restore"));
  require(restored.begin()->second.origin == "received" && restored.begin()->second.digest == written.images.begin()->second.digest,
          "the writer that ran out of memory did not receive the image afterwards");
}

// A process stopped while it writes an image, holding the image's lock, delays
// another process that needs the image by a bounded wait for the lock, after
// which that one writes a copy of its own.
void testStoppedWriterIsWaitedForBoundedly(const Setup &setup) {
  setup.source("p.bin", pattern(100000, 32));
  auto stopped = setup.engine(), other = setup.engine();
  require(stopped->ask("copy stop") == "next write stop", "a process did not take a command");
  stopped->send("copy load p.bin 0 65536");
  stopped->stopped();
  const auto started = std::chrono::steady_clock::now();
  const Copied written = parseCopied(other->ask("copy load p.bin 0 65536"));
  const double waited = secondsSince(started);
  require(written.writes == 1, "a process did not write an image whose writer stopped");
  require(waited < 6, "a stopped writer delayed a load by " + std::to_string(waited) + " s");
  stopped->resume();
  const Copied resumed = parseCopied(stopped->reply());
  require(resumed.writes == 1 && resumed.images.begin()->second.digest == written.images.begin()->second.digest,
          "a resumed writer's image differs");
}

// A holder killed leaves the others' images whole and nothing behind: the
// survivors serve the same bytes, a process that starts afterwards maps what
// they hold, the files of the process that was killed are removed, and the
// memory goes back when the last holder is gone.
void testKilledHolderLeavesOthersServing(const Setup &setup) {
  setup.checkpoint("base", targetTensors());
  auto first = setup.engine(), second = setup.engine();
  const Images written = parseImages(first->ask("load base mlx"));
  static_cast<void>(parseImages(second->ask("load base mlx")));
  const std::string dead = std::to_string(first->pid()) + "-";
  const auto leftBy = [&](const std::string &prefix) {
    size_t files = 0;
    for (const auto &entry : std::filesystem::directory_iterator(setup.registry()))
      files += entry.path().filename().string().starts_with(prefix);
    return files;
  };
  require(leftBy(dead) > 0, "a holder has no files in the registry");
  first->kill();
  const Images kept = parseImages(second->ask("contents"));
  for (const auto &[component, image] : written)
    require(kept.at(component).digest == image.digest, component + " changed when the process that wrote it was killed");

  auto restart = setup.engine();
  const Images restarted = parseImages(restart->ask("load base mlx"));
  for (const auto &[component, image] : written)
    require(restarted.at(component).origin == "received" && restarted.at(component).digest == image.digest,
            "a restart did not map " + component + " that a survivor holds");
  require(leftBy(dead) == 0, "the files of a process that was killed were left in the registry");

  // A survivor that releases and restores while a process holds the images
  // maps them again; the last holder's end ends the memory.
  static_cast<void>(second->ask("release"));
  const Images mapped = parseImages(second->ask("restore"));
  require(countOrigin(mapped, "received") == mapped.size(), "a survivor's restore wrote images that a process holds");
  second->kill();
  const Images alone = parseImages(restart->ask("contents"));
  for (const auto &[component, image] : written)
    require(alone.at(component).digest == image.digest, component + " changed when the second holder was killed");
  static_cast<void>(restart->ask("release"));
  const Images rewritten = parseImages(restart->ask("restore"));
  require(countOrigin(rewritten, "written") == rewritten.size(),
          "a restore received images that no process held, or the memory was not given back");
  for (const auto &[component, image] : written)
    require(rewritten.at(component).digest == image.digest, "a rewritten " + component + " differs");
  restart->kill();
  require(parseImages(setup.engine()->ask("load base mlx")).begin()->second.origin == "written",
          "an image outlived every process that held it");
}

// A stopped process that holds images delays another process's load, and its
// restore, once each: after one wait for an answer the other process passes
// over it and writes its images, rather than waiting per image. A restore
// after it resumed asks it again and receives.
void testStoppedHolderIsPassedOverOncePerPass(const Setup &setup) {
  setup.checkpoint("base", targetTensors());
  auto stopped = setup.engine(), other = setup.engine();
  const Images held = parseImages(stopped->ask("load base mlx"));
  stopped->stop();
  auto started = std::chrono::steady_clock::now();
  const Images loaded = parseImages(other->ask("load base mlx"));
  const double loading = secondsSince(started);
  require(countOrigin(loaded, "written") == loaded.size(), "an image of a stopped process was received");
  require(loading < 8, "a stopped process delayed a load by " + std::to_string(loading) + " s");
  static_cast<void>(other->ask("release"));
  started = std::chrono::steady_clock::now();
  const Images restored = parseImages(other->ask("restore"));
  const double restoring = secondsSince(started);
  require(countOrigin(restored, "written") == restored.size(), "a restore received from a stopped process");
  require(restoring < 8, "a stopped process delayed a restore by " + std::to_string(restoring) + " s");
  for (const auto &[component, image] : held)
    require(restored.at(component).digest == image.digest, "images written beside a stopped process differ");

  stopped->resume();
  require(parseImages(stopped->ask("contents")).size() == held.size(), "a resumed process lost its images");
  static_cast<void>(other->ask("release"));
  const Images again = parseImages(other->ask("restore"));
  require(countOrigin(again, "received") == again.size(), "a restore did not ask a holder that had stopped, once it resumed");
}

// Without sharing a process creates no registry and no digest table, and its
// images are its own memory; with it, the same images are shared memory,
// which is read-only once written.
void testUnsharedCreatesNothing(const Setup &setup) {
  setup.checkpoint("base", targetTensors());
  auto alone = setup.engine(false);
  const Images unshared = targetImages(parseImages(alone->ask("load base mlx")));
  for (const auto &[component, image] : unshared)
    require(image.origin == "private", "a process without sharing did not hold " + component + " privately");
  require(alone->ask("writable") != "writable 0", "a process without sharing holds its images in memory it cannot write");
  require(!std::filesystem::exists(setup.registry()) && std::filesystem::is_empty(setup.tables()),
          "a process without sharing created a registry or a digest table");
  auto sharing = setup.engine();
  const Images shared = targetImages(parseImages(sharing->ask("load base mlx")));
  require(std::filesystem::exists(setup.registry()) && !std::filesystem::is_empty(setup.tables()),
          "a process with sharing created no registry or digest table");
  for (const auto &[component, image] : unshared)
    require(shared.at(component).digest == image.digest, "a shared " + component + " differs from an unshared one");
}


// An image written from a file whose digest the installation did not record
// cannot be keyed: it is the process's own, as without sharing, though the
// process shares the images that it can key.
void testUnrecordedSourcesAreNotShared(const Setup &setup) {
  setup.checkpoint("base", targetTensors());
  auto first = setup.engine(), unrecorded = setup.engine(setup.self(), setup.metallib(), true, false);
  const Images written = targetImages(parseImages(first->ask("load base mlx")));
  const Images loaded = targetImages(parseImages(unrecorded->ask("load base mlx")));
  for (const auto &[component, image] : loaded)
    require(image.origin == "private" && image.digest == written.at(component).digest,
            component + " of a file without a recorded digest was not this process's own or differs");
}

// A process's writers are the code it runs: an executable and a metallib that
// replace its own on disk after it started change nothing, and the same code
// at another path writes the same images.
void testWriterIsTheRunningCode(const Setup &setup) {
  setup.checkpoint("base", targetTensors());
  const std::filesystem::path copies = setup.registry().parent_path() / "copies";
  std::filesystem::create_directory(copies);
  const std::filesystem::path executable = copies / "shared-weights", metallib = copies / "copy.metallib";
  std::filesystem::copy_file(setup.self(), executable);
  std::filesystem::copy_file(setup.metallib(), metallib);
  auto copied = setup.engine(executable, metallib), original = setup.engine();
  const std::string running = copied->ask("identity running");
  require(running.starts_with("identity ") && running == copied->ask("identity file"),
          "an engine's identity is not that of its executable file");
  for (const std::filesystem::path &path : {executable, metallib}) {
    splash::test::writeFile(copies / "replacement", std::string_view("other code"));
    std::filesystem::rename(copies / "replacement", path);
  }
  require(copied->ask("identity running") == running, "code that replaced a process's own on disk changed its identity");
  const Images written = targetImages(parseImages(copied->ask("load base mlx")));
  const Images received = targetImages(parseImages(original->ask("load base mlx")));
  for (const auto &[component, image] : written)
    require(image.origin == "written" && received.at(component).origin == "received" &&
                received.at(component).digest == image.digest,
            "a process took code that replaced its own on disk for the code it runs: " + component);
}

// Sharing keeps its registry in the user's own temporary directory, which no
// other user can enter, so that no one else can create it first; its digest
// tables go in the user's caches.
void testUserPlace() {
  const char *saved = std::getenv("HOME");
  const std::string home = saved ? saved : "";
  setenv("HOME", "/Users/someone", 1);
  const model::SharedImagesPlace place = model::userSharedImagesPlace();
  const std::filesystem::path parent = place.registry.parent_path();
  struct stat state{};
  require(lstat(parent.c_str(), &state) == 0 && S_ISDIR(state.st_mode) && state.st_uid == geteuid() &&
              !(state.st_mode & 077),
          "the image registry is in a directory other users can enter: " + parent.string());
  require(place.digestTables == "/Users/someone/Library/Caches/Splash/tensor-digests",
          "the digest tables are not in the user's caches");
  unsetenv("HOME");
  splash::test::rejects([] { static_cast<void>(model::userSharedImagesPlace()); }, "HOME",
                        "sharing without a home directory for its digests");
  if (saved) setenv("HOME", home.c_str(), 1);
}

// Sharing a model's images names each source file's digest as the model's
// record gives it.
void testConfigTakesTheRecordedDigests(const Setup &setup) {
  const std::filesystem::path root = setup.root() / "installed";
  std::filesystem::create_directories(root);
  const std::string digest(64, 'c');
  splash::test::writeFile(root / "model.json",
                          R"({"files":{"target/model.safetensors":{"digest":")" + digest + R"("}}})");
  const model::SharedImagesConfig config = model::sharedImagesConfig({"registry", "tables"}, root, "writers");
  require(config.registry == "registry" && config.digestTables == "tables" && config.writerIdentity == "writers" &&
              config.modelRoot == root &&
              config.sourceDigests ==
                  std::map<std::string, std::string, std::less<>>{{"target/model.safetensors", digest}},
          "a model's sharing is not configured as its record and place say");
}

// ---- The keys ----

// The plans of the images a loader of role makes of the model at directory, as
// the loader loads them.
std::vector<model::ImagePlan> plansOf(const Kind &kind, Role role, const std::filesystem::path &directory,
                                      metal::MetalBackend &backend) {
  model::WeightImages images(backend);
  const model::Qwen3_6MoeLayout target = syntheticTarget();
  const auto plans = [](const auto &loader) {
    std::vector<model::ImagePlan> plans;
    for (size_t index = 0; index < loader.imageCount(); ++index) plans.push_back(loader.image(index));
    return plans;
  };
  if (role == Role::Draft)
    return plans(model::DraftCheckpointLoader(backend, images, directory / "draft", syntheticDraft(target)));
  if (role == Role::Vision)
    return {model::VisionLoader(directory / "vision", kind.vision, syntheticVision()).image()};
  if (kind.target == model::TargetSource::Gguf)
    return plans(model::GgufTargetLoader(backend, images, model::findTargetGgufs(directory / "target"),
                                         target.ggufArchitecture, target));
  return plans(model::SafetensorsTargetLoader(backend, images, directory / "target", target));
}

// A SharedImages that keys images of the role's files of the model under
// root, each recorded by its SHA-256 as the installer records a file.
std::unique_ptr<model::SharedImages> keyer(const Setup &setup, const std::filesystem::path &root, Role role,
                                           const std::string &identity = "test-writer") {
  model::SharedImagesConfig config;
  config.registry = setup.registry();
  config.digestTables = setup.tables();
  config.writerIdentity = identity;
  config.modelRoot = root;
  for (const auto &entry : std::filesystem::recursive_directory_iterator(root / directoryOf(role)))
    if (entry.is_regular_file())
      config.sourceDigests[entry.path().lexically_relative(root).generic_string()] =
          model::weightDigest(splash::test::readFile(entry.path()));
  return std::make_unique<model::SharedImages>(std::move(config));
}

// The keys of the images of the role of the model called name under setup, by
// component.
std::map<std::string, std::string> keysOf(const Setup &setup, metal::MetalBackend &backend, const Kind &kind,
                                          Role role, const std::string &name,
                                          const std::string &identity = "test-writer") {
  const std::filesystem::path root = setup.root() / name;
  const auto shared = keyer(setup, root, role, identity);
  std::map<std::string, std::string> keys;
  for (const model::ImagePlan &plan : plansOf(kind, role, root, backend)) {
    const std::string key = shared->key(plan);
    require(key.size() == 64, "the key of " + plan.component + " is not a SHA-256: its loader does not describe it");
    keys[plan.component] = key;
  }
  require(keys.size() == imageCount(role), "a model's images do not have distinct names");
  return keys;
}

// The SHA-256 of each image of the role of the model at directory, as the
// writers write it, by component.
std::map<std::string, std::string> writtenImages(metal::MetalBackend &backend, const Kind &kind, Role role,
                                                 const std::filesystem::path &directory) {
  std::map<std::string, std::string> digests;
  for (const model::ImagePlan &plan : plansOf(kind, role, directory, backend)) {
    const metal::MetalBuffer buffer = backend.allocateBuffer(plan.bytes, metal::BufferStorage::Shared, plan.component);
    plan.write(model::contentsOf(buffer), buffer);
    digests[plan.component] = model::weightDigest(model::contentsOf(buffer));
  }
  return digests;
}

// Images with one key are one image: a change to any byte of a source tensor
// that changes an image the writers write changes its key. Each tensor of the
// role's files has a few of its bytes changed in turn, and the images written
// from the changed model are compared with the key's.
void testKeyCoversSourceBytes(const Setup &setup, metal::MetalBackend &backend, const Kind &kind, Role role) {
  setup.model(kind, "base");
  const std::filesystem::path directory = setup.root() / "base";
  const auto base = keysOf(setup, backend, kind, role, "base");
  const auto baseImages = writtenImages(backend, kind, role, directory);
  require(keysOf(setup, backend, kind, role, "base") == base, "the keys of one model changed between two computations");
  for (const auto &[component, key] : base)
    for (const auto &[other, otherKey] : base)
      require(component == other || key != otherKey, component + " and " + other + " have one key");
  // A GGUF's images read rows of its tensors, one expert's among the fused
  // experts', so a change at either end can fall out of the rows.
  const bool rows = (role == Role::Target && kind.target == model::TargetSource::Gguf) ||
                    (role == Role::Vision && kind.vision == model::VisionSource::Gguf);
  const std::vector<Where> positions = rows ? std::vector<Where>{Where::First, Where::Middle, Where::Last}
                                            : std::vector<Where>{Where::Middle};
  for (const Extent &tensor : tensorsOf(kind, role, directory))
    for (const Where where : positions) {
      flip(tensor, role, where);
      const auto keys = keysOf(setup, backend, kind, role, "base");
      const auto images = writtenImages(backend, kind, role, directory);
      flip(tensor, role, where);
      bool written = false;
      for (const auto &[component, digest] : images) {
        const bool differs = digest != baseImages.at(component);
        written = written || differs;
        require(!differs || keys.at(component) != base.at(component),
                "a changed byte of " + tensor.name + " changed " + component + " but not its key");
      }
      require(written, "a changed byte of " + tensor.name + " changed no image: the test does not reach it");
    }
}

// The same tensors at other offsets of the file, in another file or in
// another model, are the same images: a fine-tune's unchanged ones and its
// base model's share one.
void testKeyIgnoresWhereSourcesAre(const Setup &setup, metal::MetalBackend &backend, const Kind &kind, Role role) {
  setup.model(kind, "base");
  setup.model(kind, "reordered", true);
  require(keysOf(setup, backend, kind, role, "reordered") == keysOf(setup, backend, kind, role, "base"),
          "the offsets of the sources changed the keys");
}

// The images of the target checkpoint in a directory, planned as an engine
// plans them.
struct Planned final {
  explicit Planned(const std::filesystem::path &target)
      : checkpoint(target), images(model::safetensors::planImages(checkpoint, syntheticTarget())) {}
  model::SafetensorsCheckpoint checkpoint;
  std::vector<model::gguf::Image> images;
};

// The plan a loader makes of an image.
model::ImagePlan planOf(const model::gguf::Image &image) {
  model::gguf::Description description = model::gguf::describe(image);
  return {"target/" + image.name, image.magic,       image.layer, image.type, image.bytes, {}, image.residentBytes,
          std::move(description.text), std::move(description.sources)};
}

// An image's key covers every field of its plan: its header and size as
// ImagePlan states them, and each field of its sections the description
// states, one at a time.
void testKeyCoversThePlan(const Setup &setup) {
  setup.model(kindNamed("mlx"), "base");
  const Planned planned(setup.root() / "base" / "target");
  const auto shared = keyer(setup, setup.root() / "base", Role::Target);
  const model::gguf::Image &layer = planned.images.at(0), &experts = planned.images.at(1);
  require(layer.name == "layer-0.bin" && experts.name == "experts-0.bin" && !layer.fills.empty() &&
              !layer.copies.empty() && !layer.repacks.empty() && !experts.repacks.empty(),
          "the planned images are not those a plan test changes");
  const std::string baseline = shared->key(planOf(layer));
  const auto changes = [&](const std::string &field, const model::ImagePlan &plan) {
    require(shared->key(plan) != baseline, "the key does not cover " + field);
  };
  const auto header = [&](const std::string &field, auto change) {
    model::ImagePlan plan = planOf(layer);
    change(plan);
    changes(field, plan);
  };
  header("the component", [](model::ImagePlan &plan) { plan.component += "x"; });
  header("the magic", [](model::ImagePlan &plan) { plan.magic[0] ^= 1; });
  header("the layer", [](model::ImagePlan &plan) { ++plan.layer; });
  header("the type", [](model::ImagePlan &plan) { ++plan.type; });
  header("the size", [](model::ImagePlan &plan) { plan.bytes += 16384; });
  const auto section = [&](const std::string &field, auto change) {
    model::gguf::Image image = layer;
    change(image);
    changes(field, planOf(image));
  };
  section("a fill's offset", [](model::gguf::Image &image) { ++image.fills[0].offset; });
  section("a fill's bytes", [](model::gguf::Image &image) { image.fills[0].bytes[0] ^= 1; });
  section("a fill's length", [](model::gguf::Image &image) { image.fills[0].bytes.push_back(0); });
  section("a copy's destination", [](model::gguf::Image &image) { ++image.copies[0].destination; });
  section("a copy's conversion", [](model::gguf::Image &image) {
    image.copies[0].conversion = image.copies[0].conversion == model::gguf::Conversion::None
                                     ? model::gguf::Conversion::WidenToFloat32
                                     : model::gguf::Conversion::None;
  });
  const std::string expertsBaseline = shared->key(planOf(experts));
  for (const model::gguf::Image *image : {&layer, &experts}) {
    const std::string at = std::string(image == &layer ? "a layer's" : "an experts image's") + " repack's ";
    const std::string before = image == &layer ? baseline : expertsBaseline;
    const auto differs = [&](const std::string &field, auto change) {
      model::gguf::Image changed = *image;
      change(changed.repacks[0]);
      require(shared->key(planOf(changed)) != before, "the key does not cover " + at + field);
    };
    differs("format", [](model::gguf::Repack &repack) { ++repack.format; });
    differs("rows", [](model::gguf::Repack &repack) { repack.rows += 256; });
    differs("columns", [](model::gguf::Repack &repack) { repack.columns += 256; });
    differs("plane0", [](model::gguf::Repack &repack) { ++repack.plane0; });
    differs("plane1", [](model::gguf::Repack &repack) { ++repack.plane1; });
    differs("meta", [](model::gguf::Repack &repack) { ++repack.meta; });
  }
  // Each field of the rows an image reads, of a copy's and of a repack's.
  using Rows = model::gguf::TensorRows;
  const model::WeightSource *file = planned.checkpoint.require("language_model.model.norm.weight").file;
  const std::vector<std::pair<std::string, std::function<void(Rows &)>>> fields{
      {"type", [](Rows &rows) { ++rows.type; }},
      {"rows", [](Rows &rows) { ++rows.rows; }},
      {"row bytes", [](Rows &rows) { ++rows.rowBytes; }},
      {"first row", [](Rows &rows) { ++rows.first; }},
      {"order", [](Rows &rows) { rows.order.from = 7; }},
      {"order's head rows", [](Rows &rows) { rows.order.headRows = 3; }},
      {"order's key heads", [](Rows &rows) { rows.order.keyHeads = 3; }},
      {"order's value heads per key", [](Rows &rows) { rows.order.valueHeadsPerKey = 3; }},
      {"scale's rows per value", [file](Rows &rows) { rows.scale = {file, 0, 4, 1, false, false}; }},
      {"scale's width", [](Rows &rows) { rows.scale.bfloat16 = true; }},
      {"scale's reciprocal", [](Rows &rows) { rows.scale.reciprocal = true; }},
      // The BF16 or F16 rows a GGUF's projection is quantized into Q8_0 from.
      {"float type", [](Rows &rows) { rows.ggufFloatType = model::ggml::kF16; }},
      {"float row bytes", [](Rows &rows) { rows.ggufFloatRowBytes += 2; }},
  };
  for (const auto &[field, change] : fields) {
    section("a copy's " + field, [&](model::gguf::Image &image) { change(image.copies[0].source); });
    section("a repack source's " + field, [&](model::gguf::Image &image) { change(image.repacks[0].sources[0]); });
  }
  // A tensor's name does not decide what is written.
  model::gguf::Image renamed = layer;
  renamed.copies[0].source.name += "x";
  require(shared->key(planOf(renamed)) == baseline, "a tensor's name changed the key");
}

// Every rows of an image's plan: its copies' and its repacks' sources, in
// plan order.
std::vector<model::gguf::TensorRows *> rowsOf(model::gguf::Image &image) {
  std::vector<model::gguf::TensorRows *> rows;
  for (model::gguf::Copy &copy : image.copies) rows.push_back(&copy.source);
  for (model::gguf::Repack &repack : image.repacks)
    for (model::gguf::TensorRows &source : repack.sources) rows.push_back(&source);
  return rows;
}

// The tensor scales of NVFP4 and FP8 rows, which Model Optimizer,
// compressed-tensors and llama.cpp store, and the size of llama.cpp's NVFP4
// rows decide what their images hold: a key covers each.
void testKeyCoversTheScales(const Setup &setup) {
  for (const std::string name : {"model-optimizer", "compressed-tensors", "gguf-nvfp4"}) {
    const Kind &kind = kindNamed(name);
    setup.model(kind, name);
    const std::filesystem::path root = setup.root() / name;
    const auto shared = keyer(setup, root, Role::Target);
    std::unique_ptr<model::SafetensorsCheckpoint> checkpoint;
    std::unique_ptr<model::WeightSource> source;
    std::vector<model::gguf::Image> images;
    if (kind.target == model::TargetSource::Gguf) {
      source = std::make_unique<model::WeightSource>(model::findTargetGgufs(root / "target").front());
      images = model::gguf::planImages(model::GgufFile(*source), syntheticTarget().ggufArchitecture, syntheticTarget());
    } else {
      checkpoint = std::make_unique<model::SafetensorsCheckpoint>(root / "target");
      images = model::safetensors::planImages(*checkpoint, syntheticTarget());
    }
    using Rows = model::gguf::TensorRows;
    const std::vector<std::pair<std::string, std::function<void(Rows &)>>> fields{
        {"scale's rows per value", [](Rows &rows) { ++rows.scale.rowsPerValue; }},
        {"scale's width", [](Rows &rows) { rows.scale.bfloat16 = !rows.scale.bfloat16; }},
        {"scale's reciprocal", [](Rows &rows) { rows.scale.reciprocal = !rows.scale.reciprocal; }},
        {"scale's size", [](Rows &rows) { --rows.scale.bytes; }},
        {"NVFP4 row bytes", [](Rows &rows) { --rows.ggufNvfp4RowBytes; }},
    };
    // The kinds of rows that have a scale, or llama.cpp's NVFP4 rows.
    const std::vector<std::pair<std::string, std::function<bool(const Rows &)>>> kinds{
        {"NVFP4 rows with a scale", [](const Rows &rows) { return rows.scale.rowsPerValue && rows.type == QUANT_NVFP4_TYPE; }},
        {"FP8 rows with a scale", [](const Rows &rows) { return rows.scale.rowsPerValue && rows.type == QUANT_FP8_TYPE; }},
        {"llama.cpp NVFP4 rows", [](const Rows &rows) { return rows.ggufNvfp4RowBytes != 0; }},
    };
    for (const auto &[what, has] : kinds) {
      bool found = false;
      for (const model::gguf::Image &image : images) {
        model::gguf::Image copy = image;
        const std::vector<Rows *> all = rowsOf(copy);
        const auto at = std::find_if(all.begin(), all.end(), [&](Rows *rows) { return has(*rows); });
        if (at == all.end()) continue;
        found = true;
        const size_t index = size_t(at - all.begin());
        const std::string baseline = shared->key(planOf(image));
        for (const auto &[field, change] : fields) {
          // A field of the rows the kind has.
          const bool scale = field.starts_with("scale"), nvfp4 = field.starts_with("NVFP4");
          if ((scale && !(*at)->scale.rowsPerValue) || (nvfp4 && !(*at)->ggufNvfp4RowBytes)) continue;
          model::gguf::Image changed = image;
          change(*rowsOf(changed)[index]);
          require(shared->key(planOf(changed)) != baseline,
                  "the key of " + name + "'s " + image.name + " does not cover the " + field + " of its " + what);
        }
        break;
      }
      const bool expected = what.starts_with("llama") ? kind.target == model::TargetSource::Gguf
                                                       : !what.starts_with("FP8") || kind.target != model::TargetSource::Gguf;
      require(found == expected, name + (found ? " has " : " has no ") + what);
    }
  }
}

// What a vision tower's image reads of its plan decides what it holds: a key
// covers the layout's padding and depth, how a tensor's values are stored
// and whether the patch embedding is reordered.
void testKeyCoversTheVisionPlan(const Setup &setup) {
  const Kind &kind = kindNamed("vision-mlx");
  setup.model(kind, "base");
  const std::filesystem::path shard = setup.root() / "base" / "vision" / "model.safetensors";
  const auto key = [&](const std::string &name, const splash::ops::VisionLayout &layout) {
    const std::filesystem::path root = setup.root() / name;
    const std::string key =
        keyer(setup, root, Role::Vision)->key(model::VisionLoader(root / "vision", kind.vision, layout).image());
    require(key.size() == 64, "the vision tower of " + name + " has no key");
    return key;
  };
  const std::string baseline = key("base", syntheticVision());
  splash::ops::VisionLayout padded = syntheticVision(), shallow = syntheticVision();
  padded.paddedIntermediateSize = 224;
  shallow.depth = 1;
  require(key("base", padded) != baseline, "the key does not cover the padding of the layout");
  require(key("base", shallow) != baseline, "the key does not cover the depth of the layout");
  // The same bytes of every tensor under another model's names and with
  // another dtype: only the plan's reading of them differs.
  const auto edited = [&](const std::string &name, const std::vector<std::pair<std::string, std::string>> &changes) {
    std::filesystem::copy(setup.root() / "base", setup.root() / name, std::filesystem::copy_options::recursive);
    std::vector<uint8_t> bytes = splash::test::readFile(setup.root() / name / "vision" / "model.safetensors");
    uint64_t length = 0;
    std::memcpy(&length, bytes.data(), sizeof length);
    std::string header(reinterpret_cast<const char *>(bytes.data()) + sizeof length, length);
    for (const auto &[from, to] : changes)
      for (size_t at = header.find(from); at != std::string::npos; at = header.find(from, at + to.size()))
        header.replace(at, from.size(), to);
    const uint64_t size = header.size();
    std::vector<uint8_t> file(reinterpret_cast<const uint8_t *>(&size), reinterpret_cast<const uint8_t *>(&size) + 8);
    file.insert(file.end(), header.begin(), header.end());
    file.insert(file.end(), bytes.begin() + sizeof length + std::ptrdiff_t(length), bytes.end());
    splash::test::writeFile(setup.root() / name / "vision" / "model.safetensors", file);
  };
  edited("retyped", {{R"("vision_tower.patch_embed.proj.weight":{"dtype":"BF16")",
                      R"("vision_tower.patch_embed.proj.weight":{"dtype":"F16")"}});
  require(key("retyped", syntheticVision()) != baseline, "the key does not cover a tensor's dtype");
  // transformers' names: the patch embedding is stored in the image's order
  // already, which a writer copies as it is.
  const std::string patch = std::to_string(syntheticVision().hiddenSize);
  edited("transformers", {{"vision_tower.", "model.visual."},
                          {"\"shape\":[" + patch + ",2,16,16,3]", "\"shape\":[" + patch + ",3,2,16,16]"}});
  require(key("transformers", syntheticVision()) != baseline, "the key does not cover how the patch embedding is read");
}

// The code that writes an image is part of its key: the engine's code, as the
// process runs it, and the metallib it loaded.
void testKeyCoversTheWriters(const Setup &setup, const std::string &self, const std::string &production,
                             const std::string &other) {
  setup.checkpoint("base", targetTensors());
  metal::MetalBackend backend(production), different(other);
  const std::string identity = model::writerIdentity(backend);
  require(identity.size() == 64 && identity == model::writerIdentity(self, backend),
          "a process's writer identity is not that of its executable file");
  const std::filesystem::path copies = setup.registry().parent_path() / "copies";
  std::filesystem::create_directory(copies);
  // The metallib: another library, or the same one at another path.
  require(model::writerIdentity(self, different) != identity, "another metallib did not change the writer identity");
  std::filesystem::copy_file(production, copies / "copy.metallib");
  require(model::writerIdentity(self, metal::MetalBackend(copies / "copy.metallib")) == identity,
          "the same metallib at another path changed the writer identity");
  // The engine's code: a byte of its code, or one of the file that is not
  // code.
  std::vector<uint8_t> code = splash::test::readFile(self);
  const auto changed = [&](size_t at) {
    std::vector<uint8_t> bytes = code;
    bytes.at(at) ^= 0x5a;
    splash::test::writeFile(copies / "engine", bytes);
    return model::writerIdentity(copies / "engine", backend);
  };
  require(changed(code.size() / 8) != identity, "a changed byte of the engine's code did not change its identity");
  require(changed(code.size() - 1) == identity, "a byte of the engine's file that is not code changed its identity");
  // The identity reaches the key.
  const Kind &mlx = kindNamed("mlx");
  require(keysOf(setup, backend, mlx, Role::Target, "base", identity) !=
              keysOf(setup, backend, mlx, Role::Target, "base", changed(code.size() / 8)),
          "the writer identity does not reach the keys");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 7 && std::string_view(argv[1]) == "engine") return runEngine(argv);
    if (argc != 3 && argc != 4) {
      std::cerr << "usage: shared-weights PRODUCTION_METALLIB OTHER_METALLIB [TEST]\n";
      return 2;
    }
    const std::string only = argc == 4 ? argv[3] : "";
    const std::string self = std::filesystem::canonical(argv[0]).string();
    signal(SIGPIPE, SIG_IGN);
    // A test that waits for ever fails instead; each test has its own limit.
    signal(SIGALRM, watchdog);
    metal::MetalBackend production(argv[1]);
    const auto selected = [&](const std::string &name, unsigned seconds) {
      if (name.find(only) == std::string::npos) return false;
      std::cerr << "shared-weights: " << name << '\n';
      alarm(seconds);
      return true;
    };
    // Each loader: its images are shared between processes and keyed by what
    // determines them.
    const std::vector<std::pair<std::string, Role>> loaders{
        {"mlx", Role::Target},     {"mlx-formats", Role::Target}, {"model-optimizer", Role::Target},
        {"compressed-tensors", Role::Target}, {"gguf", Role::Target}, {"gguf-nvfp4", Role::Target},
        {"mlx", Role::Draft},      {"vision-mlx", Role::Vision},  {"vision-mmproj", Role::Vision}};
    for (const auto &[name, role] : loaders) {
      const Kind &kind = kindNamed(name);
      const std::string at = name + " " + directoryOf(role);
      try {
        if (selected(at + " processes", 300)) testProcessesShare(Setup(self, argv[1]), kind, role);
        if (selected(at + " keys", 900)) testKeyCoversSourceBytes(Setup(self, argv[1]), production, kind, role);
        if (selected(at + " offsets", 120))
          testKeyIgnoresWhereSourcesAre(Setup(self, argv[1]), production, kind, role);
      } catch (const std::exception &error) {
        throw std::runtime_error(at + ": " + error.what());
      }
    }
    if (selected("concurrent loads", 120)) testConcurrentLoadsWriteEachImageOnce(Setup(self, argv[1]));
    if (selected("fine-tune", 120)) testFineTuneSharesWhatMatches(Setup(self, argv[1]));
    if (selected("unshared", 120)) testUnsharedCreatesNothing(Setup(self, argv[1]));
    if (selected("release and restore", 120)) testReleaseAndRestoreKeepOtherHolders(Setup(self, argv[1]));
    if (selected("restore admission", 120)) testRestoreAdmitsReceivedImagesInFull(Setup(self, argv[1]));
    if (selected("restore refused", 120)) testRestoreRefusedPartWay(Setup(self, argv[1]));
    if (selected("writer out of memory", 120)) testWriterOutOfMemoryLeavesNoLockOrMapping(Setup(self, argv[1]));
    if (selected("stopped writer", 120)) testStoppedWriterIsWaitedForBoundedly(Setup(self, argv[1]));
    if (selected("killed holder", 120)) testKilledHolderLeavesOthersServing(Setup(self, argv[1]));
    if (selected("stopped holder", 180)) testStoppedHolderIsPassedOverOncePerPass(Setup(self, argv[1]));
    if (selected("unrecorded", 120)) testUnrecordedSourcesAreNotShared(Setup(self, argv[1]));
    if (selected("running code", 120)) testWriterIsTheRunningCode(Setup(self, argv[1]));
    if (selected("plan keys", 120)) testKeyCoversThePlan(Setup(self, argv[1]));
    if (selected("scale keys", 120)) testKeyCoversTheScales(Setup(self, argv[1]));
    if (selected("vision plan keys", 120)) testKeyCoversTheVisionPlan(Setup(self, argv[1]));
    if (selected("writer keys", 120)) testKeyCoversTheWriters(Setup(self, argv[1]), self, argv[1], argv[2]);
    if (selected("config", 120)) testConfigTakesTheRecordedDigests(Setup(self, argv[1]));
    if (selected("user place", 120)) testUserPlace();
    std::cout << "PASS shared-weights" << (only.empty() ? "" : " " + only) << '\n';
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
