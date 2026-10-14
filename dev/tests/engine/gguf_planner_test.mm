// The GGUF image planner on the small dense and MoE targets and a rotated
// dense one: where each tensor goes and in which row order, which F32 tensors
// it narrows to bf16, and the sources it refuses; and the safetensors planner's
// split of a MoE layer's routed experts from its other tensors.
//   gguf-planner
#include "GgufFixtures.hpp"
#include "TestCheckpoint.hpp"
#include "metal/abi/Gguf.h"
#include "model/Qwen3_6Moe.hpp"
#include "model/Qwen3_8.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/SafetensorsImage.hpp"

#include <optional>

using namespace gguf_fixtures;

namespace {

// Q8_1, a type the loader does not read.
constexpr uint32_t kQ8_1 = ggmlType("Q8_1");

// The images of the GGUF at path, planned for geometry; the planner's error
// when it refuses the file.
struct Plan {
  std::vector<model::gguf::Image> images;
  std::optional<std::string> error;
};

Plan plan(const std::filesystem::path &path, const model::QwenTargetDimensions &geometry) {
  try {
    model::WeightSource source(path);
    const model::GgufFile gguf(source);
    return {model::gguf::planImages(gguf, geometry), std::nullopt};
  } catch (const model::GgufError &error) {
    return {{}, error.what()};
  }
}

// Whether a refusal names every part.
bool names(const Plan &result, std::initializer_list<std::string_view> parts) {
  return result.error && std::all_of(parts.begin(), parts.end(), [&](std::string_view part) {
           return result.error->find(part) != std::string::npos;
         });
}

// The plan's copy of the named tensor in any image, or nullptr.
const model::gguf::Copy *copyIn(const Plan &result, const std::string &name) {
  for (const model::gguf::Image &image : result.images)
    if (const auto *copy = copyOf(image, name)) return copy;
  return nullptr;
}

// Value heads regrouped from llama.cpp's tiled order from row `from` on.
bool grouped(const model::gguf::RowOrder &order, uint64_t from, uint32_t headRows,
             const model::QwenTargetDimensions &g) {
  return order.from == from && order.headRows == headRows && order.keyHeads == g.gdnKeyHeads &&
         order.valueHeadsPerKey == g.gdnValueHeads / g.gdnKeyHeads;
}

// A copy of the tensor's rows as the GGUF stores them.
bool asStored(const model::gguf::Copy *copy, const test_gguf::Bytes &data) {
  return copy && copy->conversion == model::gguf::Conversion::None && copy->source.order.from == UINT64_MAX &&
         copy->source.rows * copy->source.rowBytes == data.size();
}

void checkDense(const std::filesystem::path &directory) {
  SmallTarget target = smallTarget(false);
  const model::QwenTargetDimensions &g = target.geometry;
  const auto path = directory / "dense.gguf";
  writeGguf(path, target.tensors, g);
  const Plan dense = plan(path, g);
  check(!dense.error, "planner plans the dense target" + (dense.error ? ": " + *dense.error : ""));
  if (dense.error) return;
  const model::gguf::Image &gdn = dense.images[0];

  for (const Tensor &tensor : target.tensors)
    if (isNorm(tensor.name))
      check(asStored(copyIn(dense, tensor.name), tensor.data), "planner keeps F32 values as stored: " + tensor.name);

  const uint32_t keyRows = g.convolutionDimension - g.gdnValueHeads * g.gdnHeadDimension;
  const auto *convolution = copyOf(gdn, "blk.0.ssm_conv1d.weight");
  check(convolution && convolution->conversion == model::gguf::Conversion::NarrowToBfloat16 &&
            convolution->source.rows == g.convolutionDimension &&
            grouped(convolution->source.order, keyRows, g.gdnHeadDimension, g),
        "planner narrows the convolution to bf16 and groups its value heads");
  const auto *decay = copyOf(gdn, "blk.0.ssm_a");
  check(decay && decay->conversion == model::gguf::Conversion::None && grouped(decay->source.order, 0, 1, g),
        "planner keeps the decay F32 and groups its value heads");
  const auto *bias = copyOf(gdn, "blk.0.ssm_dt.bias");
  check(bias && bias->conversion == model::gguf::Conversion::NarrowToBfloat16 && grouped(bias->source.order, 0, 1, g),
        "planner narrows the time bias to bf16 and groups its value heads");

  // A token table of every type the gathers read, F16 and BF16 among them, but llama.cpp's NVFP4, whose rows the
  // writer converts: copied as stored.
  std::vector<uint32_t> tableTypes;
  for (uint32_t format = 0; format < GGUF_FMT_COUNT; ++format)
    if (const std::optional<uint32_t> type = ggufType(format);
        type && gguf_embedding_format(format) && *type != model::ggml::kNVFP4)
      tableTypes.push_back(*type);
  for (const uint32_t type : tableTypes) {
    std::vector<Tensor> tensors = target.tensors;
    Tensor &table = tensorNamed(tensors, "token_embd.weight");
    const model::GgmlTypeTraits &traits = *model::ggmlTypeTraits(type);
    table.type = type;
    table.data.resize(uint64_t{g.vocabularySize} * g.hiddenSize / traits.blockElements * traits.blockBytes);
    writeGguf(path, tensors, g);
    const Plan tables = plan(path, g);
    check(!tables.error && asStored(copyIn(tables, "token_embd.weight"), table.data),
          "planner copies a token table as stored: " + model::ggmlTypeName(type) +
              (tables.error ? ": " + *tables.error : ""));
  }

  // A key or value projection of other than attentionKvHeads heads.
  for (const std::string name : {"blk.1.attn_k.weight", "blk.1.attn_v.weight"}) {
    for (uint64_t rows : {uint64_t{g.attentionHeadDimension}, uint64_t{3} * g.attentionHeadDimension}) {
      std::vector<Tensor> malformed = target.tensors;
      Tensor &t = tensorNamed(malformed, name);
      t.dims[1] = rows;
      t.data.resize(rows * gguf_reference::rowBytes(Fmt(gguf_format_of(t.type)), g.hiddenSize));
      writeGguf(path, malformed, g);
      check(names(plan(path, g), {"shape", name}),
            "planner refuses mismatched KV rows by name: " + name + " rows=" + std::to_string(rows));
    }
  }
}

// alpha/beta of any one GGUF tensor type of a format but BF16 and F16
// (llama.cpp's NVFP4 as NVFP4), on either architecture: one 256-row tensor of
// that format, the beta then the alpha rows at its native row bytes in grouped
// head order; a pair of two types or of BF16 or F16, one F32 tensor of the
// beta then the alpha rows, each decoded unless F32; a type the loader does
// not read, refused by name.
void checkQuantizedAlphaBeta(const std::filesystem::path &directory) {
  for (bool moe : {false, true}) {
    const SmallTarget target = smallTarget(moe);
    const model::QwenTargetDimensions &g = target.geometry;
    const auto path = directory / "alpha-beta.gguf";
    const auto tensorsWith = [&](uint32_t betaType, uint32_t alphaType) {
      std::vector<Tensor> tensors = target.tensors;
      for (const char *name : {"blk.0.ssm_beta.weight", "blk.0.ssm_alpha.weight"}) {
        Tensor &tensor = tensorNamed(tensors, name);
        tensor.type = tensor.name.ends_with("_beta.weight") ? betaType : alphaType;
        const model::GgmlTypeTraits &traits = *model::ggmlTypeTraits(tensor.type);
        tensor.data.resize(uint64_t{g.gdnValueHeads} * g.hiddenSize / traits.blockElements * traits.blockBytes);
      }
      return tensors;
    };
    for (uint32_t format = 0; format < GGUF_FMT_COUNT; ++format) {
      const std::optional<uint32_t> stored = ggufType(format);
      if (!stored || format == GGUF_FMT_BF16 || format == GGUF_FMT_F16) continue;
      const uint32_t type = *stored;
      writeGguf(path, tensorsWith(type, type), g);
      const Plan result = plan(path, g);
      const auto *pair = result.error ? nullptr : repackOf(result.images[0], "blk.0.ssm_beta.weight");
      // Rows of the format's native blocks (llama.cpp's NVFP4 as NVFP4's).
      const auto heads = [&](const model::gguf::TensorRows &rows, const char *name) {
        return rows.name == name && rows.type == kQuantFormats[format].ggml_type && rows.rows == g.gdnValueHeads &&
               rows.rowBytes == gguf_reference::rowBytes(Fmt(format), g.hiddenSize) && grouped(rows.order, 0, 1, g);
      };
      check(pair && pair->format == format && pair->rows == QUANT_TILE_ROWS && pair->columns == g.hiddenSize &&
                pair->sources.size() == 2 && heads(pair->sources[0], "blk.0.ssm_beta.weight") &&
                heads(pair->sources[1], "blk.0.ssm_alpha.weight"),
            "planner alpha/beta: one 256-row " + model::ggmlTypeName(type) + " tensor of beta then alpha rows (" +
                model::gguf::architecture(g.ffnKind) + ")" + (result.error ? ": " + *result.error : ""));
    }
    using model::gguf::Conversion;
    const auto conversion = [](uint32_t type) {
      return type == model::ggml::kF32 ? Conversion::None : Conversion::DequantizeToFloat32;
    };
    const uint64_t floatRows = uint64_t{g.gdnValueHeads} * g.hiddenSize * sizeof(float);
    for (const auto &[beta, alpha] : {std::pair{kIQ4_XS, kQ8_0}, std::pair{kQ4_K, model::ggml::kF32},
                                      std::pair{model::ggml::kBF16, model::ggml::kNVFP4},
                                      std::pair{model::ggml::kF32, model::ggml::kBF16},
                                      std::pair{model::ggml::kF16, kQ8_0},
                                      std::pair{model::ggml::kBF16, model::ggml::kBF16},
                                      std::pair{model::ggml::kF16, model::ggml::kF16}}) {
      writeGguf(path, tensorsWith(beta, alpha), g);
      const Plan result = plan(path, g);
      const auto *b = copyIn(result, "blk.0.ssm_beta.weight"), *a = copyIn(result, "blk.0.ssm_alpha.weight");
      check(b && a && b->conversion == conversion(beta) && a->conversion == conversion(alpha) &&
                a->destination == b->destination + floatRows && grouped(b->source.order, 0, 1, g) &&
                grouped(a->source.order, 0, 1, g) && !repackOf(result.images[0], "blk.0.ssm_beta.weight"),
            "planner widens alpha/beta into one F32 tensor: " + model::ggmlTypeName(beta) + " beta, " +
                model::ggmlTypeName(alpha) + " alpha" + (result.error ? ": " + *result.error : ""));
    }
    writeGguf(path, tensorsWith(kQ8_0, kQ8_1), g);
    check(names(plan(path, g), {"blk.0.ssm_alpha.weight (Q8_1)"}) &&
              !names(plan(path, g), {"blk.0.ssm_beta.weight"}),
          "planner refuses an alpha of a type it does not read by its name alone");
  }
}

void checkMoe(const std::filesystem::path &directory) {
  SmallTarget target = smallTarget(true);
  const model::QwenTargetDimensions &g = target.geometry;
  const auto path = directory / "moe.gguf";

  model::QwenTargetDimensions wrongExperts = g;
  wrongExperts.experts = 5;
  writeGguf(path, target.tensors, wrongExperts);
  check(names(plan(path, g), {"expert_count"}), "planner names a metadata mismatch");
  model::QwenTargetDimensions dense = g;
  dense.ffnKind = model::QwenFfnKind::Dense;
  dense.intermediateSize = g.expertIntermediateSize;
  writeGguf(path, target.tensors, dense);
  check(names(plan(path, g),
              {"architecture", model::gguf::architecture(dense.ffnKind), model::gguf::architecture(g.ffnKind)}),
        "planner checks the architecture against the model's target");

  writeGguf(path, target.tensors, g);
  const Plan moe = plan(path, g);
  check(!moe.error, "planner plans the MoE target" + (moe.error ? ": " + *moe.error : ""));
  if (moe.error) return;
  // A layer's routed experts have an image of their own after the layer's, under a header type that is neither a
  // layer's (0 or 1) nor the head's (2).
  const bool split = moe.images.size() == 4 && moe.images[0].name == "layer-0.bin" &&
                     moe.images[1].name == "experts-0.bin" && moe.images[2].name == "head.bin" &&
                     moe.images[3].name == "embedding.bin" && moe.images[1].layer == 0 && moe.images[1].type == 3 &&
                     moe.images[0].type != moe.images[1].type;
  check(split, "planner writes a MoE layer's routed experts into the image experts-N after the layer's");
  if (!split) return;
  const model::gguf::Image &layer = moe.images[0], &experts = moe.images[1];
  const auto *beta = copyOf(layer, "blk.0.ssm_beta.weight"), *alpha = copyOf(layer, "blk.0.ssm_alpha.weight");
  check(beta && alpha && alpha->destination == beta->destination + target.data(beta->source.name).size() &&
            grouped(beta->source.order, 0, 1, g) && grouped(alpha->source.order, 0, 1, g) &&
            beta->conversion == model::gguf::Conversion::None && alpha->conversion == model::gguf::Conversion::None,
        "planner F32 alpha/beta tensor: beta then alpha rows in grouped order");
  for (const std::string name : {"blk.0.ffn_gate_inp.weight", "blk.0.ffn_gate_inp_shexp.weight"})
    check(asStored(copyOf(layer, name), target.data(name)), "planner copies the F32 tensor as stored: " + name);

  // Each quantized tensor is repacked on its own, a 3-D expert tensor as one tensor of experts * N rows: the
  // routed experts' in the experts image, gate, up, then down, the others in the layer's.
  size_t quantized = 0, routed = 0;
  std::vector<std::string> routedOrder;
  for (const Tensor &tensor : target.tensors) {
    if (!tensor.name.starts_with("blk.0.") || gguf_format_of(tensor.type) == GGUF_FMT_COUNT) continue;
    const bool inExperts = tensor.name.ends_with("_exps.weight");
    ++(inExperts ? routed : quantized);
    const uint64_t rows = tensor.dims[1] * (tensor.dims.size() > 2 ? tensor.dims[2] : 1);
    const auto *repack = repackOf(inExperts ? experts : layer, tensor.name);
    check(repack && repack->sources.size() == 1 && repack->rows == rows && repack->columns == tensor.dims[0] &&
              repack->sources[0].rows == rows && !repackOf(inExperts ? layer : experts, tensor.name),
          "planner repacks " + tensor.name + " as " + std::to_string(rows) + " rows of " +
              std::to_string(tensor.dims[0]) + " in the " + (inExperts ? "experts" : "layer") + " image");
  }
  for (const model::gguf::Repack &repack : experts.repacks) routedOrder.push_back(repack.sources.front().name);
  check(layer.repacks.size() == quantized && experts.repacks.size() == routed && routed == 3 &&
            routedOrder == std::vector<std::string>{"blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight",
                                                    "blk.0.ffn_down_exps.weight"},
        "planner repacks only the layer's other quantized tensors in its image, and the routed experts' gate, up and "
        "down in the experts image");
  check(experts.copies.empty(), "planner copies nothing into the experts image but through its repacks");
}

// llama.cpp's fused ffn_gate_up_exps in the MoE target.
void checkFusedExperts(const std::filesystem::path &directory) {
  SmallTarget target = smallTarget(true);
  const model::QwenTargetDimensions &g = target.geometry;
  const auto path = directory / "moe-fused.gguf";
  // llama.cpp's fused ffn_gate_up_exps: the gate tensor takes each expert's first width rows of it, the up tensor
  // the next width; one of a type the loader does not read is refused by name.
  const uint64_t width = g.expertIntermediateSize;
  std::vector<Tensor> fused = withFusedGateUp(target.tensors, 0, g.experts);
  writeGguf(path, fused, g);
  const Plan fusedPlan = plan(path, g);
  std::vector<const model::gguf::Repack *> halves;
  bool apart = !fusedPlan.error && fusedPlan.images.size() == 4;
  if (apart) {
    for (const model::gguf::Repack &repack : fusedPlan.images[1].repacks)
      if (repack.sources.front().name == "blk.0.ffn_gate_up_exps.weight") halves.push_back(&repack);
    for (const model::gguf::Repack &repack : fusedPlan.images[0].repacks)
      apart = apart && repack.sources.front().name != "blk.0.ffn_gate_up_exps.weight" &&
                  repack.sources.front().name != "blk.0.ffn_down_exps.weight";
  }
  bool fusedSplit = apart && halves.size() == 2 && fusedPlan.images[1].repacks.size() == 3 &&
                    fusedPlan.images[1].repacks.back().sources.front().name == "blk.0.ffn_down_exps.weight";
  for (size_t half = 0; fusedSplit && half < 2; ++half) {
    fusedSplit = halves[half]->rows == g.experts * width && halves[half]->sources.size() == g.experts;
    for (uint64_t expert = 0; fusedSplit && expert < g.experts; ++expert) {
      const model::gguf::TensorRows &rows = halves[half]->sources[expert];
      fusedSplit = rows.first == (2 * expert + half) * width && rows.rows == width &&
                   rows.rowBytes == gguf_reference::rowBytes(Fmt(gguf_format_of(rows.type)), g.hiddenSize);
    }
  }
  check(fusedSplit,
        "planner splits llama.cpp's fused gate/up experts into each expert's gate and up rows, in the experts image "
        "with the down experts" + (fusedPlan.error ? ": " + *fusedPlan.error : ""));
  tensorNamed(fused, "blk.0.ffn_gate_up_exps.weight").type = kQ8_1;
  tensorNamed(fused, "blk.0.ffn_gate_up_exps.weight").data.resize(2 * g.experts * width * g.hiddenSize / 32 * 36);
  writeGguf(path, fused, g);
  check(names(plan(path, g), {"blk.0.ffn_gate_up_exps.weight (Q8_1)"}),
        "planner refuses a fused gate/up experts tensor of a type it does not read by name");
}

// A small target of Layout whose projections are whole 256-row tiles: four
// layers, the last of them full attention, and a MoE's 256 experts.
template <class Layout> Layout smallLayout() {
  Layout target;
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
  if (target.ffnKind == model::QwenFfnKind::SparseMoe) target.expertIntermediateSize = 256;
  else target.intermediateSize = 256;
  target.fullAttentionPeriod = 4;
  target.hiddenCaptureLayers.fill(target.layers - 1);
  return target;
}

// Whether the image reads a tensor whose name holds `part`.
bool readsTensor(const model::gguf::Image &image, std::string_view part) {
  const auto holds = [&](const model::gguf::TensorRows &rows) { return rows.name.find(part) != std::string::npos; };
  return std::any_of(image.copies.begin(), image.copies.end(),
                     [&](const model::gguf::Copy &copy) { return holds(copy.source); }) ||
         std::any_of(image.repacks.begin(), image.repacks.end(), [&](const model::gguf::Repack &repack) {
           return std::any_of(repack.sources.begin(), repack.sources.end(), holds);
         });
}

// A MoE target of a safetensors checkpoint: each layer's routed experts, MLX's
// stacked tensor of each projection or a transformers checkpoint's modules of
// their own, are the image after the layer's, gate, up, then down; the layer's
// keeps the router and the shared expert; a dense target has no such image.
void checkSafetensorsMoe(const std::filesystem::path &directory) {
  const auto target = smallLayout<model::Qwen3_6MoeLayout>();
  using splash::test::SyntheticTensor;
  const auto mlxBits = [](const std::string &module) {
    return module.ends_with(".gate") || module.ends_with("shared_expert_gate") ? std::pair{8u, 64u}
                                                                                : std::pair{4u, 64u};
  };
  const auto nvfp4 = [](const std::string &) -> uint32_t { return GGUF_FMT_NVFP4; };
  struct Checkpoint {
    std::string name;
    std::vector<SyntheticTensor> tensors;
    std::string routed; // what a routed expert's tensor name holds
  };
  const std::vector<Checkpoint> checkpoints{
      {"MLX", splash::test::mlxTargetTensors(target, mlxBits), ".switch_mlp."},
      {"Model Optimizer",
       splash::test::floatTargetTensors(target, nvfp4, splash::test::FloatCheckpoint::ModelOptimizer), ".experts."}};
  for (const Checkpoint &checkpoint : checkpoints) {
    const auto root = directory / checkpoint.name;
    splash::test::writeSyntheticShard(root / "model.safetensors", checkpoint.tensors);
    const model::SafetensorsCheckpoint source(root);
    const std::vector<model::gguf::Image> images = model::safetensors::planImages(source, target);
    const std::string what = "planner on a " + checkpoint.name + " MoE target: ";
    bool ordered = images.size() == 2 * target.layers + 2;
    for (uint32_t layer = 0; ordered && layer < target.layers; ++layer) {
      const model::gguf::Image &mixer = images[2 * layer], &experts = images[2 * layer + 1];
      ordered = mixer.name == "layer-" + std::to_string(layer) + ".bin" &&
                experts.name == "experts-" + std::to_string(layer) + ".bin" && experts.layer == layer &&
                experts.type == 3 && experts.magic == mixer.magic && mixer.type != experts.type;
    }
    ordered = ordered && images[images.size() - 2].name == "head.bin" && images.back().name == "embedding.bin";
    check(ordered, what + "each layer is followed by its experts-N image of header type 3, then the head and the "
                          "embedding");
    if (!ordered) continue;
    for (uint32_t layer = 0; layer < target.layers; ++layer) {
      const model::gguf::Image &mixer = images[2 * layer], &experts = images[2 * layer + 1];
      const std::string at = what + "layer " + std::to_string(layer) + " ";
      // Gate and up of experts * width rows each, then down of experts * hidden rows.
      const std::vector<uint64_t> rows{target.experts * target.expertIntermediateSize,
                                       target.experts * target.expertIntermediateSize,
                                       target.experts * target.hiddenSize};
      bool exact = experts.repacks.size() == 3 && experts.copies.empty();
      for (size_t projection = 0; exact && projection < 3; ++projection) {
        const model::gguf::Repack &repack = experts.repacks[projection];
        const std::string name = std::array{"gate_proj", "up_proj", "down_proj"}[projection];
        exact = repack.rows == rows[projection] && !repack.sources.empty() &&
                repack.sources.front().name.find(checkpoint.routed) != std::string::npos &&
                (checkpoint.routed == ".experts." || repack.sources.front().name.ends_with(name)) &&
                std::all_of(repack.sources.begin(), repack.sources.end(), [&](const model::gguf::TensorRows &source) {
                  return source.name.find(checkpoint.routed) != std::string::npos;
                });
      }
      check(exact, at + "experts-N holds exactly the routed gate, up and down experts");
      check(!readsTensor(mixer, checkpoint.routed) && readsTensor(mixer, "shared_expert.gate_proj") &&
                readsTensor(mixer, "mlp.gate"),
            at + "layer-N holds no routed expert but keeps the router and the shared expert");
    }
  }

  const auto dense = smallLayout<model::Qwen3_8Layout>();
  splash::test::writeSyntheticShard(directory / "dense" / "model.safetensors",
                                    splash::test::mlxTargetTensors(dense, mlxBits));
  const model::SafetensorsCheckpoint source(directory / "dense");
  const std::vector<model::gguf::Image> images = model::safetensors::planImages(source, dense);
  check(images.size() == dense.layers + 2 &&
            std::none_of(images.begin(), images.end(),
                         [](const model::gguf::Image &image) { return image.name.starts_with("experts-"); }),
        "planner on a dense safetensors target writes no experts image");
}

// The rotary embedding and norms the kernels compute: a GGUF declares their
// RoPE base, rotated dimensions and RMS epsilon, and no RoPE scaling.
void checkRotary(const std::filesystem::path &directory) {
  const SmallTarget target = smallTarget(false);
  const model::QwenTargetDimensions &g = target.geometry;
  const auto path = directory / "rotary.gguf";
  const auto planned = [&](const std::vector<test_gguf::Key> &keys) {
    splash::test::writeFile(path, test_gguf::file(keys, target.tensors));
    return plan(path, g);
  };
  const std::string scaling = model::gguf::architecture(g.ffnKind) + std::string(".rope.scaling.type");
  std::vector<test_gguf::Key> keys = metadata(g);
  keys.push_back(test_gguf::stringKey(scaling, "none"));
  check(!planned(keys).error, "planner accepts a GGUF that declares no RoPE scaling");
  keys.back() = test_gguf::stringKey(scaling, "yarn");
  check(names(planned(keys), {"rope.scaling.type yarn (expected none)"}), "planner refuses RoPE scaling");

  model::QwenTargetDimensions other = g;
  other.rotaryPairs = 64;
  other.rotaryTheta = 1e6F;
  keys = metadata(other);
  for (test_gguf::Key &key : keys)
    if (key.name.ends_with(".attention.layer_norm_rms_epsilon")) key = test_gguf::float32Key(key.name, 1e-5F);
  check(names(planned(keys), {"rope.dimension_count 128 (expected 64)", "rope.freq_base 1e+06 (expected 1e+07)",
                              "attention.layer_norm_rms_epsilon 1e-05 (expected 1e-06)"}),
        "planner names another rotated width, RoPE base and RMS epsilon");

  keys = metadata(g);
  std::erase_if(keys, [](const test_gguf::Key &key) {
    return key.name.find(".rope.") != std::string::npos || key.name.ends_with(".layer_norm_rms_epsilon");
  });
  check(names(planned(keys), {"rope.dimension_count missing", "rope.freq_base missing",
                              "attention.layer_norm_rms_epsilon missing"}),
        "planner requires the rotary and norm metadata");
}

// A dense target stored for rotated inputs, every rotated input one rotation
// block wide: the rotation must name exactly the tensors the planner repacks,
// so quantized alpha/beta (Q8_0 and IQ4_XS here), whose segment reads the
// rotated input, and not F32 ones, nor BF16 (Prism's) or F16 ones, which load
// as F32.
void checkRotation(const std::filesystem::path &directory) {
  model::QwenTargetDimensions g;
  g.layers = 2;
  g.hiddenSize = GGUF_ROTATION_BLOCK;
  g.vocabularySize = 256;
  g.intermediateSize = GGUF_ROTATION_BLOCK;
  g.gdnKeyHeads = 4;
  g.gdnValueHeads = 16;
  g.gdnHeadDimension = 64;
  g.convolutionDimension = 1536; // q and k of 4 heads, v of 16
  g.attentionWidth = GGUF_ROTATION_BLOCK; // four query heads of 256
  g.attentionKvHeads = 1;
  g.attentionHeadDimension = 256;
  g.rotaryPairs = 32;
  g.rotaryTheta = 1e7F;
  g.fullAttentionPeriod = 2; // layer 1
  const auto path = directory / "rotated.gguf";
  // PQ2_0 projections and token table, alpha/beta of type `alphaBeta`, and a
  // rotation that names every quantized tensor but the table, alpha/beta only
  // when `namesAlphaBeta`.
  const auto planned = [&](uint32_t alphaBeta, bool namesAlphaBeta) {
    TensorTypes types{{"ssm_beta.weight", alphaBeta}, {"ssm_alpha.weight", alphaBeta}};
    for (const char *name : {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_qkv.weight",
                             "attn_gate.weight", "ssm_out.weight", "ffn_gate.weight", "ffn_up.weight",
                             "ffn_down.weight", "output.weight", "token_embd.weight"})
      types[name] = model::ggml::kPQ2_0;
    const std::vector<Tensor> tensors = targetTensors(g, types);
    std::vector<std::string> weights;
    for (const Tensor &tensor : tensors)
      if (gguf_format_of(tensor.type) != GGUF_FMT_COUNT && tensor.name != "token_embd.weight" &&
          (namesAlphaBeta || !(tensor.name.ends_with("_alpha.weight") || tensor.name.ends_with("_beta.weight"))))
        weights.push_back(tensor.name);
    std::vector<test_gguf::Key> keys = metadata(g);
    const auto key = [](const char *name) { return std::string("prism.hadamard.") + name; };
    keys.push_back(test_gguf::uint32Key(key("version"), 1));
    keys.push_back(test_gguf::uint32Key(key("block_size"), GGUF_ROTATION_BLOCK));
    keys.push_back(test_gguf::stringKey(key("transform"), "normalized-sylvester-walsh-hadamard"));
    keys.push_back(test_gguf::stringKey(key("axis"), "input-last-dimension"));
    keys.push_back(test_gguf::stringKey(key("sign_mode"), "explicit"));
    keys.push_back(test_gguf::boolKey(key("gdn_v_grouped"), true));
    keys.push_back(test_gguf::int32ArrayKey(key("sign_widths"), {GGUF_ROTATION_BLOCK}));
    keys.push_back(test_gguf::int32ArrayKey(key("sign_values"), std::vector<int32_t>(GGUF_ROTATION_BLOCK, 1)));
    keys.push_back(test_gguf::stringArrayKey(key("weight_names"), weights));
    keys.push_back(test_gguf::stringArrayKey(key("inverse_weight_names"), {"token_embd.weight"}));
    splash::test::writeFile(path, test_gguf::file(keys, tensors));
    return plan(path, g);
  };
  for (uint32_t type : {model::ggml::kF32, model::ggml::kBF16, model::ggml::kF16}) {
    const Plan floats = planned(type, false);
    check(!floats.error, "planner plans a rotation of " + model::ggmlTypeName(type) + " alpha/beta" +
                             (floats.error ? ": " + *floats.error : ""));
  }
  for (uint32_t type : {kQ8_0, kIQ4_XS}) {
    const std::string gates = model::ggmlTypeName(type) + " alpha/beta";
    const Plan named = planned(type, true);
    check(!named.error, "planner plans a rotation that names " + gates + (named.error ? ": " + *named.error : ""));
    check(names(planned(type, false), {"the rotation must name every quantized tensor"}),
          "planner refuses a rotation that leaves out " + gates);
  }
}

} // namespace

int main() {
  @autoreleasepool {
    const splash::test::TemporaryDirectory directory("splash-gguf-planner");
    guarded("planner on the dense target", [&] { checkDense(directory.path()); });
    guarded("planner on the MoE target", [&] { checkMoe(directory.path()); });
    guarded("planner on fused gate/up experts", [&] { checkFusedExperts(directory.path()); });
    guarded("planner on a safetensors MoE target", [&] { checkSafetensorsMoe(directory.path()); });
    guarded("planner on quantized alpha/beta", [&] { checkQuantizedAlphaBeta(directory.path()); });
    guarded("planner on the rotary metadata", [&] { checkRotary(directory.path()); });
    guarded("planner on a rotated target", [&] { checkRotation(directory.path()); });
    std::printf("%s (%d failures)\n", failures ? "GGUF planner tests FAILED" : "GGUF planner tests passed", failures);
    return failures ? 1 : 0;
  }
}
