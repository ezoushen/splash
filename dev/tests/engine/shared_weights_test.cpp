// Weight images shared between processes: engine processes of this binary
// load a synthetic MLX MoE checkpoint through loadModel, with or without a
// SharedImages source, and report where each image came from and what it
// holds.
//
//   shared-weights PRODUCTION_METALLIB OTHER_METALLIB
#include "TestAdmission.hpp"
#include "TestChecks.hpp"
#include "TestCheckpoint.hpp"
#include "TestFiles.hpp"
#include "metal/MetalBackend.hpp"
#include "model/ModelFactory.hpp"
#include "model/GgufPreparation.hpp"
#include "model/SafetensorsImage.hpp"
#include "model/SharedImages.hpp"
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
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
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
  target.hiddenCaptureLayers.fill(target.layers - 1);
  return target;
}

model::DFlashDraftLayout syntheticDraft(const model::Qwen3_6MoeLayout &target) {
  model::DFlashDraftLayout draft;
  draft.layers = 2;
  draft.hiddenSize = 256;
  draft.vocabularySize = 256;
  draft.dynamicSize = 256;
  draft.qkvSize = 512;
  draft.attentionSize = 256;
  draft.intermediateSize = 256;
  draft.attentionHeadDimension = 64;
  draft.rotaryTheta = 10'000'000.0F;
  draft.targetHiddenSize = target.capturedHiddenSize();
  draft.selectorRank = 256;
  draft.kvHeads = 2;
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

model::ModelDescriptor syntheticDescriptor() {
  const model::Qwen3_6MoeLayout target = syntheticTarget();
  model::ModelDescriptor descriptor =
      model::makeModelDescriptor("Qwen MoE sharing oracle", target, syntheticDraft(target), syntheticVision(),
                                 model::TargetSource::Safetensors, model::VisionSource::None);
  descriptor.sourceIdentity = "sources";
  return descriptor;
}

// count bytes of a pattern seeded by seed.
std::vector<uint8_t> pattern(size_t count, uint32_t seed) {
  std::vector<uint8_t> bytes(count);
  for (size_t i = 0; i < count; ++i) bytes[i] = static_cast<uint8_t>((i * 2654435761u + seed) >> 13);
  return bytes;
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
  splash::test::writeSyntheticShard(root / "draft" / "model.safetensors",
                                    splash::test::draftTensors(syntheticDraft(syntheticTarget())));
}

constexpr uint32_t kLayers = 4;
// The images of the target: each layer's and its experts', the head's and the
// embedding's.
constexpr size_t kTargetImages = 2 * kLayers + 2;

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

// Arguments: METALLIB ROOT REGISTRY DIGESTS on|off|unrecorded. Every file in
// ROOT is a source, recorded by its SHA-256 as the installer records a file,
// but with unrecorded.
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
  do {
    std::istringstream command(line);
    std::string verb, name;
    command >> verb >> name;
    if (verb == "load" || verb == "release" || verb == "restore" || verb == "contents") {
      if (verb == "load") models.push_back(model::loadModel(backend, root / name, syntheticDescriptor(), shared));
      else if (verb == "release") models.back().images->release();
      else if (verb == "restore")
        while (!models.back().images->restore(splash::test::admitAll)) {
        }
      // The images' origin and the SHA-256 of their bytes, empty while
      // released, by component.
      std::string reply = "images";
      for (const auto &image : models.back().images->contents())
        reply += std::string(" ") + std::string(image.component) + "=" + originName(image.origin) + ":" +
                 (image.bytes.empty() ? "released" : model::weightDigest(image.bytes));
      std::cout << reply << std::endl;
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
    if (commands_) fclose(commands_);
    int status = 0;
    waitpid(pid_, &status, 0);
    if (slot_ < std::size(gEngines)) gEngines[slot_] = 0;
    if (replies_) fclose(replies_);
  }
  Engine(const Engine &) = delete;
  Engine &operator=(const Engine &) = delete;

  // The engine's reply to command.
  std::string ask(const std::string &command) {
    std::fprintf(commands_, "%s\n", command.c_str());
    std::fflush(commands_);
    return reply();
  }

private:
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

  pid_t pid_ = -1;
  size_t slot_ = std::size(gEngines);
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

bool isTarget(const std::string &component) { return component.starts_with("target/"); }

// The images of the target in loaded, which a test names by component.
Images targetImages(const Images &loaded) {
  Images target;
  for (const auto &[component, image] : loaded)
    if (isTarget(component)) target.insert({component, image});
  require(target.size() == kTargetImages, "an engine process loaded " + std::to_string(target.size()) +
                                              " target images, not " + std::to_string(kTargetImages));
  return target;
}

// A second process that loads the checkpoint a first holds receives every
// image of it: it writes none, and each holds the bytes an image written
// alone holds.
void testSecondProcessReceivesEveryImage(const Setup &setup) {
  setup.checkpoint("base", targetTensors());
  const Images alone = targetImages(parseImages(setup.engine(false)->ask("load base")));
  auto first = setup.engine(), second = setup.engine();
  const Images written = targetImages(parseImages(first->ask("load base")));
  const Images received = targetImages(parseImages(second->ask("load base")));
  for (const auto &[component, image] : alone) {
    require(written.at(component).origin == "written", "the first process did not write " + component);
    require(received.at(component).origin == "received",
            "the second process did not receive " + component + ": it " + received.at(component).origin + " it");
    require(written.at(component).digest == image.digest, component + " written beside others differs from alone");
    require(received.at(component).digest == image.digest, "a received " + component + " differs from alone");
  }
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
  const Images alone = targetImages(parseImages(setup.engine(false)->ask("load tuned")));
  auto first = setup.engine(), second = setup.engine();
  const Images held = targetImages(parseImages(first->ask("load base")));
  const Images fineTune = targetImages(parseImages(second->ask("load tuned")));
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
// process that holds the image keeps its bytes, and the restore writes the
// image again into memory of the restoring process alone.
void testReleaseAndRestoreKeepOtherHolders(const Setup &setup) {
  setup.checkpoint("base", targetTensors());
  auto first = setup.engine(), second = setup.engine();
  const Images written = targetImages(parseImages(first->ask("load base")));
  const Images received = targetImages(parseImages(second->ask("load base")));
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
    require(restored.at(component).digest == image.digest && restored.at(component).origin == "private",
            "a restored " + component + " is not a copy of this process's own with the same bytes");
}

// Without sharing a process creates no registry and no digest table, and its
// images are its own memory; with it, the same images are shared memory,
// which is read-only once written.
void testUnsharedCreatesNothing(const Setup &setup) {
  setup.checkpoint("base", targetTensors());
  auto alone = setup.engine(false);
  const Images unshared = targetImages(parseImages(alone->ask("load base")));
  for (const auto &[component, image] : unshared)
    require(image.origin == "private", "a process without sharing did not hold " + component + " privately");
  require(alone->ask("writable") != "writable 0", "a process without sharing holds its images in memory it cannot write");
  require(!std::filesystem::exists(setup.registry()) && std::filesystem::is_empty(setup.tables()),
          "a process without sharing created a registry or a digest table");
  auto sharing = setup.engine();
  const Images shared = targetImages(parseImages(sharing->ask("load base")));
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
  const Images written = targetImages(parseImages(first->ask("load base")));
  const Images loaded = targetImages(parseImages(unrecorded->ask("load base")));
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
  const Images written = targetImages(parseImages(copied->ask("load base")));
  const Images received = targetImages(parseImages(original->ask("load base")));
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

// The images of the target checkpoint in a directory, planned as an engine
// plans them.
struct Planned final {
  explicit Planned(const std::filesystem::path &target)
      : checkpoint(target), images(model::safetensors::planImages(checkpoint, syntheticTarget())) {}
  model::SafetensorsCheckpoint checkpoint;
  std::vector<model::gguf::Image> images;
};

// A SharedImages that keys images of the checkpoints under setup's sources,
// each file recorded by its SHA-256 as the installer records a file.
std::unique_ptr<model::SharedImages> keyer(const Setup &setup, const std::string &identity = "test-writer") {
  model::SharedImagesConfig config;
  config.registry = setup.registry();
  config.digestTables = setup.tables();
  config.writerIdentity = identity;
  config.modelRoot = setup.root();
  for (const auto &entry : std::filesystem::recursive_directory_iterator(setup.root()))
    if (entry.is_regular_file())
      config.sourceDigests[entry.path().lexically_relative(setup.root()).generic_string()] =
          model::weightDigest(splash::test::readFile(entry.path()));
  return std::make_unique<model::SharedImages>(std::move(config));
}

// The plan a loader makes of an image.
model::ImagePlan planOf(const model::gguf::Image &image) {
  model::gguf::Description description = model::gguf::describe(image);
  return {"target/" + image.name, image.magic, image.layer, image.type, image.bytes, {}, std::move(description.text),
          std::move(description.sources)};
}

// The keys of the target images of the checkpoint called name under setup,
// by component.
std::map<std::string, std::string> keysOf(const Setup &setup, const std::string &name,
                                          const std::string &identity = "test-writer") {
  const Planned planned(setup.root() / name / "target");
  const auto shared = keyer(setup, identity);
  std::map<std::string, std::string> keys;
  for (const model::gguf::Image &image : planned.images) {
    const std::string key = shared->key(planOf(image));
    require(key.size() == 64, "the key of target/" + image.name + " is not a SHA-256");
    keys["target/" + image.name] = key;
  }
  require(keys.size() == kTargetImages, "a checkpoint's images do not have distinct names");
  return keys;
}

// The SHA-256 of each image of the checkpoint under `target`, as the writers
// write it, by component.
std::map<std::string, std::string> writtenImages(metal::MetalBackend &backend, const std::filesystem::path &target) {
  const Planned planned(target);
  std::map<std::string, std::string> digests;
  for (const model::gguf::Image &image : planned.images) {
    const metal::MetalBuffer buffer = backend.allocateBuffer(image.bytes, metal::BufferStorage::Shared, image.name);
    model::writeGgufImage(backend, buffer, image);
    digests["target/" + image.name] = model::weightDigest(model::contentsOf(buffer));
  }
  return digests;
}

// Images with one key are one image: a change to any byte of a source tensor
// that changes an image the writers write changes its key. Each tensor of the
// checkpoint has two bytes in its middle changed in turn, and the images
// written from the changed checkpoint are compared with the key's.
void testKeyCoversSourceBytes(const Setup &setup, const std::string &production) {
  std::vector<SyntheticTensor> tensors = targetTensors();
  setup.checkpoint("base", tensors);
  metal::MetalBackend backend(production);
  const auto base = keysOf(setup, "base");
  const auto baseImages = writtenImages(backend, setup.root() / "base" / "target");
  require(keysOf(setup, "base") == base, "the keys of one checkpoint changed between two computations");
  for (const auto &[component, key] : base)
    for (const auto &[other, otherKey] : base)
      require(component == other || key != otherKey, component + " and " + other + " have one key");
  const Planned planned(setup.root() / "base" / "target");
  for (const SyntheticTensor &tensor : tensors) {
    const model::SourceTensor &source = planned.checkpoint.require(tensor.name);
    if (!source.bytes) continue;
    std::filesystem::remove_all(setup.root() / "flipped");
    std::filesystem::copy(setup.root() / "base", setup.root() / "flipped", std::filesystem::copy_options::recursive);
    const std::filesystem::path shard = setup.root() / "flipped" / "target" / "model.safetensors";
    std::vector<uint8_t> bytes = splash::test::readFile(shard);
    // Two bytes, so that a BF16 value changes by more than its low bits, to a
    // finite value.
    const uint64_t middle = source.file->dataOffset() + source.offset + source.bytes / 2;
    bytes.at(middle) ^= 0x3f;
    bytes.at(middle + 1) ^= 0x3f;
    splash::test::writeFile(shard, bytes);
    const auto keys = keysOf(setup, "flipped");
    const auto images = writtenImages(backend, setup.root() / "flipped" / "target");
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
void testKeyIgnoresWhereSourcesAre(const Setup &setup) {
  std::vector<SyntheticTensor> tensors = targetTensors();
  setup.checkpoint("base", tensors);
  std::reverse(tensors.begin(), tensors.end());
  setup.checkpoint("reordered", tensors);
  require(keysOf(setup, "reordered") == keysOf(setup, "base"), "the offsets of the sources changed the keys");
}

// An image's key covers every field of its plan: its header and size as
// ImagePlan states them, and each field of its sections the description
// states, one at a time.
void testKeyCoversThePlan(const Setup &setup) {
  setup.checkpoint("base", targetTensors());
  const Planned planned(setup.root() / "base" / "target");
  const auto shared = keyer(setup);
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
  require(keysOf(setup, "base", identity) != keysOf(setup, "base", changed(code.size() / 8)),
          "the writer identity does not reach the keys");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 7 && std::string_view(argv[1]) == "engine") return runEngine(argv);
    if (argc != 3) {
      std::cerr << "usage: shared-weights PRODUCTION_METALLIB OTHER_METALLIB\n";
      return 2;
    }
    const std::string self = std::filesystem::canonical(argv[0]).string();
    signal(SIGPIPE, SIG_IGN);
    // A test that waits for ever fails instead; each test has its own limit.
    signal(SIGALRM, watchdog);
    alarm(120);
    testSecondProcessReceivesEveryImage(Setup(self, argv[1]));
    alarm(120);
    testFineTuneSharesWhatMatches(Setup(self, argv[1]));
    alarm(120);
    testUnsharedCreatesNothing(Setup(self, argv[1]));
    alarm(120);
    testReleaseAndRestoreKeepOtherHolders(Setup(self, argv[1]));
    alarm(120);
    testUnrecordedSourcesAreNotShared(Setup(self, argv[1]));
    alarm(120);
    testWriterIsTheRunningCode(Setup(self, argv[1]));
    alarm(120);
    testKeyCoversSourceBytes(Setup(self, argv[1]), argv[1]);
    alarm(120);
    testKeyIgnoresWhereSourcesAre(Setup(self, argv[1]));
    alarm(120);
    testKeyCoversThePlan(Setup(self, argv[1]));
    alarm(120);
    testKeyCoversTheWriters(Setup(self, argv[1]), self, argv[1], argv[2]);
    alarm(120);
    testConfigTakesTheRecordedDigests(Setup(self, argv[1]));
    testUserPlace();
    std::cout << "PASS shared-weights\n";
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
