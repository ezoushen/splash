// Weight images shared between processes: WeightImages with a SharedImages
// source, in engine processes of this binary that load synthetic images and
// answer commands on their standard input.
//
//   shared-weights METALLIB
#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include "metal/MetalBackend.hpp"
#include "model/SharedImages.hpp"
#include "model/TensorDigests.hpp"
#include "model/WeightImages.hpp"
#include "model/WeightLayout.hpp"
#include "model/WeightSource.hpp"
#include "model/WeightStore.hpp"

#include <CommonCrypto/CommonDigest.h>
#include <fcntl.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

extern char **environ;

namespace {

using splash::test::require;
namespace model = splash::model;
namespace metal = splash::metal;

constexpr std::string_view kMagic = "TEST0001";

// ---- The engine process ----

// The images the process wrote itself.
unsigned gWrites = 0;
// Whether the process stops (SIGSTOP) as it begins to write its next image.
bool gStopInWrite = false;

// An image of a 16 KiB header block, then bytes [offset, offset + bytes) of
// file, padded to 16 KiB.
model::ImagePlan copyImage(std::shared_ptr<model::WeightSource> file, uint64_t offset, uint64_t bytes) {
  const uint64_t imageBytes = model::kWeightFileAlignment + model::alignWeightOffset(bytes);
  model::ImagePlan plan{"test/copy.bin", std::string(kMagic), 0, 0, imageBytes,
                        [file, offset, bytes](std::span<uint8_t> image, const metal::MetalBuffer &) {
                          ++gWrites;
                          if (std::exchange(gStopInWrite, false)) raise(SIGSTOP);
                          std::memset(image.data(), 0, image.size());
                          std::memcpy(image.data(), kMagic.data(), kMagic.size());
                          file->readData(offset, image.subspan(model::kWeightFileAlignment, bytes));
                          file->checkUnchanged();
                        }};
  plan.description = "copy " + std::to_string(bytes);
  plan.sources = {{file.get(), offset, bytes}};
  return plan;
}

// The SHA-256 of every image's bytes in load order, after the GPU read each
// payload back through its view as a command binds it.
std::string contentsDigest(metal::MetalBackend &backend, const model::WeightImages &images,
                           const std::vector<metal::MetalBuffer> &payloads) {
  std::string digests;
  for (const auto &image : images.contents()) digests += model::weightDigest(image.bytes) + ' ';
  for (const metal::MetalBuffer &whole : payloads) {
    const uint32_t count = static_cast<uint32_t>(whole.sizeBytes() / sizeof(uint32_t));
    const metal::MetalBuffer output =
        backend.allocateBuffer(whole.sizeBytes(), metal::BufferStorage::Shared, "readback");
    metal::ComputeDispatch dispatch;
    dispatch.pipelineName = "test_copy_u32";
    dispatch.buffers = {{0, whole}, {1, output}};
    dispatch.bytes = {{2, &count, sizeof(count)}};
    dispatch.threadgroups = {(count + 255) / 256, 1, 1};
    dispatch.threadsPerThreadgroup = {256, 1, 1};
    (void)backend.submit(dispatch);
    require(std::memcmp(output.contents(), whole.contents(), whole.sizeBytes()) == 0,
            "the GPU read other bytes than the image holds");
  }
  return model::weightDigest(digests);
}

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

// Arguments: METALLIB ROOT REGISTRY DIGESTS on|off. Every file in ROOT is a
// source, recorded by its SHA-256 as the installer records a file.
int runEngine(char **argv) {
  metal::MetalBackend backend(argv[2]);
  const std::filesystem::path root = argv[3];
  std::shared_ptr<model::SharedImages> shared;
  if (std::string_view(argv[6]) == "on") {
    model::SharedImagesConfig config;
    config.registry = argv[4];
    config.digestTables = argv[5];
    config.writerIdentity = "shared-weights-test";
    config.writeWait = std::chrono::seconds(1);
    config.modelRoot = root;
    for (const auto &entry : std::filesystem::directory_iterator(root))
      config.sourceDigests[entry.path().filename().string()] =
          model::weightDigest(splash::test::readFile(entry.path()));
    shared = std::make_shared<model::SharedImages>(std::move(config));
  }
  model::WeightImages images(backend, "sources", shared);
  std::vector<metal::MetalBuffer> payloads;
  std::map<std::string, std::shared_ptr<model::WeightSource>> sources;
  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream command(line);
    std::string verb;
    command >> verb;
    const unsigned before = gWrites;
    if (verb == "stop-in-write") {
      gStopInWrite = true;
      std::cout << "stopping" << std::endl;
      continue;
    } else if (verb == "load") {
      std::string name;
      uint64_t offset = 0, bytes = 0;
      command >> name >> offset >> bytes;
      auto &source = sources[name];
      if (!source) source = std::make_shared<model::WeightSource>(root / name);
      model::WeightFile file = images.load(copyImage(source, offset, bytes));
      payloads.push_back(file.section(bytes, "payload"));
    } else if (verb == "writable") {
      unsigned count = 0;
      for (const auto &image : images.contents()) count += writable(image.bytes.data());
      std::cout << "writable " << count << std::endl;
      continue;
    } else if (verb == "release") {
      images.release();
      std::cout << "released" << std::endl;
      continue;
    } else if (verb == "restore") {
      while (!images.restore()) {
      }
    } else if (verb != "contents") {
      std::cout << "unknown command " << verb << std::endl;
      continue;
    }
    std::cout << "writes " << gWrites - before << " contents " << contentsDigest(backend, images, payloads)
              << std::endl;
  }
  return 0;
}

// ---- The tests ----

// The engine processes running, which a test that waits for ever kills as
// it fails (watchdog), so that none is left stopped.
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
  }
  ~Engine() {
    if (stopped_) resume();
    if (commands_) fclose(commands_);
    if (pid_ > 0) {
      int status = 0;
      waitpid(pid_, &status, 0);
      ended();
    }
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
    char line[512];
    require(std::fgets(line, sizeof(line), replies_) != nullptr, "an engine process ended");
    std::string reply(line);
    if (!reply.empty() && reply.back() == '\n') reply.pop_back();
    return reply;
  }
  pid_t pid() const { return pid_; }
  // Stops the process, as SIGSTOP or a debugger does, until resume().
  void stop() {
    ::kill(pid_, SIGSTOP);
    stopped();
  }
  // Waits until the process stopped itself.
  void stopped() {
    int status = 0;
    require(waitpid(pid_, &status, WUNTRACED) == pid_ && WIFSTOPPED(status), "an engine process did not stop");
    stopped_ = true;
  }
  void resume() {
    ::kill(pid_, SIGCONT);
    stopped_ = false;
  }
  void kill() {
    ::kill(pid_, SIGKILL);
    int status = 0;
    waitpid(pid_, &status, 0);
    ended();
    pid_ = -1;
  }

private:
  // The process was waited for: the watchdog leaves its pid alone.
  void ended() {
    if (slot_ < std::size(gEngines)) gEngines[slot_] = 0;
  }

  pid_t pid_ = -1;
  size_t slot_ = std::size(gEngines);
  bool stopped_ = false;
  FILE *commands_ = nullptr;
  FILE *replies_ = nullptr;
};

struct Reply final {
  unsigned writes = 0;
  std::string contents;
};

Reply parse(const std::string &line) {
  std::istringstream reply(line);
  std::string writes, contents;
  Reply result;
  reply >> writes >> result.writes >> contents >> result.contents;
  require(writes == "writes" && contents == "contents" && !result.contents.empty(),
          "an engine process answered " + line);
  return result;
}

// A registry, digest tables and sources of this test alone. The registry is
// short, under /tmp, as socket paths are.
class Setup final {
public:
  explicit Setup(std::string self, std::string metallib)
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
  std::unique_ptr<Engine> engine(bool shared = true) const {
    return std::make_unique<Engine>(self_, std::vector<std::string>{metallib_, root_.path(), registry_,
                                                                    tables_.path(), shared ? "on" : "off"});
  }
  void source(const std::string &name, const std::vector<uint8_t> &bytes) const {
    splash::test::writeFile(root_.path() / name, bytes);
  }
  const std::filesystem::path &registry() const { return registry_; }
  const std::filesystem::path &tables() const { return tables_.path(); }
  const std::filesystem::path &root() const { return root_.path(); }

private:
  std::string self_, metallib_;
  splash::test::TemporaryDirectory tables_, root_;
  std::filesystem::path registry_;
};

// count bytes of a pattern seeded by seed.
std::vector<uint8_t> pattern(size_t count, uint32_t seed) {
  std::vector<uint8_t> bytes(count);
  for (size_t i = 0; i < count; ++i) bytes[i] = static_cast<uint8_t>((i * 2654435761u + seed) >> 13);
  return bytes;
}

// A second process loading an image the first holds receives it: it writes
// nothing and holds the same bytes.
void testSecondProcessReceives(const Setup &setup) {
  setup.source("a.safetensors", pattern(100000, 1));
  auto first = setup.engine(), second = setup.engine();
  const Reply written = parse(first->ask("load a.safetensors 4096 50000"));
  const Reply received = parse(second->ask("load a.safetensors 4096 50000"));
  require(written.writes == 1, "the first process did not write its image");
  require(received.writes == 0, "the second process wrote an image another process holds");
  require(received.contents == written.contents, "a received image holds other bytes");
}

// An image's key covers the content of the bytes it reads: other bytes of the
// same plan are another image.
void testOtherBytesAreAnotherImage(const Setup &setup) {
  setup.source("b.safetensors", pattern(100000, 2));
  setup.source("c.safetensors", pattern(100000, 3));
  auto first = setup.engine(), second = setup.engine();
  const Reply written = parse(first->ask("load b.safetensors 4096 50000"));
  const Reply other = parse(second->ask("load c.safetensors 4096 50000"));
  require(written.writes == 1 && other.writes == 1, "an image of other source bytes was received");
  require(other.contents != written.contents, "images of other source bytes hold the same bytes");
}

// The same bytes in another file, at another offset, are the same image: a
// fine-tune's unchanged experts and its base model's share one.
void testSameBytesElsewhereAreOneImage(const Setup &setup) {
  const std::vector<uint8_t> tensor = pattern(40000, 4);
  std::vector<uint8_t> base = pattern(60000, 5), tuned = pattern(90000, 6);
  std::copy(tensor.begin(), tensor.end(), base.begin() + 8192);
  std::copy(tensor.begin(), tensor.end(), tuned.begin() + 20000);
  setup.source("base.safetensors", base);
  setup.source("tuned.safetensors", tuned);
  auto first = setup.engine(), second = setup.engine();
  const Reply written = parse(first->ask("load base.safetensors 8192 40000"));
  const Reply received = parse(second->ask("load tuned.safetensors 20000 40000"));
  require(written.writes == 1 && received.writes == 0, "the same source bytes in another file were written again");
  require(received.contents == written.contents, "a received image holds other bytes");
}

// A process's release lets go of its view only: another process that holds
// the image keeps its bytes.
void testReleaseKeepsOtherHolders(const Setup &setup) {
  setup.source("f.safetensors", pattern(100000, 7));
  auto first = setup.engine(), second = setup.engine();
  const Reply written = parse(first->ask("load f.safetensors 0 65536"));
  require(parse(second->ask("load f.safetensors 0 65536")).writes == 0, "an image another process holds was written");
  require(first->ask("release") == "released", "a process did not release its images");
  require(parse(second->ask("contents")).contents == written.contents,
          "a release in one process changed another's image");
}

// The memory lives while any process holds it: SIGKILL of the process that
// wrote an image leaves another process's view of it whole.
void testKilledWriterKeepsReaders(const Setup &setup) {
  setup.source("g.safetensors", pattern(100000, 8));
  auto writer = setup.engine(), reader = setup.engine();
  const Reply written = parse(writer->ask("load g.safetensors 0 65536"));
  require(parse(reader->ask("load g.safetensors 0 65536")).writes == 0, "an image another process holds was written");
  writer->kill();
  require(parse(reader->ask("contents")).contents == written.contents,
          "an image changed when the process that wrote it was killed");
}

// Restore takes an image from a process that holds it and writes it only
// when none does: after its writer was killed, from the only other holder,
// or by writing it again once no process holds it.
void testRestoreReceivesOrRewrites(const Setup &setup) {
  setup.source("h.safetensors", pattern(100000, 9));
  auto writer = setup.engine(), reader = setup.engine(), other = setup.engine();
  const Reply written = parse(writer->ask("load h.safetensors 0 65536"));
  require(parse(reader->ask("load h.safetensors 0 65536")).writes == 0 &&
              parse(other->ask("load h.safetensors 0 65536")).writes == 0,
          "an image another process holds was written");
  writer->kill();
  require(reader->ask("release") == "released", "a process did not release its images");
  const Reply received = parse(reader->ask("restore"));
  require(received.writes == 0, "a restore wrote an image another process holds");
  require(received.contents == written.contents, "a restored image holds other bytes");
  require(reader->ask("release") == "released" && other->ask("release") == "released",
          "a process did not release its images");
  const Reply rewritten = parse(reader->ask("restore"));
  require(rewritten.writes == 1, "a restore did not write an image no process holds");
  require(rewritten.contents == written.contents, "a rewritten image holds other bytes");
}

// Two processes that load one image at the same time hold one copy: one
// writes it and the other receives it.
void testConcurrentLoadsWriteOnce(const Setup &setup) {
  setup.source("m.safetensors", pattern(400000, 16));
  auto first = setup.engine(), second = setup.engine();
  for (uint64_t offset = 0; offset < 320000; offset += 65536) {
    const std::string command = "load m.safetensors " + std::to_string(offset) + " 65536";
    first->send(command);
    second->send(command);
    const Reply one = parse(first->reply()), other = parse(second->reply());
    require(one.writes + other.writes == 1, "two processes that loaded one image at once wrote it " +
                                                std::to_string(one.writes + other.writes) + " times");
    require(one.contents == other.contents, "two processes that loaded one image at once hold other bytes");
  }
}

// The files of a process that ended are removed by the next process that
// looks for an image, and so are those of one that ended while it joined:
// a socket, or a holder file it had not published.
void testEndedProcessFilesAreRemoved(const Setup &setup) {
  setup.source("n.safetensors", pattern(100000, 17));
  auto ended = setup.engine();
  require(parse(ended->ask("load n.safetensors 0 65536")).writes == 1, "a process did not write its image");
  const std::string name = std::to_string(ended->pid()) + "-";
  ended->kill();
  splash::test::writeFile(setup.registry() / "1-ended.socket", std::string_view());
  splash::test::writeFile(setup.registry() / "2-ended.new", std::string_view());
  require(parse(setup.engine()->ask("load n.safetensors 0 65536")).writes == 1, "an ended process's image was received");
  for (const auto &entry : std::filesystem::directory_iterator(setup.registry())) {
    const std::string file = entry.path().filename().string();
    require(!file.starts_with(name) && !file.starts_with("1-ended") && !file.starts_with("2-ended"),
            "a file of an ended process was left: " + file);
  }
}

// Seconds since started.
double secondsSince(std::chrono::steady_clock::time_point started) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
}

// A stopped process that holds images delays another process's load, and
// its restore, once each: after one wait for an answer the other process
// passes over it and writes its images, rather than waiting for it per image
// and per lookup.
void testStoppedHolderIsPassedOver(const Setup &setup) {
  setup.source("o.safetensors", pattern(300000, 18));
  auto stopped = setup.engine(), other = setup.engine();
  Reply held;
  for (int image = 0; image < 4; ++image)
    held = parse(stopped->ask("load o.safetensors " + std::to_string(image * 65536) + " 65536"));
  stopped->stop();
  auto started = std::chrono::steady_clock::now();
  Reply loaded;
  for (int image = 0; image < 4; ++image) {
    loaded = parse(other->ask("load o.safetensors " + std::to_string(image * 65536) + " 65536"));
    require(loaded.writes == 1, "an image of a stopped process was received");
  }
  const double loading = secondsSince(started);
  require(loaded.contents == held.contents, "images written beside a stopped process hold other bytes");
  require(loading < 6, "a stopped process delayed four loads by " + std::to_string(loading) + " s");
  require(other->ask("release") == "released", "a process did not release its images");
  started = std::chrono::steady_clock::now();
  const Reply restored = parse(other->ask("restore"));
  const double restoring = secondsSince(started);
  require(restored.writes == 4 && restored.contents == held.contents, "a restore beside a stopped process failed");
  require(restoring < 6, "a stopped process delayed a restore by " + std::to_string(restoring) + " s");
  stopped->resume();
  require(parse(stopped->ask("contents")).contents == held.contents, "a resumed process lost its images");
}

// An image is read-only once written, in the process that wrote it as in
// one that received it.
void testImagesAreReadOnly(const Setup &setup) {
  setup.source("q.safetensors", pattern(100000, 20));
  auto first = setup.engine(), second = setup.engine();
  require(parse(first->ask("load q.safetensors 0 65536")).writes == 1, "a process did not write its image");
  require(parse(second->ask("load q.safetensors 0 65536")).writes == 0, "an image another process holds was written");
  require(first->ask("writable") == "writable 0", "the process that wrote an image can still write it");
  require(second->ask("writable") == "writable 0", "a process that received an image can write it");
}

// A process stopped while it writes an image, holding the image's lock,
// delays another process that needs the image by a bounded wait for the
// lock, after which that one writes a copy of its own.
void testStoppedWriterIsWaitedForBoundedly(const Setup &setup) {
  setup.source("p.safetensors", pattern(100000, 19));
  auto stopped = setup.engine(), other = setup.engine();
  require(stopped->ask("stop-in-write") == "stopping", "a process did not take a command");
  stopped->send("load p.safetensors 0 65536");
  stopped->stopped();
  const auto started = std::chrono::steady_clock::now();
  const Reply written = parse(other->ask("load p.safetensors 0 65536"));
  const double waited = secondsSince(started);
  require(written.writes == 1, "a process did not write an image whose writer stopped");
  require(waited < 6, "a stopped writer delayed a load by " + std::to_string(waited) + " s");
  stopped->resume();
  const Reply resumed = parse(stopped->reply());
  require(resumed.writes == 1 && resumed.contents == written.contents,
          "a resumed writer's image holds other bytes");
}

// The digest of bytes as an image's key covers them: the SHA-256 of the
// SHA-256s of its 4 MiB chunks.
std::string treeDigest(std::span<const uint8_t> bytes) {
  std::vector<uint8_t> chunks;
  for (size_t at = 0; at < bytes.size(); at += model::kDigestChunkBytes) {
    uint8_t digest[CC_SHA256_DIGEST_LENGTH];
    const size_t count = std::min<size_t>(model::kDigestChunkBytes, bytes.size() - at);
    CC_SHA256(bytes.data() + at, static_cast<CC_LONG>(count), digest);
    chunks.insert(chunks.end(), digest, digest + sizeof(digest));
  }
  return model::weightDigest(chunks);
}

// The digest TensorDigests gives bytes [offset, offset + bytes) of the
// source name, recorded with record, keeping its tables in setup's.
std::string tensorDigest(const Setup &setup, const std::string &name, const std::string &record, uint64_t offset,
                         uint64_t bytes) {
  model::WeightSource file(setup.root() / name);
  model::TensorDigests digests(setup.tables(), setup.root(), {{name, record}});
  const model::SourceBytes source{&file, offset, bytes};
  const auto result = digests.digests({&source, 1});
  require(result && result->size() == 1, "a recorded file has no digests");
  return result->front();
}

// A digest table is of one file as it was: a file written again, of the size
// and installation record it had, is digested again, so that a fine-tune
// written over a copy of its base model is not taken for the base.
void testRewrittenFileIsDigestedAgain(const Setup &setup) {
  const std::vector<uint8_t> before = pattern(100000, 13), after = pattern(100000, 14);
  setup.source("k.safetensors", before);
  const std::string record = model::weightDigest(before);
  require(tensorDigest(setup, "k.safetensors", record, 4096, 50000) ==
              treeDigest(std::span(before).subspan(4096, 50000)),
          "a tensor digest is not that of its bytes");
  setup.source("k.safetensors", after);
  require(tensorDigest(setup, "k.safetensors", record, 4096, 50000) ==
              treeDigest(std::span(after).subspan(4096, 50000)),
          "a digest table was used for a file written again");
}

// A digest table that was damaged is not used: a digest changed to another,
// or to text that is no digest, is computed again.
void testDamagedTableIsDigestedAgain(const Setup &setup) {
  const std::vector<uint8_t> bytes = pattern(100000, 15);
  setup.source("l.safetensors", bytes);
  const std::string record = model::weightDigest(bytes);
  const std::string expected = treeDigest(std::span(bytes).subspan(0, 65536));
  require(tensorDigest(setup, "l.safetensors", record, 0, 65536) == expected,
          "a tensor digest is not that of its bytes");
  const std::filesystem::path table = setup.tables() / (record + ".digests");
  for (const char damage : {expected[0] == '0' ? '1' : '0', 'g'}) {
    const std::vector<uint8_t> written = splash::test::readFile(table);
    std::string text(written.begin(), written.end());
    const size_t at = text.find(expected);
    require(at != std::string::npos, "a digest table does not hold its digest");
    text[at] = damage;
    splash::test::writeFile(table, text);
    require(tensorDigest(setup, "l.safetensors", record, 0, 65536) == expected,
            std::string("a damaged digest table was used: ") + damage);
  }
}

// An engine process ends when its commands end, whatever other engine
// processes the test started after it.
void testEnginesEndInAnyOrder(const Setup &setup) {
  setup.source("j.safetensors", pattern(100000, 12));
  auto first = setup.engine(), second = setup.engine();
  require(parse(first->ask("load j.safetensors 0 65536")).writes == 1, "a process did not write its image");
  first.reset();
  require(parse(second->ask("load j.safetensors 0 65536")).writes == 1, "an ended process's image was received");
}

// Without sharing a process creates no registry and no digest table, and its
// images hold the bytes of shared ones.
void testUnsharedCreatesNothing(const Setup &setup) {
  setup.source("i.safetensors", pattern(100000, 10));
  auto alone = setup.engine(false);
  const Reply unshared = parse(alone->ask("load i.safetensors 4096 65536"));
  require(unshared.writes == 1, "a process without sharing did not write its image");
  require(!std::filesystem::exists(setup.registry()) && std::filesystem::is_empty(setup.tables()),
          "a process without sharing created a registry or a digest table");
  const Reply shared = parse(setup.engine()->ask("load i.safetensors 4096 65536"));
  require(shared.writes == 1 && shared.contents == unshared.contents,
          "a shared image holds other bytes than an unshared one");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 7 && std::string_view(argv[1]) == "engine") return runEngine(argv);
    if (argc != 2) {
      std::cerr << "usage: shared-weights METALLIB\n";
      return 2;
    }
    const std::string self = std::filesystem::canonical(argv[0]).string();
    signal(SIGPIPE, SIG_IGN);
    // A test that waits for ever fails instead.
    signal(SIGALRM, watchdog);
    alarm(120);
    testSecondProcessReceives(Setup(self, argv[1]));
    testOtherBytesAreAnotherImage(Setup(self, argv[1]));
    testSameBytesElsewhereAreOneImage(Setup(self, argv[1]));
    testReleaseKeepsOtherHolders(Setup(self, argv[1]));
    testKilledWriterKeepsReaders(Setup(self, argv[1]));
    testRestoreReceivesOrRewrites(Setup(self, argv[1]));
    testEnginesEndInAnyOrder(Setup(self, argv[1]));
    testConcurrentLoadsWriteOnce(Setup(self, argv[1]));
    testEndedProcessFilesAreRemoved(Setup(self, argv[1]));
    testStoppedHolderIsPassedOver(Setup(self, argv[1]));
    testStoppedWriterIsWaitedForBoundedly(Setup(self, argv[1]));
    testImagesAreReadOnly(Setup(self, argv[1]));
    testRewrittenFileIsDigestedAgain(Setup(self, argv[1]));
    testDamagedTableIsDigestedAgain(Setup(self, argv[1]));
    testUnsharedCreatesNothing(Setup(self, argv[1]));
    std::cout << "PASS SharedWeights\n";
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
