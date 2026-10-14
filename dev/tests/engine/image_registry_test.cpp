// The registry that processes share weight images through (ImageRegistry),
// with several registries in one process: each serves on a thread of its own
// and asks the others through its socket as a process would, so that the
// sanitizers watch the threads and the mappings. The processes that share
// images are tested in shared_weights_test.cpp.
#include "TestChecks.hpp"
#include "model/ImageRegistry.hpp"

#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using splash::metal::MappedMemory;
using splash::model::ImageRegistry;
using splash::test::require;

constexpr uint64_t kImageBytes = 3 * 16384 + 123;

// A registry directory of this test alone, which no other user can enter.
class Directory final {
public:
  Directory() {
    char pattern[] = "/tmp/splash-ir-XXXXXX";
    require(mkdtemp(pattern) != nullptr, "cannot create a registry directory");
    root_ = pattern;
  }
  ~Directory() {
    std::error_code ignored;
    std::filesystem::remove_all(root_, ignored);
  }
  Directory(const Directory &) = delete;
  Directory &operator=(const Directory &) = delete;
  [[nodiscard]] std::filesystem::path registry() const { return root_ / "registry"; }

private:
  std::filesystem::path root_;
};

std::string keyOf(char digit) { return std::string(64, digit); }

uint8_t patternAt(uint64_t index) { return static_cast<uint8_t>(index * 31 + 7); }

// What a registry gave a process for an image, which the test lets go of as
// the backend lets go of a buffer's memory.
struct Held final {
  MappedMemory memory;
  bool written = false;
  void letGo() {
    if (memory.unmap) std::exchange(memory.unmap, {})();
  }
};

// One acquire of key through registry, counting the writes in writes: the
// image's bytes are patternAt, unless write fails.
Held acquire(ImageRegistry &registry, const std::string &key, std::atomic<unsigned> &writes, bool fail = false) {
  Held held;
  held.written = registry.acquire(
      key, kImageBytes, [&](MappedMemory memory) { held.memory = std::move(memory); },
      [&] {
        ++writes;
        if (fail) throw std::runtime_error("the writer failed");
        auto *bytes = static_cast<uint8_t *>(held.memory.address);
        for (uint64_t i = 0; i < kImageBytes; ++i) bytes[i] = patternAt(i);
      });
  return held;
}

bool holdsPattern(const Held &held) {
  const auto *bytes = static_cast<const uint8_t *>(held.memory.address);
  for (uint64_t i = 0; i < kImageBytes; ++i)
    if (bytes[i] != patternAt(i)) return false;
  return true;
}

// Registries that ask for one image at the same time write it once and the
// others receive it, each of them holding the same bytes.
void testConcurrentAcquiresWriteOnce() {
  const Directory directory;
  constexpr size_t count = 6;
  std::vector<std::unique_ptr<ImageRegistry>> registries;
  for (size_t i = 0; i < count; ++i)
    registries.push_back(std::make_unique<ImageRegistry>(directory.registry(), std::chrono::seconds(10)));
  std::atomic<unsigned> writes{0};
  std::vector<Held> held(count);
  std::vector<std::thread> threads;
  for (size_t i = 0; i < count; ++i)
    threads.emplace_back([&, i] { held[i] = acquire(*registries[i], keyOf('1'), writes); });
  for (std::thread &thread : threads) thread.join();
  require(writes == 1, "registries that asked for one image at once wrote it " + std::to_string(writes.load()) + " times");
  size_t writers = 0;
  for (Held &image : held) {
    writers += image.written;
    require(holdsPattern(image), "a registry holds other bytes than the image's");
  }
  require(writers == 1, "not one registry wrote the image");
  for (Held &image : held) image.letGo();
}

// An image is held while some registry's process maps it: another registry
// then receives it, and once every mapping is let go, writes it again.
void testImageLivesWhileAnyoneMapsIt() {
  const Directory directory;
  ImageRegistry first(directory.registry(), std::chrono::seconds(10)), second(directory.registry(), std::chrono::seconds(10));
  std::atomic<unsigned> writes{0};
  Held written = acquire(first, keyOf('2'), writes);
  Held received = acquire(second, keyOf('2'), writes);
  require(writes == 1 && !received.written && holdsPattern(received), "a registry did not receive a held image");
  written.letGo();
  Held again = acquire(second, keyOf('2'), writes);
  require(writes == 1 && !again.written && holdsPattern(again), "an image another registry still maps was written again");
  received.letGo();
  again.letGo();
  Held rewritten = acquire(first, keyOf('2'), writes);
  require(writes == 2 && rewritten.written && holdsPattern(rewritten),
          "an image that no registry maps any more was received");
  rewritten.letGo();
}

// A writer that fails frees the lock of its image's key at once and serves
// nothing: another registry that asks for the key writes it, without waiting
// for the wait the writer's lock allows, and serves it to the next.
void testFailedWriterLeavesNoLock() {
  const Directory directory;
  ImageRegistry failing(directory.registry(), std::chrono::seconds(10)), other(directory.registry(), std::chrono::seconds(10)),
      third(directory.registry(), std::chrono::seconds(10));
  std::atomic<unsigned> writes{0};
  Held failed;
  splash::test::rejects(
      [&] {
        failed.written = failing.acquire(
            keyOf('3'), kImageBytes, [&](MappedMemory memory) { failed.memory = std::move(memory); },
            [&] {
              ++writes;
              throw std::runtime_error("the writer failed");
            });
      },
      "the writer failed", "a writer that failed did not fail its acquire");
  require(writes == 1, "the acquire did not reach its writer");
  failed.letGo();
  const auto started = std::chrono::steady_clock::now();
  Held written = acquire(other, keyOf('3'), writes);
  const double waited = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  require(writes == 2 && written.written && holdsPattern(written) && waited < 5,
          "a registry that asked for the key of a failed writer waited " + std::to_string(waited) + " s");
  Held received = acquire(third, keyOf('3'), writes);
  require(writes == 2 && !received.written && holdsPattern(received), "the image written after a failed writer was not served");
  written.letGo();
  received.letGo();
}

} // namespace

int main() {
  try {
    testConcurrentAcquiresWriteOnce();
    testImageLivesWhileAnyoneMapsIt();
    testFailedWriterLeavesNoLock();
    std::cout << "PASS image-registry\n";
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
