#pragma once

// Weight images that processes share (serve --share-weights): an image is
// keyed by what determines its bytes, the code that writes it, its plan and
// the content of every source byte it reads, so any two processes whose
// images have one key hold one copy of it, whatever model or file the bytes
// came from (ImageRegistry). WeightImages loads, releases and restores its
// images through it.

#include "metal/MetalBackend.hpp"
#include "model/WeightImages.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <string>

namespace splash::model {

struct SharedImagesConfig final {
  // The registry of the processes that share images: a per-user directory
  // with a short path, as it holds their sockets.
  std::filesystem::path registry;
  // Where the digests of source tensors are kept, a small table per file.
  std::filesystem::path digestTables;
  // The writers' code (writerIdentity).
  std::string writerIdentity;
  // The model's directory, and the content digest its installation record
  // gives each source file, by its path relative to the directory
  // (recordedSourceDigests).
  std::filesystem::path modelRoot;
  std::map<std::string, std::string, std::less<>> sourceDigests;
  // How long a load or restore waits for another process that writes an
  // image it needs before it writes a copy of its own (ImageRegistry).
  std::chrono::milliseconds writeWait{10000};
};

class SharedImages final {
public:
  // Joins the registry.
  explicit SharedImages(SharedImagesConfig config);
  ~SharedImages();
  SharedImages(const SharedImages &) = delete;
  SharedImages &operator=(const SharedImages &) = delete;

  // plan's key, or empty when the plan cannot be shared: its loader does
  // not describe it, or a file it reads has no recorded digest.
  [[nodiscard]] std::string key(const ImagePlan &plan);
  // Gives attach the memory of the image under key, of bytes: another
  // process's, or new memory that write then fills. Returns whether this
  // process wrote it (ImageRegistry::acquire).
  bool acquire(const std::string &key, uint64_t bytes, const std::function<void(metal::MappedMemory)> &attach,
               const std::function<void()> &write);
  // Begins a pass of acquires (ImageRegistry::newPass).
  void newPass();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// The SHA-256 of the code that writes images: the engine's code as this
// process runs it (the __TEXT segment of the executable that holds it, as
// mapped) and the metallib backend loaded, which writes a GGUF image's on
// the GPU. Files that replace them on disk change nothing. The toolchain and
// the system's libraries and frameworks are not covered.
[[nodiscard]] std::string writerIdentity(const metal::MetalBackend &backend);
// What writerIdentity gives in a process of the engine executable at path
// that loaded backend's metallib.
[[nodiscard]] std::string writerIdentity(const std::filesystem::path &engine, const metal::MetalBackend &backend);

// The directory of this user's image registry, inside a directory that only
// this user may enter, so that no other user can create it first.
[[nodiscard]] std::filesystem::path userRegistryDirectory();

// Sharing for the model at root by writers of writerIdentity: the user's
// registry (userRegistryDirectory), and digest tables, under
// ~/Library/Caches/Splash/tensor-digests.
[[nodiscard]] SharedImagesConfig sharedImagesConfig(const std::filesystem::path &root, std::string writerIdentity);

} // namespace splash::model
