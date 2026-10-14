#pragma once

// Weight images in memory that processes of one user share: each image lives
// in an unnamed POSIX shared-memory object, which exists exactly as long as
// some process maps it or holds its descriptor, so no process's end, even
// SIGKILL, leaves memory behind. Every process that holds images serves them
// to the others: it holds a lock on a holder file in the registry directory
// for its lifetime, and answers on a Unix socket beside it with an image's
// descriptor (SCM_RIGHTS) for its key. It marks both files as in use every
// hour, so that a cleaner of temporary files leaves them; the lock files of
// an image's writers are in a subdirectory, and each is created again
// whenever it is missing.

#include "metal/MetalBackend.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>

namespace splash::model {

class ImageRegistry final {
public:
  // Joins the registry at directory (requirePrivateDirectory). An acquire
  // waits up to writeWait for another process that writes the same image.
  ImageRegistry(const std::filesystem::path &directory, std::chrono::milliseconds writeWait);
  // Stops serving and leaves the registry. Images stay mapped while their
  // buffers hold them, and in other processes.
  ~ImageRegistry();
  ImageRegistry(const ImageRegistry &) = delete;
  ImageRegistry &operator=(const ImageRegistry &) = delete;

  // Gives attach the memory of the image under key, of bytes: mapped
  // read-only from a process that holds it, or, when none does, created and
  // mapped writable, which write then fills before any process is served it
  // and which is then read-only.
  // One process at a time writes a key; others wait for it, then receive.
  // Past writeWait they write a copy of their own, which they do not serve.
  // Returns whether this process wrote the image.
  bool acquire(const std::string &key, uint64_t bytes, const std::function<void(metal::MappedMemory)> &attach,
               const std::function<void()> &write);
  // Whether a process other than this one holds the image under key, of
  // bytes, now. It asks the holders as acquire does, and maps nothing that
  // outlives the call.
  [[nodiscard]] bool heldByAnother(const std::string &key, uint64_t bytes);
  // Begins a pass of acquires, a load or a restore of this process's images:
  // a holder that did not answer in time is passed over for the rest of the
  // pass it did not answer in, so that a stopped process delays each pass by
  // one wait, not one per image. One lookup among the holders waits at most
  // 5 s, each holder at most 2 s.
  void newPass();

  // For tests: hook runs on the thread of a lookup that has found a process
  // ended, before the lookup removes that process's files; a null hook
  // clears it. It applies to every registry of the process.
  static void setRemovalHook(void (*hook)());

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::model
