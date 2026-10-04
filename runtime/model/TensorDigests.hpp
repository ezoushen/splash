#pragma once

// The SHA-256 digests of the source bytes weight images read, which key the
// images processes share, kept so that a start reads no source bytes for
// them: one small table per source file, named by the file's content digest
// as the model's installation records it, maps the bytes an image reads, by
// offset and length in the file, to their digest. A table is of the file
// as it was open when the table was made: its device, inode, size and times
// of modification and change. One that is missing, unreadable, damaged (it
// ends in the SHA-256 of its text) or of another file, or of the file before
// it was written again, is written again, from one parallel uncached read
// of the bytes it lacks.

#include "model/WeightSource.hpp"

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace splash::model {

// A digest covers its bytes in chunks of this size, the last one shorter: it
// is the SHA-256 of the chunks' SHA-256s, so threads digest one tensor's
// chunks in parallel.
inline constexpr uint64_t kDigestChunkBytes = 4 << 20;

class TensorDigests final {
public:
  // Tables live in directory, created 0700. The files are those of the model
  // at root, each with its recorded digest by its path relative to root.
  TensorDigests(std::filesystem::path directory, std::filesystem::path root,
                std::map<std::string, std::string, std::less<>> recordedDigests);
  ~TensorDigests();
  TensorDigests(const TensorDigests &) = delete;
  TensorDigests &operator=(const TensorDigests &) = delete;

  // The lowercase hex digest of each source's bytes, in order; none when a
  // file they read has no recorded digest.
  [[nodiscard]] std::optional<std::vector<std::string>> digests(std::span<const SourceBytes> sources);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::model
