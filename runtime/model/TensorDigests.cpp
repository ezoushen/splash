#include "model/TensorDigests.hpp"
#include "model/WeightImages.hpp"
#include "model/WeightStore.hpp"

#include <CommonCrypto/CommonDigest.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace splash::model {
namespace {

constexpr std::string_view kTableHeader = "splash-tensor-digests 2";

using Digest = std::array<uint8_t, CC_SHA256_DIGEST_LENGTH>;

std::span<const uint8_t> asBytes(std::string_view text) {
  return {reinterpret_cast<const uint8_t *>(text.data()), text.size()};
}

Digest sha256(std::span<const uint8_t> bytes) {
  // Chunks and digest lists are far below CommonCrypto's 32-bit lengths.
  Digest digest;
  CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.data());
  return digest;
}

bool lowercaseHex(std::string_view text) {
  return std::all_of(text.begin(), text.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

// A table ends in a line of the SHA-256 of all before it, so a damaged
// table is not used.
constexpr std::string_view kTableTrailer = "sha256 ";

// Bytes of a file at an offset from its start.
using Extent = std::pair<uint64_t, uint64_t>;

// The open file a table was made from: its device, inode, size and times of
// change in nanoseconds. A file written again, even to its size, or another
// file with its name and record, is another file.
struct FileIdentity final {
  uint64_t device = 0, inode = 0, bytes = 0, modified = 0, changed = 0;
  bool operator==(const FileIdentity &) const = default;
};

uint64_t nanoseconds(const timespec &time) {
  return uint64_t(time.tv_sec) * 1'000'000'000 + uint64_t(time.tv_nsec);
}

FileIdentity identity(const WeightSource &file) {
  struct stat state{};
  if (fstat(file.descriptor(), &state) == -1)
    throw std::system_error(errno, std::generic_category(), "cannot inspect " + file.path().string());
  return {uint64_t(state.st_dev), uint64_t(state.st_ino), uint64_t(state.st_size),
          nanoseconds(state.st_mtimespec), nanoseconds(state.st_ctimespec)};
}

std::ostream &operator<<(std::ostream &output, const FileIdentity &file) {
  return output << file.device << ' ' << file.inode << ' ' << file.bytes << ' ' << file.modified << ' '
                << file.changed;
}

std::istream &operator>>(std::istream &input, FileIdentity &file) {
  return input >> file.device >> file.inode >> file.bytes >> file.modified >> file.changed;
}

// A file's table: the file, and the digest of each extent.
struct Table final {
  FileIdentity file;
  std::map<Extent, std::string> digests;
};

// The table at path for the file of record digest, or an empty one: a table
// that is missing, unreadable, damaged or of another file is not used.
Table readTable(const std::filesystem::path &path, std::string_view record, const FileIdentity &source) {
  Table table;
  table.file = source;
  const uint64_t fileBytes = source.bytes;
  std::ifstream file(path, std::ios::binary);
  const std::string contents{std::istreambuf_iterator<char>(file), {}};
  const size_t trailer = contents.rfind('\n', contents.size() >= 2 ? contents.size() - 2 : 0);
  if (trailer == std::string::npos || !contents.ends_with('\n')) return table;
  const std::string_view body = std::string_view(contents).substr(0, trailer + 1);
  if (contents.substr(trailer + 1) != std::string(kTableTrailer) + digestHex(sha256(asBytes(body))) + '\n')
    return table;
  std::istringstream input{std::string(body)};
  std::string header, digest;
  FileIdentity made;
  if (!std::getline(input, header) || header != kTableHeader || !(input >> digest >> made) || digest != record ||
      made != source)
    return table;
  uint64_t offset = 0, length = 0;
  while (input >> offset >> length >> digest) {
    if (digest.size() != 2 * CC_SHA256_DIGEST_LENGTH || !lowercaseHex(digest) || offset > fileBytes ||
        length > fileBytes - offset)
      return {source, {}};
    table.digests[{offset, length}] = digest;
  }
  if (!input.eof()) return {source, {}};
  return table;
}

// Replaces the table at path in one rename, so a reader sees the old table
// or the new one.
void writeTable(const std::filesystem::path &path, std::string_view record, const Table &table) {
  std::ostringstream text;
  text << kTableHeader << '\n' << record << ' ' << table.file << '\n';
  for (const auto &[extent, digest] : table.digests)
    text << extent.first << ' ' << extent.second << ' ' << digest << '\n';
  std::string contents = text.str();
  contents += std::string(kTableTrailer) + digestHex(sha256(asBytes(contents))) + '\n';
  std::string staged = (path.parent_path() / ".digests-XXXXXX").string();
  const int file = mkstemp(staged.data());
  if (file == -1) throw std::system_error(errno, std::generic_category(), "cannot write a tensor digest table");
  const ssize_t written = write(file, contents.data(), contents.size());
  const int error = errno;
  if (close(file) == -1 || written != static_cast<ssize_t>(contents.size()) ||
      rename(staged.c_str(), path.c_str()) == -1) {
    const int failure = written == -1 ? error : errno;
    unlink(staged.c_str());
    throw std::system_error(failure, std::generic_category(), "cannot write the tensor digest table " + path.string());
  }
}

void requireTableDirectory(const std::filesystem::path &directory) {
  std::error_code error;
  std::filesystem::create_directories(directory.parent_path(), error);
  if (error) throw std::system_error(error, "cannot create " + directory.parent_path().string());
  if (mkdir(directory.c_str(), 0700) == -1 && errno != EEXIST)
    throw std::system_error(errno, std::generic_category(), "cannot create " + directory.string());
}

// The digest of each extent of file, read uncached on a thread per core.
std::vector<std::string> digestExtents(const WeightSource &file, const std::vector<Extent> &extents) {
  struct Chunk final {
    size_t extent;
    uint64_t offset, bytes;
  };
  std::vector<Chunk> chunks;
  std::vector<std::vector<Digest>> digests(extents.size());
  for (size_t index = 0; index < extents.size(); ++index) {
    const auto [offset, bytes] = extents[index];
    digests[index].resize((bytes + kDigestChunkBytes - 1) / kDigestChunkBytes);
    for (uint64_t at = 0; at < bytes; at += kDigestChunkBytes)
      chunks.push_back({index, offset + at, std::min(kDigestChunkBytes, bytes - at)});
  }
  std::vector<std::vector<uint8_t>> staging(loadThreads());
  parallelFor(chunks.size(), [&](size_t index, unsigned thread) {
    const Chunk &chunk = chunks[index];
    std::vector<uint8_t> &buffer = staging[thread];
    buffer.resize(kDigestChunkBytes);
    const std::span<uint8_t> bytes(buffer.data(), chunk.bytes);
    readWeightBytes(file.descriptor(), chunk.offset, bytes);
    digests[chunk.extent][(chunk.offset - extents[chunk.extent].first) / kDigestChunkBytes] = sha256(bytes);
  });
  file.checkUnchanged();
  std::vector<std::string> result;
  for (const std::vector<Digest> &parts : digests) {
    const Digest digest = sha256({reinterpret_cast<const uint8_t *>(parts.data()), parts.size() * sizeof(Digest)});
    result.push_back(digestHex(digest));
  }
  return result;
}

} // namespace

struct TensorDigests::Impl {
  std::filesystem::path directory;
  std::filesystem::path root;
  std::map<std::string, std::string, std::less<>> recorded;
  // The tables read or written so far, by record digest.
  std::map<std::string, Table, std::less<>> tables;
  bool directoryReady = false;

  // The recorded digest of file, which names its table: a Git SHA-1 or a
  // SHA-256 in lowercase hex; empty for none.
  [[nodiscard]] std::string record(const WeightSource &file) const {
    const auto found = recorded.find(file.path().lexically_relative(root).generic_string());
    if (found == recorded.end()) return {};
    const std::string &digest = found->second;
    return lowercaseHex(digest) && (digest.size() == 40 || digest.size() == 64) ? digest : std::string();
  }

  // file's table, holding every extent of extents.
  Table &table(const WeightSource &file, const std::string &record, const std::vector<Extent> &extents) {
    const std::filesystem::path path = directory / (record + ".digests");
    const FileIdentity opened = identity(file);
    auto found = tables.find(record);
    if (found == tables.end() || found->second.file != opened)
      found = tables.insert_or_assign(record, readTable(path, record, opened)).first;
    Table &result = found->second;
    std::vector<Extent> missing;
    for (const Extent &extent : extents)
      if (!result.digests.contains(extent) && std::find(missing.begin(), missing.end(), extent) == missing.end())
        missing.push_back(extent);
    if (missing.empty()) return result;
    const std::vector<std::string> digests = digestExtents(file, missing);
    for (size_t index = 0; index < missing.size(); ++index) result.digests[missing[index]] = digests[index];
    if (!directoryReady) requireTableDirectory(directory);
    directoryReady = true;
    writeTable(path, record, result);
    return result;
  }
};

TensorDigests::TensorDigests(std::filesystem::path directory, std::filesystem::path root,
                             std::map<std::string, std::string, std::less<>> recordedDigests)
    : impl_(std::make_unique<Impl>()) {
  impl_->directory = std::move(directory);
  impl_->root = std::move(root);
  impl_->recorded = std::move(recordedDigests);
}

TensorDigests::~TensorDigests() = default;

std::optional<std::vector<std::string>> TensorDigests::digests(std::span<const SourceBytes> sources) {
  // The extents each file's sources read, from the start of the file.
  std::map<const WeightSource *, std::vector<Extent>> extents;
  for (const SourceBytes &source : sources) {
    if (!source.file) throw std::invalid_argument("a shared image's source names no file");
    if (source.offset > source.file->bytes() - source.file->dataOffset() ||
        source.bytes > source.file->bytes() - source.file->dataOffset() - source.offset)
      throw std::out_of_range("a shared image's source is out of its file: " + source.file->path().string());
    extents[source.file].emplace_back(source.file->dataOffset() + source.offset, source.bytes);
  }
  std::map<const WeightSource *, Table *> tables;
  for (const auto &[file, fileExtents] : extents) {
    const std::string record = impl_->record(*file);
    if (record.empty()) return std::nullopt;
    tables[file] = &impl_->table(*file, record, fileExtents);
  }
  std::vector<std::string> result;
  for (const SourceBytes &source : sources)
    result.push_back(
        tables.at(source.file)->digests.at({source.file->dataOffset() + source.offset, source.bytes}));
  return result;
}

} // namespace splash::model
