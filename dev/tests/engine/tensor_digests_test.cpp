// The digest tables of source files: a tensor's digest is the SHA-256 of its
// 4 MiB chunks' SHA-256s, whatever table or read produced it; a table that is
// stale, damaged or another file's is refused and rebuilt, one that is valid
// is trusted without reading the file again, and a table directory that is
// not the user's own is refused.
#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include "model/TensorDigests.hpp"
#include "model/WeightSource.hpp"
#include "model/WeightStore.hpp"

#include <CommonCrypto/CommonDigest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <map>
#include <span>
#include <string>
#include <vector>

namespace {

using splash::test::rejects;
using splash::test::require;
namespace model = splash::model;

std::vector<uint8_t> pattern(size_t count, uint32_t seed) {
  std::vector<uint8_t> bytes(count);
  for (size_t i = 0; i < count; ++i) bytes[i] = static_cast<uint8_t>((i * 2654435761u + seed) >> 13);
  return bytes;
}

std::string hex(const uint8_t *digest) {
  std::string text;
  for (size_t i = 0; i < CC_SHA256_DIGEST_LENGTH; ++i) {
    text += "0123456789abcdef"[digest[i] >> 4];
    text += "0123456789abcdef"[digest[i] & 15];
  }
  return text;
}

// The digest the format promises, from CommonCrypto alone.
std::string expectedDigest(std::span<const uint8_t> bytes) {
  std::vector<uint8_t> chunks;
  for (size_t at = 0; at < bytes.size(); at += model::kDigestChunkBytes) {
    uint8_t digest[CC_SHA256_DIGEST_LENGTH];
    const size_t count = std::min<size_t>(model::kDigestChunkBytes, bytes.size() - at);
    CC_SHA256(bytes.data() + at, static_cast<CC_LONG>(count), digest);
    chunks.insert(chunks.end(), digest, digest + sizeof(digest));
  }
  uint8_t digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(chunks.data(), static_cast<CC_LONG>(chunks.size()), digest);
  return hex(digest);
}

// The installation's record of a file.
std::string recordOf(std::span<const uint8_t> bytes) { return model::weightDigest(bytes); }

// A directory of a model's sources and one for the tables, which the
// tables' own directory is made in.
struct Fixture final {
  splash::test::TemporaryDirectory root{"tensor-digests-sources"}, tables{"tensor-digests-tables"};
  std::filesystem::path tableDirectory() const { return tables.path() / "tables"; }
  std::filesystem::path source(const std::string &name, const std::vector<uint8_t> &bytes) const {
    splash::test::writeFile(root.path() / name, bytes);
    return root.path() / name;
  }
  model::TensorDigests digests(const std::map<std::string, std::string, std::less<>> &records) const {
    return model::TensorDigests(tableDirectory(), root.path(), records);
  }
};

// The digests of ranges of the file called name, recorded as record.
std::vector<std::string> digestsOf(const Fixture &fixture, const std::string &name, const std::string &record,
                                   const std::vector<std::pair<uint64_t, uint64_t>> &ranges, uint64_t dataOffset = 0) {
  model::WeightSource file(fixture.root.path() / name);
  file.setDataOffset(dataOffset);
  auto digests = fixture.digests({{name, record}});
  std::vector<model::SourceBytes> sources;
  for (const auto &[offset, bytes] : ranges) sources.push_back({&file, offset, bytes});
  const auto result = digests.digests(sources);
  require(result && result->size() == ranges.size(), "a recorded file has no digests");
  return *result;
}

std::filesystem::path tableOf(const Fixture &fixture, const std::string &record) {
  return fixture.tableDirectory() / (record + ".digests");
}

// A table ends in a line of the SHA-256 of the text before it; rewrites the
// table with `edit` applied to that text and the line made right again.
void rewriteTable(const std::filesystem::path &path, const std::function<void(std::string &)> &edit) {
  const std::vector<uint8_t> written = splash::test::readFile(path);
  std::string text(written.begin(), written.end());
  text.resize(text.rfind('\n', text.size() - 2) + 1);
  edit(text);
  uint8_t digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(text.data(), static_cast<CC_LONG>(text.size()), digest);
  splash::test::writeFile(path, text + "sha256 " + hex(digest) + "\n");
}

void testDigestsAreOfTheirBytes() {
  Fixture fixture;
  // A file of tensors of one chunk, several, an exact multiple and none that
  // fills the file, after 100 bytes of header.
  const uint64_t chunk = model::kDigestChunkBytes;
  const std::vector<uint8_t> bytes = pattern(100 + 3 * chunk + 12345, 1);
  fixture.source("m.safetensors", bytes);
  const std::vector<std::pair<uint64_t, uint64_t>> ranges{{0, 4096}, {4096, chunk}, {chunk + 4096, 2 * chunk + 7},
                                                         {3 * chunk + 4103, 1000}};
  const auto digests = digestsOf(fixture, "m.safetensors", recordOf(bytes), ranges, 100);
  for (size_t i = 0; i < ranges.size(); ++i)
    require(digests[i] == expectedDigest(std::span(bytes).subspan(100 + ranges[i].first, ranges[i].second)),
            "a tensor digest is not that of its bytes: range " + std::to_string(i));
  const auto again = digestsOf(fixture, "m.safetensors", recordOf(bytes), ranges, 100);
  require(again == digests, "a digest table gave other digests than the read");
}

void testValidTableIsTrustedWithoutReading() {
  Fixture fixture;
  const std::vector<uint8_t> bytes = pattern(70000, 2);
  fixture.source("n.safetensors", bytes);
  const std::string record = recordOf(bytes);
  const std::string real = digestsOf(fixture, "n.safetensors", record, {{0, 65536}}).front();
  const std::string stand(64, 'a');
  rewriteTable(tableOf(fixture, record), [&](std::string &text) {
    const size_t at = text.find(real);
    require(at != std::string::npos, "a digest table does not hold its digest");
    text.replace(at, real.size(), stand);
  });
  require(digestsOf(fixture, "n.safetensors", record, {{0, 65536}}).front() == stand,
          "a valid digest table was not used: the file was read again");
}

void testRewrittenFileIsDigestedAgain() {
  Fixture fixture;
  const std::vector<uint8_t> before = pattern(100000, 3), after = pattern(100000, 4);
  fixture.source("k.safetensors", before);
  const std::string record = recordOf(before);
  require(digestsOf(fixture, "k.safetensors", record, {{4096, 50000}}).front() ==
              expectedDigest(std::span(before).subspan(4096, 50000)),
          "a tensor digest is not that of its bytes");
  fixture.source("k.safetensors", after);
  require(digestsOf(fixture, "k.safetensors", record, {{4096, 50000}}).front() ==
              expectedDigest(std::span(after).subspan(4096, 50000)),
          "a digest table was used for a file written again");
}

void testDamagedTableIsDigestedAgain() {
  Fixture fixture;
  const std::vector<uint8_t> bytes = pattern(100000, 5);
  fixture.source("l.safetensors", bytes);
  const std::string record = recordOf(bytes);
  const std::string expected = expectedDigest(std::span(bytes).subspan(0, 65536));
  require(digestsOf(fixture, "l.safetensors", record, {{0, 65536}}).front() == expected,
          "a tensor digest is not that of its bytes");
  const std::filesystem::path table = tableOf(fixture, record);
  // A digest changed under the table's own check, then one that is no hex
  // under a check made right.
  for (const bool fixCheck : {false, true}) {
    for (const char damage : {expected[0] == '0' ? '1' : '0', 'g'}) {
      if (fixCheck != (damage == 'g')) continue;
      const auto damaged = [&](std::string &text) {
        const size_t at = text.find(expected);
        require(at != std::string::npos, "a digest table does not hold its digest");
        text[at] = damage;
      };
      if (fixCheck) {
        rewriteTable(table, damaged);
      } else {
        const std::vector<uint8_t> written = splash::test::readFile(table);
        std::string text(written.begin(), written.end());
        damaged(text);
        splash::test::writeFile(table, text);
      }
      require(digestsOf(fixture, "l.safetensors", record, {{0, 65536}}).front() == expected,
              std::string("a damaged digest table was used: ") + damage);
    }
  }
}

// A table is of the record digest that names it: the table of one record
// found under another's name is not used, even for the very file it was made
// from, and a table of another file is not used either.
void testAnotherRecordsTableIsRefused() {
  Fixture fixture;
  const std::vector<uint8_t> first = pattern(80000, 6), second = pattern(80000, 7);
  fixture.source("a.safetensors", first);
  fixture.source("b.safetensors", second);
  const std::string firstRecord = recordOf(first), secondRecord = recordOf(second), otherRecord(64, 'b');
  const std::string real = digestsOf(fixture, "a.safetensors", firstRecord, {{0, 65536}}).front();
  // The table of A, of a digest that is not A's, under the name of another
  // record that A is recorded as.
  rewriteTable(tableOf(fixture, firstRecord), [&](std::string &text) { text.replace(text.find(real), real.size(), 64, 'a'); });
  std::filesystem::copy_file(tableOf(fixture, firstRecord), tableOf(fixture, otherRecord));
  require(digestsOf(fixture, "a.safetensors", otherRecord, {{0, 65536}}).front() == real,
          "a digest table of another record digest was used");
  // The same table under the name of B's record, for B.
  std::filesystem::copy_file(tableOf(fixture, firstRecord), tableOf(fixture, secondRecord));
  require(digestsOf(fixture, "b.safetensors", secondRecord, {{0, 65536}}).front() ==
              expectedDigest(std::span(second).subspan(0, 65536)),
          "another file's digest table was used");
}

void testFilesWithoutARecordHaveNoDigests() {
  Fixture fixture;
  const std::vector<uint8_t> bytes = pattern(5000, 8);
  fixture.source("r.safetensors", bytes);
  model::WeightSource file(fixture.root.path() / "r.safetensors");
  const model::SourceBytes source{&file, 0, 1000};
  for (const std::string &record : {std::string(), std::string("xyz"), std::string(40, 'G'), std::string(41, 'a')}) {
    auto digests = fixture.digests(record.empty() ? std::map<std::string, std::string, std::less<>>{}
                                                  : std::map<std::string, std::string, std::less<>>{
                                                        {"r.safetensors", record}});
    require(!digests.digests({&source, 1}), "a file without a usable record was given digests: " + record);
  }
  const model::SourceBytes outside{&file, 0, 6000};
  auto digests = fixture.digests({{"r.safetensors", recordOf(bytes)}});
  rejects([&] { (void)digests.digests({&outside, 1}); }, "out of its file", "a range beyond its file was digested");
}

void testTableDirectoryMustBeTheUsers() {
  Fixture fixture;
  const std::filesystem::path directory = fixture.tableDirectory();
  const std::filesystem::path elsewhere = fixture.tables.path() / "elsewhere";
  std::filesystem::create_directories(elsewhere);
  chmod(elsewhere.c_str(), 0700);
  std::filesystem::create_directory_symlink(elsewhere, directory);
  rejects([&] { (void)fixture.digests({}); }, "refusing", "a symbolic link was taken for the table directory");
  std::filesystem::remove(directory);
  std::filesystem::create_directory(directory);
  for (const mode_t mode : {mode_t(0777), mode_t(0755), mode_t(0750)}) {
    chmod(directory.c_str(), mode);
    rejects([&] { (void)fixture.digests({}); }, "refusing", "a directory others may use was taken for the tables");
  }
  chmod(directory.c_str(), 0700);
  (void)fixture.digests({});
  // Made when it is missing.
  std::filesystem::remove(directory);
  (void)fixture.digests({});
  struct stat state{};
  require(stat(directory.c_str(), &state) == 0 && S_ISDIR(state.st_mode) && (state.st_mode & 0777) == 0700,
          "the table directory was not made 0700");
}

} // namespace

int main() {
  try {
    testDigestsAreOfTheirBytes();
    testValidTableIsTrustedWithoutReading();
    testRewrittenFileIsDigestedAgain();
    testDamagedTableIsDigestedAgain();
    testAnotherRecordsTableIsRefused();
    testFilesWithoutARecordHaveNoDigests();
    testTableDirectoryMustBeTheUsers();
    std::cout << "tensor digests: digests, tables, refusals and the table directory PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
