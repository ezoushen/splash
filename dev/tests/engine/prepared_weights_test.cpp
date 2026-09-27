#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include "model/PreparedWeights.hpp"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

using namespace splash::model;
using splash::test::rejects;
using splash::test::require;

namespace {

// Runs body in a child process. The child leaves through _exit with body's
// status, 1 when it throws, so it never returns into the parent's scenarios
// or removes their directory.
template <class F> pid_t spawn(F body) {
  const pid_t child = fork();
  require(child >= 0, "fork a child process");
  if (child == 0) {
    int status = 1;
    try {
      status = body();
    } catch (...) {
    }
    _exit(status);
  }
  return child;
}

// The exit status of child, -1 when a signal ended it.
int exitStatus(pid_t child) {
  int status = 0;
  require(waitpid(child, &status, 0) == child, "wait for a child process");
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

// Writes one byte other than the fixture's at offset of a prepared or source
// file, which preparation left read-only.
void corrupt(const std::filesystem::path &path, uint64_t offset) {
  require(chmod(path.c_str(), 0600) == 0, "chmod " + path.string());
  const int fd = open(path.c_str(), O_WRONLY);
  require(fd >= 0, "open " + path.string());
  const uint8_t bad = 9;
  writeWeightBytes(fd, offset, std::span(&bad, 1));
  close(fd);
}

std::string key(uint8_t value) { return weightDigest(std::span(&value, 1)); }

constexpr uint64_t kGiB = uint64_t{1} << 30;

// The store under SPLASH_WEIGHT_CACHE and the content every entry is
// written with; builds counts the writes.
struct Cache {
  std::filesystem::path root;
  PreparedWeights store;
  std::vector<uint8_t> bytes;
  int builds = 0;

  explicit Cache(const std::filesystem::path &directory) : root(directory), bytes(128 * 1024) {
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<uint8_t>(i * 37);
  }
  // An entry of its own component, so that publishing it supersedes nothing.
  [[nodiscard]] PreparedWeight entry(uint8_t value, uint64_t size) const {
    return {key(value), size, "test/" + std::to_string(value), key(value), "/test"};
  }
  [[nodiscard]] PreparedWeight entry(uint8_t value) const { return entry(value, bytes.size()); }
  [[nodiscard]] WeightWriter writer() {
    return [this](int fd, const PreparationCheck &) {
      ++builds;
      writeWeightBytes(fd, 0, bytes);
    };
  }
  std::filesystem::path prepare(const PreparedWeight &weight, const PreparationGuards &guards = {}) {
    return store.prepare(weight, writer(), guards);
  }
  std::filesystem::path prepare(uint8_t value, const PreparationGuards &guards = {}) {
    return prepare(entry(value), guards);
  }
  // The writes run by preparing value.
  int buildsOf(uint8_t value, const PreparationGuards &guards = {}) {
    const int before = builds;
    static_cast<void>(prepare(value, guards));
    return builds - before;
  }
};

const PreparationCheck noWorkspace = [] { throw std::runtime_error("no conversion workspace"); };

void coldPreparationWritesOnceAndWarmReuses(Cache &cache) {
  require(cache.buildsOf(1) == 1, "cold preparation did not write once");
  const auto path = cache.prepare(1);
  require(std::filesystem::file_size(path) == cache.bytes.size() &&
              WeightSource(path).digest() == weightDigest(cache.bytes),
          "prepared content differs");
  struct stat info{};
  require(stat(path.c_str(), &info) == 0 && !(info.st_mode & 0222), "prepared file is writable");
  require(cache.buildsOf(1) == 0, "warm preparation rebuilt weights");
}

void onlyAMissAdmitsWorkspace(Cache &cache) {
  static_cast<void>(cache.prepare(1));
  require(cache.buildsOf(1, {{}, noWorkspace}) == 0, "warm hit required workspace admission");
  rejects([&] { static_cast<void>(cache.prepare(10, {{}, noWorkspace})); }, "no conversion workspace",
          "cold preparation ignored workspace admission");
}

void warmHitsAreCancellable(Cache &cache) {
  static_cast<void>(cache.prepare(1));
  rejects([&] { static_cast<void>(cache.prepare(1, {[] { throw std::runtime_error("cancelled"); }})); },
          "cancelled", "warm hit ignored cancellation");
}

// A warm load must finish while an unrelated converter holds the lock. Pipes
// order the two processes; the holder's alarm turns a deadlock into a failure.
void warmLoadDoesNotWaitForTheConverterLock(Cache &cache) {
  static_cast<void>(cache.prepare(1));
  int ready[2], release[2];
  require(pipe(ready) == 0 && pipe(release) == 0, "lock fixture pipes");
  const pid_t holder = spawn([&] {
    alarm(5);
    const int lock = open((cache.root / "prepare.lock").c_str(), O_RDWR);
    if (lock < 0 || flock(lock, LOCK_EX)) return 1;
    char byte = 'x';
    if (write(ready[1], &byte, 1) != 1 || read(release[0], &byte, 1) != 1) return 2;
    return 0;
  });
  // With the holder the only writer of ready, the read ends if it exits
  // before taking the lock. The parent keeps its reading end of release, so
  // releasing a holder the alarm ended fails the check below, not the write.
  close(ready[1]);
  char signal = 'x';
  require(read(ready[0], &signal, 1) == 1, "the lock holder exited before taking the converter lock");
  const std::array<PreparedWeight, 1> warm{{cache.entry(1)}};
  cache.store.requireSpace(warm);
  static_cast<void>(cache.prepare(1, {{}, noWorkspace}));
  require(write(release[1], &signal, 1) == 1, "release the lock holder");
  for (int end : {ready[0], release[0], release[1]}) close(end);
  require(exitStatus(holder) == 0, "warm load waited for the converter lock");
}

void diskChecksKeepTheReserveAndCoverTheModel(Cache &cache) {
  // 2 GiB stay free, to the byte, and nothing to write needs no reserve.
  constexpr uint64_t kReserve = uint64_t{2} << 30;
  requireWeightDiskSpace(0, 0);
  requireWeightDiskSpace(kReserve + 128, 128);
  rejects([] { requireWeightDiskSpace(kReserve + 127, 128); }, "not enough disk space", "disk reserve boundary");
  // The store checks its volume: a file 1 GiB smaller than the free space
  // does not fit.
  const uint64_t available = std::filesystem::space(cache.root).available;
  const std::array<PreparedWeight, 1> reserve{{cache.entry(22, available > kGiB ? available - kGiB : 1)}};
  rejects([&] { cache.store.requireSpace(reserve); }, "not enough disk space", "disk reserve ignored");
  const std::array<PreparedWeight, 2> tooLarge{{cache.entry(20, UINT64_MAX / 2), cache.entry(21, UINT64_MAX / 2)}};
  rejects([&] { cache.store.requireSpace(tooLarge); }, "not enough disk space", "model-wide disk budget ignored");
  require(!std::filesystem::exists(cache.root / key(20)), "disk preflight wrote a partial model");
}

// Publishing a file evicts the entries it supersedes, so preparing a model
// again under a new identity needs room for its largest file, not for all of
// it; a model that supersedes nothing still needs room for every file.
void supersededEntriesAreCredited(Cache &cache) {
  const uint64_t available = std::filesystem::space(cache.root).available;
  const uint64_t size = available / 4;
  std::vector<PreparedWeight> model, earlier;
  for (uint8_t i = 0; i < 8; ++i) {
    const std::string component = "model/layer-" + std::to_string(i) + ".bin";
    model.push_back({key(90 + i), size, component, key(110 + i), "/model"});
    earlier.push_back({key(100 + i), cache.bytes.size(), component, key(110 + i), "/model"});
  }
  rejects([&] { cache.store.requireSpace(model); }, "not enough disk space", "a new model's budget was credited");
  // The earlier generation of each file, as large as its replacement, sparse.
  for (const auto &weight : earlier) {
    const auto path = cache.prepare(weight);
    std::filesystem::permissions(path, std::filesystem::perms::owner_write, std::filesystem::perm_options::add);
    std::filesystem::resize_file(path, size);
  }
  cache.store.requireSpace(model);
  // A file is written while the entry it replaces remains.
  const std::array<PreparedWeight, 1> large{{{key(98), available - kGiB, model[0].component, model[0].inputs, "/model"}}};
  rejects([&] { cache.store.requireSpace(large); }, "not enough disk space", "a replaced entry was counted free");
  for (const auto &weight : earlier) std::filesystem::remove_all(cache.root / weight.key);
}

// A complete entry of the first provenance version at key(value), whose
// sparse file is bytes long; its inputs are key(value) unless given.
void firstVersionEntry(Cache &cache, uint8_t value, const std::string &component, const std::string &source,
                       uint64_t bytes, const std::string &inputs = {}) {
  const auto directory = cache.root / key(value);
  std::filesystem::create_directory(directory);
  splash::test::writeFile(directory / "weights", "");
  std::filesystem::resize_file(directory / "weights", bytes);
  splash::test::writeFile(directory / "source", "splash-prepared-weight-v1\ncomponent " + component + "\ninputs " +
                                                    (inputs.empty() ? key(value) : inputs) + "\nsource " + source +
                                                    "\n");
}

// Every first-version entry of a source path goes when the first file of
// that path is published, so the disk check pools them and spreads them over
// the path's missing files, each up to its size: two missing files, whatever
// the entries' sizes and order, need room for one. A file larger than the
// free space still does not fit.
void firstVersionCreditIsPooledBySource(Cache &cache) {
  constexpr uint64_t kReserve = uint64_t{2} << 30;
  const uint64_t available = std::filesystem::space(cache.root).available;
  const uint64_t size = (available - kReserve) / 5 * 4;
  const std::string source = "/models/pooled";
  firstVersionEntry(cache, 230, "model/a.bin", source, size / 2);
  firstVersionEntry(cache, 231, "model/b.bin", source, size / 2 * 3);
  const std::array<PreparedWeight, 2> model{{{key(232), size, "model/a.bin", key(233), source, {key(233)}},
                                            {key(234), size, "model/b.bin", key(235), source, {key(235)}}}};
  cache.store.requireSpace(model);
  const std::array<PreparedWeight, 1> large{{{key(236), available, "model/c.bin", key(237), source, {key(237)}}}};
  rejects([&] { cache.store.requireSpace(large); }, "not enough disk space", "a file larger than the free space fit");
  for (uint8_t value : {230, 231}) std::filesystem::remove_all(cache.root / key(value));
}

// A re-assembly gives a model's root a new path while its blobs stay, so the
// first version may have prepared the same components under another path. A
// file keyed by its tensors removes the first-version entry of its component
// from any path, and the disk check credits it to that file: two files, each
// replacing an entry as large, need room for one. A GGUF target's
// first-version entry of that component is current and stays, and so does an
// entry of another component from another path.
void firstVersionComponentsOfAnotherPathAreSuperseded(Cache &cache) {
  constexpr uint64_t kReserve = uint64_t{2} << 30;
  const uint64_t size = (std::filesystem::space(cache.root).available - kReserve) / 3 * 2;
  const std::string earlier = "/models/.resolved/earlier/target", now = "/models/.resolved/now/target";
  firstVersionEntry(cache, 240, "target/layer-0.bin", earlier, size);
  firstVersionEntry(cache, 241, "target/layer-1.bin", earlier, size);
  const std::array<PreparedWeight, 2> model{{{key(242), size, "target/layer-0.bin", key(243), now, {key(243)}},
                                            {key(244), size, "target/layer-1.bin", key(245), now, {key(245)}}}};
  cache.store.requireSpace(model);
  firstVersionEntry(cache, 246, "target/layer-0.bin", "/models/gguf/target/model.gguf", cache.bytes.size(), key(247));
  firstVersionEntry(cache, 248, "target/head.bin", earlier, cache.bytes.size());
  static_cast<void>(cache.prepare({key(249), cache.bytes.size(), "target/layer-0.bin", key(250), now, {key(250)}}));
  const auto kept = [&](uint8_t value) { return std::filesystem::exists(cache.root / key(value)); };
  require(!kept(240) && kept(241) && kept(246) && kept(248) && kept(249),
          "a first-version entry of the component under another path was kept, or others removed");
  for (uint8_t value : {241, 246, 248, 249}) std::filesystem::remove_all(cache.root / key(value));
}

// A GGUF target's inputs are located in its file, as the first version's
// were, so its first-version entries are current entries. Publishing a
// missing image removes the first-version entry of its component and
// inputs; its siblings from the same file stay.
void firstVersionGgufSiblingsStay(Cache &cache) {
  const std::string gguf = "/models/gguf/target/model.gguf";
  firstVersionEntry(cache, 220, "target/layer-0.bin", gguf, cache.bytes.size(), key(223));
  firstVersionEntry(cache, 221, "target/layer-1.bin", gguf, cache.bytes.size(), key(223));
  static_cast<void>(cache.prepare({key(222), cache.bytes.size(), "target/layer-0.bin", key(223), gguf}));
  const auto kept = [&](uint8_t value) { return std::filesystem::exists(cache.root / key(value)); };
  require(!kept(220) && kept(221) && kept(222), "a GGUF target's current first-version entry was removed");
  for (uint8_t value : {221, 222}) std::filesystem::remove_all(cache.root / key(value));
}

// Entries of the first provenance version recorded the digests of whole
// shards as their inputs, which no entry keyed by its tensors matches.
// Publishing one removes every first-version entry prepared from its source
// path, whatever its component, and the disk check credits them; those of
// other paths stay.
void firstVersionEntriesAreSupersededBySource(Cache &cache) {
  const uint64_t available = std::filesystem::space(cache.root).available;
  const uint64_t size = available / 4;
  const auto firstVersion = [&](uint8_t value, const std::string &component, const std::string &source,
                                uint64_t bytes) { firstVersionEntry(cache, value, component, source, bytes); };
  std::vector<PreparedWeight> model;
  for (uint8_t i = 0; i < 8; ++i) {
    const std::string component = "model/layer-" + std::to_string(i) + ".bin";
    model.push_back({key(140 + i), size, component, key(150 + i), "/models/v1", {key(150 + i)}});
    firstVersion(160 + i, component, "/models/v1", size);
  }
  cache.store.requireSpace(model);
  firstVersion(170, "model/layer-0.bin", "/models/other", cache.bytes.size());
  firstVersion(171, "model/head.bin", "/models/v1", cache.bytes.size());
  static_cast<void>(
      cache.prepare({key(172), cache.bytes.size(), "model/layer-0.bin", key(173), "/models/v1", {key(173)}}));
  const auto kept = [&](uint8_t value) { return std::filesystem::exists(cache.root / key(value)); };
  for (uint8_t value = 160; value < 168; ++value)
    require(!kept(value), "a first-version entry of the source path was kept");
  require(!kept(171) && !kept(170) && kept(172), "first-version entries were kept or others removed");
  std::filesystem::remove_all(cache.root / key(172));
}

// A fine-tune and its base, both prepared by the first version; the
// fine-tune left layer 0 unchanged. The fine-tune loads first and publishes
// its files, layer 0 among them; the base reuses that layer and publishes its
// head, which removes every first-version entry of the base, its layer 0's
// too.
void firstVersionEntriesOfASharingModelAreRemoved(Cache &cache) {
  uint8_t value = 210;
  for (const std::string source : {"/models/base", "/models/tuned"})
    for (const std::string component : {"target/layer-0.bin", "target/head.bin"})
      firstVersionEntry(cache, value++, component, source, cache.bytes.size());
  const auto weight = [&](uint8_t value, const std::string &component, const std::string &source) {
    return PreparedWeight{key(value), cache.bytes.size(), component, key(value + 1), source, {key(value + 1)}};
  };
  const int builds = cache.builds;
  static_cast<void>(cache.prepare(weight(214, "target/layer-0.bin", "/models/tuned")));
  static_cast<void>(cache.prepare(weight(216, "target/head.bin", "/models/tuned")));
  static_cast<void>(cache.prepare(weight(214, "target/layer-0.bin", "/models/base")));
  static_cast<void>(cache.prepare(weight(218, "target/head.bin", "/models/base")));
  const auto kept = [&](uint8_t value) { return std::filesystem::exists(cache.root / key(value)); };
  require(cache.builds == builds + 3 && !kept(210) && !kept(211) && !kept(212) && !kept(213) && kept(214) &&
              kept(216) && kept(218),
          "a first-version entry of a model sharing another's files was kept");
  for (uint8_t value : {214, 216, 218}) std::filesystem::remove_all(cache.root / key(value));
}

// Same-size corruption must not pass a metadata-only check.
void corruptionIsRepaired(Cache &cache) {
  const auto path = cache.prepare(1);
  corrupt(path, 0);
  require(cache.buildsOf(1) == 1 && WeightSource(path).digest() == weightDigest(cache.bytes),
          "corruption not repaired");
}

// Interrupted and ENOSPC writes do not publish anything and can be retried.
void failedWritesPublishNothing(Cache &cache) {
  rejects([&] {
    static_cast<void>(cache.store.prepare(cache.entry(2), [&](int output, const PreparationCheck &) {
      writeWeightBytes(output, 0, std::span(cache.bytes).first(64));
      throw std::system_error(ENOSPC, std::generic_category(), "fixture disk full");
    }));
  }, "fixture disk full", "failed write accepted");
  require(!std::filesystem::exists(cache.root / key(2)), "partial file published");
  require(cache.buildsOf(2) == 1, "retry after a failed write did not write");
  rejects([&] { static_cast<void>(cache.prepare(3, {[] { throw std::runtime_error("memory pressure"); }})); },
          "memory pressure", "pressure ignored");
  require(!std::filesystem::exists(cache.root / key(3)), "pressure rejection published weights");
  rejects([&] { static_cast<void>(cache.prepare({"../outside", cache.bytes.size(), "test/outside", key(6), "/test"})); },
          "invalid prepared weight identity", "unsafe cache key accepted");
}

void crashedWriteIsReclaimed(Cache &cache) {
  constexpr int kCrashed = 7;
  const pid_t crash = spawn([&] {
    static_cast<void>(cache.store.prepare(cache.entry(4), [&](int output, const PreparationCheck &) {
      writeWeightBytes(output, 0, std::span(cache.bytes).first(64));
      _exit(kCrashed);
    }));
    return 0;
  });
  require(exitStatus(crash) == kCrashed, "the writer did not crash");
  require(!std::filesystem::exists(cache.root / key(4)), "crash published partial weights");
  const std::array<PreparedWeight, 1> retry{{cache.entry(4)}};
  cache.store.requireSpace(retry);
  require(!std::filesystem::exists(cache.root / (key(4) + ".partial")), "abandoned staging not cleaned");
  require(cache.buildsOf(4) == 1, "retry after a crash did not write");
}

// Two processes requesting the same identity must run its writer only once.
void concurrentMissesWriteOnce(Cache &cache) {
  const auto counter = cache.root / "builds";
  const WeightWriter competing = [&](int output, const PreparationCheck &) {
    const int count = open(counter.c_str(), O_CREAT | O_WRONLY | O_APPEND, 0600);
    require(count >= 0 && write(count, "x", 1) == 1, "write build counter");
    close(count);
    writeWeightBytes(output, 0, cache.bytes);
  };
  const pid_t child = spawn([&] {
    static_cast<void>(cache.store.prepare(cache.entry(5), competing));
    return 0;
  });
  static_cast<void>(cache.store.prepare(cache.entry(5), competing));
  require(exitStatus(child) == 0 && std::filesystem::file_size(counter) == 1,
          "concurrent cache miss rebuilt or corrupted weights");
}

void changedSourcesAreRejected(Cache &cache) {
  const auto path = cache.prepare(8);
  const WeightSource source(path);
  corrupt(path, 5);
  rejects([&] { source.checkUnchanged(); }, "source weights changed", "changed source was accepted");
  // An open descriptor surviving a rename must not hide path replacement.
  const WeightSource beforeReplacement(path);
  std::filesystem::rename(path, cache.root / "old-source");
  std::filesystem::copy_file(cache.root / "old-source", path);
  rejects([&] { beforeReplacement.checkUnchanged(); }, "source weights changed", "replaced source was accepted");
  // A source that cannot be opened is named, not taken for the cache.
  const auto missing = cache.root / "missing.gguf";
  rejects([&] { static_cast<void>(WeightSource(missing)); }, "open weight source " + missing.string(),
          "a missing source was reported without its path");
}

constexpr uint64_t kTensor = 4 << 20;

// A source of two tensors after a 16-byte header; its bytes.
std::vector<uint8_t> tensorSource(const std::filesystem::path &path) {
  std::vector<uint8_t> data(16 + 2 * kTensor);
  for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<uint8_t>(i * 13);
  splash::test::writeFile(path, data);
  return data;
}

// The digests of the tensors of the source at path, opened now; checks
// counts the checks run.
std::array<std::string, 2> tensorDigests(const std::filesystem::path &path, int &checks) {
  WeightSource source(path, [&] { ++checks; });
  source.setDataOffset(16);
  source.addTensor(kTensor, kTensor);
  source.addTensor(0, kTensor);
  return {source.tensorDigest(0, kTensor), source.tensorDigest(kTensor, kTensor)};
}

// A source's tensors are hashed once, in one pass that checks between its
// bounded reads: a warm start reads the digests remembered for the unchanged
// file, not its tensor data.
void warmSourcesReadNoTensorData(Cache &cache) {
  const auto path = cache.root / "tensors.safetensors";
  const auto data = tensorSource(path);
  int checks = 0;
  const auto cold = tensorDigests(path, checks);
  require(cold[0] == weightDigest(std::span(data).subspan(16, kTensor)) &&
              cold[1] == weightDigest(std::span(data).subspan(16 + kTensor, kTensor)),
          "tensor digests differ");
  checks = 0;
  require(tensorDigests(path, checks) == cold && checks <= 1, "a warm start read tensor data");
  // A digest is only of a tensor the parser added.
  WeightSource source(path);
  source.setDataOffset(16);
  source.addTensor(0, kTensor);
  rejects([&] { static_cast<void>(source.tensorDigest(0, kTensor - 1)); }, "source tensor was not added",
          "a range that was not added was hashed");
}

// The records the cache remembers.
std::set<std::filesystem::path> records(Cache &cache) {
  std::set<std::filesystem::path> result;
  for (const auto &entry : std::filesystem::directory_iterator(cache.root / "verified")) result.insert(entry.path());
  return result;
}

// A damaged table of tensor digests is recomputed, not trusted, and
// remembered again.
void damagedTablesAreRecomputed(Cache &cache) {
  const auto path = cache.root / "damaged.safetensors";
  static_cast<void>(tensorSource(path));
  const auto before = records(cache);
  int checks = 0;
  const auto cold = tensorDigests(path, checks);
  // The source's table, the same size, with its first digit changed.
  const auto after = records(cache);
  std::vector<std::filesystem::path> tables;
  std::set_difference(after.begin(), after.end(), before.begin(), before.end(), std::back_inserter(tables));
  require(tables.size() == 1, "hashing a source's tensors did not remember one table");
  const int record = open(tables.front().c_str(), O_RDWR);
  require(record >= 0, "open table fixture");
  uint8_t digit = 0;
  readWeightBytes(record, 0, std::span(&digit, 1));
  digit = digit == '0' ? '1' : '0';
  writeWeightBytes(record, 0, std::span(&digit, 1));
  close(record);
  checks = 0;
  require(tensorDigests(path, checks) == cold && checks > 1, "a damaged table was trusted");
  checks = 0;
  require(tensorDigests(path, checks) == cold && checks <= 1, "a recomputed table was not remembered");
}

// An entry keyed by the content of its source tensors lists their sorted
// digests beside its file, one per line and read-only, which release tooling
// joins on; an entry of located inputs lists none.
void entriesListTheirTensors(Cache &cache) {
  const uint64_t shape[] = {1};
  WeightIdentity content("fixture");
  content.input(key(201), 1, "U8", shape).input(key(200), 1, "U8", shape).input(key(201), 1, "U8", shape);
  const auto listed = cache.prepare(content.weight(cache.bytes.size(), "test/tensors", "/test")).parent_path() /
                      "tensors";
  std::string expected;
  for (const auto &digest : std::set{key(200), key(201)}) expected += digest + "\n";
  const auto bytes = splash::test::readFile(listed);
  struct stat info{};
  require(std::string(bytes.begin(), bytes.end()) == expected && stat(listed.c_str(), &info) == 0 &&
              !(info.st_mode & 0222),
          "an entry did not list its tensors");
  const auto path = cache.root / "located.gguf";
  static_cast<void>(tensorSource(path));
  WeightSource source(path);
  source.setDataOffset(16);
  WeightIdentity located("fixture");
  located.input(source, 0, kTensor, "U8", shape);
  require(!std::filesystem::exists(
              cache.prepare(located.weight(cache.bytes.size(), "test/located", "/test")).parent_path() / "tensors"),
          "an entry of located inputs listed tensors");
}

// A prepared file is mapped only as the cache verified it (WeightFile): not
// once replaced, even by the same bytes, until prepare verifies it again, and
// not once modified.
void onlyVerifiedFilesAreMapped(Cache &cache) {
  const auto path = cache.prepare(11);
  const auto map = [&] {
    const int file = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    require(file >= 0, "open " + path.string());
    try {
      requireVerifiedFile(file, path);
    } catch (...) {
      close(file);
      throw;
    }
    close(file);
  };
  map();
  std::filesystem::copy_file(path, cache.root / "copy");
  std::filesystem::rename(cache.root / "copy", path);
  rejects(map, "changed after verification", "a replaced prepared file was mapped");
  require(cache.buildsOf(11) == 0, "an identical replacement was prepared again");
  map();
  corrupt(path, 0);
  rejects(map, "changed after verification", "a modified prepared file was mapped");
}

// Damaged memoization proofs must be recomputed, not trusted.
void damagedProofsAreRecomputed(Cache &cache) {
  static_cast<void>(cache.prepare(9));
  int damaged = 0;
  for (const auto &entry : std::filesystem::directory_iterator(cache.root / "verified")) {
    const int proof = open(entry.path().c_str(), O_WRONLY | O_TRUNC);
    require(proof >= 0, "open proof fixture");
    const uint8_t bad = 9;
    writeWeightBytes(proof, 0, std::span(&bad, 1));
    close(proof);
    ++damaged;
  }
  require(damaged > 0, "preparation left no proof to damage");
  require(cache.buildsOf(9) == 0, "damaged proof forced an unnecessary rebuild");
}

// Publishing an entry removes the earlier preparations of its component from
// the same source data, whichever model's, or from its source path, whose
// planner may now read other tensors, and the entries earlier versions
// prepared from its source path; others stay, another model's preparation
// of the component from other data too.
void publishingSupersedesEarlierPreparations(Cache &cache) {
  static_cast<void>(cache.prepare(36));
  const PreparedWeight older{key(30), cache.bytes.size(), "target/layer-0.bin", key(40), "/models/z"};
  const PreparedWeight otherData{key(31), cache.bytes.size(), "target/layer-0.bin", key(41), "/models/b"};
  const PreparedWeight otherComponent{key(32), cache.bytes.size(), "target/head.bin", key(40), "/models/a"};
  const PreparedWeight otherTensors{key(37), cache.bytes.size(), "target/layer-0.bin", key(42), "/models/a"};
  for (const auto &weight : {older, otherData, otherComponent, otherTensors})
    static_cast<void>(cache.prepare(weight));
  std::filesystem::create_directory(cache.root / key(33));
  splash::test::writeFile(cache.root / key(33) / "source", "/models/a\nhead.bin\n");
  std::filesystem::create_directory(cache.root / key(34));
  splash::test::writeFile(cache.root / key(34) / "source", "/models/c\nhead.bin\n");
  static_cast<void>(cache.prepare({key(35), cache.bytes.size(), "target/layer-0.bin", key(40), "/models/a"}));
  const auto kept = [&](uint8_t value) { return std::filesystem::exists(cache.root / key(value)); };
  require(!kept(30) && !kept(33) && !kept(37) && kept(31) && kept(32) && kept(34) && kept(35) && kept(36),
          "superseded entries were kept or others removed");
}

// Two models whose sources hold the same tensors share an entry, whichever
// prepared it. After a preparation-identity change, the first of them to
// prepare it publishes the new key and removes the earlier entry; the other
// reuses the new one.
void sharedEntriesAreSupersededOnce(Cache &cache) {
  for (const bool baseFirst : {true, false}) {
    const uint8_t generation = baseFirst ? 180 : 190;
    const auto shared = [&](uint8_t identity, const std::string &source) {
      return PreparedWeight{key(generation + identity), cache.bytes.size(), "target/layer-0.bin", key(179), source};
    };
    static_cast<void>(cache.prepare(shared(0, "/models/base")));
    const int builds = cache.builds;
    static_cast<void>(cache.prepare(shared(1, baseFirst ? "/models/base" : "/models/tuned")));
    static_cast<void>(cache.prepare(shared(1, baseFirst ? "/models/tuned" : "/models/base")));
    require(cache.builds == builds + 1 && !std::filesystem::exists(cache.root / key(generation)),
            "a shared entry was prepared twice or its earlier preparation kept");
    std::filesystem::remove_all(cache.root / key(generation + 1));
  }
}

} // namespace

int main() {
  try {
    const splash::test::TemporaryDirectory directory("splash-prepared-weights");
    setenv("SPLASH_WEIGHT_CACHE", directory.path().c_str(), 1);
    Cache cache(directory.path());
    coldPreparationWritesOnceAndWarmReuses(cache);
    onlyAMissAdmitsWorkspace(cache);
    warmHitsAreCancellable(cache);
    warmLoadDoesNotWaitForTheConverterLock(cache);
    diskChecksKeepTheReserveAndCoverTheModel(cache);
    supersededEntriesAreCredited(cache);
    firstVersionEntriesAreSupersededBySource(cache);
    firstVersionEntriesOfASharingModelAreRemoved(cache);
    firstVersionGgufSiblingsStay(cache);
    firstVersionComponentsOfAnotherPathAreSuperseded(cache);
    firstVersionCreditIsPooledBySource(cache);
    corruptionIsRepaired(cache);
    failedWritesPublishNothing(cache);
    crashedWriteIsReclaimed(cache);
    concurrentMissesWriteOnce(cache);
    changedSourcesAreRejected(cache);
    warmSourcesReadNoTensorData(cache);
    damagedTablesAreRecomputed(cache);
    entriesListTheirTensors(cache);
    onlyVerifiedFilesAreMapped(cache);
    damagedProofsAreRecomputed(cache);
    publishingSupersedesEarlierPreparations(cache);
    sharedEntriesAreSupersededOnce(cache);
    std::cout << "prepared weights: content, reuse, corruption, interruption, pressure, concurrency, tensor "
                 "digests and superseded entries PASS\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
