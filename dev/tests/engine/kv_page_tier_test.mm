#include "model/KvPageTier.hpp"
#include "tests/engine/AllocationFailure.hpp"

#include "engine/MemoryGovernor.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace splash;
using model::KvPageTier;
using model::KvTransfer;
using model::SlotFile;

namespace {

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

std::vector<std::byte> pattern(uint64_t bytes, uint32_t seed) {
  std::vector<std::byte> result(bytes);
  uint32_t state = seed;
  for (std::byte &value : result) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    value = static_cast<std::byte>(state);
  }
  return result;
}

// A page's bytes through the extent's host memory, in the order the tier
// stores them.
std::vector<std::byte> contents(const kv::PageStorage &pages, uint32_t page) {
  std::vector<std::byte> bytes;
  for (const auto span : pages.spans(page)) bytes.insert(bytes.end(), span.begin(), span.end());
  return bytes;
}

// Writes a pattern to a page the same way and returns it.
std::vector<std::byte> fill(const kv::PageStorage &pages, uint32_t page, uint32_t seed) {
  const auto bytes = pattern(pages.bytesPerPage(), seed);
  auto next = bytes.begin();
  for (const auto span : pages.spans(page)) {
    std::copy_n(next, span.size(), span.begin());
    next += static_cast<std::ptrdiff_t>(span.size());
  }
  return bytes;
}

std::shared_ptr<SlotFile::Slot> fileSlot(const std::shared_ptr<model::KvDiskSlot> &slot) {
  return static_cast<KvPageTier::DiskSlot &>(*slot).slot;
}

std::vector<std::byte> readSlot(SlotFile &file, const std::shared_ptr<model::KvDiskSlot> &slot) {
  std::vector<std::byte> bytes(file.slotBytes());
  require(file.read(fileSlot(slot), {bytes}, {})->wait(), "reading a disk slot failed");
  return bytes;
}

// A disk slot holds its page's bytes and zeros up to its aligned size.
bool holdsPage(const std::vector<std::byte> &slot, const std::vector<std::byte> &page) {
  return slot.size() >= page.size() && std::equal(page.begin(), page.end(), slot.begin()) &&
         std::all_of(slot.begin() + static_cast<std::ptrdiff_t>(page.size()), slot.end(),
                     [](std::byte value) { return value == std::byte{0}; });
}

// Drives the tier the way the engine does: poll until the worker's IO is
// retired.
void runUntilReady(KvPageTier &tier, KvTransfer &transfer) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  while (std::chrono::steady_clock::now() < deadline) {
    tier.poll();
    if (transfer.ready()) return;
    std::this_thread::yield();
  }
  throw std::runtime_error("KV transfer did not complete");
}

// Two pages in different extents go from their extents to disk and back.
// Between the two the host overwrites them, so what a restore leaves in a
// page is what the worker read; the pages beside them keep their own bytes.
void roundTrip(metal::MetalBackend &backend, engine::MemoryGovernor &governor,
               kv::Layout layout) {
  const uint32_t extent = layout.minimumExtentPages();
  kv::PageStorage pages(backend, governor.allocationAdmission(), layout, 2 * extent,
                        extent);
  const uint32_t pageA = extent - 1;
  const uint32_t pageB = extent;
  require(pages.isResident(pageA) && pages.ensureResident(pageB) && pages.isResident(pageB),
          "test pages were not mapped");
  const uint64_t slotBytes = KvPageTier::slotBytesFor(pages);
  const uint64_t payload = pages.bytesPerPage();
  require(slotBytes >= payload && slotBytes < payload + SlotFile::kAlignmentBytes &&
              slotBytes % SlotFile::kAlignmentBytes == 0,
          "KV slot size is not the page rounded up for uncached IO");
  auto file = std::make_shared<SlotFile>(slotBytes, 2 * slotBytes);
  KvPageTier tier(pages, file);
  require(tier.slotBytes() == slotBytes, "tier reports another slot size");

  const auto first = fill(pages, pageA, 0x1234567u);
  const auto second = fill(pages, pageB, 0x89abcdefu);
  const auto beforeA = fill(pages, pageA - 1, 0x600df00du);
  const auto afterB = fill(pages, pageB + 1, 0x0ddba11u);
  require(contents(pages, pageA) == first && contents(pages, pageB) == second,
          "a page does not hold what the host wrote through its spans");

  auto slotA = tier.acquireSlot();
  auto slotB = tier.acquireSlot();
  require(slotA && slotB && !tier.acquireSlot(), "disk quota was not enforced");
  auto demoteA = tier.demote(pageA, slotA, {});
  auto demoteB = tier.demote(pageB, slotB, {});
  require(demoteA && demoteB, "demotions within the tier's limit were refused");
  runUntilReady(tier, *demoteA);
  runUntilReady(tier, *demoteB);
  require(demoteA->finish() && demoteB->finish(), "demotions did not succeed");
  require(contents(pages, pageA) == first && contents(pages, pageB) == second,
          "a demotion changed its page");
  require(holdsPage(readSlot(*file, slotA), first) && holdsPage(readSlot(*file, slotB), second),
          "disk slots do not hold their pages byte for byte");

  static_cast<void>(fill(pages, pageA, 0xdeadbeefu));
  static_cast<void>(fill(pages, pageB, 0xfeedfaceu));
  auto restoreB = tier.restore(slotB, pageB, {});
  auto restoreA = tier.restore(slotA, pageA, {});
  require(restoreA && restoreB, "restores within the tier's limit were refused");
  runUntilReady(tier, *restoreA);
  runUntilReady(tier, *restoreB);
  require(restoreA->finish() && restoreB->finish(), "restores did not succeed");
  require(contents(pages, pageA) == first && contents(pages, pageB) == second,
          "pages did not come back from disk byte for byte");
  require(contents(pages, pageA - 1) == beforeA && contents(pages, pageB + 1) == afterB,
          "a transfer wrote outside its page");

  // A finished transfer holds neither room in the tier nor its slot.
  slotA.reset();
  require(tier.canDemote() && tier.acquireSlot() != nullptr,
          "a released disk slot was not reusable");
  std::cout << "round trip " << kv::formatName(layout.format) << " "
            << layout.attentionLayers << "x" << layout.kvHeads << "x"
            << layout.headDimension << ": page_bytes=" << payload
            << " slot_bytes=" << slotBytes << '\n';
}

// The transfers in flight are bounded, in all and by kind: demotions take at
// most half of the bound and restores at most three quarters, so a burst of
// restores leaves a demotion its turn and a burst of demotions leaves
// restores theirs. A transfer counts until poll() has retired it, and a read
// that fails gives its room back like any other.
void limits(metal::MetalBackend &backend, engine::MemoryGovernor &governor,
            kv::Format format) {
  const kv::Layout layout{2, 2, 256, format};
  kv::PageStorage pages(backend, governor.allocationAdmission(), layout,
                        layout.minimumExtentPages(), layout.minimumExtentPages());
  const uint64_t slotBytes = KvPageTier::slotBytesFor(pages);
  auto file = std::make_shared<SlotFile>(slotBytes, 8 * slotBytes);
  KvPageTier tier(pages, file, 4);
  const auto bytes = fill(pages, 9, 0x51a5e5u);
  std::vector<std::shared_ptr<model::KvDiskSlot>> written;
  for (int index = 0; index < 2; ++index) {
    written.push_back(tier.acquireSlot());
    require(file->write(fileSlot(written.back()), {bytes}, {})->wait(), "seeding failed");
  }
  // An unwritten slot cannot be restored.
  auto empty = tier.acquireSlot();
  std::vector<std::unique_ptr<KvTransfer>> restores;
  restores.push_back(tier.restore(written[0], 0, {}));
  restores.push_back(tier.restore(empty, 1, {}));
  restores.push_back(tier.restore(written[1], 2, {}));
  require(restores[0] && restores[1] && restores[2], "a restore within the share was refused");
  require(!tier.restore(written[0], 3, {}), "restores took more than three quarters of the bound");
  auto target = tier.acquireSlot();
  auto demotion = tier.demote(9, target, {});
  require(demotion != nullptr, "a burst of restores left no room for a demotion");
  require(!tier.canDemote() && !tier.demote(9, tier.acquireSlot(), {}),
          "the tier took more transfers than its bound");
  for (auto &restore : restores) runUntilReady(tier, *restore);
  runUntilReady(tier, *demotion);
  require(restores[0]->finish() && !restores[1]->finish() && restores[2]->finish(),
          "a restore beside a failed one did not succeed, or an unwritten slot was restored");
  require(demotion->finish() && contents(pages, 0) == bytes && contents(pages, 2) == bytes &&
              holdsPage(readSlot(*file, target), bytes),
          "transfers of both kinds side by side moved the wrong bytes");

  // With nothing in flight, demotions take half of the bound and no more.
  auto first = tier.demote(0, tier.acquireSlot(), {});
  auto second = tier.demote(2, tier.acquireSlot(), {});
  require(first && second && !tier.canDemote() && !tier.demote(9, tier.acquireSlot(), {}),
          "demotions took more than half of the bound");
  require(tier.restore(written[0], 4, {}) != nullptr,
          "a burst of demotions left no room for a restore");
  runUntilReady(tier, *first);
  runUntilReady(tier, *second);
  require(first->finish() && second->finish() && tier.canDemote() && tier.writable(),
          "demotions did not succeed or did not give their room back");
  std::cout << "transfer limit tests passed\n";
}

// A transfer that cannot be started leaves nothing behind: no room taken in
// the tier and no IO the tier does not track.
void allocationFailure(metal::MetalBackend &backend, engine::MemoryGovernor &governor,
                       kv::Format format) {
  const kv::Layout layout{2, 2, 256, format};
  kv::PageStorage pages(backend, governor.allocationAdmission(), layout,
                        layout.minimumExtentPages(), layout.minimumExtentPages());
  const uint64_t bytes = KvPageTier::slotBytesFor(pages);
  for (bool restoring : {false, true}) {
    bool completed = false;
    for (int failure = 0; failure < 64; ++failure) {
      auto file = std::make_shared<SlotFile>(bytes, bytes);
      KvPageTier tier(pages, file, 1);
      auto slot = tier.acquireSlot();
      const auto payload = fill(pages, 0, 123);
      require(file->write(fileSlot(slot), {payload}, {})->wait(), "fault slot seed failed");
      std::unique_ptr<KvTransfer> transfer;
      allocationFailureAfter = failure;
      try {
        transfer = restoring ? tier.restore(slot, 0, {}) : tier.demote(0, slot, {});
        allocationFailureAfter = -1;
      } catch (const std::bad_alloc &) {
        allocationFailureAfter = -1;
        require(tier.canDemote() && file->idle(),
                "failed transfer submission took room or left untracked IO");
        continue;
      }
      require(transfer != nullptr, "transfer refused after allocation failures");
      runUntilReady(tier, *transfer);
      require(transfer->finish() && tier.canDemote() && contents(pages, 0) == payload,
              "transfer did not recover");
      completed = true;
      break;
    }
    require(completed, "transfer allocation sweep never reached success");
  }
}

// Kernels and the worker share an extent. Kernels write a page through its
// address, and the worker writes those bytes to disk. Then one command keeps
// writing and checking the pages around it while the worker reads the page
// back and writes another one out; the read lands while the command runs.
// The command sees its own data, the worker's page holds the disk's bytes,
// and the next command reads them through the page's address.
void besideACommand(metal::MetalBackend &backend, engine::MemoryGovernor &governor,
                    kv::Format format) {
  const kv::Layout layout{10, 2, 256, format}; // Qwen3.6-35B
  const uint32_t extent = layout.minimumExtentPages();
  kv::PageStorage pages(backend, governor.allocationAdmission(), layout, extent, extent);
  const uint64_t slotBytes = KvPageTier::slotBytesFor(pages);
  auto file = std::make_shared<SlotFile>(slotBytes, 2 * slotBytes);
  KvPageTier tier(pages, file);
  constexpr uint32_t restored = 16, demoted = 25, neighbours = 8;
  const uint64_t data = layout.dataBytesPerLayerPage();
  const uint32_t words = static_cast<uint32_t>(data / sizeof(uint32_t));

  // The GPU addresses of a page's key and value slabs in every layer.
  std::byte *const base = pages.spans(0).front().data();
  const uint64_t address = pages.entry(0);
  const auto slabs = [&](uint32_t page, std::vector<uint64_t> &table) {
    for (const auto span : pages.spans(page)) {
      if (span.size() == data) table.push_back(address + static_cast<uint64_t>(span.data() - base));
    }
  };
  std::vector<uint64_t> own, around;
  slabs(restored, own);
  for (uint32_t page = restored - neighbours; page <= restored + neighbours; ++page) {
    if (page != restored) slabs(page, around);
  }
  const auto tableOf = [&](const std::vector<uint64_t> &entries) {
    metal::MetalBuffer table = backend.allocateBuffer(entries.size() * sizeof(uint64_t));
    std::memcpy(table.contents(), entries.data(), entries.size() * sizeof(uint64_t));
    return table;
  };
  const metal::MetalBuffer ownTable = tableOf(own), aroundTable = tableOf(around);
  metal::MetalBuffer mismatches = backend.allocateBuffer(sizeof(uint32_t));
  auto *mismatched = static_cast<uint32_t *>(mismatches.contents());
  // Writes seed ^ (131071 * entry + word) to every word of every slab of a
  // table, or counts the words that hold something else.
  const auto write = [&](const metal::MetalBuffer &table, size_t entries, const uint32_t &seed) {
    return metal::ComputeDispatch{"addressed_write_u32", {{0, table}},
        {{1, &words, sizeof(words)}, {2, &seed, sizeof(seed)}},
        {words / 256, entries, 1}, {256, 1, 1}};
  };
  const auto check = [&](const metal::MetalBuffer &table, size_t entries, const uint32_t &seed) {
    return metal::ComputeDispatch{"addressed_check_u32", {{0, table}, {3, mismatches}},
        {{1, &words, sizeof(words)}, {2, &seed, sizeof(seed)}},
        {words / 256, entries, 1}, {256, 1, 1}};
  };

  // Kernels write the page's slabs, the host its scales; the worker writes
  // what both left in the extent.
  const uint32_t ownSeed = 0x9e3779b9u;
  static_cast<void>(fill(pages, restored, 0x5ca1ab1eu));
  static_cast<void>(backend.submit(write(ownTable, own.size(), ownSeed)));
  const auto expected = contents(pages, restored);
  const auto leaving = fill(pages, demoted, 0xc0ffee11u);
  auto slot = tier.acquireSlot();
  auto demotion = tier.demote(restored, slot, {});
  require(demotion != nullptr, "demotion was refused");
  runUntilReady(tier, *demotion);
  require(demotion->finish() && holdsPage(readSlot(*file, slot), expected),
          "the disk does not hold what kernels wrote to the page");
  static_cast<void>(fill(pages, restored, 0xdeadbeefu));

  // The command grows until it outlasts the read.
  uint32_t rounds = 16;
  for (bool overlapped = false; !overlapped;) {
    rounds *= 4;
    require(rounds <= 4096, "no restore landed while a command was running");
    std::vector<uint32_t> seeds(rounds);
    std::vector<metal::ComputeDispatch> command;
    for (uint32_t round = 0; round < rounds; ++round) {
      seeds[round] = 0x85ebca6bu * (round + 1);
      command.push_back(write(aroundTable, around.size(), seeds[round]));
      command.push_back(check(aroundTable, around.size(), seeds[round]));
    }
    *mismatched = 0;
    static_cast<void>(fill(pages, restored, 0xdeadbeefu + rounds));
    auto other = tier.acquireSlot();
    auto ticket = backend.submitCommandAsync(command);
    auto restore = tier.restore(slot, restored, {});
    auto leave = tier.demote(demoted, other, {});
    require(restore && leave, "transfers beside a command were refused");
    runUntilReady(tier, *restore);
    overlapped = !ticket.ready();
    runUntilReady(tier, *leave);
    static_cast<void>(ticket.wait());
    require(restore->finish() && leave->finish(), "transfers beside a command failed");
    require(*mismatched == 0, "a command read other data than it wrote beside a transfer");
    require(contents(pages, restored) == expected,
            "a page restored beside a command does not hold the disk's bytes");
    require(holdsPage(readSlot(*file, other), leaving) && contents(pages, demoted) == leaving,
            "a page demoted beside a command reached the disk changed");
    for (size_t entry = 0; entry < around.size(); ++entry) {
      const auto *slab = reinterpret_cast<const uint32_t *>(base + (around[entry] - address));
      for (uint32_t word = 0; word < words; ++word) {
        require(slab[word] == (seeds.back() ^ (static_cast<uint32_t>(entry) * 131071u + word)),
                "a transfer changed a page the command was writing");
      }
    }
  }
  // The next command reads the restored page through its address.
  *mismatched = 0;
  static_cast<void>(backend.submit(check(ownTable, own.size(), ownSeed)));
  require(*mismatched == 0, "kernels read other data than the restore left in the page");
  std::cout << "transfers beside a command " << kv::formatName(format) << ": the restore landed within "
            << rounds << " rounds of writes and checks\n";
}

// A tier goes away with transfers still in flight. They read their pages in
// place, so the worker has to be done with them first.
void teardown(metal::MetalBackend &backend, engine::MemoryGovernor &governor,
              kv::Format format) {
  const kv::Layout layout{16, 4, 256, format}; // Qwen3.8-27B
  kv::PageStorage pages(backend, governor.allocationAdmission(), layout,
                        layout.minimumExtentPages(), layout.minimumExtentPages());
  const uint64_t slotBytes = KvPageTier::slotBytesFor(pages);
  auto file = std::make_shared<SlotFile>(slotBytes, 16 * slotBytes);
  {
    KvPageTier tier(pages, file, 16);
    std::vector<std::unique_ptr<KvTransfer>> demotions;
    for (uint32_t page = 0; page < 8; ++page) {
      auto slot = tier.acquireSlot();
      require(slot != nullptr, "the quota did not hold a demotion");
      demotions.push_back(tier.demote(page, std::move(slot), {}));
      require(demotions.back() != nullptr, "a demotion within the share was refused");
    }
  }
  require(file->idle(), "the tier left its pages with disk IO still running");
  std::cout << "teardown tests passed\n";
}

void run(const std::string &metallib) {
  metal::MetalBackend backend(metallib);
  engine::MemoryGovernor governor(
      backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1);
  for (const auto format : {kv::Format::Int8, kv::Format::BFloat16}) {
    roundTrip(backend, governor, kv::Layout{2, 2, 256, format});
    roundTrip(backend, governor, kv::Layout{10, 2, 256, format}); // Qwen3.6-35B
    roundTrip(backend, governor, kv::Layout{16, 4, 256, format}); // Qwen3.8-27B
    limits(backend, governor, format);
    allocationFailure(backend, governor, format);
    besideACommand(backend, governor, format);
    teardown(backend, governor, format);
  }
  std::cout << "kv page tier tests passed\n";
}

} // namespace

int main(int argc, const char **argv) {
  if (argc < 2) {
    std::cerr << "usage: kv_page_tier_test METALLIB\n";
    return 2;
  }
  try {
    run(argv[1]);
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
