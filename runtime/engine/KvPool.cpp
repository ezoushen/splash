#include "engine/KvPool.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace splash::engine {
namespace {

double millisecondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

} // namespace

KvPool::KvPool(kv::ExtentStorage &storage)
    : storage_(storage), pages_(storage.pageCount()) {
  if (pages_.empty() || !storage_.bytesPerPage()) {
    throw std::invalid_argument("invalid KV extent storage");
  }
  for (uint32_t first = 0; first < pages_.size();) {
    const uint32_t count = storage_.extentPageCount(first);
    if (!count || storage_.extentFirstPage(first) != first ||
        count > pages_.size() - first) {
      throw std::invalid_argument("invalid KV extent geometry");
    }
    const uint32_t extent = static_cast<uint32_t>(extents_.size());
    ExtentRecord record;
    record.firstPage = first;
    record.pageCount = count;
    record.allocated = storage_.isAllocated(first);
    extents_.push_back(record);
    if (record.allocated)
      allocatedPages_ += count;
    for (uint32_t page = first; page < first + count; ++page)
      pages_[page].extent = extent;
    first += count;
  }
  for (uint32_t page = static_cast<uint32_t>(pages_.size()); page > 0;) {
    --page;
    insertFree(page, extents_[pages_[page].extent].allocated
                         ? FreeClass::Allocated
                         : FreeClass::Unallocated);
  }
  for (uint32_t extent = 0; extent < extents_.size(); ++extent) {
    if (extents_[extent].allocated)
      setExtentReclaimable(extent, true);
  }
}

KvPageAcquisition KvPool::acquirePages(uint32_t count, bool prefixOwner) {
  if (!count)
    return {};

  std::vector<uint32_t> selected;
  selected.reserve(count);
  auto returnSelected = [&] {
    for (auto p = selected.rbegin(); p != selected.rend(); ++p)
      insertFree(*p, FreeClass::Allocated);
  };
  while (selected.size() < count) {
    if (freePages_) {
      selected.push_back(popFree());
      continue;
    }
    // Page ids cover every extent the budget could hold, so only a storage
    // with fewer of them than its budget (a test's) runs out.
    if (!unallocated_.count) {
      returnSelected();
      return {{}, KvPageAcquireFailure::Denied, metal::AllocationFailure::Capacity};
    }
    const uint32_t page = unallocated_.head;
    const uint32_t extent = pages_[page].extent;
    metal::AllocationResult allocated = false;
    const auto growth = std::chrono::steady_clock::now();
    try {
      allocated = storage_.ensureAllocated(page);
    } catch (...) {
      returnSelected();
      throw;
    }
    if (!allocated) {
      // The extents this acquisition allocated stay, reclaimable: the
      // budget admitted them, and the retry that follows a reclaim takes
      // their pages first instead of allocating them again. A reclaim pass
      // returns them if they stay unused.
      returnSelected();
      return {{}, KvPageAcquireFailure::Denied, allocated.failure};
    }
    ++extentAllocations_;
    extentAllocateMaxMilliseconds_ =
        std::max(extentAllocateMaxMilliseconds_, millisecondsSince(growth));
    setExtentAllocated(extent, true);
  }

  for (uint32_t page : selected) {
    PageRecord &record = pages_[page];
    markUsed(page);
    uint32_t &references =
        prefixOwner ? record.prefixReferences : record.activeReferences;
    references = 1;
    ++(prefixOwner ? prefixPages_ : activePages_);
  }
  return {std::move(selected), KvPageAcquireFailure::None};
}

void KvPool::retainPage(uint32_t page, bool prefixOwner) {
  if (!storage_.isAllocated(page)) {
    throw std::logic_error("cannot retain a KV page of an unallocated extent");
  }
  PageRecord &record = pages_.at(page);
  uint32_t &references =
      prefixOwner ? record.prefixReferences : record.activeReferences;
  if (references == std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error("KV page reference overflow");
  }
  if (pageFree(page))
    markUsed(page);
  if (!references)
    ++(prefixOwner ? prefixPages_ : activePages_);
  ++references;
}

void KvPool::releasePage(uint32_t page, bool prefixOwner) {
  PageRecord &record = pages_.at(page);
  uint32_t &references =
      prefixOwner ? record.prefixReferences : record.activeReferences;
  if (!references)
    throw std::logic_error("invalid KV page release");
  --references;
  if (!references)
    --(prefixOwner ? prefixPages_ : activePages_);
  if (pageFree(page))
    markFree(page);
}

uint32_t KvPool::pageCount() const noexcept {
  return static_cast<uint32_t>(pages_.size());
}

uint64_t KvPool::bytesPerPage() const noexcept {
  return storage_.bytesPerPage();
}

uint32_t KvPool::freePageCount() const noexcept { return freePages_; }

uint32_t KvPool::activeReferences(uint32_t page) const {
  return pages_.at(page).activeReferences;
}

bool KvPool::pageFree(uint32_t page) const {
  const PageRecord &record = pages_.at(page);
  return !record.activeReferences && !record.prefixReferences;
}

uint64_t KvPool::allocatedBytes() const noexcept {
  return uint64_t{allocatedPages_} * bytesPerPage();
}

uint32_t KvPool::reclaimEmptyExtents(bool keepRunway, uint32_t limit) {
  uint32_t reclaimed = 0;
  bool kept = false;
  uint32_t extent = reclaimableExtents_.head;
  while (extent != noIndex && reclaimed < limit) {
    const uint32_t next = extents_[extent].nextReclaimable;
    if (keepRunway && !kept) {
      kept = true;
    } else if (releaseExtent(extent)) {
      setExtentAllocated(extent, false);
      ++reclaimed;
    }
    extent = next;
  }
  return reclaimed;
}

bool KvPool::releaseExtent(uint32_t extent) {
  const auto start = std::chrono::steady_clock::now();
  if (!storage_.releaseExtentOf(extents_[extent].firstPage))
    return false;
  ++extentReleases_;
  extentReleaseMaxMilliseconds_ =
      std::max(extentReleaseMaxMilliseconds_, millisecondsSince(start));
  return true;
}

KvPageMoves KvPool::compactExtent(std::span<const uint32_t> fixed) {
  // Pages move only into extents that hold pages already: an empty extent is
  // released as it is, never filled to release another.
  uint32_t freeInUse = 0;
  for (const ExtentRecord &extent : extents_) {
    if (extent.allocated && extent.usedPages)
      freeInUse += extent.freePages.count;
  }
  uint32_t emptied = noIndex;
  for (uint32_t index = 0; index < extents_.size(); ++index) {
    const ExtentRecord &extent = extents_[index];
    if (!extent.allocated || !extent.usedPages ||
        extent.usedPages > freeInUse - extent.freePages.count ||
        (emptied != noIndex && extent.usedPages >= extents_[emptied].usedPages))
      continue;
    if (std::none_of(fixed.begin(), fixed.end(), [&](uint32_t page) {
          return pages_.at(page).extent == index;
        }))
      emptied = index;
  }
  if (emptied == noIndex)
    return {};

  std::vector<uint32_t> receivers;
  for (uint32_t index = 0; index < extents_.size(); ++index) {
    const ExtentRecord &extent = extents_[index];
    if (index != emptied && extent.allocated && extent.usedPages &&
        extent.freePages.count)
      receivers.push_back(index);
  }
  std::stable_sort(receivers.begin(), receivers.end(),
                   [&](uint32_t left, uint32_t right) {
                     return extents_[left].usedPages > extents_[right].usedPages;
                   });
  const ExtentRecord &source = extents_[emptied];
  std::vector<kv::PageCopy> copies;
  copies.reserve(source.usedPages);
  auto receiver = receivers.begin();
  uint32_t to = extents_[*receiver].freePages.head;
  for (uint32_t page = source.firstPage;
       page < source.firstPage + source.pageCount; ++page) {
    if (pageFree(page))
      continue;
    // The receivers' free pages cover the extent, so one always follows.
    while (to == noIndex)
      to = extents_[*++receiver].freePages.head;
    copies.push_back({page, to});
    to = pages_[to].nextFree;
  }

  const auto start = std::chrono::steady_clock::now();
  storage_.copyPages(copies);
  KvPageMoves moves{source.firstPage, std::vector<uint32_t>(source.pageCount)};
  std::iota(moves.destinations.begin(), moves.destinations.end(), source.firstPage);
  for (const kv::PageCopy &copy : copies) {
    markUsed(copy.to);
    pages_[copy.to].activeReferences =
        std::exchange(pages_[copy.from].activeReferences, 0);
    pages_[copy.to].prefixReferences =
        std::exchange(pages_[copy.from].prefixReferences, 0);
    markFree(copy.from);
    moves.destinations[copy.from - moves.firstPage] = copy.to;
  }
  ++extentCompactions_;
  pagesMoved_ += copies.size();
  extentCompactMaxMilliseconds_ =
      std::max(extentCompactMaxMilliseconds_, millisecondsSince(start));
  return moves;
}

KvPoolSnapshot KvPool::snapshot() const {
  KvPoolSnapshot result;
  result.pagesAllocated = allocatedPages_;
  result.pagesActive = activePages_;
  result.pagesPrefix = prefixPages_;
  result.pagesFree = freePages_;
  result.allocatedBytes = allocatedBytes();
  result.reclaimableExtents = reclaimableExtents_.count;
  result.reclaimableBytes = reclaimableBytes_;
  result.extentAllocations = extentAllocations_;
  result.extentReleases = extentReleases_;
  result.extentAllocateMaxMilliseconds = extentAllocateMaxMilliseconds_;
  result.extentReleaseMaxMilliseconds = extentReleaseMaxMilliseconds_;
  result.extentCompactions = extentCompactions_;
  result.pagesMoved = pagesMoved_;
  result.extentCompactMaxMilliseconds = extentCompactMaxMilliseconds_;
  return result;
}

KvPool::IndexList &KvPool::freeList(FreeClass kind, uint32_t page) noexcept {
  if (kind == FreeClass::Allocated)
    return extents_[pages_[page].extent].freePages;
  if (kind == FreeClass::Unallocated)
    return unallocated_;
  std::terminate();
}

void KvPool::insertFree(uint32_t page, FreeClass kind) noexcept {
  PageRecord &record = pages_[page];
  if (kind == FreeClass::None || record.freeClass != FreeClass::None ||
      !pageFree(page)) {
    std::terminate();
  }
  IndexList &list = freeList(kind, page);
  record.previousFree = noIndex;
  record.nextFree = list.head;
  record.freeClass = kind;
  if (list.head != noIndex)
    pages_[list.head].previousFree = page;
  list.head = page;
  ++list.count;
  if (kind == FreeClass::Allocated) {
    ++freePages_;
    packingExtent_ = noIndex;
  }
}

void KvPool::removeFree(uint32_t page) noexcept {
  PageRecord &record = pages_[page];
  if (record.freeClass == FreeClass::None)
    std::terminate();
  const FreeClass kind = record.freeClass;
  IndexList &list = freeList(kind, page);
  if (record.previousFree == noIndex) {
    if (list.head != page)
      std::terminate();
    list.head = record.nextFree;
  } else {
    pages_[record.previousFree].nextFree = record.nextFree;
  }
  if (record.nextFree != noIndex)
    pages_[record.nextFree].previousFree = record.previousFree;
  record.previousFree = noIndex;
  record.nextFree = noIndex;
  record.freeClass = FreeClass::None;
  if (!list.count)
    std::terminate();
  --list.count;
  if (kind == FreeClass::Allocated) {
    if (!freePages_)
      std::terminate();
    --freePages_;
  }
}

uint32_t KvPool::popFree() noexcept {
  const uint32_t page = extents_[packingExtent()].freePages.head;
  if (page == noIndex)
    std::terminate();
  removeFree(page);
  return page;
}

// The allocated extent with the most live pages that still has a free page;
// ties go to the lowest index, and empty extents lose to any used one. The
// answer only changes when another extent gains or loses a page, so it is
// reused until then.
uint32_t KvPool::packingExtent() noexcept {
  if (packingExtent_ != noIndex && extents_[packingExtent_].freePages.count)
    return packingExtent_;
  uint32_t best = noIndex;
  for (uint32_t index = 0; index < extents_.size(); ++index) {
    const ExtentRecord &extent = extents_[index];
    if (extent.freePages.count &&
        (best == noIndex || extent.usedPages > extents_[best].usedPages)) {
      best = index;
    }
  }
  if (best == noIndex)
    std::terminate();
  packingExtent_ = best;
  return best;
}

void KvPool::markUsed(uint32_t page) noexcept {
  PageRecord &record = pages_[page];
  if (!pageFree(page))
    std::terminate();
  if (record.freeClass != FreeClass::None) {
    removeFree(page);
    if (record.extent != packingExtent_)
      packingExtent_ = noIndex;
  }
  ExtentRecord &extent = extents_[record.extent];
  if (!extent.usedPages)
    setExtentReclaimable(record.extent, false);
  ++extent.usedPages;
}

void KvPool::markFree(uint32_t page) noexcept {
  PageRecord &record = pages_[page];
  if (!pageFree(page) || record.freeClass != FreeClass::None)
    std::terminate();
  ExtentRecord &extent = extents_[record.extent];
  if (!extent.usedPages)
    std::terminate();
  --extent.usedPages;
  insertFree(page, extent.allocated ? FreeClass::Allocated : FreeClass::Unallocated);
  if (!extent.usedPages && extent.allocated)
    setExtentReclaimable(record.extent, true);
}

void KvPool::setExtentAllocated(uint32_t extentIndex, bool allocated) noexcept {
  ExtentRecord &extent = extents_[extentIndex];
  if (extent.allocated == allocated)
    return;
  if (extent.usedPages)
    std::terminate();
  setExtentReclaimable(extentIndex, false);
  extent.allocated = allocated;
  if (allocated) {
    allocatedPages_ += extent.pageCount;
  } else {
    if (allocatedPages_ < extent.pageCount)
      std::terminate();
    allocatedPages_ -= extent.pageCount;
  }
  for (uint32_t page = extent.firstPage + extent.pageCount;
       page > extent.firstPage;) {
    --page;
    if (pages_[page].freeClass == FreeClass::None)
      continue;
    removeFree(page);
    insertFree(page, allocated ? FreeClass::Allocated : FreeClass::Unallocated);
  }
  if (allocated)
    setExtentReclaimable(extentIndex, true);
}

void KvPool::setExtentReclaimable(uint32_t extentIndex,
                                  bool reclaimable) noexcept {
  ExtentRecord &extent = extents_[extentIndex];
  if (extent.reclaimable == reclaimable)
    return;
  if (reclaimable) {
    if (!extent.allocated || extent.usedPages)
      std::terminate();
    extent.previousReclaimable = noIndex;
    extent.nextReclaimable = reclaimableExtents_.head;
    if (reclaimableExtents_.head != noIndex) {
      extents_[reclaimableExtents_.head].previousReclaimable = extentIndex;
    }
    reclaimableExtents_.head = extentIndex;
    ++reclaimableExtents_.count;
    reclaimableBytes_ += uint64_t{extent.pageCount} * bytesPerPage();
    extent.reclaimable = true;
    return;
  }
  if (extent.previousReclaimable == noIndex) {
    if (reclaimableExtents_.head != extentIndex)
      std::terminate();
    reclaimableExtents_.head = extent.nextReclaimable;
  } else {
    extents_[extent.previousReclaimable].nextReclaimable =
        extent.nextReclaimable;
  }
  if (extent.nextReclaimable != noIndex) {
    extents_[extent.nextReclaimable].previousReclaimable =
        extent.previousReclaimable;
  }
  extent.previousReclaimable = noIndex;
  extent.nextReclaimable = noIndex;
  extent.reclaimable = false;
  if (!reclaimableExtents_.count ||
      reclaimableBytes_ < uint64_t{extent.pageCount} * bytesPerPage()) {
    std::terminate();
  }
  --reclaimableExtents_.count;
  reclaimableBytes_ -= uint64_t{extent.pageCount} * bytesPerPage();
}

} // namespace splash::engine
