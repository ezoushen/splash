#include "engine/KvPool.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>

namespace splash::engine {
namespace {

double millisecondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

} // namespace

KvPool::KvPool(KvBacking &backing)
    : backing_(backing), pages_(backing.pageCount()) {
  if (pages_.empty() || !backing_.bytesPerPage()) {
    throw std::invalid_argument("invalid elastic KV backing");
  }
  for (uint32_t first = 0; first < pages_.size();) {
    const uint32_t count = backing_.extentPageCount(first);
    if (!count || backing_.extentFirstPage(first) != first ||
        count > pages_.size() - first) {
      throw std::invalid_argument("invalid elastic KV extent geometry");
    }
    const uint32_t extent = static_cast<uint32_t>(extents_.size());
    ExtentRecord record;
    record.firstPage = first;
    record.pageCount = count;
    record.resident = backing_.isResident(first);
    extents_.push_back(record);
    if (record.resident)
      residentPages_ += count;
    for (uint32_t page = first; page < first + count; ++page)
      pages_[page].extent = extent;
    first += count;
  }
  for (uint32_t page = static_cast<uint32_t>(pages_.size()); page > 0;) {
    --page;
    insertFree(page, extents_[pages_[page].extent].resident
                         ? FreeClass::Resident
                         : FreeClass::Unbacked);
  }
  for (uint32_t extent = 0; extent < extents_.size(); ++extent) {
    if (extents_[extent].resident)
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
      insertFree(*p, FreeClass::Resident);
  };
  while (selected.size() < count) {
    if (freeResidentPages_) {
      selected.push_back(popFreeResident());
      continue;
    }
    // Page ids cover every extent the budget could hold, so only a backing
    // with fewer of them than its budget (a test's) runs out.
    if (!freeUnbacked_.count) {
      returnSelected();
      return {{}, KvPageAcquireFailure::Denied, metal::AllocationFailure::Capacity};
    }
    const uint32_t page = freeUnbacked_.head;
    const uint32_t extent = pages_[page].extent;
    metal::AllocationResult allocated = false;
    const auto growth = std::chrono::steady_clock::now();
    try {
      allocated = backing_.ensureResident(page);
    } catch (...) {
      returnSelected();
      throw;
    }
    if (!allocated) {
      // The extents this acquisition allocated stay resident and
      // reclaimable: the budget admitted them, and the retry that follows a
      // reclaim takes their pages first instead of allocating them again.
      // A reclaim pass returns them if they stay unused.
      returnSelected();
      return {{}, KvPageAcquireFailure::Denied, allocated.failure};
    }
    ++extentAllocations_;
    extentAllocateMaxMilliseconds_ =
        std::max(extentAllocateMaxMilliseconds_, millisecondsSince(growth));
    setExtentResident(extent, true);
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
  if (!backing_.isResident(page)) {
    throw std::logic_error("cannot retain an unbacked KV page");
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
  return backing_.bytesPerPage();
}

uint32_t KvPool::freePageCount() const noexcept {
  return freeResidentPages_ + freeUnbacked_.count;
}

uint32_t KvPool::freeResidentPageCount() const noexcept { return freeResidentPages_; }

uint32_t KvPool::activeReferences(uint32_t page) const {
  return pages_.at(page).activeReferences;
}

bool KvPool::pageFree(uint32_t page) const {
  const PageRecord &record = pages_.at(page);
  return !record.activeReferences && !record.prefixReferences;
}

uint64_t KvPool::residentBackingBytes() const noexcept {
  return uint64_t{residentPages_} * bytesPerPage();
}

uint32_t KvPool::reclaimEmptyExtents(bool keepRunway, uint32_t limit) {
  uint32_t reclaimed = 0;
  bool kept = false;
  uint32_t extent = reclaimableExtents_.head;
  while (extent != noIndex && reclaimed < limit) {
    const uint32_t next = extents_[extent].nextReclaimable;
    if (keepRunway && !kept) {
      kept = true;
    } else if (releaseBacking(extents_[extent].firstPage)) {
      setExtentResident(extent, false);
      ++reclaimed;
    }
    extent = next;
  }
  return reclaimed;
}

bool KvPool::releaseBacking(uint32_t page) {
  const auto start = std::chrono::steady_clock::now();
  if (!backing_.releaseBackingForPage(page))
    return false;
  ++extentReleases_;
  extentReleaseMaxMilliseconds_ =
      std::max(extentReleaseMaxMilliseconds_, millisecondsSince(start));
  return true;
}

KvPoolSnapshot KvPool::snapshot() const {
  KvPoolSnapshot result;
  result.pagesTotal = pageCount();
  result.pagesFree = freePageCount();
  result.pagesResident = residentPages_;
  result.pagesActive = activePages_;
  result.pagesPrefix = prefixPages_;
  result.pagesFreeResident = freeResidentPages_;
  result.residentBackingBytes = residentBackingBytes();
  result.reclaimableExtents = reclaimableExtents_.count;
  result.reclaimableBackingBytes = reclaimableBackingBytes_;
  result.extentAllocations = extentAllocations_;
  result.extentReleases = extentReleases_;
  result.extentAllocateMaxMilliseconds = extentAllocateMaxMilliseconds_;
  result.extentReleaseMaxMilliseconds = extentReleaseMaxMilliseconds_;
  return result;
}

KvPool::IndexList &KvPool::freeList(FreeClass kind, uint32_t page) noexcept {
  if (kind == FreeClass::Resident)
    return extents_[pages_[page].extent].freeResident;
  if (kind == FreeClass::Unbacked)
    return freeUnbacked_;
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
  if (kind == FreeClass::Resident) {
    ++freeResidentPages_;
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
  if (kind == FreeClass::Resident) {
    if (!freeResidentPages_)
      std::terminate();
    --freeResidentPages_;
  }
}

uint32_t KvPool::popFreeResident() noexcept {
  const uint32_t page = extents_[packingExtent()].freeResident.head;
  if (page == noIndex)
    std::terminate();
  removeFree(page);
  return page;
}

// The extent with the most live pages that still has a free resident page;
// ties go to the lowest index, and empty extents lose to any used one. The
// answer only changes when another extent gains or loses a page, so it is
// reused until then.
uint32_t KvPool::packingExtent() noexcept {
  if (packingExtent_ != noIndex && extents_[packingExtent_].freeResident.count)
    return packingExtent_;
  uint32_t best = noIndex;
  for (uint32_t index = 0; index < extents_.size(); ++index) {
    const ExtentRecord &extent = extents_[index];
    if (extent.freeResident.count &&
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
  insertFree(page, extent.resident ? FreeClass::Resident : FreeClass::Unbacked);
  if (!extent.usedPages && extent.resident)
    setExtentReclaimable(record.extent, true);
}

void KvPool::setExtentResident(uint32_t extentIndex, bool resident) noexcept {
  ExtentRecord &extent = extents_[extentIndex];
  if (extent.resident == resident)
    return;
  if (extent.usedPages)
    std::terminate();
  setExtentReclaimable(extentIndex, false);
  extent.resident = resident;
  if (resident) {
    residentPages_ += extent.pageCount;
  } else {
    if (residentPages_ < extent.pageCount)
      std::terminate();
    residentPages_ -= extent.pageCount;
  }
  for (uint32_t page = extent.firstPage + extent.pageCount;
       page > extent.firstPage;) {
    --page;
    if (pages_[page].freeClass == FreeClass::None)
      continue;
    removeFree(page);
    insertFree(page, resident ? FreeClass::Resident : FreeClass::Unbacked);
  }
  if (resident)
    setExtentReclaimable(extentIndex, true);
}

void KvPool::setExtentReclaimable(uint32_t extentIndex,
                                  bool reclaimable) noexcept {
  ExtentRecord &extent = extents_[extentIndex];
  if (extent.reclaimable == reclaimable)
    return;
  if (reclaimable) {
    if (!extent.resident || extent.usedPages)
      std::terminate();
    extent.previousReclaimable = noIndex;
    extent.nextReclaimable = reclaimableExtents_.head;
    if (reclaimableExtents_.head != noIndex) {
      extents_[reclaimableExtents_.head].previousReclaimable = extentIndex;
    }
    reclaimableExtents_.head = extentIndex;
    ++reclaimableExtents_.count;
    reclaimableBackingBytes_ += uint64_t{extent.pageCount} * bytesPerPage();
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
      reclaimableBackingBytes_ < uint64_t{extent.pageCount} * bytesPerPage()) {
    std::terminate();
  }
  --reclaimableExtents_.count;
  reclaimableBackingBytes_ -= uint64_t{extent.pageCount} * bytesPerPage();
}

} // namespace splash::engine
