#pragma once

#include "ops/PagedKv.hpp"

#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace splash::engine {

struct KvPoolSnapshot {
  // Pages of allocated extents: all of them, those nothing holds, and those
  // requests and the cache hold.
  uint32_t pagesAllocated = 0;
  uint32_t pagesFree = 0;
  uint32_t pagesActive = 0;
  uint32_t pagesPrefix = 0;
  // Extents none of whose pages is held, which a reclaim releases at once.
  uint32_t reclaimableExtents = 0;
  uint64_t allocatedBytes = 0;
  uint64_t reclaimableBytes = 0;
  // Extents allocated and released through the pool, and the longest
  // allocation and release of one: what memory costs the serving loop per
  // extent. How long a whole reclaim pass holds the loop shows in its
  // longest tick.
  uint64_t extentAllocations = 0;
  uint64_t extentReleases = 0;
  double extentAllocateMaxMilliseconds = 0.0;
  double extentReleaseMaxMilliseconds = 0.0;
  // Extents emptied by moving their pages, the pages moved, and the longest
  // emptying of one.
  uint64_t extentCompactions = 0;
  uint64_t pagesMoved = 0;
  double extentCompactMaxMilliseconds = 0.0;
};

// Where the pages of an emptied extent went. Whoever names one of its pages
// follows it.
struct KvPageMoves final {
  uint32_t firstPage = 0;
  // By offset in the extent: the page that took the page's content, or the
  // page itself where it was free.
  std::vector<uint32_t> destinations;

  [[nodiscard]] bool empty() const noexcept { return destinations.empty(); }
  // The page that holds what `page` held.
  [[nodiscard]] uint32_t follow(uint32_t page) const noexcept {
    return page >= firstPage && page - firstPage < destinations.size()
               ? destinations[page - firstPage]
               : page;
  }
};

enum class KvPageAcquireFailure : uint8_t {
  None,
  // The pool could not grow by an extent; allocationFailure says why.
  Denied,
  // A transfer in flight (a KV demotion, a KV restore or the one state write)
  // holds what the request needs; retry when it lands.
  Pending,
};

struct KvPageAcquisition {
  std::vector<uint32_t> pages;
  KvPageAcquireFailure failure = KvPageAcquireFailure::None;
  metal::AllocationFailure allocationFailure = metal::AllocationFailure::None;

  [[nodiscard]] bool granted() const noexcept {
    return failure == KvPageAcquireFailure::None;
  }
};

// Sole owner of KV page references and of which extents are allocated.
// Resource policy may ask for pages or release references, but cannot
// directly allocate or release Metal memory. Free pages are handed out from
// the allocated extent with the most live pages first, so partially used
// extents fill up, empty extents are touched last, and cold extents drain to
// empty, the only state in which an extent can be released. Held pages are
// scattered over the extents all the same; compactExtent() empties one more
// extent whenever the free pages of the others cover it.
class KvPool final {
public:
  explicit KvPool(kv::ExtentStorage &storage);

  [[nodiscard]] KvPageAcquisition acquirePages(uint32_t count,
                                               bool prefixOwner);
  void retainPage(uint32_t page, bool prefixOwner);
  void releasePage(uint32_t page, bool prefixOwner);

  [[nodiscard]] uint32_t pageCount() const noexcept;
  [[nodiscard]] uint64_t bytesPerPage() const noexcept;
  // Free pages of allocated extents; acquisition hands these out first.
  [[nodiscard]] uint32_t freePageCount() const noexcept;
  [[nodiscard]] uint32_t activeReferences(uint32_t page) const;
  [[nodiscard]] bool pageFree(uint32_t page) const;
  [[nodiscard]] uint64_t allocatedBytes() const noexcept;

  // Releases completely unreferenced extents, at most `limit` of them.
  // keepRunway retains one of them to avoid adding allocation latency to the
  // next request.
  [[nodiscard]] uint32_t
  reclaimEmptyExtents(bool keepRunway,
                      uint32_t limit = std::numeric_limits<uint32_t>::max());
  // Empties the allocated extent that holds the fewest pages, by moving each
  // of them to a free page of the other extents that hold pages, fullest
  // first: free pages scattered over the pool become an empty extent, which
  // a reclaim releases. References move with their pages, and the result
  // says where each went, for the caller to re-point whoever names them.
  // Nothing moves, and the result is empty, unless those free pages cover
  // the extent. An extent that holds a page of `fixed` is not emptied.
  [[nodiscard]] KvPageMoves compactExtent(std::span<const uint32_t> fixed);
  [[nodiscard]] KvPoolSnapshot snapshot() const;

private:
  static constexpr uint32_t noIndex = std::numeric_limits<uint32_t>::max();
  enum class FreeClass : uint8_t { None, Allocated, Unallocated };

  struct PageRecord {
    uint32_t activeReferences = 0;
    uint32_t prefixReferences = 0;
    uint32_t extent = noIndex;
    uint32_t previousFree = noIndex;
    uint32_t nextFree = noIndex;
    FreeClass freeClass = FreeClass::None;
  };

  struct IndexList {
    uint32_t head = noIndex;
    uint32_t count = 0;
  };

  struct ExtentRecord {
    uint32_t firstPage = 0;
    uint32_t pageCount = 0;
    uint32_t usedPages = 0;
    uint32_t previousReclaimable = noIndex;
    uint32_t nextReclaimable = noIndex;
    IndexList freePages;
    bool allocated = false;
    bool reclaimable = false;
  };

  [[nodiscard]] IndexList &freeList(FreeClass kind, uint32_t page) noexcept;
  void insertFree(uint32_t page, FreeClass kind) noexcept;
  void removeFree(uint32_t page) noexcept;
  [[nodiscard]] uint32_t popFree() noexcept;
  [[nodiscard]] uint32_t packingExtent() noexcept;
  void markUsed(uint32_t page) noexcept;
  void markFree(uint32_t page) noexcept;
  void setExtentAllocated(uint32_t extent, bool allocated) noexcept;
  void setExtentReclaimable(uint32_t extent, bool reclaimable) noexcept;

  bool releaseExtent(uint32_t extent);

  kv::ExtentStorage &storage_;
  uint64_t extentAllocations_ = 0;
  uint64_t extentReleases_ = 0;
  double extentAllocateMaxMilliseconds_ = 0.0;
  double extentReleaseMaxMilliseconds_ = 0.0;
  uint64_t extentCompactions_ = 0;
  uint64_t pagesMoved_ = 0;
  double extentCompactMaxMilliseconds_ = 0.0;
  std::vector<PageRecord> pages_;
  std::vector<ExtentRecord> extents_;
  uint32_t freePages_ = 0;
  // The extent currently being filled. Stays valid while only this extent
  // changes, so a burst of allocations rescans the extents once per extent
  // it moves into.
  uint32_t packingExtent_ = noIndex;
  IndexList unallocated_;
  IndexList reclaimableExtents_;
  uint32_t activePages_ = 0;
  uint32_t prefixPages_ = 0;
  uint32_t allocatedPages_ = 0;
  uint64_t reclaimableBytes_ = 0;
};

} // namespace splash::engine
