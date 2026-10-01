#include "engine/KvPool.hpp"

#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

namespace {

using splash::engine::KvPool;
using splash::kv::ExtentStorage;

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

class TestStorage final : public ExtentStorage {
public:
    TestStorage(uint32_t pages, uint32_t extentPages,
                uint32_t allocatedExtents = 0)
        : TestStorage(std::vector<uint32_t>((pages + extentPages - 1) /
                                                extentPages,
                                            extentPages),
                      allocatedExtents) {
        if (!pages || !extentPages || pages % extentPages) {
            throw std::invalid_argument("invalid test extent geometry");
        }
    }
    // Explicit extent sizes: the pool accepts extents of different sizes.
    TestStorage(std::vector<uint32_t> extentSizes, uint32_t allocatedExtents)
        : allocated_(extentSizes.size(), false) {
        for (uint32_t size : extentSizes) {
            if (!size) throw std::invalid_argument("empty test extent");
            for (uint32_t page = 0; page < size; ++page) {
                extentOf_.push_back(static_cast<uint32_t>(firstPage_.size()));
            }
            firstPage_.push_back(pages_);
            extentPages_.push_back(size);
            pages_ += size;
        }
        for (uint32_t index = 0;
             index < allocatedExtents && index < allocated_.size(); ++index) {
            allocated_[index] = true;
        }
    }

    uint32_t pageCount() const noexcept override { return pages_; }
    uint64_t bytesPerPage() const noexcept override { return 100; }
    bool isAllocated(uint32_t page) const override {
        return allocated_.at(extentOf_.at(page));
    }
    splash::metal::AllocationResult ensureAllocated(uint32_t page) override {
        uint32_t extent = extentOf_.at(page);
        ++allocationAttempts;
        if (!admission) return false;
        if (failExtent && extent == *failExtent) return false;
        if (throwExtent && extent == *throwExtent)
            throw std::runtime_error("test allocation failure");
        allocated_.at(extent) = true;
        return true;
    }
    bool releaseExtentOf(uint32_t page) override {
        uint32_t extent = extentOf_.at(page);
        if (!allocated_.at(extent)) return false;
        allocated_[extent] = false;
        ++releasedExtents;
        return true;
    }
    uint32_t extentFirstPage(uint32_t page) const override {
        return firstPage_.at(extentOf_.at(page));
    }
    uint32_t extentPageCount(uint32_t page) const override {
        return extentPages_.at(extentOf_.at(page));
    }
    bool admission = true;
    std::optional<uint32_t> failExtent;
    std::optional<uint32_t> throwExtent;
    uint32_t allocationAttempts = 0;
    uint32_t releasedExtents = 0;

private:
    uint32_t pages_ = 0;
    std::vector<uint32_t> extentPages_;
    std::vector<uint32_t> firstPage_;
    std::vector<uint32_t> extentOf_;
    std::vector<bool> allocated_;
};

void release(KvPool &pool, const std::vector<uint32_t> &pages,
             bool prefix = false) {
    for (uint32_t page : pages) pool.releasePage(page, prefix);
}

void testGrowthPacksAllocatedExtents() {
    TestStorage storage(12, 4, 1);
    KvPool pool(storage);
    auto pages = pool.acquirePages(5, false);
    require(pages.granted() && pages.pages.size() == 5,
            "elastic pool did not acquire requested pages");
    require(pages.pages[0] == 0 && pages.pages[3] == 3 &&
                pages.pages[4] == 4,
            "elastic pool did not fill its allocated runway first");
    auto live = pool.snapshot();
    require(live.pagesAllocated == 8 && live.pagesActive == 5 &&
                live.pagesFree == 3 &&
                live.allocatedBytes == 800,
            "elastic growth accounting is incorrect");

    release(pool, pages.pages);
    require(pool.reclaimEmptyExtents(true) == 1,
            "reclaim did not retain exactly one warm runway");
    auto reclaimed = pool.snapshot();
    require(reclaimed.pagesAllocated == 4 &&
                reclaimed.reclaimableExtents == 1 &&
                storage.releasedExtents == 1,
            "empty extent was not returned exactly");
    require(reclaimed.extentAllocations == 1 && reclaimed.extentReleases == 1,
            "the pool did not count its growth and release");
}

void testFailedGrowthRollsBackAtomically() {
    TestStorage storage(12, 4);
    storage.failExtent = 1;
    KvPool pool(storage);
    auto pages = pool.acquirePages(5, false);
    require(!pages.granted() &&
                pages.failure ==
                    splash::engine::KvPageAcquireFailure::Denied,
            "failed growth was not reported as denied");
    auto status = pool.snapshot();
    require(status.pagesAllocated == 4 && status.pagesFree == 4 &&
                status.pagesActive == 0 && status.pagesPrefix == 0 &&
                storage.allocationAttempts == 2,
            "failed growth leaked references");
}

// A failed acquisition keeps the extents it allocated, reclaimable: the
// retry takes their pages instead of allocating them again, and a reclaim
// pass returns them if nothing does.
void testFailedGrowthKeepsItsExtentsForTheRetry() {
    TestStorage storage(16, 4);
    storage.failExtent = 2;
    KvPool pool(storage);
    const auto before = pool.snapshot().extentReleases;
    auto pages = pool.acquirePages(9, false);
    auto status = pool.snapshot();
    require(!pages.granted() && storage.releasedExtents == 0 &&
                pool.snapshot().extentReleases == before &&
                status.pagesAllocated == 8 && status.reclaimableExtents == 2 &&
                status.pagesFree == 8 && status.pagesActive == 0 &&
                status.extentAllocations == 2,
            "a failed acquisition did not keep the extents it allocated");
    storage.failExtent.reset();
    pages = pool.acquirePages(9, false);
    require(pages.granted() && pool.snapshot().extentAllocations == 3 &&
                storage.releasedExtents == 0,
            "the retry allocated again the extents it was denied with");
    release(pool, pages.pages);
    require(pool.reclaimEmptyExtents(false) == 3 &&
                pool.snapshot().pagesAllocated == 0,
            "a reclaim pass did not return the extents the retry left");
}

// A storage that throws while allocating leaves every page free and the
// accounting whole; the extent allocated before it stays, reclaimable, and
// the pool keeps serving.
void testThrowingStorageKeepsAccounting() {
    TestStorage storage(12, 4);
    storage.throwExtent = 1;
    KvPool pool(storage);
    bool threw = false;
    try {
        static_cast<void>(pool.acquirePages(5, false));
    } catch (const std::runtime_error &) {
        threw = true;
    }
    auto status = pool.snapshot();
    require(threw && status.pagesActive == 0 && status.pagesFree == 4 &&
                status.pagesAllocated == 4 && status.reclaimableExtents == 1,
            "throwing storage leaked pages or broke the extent accounting");
    storage.throwExtent.reset();
    auto pages = pool.acquirePages(5, false);
    require(pages.granted() && pool.snapshot().pagesActive == 5,
            "pool did not serve after a throwing storage");
}

void testPressureReusesFreePagesAndDeniesGrowth() {
    TestStorage storage(8, 4, 1);
    KvPool pool(storage);
    auto active = pool.acquirePages(2, false);
    require(active.granted() && active.pages.size() == 2,
            "pressure setup did not acquire active pages");
    storage.admission = false;
    auto reused = pool.acquirePages(1, false);
    require(reused.granted() && reused.pages.size() == 1 &&
                reused.pages.front() == 2,
            "critical pressure rejected a free page of an allocated extent");
    auto denied = pool.acquirePages(2, false);
    require(!denied.granted() &&
                denied.failure ==
                    splash::engine::KvPageAcquireFailure::Denied,
            "critical pressure admitted a new extent");
    require(pool.activeReferences(active.pages[0]) == 1 &&
                pool.activeReferences(active.pages[1]) == 1 &&
                pool.activeReferences(reused.pages.front()) == 1 &&
                pool.snapshot().pagesActive == 3,
            "critical pressure corrupted existing active references");
    release(pool, reused.pages);
    release(pool, active.pages);
    require(pool.reclaimEmptyExtents(false) == 1 &&
                pool.snapshot().pagesAllocated == 0,
            "pressure cleanup did not reclaim the empty extent");
}

// One pass releases every empty extent but the runway, however many there
// are; a pass without the runway releases that one too.
void testPassReleasesEveryEmptyExtent() {
    constexpr uint32_t extents = 200;
    TestStorage storage(4 * extents, 4, extents);
    KvPool pool(storage);
    auto pages = pool.acquirePages(4 * extents, false);
    require(pages.granted() && pool.snapshot().pagesAllocated == 4 * extents,
            "release setup did not acquire every page");
    release(pool, pages.pages);
    require(pool.snapshot().reclaimableExtents == extents,
            "every empty extent was not reclaimable");
    const auto before = pool.snapshot().extentReleases;
    require(pool.reclaimEmptyExtents(true) == extents - 1 &&
                storage.releasedExtents == extents - 1 &&
                pool.snapshot().extentReleases == before + extents - 1 &&
                pool.snapshot().reclaimableExtents == 1,
            "a pass did not release every empty extent but the runway");
    require(pool.reclaimEmptyExtents(false) == 1 &&
                pool.snapshot().pagesAllocated == 0 &&
                pool.snapshot().extentReleases == extents,
            "a pass without the runway did not release it");
}

void testFullestExtentFillsFirstSoColdExtentsDrain() {
    TestStorage storage(12, 4, 3);
    KvPool pool(storage);
    auto all = pool.acquirePages(12, false);
    require(all.granted() && all.pages.size() == 12,
            "fill setup did not acquire every page");
    // Leave extent 0 with three holes, extent 1 with one and extent 2 with two.
    release(pool, {0, 1, 2, 5, 8, 9});
    require(pool.snapshot().pagesFree == 6 &&
                pool.snapshot().reclaimableExtents == 0,
            "partial release accounting is incorrect");

    // New pages come from the fullest extents; the coldest keeps its holes.
    auto refill = pool.acquirePages(2, false);
    require(refill.granted() && refill.pages.size() == 2 &&
                refill.pages[0] == 5 &&
                (refill.pages[1] == 8 || refill.pages[1] == 9),
            "refill did not take pages from the fullest extents first");
    auto again = pool.acquirePages(1, false);
    require(again.granted() && again.pages.front() / 4 == 2,
            "allocation did not continue with the fullest extent");

    // Its last page going cold empties the extent so it can be released.
    release(pool, {3});
    require(pool.snapshot().reclaimableExtents == 1 &&
                pool.reclaimEmptyExtents(false) == 1 &&
                pool.snapshot().pagesAllocated == 8 &&
                storage.releasedExtents == 1,
            "drained extent was not released");
}

void testShorterTrailingExtentIsNotPreferredForBeingSmall() {
    // Having fewer free pages only because an extent is small must not rank
    // it as the fullest.
    TestStorage storage({4, 4, 2}, 3);
    KvPool pool(storage);
    auto all = pool.acquirePages(10, false);
    require(all.granted() && all.pages.size() == 10,
            "trailing extent setup did not acquire every page");

    // Partially used: extent 0 keeps two live pages, the tail keeps one.
    release(pool, {0, 1, 8});
    auto page = pool.acquirePages(1, false);
    require(page.granted() && page.pages.front() < 4,
            "partial trailing extent was refilled ahead of a fuller extent");
    release(pool, {9});
    require(pool.snapshot().reclaimableExtents == 1,
            "trailing extent did not drain after its last page was freed");

    // Empty: the tail's two free pages still lose to extent 0's one live page.
    release(pool, {page.pages.front(), 2});
    auto next = pool.acquirePages(1, false);
    require(next.granted() && next.pages.front() < 4,
            "empty trailing extent was refilled ahead of a partial extent");
    require(pool.reclaimEmptyExtents(false) == 1 &&
                pool.snapshot().pagesAllocated == 8,
            "empty trailing extent was not released");
}

void testPrefixAndActiveReferencesHoldTheExtent() {
    TestStorage storage(8, 4, 1);
    KvPool pool(storage);
    auto active = pool.acquirePages(1, false);
    require(active.granted(), "shared reference setup failed");
    pool.retainPage(active.pages.front(), true);
    pool.releasePage(active.pages.front(), false);
    require(pool.reclaimEmptyExtents(false) == 0 &&
                pool.snapshot().pagesPrefix == 1,
            "prefix-owned extent was reclaimed while live");
    pool.releasePage(active.pages.front(), true);
    require(pool.reclaimEmptyExtents(false) == 1,
            "last prefix release did not make extent reclaimable");
}

}  // namespace

int main() {
    try {
        testGrowthPacksAllocatedExtents();
        testFailedGrowthRollsBackAtomically();
        testFailedGrowthKeepsItsExtentsForTheRetry();
        testThrowingStorageKeepsAccounting();
        testPressureReusesFreePagesAndDeniesGrowth();
        testPassReleasesEveryEmptyExtent();
        testFullestExtentFillsFirstSoColdExtentsDrain();
        testShorterTrailingExtentIsNotPreferredForBeingSmall();
        testPrefixAndActiveReferencesHoldTheExtent();
        std::cout << "elastic KV pool tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "elastic KV pool test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
