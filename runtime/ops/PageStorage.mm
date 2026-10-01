#include "ops/PageStorage.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::kv {

PageStorage::PageStorage(metal::MetalBackend &backend,
    metal::AllocationAdmission admitAllocation, Layout layout,
    uint32_t pageCount, uint32_t extentPages)
    : backend_(backend), admitAllocation_(std::move(admitAllocation)),
      layout_(layout), pageCount_(pageCount), extentPages_(extentPages) {
    if (!admitAllocation_) {
        throw std::invalid_argument(
            "KV page storage requires physical allocation admission");
    }
    if (!layout_.valid()) {
        throw std::invalid_argument("KV page storage layout is invalid");
    }
    // Page entries carry the index in an extent in their low bits; kernels
    // address an extent's regions with 32-bit offsets.
    if (!extentPages_ || extentPages_ % layout_.extentAlignmentPages() ||
        extentPages_ > SPLASH_KV_PAGE_INDEX_MASK ||
        extentBytes() > std::numeric_limits<uint32_t>::max() ||
        extentBytes() > backend_.capabilities().maxBufferLengthBytes) {
        throw std::invalid_argument("KV extent geometry is invalid");
    }
    if (!pageCount_ || pageCount_ % extentPages_) {
        throw std::invalid_argument(
            "KV page pool is not a whole number of extents");
    }
    extents_.resize(pageCount_ / extentPages_);
    // One small runway makes startup warmup and the first requests allocation
    // free. Every later extent is allocated when real tokens need it.
    if (auto result = ensureResident(0); !result) {
        throw metal::MetalAllocationError(
            std::string("unable to allocate initial KV backing extent: ") +
                metal::allocationFailureName(result.failure), result.failure);
    }
}

uint64_t PageStorage::declaredBytes() const noexcept {
    return uint64_t(pageCount_) * layout_.bytesPerModelPage();
}

uint64_t PageStorage::actualAllocatedBytes() const noexcept {
    return residentBackingBytes_;
}

uint32_t PageStorage::residentPages() const noexcept {
    return residentPages_;
}

size_t PageStorage::extentIndex(uint32_t page) const {
    if (page >= pageCount_) throw std::out_of_range("invalid KV page id");
    return page / extentPages_;
}

uint32_t PageStorage::extentFirstPage(uint32_t page) const {
    return static_cast<uint32_t>(extentIndex(page)) * extentPages_;
}

uint32_t PageStorage::extentPageCount(uint32_t page) const {
    static_cast<void>(extentIndex(page));
    return extentPages_;
}

bool PageStorage::isResident(uint32_t page) const {
    return static_cast<bool>(extents_[extentIndex(page)]);
}

metal::AllocationResult PageStorage::ensureResident(uint32_t page) {
    metal::MetalBuffer &extent = extents_[extentIndex(page)];
    if (extent) return true;
    const uint64_t bytes = extentBytes();
    try {
        return admitAllocation_(bytes, [&] {
            metal::MetalBuffer allocated = backend_.allocateAddressed(
                bytes, "kv-extent-" + std::to_string(extentFirstPage(page)));
            if (allocated.gpuAddress() & SPLASH_KV_PAGE_INDEX_MASK) {
                throw std::logic_error(
                    "KV extent address leaves no room for the page index");
            }
            extent = std::move(allocated);
            residentBackingBytes_ += bytes;
            residentPages_ += extentPages_;
            ++generation_;
        });
    } catch (const metal::MetalAllocationError &error) {
        return error.failure();
    }
}

bool PageStorage::releaseBackingForPage(uint32_t page) {
    metal::MetalBuffer &extent = extents_[extentIndex(page)];
    if (!extent) return false;
    if (backend_.commandInFlight()) {
        throw std::logic_error(
            "cannot release KV backing while a command is in flight");
    }
    if (residentBackingBytes_ < extentBytes() || residentPages_ < extentPages_) {
        throw std::logic_error("KV resident accounting underflowed");
    }
    extent = {};
    residentBackingBytes_ -= extentBytes();
    residentPages_ -= extentPages_;
    ++generation_;
    return true;
}

LayerStorage PageStorage::layer(uint32_t index) const {
    if (index >= layout_.attentionLayers) {
        throw std::out_of_range("invalid attention layer index");
    }
    const uint64_t offset = splash_kv_offset(
        extentPages_, static_cast<uint32_t>(layout_.dataBytesPerLayerPage()),
        static_cast<uint32_t>(layout_.scaleBytesPerLayerPage()), index,
        SPLASH_KV_KEYS, 0);
    return {{extentPages_, static_cast<uint32_t>(offset)}, layout_.format};
}

SplashKvPage PageStorage::entry(uint32_t page) const {
    const metal::MetalBuffer &extent = extents_[extentIndex(page)];
    if (!extent) {
        throw std::logic_error("KV page " + std::to_string(page) +
                               " has no backing");
    }
    return extent.gpuAddress() | (page % extentPages_);
}

void PageStorage::writeEntries(std::span<const uint32_t> pages,
                               const metal::MetalBuffer &table) const {
    auto *entries = static_cast<SplashKvPage *>(table.contents());
    if (!entries || table.sizeBytes() / sizeof(SplashKvPage) < pages.size()) {
        throw std::logic_error(
            "KV page table is not CPU-visible or too small for its entries");
    }
    for (size_t index = 0; index < pages.size(); ++index)
        entries[index] = entry(pages[index]);
}

}  // namespace splash::kv
