#pragma once

#include "metal/MetalBackend.hpp"
#include "ops/PagedKv.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace splash::kv {

// Storage for paged KV in extents: ordinary shared Metal buffers of
// extentPages pages each, allocated when the pool needs one of their pages and
// released when the last of them is free. Inside an extent every attention
// layer has a region that holds the keys of all its pages, then their key
// scales, values and value scales (abi/KvExtent.h). Kernels reach a page
// through its entry in a request's GPU page table, and the backend's
// residency set keeps every extent resident for every command; the host
// reaches the same bytes through the page's spans(). Active requests may
// overwrite slots at or beyond their logical commit index. Cached KV blocks
// reference only fully committed pages, which are immutable while shared.
class PageStorage final : public ExtentStorage {
public:
  // pageCount must be a whole number of extents of extentPages pages, a
  // whole number of the layout's alignment units (Layout::extentPagesFor).
  PageStorage(metal::MetalBackend &backend,
              metal::AllocationAdmission admitAllocation,
              Layout layout,
              uint32_t pageCount,
              uint32_t extentPages);

  PageStorage(const PageStorage &) = delete;
  PageStorage &operator=(const PageStorage &) = delete;

  [[nodiscard]] uint32_t pageCount() const noexcept override {
    return pageCount_;
  }
  [[nodiscard]] uint64_t bytesPerPage() const noexcept override {
    return layout_.bytesPerModelPage();
  }
  [[nodiscard]] Layout layout() const noexcept { return layout_; }
  [[nodiscard]] uint32_t extentPages() const noexcept { return extentPages_; }
  [[nodiscard]] uint64_t extentBytes() const noexcept {
    return uint64_t{extentPages_} * layout_.bytesPerModelPage();
  }
  // The extents that hold memory now, and their bytes.
  [[nodiscard]] uint32_t allocatedExtents() const noexcept { return allocatedExtents_; }
  [[nodiscard]] uint64_t actualAllocatedBytes() const noexcept {
    return uint64_t{allocatedExtents_} * extentBytes();
  }
  [[nodiscard]] bool isAllocated(uint32_t page) const override;
  [[nodiscard]] metal::AllocationResult ensureAllocated(uint32_t page) override;
  // The caller must prove that no active, prefix, reserved, or in-flight
  // reference remains anywhere in this extent. A command reaches extents
  // through its tables without retaining them, so releasing one while a
  // command is in flight throws std::logic_error and leaves it intact.
  [[nodiscard]] bool releaseExtentOf(uint32_t page) override;
  // A command reaches both pages through its tables and may still write the
  // source, so, like a release, copying while one is in flight throws
  // std::logic_error.
  void copyPages(std::span<const PageCopy> copies) override;
  [[nodiscard]] uint32_t extentFirstPage(uint32_t page) const override;
  [[nodiscard]] uint32_t extentPageCount(uint32_t page) const override;
  [[nodiscard]] LayerStorage layer(uint32_t index) const;

  // The entry kernels reach a page by. Throws std::logic_error for a page
  // whose extent is not allocated: a GPU table holds only pages of allocated
  // extents.
  [[nodiscard]] SplashKvPage entry(uint32_t page) const;
  // Writes the entries of `pages` to the start of a CPU-visible GPU page
  // table, which must hold all of them; throws std::logic_error otherwise.
  void writeEntries(std::span<const uint32_t> pages,
                    const metal::MetalBuffer &table) const;
  // The page's memory as the host reaches it, and the only code that names
  // it: the page's bytes of each tensor in every layer's region, layer by
  // layer as keys, key scales, values and value scales, where
  // splash_kv_offset places them for the kernels. BF16 pages have no scale
  // bytes. Throws std::logic_error for a page whose extent is not allocated.
  [[nodiscard]] std::vector<std::span<std::byte>> spans(uint32_t page) const;
  // Advances whenever an extent is allocated or released. A GPU table
  // written at an earlier generation may hold an entry of a released extent.
  [[nodiscard]] uint64_t generation() const noexcept { return generation_; }

private:
  [[nodiscard]] size_t extentIndex(uint32_t page) const;

  metal::MetalBackend &backend_;
  metal::AllocationAdmission admitAllocation_;
  Layout layout_;
  uint32_t pageCount_ = 0;
  uint32_t extentPages_ = 0;
  // Empty while the extent is not allocated.
  std::vector<metal::MetalBuffer> extents_;
  uint32_t allocatedExtents_ = 0;
  uint64_t generation_ = 0;
};

} // namespace splash::kv
