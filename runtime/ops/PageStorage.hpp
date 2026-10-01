#pragma once

#include "metal/MetalBackend.hpp"
#include "ops/PagedKv.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace splash::kv {

// Physical storage for paged KV in extents: ordinary shared Metal buffers of
// extentPages pages each, allocated when the pool needs one of their pages and
// released when the last of them is free. Inside an extent every attention
// layer has a region that holds the keys of all its pages, then their key
// scales, values and value scales (abi/KvExtent.h). Kernels reach a page
// through its entry in a request's GPU page table, and the backend's
// residency set keeps every extent resident for every command. Active
// requests may overwrite slots at or beyond their logical commit index.
// Cached KV blocks reference only fully committed pages, which are immutable
// while shared.
class PageStorage final : public Backing {
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
  [[nodiscard]] uint64_t actualAllocatedBytes() const noexcept;
  [[nodiscard]] uint32_t residentPages() const noexcept;
  [[nodiscard]] bool isResident(uint32_t page) const override;
  [[nodiscard]] metal::AllocationResult ensureResident(uint32_t page) override;
  // The caller must prove that no active, prefix, reserved, or in-flight
  // reference remains anywhere in this extent. A command reaches extents
  // through its tables without retaining them, so releasing one while a
  // command is in flight throws std::logic_error and leaves it intact.
  [[nodiscard]] bool releaseBackingForPage(uint32_t page) override;
  [[nodiscard]] uint32_t extentFirstPage(uint32_t page) const override;
  [[nodiscard]] uint32_t extentPageCount(uint32_t page) const override;
  [[nodiscard]] LayerStorage layer(uint32_t index) const;

  // The entry kernels reach a page by. Throws std::logic_error for a page
  // whose extent has no backing: a GPU table holds only backed pages.
  [[nodiscard]] SplashKvPage entry(uint32_t page) const;
  // Writes the entries of `pages` to the start of a CPU-visible GPU page
  // table, which must hold all of them; throws std::logic_error otherwise.
  void writeEntries(std::span<const uint32_t> pages,
                    const metal::MetalBuffer &table) const;
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
  // Empty while the extent has no backing.
  std::vector<metal::MetalBuffer> extents_;
  uint64_t residentBackingBytes_ = 0;
  uint32_t residentPages_ = 0;
  uint64_t generation_ = 0;
};

} // namespace splash::kv
