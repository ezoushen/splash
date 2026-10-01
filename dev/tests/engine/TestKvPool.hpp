#pragma once

#include "engine/KvPool.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace splash::test {

// Extents of extentPages pages, all allocated at first. Every page holds one
// value, which a test sets and a copy carries, so a test can follow the
// content of pages the pool moves.
class TestKvStorage final : public kv::ExtentStorage {
public:
    TestKvStorage(uint32_t pages, uint64_t bytesPerPage, uint32_t extentPages = 1)
        : content(pages), pageCount_(pages), bytesPerPage_(bytesPerPage),
          extentPages_(extentPages) {
        if (!pages || !bytesPerPage || !extentPages || pages % extentPages) {
            throw std::invalid_argument("invalid test KV extent storage");
        }
        allocated_.assign(pages / extentPages, true);
    }

    uint32_t pageCount() const noexcept override { return pageCount_; }
    uint64_t bytesPerPage() const noexcept override { return bytesPerPage_; }
    bool isAllocated(uint32_t page) const override {
        return allocated_.at(extent(page));
    }
    splash::metal::AllocationResult ensureAllocated(uint32_t page) override {
        allocated_.at(extent(page)) = true;
        return true;
    }
    bool releaseExtentOf(uint32_t page) override {
        bool wasAllocated = allocated_.at(extent(page));
        allocated_[extent(page)] = false;
        return wasAllocated;
    }
    void copyPages(std::span<const kv::PageCopy> pages) override {
        for (const kv::PageCopy &copy : pages) {
            content.at(copy.to) = content.at(copy.from);
            copies.push_back(copy);
        }
    }
    uint32_t extentFirstPage(uint32_t page) const override {
        return extent(page) * extentPages_;
    }
    uint32_t extentPageCount(uint32_t page) const override {
        static_cast<void>(extent(page));
        return extentPages_;
    }

    std::vector<uint64_t> content;
    std::vector<kv::PageCopy> copies;

private:
    uint32_t extent(uint32_t page) const {
        if (page >= pageCount_) throw std::out_of_range("invalid test page");
        return page / extentPages_;
    }

    uint32_t pageCount_ = 0;
    uint64_t bytesPerPage_ = 0;
    uint32_t extentPages_ = 0;
    std::vector<bool> allocated_;
};

}  // namespace splash::test
