#pragma once

#include "engine/KvPool.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace splash::test {

class TestKvStorage final : public kv::ExtentStorage {
public:
    TestKvStorage(uint32_t pages, uint64_t bytesPerPage)
        : pageCount_(pages), bytesPerPage_(bytesPerPage), allocated_(pages, true) {
        if (!pages || !bytesPerPage) {
            throw std::invalid_argument("invalid test KV extent storage");
        }
    }

    uint32_t pageCount() const noexcept override { return pageCount_; }
    uint64_t bytesPerPage() const noexcept override { return bytesPerPage_; }
    bool isAllocated(uint32_t page) const override {
        return allocated_.at(page);
    }
    splash::metal::AllocationResult ensureAllocated(uint32_t page) override {
        allocated_.at(page) = true;
        return true;
    }
    bool releaseExtentOf(uint32_t page) override {
        bool wasAllocated = allocated_.at(page);
        allocated_[page] = false;
        return wasAllocated;
    }
    uint32_t extentFirstPage(uint32_t page) const override {
        if (page >= pageCount_) throw std::out_of_range("invalid test page");
        return page;
    }
    uint32_t extentPageCount(uint32_t page) const override {
        if (page >= pageCount_) throw std::out_of_range("invalid test page");
        return 1;
    }
private:
    uint32_t pageCount_ = 0;
    uint64_t bytesPerPage_ = 0;
    std::vector<bool> allocated_;
};

}  // namespace splash::test
