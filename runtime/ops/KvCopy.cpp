#include "ops/KvCopy.hpp"

#include "metal/abi/KvCopy.h"

#include <utility>

namespace splash::ops {

namespace {
constexpr uint32_t kCopyThreads = 256;

uint32_t direction(KvCopy::Direction value) noexcept {
  switch (value) {
  case KvCopy::Direction::ToStaging:
    return SPLASH_KV_COPY_TO_STAGING;
  case KvCopy::Direction::ToPage:
    return SPLASH_KV_COPY_TO_PAGE;
  case KvCopy::Direction::None:
    break;
  }
  return SPLASH_KV_COPY_NONE;
}
} // namespace

uint64_t KvCopy::tableBytes(uint32_t stagingSlots) noexcept {
  return uint64_t{stagingSlots} * sizeof(SplashKvCopySlot);
}

void KvCopy::setEntry(const metal::MetalBuffer &table, uint32_t slot, SplashKvPage page,
                      Direction value) noexcept {
  static_cast<SplashKvCopySlot *>(table.contents())[slot] = {page, direction(value), 0};
}

void KvCopy::addPages(metal::CommandGraph &graph, const kv::PageStorage &pages,
                      metal::MetalBuffer staging, metal::MetalBuffer table,
                      uint32_t stagingSlots, uint64_t slotBytes) {
  const kv::Layout layout = pages.layout();
  SplashKvCopyParams params{};
  params.data_bytes = static_cast<uint32_t>(layout.dataBytesPerLayerPage());
  params.scale_bytes = static_cast<uint32_t>(layout.scaleBytesPerLayerPage());
  params.slot_bytes = static_cast<uint32_t>(slotBytes);
  params.extent_pages = pages.extentPages();
  params.staging_slots = stagingSlots;
  for (uint32_t layer = 0; layer < layout.attentionLayers; ++layer) {
    params.layer_offset = static_cast<uint32_t>(layer * layout.bytesPerLayerPage());
    params.layer = layer;
    graph.add("kv_copy_pages", {staging, table}, params, {stagingSlots, 1, 1},
              {kCopyThreads, 1, 1});
  }
}

} // namespace splash::ops
