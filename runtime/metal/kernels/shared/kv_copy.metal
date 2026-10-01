#include "metal/abi/KvCopy.h"
#include "metal/kernels/common/kv_extent.h"
#include <metal_stdlib>

using namespace metal;

// Moves whole KV pages of one attention layer between their extents and
// host-visible staging, 16 bytes per thread per step. Idle slots and empty
// entries copy nothing.
kernel void kv_copy_pages(device uchar *staging [[buffer(0)]],
                          device const SplashKvCopySlot *table [[buffer(1)]],
                          constant SplashKvCopyParams &params [[buffer(2)]],
                          uint slot [[threadgroup_position_in_grid]],
                          uint lane [[thread_position_in_threadgroup]],
                          uint lanes [[threads_per_threadgroup]]) {
  if (slot >= params.staging_slots) return;
  const SplashKvCopySlot entry = table[slot];
  if (entry.direction == SPLASH_KV_COPY_NONE || !entry.page) return;
  const bool toStaging = entry.direction == SPLASH_KV_COPY_TO_STAGING;
  const uint index = splash_kv_page_index(entry.page);
  device uchar *extent = splash_kv_extent(entry.page, index);
  device uchar *stage =
      staging + ulong(slot) * params.slot_bytes + params.layer_offset;
  for (uint tensor = SPLASH_KV_KEYS; tensor <= SPLASH_KV_VALUE_SCALES; ++tensor) {
    const uint bytes =
        splash_kv_page_bytes(params.data_bytes, params.scale_bytes, tensor);
    if (!bytes) continue;
    device uint4 *pageWords = reinterpret_cast<device uint4 *>(
        extent + splash_kv_offset(params.extent_pages, params.data_bytes,
                                  params.scale_bytes, params.layer, tensor, index));
    device uint4 *stageWords = reinterpret_cast<device uint4 *>(stage);
    const uint words = bytes / 16;
    for (uint word = lane; word < words; word += lanes) {
      if (toStaging) {
        stageWords[word] = pageWords[word];
      } else {
        pageWords[word] = stageWords[word];
      }
    }
    stage += bytes;
  }
}
