#include "metal/kernels/common/paged_attention_tile.h"

// Prototype variant 3 converts legacy [2 KV heads][row][16 query heads][D]
// staging to [4 query groups][row][8 query heads][D], and back. Production
// preparation and gates address the four-group layout directly.
kernel void prefill_attention_vgroups_pack(
    device const bfloat *input [[buffer(0)]], device bfloat *output [[buffer(1)]],
    constant SplashPrefillAttentionParams &params [[buffer(2)]],
    uint3 group [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) {
  const uint g = group.x, row = group.y, head = group.z;
  const ulong source = ((ulong(g / 2) * params.chunk_stride + row) * 16 +
                         g % 2 * 8 + head) * SplashKvHeadDimension + d;
  const ulong target = ((ulong(g) * params.chunk_stride + row) * 8 + head) *
                          SplashKvHeadDimension + d;
  output[target] = row < params.rows ? input[source] : bfloat(0.0f);
}

kernel void prefill_attention_vgroups_unpack(
    device const bfloat *input [[buffer(0)]], device bfloat *output [[buffer(1)]],
    constant SplashPrefillAttentionParams &params [[buffer(2)]],
    uint3 group [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) {
  const uint g = group.x, row = group.y, head = group.z;
  const ulong source = ((ulong(g) * params.chunk_stride + row) * 8 + head) *
                          SplashKvHeadDimension + d;
  const ulong target = ((ulong(g / 2) * params.chunk_stride + row) * 16 +
                         g % 2 * 8 + head) * SplashKvHeadDimension + d;
  output[target] = row < params.rows ? input[source] : bfloat(0.0f);
}

template <typename CacheElement>
inline void splash_vgroups_split(
    device bfloat *queries, device float *partials, device float *statistics,
    device const SplashKvPage *page_table, constant SplashPrefillAttentionParams &params,
    threadgroup float *scores, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale,
    uint3 group, uint thread_index) {
  // Four query groups own slots; two physical KV heads own the cache bytes.
  const SplashAttentionTile tile = splash_prefill_attention_tile<4, 8>(params, group.x, group.y);
  if (!tile.active || group.z >= tile.splits) return;
  splash_paged_attention_tile<2, 8, SPLASH_PREFILL_ATTENTION_TILE_ROWS, CacheElement>(
      queries + tile.offset, page_table, params.kv, group.x / 2, tile.first_position,
      tile.active_rows, tile.splits, group.z, partials, statistics, tile.first_slot + group.z,
      scores, probabilities, row_max, row_sum, previous_scale, rescale, thread_index);
}

#define VGROUPS_SPLIT(Name, CacheElement)                                      \
  kernel void Name(                                                          \
      device bfloat *queries [[buffer(0)]], device float *partials [[buffer(1)]], \
      device float *statistics [[buffer(2)]],                                 \
      device const SplashKvPage *page_table [[buffer(3)]],                     \
      constant SplashPrefillAttentionParams &params [[buffer(4)]],             \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    constexpr uint M = 8 * SPLASH_PREFILL_ATTENTION_TILE_ROWS;                 \
    constexpr uint N = SplashKvPageTokens;                                    \
    alignas(16) threadgroup float scores[M * N];                              \
    alignas(16) threadgroup bfloat probabilities[M * N];                       \
    threadgroup float row_max[M], row_sum[M], previous_scale[M];               \
    threadgroup atomic_uint rescale;                                          \
    splash_vgroups_split<CacheElement>(queries, partials, statistics, page_table, \
        params, scores, probabilities, row_max, row_sum, previous_scale,       \
        &rescale, group, thread_index);                                       \
  }
VGROUPS_SPLIT(prefill_attention_q8_split_kv2_vgroups, int8_t)
VGROUPS_SPLIT(prefill_attention_bf16_split_kv2_vgroups, bfloat)
#undef VGROUPS_SPLIT

kernel void prefill_attention_reduce_kv2_vgroups(
    device const float *partials [[buffer(0)]], device const float *statistics [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant SplashPrefillAttentionParams &params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]]) {
  const SplashAttentionTile tile = splash_prefill_attention_tile<4, 8>(params, group.x, group.z);
  if (!tile.active || group.y >= 64 || thread_index >= SplashKvHeadDimension) return;
  threadgroup float weights[SplashPrefillMaximumSplits], group_values[8];
  splash_attention_reduce_row<8, SPLASH_PREFILL_ATTENTION_TILE_ROWS>(
      partials, statistics, output + tile.offset, tile.first_position, tile.active_rows,
      tile.splits, tile.first_slot, group.y, thread_index, weights, group_values);
}
