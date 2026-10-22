#include "metal/kernels/common/lane_bindings.h"
#include "metal/kernels/common/paged_attention_tile.h"

// Four query groups per lane share its two physical KV heads. The page loop
// and split order are the production g8 tile's; only group/KV addressing differs.
#define VGROUPS_VERIFY_SPLIT(Name, CacheElement)                               \
  kernel void Name(                                                          \
      device bfloat *queries [[buffer(0)]], device float *partials [[buffer(1)]], \
      device float *statistics [[buffer(2)]],                                 \
      device const SplashKvPage *table0 [[buffer(3)]],                         \
      device const SplashKvPage *table1 [[buffer(4)]],                         \
      device const SplashKvPage *table2 [[buffer(5)]],                         \
      device const SplashKvPage *table3 [[buffer(6)]],                         \
      constant SplashVerifyAttentionParams *params [[buffer(7)]],              \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    constexpr uint M = 8 * SPLASH_TARGET_VERIFY_ROWS;                          \
    constexpr uint N = SplashKvPageTokens;                                    \
    alignas(16) threadgroup float scores[M * N];                              \
    alignas(16) threadgroup bfloat probabilities[M * N];                       \
    threadgroup float row_max[M], row_sum[M], previous_scale[M];               \
    threadgroup atomic_uint rescale;                                          \
    const SplashAttentionTile tile = splash_verify_attention_tile<4, 8>(       \
        params[group.z], group.x, group.z);                                   \
    if (!tile.active || group.y >= tile.splits) return;                        \
    splash_paged_attention_tile<2, 8, SPLASH_TARGET_VERIFY_ROWS, CacheElement>( \
        queries + tile.offset, SPLASH_LANE_BINDING(group.z, table0, table1, table2, table3), \
        params[group.z].kv, group.x / 2, tile.first_position, tile.active_rows, \
        tile.splits, group.y, partials, statistics, tile.first_slot + group.y, \
        scores, probabilities, row_max, row_sum, previous_scale, &rescale,      \
        thread_index);                                                       \
  }
VGROUPS_VERIFY_SPLIT(verify_attention_q8_split_kv2_vgroups, int8_t)
VGROUPS_VERIFY_SPLIT(verify_attention_bf16_split_kv2_vgroups, bfloat)
#undef VGROUPS_VERIFY_SPLIT

kernel void verify_attention_reduce_kv2_vgroups(
    device const float *partials [[buffer(0)]], device const float *statistics [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant SplashVerifyAttentionParams *params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]]) {
  const SplashAttentionTile tile = splash_verify_attention_tile<4, 8>(params[group.z], group.x, group.z);
  if (!tile.active || group.y >= 64 || thread_index >= SplashKvHeadDimension) return;
  threadgroup float weights[SplashVerifyMaximumSplits], group_values[8];
  splash_attention_reduce_row<8, SPLASH_TARGET_VERIFY_ROWS>(
      partials, statistics, output + tile.offset, tile.first_position, tile.active_rows,
      tile.splits, tile.first_slot, group.y, thread_index, weights, group_values);
}
