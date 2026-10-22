#define SPLASH_ATTENTION_TILE_SIMDGROUPS 16
#define SplashAttentionTileThreads SplashAttentionTileThreadsT512
#define splash_attention_softmax_passes splash_attention_softmax_passes_t512
#define splash_paged_attention_tile splash_paged_attention_tile_t512
#include "metal/kernels/common/paged_attention_tile.h"

// GQA-16 prefill split; dispatch with 512 threads per threadgroup.
#define PREFILL_T512(Name, CacheElement)                                      \
  kernel void Name(                                                          \
      device bfloat *queries [[buffer(0)]],                                  \
      device float *partials [[buffer(1)]],                                  \
      device float *statistics [[buffer(2)]],                                \
      device const SplashKvPage *page_table [[buffer(3)]],                   \
      constant SplashPrefillAttentionParams &params [[buffer(4)]],           \
      uint3 group [[threadgroup_position_in_grid]],                          \
      uint thread_index [[thread_index_in_threadgroup]]) {                   \
    constexpr uint M = 16 * SPLASH_PREFILL_ATTENTION_TILE_ROWS;              \
    constexpr uint N = SplashKvPageTokens;                                   \
    alignas(16) threadgroup float scores[M * N];                             \
    alignas(16) threadgroup bfloat probabilities[M * N];                     \
    threadgroup float row_max[M], row_sum[M], previous_scale[M];             \
    threadgroup atomic_uint rescale;                                         \
    const SplashAttentionTile tile =                                        \
        splash_prefill_attention_tile<2, 16>(params, group.x, group.y);       \
    if (!tile.active || group.z >= tile.splits) return;                      \
    splash_paged_attention_tile_t512<2, 16,                                  \
        SPLASH_PREFILL_ATTENTION_TILE_ROWS, CacheElement>(                   \
        queries + tile.offset, page_table, params.kv, group.x,                \
        tile.first_position, tile.active_rows, tile.splits, group.z,          \
        partials, statistics, tile.first_slot + group.z, scores,              \
        probabilities, row_max, row_sum, previous_scale, &rescale,            \
        thread_index);                                                       \
  }

PREFILL_T512(prefill_attention_q8_split_kv2_g16_t512, int8_t)
PREFILL_T512(prefill_attention_bf16_split_kv2_g16_t512, bfloat)
#undef PREFILL_T512
