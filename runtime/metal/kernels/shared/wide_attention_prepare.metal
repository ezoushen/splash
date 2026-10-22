#include "metal/kernels/common/gguf_sgmatrix.h"
#include "metal/kernels/common/attention_gate.h"
#include "metal/kernels/common/attention_qkv_prepare.h"

// The packed projection and physical K/V staging stay unchanged. Only query
// destinations use four groups of eight, directly in the attention layout.
template <class W>
inline void vgroups_qkv(
    device const bfloat *qkv, device const W *q_norm, device const W *k_norm,
    device const float *rope_cos, device const float *rope_sin,
    device bfloat *queries, device bfloat *keys, device bfloat *values,
    FullPrefillParams params, threadgroup float *reductions,
    threadgroup bfloat *normalized, uint task, uint batch, uint thread_index,
    uint lane, uint simd_group) {
  const uint query_tasks = params.tokens * 32;
  const bool query = task < query_tasks;
  const uint local_task = query ? task : task - query_tasks;
  const uint row = local_task / (query ? 32 : 2);
  const uint head = local_task % (query ? 32 : 2);
  const ulong packed_offset = (ulong(batch) * params.tokens + row) * 68 * 256;
  const ulong rope_offset = (ulong(batch) * params.tokens + row) * SPLASH_TARGET_ROPE_PAIRS;
  const ulong query_offset = ulong(batch) * 32 * params.stride * 256;
  const ulong kv_offset = ulong(batch) * 2 * params.stride * 256;
  // Every call sees row zero of a one-row view. The unchanged g8 helper
  // therefore never advances by its 16-head packed stride. Half 0/1 starts
  // at query group 0/2; K/V still use the physical 32-query, 2-KV-head view.
  const FullPrefillParams one_row{1, params.stride};
  if (query) {
    const uint query_half = head / 16;
    full_qkv_storage_phase<16, 2>(
        qkv + packed_offset + query_half * 16 * 512, q_norm, k_norm,
        rope_cos + rope_offset, rope_sin + rope_offset,
        queries + query_offset + (ulong(query_half) * 2 * params.stride + row) * 8 * 256,
        keys, values, one_row, reductions, normalized, head % 16, thread_index,
        lane, simd_group);
  } else {
    full_qkv_storage_phase<32, 2>(
        qkv + packed_offset, q_norm, k_norm, rope_cos + rope_offset,
        rope_sin + rope_offset, queries, keys + kv_offset + ulong(row) * 256,
        values + kv_offset + row, one_row, reductions, normalized, 32 + head,
        thread_index, lane, simd_group);
  }
}

#define PREFILL_VGROUPS_QKV(Name, W)                                          \
  kernel void Name(                                                          \
      device const bfloat *qkv [[buffer(0)]], device const W *q_norm [[buffer(1)]], \
      device const W *k_norm [[buffer(2)]],                                   \
      device const float *rope_cos [[buffer(3)]],                            \
      device const float *rope_sin [[buffer(4)]],                            \
      device bfloat *queries [[buffer(5)]], device bfloat *keys [[buffer(6)]], \
      device bfloat *values [[buffer(7)]],                                   \
      constant FullPrefillParams &params [[buffer(8)]],                      \
      uint task [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]],                     \
      uint lane [[thread_index_in_simdgroup]],                               \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                  \
    threadgroup float reductions[8];                                        \
    threadgroup bfloat normalized[256];                                     \
    vgroups_qkv(qkv, q_norm, k_norm, rope_cos, rope_sin, queries, keys, values, \
                params, reductions, normalized, task, 0, thread_index, lane, simd_group); \
  }
PREFILL_VGROUPS_QKV(prefill_attention_qkv_kv2_vgroups, bfloat)
PREFILL_VGROUPS_QKV(prefill_attention_qkv_kv2_vgroups_f32, float)
#undef PREFILL_VGROUPS_QKV

#define VERIFY_VGROUPS_QKV(Name, W)                                           \
  kernel void Name(                                                          \
      device const bfloat *qkv [[buffer(0)]], device const W *q_norm [[buffer(1)]], \
      device const W *k_norm [[buffer(2)]],                                   \
      device const float *rope_cos [[buffer(3)]],                            \
      device const float *rope_sin [[buffer(4)]],                            \
      device bfloat *queries [[buffer(5)]], device bfloat *keys [[buffer(6)]], \
      device bfloat *values [[buffer(7)]],                                   \
      uint2 group [[threadgroup_position_in_grid]],                          \
      uint thread_index [[thread_index_in_threadgroup]],                     \
      uint lane [[thread_index_in_simdgroup]],                               \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                  \
    threadgroup float reductions[8];                                        \
    threadgroup bfloat normalized[256];                                     \
    vgroups_qkv(qkv, q_norm, k_norm, rope_cos, rope_sin, queries, keys, values, \
                {SPLASH_TARGET_VERIFY_ROWS, SPLASH_VERIFY_CHUNK_STRIDE}, reductions, \
                normalized, group.x, group.y, thread_index, lane, simd_group); \
  }
VERIFY_VGROUPS_QKV(verify_attention_qkv_kv2_vgroups, bfloat)
VERIFY_VGROUPS_QKV(verify_attention_qkv_kv2_vgroups_f32, float)
#undef VERIFY_VGROUPS_QKV

kernel void prefill_attention_gate_kv2_vgroups(
    device const bfloat *packed [[buffer(0)]], device const bfloat *attention [[buffer(1)]],
    device bfloat *hidden [[buffer(2)]], constant FullPrefillParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]], uint grid_size [[threads_per_grid]]) {
  for (uint element = index; element < params.tokens * 32 * 256; element += grid_size)
    hidden[element] = full_attention_gate_value<32, 2, 8>(
        packed, attention, 0, 0, params.stride, element);
}

inline bfloat vgroups_verify_gate_value(device const bfloat *packed,
                                         device const bfloat *attention, uint element) {
  constexpr uint per_lane = SPLASH_TARGET_VERIFY_ROWS * 32 * 256;
  return full_attention_gate_value<32, 2, 8>(packed, attention, element / per_lane,
      SPLASH_TARGET_VERIFY_ROWS, SPLASH_VERIFY_CHUNK_STRIDE, element % per_lane);
}

kernel void verify_attention_gate_kv2_vgroups(
    device const bfloat *packed [[buffer(0)]], device const bfloat *attention [[buffer(1)]],
    device bfloat *hidden [[buffer(2)]], constant FullDecodeBatchParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]], uint grid_size [[threads_per_grid]]) {
  const uint count = params.lanes * SPLASH_TARGET_VERIFY_ROWS * 32 * 256;
  for (uint element = index; element < count; element += grid_size)
    hidden[element] = vgroups_verify_gate_value(packed, attention, element);
}

kernel void verify_attention_gate_table16_kv2_vgroups(
    device const bfloat *packed [[buffer(0)]], device const bfloat *attention [[buffer(1)]],
    device bfloat *hidden [[buffer(2)]], device bfloat *table [[buffer(3)]],
    device float *sums [[buffer(4)]], uint index [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint width = 32 * 256;
  const uint element = 2 * index;
  const bfloat a = vgroups_verify_gate_value(packed, attention, element);
  const bfloat b = vgroups_verify_gate_value(packed, attention, element + 1);
  hidden[element] = a;
  hidden[element + 1] = b;
  const uint row = element / width;
  gguf_sg::Table16::write(table + ulong(row / 8) * width * 8,
      sums + ulong(row / 8) * gguf_sg::Table16::sums_per_tile(width), width,
      (element % width) / 64, row % 8, lane, a, b);
}
