#pragma once

#include "metal/CommandGraph.hpp"
#include "ops/PagedAttention.hpp"

namespace splash::test {

// Sweep-only plan: four groups of eight query heads share two real KV heads.
// The tile/split counts stay the production plan's, so 4 * 64 fused rows
// consume exactly the same partial/statistic bytes as 2 * 128 fused rows.
// Two BF16 layout buffers add 64 MiB at 2048 rows (32 * 2048 * 256 * 2 each).
inline void addVirtualGroupPrefill(
    metal::MetalBackend &backend, metal::CommandGraph &graph, SplashKvLayer layer,
    metal::MetalBuffer queries, metal::MetalBuffer output, metal::MetalBuffer partials,
    metal::MetalBuffer statistics, metal::MetalBuffer table,
    const kv::ChunkedPrefillParams &chunk, const ops::PrefillAttentionPlan &plan) {
  const auto packed = backend.allocateBuffer(queries.sizeBytes(), metal::BufferStorage::Shared,
                                             "virtual-group-queries");
  const auto result = backend.allocateBuffer(output.sizeBytes(), metal::BufferStorage::Shared,
                                             "virtual-group-output");
  const kv::PrefillAttentionParams params{
      chunk.committed_tokens, chunk.chunk_tokens, chunk.chunk_stride,
      chunk.page_table_entries, layer, plan.splits, 0, 0};
  graph.add("prefill_attention_vgroups_pack", {queries, packed}, params,
            {4, chunk.chunk_stride, 8});
  graph.add(plan.layout.format == kv::Format::Int8
                ? "prefill_attention_q8_split_kv2_vgroups"
                : "prefill_attention_bf16_split_kv2_vgroups",
            {packed, partials, statistics, table}, params, {4, plan.tiles, plan.splits});
  graph.add("prefill_attention_reduce_kv2_vgroups", {partials, statistics, result},
            params, {4, 64, plan.tiles});
  graph.add("prefill_attention_vgroups_unpack", {result, output}, params,
            {4, chunk.chunk_stride, 8});
}

} // namespace splash::test
