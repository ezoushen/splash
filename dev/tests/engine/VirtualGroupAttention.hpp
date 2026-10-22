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

// Verify keeps each lane's production split count and the batch-wide slot
// stride. Packing is per lane; the attention grid owns four groups per lane.
inline void addVirtualGroupVerify(
    metal::MetalBackend &backend, metal::CommandGraph &graph, SplashKvLayer layer,
    ops::PagedVerifyBuffers buffers, std::span<const kv::ChunkedPrefillParams> chunks,
    const ops::VerifyAttentionPlan &plan) {
  const auto packed = backend.allocateBuffer(buffers.queries.sizeBytes(), metal::BufferStorage::Shared,
                                             "virtual-group-verify-queries");
  const auto result = backend.allocateBuffer(buffers.output.sizeBytes(), metal::BufferStorage::Shared,
                                             "virtual-group-verify-output");
  constexpr uint32_t maximumLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
  std::array<kv::VerifyAttentionParams, maximumLanes> attention{};
  std::array<kv::PrefillAttentionParams, maximumLanes> packing{};
  const uint64_t queryBytes = uint64_t{plan.queryHeads} * kv::kVerifyChunkStride *
                              plan.layout.headDimension * 2;
  const uint64_t chunkBytes = uint64_t{plan.layout.kvHeads} * kv::kVerifyChunkStride *
                              plan.layout.headDimension * 2;
  for (uint32_t lane = 0; lane < plan.lanes; ++lane) {
    const auto &chunk = chunks[lane];
    attention[lane] = {chunk.committed_tokens, chunk.page_table_entries, layer,
                        plan.laneSplits[lane], plan.splits, 0, 0};
    packing[lane] = {chunk.committed_tokens, kv::kVerifyRows, kv::kVerifyChunkStride,
                      chunk.page_table_entries, layer, 1, 0, 0};
    ops::PagedAttention::addPrefillStore(
        graph, layer, backend.view(buffers.chunkKeys, lane * chunkBytes, chunkBytes),
        backend.view(buffers.chunkValues, lane * chunkBytes, chunkBytes), buffers.pageTables[lane],
        chunk, plan.layout);
    graph.add("prefill_attention_vgroups_pack",
              {backend.view(buffers.queries, lane * queryBytes, queryBytes),
               backend.view(packed, lane * queryBytes, queryBytes)}, packing[lane],
              {4, kv::kVerifyChunkStride, 8});
  }
  const auto &tables = buffers.pageTables;
  graph.add(plan.layout.format == kv::Format::Int8
                ? "verify_attention_q8_split_kv2_vgroups"
                : "verify_attention_bf16_split_kv2_vgroups",
            {packed, buffers.partials, buffers.statistics, tables[0], tables[1], tables[2], tables[3]},
            attention, {4, plan.splits, plan.lanes});
  graph.add("verify_attention_reduce_kv2_vgroups", {buffers.partials, buffers.statistics, result},
            attention, {4, 64, plan.lanes});
  for (uint32_t lane = 0; lane < plan.lanes; ++lane)
    graph.add("prefill_attention_vgroups_unpack",
              {backend.view(result, lane * queryBytes, queryBytes),
               backend.view(buffers.output, lane * queryBytes, queryBytes)}, packing[lane],
              {4, kv::kVerifyChunkStride, 8});
}

} // namespace splash::test
