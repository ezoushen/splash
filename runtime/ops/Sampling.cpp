#include "ops/Sampling.hpp"

#include "metal/abi/Sampling.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace splash::ops {
namespace {

constexpr uint32_t kMaximumLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kTargetShards = SPLASH_TARGET_SAMPLING_SHARDS;
constexpr uint32_t kVocabularyThreads = SPLASH_TARGET_VOCABULARY_THREADS;
constexpr uint32_t kVocabularyGroups = SPLASH_TARGET_VOCABULARY_GROUPS;
// A sampled row's draw sums one range per simdgroup of the row's groups.
constexpr uint32_t kVocabularyRanges =
    kVocabularyGroups * kVocabularyThreads / 32;

// A sampled lane keeps its topK most likely tokens, and every token for 0 or
// a topK past the vocabulary (top-k disabled).
uint32_t effectiveTopK(const SamplingPolicy &policy,
                       uint32_t vocabulary) noexcept {
  return policy.topK && policy.topK < vocabulary ? policy.topK : vocabulary;
}
constexpr uint32_t kDraftShards = SPLASH_DRAFT_SAMPLING_SHARDS;
constexpr uint32_t kDraftCandidates = 16;
// Each position's group scores its 16 x 16 edge table eight edges per
// simdgroup task; eight simdgroups balance the seven-group B1 dispatch
// against the 28 groups of B4 (wider groups speed up B1 and slow down B4).
constexpr uint32_t kEdgeThreads = 256;
constexpr uint32_t kPenaltyThreads = 256;

void requireVocabulary(std::span<const uint32_t> tokens, size_t vocabulary) {
  if (std::any_of(tokens.begin(), tokens.end(),
                  [&](uint32_t token) { return token >= vocabulary; }))
    throw std::invalid_argument("penalty token is outside the vocabulary");
}

} // namespace

SamplingWorkspace Sampling::workspace(uint32_t rows) {
  if (!rows)
    throw std::invalid_argument("invalid sampling workspace row count");
  const uint64_t shards = uint64_t{rows} * kTargetShards;
  return {shards * sizeof(float),
          shards * sizeof(uint32_t),
          shards * sizeof(TargetShardMass),
          uint64_t{rows} * sizeof(TargetVocabularyRow),
          uint64_t{rows} * kVocabularyRanges * sizeof(TargetVocabularyRange),
          uint64_t{rows} * sizeof(uint32_t)};
}

DraftSelectorWorkspace Sampling::draftWorkspace(uint32_t positions) {
  if (!positions)
    throw std::invalid_argument("invalid draft selector workspace position count");
  const uint64_t candidates = uint64_t{positions} * kDraftCandidates;
  // The partial values are followed by each position's 16 x 16 edge table.
  return {candidates * kDraftShards * sizeof(uint32_t),
          candidates * (kDraftShards + kDraftCandidates) * sizeof(float),
          candidates * sizeof(uint32_t), candidates * sizeof(float),
          candidates * sizeof(float)};
}

void Sampling::loadPenaltyWords(std::span<uint32_t> words,
                                std::span<const uint32_t> prompt,
                                std::span<const uint32_t> selected,
                                bool markPrompt) {
  requireVocabulary(prompt, words.size());
  requireVocabulary(selected, words.size());
  std::fill(words.begin(), words.end(), 0U);
  if (markPrompt) {
    for (const uint32_t token : prompt)
      words[token] |= SPLASH_PENALTY_PROMPT_BIT;
  }
  countPenaltyTokens(words, selected);
}

// Counts stay far below the prompt bit: a request selects at most one token
// per position of its context.
void Sampling::countPenaltyTokens(std::span<uint32_t> words,
                                  std::span<const uint32_t> selected) {
  requireVocabulary(selected, words.size());
  for (const uint32_t token : selected)
    ++words[token];
}

Sampling::Sampling(uint32_t vocabulary, uint32_t rowsPerLane)
    : vocabulary_(vocabulary), rowsPerLane_(rowsPerLane),
      maskWords_((vocabulary + 31) / 32) {
  if (!vocabulary || rowsPerLane != SPLASH_TARGET_VERIFY_ROWS)
    throw std::invalid_argument("invalid sampling geometry");
}

void Sampling::addPenalties(metal::CommandGraph &graph,
                            std::span<const SamplingPolicy> policies,
                            const SamplingBuffers &buffers,
                            const PenaltyTable &table, uint32_t rowOffset,
                            bool verify) const {
  SamplingPenaltyParams params{};
  params.vocabulary = vocabulary_;
  params.rows = verify ? rowsPerLane_ : 1;
  params.row_stride = rowsPerLane_;
  params.row_offset = rowOffset;
  const uint64_t rowBytes = uint64_t{vocabulary_} * sizeof(uint32_t);
  for (uint32_t lane = 0; lane < policies.size(); ++lane) {
    const SamplingPenalties &penalties = policies[lane].penalties;
    if (!penalties.active())
      continue;
    if (!std::isfinite(penalties.repetition) || penalties.repetition <= 0.0F ||
        !std::isfinite(penalties.presence) ||
        !std::isfinite(penalties.frequency))
      throw std::invalid_argument("invalid sampling penalties");
    // The kernel indexes the whole table by this row.
    if (lane >= table.rows.size() ||
        (uint64_t{table.rows[lane]} + 1) * rowBytes > table.words.sizeBytes())
      throw std::invalid_argument("penalized lane has no penalty table row");
    const uint32_t entry = params.entries++;
    params.logits_lane[entry] = lane;
    params.table_row[entry] = table.rows[lane];
    params.repetition[entry] = penalties.repetition;
    // 1 / 2^-149 overflows; the saturated inverse keeps the product finite.
    params.repetition_inverse[entry] = std::min(
        1.0F / penalties.repetition, std::numeric_limits<float>::max());
    params.presence[entry] = penalties.presence;
    params.frequency[entry] = penalties.frequency;
  }
  if (!params.entries)
    return;
  const metal::DispatchSize groups{
      (vocabulary_ + kPenaltyThreads - 1) / kPenaltyThreads, params.entries, 1};
  if (!verify) {
    graph.add("decode_sample_penalize", {buffers.logits, table.words}, params,
              groups, {kPenaltyThreads, 1, 1});
    return;
  }
  if (!buffers.inputTokens)
    throw std::invalid_argument("penalized verify rows need their input tokens");
  graph.add("decode_sample_penalize_verify",
            {buffers.logits, table.words, buffers.inputTokens}, params, groups,
            {kPenaltyThreads, 1, 1});
}

void Sampling::addInitial(metal::CommandGraph &graph,
                          const SamplingPolicy &policy,
                          SamplingBuffers buffers, uint32_t rowOffset,
                          uint32_t stopToken0, uint32_t stopToken1,
                          PenaltyTable penalties) const {
  if (rowOffset >= rowsPerLane_)
    throw std::invalid_argument("invalid initial sampling row");
  if (policy.samples() &&
      (!buffers.vocabularyRanges || !buffers.vocabularyArrivals))
    throw std::invalid_argument(
        "a sampled row needs its draw ranges and arrival counts");
  addPenalties(graph, {&policy, 1}, buffers, penalties, rowOffset, false);
  const TargetSamplingParams params{vocabulary_,
                                    rowOffset,
                                    effectiveTopK(policy, vocabulary_),
                                    policy.temperature,
                                    policy.topP,
                                    policy.minP,
                                    maskWords_,
                                    0,
                                    policy.constrained ? 1U : 0U,
                                    policy.excludesStopTokens ? 1U : 0U,
                                    stopToken0,
                                    stopToken1};
  if (!policy.samples()) {
    graph.add("decode_sample_argmax_sharded",
              {buffers.logits, buffers.constraintMasks, buffers.argmaxValues,
               buffers.argmaxIndices},
              params, {kTargetShards, 1, 1});
    graph.add("decode_sample_argmax_reduce",
              {buffers.argmaxValues, buffers.argmaxIndices,
               buffers.outputTokens},
              {1, 1, 1}, {32, 1, 1});
    return;
  }
  graph.add("decode_sample_mass_sharded",
            {buffers.logits, buffers.constraintMasks, buffers.partialMasses},
            params, {kTargetShards, 1, 1});
  graph.add("decode_sample_vocabulary_search",
            {buffers.logits, buffers.constraintMasks, buffers.partialMasses,
             buffers.vocabularyRows},
            params, {1, 1, 1}, {kVocabularyThreads, 1, 1});
  graph.add("decode_sample_vocabulary_draw",
            {buffers.logits, buffers.constraintMasks, buffers.vocabularyRows,
             buffers.uniforms, buffers.outputTokens, buffers.vocabularyRanges,
             buffers.vocabularyArrivals},
            params, {kVocabularyGroups, 1, 1}, {kVocabularyThreads, 1, 1});
}

void Sampling::addVerify(metal::CommandGraph &graph,
                         std::span<const SamplingPolicy> policies,
                         SamplingBuffers buffers, uint32_t stopToken0,
                         uint32_t stopToken1, PenaltyTable penalties) const {
  if (policies.empty() || policies.size() > kMaximumLanes)
    throw std::invalid_argument("invalid sampling batch width");
  const bool sampling = std::any_of(
      policies.begin(), policies.end(),
      [](const SamplingPolicy &policy) { return policy.samples(); });
  const bool greedy = std::any_of(
      policies.begin(), policies.end(),
      [](const SamplingPolicy &policy) { return !policy.samples(); });
  if (sampling && (!buffers.inputTokens || !buffers.draftCandidates ||
                   !buffers.draftProbabilities || !buffers.vocabularyRanges ||
                   !buffers.vocabularyArrivals))
    throw std::invalid_argument(
        "sampled verify rows need their draft tokens and candidates and "
        "their draw ranges and arrival counts");
  addPenalties(graph, policies, buffers, penalties, 0, true);
  const uint32_t lanes = static_cast<uint32_t>(policies.size());
  const uint32_t rows = lanes * rowsPerLane_;
  TargetSamplingBatchParams params{};
  params.vocabulary = vocabulary_;
  params.rows_per_lane = rowsPerLane_;
  params.lanes = lanes;
  params.mask_words = maskWords_;
  params.stop_token_0 = stopToken0;
  params.stop_token_1 = stopToken1;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const SamplingPolicy &policy = policies[lane];
    if (policy.samples()) {
      params.top_k[lane] = effectiveTopK(policy, vocabulary_);
      params.temperature[lane] = policy.temperature;
      params.top_p[lane] = policy.topP;
      params.min_p[lane] = policy.minP;
      params.sampling_mask |= uint32_t{1} << lane;
    }
    if (policy.constrained)
      params.constrained_mask |= uint32_t{1} << lane;
    if (policy.excludesStopTokens)
      params.exclude_stop_mask |= uint32_t{1} << lane;
  }
  // Greedy and sampled lanes run their own kernels, each over the rows of
  // the batch; the groups of the other kind's lanes return at once.
  if (greedy) {
    graph.add("decode_sample_argmax_sharded_batch",
              {buffers.logits, buffers.constraintMasks, buffers.argmaxValues,
               buffers.argmaxIndices},
              params, {uint64_t{rows} * kTargetShards, 1, 1});
    graph.add("decode_sample_argmax_reduce_batch",
              {buffers.argmaxValues, buffers.argmaxIndices,
               buffers.outputTokens},
              params, {rows, 1, 1}, {32, 1, 1});
  }
  if (sampling) {
    graph.add("decode_sample_mass_sharded_batch",
              {buffers.logits, buffers.constraintMasks, buffers.partialMasses},
              params, {uint64_t{rows} * kTargetShards, 1, 1});
    graph.add("decode_sample_vocabulary_search_batch",
              {buffers.logits, buffers.constraintMasks, buffers.partialMasses,
               buffers.vocabularyRows},
              params, {rows, 1, 1}, {kVocabularyThreads, 1, 1});
    graph.add("decode_sample_vocabulary_draw_batch",
              {buffers.logits, buffers.constraintMasks, buffers.vocabularyRows,
               buffers.inputTokens, buffers.draftCandidates,
               buffers.draftProbabilities, buffers.uniforms,
               buffers.vocabularyRanges, buffers.vocabularyArrivals},
              params, {uint64_t{rows} * kVocabularyGroups, 1, 1},
              {kVocabularyThreads, 1, 1});
  }
}

void Sampling::addDraftSelector(
    metal::CommandGraph &graph, DraftSelectorBuffers buffers,
    std::span<const uint32_t> anchors,
    std::span<const SamplingPolicy> policies, uint32_t proposalTokens) const {
  if (anchors.empty() || anchors.size() != policies.size() ||
      anchors.size() > kMaximumLanes ||
      proposalTokens != SPLASH_DRAFT_PROPOSAL_TOKENS)
    throw std::invalid_argument("invalid draft selector batch");
  const uint32_t lanes = static_cast<uint32_t>(anchors.size());
  SelectorBatchParams params{};
  params.lanes = lanes;
  params.vocabulary = vocabulary_;
  for (uint32_t lane = 0; lane < kMaximumLanes; ++lane) {
    const uint32_t source = std::min(lane, lanes - 1);
    params.anchor[lane] = anchors[source];
    params.temperature[lane] = policies[source].temperature;
    if (lane < lanes && policies[lane].samples())
      params.sampling_mask |= uint32_t{1} << lane;
  }
  graph.add("draft_select_top16_sharded",
            {buffers.logits, buffers.partialIds, buffers.partialValues},
            vocabulary_,
            {uint64_t{lanes} * proposalTokens * kDraftShards, 1, 1});
  graph.add("draft_select_edges",
            {buffers.partialIds, buffers.partialValues, buffers.candidates,
             buffers.unary, buffers.selectorHidden,
             buffers.predecessorCodebook, buffers.successorCodebook},
            params, {uint64_t{lanes} * proposalTokens, 1, 1},
            {kEdgeThreads, 1, 1});
  graph.add("draft_select_dflash",
            {buffers.candidates, buffers.unary, buffers.partialValues,
             buffers.uniforms, buffers.proposedTokens,
             buffers.proposalProbabilities},
            params, {lanes, 1, 1}, {1, 1, 1});
}

void Sampling::addAcceptance(
    metal::CommandGraph &graph, AcceptanceBuffers buffers,
    std::span<const uint32_t> maximumRetained,
    std::span<const SamplingPolicy> policies, uint32_t stopToken0,
    uint32_t stopToken1) const {
  if (maximumRetained.empty() || maximumRetained.size() != policies.size() ||
      maximumRetained.size() > kMaximumLanes)
    throw std::invalid_argument("invalid DFlash acceptance batch");
  AcceptBatchParams params{};
  params.stop_token_0 = stopToken0;
  params.stop_token_1 = stopToken1;
  params.lanes = static_cast<uint32_t>(maximumRetained.size());
  for (uint32_t lane = 0; lane < params.lanes; ++lane) {
    if (!maximumRetained[lane] || maximumRetained[lane] > rowsPerLane_)
      throw std::invalid_argument("invalid DFlash retention limit");
    params.remaining[lane] = maximumRetained[lane];
    if (policies[lane].samples())
      params.sampling_mask |= uint32_t{1} << lane;
  }
  graph.add("decode_accept_dflash",
            {buffers.proposedTokens, buffers.candidates,
             buffers.proposalProbabilities, buffers.targetVocabularyRows,
             buffers.uniforms, buffers.outputTokens, buffers.retainedCounts,
             buffers.nextAnchors, buffers.acceptedCounts},
            params, {params.lanes, 1, 1}, {1, 1, 1});
}

void Sampling::addVerifyInput(metal::CommandGraph &graph,
                              metal::MetalBuffer draftInputTokens,
                              metal::MetalBuffer proposedTokens,
                              metal::MetalBuffer verifyInputTokens,
                              uint32_t lanes) const {
  if (!lanes || lanes > kMaximumLanes)
    throw std::invalid_argument("invalid verify input batch");
  const VerifyInputBatchParams params{lanes, vocabulary_};
  graph.add("verify_input_tokens",
            {std::move(draftInputTokens), std::move(proposedTokens),
             std::move(verifyInputTokens)},
            params, {uint64_t{lanes} * rowsPerLane_, 1, 1}, {1, 1, 1});
}

} // namespace splash::ops
