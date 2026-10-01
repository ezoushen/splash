#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/MetalBackend.hpp"

#include <cstdint>
#include <span>

namespace splash::ops {

// The sampling penalties, which rewrite a lane's target logits before its
// policy selects from them: repetition scales the logit of every token the
// prompt or the output holds, presence and frequency lower that of every
// token the output holds. The defaults leave the logits unchanged.
struct SamplingPenalties final {
  float repetition = 1.0F;
  float presence = 0.0F;
  float frequency = 0.0F;

  [[nodiscard]] bool active() const noexcept {
    return repetition != 1.0F || presence != 0.0F || frequency != 0.0F;
  }
};

struct SamplingPolicy final {
  // A sampling lane keeps the tokens minP leaves it, then its topK most
  // likely of those, or every one for 0 or a topK past the vocabulary (top-k
  // disabled), then its top-p nucleus of those.
  uint32_t topK = 1;
  float temperature = 1.0F;
  float topP = 1.0F;
  bool constrained = false;
  // The lane ignores end-of-sequence: the target never selects a stop token,
  // though the draft may still propose one.
  bool excludesStopTokens = false;
  // Greedy lanes take the argmax of the penalized logits.
  SamplingPenalties penalties{};
  // A sampling lane first drops the tokens less likely than minP times its
  // most likely one; 0 drops none.
  float minP = 0.0F;

  [[nodiscard]] bool samples() const noexcept { return temperature > 0.0F; }
};

// The penalty words of every state slot, one row of vocabulary words each,
// and the row each lane of a dispatch reads. Lanes follow the batch plan,
// not slots, so a lane never binds a slot's row by its own index.
struct PenaltyTable final {
  metal::MetalBuffer words;
  std::span<const uint32_t> rows;
};

struct SamplingWorkspace final {
  uint64_t argmaxValuesBytes = 0;
  uint64_t argmaxIndicesBytes = 0;
  uint64_t partialMassesBytes = 0;
  uint64_t vocabularyRowsBytes = 0;
  uint64_t vocabularyRangesBytes = 0;
  uint64_t vocabularyArrivalsBytes = 0;
};

struct DraftSelectorWorkspace final {
  uint64_t partialIdsBytes = 0;
  uint64_t partialValuesBytes = 0;
  uint64_t candidatesBytes = 0;
  uint64_t unaryBytes = 0;
  uint64_t proposalProbabilitiesBytes = 0;
};

struct SamplingBuffers final {
  // fp32 [rows][vocabulary].
  metal::MetalBuffer logits;
  // Per shard of a sampled row, its share of the row's softmax denominator
  // (metal/abi/Sampling.h TargetShardMass).
  metal::MetalBuffer partialMasses;
  // Per sampled row, where its distribution ends and its draw
  // (TargetVocabularyRow); acceptance reads a verify row's.
  metal::MetalBuffer vocabularyRows;
  metal::MetalBuffer uniforms;
  metal::MetalBuffer constraintMasks;
  metal::MetalBuffer outputTokens;
  metal::MetalBuffer argmaxValues;
  metal::MetalBuffer argmaxIndices;
  // Verify input tokens [rows]: row 0 of a lane is its anchor, rows 1..7 its
  // draft tokens. Only penalized and sampled verify rows read them.
  metal::MetalBuffer inputTokens{};
  // The draft's candidates and their probabilities at each proposal
  // position (AcceptanceBuffers), which a sampled verify row draws its
  // correction's residual from.
  metal::MetalBuffer draftCandidates{};
  metal::MetalBuffer draftProbabilities{};
  // Per sampled row, the ranges of its draw (TargetVocabularyRange), which
  // the groups that share the draw sum, and how many of those groups have
  // finished: a count every draw returns to zero, where it starts.
  metal::MetalBuffer vocabularyRanges{};
  metal::MetalBuffer vocabularyArrivals{};
};

struct DraftSelectorBuffers final {
  // fp32 [rows][vocabulary].
  metal::MetalBuffer logits;
  metal::MetalBuffer partialIds;
  metal::MetalBuffer partialValues;
  metal::MetalBuffer candidates;
  metal::MetalBuffer unary;
  metal::MetalBuffer selectorHidden;
  metal::MetalBuffer predecessorCodebook;
  metal::MetalBuffer successorCodebook;
  metal::MetalBuffer uniforms;
  metal::MetalBuffer proposedTokens;
  metal::MetalBuffer proposalProbabilities;
};

struct AcceptanceBuffers final {
  metal::MetalBuffer proposedTokens;
  metal::MetalBuffer candidates;
  metal::MetalBuffer proposalProbabilities;
  metal::MetalBuffer targetVocabularyRows;
  metal::MetalBuffer uniforms;
  metal::MetalBuffer outputTokens;
  metal::MetalBuffer retainedCounts;
  metal::MetalBuffer nextAnchors;
  metal::MetalBuffer acceptedCounts;
};

// Target token policy. This operator owns the sampling penalties,
// min-p/top-k/top-p, constrained selection, stop-token exclusion and greedy
// argmax pipeline ABIs; the model only supplies policy, buffers, penalty
// words and its stop tokens.
class Sampling final {
public:
  // rowsPerLane is the kernels' SPLASH_TARGET_VERIFY_ROWS.
  Sampling(uint32_t vocabulary, uint32_t rowsPerLane);

  // Exact scratch/output bytes for the fixed precompiled sampling ABI.
  // Counts may cover one lane or a packed batch; the operator owns sharding.
  [[nodiscard]] static SamplingWorkspace workspace(uint32_t rows);
  [[nodiscard]] static DraftSelectorWorkspace draftWorkspace(uint32_t positions);

  // A penalized request's penalty words (metal/abi/Sampling.h): the prompt
  // bit of every prompt token when markPrompt, as only repetition reads it,
  // and the count of every token the target selected. Every token must be
  // inside the vocabulary, one word each.
  static void loadPenaltyWords(std::span<uint32_t> words,
                               std::span<const uint32_t> prompt,
                               std::span<const uint32_t> selected,
                               bool markPrompt);
  static void countPenaltyTokens(std::span<uint32_t> words,
                                 std::span<const uint32_t> selected);

  // A lane whose policy has active penalties has its logits rewritten in
  // place first, from its row of the penalty table; the rows must hold the
  // LM head's fresh output. Other lanes dispatch nothing new. A greedy lane
  // then takes each row's argmax and a sampled lane draws from each row's
  // distribution over the whole vocabulary, both among the tokens the lane
  // admits: the first token directly, a verify row as its draft token's
  // probability and correction, which acceptance reads from the vocabulary
  // rows.
  void addInitial(metal::CommandGraph &graph, const SamplingPolicy &policy,
                  SamplingBuffers buffers, uint32_t rowOffset,
                  uint32_t stopToken0, uint32_t stopToken1,
                  PenaltyTable penalties = {}) const;
  void addVerify(metal::CommandGraph &graph,
                 std::span<const SamplingPolicy> policies,
                 SamplingBuffers buffers, uint32_t stopToken0,
                 uint32_t stopToken1, PenaltyTable penalties = {}) const;
  // proposalTokens is the kernels' SPLASH_DRAFT_PROPOSAL_TOKENS.
  void addDraftSelector(
      metal::CommandGraph &graph, DraftSelectorBuffers buffers,
      std::span<const uint32_t> anchors,
      std::span<const SamplingPolicy> policies, uint32_t proposalTokens) const;
  void addAcceptance(
      metal::CommandGraph &graph, AcceptanceBuffers buffers,
      std::span<const uint32_t> maximumRetained,
      std::span<const SamplingPolicy> policies, uint32_t stopToken0,
      uint32_t stopToken1) const;
  void addVerifyInput(metal::CommandGraph &graph,
                      metal::MetalBuffer draftInputTokens,
                      metal::MetalBuffer proposedTokens,
                      metal::MetalBuffer verifyInputTokens,
                      uint32_t lanes) const;

private:
  // Penalizes the row at rowOffset of each penalized lane or, for verify,
  // all its rows, each also counting the draft tokens its context adds.
  void addPenalties(metal::CommandGraph &graph,
                    std::span<const SamplingPolicy> policies,
                    const SamplingBuffers &buffers, const PenaltyTable &table,
                    uint32_t rowOffset, bool verify) const;

  uint32_t vocabulary_ = 0;
  uint32_t rowsPerLane_ = 0;
  uint32_t maskWords_ = 0;
};

} // namespace splash::ops
