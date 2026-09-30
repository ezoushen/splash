// Target sampling policy against a direct CPU reference.
//
// The penalty kernels must rewrite exactly the logits of the penalized lanes'
// prompt and output tokens, within float rounding of the host arithmetic,
// from each lane's own penalty table row, counting at verify row r the draft
// tokens of rows 1..r; every other logit stays bitwise. The unchanged sampler
// then selects from them: every row's sparse distribution (top-k, top-p,
// renormalized, ids ascending) and every greedy argmax must match an
// evaluation of the same rules in double, and DFlash acceptance must accept
// and correct as a sequential decode would. The data are tie-free at every
// decision the reference makes; one within float rounding of a top_p boundary
// fails as ambiguous instead of passing by chance. Extreme
// repetition penalties saturate to exact, finite outcomes. The host word
// helpers and the lifecycle that rebuilds a resumed request's words are
// checked bitwise.
#include "metal/MetalBackend.hpp"
#include "ops/Sampling.hpp"

#import <Foundation/Foundation.h>

#include "metal/abi/Sampling.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using namespace splash::ops;

constexpr uint32_t kRows = SPLASH_TARGET_VERIFY_ROWS;
constexpr uint32_t kPositions = SPLASH_DRAFT_PROPOSAL_TOKENS;
constexpr uint32_t kLanes = SPLASH_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kCandidates = kTargetSamplingCandidates;
constexpr uint32_t kDraftCandidates = 16;
constexpr float kFloatMax = std::numeric_limits<float>::max();
// Both stop tokens sit below every spike of fillRow.
constexpr std::array<uint32_t, 2> kStopTokens{1, 2};

void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <class Function> void rejects(Function function, const char *what) {
  try {
    function();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error(std::string("accepted ") + what);
}

class Random final {
public:
  explicit Random(uint64_t seed) : state_(seed) {}
  // Uniform in [-1, 1).
  float unit() {
    return static_cast<float>(next() & 0xFFFFFF) / 8388608.0F - 1.0F;
  }
  uint32_t next() {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(state_ >> 33);
  }

private:
  uint64_t state_;
};

MetalBuffer allocate(MetalBackend &backend, uint64_t bytes) {
  MetalBuffer buffer =
      backend.allocateBuffer(bytes, BufferStorage::Shared, "target sampling");
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

struct Batch final {
  SamplingBuffers buffers;
  uint32_t vocabulary = 0;
  uint32_t rows = 0;

  [[nodiscard]] float *logits() const {
    return static_cast<float *>(buffers.logits.contents());
  }
  [[nodiscard]] float *row(uint32_t index) const {
    return logits() + uint64_t{index} * vocabulary;
  }
  [[nodiscard]] const uint32_t *topIds() const {
    return static_cast<const uint32_t *>(buffers.topIds.contents());
  }
  [[nodiscard]] const float *topProbabilities() const {
    return static_cast<const float *>(buffers.topProbabilities.contents());
  }
  [[nodiscard]] const uint32_t *outputTokens() const {
    return static_cast<const uint32_t *>(buffers.outputTokens.contents());
  }
  [[nodiscard]] uint32_t *inputTokens() const {
    return static_cast<uint32_t *>(buffers.inputTokens.contents());
  }
  [[nodiscard]] uint32_t *masks() const {
    return static_cast<uint32_t *>(buffers.constraintMasks.contents());
  }
  [[nodiscard]] float *uniforms() const {
    return static_cast<float *>(buffers.uniforms.contents());
  }
  [[nodiscard]] uint32_t maskWords() const { return (vocabulary + 31) / 32; }
  // Outputs a missing dispatch would leave behind cannot pass as results.
  void poison() const {
    for (const auto &buffer :
         {buffers.partialIds, buffers.partialValues, buffers.topIds,
          buffers.topProbabilities, buffers.outputTokens})
      std::memset(buffer.contents(), 0xA5, buffer.sizeBytes());
  }
};

Batch makeBatch(MetalBackend &backend, uint32_t vocabulary, uint32_t lanes) {
  const uint32_t rows = lanes * kRows;
  const auto space = Sampling::workspace(rows);
  return {SamplingBuffers{
              allocate(backend, uint64_t{rows} * vocabulary * sizeof(float)),
              allocate(backend, space.partialIdsBytes),
              allocate(backend, space.partialValuesBytes),
              allocate(backend, space.topIdsBytes),
              allocate(backend, space.topProbabilitiesBytes),
              allocate(backend, uint64_t{lanes} * 2 * kRows * sizeof(float)),
              allocate(backend,
                       uint64_t{lanes} * (kRows + 1) * ((vocabulary + 31) / 32) * 4),
              allocate(backend, uint64_t{rows} * sizeof(uint32_t)),
              allocate(backend, space.argmaxValuesBytes),
              allocate(backend, space.argmaxIndicesBytes),
              allocate(backend, uint64_t{rows} * sizeof(uint32_t))},
          vocabulary, rows};
}

// Low noise with forty distinct spikes: the production shape, whose top-32
// holds no ties. The stop tokens stay below every spike.
void fillRow(float *row, uint32_t vocabulary, Random &random) {
  for (uint32_t token = 0; token < vocabulary; ++token)
    row[token] = -8.0F + 3.0F * random.unit() * random.unit();
  for (uint32_t spike = 0; spike < 40; ++spike) {
    const uint32_t token = 3 + random.next() % (vocabulary - 3);
    row[token] = 1.0F + 0.25F * float(spike) + 0.1F * random.unit();
  }
}

// The tokens a row may select: those its constraint mask row allows, if it
// has one, less the stop tokens when its lane ignores end-of-sequence.
struct Admission final {
  const uint32_t *mask = nullptr;
  bool excludesStop = false;

  [[nodiscard]] bool operator()(uint32_t token) const {
    if (mask && !(mask[token / 32] & (1U << (token % 32))))
      return false;
    return !excludesStop ||
           (token != kStopTokens[0] && token != kStopTokens[1]);
  }
};

// The row's candidates as target_top_shard keeps them: value descending, id
// ascending, at most kCandidates.
std::vector<uint32_t> referenceCandidates(const float *row,
                                          uint32_t vocabulary,
                                          Admission admits = {}) {
  std::vector<uint32_t> order;
  for (uint32_t token = 0; token < vocabulary; ++token)
    if (admits(token))
      order.push_back(token);
  const auto beats = [&](uint32_t a, uint32_t b) {
    return row[a] > row[b] || (row[a] == row[b] && a < b);
  };
  const size_t keep = std::min<size_t>(kCandidates, order.size());
  std::partial_sort(order.begin(), order.begin() + keep, order.end(), beats);
  order.resize(keep);
  return order;
}

// The distance to a top_p boundary below which the GPU's float arithmetic
// could decide either way; data that close fails as ambiguous. A top_p
// prefix sums at most 32 normalized probabilities, each addition within half
// an ulp of 1.
constexpr double kTopPMargin = 32 * 0x1p-24;

// The sparse distribution top32_probs_row writes: every candidate in
// ascending id order, with its probability after top-k and top-p.
struct Distribution final {
  std::vector<uint32_t> ids;
  std::vector<double> probabilities;

  [[nodiscard]] double probability(uint32_t token) const {
    for (size_t index = 0; index < ids.size(); ++index)
      if (ids[index] == token)
        return probabilities[index];
    return 0.0;
  }
};

Distribution referenceDistribution(const float *row, uint32_t vocabulary,
                                   const SamplingPolicy &policy,
                                   Admission admits = {}) {
  const std::vector<uint32_t> candidates =
      referenceCandidates(row, vocabulary, admits);
  const size_t selected = std::min<size_t>(policy.topK, candidates.size());
  std::vector<double> weights(candidates.size(), 0.0);
  for (size_t rank = 0; rank < selected; ++rank)
    weights[rank] = std::exp((double(row[candidates[rank]]) - row[candidates[0]]) /
                             policy.temperature);
  const double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
  double prefix = 0.0;
  double kept = 0.0;
  for (size_t rank = 0; rank < selected; ++rank) {
    const double probability = weights[rank] / sum;
    // A rank too unlikely to move any probability past the tolerance may
    // be kept or cut.
    require(std::fabs(prefix - policy.topP) > kTopPMargin ||
                probability < 5e-7,
            "test data is ambiguous at top_p " + std::to_string(policy.topP) +
                ": a rank of probability " + std::to_string(probability) +
                " follows " + std::to_string(prefix));
    weights[rank] = prefix <= policy.topP ? probability : 0.0;
    prefix += probability;
    kept += weights[rank];
  }
  std::vector<size_t> order(candidates.size());
  std::iota(order.begin(), order.end(), size_t{0});
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return candidates[a] < candidates[b];
  });
  Distribution result;
  for (size_t rank : order) {
    result.ids.push_back(candidates[rank]);
    result.probabilities.push_back(weights[rank] / kept);
  }
  return result;
}

void requireDistribution(const Batch &batch, uint32_t outputRow,
                         const Distribution &expected,
                         const std::string &label) {
  const uint64_t origin = uint64_t{outputRow} * kCandidates;
  for (uint32_t rank = 0; rank < kCandidates; ++rank) {
    const uint32_t id = batch.topIds()[origin + rank];
    const float probability = batch.topProbabilities()[origin + rank];
    if (rank >= expected.ids.size()) {
      require(id == 0xFFFFFFFFU && probability == 0.0F,
              label + ": padding lost the empty sentinel");
      continue;
    }
    require(id == expected.ids[rank], label + ": candidate ids differ");
    require(std::isfinite(probability) &&
                std::fabs(probability - expected.probabilities[rank]) < 1e-6,
            label + ": probability " + std::to_string(probability) +
                " differs from " + std::to_string(expected.probabilities[rank]));
  }
}

uint32_t referenceArgmax(const float *row, uint32_t vocabulary,
                         Admission admits = {}) {
  return referenceCandidates(row, vocabulary, admits).front();
}

// The penalty kernel's arithmetic on the host: the logit of a token the
// output holds count times, or the prompt holds.
float referencePenalty(float value, uint32_t count, bool prompt,
                       const SamplingPenalties &penalties) {
  if (!count && !prompt)
    return value;
  value *= value > 0.0F
               ? std::min(1.0F / penalties.repetition, kFloatMax)
               : penalties.repetition;
  if (count)
    value -= penalties.frequency * float(count) + penalties.presence;
  return std::clamp(value, -kFloatMax, kFloatMax);
}

// A penalized logit as the kernel must write it. The scaling is one IEEE
// product on both sides, so a prompt-only token matches bitwise; the GPU may
// fuse frequency * count + presence, which leaves the difference within an
// ulp of the larger operand of the final subtraction.
bool penaltyMatches(float actual, float value, uint32_t count, bool prompt,
                    const SamplingPenalties &penalties) {
  const float expected = referencePenalty(value, count, prompt, penalties);
  if (!count || std::fabs(expected) == kFloatMax)
    return std::memcmp(&actual, &expected, sizeof(float)) == 0;
  const float subtrahend =
      penalties.frequency * float(count) + penalties.presence;
  const float magnitude = std::max(
      {std::fabs(expected + subtrahend), std::fabs(subtrahend),
       std::fabs(expected)});
  const float ulp = std::nextafter(magnitude, INFINITY) - magnitude;
  return std::fabs(actual - expected) <= 2.0F * ulp;
}

std::span<uint32_t> tableRow(const MetalBuffer &table, uint32_t vocabulary,
                             uint32_t row) {
  return {static_cast<uint32_t *>(table.contents()) + uint64_t{row} * vocabulary,
          vocabulary};
}

// A table of kLanes state slots, each row with its own words: prompt bits at
// a density of its own, small counts, and a few counts at the 2^20-token
// output limit.
MetalBuffer penaltyTable(MetalBackend &backend, uint32_t vocabulary) {
  MetalBuffer table = allocate(backend, uint64_t{kLanes} * vocabulary * 4);
  for (uint32_t slot = 0; slot < kLanes; ++slot) {
    Random random(0x7461626c65 + slot);
    std::span<uint32_t> words = tableRow(table, vocabulary, slot);
    for (uint32_t token = 0; token < vocabulary; ++token) {
      const uint32_t draw = random.next() % 64;
      uint32_t word = draw < 8 + 6 * slot ? SPLASH_PENALTY_PROMPT_BIT : 0U;
      if (draw % 3 == 0)
        word += 1 + draw % (slot + 3);
      if (draw == 63 && token % 7 == 0)
        word = (word & SPLASH_PENALTY_PROMPT_BIT) + (1U << 20);
      words[token] = word;
    }
  }
  return table;
}

// The draft tokens verify row r's context adds: the token count among the
// lane's verify input rows 1..r.
uint32_t draftedCount(const uint32_t *inputs, uint32_t row, uint32_t token) {
  uint32_t count = 0;
  for (uint32_t input = 1; input <= row; ++input)
    count += inputs[input] == token ? 1U : 0U;
  return count;
}

// Checks every logit of a lane's rows [first, first + rows) against the
// originals: penalized from the lane's words, or bitwise unchanged.
void requirePenalizedRows(const Batch &batch, const std::vector<float> &original,
                          uint32_t lane, uint32_t first, uint32_t rows,
                          std::span<const uint32_t> words,
                          const SamplingPenalties &penalties, bool verify,
                          const std::string &label) {
  const uint32_t vocabulary = batch.vocabulary;
  const uint32_t *inputs = batch.inputTokens() + uint64_t{lane} * kRows;
  for (uint32_t row = first; row < first + rows; ++row) {
    const uint64_t origin = (uint64_t{lane} * kRows + row) * vocabulary;
    for (uint32_t token = 0; token < vocabulary; ++token) {
      const float before = original[origin + token];
      const float after = batch.logits()[origin + token];
      const uint32_t word = words.empty() ? 0U : words[token];
      const uint32_t count = (word & SPLASH_PENALTY_COUNT_MASK) +
                             (verify ? draftedCount(inputs, row, token) : 0U);
      const bool prompt = (word & SPLASH_PENALTY_PROMPT_BIT) != 0;
      require(penalties.active()
                  ? penaltyMatches(after, before, count, prompt, penalties)
                  : std::memcmp(&after, &before, sizeof(float)) == 0,
              label + ": row " + std::to_string(row) + " token " +
                  std::to_string(token) + " wrote " + std::to_string(after) +
                  " for " + std::to_string(before));
    }
  }
}

// Penalized lanes read table rows that are neither their lane indices nor
// within the batch width. Plan lanes do not follow state slots.
constexpr std::array<std::array<uint32_t, kLanes>, kLanes> kTableRows{
    {{3}, {2, 0}, {1, 3, 0}, {2, 0, 3, 1}}};

// A sampled lane with every penalty, a constrained greedy lane with
// presence alone, an unpenalized sampled lane that ignores end-of-sequence,
// and a greedy lane whose repetition below 1 raises repeated tokens.
const std::array<SamplingPolicy, kLanes> kMixedPolicies{
    SamplingPolicy{32, 0.8F, 0.95F, false, false, {1.3F, 0.5F, 0.25F}},
    SamplingPolicy{1, 0.0F, 1.0F, true, false, {1.0F, 1.5F, 0.0F}},
    SamplingPolicy{20, 1.2F, 0.9F, false, true, {}},
    SamplingPolicy{1, 0.0F, 1.0F, false, false, {0.8F, -2.0F, 2.0F}}};

// Penalized logits of every verify row and of the first token at an offset,
// and the selections the unchanged sampler makes from them, at B1-B4 with
// penalized and unpenalized lanes side by side. With greedy lanes only, the
// batch keeps the argmax kernels.
void penalties(MetalBackend &backend, uint32_t vocabulary, uint32_t lanes,
               bool argmaxPath, uint32_t &changedSelections) {
  Random random(0x70656e + uint64_t{vocabulary} * 8 + lanes * 2 + argmaxPath);
  Sampling sampling(backend, vocabulary, kRows);
  const Batch batch = makeBatch(backend, vocabulary, lanes);
  const MetalBuffer table = penaltyTable(backend, vocabulary);
  for (uint32_t row = 0; row < batch.rows; ++row) {
    fillRow(batch.row(row), vocabulary, random);
    // A stop token leads every other row, so a lane that ignores
    // end-of-sequence has something to skip.
    if (row % 2)
      batch.row(row)[kStopTokens[0]] = 13.0F;
  }
  for (uint32_t index = 0; index < lanes * (kRows + 1) * batch.maskWords();
       ++index)
    batch.masks()[index] = 0xB6DB6DB6U ^ index;
  std::vector<SamplingPolicy> policies;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    SamplingPolicy policy = kMixedPolicies[lane];
    if (argmaxPath)
      policy = {1, 0.0F, 1.0F, false, false,
                kMixedPolicies[lane == 2 ? 0 : lane].penalties};
    policies.push_back(policy);
    // The draft repeats a token, and its tokens are among the row's best, so
    // the counts they add move selections.
    const std::vector<uint32_t> best =
        referenceCandidates(batch.row(lane * kRows), vocabulary);
    const std::array<uint32_t, kRows> inputs{
        best[9], best[0], best[1], best[0], best[2], best[3], best[1], best[4]};
    std::copy(inputs.begin(), inputs.end(),
              batch.inputTokens() + uint64_t{lane} * kRows);
  }
  const std::span<const uint32_t> rows =
      std::span(kTableRows[lanes - 1]).first(lanes);
  const std::vector<float> original(batch.logits(),
                                    batch.logits() + uint64_t{batch.rows} * vocabulary);
  const std::string label = std::string(argmaxPath ? "argmax" : "mixed") +
                            " penalties B" + std::to_string(lanes) +
                            " vocabulary " + std::to_string(vocabulary);
  const auto words = [&](uint32_t lane) {
    return tableRow(table, vocabulary, rows[lane]);
  };

  batch.poison();
  CommandGraph verify;
  sampling.addVerify(verify, policies, batch.buffers, kStopTokens[0],
                     kStopTokens[1], {table, rows});
  static_cast<void>(backend.submitCommand(verify.dispatches()));
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    requirePenalizedRows(batch, original, lane, 0, kRows, words(lane),
                         policies[lane].penalties, true, label + " verify");
    for (uint32_t row = lane * kRows; row < (lane + 1) * kRows; ++row) {
      const SamplingPolicy &policy = policies[lane];
      const Admission admits{
          policy.constrained
              ? batch.masks() + (uint64_t{lane} * (kRows + 1) + row % kRows + 1) *
                                    batch.maskWords()
              : nullptr,
          policy.excludesStopTokens};
      if (policy.samples()) {
        requireDistribution(batch, row,
                            referenceDistribution(batch.row(row), vocabulary,
                                                  policy, admits),
                            label + " verify row " + std::to_string(row));
        continue;
      }
      const uint32_t token = batch.outputTokens()[row];
      require(token == referenceArgmax(batch.row(row), vocabulary, admits),
              label + ": a greedy verify row lost its penalized argmax");
      const float *unpenalized = original.data() + uint64_t{row} * vocabulary;
      changedSelections +=
          token != referenceArgmax(unpenalized, vocabulary, admits) ? 1U : 0U;
    }
  }

  // The first token after a prompt: one row of lane 0 at an offset, no draft
  // tokens; every other row and lane keeps its logits.
  constexpr uint32_t kOffset = 5;
  std::copy(original.begin(), original.end(), batch.logits());
  batch.poison();
  CommandGraph initial;
  sampling.addInitial(initial, policies.front(), batch.buffers, kOffset,
                      kStopTokens[0], kStopTokens[1], {table, rows.first(1)});
  static_cast<void>(backend.submitCommand(initial.dispatches()));
  requirePenalizedRows(batch, original, 0, kOffset, 1, words(0),
                       policies.front().penalties, false, label + " initial");
  for (uint32_t lane = 0; lane < lanes; ++lane)
    for (uint32_t row = 0; row < kRows; ++row)
      if (lane || row != kOffset)
        requirePenalizedRows(batch, original, lane, row, 1, {}, {}, false,
                             label + " initial, unselected row");
  const SamplingPolicy &first = policies.front();
  const Admission admits{first.constrained ? batch.masks() : nullptr,
                         first.excludesStopTokens};
  if (first.samples())
    requireDistribution(batch, 0,
                        referenceDistribution(batch.row(kOffset), vocabulary,
                                              first, admits),
                        label + " initial");
  else
    require(batch.outputTokens()[0] ==
                referenceArgmax(batch.row(kOffset), vocabulary, admits),
            label + ": the initial token lost its penalized argmax");
}

// A request's DFlash cycle against a sequential decode. Every verify row of
// a lane has nearly the same logits, as a model that would repeat itself,
// so only the draft tokens its context adds keep row r from selecting what
// row 0 did. Row r must select from the distribution a non-speculative decode
// would, given the context through draft token r - 1; the drafts are what
// that decode selects up to a lane's first wrong one, and acceptance must
// stop there with the same correction and next anchor. Sampled lanes accept
// a one-hot draft token while its uniform is below the token's penalized
// probability.
void speculativeExactness(MetalBackend &backend, uint32_t samplingMask) {
  constexpr uint32_t vocabulary = 1003;
  constexpr uint32_t lanes = 2;
  constexpr std::array<uint32_t, lanes> kSlots{3, 1};
  // Lane 0 drafts its first wrong token at position 3; lane 1 drafts every
  // token right and takes the bonus row.
  constexpr std::array<uint32_t, lanes> kWrongAt{3, kPositions};
  const std::array<SamplingPenalties, lanes> penalties{
      SamplingPenalties{1.2F, 2.0F, 2.0F}, SamplingPenalties{1.0F, 1.5F, 0.0F}};
  Random random(0x73706563 + samplingMask);
  Sampling sampling(backend, vocabulary, kRows);
  const Batch batch = makeBatch(backend, vocabulary, lanes);
  const MetalBuffer table = penaltyTable(backend, vocabulary);
  const auto proposals = Sampling::draftWorkspace(lanes * kPositions);
  AcceptanceBuffers acceptance{
      allocate(backend, uint64_t{lanes} * kPositions * sizeof(uint32_t)),
      allocate(backend, proposals.candidatesBytes),
      allocate(backend, proposals.proposalProbabilitiesBytes),
      batch.buffers.topIds,
      batch.buffers.topProbabilities,
      batch.buffers.uniforms,
      batch.buffers.outputTokens,
      allocate(backend, lanes * sizeof(uint32_t)),
      allocate(backend, lanes * sizeof(uint32_t)),
      allocate(backend, lanes * sizeof(uint32_t))};
  auto *proposed = static_cast<uint32_t *>(acceptance.proposedTokens.contents());
  auto *candidates = static_cast<uint32_t *>(acceptance.candidates.contents());
  auto *proposal =
      static_cast<float *>(acceptance.proposalProbabilities.contents());
  std::fill(candidates, candidates + lanes * kPositions * kDraftCandidates,
            0xFFFFFFFFU);

  std::vector<SamplingPolicy> policies;
  std::array<std::array<uint32_t, kRows>, lanes> expectedOutput{};
  std::array<uint32_t, lanes> expectedAccepted{};
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const bool sampled = (samplingMask >> lane) & 1U;
    policies.push_back({sampled ? 32U : 1U, sampled ? 0.9F : 0.0F, 0.9F, false,
                        false, penalties[lane]});
    std::span<const uint32_t> words = tableRow(table, vocabulary, kSlots[lane]);
    std::vector<uint32_t> counts(vocabulary);
    for (uint32_t token = 0; token < vocabulary; ++token)
      counts[token] = words[token] & SPLASH_PENALTY_COUNT_MASK;
    uint32_t *inputs = batch.inputTokens() + lane * kRows;
    float *uniforms = batch.uniforms() + lane * 2 * kRows;
    inputs[0] = 5 + lane;
    std::vector<float> base(vocabulary);
    fillRow(base.data(), vocabulary, random);
    std::vector<Distribution> targets;
    std::vector<uint32_t> selections;
    for (uint32_t row = 0; row < kRows; ++row) {
      float *logits = batch.row(lane * kRows + row);
      for (uint32_t token = 0; token < vocabulary; ++token)
        logits[token] = base[token] + 0.01F * random.unit();
      // What a sequential decode selects from at this position.
      std::vector<float> penalized(logits, logits + vocabulary);
      for (uint32_t token = 0; token < vocabulary; ++token)
        penalized[token] = referencePenalty(
            logits[token], counts[token],
            (words[token] & SPLASH_PENALTY_PROMPT_BIT) != 0, penalties[lane]);
      const std::vector<uint32_t> order =
          referenceCandidates(penalized.data(), vocabulary);
      require(penalized[order[0]] - penalized[order[1]] > 1e-4F,
              "test data is ambiguous at a greedy selection");
      targets.push_back(sampled ? referenceDistribution(penalized.data(),
                                                        vocabulary,
                                                        policies[lane])
                                : Distribution{});
      selections.push_back(order[0]);
      if (row == kPositions)
        break;
      // The draft: the best penalized candidate until the wrong position,
      // where it takes the eighth best, whose probability is small.
      const uint32_t draft = row < kWrongAt[lane] ? order[0] : order[7];
      inputs[row + 1] = draft;
      proposed[lane * kPositions + row] = draft;
      candidates[(lane * kPositions + row) * kDraftCandidates] = draft;
      proposal[(lane * kPositions + row) * kDraftCandidates] = 1.0F;
      ++counts[draft];
      const double p = targets.back().probability(draft);
      // Accept a right token surely and refuse the wrong one.
      uniforms[kRows + row] =
          row < kWrongAt[lane] ? float(p * 0.5) : float(std::min(1.0, p * 2.0 + 0.01));
      require(!sampled || row >= kWrongAt[lane] || p > 1e-4,
              "a sampled lane drafted a token its target cannot accept");
    }
    uniforms[2 * kRows - 1] = 0.37F;

    // Acceptance as accept_greedy_lane and accept_sampled_lane define it.
    const uint32_t accepted = kWrongAt[lane];
    expectedAccepted[lane] = accepted;
    for (uint32_t position = 0; position < accepted; ++position)
      expectedOutput[lane][position] = inputs[position + 1];
    const Distribution &last = targets[accepted];
    if (!sampled) {
      expectedOutput[lane][accepted] = selections[accepted];
    } else {
      // The bonus row samples the target; a rejected row samples its
      // residual, the target without the one-hot draft token.
      const uint32_t excluded =
          accepted < kPositions ? inputs[accepted + 1] : 0xFFFFFFFFU;
      double total = 0.0;
      for (size_t index = 0; index < last.ids.size(); ++index)
        total += last.ids[index] == excluded ? 0.0 : last.probabilities[index];
      const double threshold = double(uniforms[2 * kRows - 1]) * total;
      double cumulative = 0.0;
      std::optional<uint32_t> drawn;
      for (size_t index = 0; index < last.ids.size() && !drawn; ++index) {
        if (last.ids[index] == excluded || !(last.probabilities[index] > 0.0))
          continue;
        cumulative += last.probabilities[index];
        require(std::fabs(cumulative - threshold) > 1e-5,
                "test data is ambiguous at the correction draw");
        if (cumulative > threshold)
          drawn = last.ids[index];
      }
      require(drawn.has_value(), "the reference correction drew nothing");
      expectedOutput[lane][accepted] = *drawn;
    }
  }

  batch.poison();
  CommandGraph graph;
  sampling.addVerify(graph, policies, batch.buffers, kStopTokens[0],
                     kStopTokens[1], {table, kSlots});
  const std::array<uint32_t, lanes> maximumRetained{kRows, kRows};
  sampling.addAcceptance(graph, acceptance, maximumRetained, policies,
                         kStopTokens[0], kStopTokens[1]);
  static_cast<void>(backend.submitCommand(graph.dispatches()));
  const auto *retained =
      static_cast<const uint32_t *>(acceptance.retainedCounts.contents());
  const auto *acceptedCounts =
      static_cast<const uint32_t *>(acceptance.acceptedCounts.contents());
  const auto *anchors =
      static_cast<const uint32_t *>(acceptance.nextAnchors.contents());
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const std::string label = "speculative lane " + std::to_string(lane) +
                              " sampling mask " + std::to_string(samplingMask);
    const uint32_t accepted = expectedAccepted[lane];
    require(acceptedCounts[lane] == accepted && retained[lane] == accepted + 1,
            label + ": accepted " + std::to_string(acceptedCounts[lane]) +
                " draft tokens, not " + std::to_string(accepted));
    for (uint32_t position = 0; position <= accepted; ++position)
      require(batch.outputTokens()[lane * kRows + position] ==
                  expectedOutput[lane][position],
              label + ": output differs from a sequential decode");
    require(anchors[lane] == expectedOutput[lane][accepted],
            label + ": the next anchor differs from a sequential decode");
  }
}

// Extreme repetition penalties saturate instead of overflowing: every
// candidate the penalty drives to the same float limit ties, greedy takes the
// lowest id among them, and sampling spreads evenly over those top-k keeps,
// with finite probabilities, and draws one of them.
void extremes(MetalBackend &backend) {
  constexpr uint32_t vocabulary = 1003;
  constexpr uint32_t kSaturated = 50;
  constexpr uint32_t kTopK = 20;
  Sampling sampling(backend, vocabulary, kRows);
  const Batch batch = makeBatch(backend, vocabulary, 1);
  const MetalBuffer table = allocate(backend, uint64_t{kLanes} * vocabulary * 4);
  const std::span<uint32_t> words = tableRow(table, vocabulary, 2);
  const std::array<uint32_t, 1> rows{2};
  // Tokens 100, 107, ... hold the prompt bit and positive logits large enough
  // to overflow, or, for the mask case, negative ones.
  std::vector<uint32_t> saturating;
  for (uint32_t index = 0; index < kSaturated; ++index)
    saturating.push_back(100 + 7 * (kSaturated - 1 - index));
  std::sort(saturating.begin(), saturating.end());
  struct Case final {
    float repetition;
    float saturatingLogit;
    bool masked;
  };
  for (const Case c : {Case{1e-37F, 45.0F, false},
                       Case{std::ldexp(1.0F, -149), 45.0F, false},
                       Case{1e38F, -6.0F, true}}) {
    float *row = batch.logits();
    for (uint32_t token = 0; token < vocabulary; ++token) {
      row[token] = -3.0F + float(token % 17) * 0.1F;
      words[token] = token % 5 == 0 ? SPLASH_PENALTY_PROMPT_BIT : 0U;
    }
    for (const uint32_t token : saturating) {
      row[token] = c.saturatingLogit + float(token % 3);
      words[token] = SPLASH_PENALTY_PROMPT_BIT;
    }
    // The mask leaves only the saturating tokens.
    std::fill(batch.masks(), batch.masks() + batch.maskWords(), 0U);
    for (const uint32_t token : saturating)
      batch.masks()[token / 32] |= 1U << (token % 32);
    const std::vector<float> original(row, row + kRows * vocabulary);
    for (const float temperature : {0.0F, 1.0F}) {
      for (const float uniform : {0.0F, 0.51F, 0.999F}) {
        std::copy(original.begin(), original.end(), row);
        batch.poison();
        batch.uniforms()[0] = uniform;
        const SamplingPolicy policy{temperature > 0.0F ? kTopK : 1U,
                                    temperature,
                                    1.0F,
                                    c.masked,
                                    false,
                                    {c.repetition, 0.0F, 0.0F}};
        CommandGraph graph;
        sampling.addInitial(graph, policy, batch.buffers, 0, kStopTokens[0],
                            kStopTokens[1], {table, rows});
        static_cast<void>(backend.submitCommand(graph.dispatches()));
        const std::string label =
            "repetition " + std::to_string(c.repetition) + " temperature " +
            std::to_string(temperature);
        for (const uint32_t token : saturating)
          require(row[token] == (c.saturatingLogit > 0.0F ? kFloatMax : -kFloatMax),
                  label + ": a penalized logit did not saturate");
        for (uint32_t token = 0; token < vocabulary; ++token)
          require(std::isfinite(row[token]),
                  label + ": a penalized logit is not finite");
        const uint32_t token = batch.outputTokens()[0];
        if (temperature == 0.0F) {
          require(token == saturating.front(),
                  label + ": greedy did not take the lowest saturated id");
          continue;
        }
        const std::vector<uint32_t> kept(saturating.begin(),
                                         saturating.begin() + kTopK);
        require(std::find(kept.begin(), kept.end(), token) != kept.end(),
                label + ": sampling drew a token top-k does not keep");
        for (uint32_t rank = 0; rank < kCandidates; ++rank) {
          const float probability = batch.topProbabilities()[rank];
          const bool isKept =
              std::find(kept.begin(), kept.end(), batch.topIds()[rank]) !=
              kept.end();
          require(std::isfinite(probability) &&
                      std::fabs(probability - (isKept ? 1.0F / kTopK : 0.0F)) <
                          1e-6F,
                  label + ": saturated candidates are not uniform");
        }
      }
    }
  }
}

// A penalized lane needs a table row inside the table and, for verify, its
// input tokens; penalties must be finite with a positive repetition. A
// refused request encodes nothing.
void invalidPenalties(MetalBackend &backend) {
  constexpr uint32_t vocabulary = 1003;
  Sampling sampling(backend, vocabulary, kRows);
  Batch batch = makeBatch(backend, vocabulary, 1);
  const MetalBuffer table = penaltyTable(backend, vocabulary);
  const SamplingPolicy penalized{1, 0.0F, 1.0F, false, false,
                                 {1.0F, 1.0F, 0.0F}};
  const std::array<uint32_t, 1> inside{3};
  const std::array<uint32_t, 1> outside{kLanes};
  CommandGraph graph;
  rejects([&] { sampling.addVerify(graph, {&penalized, 1}, batch.buffers, 1, 2); },
          "penalties without a table");
  rejects([&] {
    sampling.addVerify(graph, {&penalized, 1}, batch.buffers, 1, 2,
                       {table, outside});
  }, "a table row outside the table");
  rejects([&] {
    sampling.addInitial(graph, penalized, batch.buffers, 0, 1, 2,
                        {table, outside});
  }, "an initial table row outside the table");
  for (const SamplingPenalties invalid :
       {SamplingPenalties{0.0F, 0.0F, 0.0F},
        SamplingPenalties{-1.0F, 0.0F, 0.0F},
        SamplingPenalties{INFINITY, 0.0F, 0.0F},
        SamplingPenalties{1.0F, NAN, 0.0F},
        SamplingPenalties{1.0F, 0.0F, INFINITY}}) {
    SamplingPolicy policy = penalized;
    policy.penalties = invalid;
    rejects([&] {
      sampling.addInitial(graph, policy, batch.buffers, 0, 1, 2,
                          {table, inside});
    }, "invalid penalties");
  }
  batch.buffers.inputTokens = {};
  rejects([&] {
    sampling.addVerify(graph, {&penalized, 1}, batch.buffers, 1, 2,
                       {table, inside});
  }, "penalized verify rows without input tokens");
  require(graph.empty(), "a refused penalty request encoded a dispatch");
  // Unpenalized lanes need neither.
  const SamplingPolicy greedy{1, 0.0F, 1.0F, false};
  sampling.addVerify(graph, {&greedy, 1}, batch.buffers, 1, 2);
  require(!graph.empty(), "an unpenalized batch encoded nothing");
}

// The host words: prompt bits only when repetition reads them, counts of
// every selected token, and no token outside the vocabulary.
void penaltyWords() {
  std::vector<uint32_t> words(8, 0xDEADBEEFU);
  const std::vector<uint32_t> prompt{1, 3, 3, 7};
  const std::vector<uint32_t> selected{3, 5, 5, 0};
  Sampling::loadPenaltyWords(words, prompt, selected, true);
  constexpr uint32_t kPrompt = SPLASH_PENALTY_PROMPT_BIT;
  require(words == std::vector<uint32_t>{1, kPrompt, 0, kPrompt + 1, 0, 2, 0,
                                         kPrompt},
          "penalty words with prompt bits differ");
  Sampling::loadPenaltyWords(words, prompt, selected, false);
  require(words == std::vector<uint32_t>{1, 0, 0, 1, 0, 2, 0, 0},
          "penalty words without prompt bits differ");
  const std::vector<uint32_t> step{5, 6};
  Sampling::countPenaltyTokens(words, step);
  require(words == std::vector<uint32_t>{1, 0, 0, 1, 0, 3, 1, 0},
          "counted penalty words differ");
  const std::vector<uint32_t> before = words;
  const std::vector<uint32_t> outside{2, 8};
  rejects([&] { Sampling::countPenaltyTokens(words, outside); },
          "a counted token outside the vocabulary");
  rejects([&] { Sampling::loadPenaltyWords(words, outside, {}, true); },
          "a prompt token outside the vocabulary");
  rejects([&] { Sampling::loadPenaltyWords(words, {}, outside, false); },
          "a selected token outside the vocabulary");
  require(words == before, "a refused token changed the penalty words");
}

// A request's lifetime as the model runtime keeps its words: activation loads
// the prompt, every selection counts the step's tokens (the first token after
// the prompt, a constrained request's first token, verify commits whose last
// token is the new anchor, cut at a stop token), and a terminal anchor is
// emitted without a step of its own. A resume rebuilds the words from the
// history the engine passes (the prompt, then the emitted outputs) and the
// pending anchor, and continues from them. Wherever a request can be
// suspended (during the prompt, at its end with and without a pending
// anchor, mid-generation, and again during the replay) the rebuild must
// equal the words counted step by step.
void penaltyLifecycle() {
  constexpr uint32_t vocabulary = 97;
  constexpr uint32_t kStop = 96;
  for (const bool markPrompt : {false, true}) {
    for (const bool constrained : {false, true}) {
      for (uint32_t seed = 0; seed < 8; ++seed) {
        Random random(0x6c696665 + seed * 4 + constrained * 2 + markPrompt);
        std::vector<uint32_t> prompt(40);
        for (uint32_t &token : prompt)
          token = random.next() % (vocabulary - 1);
        // The runtime's state: its slot's words, emitted count and anchor;
        // the engine's history, which resume passes as the new prompt.
        std::vector<uint32_t> words(vocabulary);
        std::vector<uint32_t> history = prompt;
        uint32_t generated = 0;
        std::optional<uint32_t> pending;
        Sampling::loadPenaltyWords(words, prompt, {}, markPrompt);
        const auto commit = [&](std::span<const uint32_t> tokens) {
          Sampling::countPenaltyTokens(words, tokens);
          pending = tokens.back();
        };
        const auto emit = [&](std::span<const uint32_t> tokens) {
          history.insert(history.end(), tokens.begin(), tokens.end());
          generated += static_cast<uint32_t>(tokens.size());
        };
        const auto resume = [&](const char *stage) {
          std::vector<uint32_t> rebuilt(vocabulary, 0xA5A5A5A5U);
          require(history.size() > generated, "history lost its prompt");
          const size_t promptSize = history.size() - generated;
          Sampling::loadPenaltyWords(
              rebuilt, std::span(history).first(promptSize),
              std::span(history).subspan(promptSize), markPrompt);
          if (pending)
            Sampling::countPenaltyTokens(rebuilt, {&*pending, 1});
          require(rebuilt == words,
                  std::string("resumed penalty words differ ") + stage);
          words = std::move(rebuilt);
        };
        resume("during the prompt");
        if (constrained)
          resume("while the first token waits for its mask");
        const uint32_t first = random.next() % (vocabulary - 1);
        commit({&first, 1});
        resume("with the first token pending");
        for (uint32_t cycle = 0; cycle < 9 && *pending != kStop; ++cycle) {
          const uint32_t retained = 1 + random.next() % kRows;
          std::vector<uint32_t> tokens(retained);
          for (uint32_t &token : tokens)
            token = random.next() % vocabulary;
          // Acceptance cuts the retained tokens after the first stop token.
          const auto stop = std::find(tokens.begin(), tokens.end(), kStop);
          if (stop != tokens.end())
            tokens.erase(stop + 1, tokens.end());
          std::vector<uint32_t> output{*pending};
          output.insert(output.end(), tokens.begin(), tokens.end() - 1);
          emit(output);
          commit(tokens);
          resume("mid-generation");
          resume("during the replay");
        }
        // The terminal anchor was counted when selected.
        emit({&*pending, 1});
        pending.reset();
        resume("after the terminal anchor");
      }
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  std::string stage = "setup";
  try {
    if (argc != 2)
      throw std::invalid_argument("usage: target-sampling METALLIB");
    stage = "penalty words";
    penaltyWords();
    stage = "penalty lifecycle";
    penaltyLifecycle();
    MetalBackend backend(argv[1]);
    stage = "invalid penalties";
    invalidPenalties(backend);
    stage = "extreme penalties";
    extremes(backend);
    for (const uint32_t samplingMask : {0U, 1U, 2U, 3U}) {
      stage = "speculative exactness, sampling mask " +
              std::to_string(samplingMask);
      speculativeExactness(backend, samplingMask);
    }
    for (const uint32_t vocabulary : {1003U, 248320U}) {
      uint32_t changedSelections = 0;
      for (uint32_t lanes = 1; lanes <= kLanes; ++lanes) {
        const std::string batch = " B" + std::to_string(lanes) +
                                  ", vocabulary " + std::to_string(vocabulary);
        stage = "mixed penalties" + batch;
        penalties(backend, vocabulary, lanes, false, changedSelections);
        stage = "argmax penalties" + batch;
        penalties(backend, vocabulary, lanes, true, changedSelections);
      }
      require(changedSelections > 0,
              "the penalties changed no greedy selection");
    }
    std::cout << "target_sampling_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "target_sampling_metal_test: FAIL (" << stage
              << "): " << error.what() << '\n';
    return 1;
  }
}
