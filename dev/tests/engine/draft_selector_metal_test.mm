// DFlash draft selector against a direct CPU reference: the sharded top-16
// scan and its reduce must return exactly the sixteen largest logits of every
// proposal row in (value desc, id asc) order, and the codebook walk must pick
// the same tokens as a double-precision evaluation of the same scores. The
// vocabularies cover the production size, an odd size that misaligns the
// 16-byte vectors and leaves shards with only a few tokens, and one wider
// than a single register chunk per thread. The logits are fp32, and their
// order is decided below the bf16 spacing.
#include "metal/MetalBackend.hpp"
#include "metal/abi/Sampling.h"
#include "ops/Sampling.hpp"
#include "tuning/LinearNumerics.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

using splash::metal::BufferStorage;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using namespace splash::ops;

constexpr uint32_t kRows = SPLASH_DRAFT_QUERY_ROWS;
constexpr uint32_t kPositions = SPLASH_DRAFT_PROPOSAL_TOKENS;
constexpr uint32_t kCandidates = 16;
constexpr uint32_t kRank = 256;
constexpr uint32_t kLanes = SPLASH_MAXIMUM_BATCH_WIDTH;

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <class Function> void rejects(Function function) {
  try {
    function();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error("invalid draft selector request was accepted");
}

class Random final {
public:
  explicit Random(uint64_t seed) : state_(seed) {}
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

// Target sampling takes the model's stop tokens: here one in the first shard
// and one in the last, so a lane that excludes them skips them in both.
std::array<uint32_t, 2> stopTokens(uint32_t vocabulary) {
  return {1, vocabulary - 2};
}

MetalBuffer allocate(MetalBackend &backend, uint64_t bytes) {
  MetalBuffer buffer =
      backend.allocateBuffer(bytes, BufferStorage::Shared, "draft selector");
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

MetalBuffer randomBfloat(MetalBackend &backend, uint64_t count, Random &random,
                         float scale) {
  MetalBuffer buffer = allocate(backend, count * sizeof(uint16_t));
  auto *values = static_cast<uint16_t *>(buffer.contents());
  for (uint64_t index = 0; index < count; ++index)
    values[index] = tuning::floatToBf16(random.unit() * scale);
  return buffer;
}

// Row patterns: peaked logits with a few spikes (the production shape),
// uniform noise, heavy ties from a seven-value alphabet with -inf runs, and
// a row with ten finite tokens so -inf tokens fill the tail by id.
enum class Pattern : uint8_t { Peaked, Uniform, Ties, Sparse };

void fillRow(float *row, uint32_t vocabulary, Pattern pattern,
             Random &random) {
  for (uint32_t token = 0; token < vocabulary; ++token) {
    float value = 0.0F;
    switch (pattern) {
    case Pattern::Peaked:
      value = -8.0F + 3.0F * random.unit() * random.unit();
      break;
    case Pattern::Uniform:
      value = 4.0F * random.unit();
      break;
    case Pattern::Ties:
      value = random.next() % 11 == 0
                  ? -INFINITY
                  : static_cast<float>(int(random.next() % 7)) - 3.0F;
      break;
    case Pattern::Sparse:
      value = -INFINITY;
      break;
    }
    row[token] = value;
  }
  if (pattern == Pattern::Peaked) {
    for (uint32_t spike = 0; spike < 40; ++spike)
      row[random.next() % vocabulary] = 4.0F + 6.0F * random.unit();
  }
  if (pattern == Pattern::Sparse) {
    for (uint32_t finite = 0; finite < 10; ++finite)
      row[random.next() % vocabulary] = random.unit();
  }
}

// The row's sixteen largest tokens: value descending, id ascending on ties.
std::vector<uint32_t> referenceTop16(const float *row, uint32_t vocabulary) {
  std::vector<uint32_t> order(vocabulary);
  std::iota(order.begin(), order.end(), 0U);
  const auto beats = [&](uint32_t a, uint32_t b) {
    return row[a] > row[b] || (row[a] == row[b] && a < b);
  };
  const size_t keep = std::min<size_t>(kCandidates, vocabulary);
  std::partial_sort(order.begin(), order.begin() + keep, order.end(), beats);
  order.resize(keep);
  return order;
}

struct Case final {
  uint32_t vocabulary;
  uint32_t lanes;
  bool sampling;
};

void runCase(MetalBackend &backend, const Case &c) {
  Random random(0x5e1ec7 + uint64_t{c.vocabulary} * 8 + c.lanes * 2 + c.sampling);
  const uint32_t rows = c.lanes * kRows;
  const uint32_t positions = c.lanes * kPositions;
  const auto workspace = Sampling::draftWorkspace(positions);
  Sampling sampling(c.vocabulary, kRows);

  MetalBuffer logits = allocate(backend, uint64_t{rows} * c.vocabulary * sizeof(float));
  auto *logitRows = static_cast<float *>(logits.contents());
  const std::array patterns{Pattern::Peaked, Pattern::Uniform, Pattern::Ties,
                            Pattern::Sparse};
  for (uint32_t row = 0; row < rows; ++row)
    fillRow(logitRows + uint64_t{row} * c.vocabulary, c.vocabulary,
            patterns[(row / kRows + row % kRows) % patterns.size()], random);
  DraftSelectorBuffers buffers{
      logits,
      allocate(backend, workspace.partialIdsBytes),
      allocate(backend, workspace.partialValuesBytes),
      allocate(backend, workspace.candidatesBytes),
      allocate(backend, workspace.unaryBytes),
      randomBfloat(backend, uint64_t{rows} * kRank, random, 0.1F),
      randomBfloat(backend, uint64_t{c.vocabulary} * kRank, random, 0.1F),
      randomBfloat(backend, uint64_t{c.vocabulary} * kRank, random, 0.1F),
      allocate(backend, uint64_t{c.lanes} * 2 * kRows * sizeof(float)),
      allocate(backend, uint64_t{positions} * sizeof(uint32_t)),
      allocate(backend, workspace.proposalProbabilitiesBytes)};
  auto *uniforms = static_cast<float *>(buffers.uniforms.contents());
  for (uint32_t index = 0; index < c.lanes * 2 * kRows; ++index)
    uniforms[index] = (random.unit() + 1.0F) * 0.5F;
  std::vector<uint32_t> anchors(c.lanes);
  std::vector<SamplingPolicy> policies(c.lanes);
  for (uint32_t lane = 0; lane < c.lanes; ++lane) {
    anchors[lane] = random.next() % c.vocabulary;
    policies[lane] = SamplingPolicy{16, c.sampling ? 0.8F : 0.0F, 1.0F, false};
  }

  CommandGraph graph;
  sampling.addDraftSelector(graph, buffers, anchors, policies, kPositions);
  require(graph.dispatches().size() == 3,
          "draft selector dispatch count changed");
  static_cast<void>(backend.submitCommand(graph.dispatches()));

  const auto *candidates =
      static_cast<const uint32_t *>(buffers.candidates.contents());
  const auto *unary = static_cast<const float *>(buffers.unary.contents());
  const auto *tokens =
      static_cast<const uint32_t *>(buffers.proposedTokens.contents());
  const auto *probabilities =
      static_cast<const float *>(buffers.proposalProbabilities.contents());
  const auto *hidden =
      static_cast<const uint16_t *>(buffers.selectorHidden.contents());
  const auto *predecessors =
      static_cast<const uint16_t *>(buffers.predecessorCodebook.contents());
  const auto *successors =
      static_cast<const uint16_t *>(buffers.successorCodebook.contents());

  for (uint32_t lane = 0; lane < c.lanes; ++lane) {
    uint32_t predecessor = anchors[lane];
    for (uint32_t position = 0; position < kPositions; ++position) {
      const uint32_t global = lane * kPositions + position;
      const float *row =
          logitRows + (uint64_t{lane} * kRows + position + 1) * c.vocabulary;
      const auto expected = referenceTop16(row, c.vocabulary);
      for (uint32_t rank = 0; rank < kCandidates; ++rank) {
        const uint32_t id = candidates[global * kCandidates + rank];
        const float value = unary[global * kCandidates + rank];
        if (rank < expected.size()) {
          require(id == expected[rank] && value == row[expected[rank]],
                  "draft top-16 candidates differ from the exact sorted order");
        } else {
          require(id == 0xFFFFFFFFU && value == -INFINITY,
                  "draft top-16 padding lost the empty sentinel");
        }
      }

      // Scores as the kernel defines them, in double.
      std::array<double, kCandidates> scores{};
      for (uint32_t rank = 0; rank < kCandidates; ++rank) {
        const uint32_t candidate =
            std::min(candidates[global * kCandidates + rank], c.vocabulary - 1);
        double edge = 0.0;
        for (uint32_t dim = 0; dim < kRank; ++dim) {
          edge += double(tuning::bf16ToFloat(predecessors[uint64_t{predecessor} * kRank + dim])) *
                  tuning::bf16ToFloat(hidden[(uint64_t{lane} * kRows + position + 1) * kRank + dim]) *
                  tuning::bf16ToFloat(successors[uint64_t{candidate} * kRank + dim]);
        }
        scores[rank] = double(unary[global * kCandidates + rank]) + edge;
      }
      const uint32_t token = tokens[global];
      uint32_t selected = kCandidates;
      for (uint32_t rank = 0; rank < kCandidates; ++rank)
        if (candidates[global * kCandidates + rank] == token)
          selected = selected == kCandidates ? rank : selected;
      require(selected < kCandidates,
              "draft selector proposed a token outside its candidates");
      if (c.sampling) {
        const double maximum = *std::max_element(scores.begin(), scores.end());
        double sum = 0.0;
        std::array<double, kCandidates> reference{};
        for (uint32_t rank = 0; rank < kCandidates; ++rank) {
          reference[rank] = std::exp((scores[rank] - maximum) / 0.8);
          sum += reference[rank];
        }
        const float uniform = uniforms[lane * 2 * kRows + position + 1];
        std::array<double, kCandidates> cumulative{};
        uint32_t expectedSelection = kCandidates - 1;
        for (uint32_t rank = 0; rank < kCandidates; ++rank) {
          const float probability = probabilities[global * kCandidates + rank];
          require(std::fabs(probability - reference[rank] / sum) < 1e-4,
                  "draft selector probabilities diverged from the softmax");
          cumulative[rank] = (rank ? cumulative[rank - 1] : 0.0) + reference[rank] / sum;
          if (expectedSelection == kCandidates - 1 && cumulative[rank] > uniform)
            expectedSelection = rank;
        }
        // A draw within fp32 rounding of the crossed boundary may go either way.
        const double boundary =
            std::fabs(cumulative[std::min(selected, expectedSelection)] - uniform);
        require(selected == expectedSelection || boundary < 1e-5,
                "draft selector drew a different candidate than the reference");
      } else {
        uint32_t expectedSelection = 0;
        for (uint32_t rank = 1; rank < kCandidates; ++rank)
          if (scores[rank] > scores[expectedSelection])
            expectedSelection = rank;
        // fp32 accumulation over 256 products of magnitude 1e-3 stays far
        // below this margin; only an exact tie could legitimately differ.
        require(selected == expectedSelection ||
                    std::fabs(scores[selected] - scores[expectedSelection]) < 1e-5,
                "draft selector picked a different greedy candidate");
      }
      predecessor = token;
    }
  }
}

// Target sampling buffers for a batch of lanes: every scratch and output,
// and the draft's verify inputs and candidates, which sampled verify rows
// read.
SamplingBuffers targetBuffers(MetalBackend &backend, uint32_t vocabulary,
                              uint32_t lanes) {
  const uint32_t rows = lanes * kRows;
  const auto space = Sampling::workspace(rows);
  const auto proposals = Sampling::draftWorkspace(lanes * kPositions);
  return {allocate(backend, uint64_t{rows} * vocabulary * sizeof(float)),
          allocate(backend, space.partialMassesBytes),
          allocate(backend, space.vocabularyRowsBytes),
          allocate(backend, uint64_t{lanes} * 2 * kRows * sizeof(float)),
          allocate(backend,
                   uint64_t{lanes} * (kRows + 1) * ((vocabulary + 31) / 32) * 4),
          allocate(backend, uint64_t{rows} * sizeof(uint32_t)),
          allocate(backend, space.argmaxValuesBytes),
          allocate(backend, space.argmaxIndicesBytes),
          allocate(backend, uint64_t{rows} * sizeof(uint32_t)),
          allocate(backend, proposals.candidatesBytes),
          allocate(backend, proposals.proposalProbabilitiesBytes),
          allocate(backend, space.vocabularyRangesBytes),
          allocate(backend, space.vocabularyArrivalsBytes)};
}

const TargetVocabularyRow *vocabularyRows(const SamplingBuffers &buffers) {
  return static_cast<const TargetVocabularyRow *>(
      buffers.vocabularyRows.contents());
}

// Compare every mixed policy mask against isolated lane execution: a lane's
// rows select the same, bit for bit, whatever lanes share its batch. Poison
// the outputs so a missing argmax or draw cannot pass by reading old data.
void mixedVerify(MetalBackend &backend, uint32_t lanes, uint32_t samplingMask) {
  constexpr uint32_t vocabulary = 1003;
  Sampling sampling(vocabulary, kRows);
  const auto poison = [](const SamplingBuffers &buffers) {
    for (const auto &buffer : {buffers.outputTokens, buffers.vocabularyRows})
      std::memset(buffer.contents(), 0xA5, buffer.sizeBytes());
  };
  const SamplingBuffers batch = targetBuffers(backend, vocabulary, lanes);
  poison(batch);
  Random random(9831 + lanes);
  std::vector<SamplingPolicy> policies;
  auto *logits = static_cast<float *>(batch.logits.contents());
  auto *inputs = static_cast<uint32_t *>(batch.inputTokens.contents());
  auto *candidates = static_cast<uint32_t *>(batch.draftCandidates.contents());
  auto *proposal = static_cast<float *>(batch.draftProbabilities.contents());
  auto *uniforms = static_cast<float *>(batch.uniforms.contents());
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    policies.push_back(
        {8 + lane, (samplingMask & (1U << lane)) ? 0.7F : 0.0F, 0.9F, false});
    for (uint32_t row = 0; row < kRows; ++row)
      fillRow(logits + (lane * kRows + row) * vocabulary, vocabulary,
              row % 2 ? Pattern::Ties : Pattern::Peaked, random);
  }
  for (uint32_t row = 0; row < lanes * kRows; ++row)
    inputs[row] = random.next() % vocabulary;
  for (uint32_t entry = 0; entry < lanes * kPositions * kCandidates; ++entry) {
    candidates[entry] = random.next() % vocabulary;
    proposal[entry] = 0.5F * (random.unit() + 1.0F) / kCandidates;
  }
  for (uint32_t uniform = 0; uniform < lanes * 2 * kRows; ++uniform)
    uniforms[uniform] = 0.5F * (random.unit() + 1.0F);
  const auto stops = stopTokens(vocabulary);
  CommandGraph graph;
  sampling.addVerify(graph, policies, batch, stops[0], stops[1]);
  static_cast<void>(backend.submitCommand(graph.dispatches()));
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    const SamplingBuffers single = targetBuffers(backend, vocabulary, 1);
    poison(single);
    const auto copy = [&](const MetalBuffer &from, const MetalBuffer &to) {
      std::memcpy(to.contents(),
                  static_cast<const uint8_t *>(from.contents()) +
                      lane * to.sizeBytes(),
                  to.sizeBytes());
    };
    copy(batch.logits, single.logits);
    copy(batch.inputTokens, single.inputTokens);
    copy(batch.draftCandidates, single.draftCandidates);
    copy(batch.draftProbabilities, single.draftProbabilities);
    copy(batch.uniforms, single.uniforms);
    CommandGraph reference;
    sampling.addVerify(reference, std::span(policies).subspan(lane, 1), single,
                       stops[0], stops[1]);
    static_cast<void>(backend.submitCommand(reference.dispatches()));
    if (policies[lane].samples())
      require(std::memcmp(vocabularyRows(batch) + lane * kRows,
                          vocabularyRows(single),
                          single.vocabularyRows.sizeBytes()) == 0,
              "mixed verification changed sampled rows");
    else
      require(std::memcmp(static_cast<uint32_t *>(batch.outputTokens.contents()) +
                              lane * kRows,
                          single.outputTokens.contents(),
                          single.outputTokens.sizeBytes()) == 0,
              "mixed verification changed greedy tokens");
  }
}

// Constrained greedy lanes (the argmax kernels) and constrained sampled
// lanes with top-k 1 (a distribution of one token) must agree with a
// full-vocabulary CPU argmax, including ties, row offsets and masks: the
// first token, every greedy verify row's token, and every sampled verify
// row's draw and its draft token's probability, 1 for the argmax and 0
// otherwise. Poison every scratch and output buffer, so nothing passes on
// stale data. The fp32 logits carry offsets below the bf16 spacing of their
// values, which decide the argmax among equal integer parts: reading them
// rounded would pick the lowest id instead.
void targetTop1(MetalBackend &backend, uint32_t vocabulary, uint32_t lanes) {
  const uint32_t rows = lanes * kRows;
  const uint32_t words = (vocabulary + 31) / 32;
  Sampling sampling(vocabulary, kRows);
  const SamplingBuffers b = targetBuffers(backend, vocabulary, lanes);
  auto *logits = static_cast<float *>(b.logits.contents());
  auto *masks = static_cast<uint32_t *>(b.constraintMasks.contents());
  auto *inputs = static_cast<uint32_t *>(b.inputTokens.contents());
  const auto *tokens = static_cast<const uint32_t *>(b.outputTokens.contents());
  for (uint32_t row = 0; row < rows; ++row)
    for (uint32_t token = 0; token < vocabulary; ++token)
      logits[uint64_t{row} * vocabulary + token] =
          float(int((token * 7 + row * 13) % 23) - 11) + float(token % 3) * 0x1p-12F;
  for (uint32_t row = 0; row < lanes * (kRows + 1); ++row)
    for (uint32_t token = 0; token < vocabulary; ++token)
      if ((token + row) % 17 == 0)
        masks[uint64_t{row} * words + token / 32] |= 1U << (token % 32);
  constexpr uint32_t kUnmasked = ~0U;
  auto expected = [&](uint32_t row, uint32_t maskRow) {
    float best = -INFINITY;
    uint32_t id = ~0U;
    for (uint32_t token = 0; token < vocabulary; ++token) {
      if (maskRow != kUnmasked &&
          !(masks[uint64_t{maskRow} * words + token / 32] & (1U << (token % 32))))
        continue;
      const float value = logits[uint64_t{row} * vocabulary + token];
      if (value > best) { best = value; id = token; }
    }
    return id;
  };
  auto poison = [&] {
    for (const auto &buffer : {b.argmaxValues, b.argmaxIndices, b.partialMasses,
                               b.vocabularyRows, b.outputTokens})
      std::memset(buffer.contents(), 0xA5, buffer.sizeBytes());
  };
  const auto stops = stopTokens(vocabulary);
  for (const float temperature : {0.0F, 0.8F}) {
    poison();
    CommandGraph initial;
    sampling.addInitial(initial, {1, temperature, 0.5F, true}, b, kRows - 1,
                        stops[0], stops[1]);
    static_cast<void>(backend.submitCommand(initial.dispatches()));
    require(tokens[0] == expected(kRows - 1, 0),
            "initial target differs from masked CPU argmax");
  }
  // Odd verify rows draft their argmax, even ones another token.
  for (uint32_t row = 0; row < rows; ++row) {
    if (row % kRows == kRows - 1)
      continue;
    const uint32_t maskRow = row / kRows * (kRows + 1) + row % kRows + 1;
    const uint32_t id = expected(row, maskRow);
    inputs[row + 1] = row % 2 ? id : (id + 1) % vocabulary;
  }
  poison();
  std::vector<SamplingPolicy> policies(lanes);
  for (uint32_t lane = 0; lane < lanes; ++lane)
    policies[lane] = {1, lane % 2 ? 0.8F : 0.0F, 0.5F, true};
  CommandGraph verify;
  sampling.addVerify(verify, policies, b, stops[0], stops[1]);
  static_cast<void>(backend.submitCommand(verify.dispatches()));
  for (uint32_t row = 0; row < rows; ++row) {
    const uint32_t maskRow = row / kRows * (kRows + 1) + row % kRows + 1;
    const uint32_t id = expected(row, maskRow);
    if (!policies[row / kRows].samples()) {
      require(tokens[row] == id,
              "batched target differs from masked CPU argmax");
      continue;
    }
    const TargetVocabularyRow &record = vocabularyRows(b)[row];
    require(record.token == id,
            "a sampled top-1 row drew other than its masked CPU argmax");
    if (row % kRows != kRows - 1)
      require(record.draft_probability == (inputs[row + 1] == id ? 1.0F : 0.0F),
              "a sampled top-1 row's draft probability is not one-hot");
  }
  const SamplingPolicy greedy{1, 0.0F, 1.0F, false};
  poison();
  CommandGraph initialArgmax;
  sampling.addInitial(initialArgmax, greedy, b, kRows - 1, stops[0], stops[1]);
  static_cast<void>(backend.submitCommand(initialArgmax.dispatches()));
  require(tokens[0] == expected(kRows - 1, kUnmasked),
          "initial argmax differs from CPU argmax");
  poison();
  CommandGraph verifyArgmax;
  sampling.addVerify(verifyArgmax, std::vector<SamplingPolicy>(lanes, greedy), b,
                     stops[0], stops[1]);
  static_cast<void>(backend.submitCommand(verifyArgmax.dispatches()));
  for (uint32_t row = 0; row < rows; ++row)
    require(tokens[row] == expected(row, kUnmasked),
            "batched argmax differs from CPU argmax");
}

// A lane that ignores end-of-sequence never selects a stop token: not in the
// sample after a prefill chunk, not in any verify row, greedy or sampled,
// alone or batched beside lanes that keep them, and a stop token its draft
// proposes is rejected. The stop tokens lead every row, so a lane that keeps
// them selects one; the other lanes take the row's best remaining token.
void excludedStopTokens(MetalBackend &backend, uint32_t vocabulary) {
  const uint32_t rows = kLanes * kRows;
  const auto stops = stopTokens(vocabulary);
  Sampling sampling(vocabulary, kRows);
  const SamplingBuffers b = targetBuffers(backend, vocabulary, kLanes);
  const auto best = [&](uint32_t row) { return 5 + row * 7919 % (vocabulary - 8); };
  auto *logits = static_cast<float *>(b.logits.contents());
  for (uint32_t row = 0; row < rows; ++row) {
    float *values = logits + uint64_t{row} * vocabulary;
    for (uint32_t token = 0; token < vocabulary; ++token)
      values[token] = -4.0F + float(token % 7) * 0.01F;
    values[best(row)] = 8.0F;
    values[stops[0]] = 12.0F;
    values[stops[1]] = 11.0F;
  }
  auto *uniforms = static_cast<float *>(b.uniforms.contents());
  const auto *tokens = static_cast<const uint32_t *>(b.outputTokens.contents());
  const auto stop = [&](uint32_t token) {
    return token == stops[0] || token == stops[1];
  };
  auto poison = [&] {
    for (const auto &buffer : {b.partialMasses, b.vocabularyRows, b.outputTokens})
      std::memset(buffer.contents(), 0xA5, buffer.sizeBytes());
  };

  for (const bool excludes : {false, true}) {
    for (const float temperature : {0.0F, 0.8F}) {
      for (const float uniform : {0.0F, 0.5F, 0.999F}) {
        poison();
        uniforms[0] = uniform;
        CommandGraph initial;
        sampling.addInitial(initial, {32, temperature, 1.0F, false, excludes}, b,
                            kRows - 1, stops[0], stops[1]);
        static_cast<void>(backend.submitCommand(initial.dispatches()));
        if (excludes)
          require(tokens[0] == best(kRows - 1) ||
                      (temperature > 0.0F && tokens[0] < vocabulary &&
                       !stop(tokens[0])),
                  "an excluding lane selected a stop token after prefill");
        else
          require(stop(tokens[0]), "the initial sample skipped the stop tokens");
      }
    }
  }

  // Drafts propose the first stop token, with all of their mass on it; it
  // is verify input row 1 of each lane.
  AcceptanceBuffers acceptance{
      allocate(backend, uint64_t{kLanes} * kPositions * sizeof(uint32_t)),
      b.draftCandidates,
      b.draftProbabilities,
      b.vocabularyRows,
      b.uniforms,
      b.outputTokens,
      allocate(backend, kLanes * sizeof(uint32_t)),
      allocate(backend, kLanes * sizeof(uint32_t)),
      allocate(backend, kLanes * sizeof(uint32_t))};
  for (uint32_t lane = 0; lane < kLanes; ++lane) {
    static_cast<uint32_t *>(acceptance.proposedTokens.contents())[lane * kPositions] =
        stops[0];
    static_cast<uint32_t *>(b.inputTokens.contents())[lane * kRows + 1] = stops[0];
    static_cast<uint32_t *>(acceptance.candidates.contents())
        [uint64_t{lane} * kPositions * kCandidates] = stops[0];
    static_cast<float *>(acceptance.proposalProbabilities.contents())
        [uint64_t{lane} * kPositions * kCandidates] = 1.0F;
  }
  std::fill(uniforms, uniforms + kLanes * 2 * kRows, 0.5F);
  const std::array<uint32_t, kLanes> maximumRetained{kRows, kRows, kRows, kRows};
  for (uint32_t lanes = 1; lanes <= kLanes; ++lanes) {
    for (uint32_t excludeMask = 0; excludeMask < (1U << lanes); ++excludeMask) {
      for (const uint32_t samplingMask : {0U, 0b0101U, 0b1111U}) {
        std::vector<SamplingPolicy> policies;
        for (uint32_t lane = 0; lane < lanes; ++lane)
          policies.push_back({32, (samplingMask >> lane & 1U) ? 0.8F : 0.0F,
                              1.0F, false, (excludeMask >> lane & 1U) != 0});
        poison();
        CommandGraph verify;
        sampling.addVerify(verify, policies, b, stops[0], stops[1]);
        static_cast<void>(backend.submitCommand(verify.dispatches()));
        // Only an excluding lane's distribution leaves out the stop token
        // drafted at row 0, and none of its rows draws a stop token.
        for (uint32_t row = 0; row < lanes * kRows; ++row) {
          const SamplingPolicy &policy = policies[row / kRows];
          if (!policy.samples()) {
            require(tokens[row] ==
                        (policy.excludesStopTokens ? best(row) : stops[0]),
                    "a greedy verify row mishandled the stop tokens");
            continue;
          }
          const TargetVocabularyRow &record = vocabularyRows(b)[row];
          require(row % kRows != 0 || (record.draft_probability > 0.0F) !=
                                          policy.excludesStopTokens,
                  "a verify distribution mishandled the stop tokens");
          require(!policy.excludesStopTokens ||
                      (record.token < vocabulary && !stop(record.token)),
                  "an excluding verify row drew a stop token");
        }
        CommandGraph accept;
        sampling.addAcceptance(accept, acceptance,
                               std::span(maximumRetained).first(lanes), policies,
                               stops[0], stops[1]);
        static_cast<void>(backend.submitCommand(accept.dispatches()));
        const auto *retained =
            static_cast<const uint32_t *>(acceptance.retainedCounts.contents());
        const auto *anchors =
            static_cast<const uint32_t *>(acceptance.nextAnchors.contents());
        for (uint32_t lane = 0; lane < lanes; ++lane) {
          const uint32_t *output = tokens + lane * kRows;
          if (!policies[lane].excludesStopTokens) {
            require(retained[lane] == 1 && output[0] == stops[0] &&
                        anchors[lane] == stops[0],
                    "the target did not accept the drafted stop token");
            continue;
          }
          require(retained[lane] >= 1 && retained[lane] <= kRows &&
                      !stop(anchors[lane]),
                  "an excluding lane accepted a stop token");
          for (uint32_t index = 0; index < retained[lane]; ++index)
            require(output[index] < vocabulary && !stop(output[index]),
                    "an excluding lane accepted a stop token");
        }
      }
    }
  }
}

void invalidRequests(MetalBackend &backend) {
  Sampling sampling(1024, kRows);
  const auto workspace = Sampling::draftWorkspace(kPositions);
  DraftSelectorBuffers buffers{
      allocate(backend, uint64_t{kRows} * 1024 * sizeof(float)),
      allocate(backend, workspace.partialIdsBytes),
      allocate(backend, workspace.partialValuesBytes),
      allocate(backend, workspace.candidatesBytes),
      allocate(backend, workspace.unaryBytes),
      allocate(backend, uint64_t{kRows} * kRank * 2),
      allocate(backend, uint64_t{1024} * kRank * 2),
      allocate(backend, uint64_t{1024} * kRank * 2),
      allocate(backend, 2 * kRows * sizeof(float)),
      allocate(backend, kPositions * sizeof(uint32_t)),
      allocate(backend, workspace.proposalProbabilitiesBytes)};
  const std::array<uint32_t, 2> anchors{1, 2};
  const std::array<SamplingPolicy, 1> policies{SamplingPolicy{}};
  CommandGraph graph;
  rejects([&] {
    sampling.addDraftSelector(graph, buffers, anchors, policies, kPositions);
  });
  // The kernels compile the proposal count and a lane's rows in.
  for (const uint32_t proposals : {0U, kPositions - 1, kPositions + 1})
    rejects([&] {
      sampling.addDraftSelector(graph, buffers, std::span(anchors).first(1),
                                policies, proposals);
    });
  for (const uint32_t rows : {0U, kRows - 1, kRows + 1})
    rejects([&] { (void)Sampling(1024, rows); });
  require(graph.empty(), "invalid draft selector request encoded a graph");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("usage: draft-selector METALLIB");
    MetalBackend backend(argv[1]);
    invalidRequests(backend);
    for (const uint32_t vocabulary : {1003U, 248320U}) {
      for (uint32_t lanes = 1; lanes <= kLanes; ++lanes)
        targetTop1(backend, vocabulary, lanes);
      excludedStopTokens(backend, vocabulary);
    }
    for (uint32_t lanes = 1; lanes <= kLanes; ++lanes)
      for (uint32_t mask = 0; mask < (1U << lanes); ++mask)
        mixedVerify(backend, lanes, mask);
    for (const uint32_t vocabulary : {248320U, 1003U, 270005U}) {
      for (uint32_t lanes = 1; lanes <= kLanes; ++lanes) {
        runCase(backend, {vocabulary, lanes, false});
        runCase(backend, {vocabulary, lanes, true});
      }
    }
    std::cout << "draft_selector_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "draft_selector_metal_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
