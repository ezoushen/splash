#include "metal/MetalBackend.hpp"
#include "metal/abi/Sampling.h"
#include "model/Model.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using splash::metal::BufferStorage;
using splash::metal::ComputeDispatch;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;

constexpr uint32_t kRows = splash::model::ExecutionLimits::targetVerifyRows;
constexpr uint32_t kProposals =
    splash::model::ExecutionLimits::draftProposalTokens;
constexpr uint32_t kLanes =
    splash::model::ExecutionLimits::maximumBatchWidth;

MetalBuffer shared(MetalBackend &backend, uint64_t bytes, const char *label) {
  return backend.allocateBuffer(bytes, BufferStorage::Shared, label);
}

template <class T> T *contents(const MetalBuffer &buffer) {
  return static_cast<T *>(buffer.contents());
}

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

void runWidth(MetalBackend &backend, uint32_t width,
              const std::array<uint32_t, kLanes> &acceptedReference,
              const std::array<uint32_t, kLanes> &remainingReference,
              uint32_t stopLane = kLanes, uint32_t stopRow = 0) {
  require(width >= 1 && width <= kLanes, "invalid test width");
  MetalBuffer draft = shared(backend, kLanes * kProposals * sizeof(uint32_t),
                             "accept-draft");
  MetalBuffer draftIds =
      shared(backend, kLanes * kProposals * 16 * sizeof(uint32_t),
             "accept-draft-ids");
  MetalBuffer draftProbabilities =
      shared(backend, kLanes * kProposals * 16 * sizeof(float),
             "accept-draft-probabilities");
  // Every lane is greedy here; sampled lanes read their target rows.
  MetalBuffer targetRows =
      shared(backend, kLanes * kRows * sizeof(TargetVocabularyRow),
             "accept-target-rows");
  MetalBuffer uniforms =
      shared(backend, kLanes * 2 * kRows * sizeof(float), "accept-uniforms");
  MetalBuffer output = shared(backend, kLanes * kRows * sizeof(uint32_t),
                              "accept-output");
  MetalBuffer retained =
      shared(backend, kLanes * sizeof(uint32_t), "accept-retained");
  MetalBuffer next =
      shared(backend, kLanes * sizeof(uint32_t), "accept-next");
  MetalBuffer accepted =
      shared(backend, kLanes * sizeof(uint32_t), "accept-count");

  auto *draftTokens = contents<uint32_t>(draft);
  auto *targetTokens = contents<uint32_t>(output);
  std::memset(draftIds.contents(), 0, draftIds.sizeBytes());
  std::memset(draftProbabilities.contents(), 0,
              draftProbabilities.sizeBytes());
  std::memset(targetRows.contents(), 0, targetRows.sizeBytes());
  std::memset(uniforms.contents(), 0, uniforms.sizeBytes());
  std::memset(retained.contents(), 0, retained.sizeBytes());
  std::memset(next.contents(), 0, next.sizeBytes());
  std::memset(accepted.contents(), 0, accepted.sizeBytes());
  for (uint32_t lane = 0; lane < kLanes; ++lane) {
    for (uint32_t token = 0; token < kProposals; ++token) {
      const uint32_t proposal = 1000 + lane * 100 + token;
      draftTokens[lane * kProposals + token] = proposal;
      targetTokens[lane * kRows + token] =
          token < acceptedReference[lane] ? proposal : proposal + 50;
    }
    targetTokens[lane * kRows + kRows - 1] = 9000 + lane;
  }

  constexpr uint32_t kStopToken = 248044;
  if (stopLane < width) {
    require(stopRow < kRows, "invalid stop row");
    targetTokens[stopLane * kRows + stopRow] = kStopToken;
    if (stopRow < acceptedReference[stopLane])
      draftTokens[stopLane * kProposals + stopRow] = kStopToken;
  }

  AcceptBatchParams params{};
  std::copy(remainingReference.begin(), remainingReference.end(),
            std::begin(params.remaining));
  params.stop_token_0 = kStopToken;
  params.stop_token_1 = 248046;
  params.lanes = width;
  ComputeDispatch dispatch;
  dispatch.pipelineName = "decode_accept_dflash";
  dispatch.buffers = {{0, draft},
                      {1, draftIds},
                      {2, draftProbabilities},
                      {3, targetRows},
                      {4, uniforms},
                      {5, output},
                      {6, retained},
                      {7, next},
                      {8, accepted}};
  dispatch.bytes = {{9, &params, sizeof(params)}};
  dispatch.threadgroups = {width, 1, 1};
  dispatch.threadsPerThreadgroup = {1, 1, 1};
  static_cast<void>(backend.submit(dispatch));

  const auto *retainedCounts = contents<uint32_t>(retained);
  const auto *nextTokens = contents<uint32_t>(next);
  const auto *acceptedCounts = contents<uint32_t>(accepted);
  for (uint32_t lane = 0; lane < width; ++lane) {
    uint32_t expectedRetained =
        std::min(acceptedReference[lane] + 1, remainingReference[lane]);
    if (lane == stopLane && stopRow < expectedRetained)
      expectedRetained = stopRow + 1;
    require(acceptedCounts[lane] == acceptedReference[lane],
            "accepted proposal count mismatch");
    require(retainedCounts[lane] == expectedRetained,
            "retained token count mismatch");
    require(nextTokens[lane] ==
                targetTokens[lane * kRows + expectedRetained - 1],
            "next anchor mismatch");
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("usage: dflash-batch-control METALLIB");
    MetalBackend backend(argv[1]);
    constexpr std::array<uint32_t, kLanes> lowerAccepted{0, 1, 2, 3};
    constexpr std::array<uint32_t, kLanes> upperAccepted{4, 5, 6, 7};
    constexpr std::array<uint32_t, kLanes> fullRemaining{8, 8, 8, 8};
    for (uint32_t width = 1; width <= kLanes; ++width) {
      runWidth(backend, width, lowerAccepted, fullRemaining);
      runWidth(backend, width, upperAccepted, fullRemaining);
    }

    // Output limits and stop tokens shorten the committed prefix without
    // changing the physical eight-row graph.  The accepted proposal count is
    // still seven in every lane; only retained rows and the next anchor move.
    constexpr std::array<uint32_t, kLanes> allAccepted{7, 7, 7, 7};
    constexpr std::array<uint32_t, kLanes> shortRemaining{1, 2, 3, 8};
    runWidth(backend, kLanes, allAccepted, shortRemaining, 3, 3);
    std::cout << "dflash_batch_control_metal_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "dflash_batch_control_metal_test: FAIL: " << error.what()
              << '\n';
    return 1;
  }
}
