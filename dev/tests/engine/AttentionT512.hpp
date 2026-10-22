#pragma once

#include "metal/MetalBackend.hpp"

#include <span>
#include <vector>

namespace splash::test {

// Select the 512-thread GQA-16 prefill split in a store/split/reduce graph.
// The returned dispatches borrow the input graph's parameter storage.
inline std::vector<metal::ComputeDispatch> prefillT512Dispatches(
    std::span<const metal::ComputeDispatch> source) {
  std::vector<metal::ComputeDispatch> result(source.begin(), source.end());
  for (auto &dispatch : result) {
    if (dispatch.pipelineName == "prefill_attention_q8_split_kv2_g16" ||
        dispatch.pipelineName == "prefill_attention_bf16_split_kv2_g16") {
      dispatch.pipelineName += "_t512";
      dispatch.threadsPerThreadgroup = {512, 1, 1};
    }
  }
  return result;
}

} // namespace splash::test
