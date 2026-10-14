#include "Qwen3_6Moe.hpp"
#include "model/QwenTargetLoader.hpp"

#include <utility>

namespace splash::model {
namespace {

// Block images keep the routed experts of each projection as one tensor, in
// the layer's experts image (`experts`), the shared expert's as another, and
// the router and the shared-expert scalar gate in F32, in the layer's.
void readFfn(WeightFile &file, WeightFile &experts, Qwen3_6MoeLayerWeights &layer) {
  ops::BlockMoeWeights ffn;
  ffn.router = readQuantizedSegment(file, "router");
  ffn.gate.routed = readQuantizedSegment(experts, "experts-gate");
  ffn.up.routed = readQuantizedSegment(experts, "experts-up");
  ffn.down.routed = readQuantizedSegment(experts, "experts-down");
  ffn.gate.shared = readQuantizedSegment(file, "shared-expert-gate");
  ffn.up.shared = readQuantizedSegment(file, "shared-expert-up");
  ffn.down.shared = readQuantizedSegment(file, "shared-expert-down");
  ffn.sharedScalarGate = readQuantizedSegment(file, "shared-expert-scalar-gate");
  layer.ffn = std::move(ffn);
}

} // namespace

Qwen3_6MoeWeights
loadQwen3_6MoeWeights(metal::MetalBackend &backend, Qwen3_6MoeLayout layout,
                      const QwenTargetFiles &files) {
  return loadQwenTarget<Qwen3_6MoeWeights>(
      backend, layout, files,
      [](WeightFile &file, WeightFile &experts, Qwen3_6MoeLayerWeights &layer, const BlockTargetFormat &) {
        readFfn(file, experts, layer);
      });
}

} // namespace splash::model
