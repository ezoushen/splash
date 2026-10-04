#pragma once

#include "model/AffinePreparation.hpp"
#include "model/WeightImages.hpp"

#include <filesystem>
#include <memory>
#include <vector>

namespace splash::model {

struct Qwen3_8Layout;
struct Qwen3_6MoeLayout;

namespace affine {
struct PlannedCheckpoint;
}

// Native MLX affine source -> the existing packed target ABI; neither this
// adapter nor the block-quantized one changes inference kernels. The
// checkpoint is planned once; each image is written into memory when it is
// opened.
class AffineTargetLoader final {
public:
  AffineTargetLoader(WeightImages &images, const std::filesystem::path &directory, const Qwen3_8Layout &layout);
  AffineTargetLoader(WeightImages &images, const std::filesystem::path &directory, const Qwen3_6MoeLayout &layout);
  ~AffineTargetLoader();
  [[nodiscard]] WeightFile layer(uint32_t index);
  // A sparse MoE layer's routed experts, which its layer image does not hold.
  [[nodiscard]] WeightFile experts(uint32_t index);
  [[nodiscard]] WeightFile head();
  [[nodiscard]] WeightFile embedding();
private:
  // Image `part` of target layer `index`: the layer's, or its experts'.
  [[nodiscard]] WeightFile openLayer(uint32_t index, size_t part);
  WeightImages &images_;
  std::shared_ptr<affine::PlannedCheckpoint> planned_; // as affineTargetImages plans them
  size_t layerImages_; // images per layer: 2 for a sparse MoE layer, else 1
};

// Every planned image of a layout, its sections at their offsets: each layer,
// a sparse MoE layer followed by its routed experts, then the head and the
// embedding.
[[nodiscard]] std::vector<affine::Image> affineTargetImages(const Qwen3_8Layout &layout);
[[nodiscard]] std::vector<affine::Image> affineTargetImages(const Qwen3_6MoeLayout &layout);

} // namespace splash::model
