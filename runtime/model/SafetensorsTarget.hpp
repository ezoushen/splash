#pragma once

// Source adapter for a safetensors target (MLX's, Model Optimizer's or
// compressed-tensors'): its block images (model/SafetensorsImage.hpp) are
// written into memory, and QwenTargetLoader reads them as block-quantized
// weights (BlockTargetFormat).

#include "model/GgufImage.hpp"
#include "model/QwenHybridLayout.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/WeightImages.hpp"

#include <filesystem>
#include <memory>
#include <vector>

namespace splash::model {

class SafetensorsTargetLoader final {
public:
  // Plans every image of the checkpoint in directory once.
  SafetensorsTargetLoader(metal::MetalBackend &backend, WeightImages &images, const std::filesystem::path &directory,
                          const QwenTargetDimensions &geometry);
  SafetensorsTargetLoader(const SafetensorsTargetLoader &) = delete;
  SafetensorsTargetLoader &operator=(const SafetensorsTargetLoader &) = delete;

  [[nodiscard]] WeightFile layer(uint32_t index);
  // A sparse MoE layer's routed experts, which its layer image does not hold.
  [[nodiscard]] WeightFile experts(uint32_t index);
  [[nodiscard]] WeightFile head();
  [[nodiscard]] WeightFile embedding();
  // The plan of the image at `index` among those planned, in load order: each
  // layer's, followed by its routed experts' for a sparse MoE, then the head's
  // and the embedding's.
  [[nodiscard]] ImagePlan image(size_t index) const;
  [[nodiscard]] size_t imageCount() const noexcept { return planned_->images.size(); }
  // Whether its images hold F32 norms (safetensors::float32Norms).
  [[nodiscard]] bool float32Norms() const noexcept { return float32Norms_; }

private:
  // The checkpoint and its images, which their writers share.
  struct Planned {
    explicit Planned(const std::filesystem::path &directory) : checkpoint(directory) {}
    SafetensorsCheckpoint checkpoint;
    std::vector<gguf::Image> images; // as safetensors::planImages orders them
  };
  [[nodiscard]] WeightFile open(size_t index);
  // Image `part` of target layer `index`: the layer's, or its experts'.
  [[nodiscard]] WeightFile openLayer(uint32_t index, size_t part);

  metal::MetalBackend &backend_;
  WeightImages &images_;
  std::shared_ptr<Planned> planned_;
  size_t layerImages_; // images per layer: gguf::imagesPerLayer
  bool float32Norms_ = false;
};

// The bytes of every image of the safetensors target in directory.
[[nodiscard]] uint64_t safetensorsTargetImageBytes(const std::filesystem::path &directory,
                                                   const QwenTargetDimensions &geometry);

} // namespace splash::model
