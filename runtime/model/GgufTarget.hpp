#pragma once

// Source adapter for a Qwen GGUF: its images (model/GgufImageLayout.hpp) are
// written into memory, and QwenTargetLoader reads them as block-quantized
// weights (BlockTargetFormat).

#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

#include "model/GgufFile.hpp"
#include "model/GgufImage.hpp"
#include "model/QwenHybridLayout.hpp"
#include "model/WeightImages.hpp"

namespace splash::model {

// The single .gguf in a target directory (shards are not supported).
[[nodiscard]] std::filesystem::path findTargetGguf(const std::filesystem::path &directory);

class GgufTargetLoader final {
public:
  // Plans every image from the GGUF's metadata once.
  GgufTargetLoader(metal::MetalBackend &backend, WeightImages &images, const std::filesystem::path &path,
                   const QwenTargetDimensions &geometry);
  GgufTargetLoader(const GgufTargetLoader &) = delete;
  GgufTargetLoader &operator=(const GgufTargetLoader &) = delete;

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
  // The input rotation of a Prism ML GGUF, which planImages checked names
  // every quantized tensor of the target and its token table.
  [[nodiscard]] const std::optional<GgufRotation> &rotation() const noexcept { return rotation_; }

private:
  // The GGUF and its images, which their writers share.
  struct Planned {
    explicit Planned(const std::filesystem::path &path) : source(path) {}
    WeightSource source;
    std::vector<gguf::Image> images; // as gguf::planImages orders them
  };
  [[nodiscard]] WeightFile open(size_t index);
  // Image `part` of target layer `index`: the layer's, or its experts'.
  [[nodiscard]] WeightFile openLayer(uint32_t index, size_t part);

  metal::MetalBackend &backend_;
  WeightImages &images_;
  std::shared_ptr<Planned> planned_;
  size_t layerImages_; // images per layer: gguf::imagesPerLayer
  std::optional<GgufRotation> rotation_;
};

} // namespace splash::model
