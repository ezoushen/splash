#include "model/SafetensorsTarget.hpp"
#include "model/GgufPreparation.hpp"
#include "model/SafetensorsImage.hpp"

namespace splash::model {

SafetensorsTargetLoader::SafetensorsTargetLoader(metal::MetalBackend &backend, WeightImages &images,
                                                 const std::filesystem::path &directory,
                                                 const QwenTargetDimensions &geometry)
    : backend_(backend), images_(images), planned_(std::make_shared<Planned>(directory)),
      layerImages_(gguf::imagesPerLayer(geometry)) {
  planned_->images = safetensors::planImages(planned_->checkpoint, geometry);
  float32Norms_ = safetensors::float32Norms(safetensors::moduleNames(planned_->checkpoint));
}

WeightFile SafetensorsTargetLoader::open(size_t index) {
  const gguf::Image &plan = planned_->images[index];
  return images_.load({"target/" + plan.name, plan.magic, plan.layer, plan.type, plan.bytes,
                       [&backend = backend_, planned = planned_, index](std::span<uint8_t>,
                                                                         const metal::MetalBuffer &buffer) {
                         writeGgufImage(backend, buffer, planned->images[index]);
                         planned->checkpoint.checkUnchanged();
                       }});
}

WeightFile SafetensorsTargetLoader::openLayer(uint32_t index, size_t part) {
  if (index >= (planned_->images.size() - 2) / layerImages_) throw WeightStoreError("target layer is out of range");
  return open(index * layerImages_ + part);
}

WeightFile SafetensorsTargetLoader::layer(uint32_t index) { return openLayer(index, 0); }

WeightFile SafetensorsTargetLoader::experts(uint32_t index) {
  if (layerImages_ != 2) throw WeightStoreError("a dense target has no experts image");
  return openLayer(index, 1);
}

WeightFile SafetensorsTargetLoader::head() { return open(planned_->images.size() - 2); }

WeightFile SafetensorsTargetLoader::embedding() { return open(planned_->images.size() - 1); }

uint64_t safetensorsTargetImageBytes(const std::filesystem::path &directory, const QwenTargetDimensions &geometry) {
  const SafetensorsCheckpoint checkpoint(directory);
  uint64_t bytes = 0;
  for (const gguf::Image &image : safetensors::planImages(checkpoint, geometry)) bytes += image.bytes;
  return bytes;
}

} // namespace splash::model
