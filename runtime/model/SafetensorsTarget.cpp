#include "model/SafetensorsTarget.hpp"
#include "model/GgufPreparation.hpp"
#include "model/SafetensorsImage.hpp"

namespace splash::model {

SafetensorsTargetLoader::SafetensorsTargetLoader(metal::MetalBackend &backend, WeightImages &images,
                                                 const std::filesystem::path &directory,
                                                 const QwenTargetDimensions &geometry, ExpertStoreOptions experts)
    : backend_(backend), images_(images), planned_(std::make_shared<Planned>(directory)),
      layerImages_(gguf::imagesPerLayer(geometry)), experts_(experts) {
  planned_->images =
      safetensors::planImages(planned_->checkpoint, geometry, {.streamExperts = experts.memoryBytes != 0});
  float32Norms_ = safetensors::float32Norms(safetensors::moduleNames(planned_->checkpoint));
}

StreamedBlocks SafetensorsTargetLoader::streamedExperts() const {
  return {gguf::streamedExperts(std::span(planned_->images).first(planned_->images.size() - 2)), planned_,
          {[planned = planned_] { planned->checkpoint.checkUnchanged(); }}};
}

WeightFile SafetensorsTargetLoader::open(size_t index) {
  const gguf::Image &plan = planned_->images[index];
  gguf::Description description = gguf::describe(plan);
  return images_.load({"target/" + plan.name, plan.magic, plan.layer, plan.type, plan.bytes,
                       [&backend = backend_, planned = planned_, index](std::span<uint8_t>,
                                                                         const metal::MetalBuffer &buffer) {
                         writeGgufImage(backend, buffer, planned->images[index]);
                         planned->checkpoint.checkUnchanged();
                       },
                       plan.residentBytes, std::move(description.text), std::move(description.sources)});
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

} // namespace splash::model
