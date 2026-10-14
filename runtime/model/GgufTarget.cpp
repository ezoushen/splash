#include "model/GgufTarget.hpp"
#include "model/GgufPreparation.hpp"

namespace splash::model {
std::filesystem::path findTargetGguf(const std::filesystem::path &directory) {
  std::filesystem::path found;
  std::error_code error;
  for (const auto &entry : std::filesystem::directory_iterator(directory, error)) {
    if (entry.path().extension() != ".gguf") continue;
    if (!found.empty()) throw GgufError("target directory holds more than one GGUF: " + directory.string());
    found = entry.path();
  }
  if (error) throw GgufError("cannot list target directory: " + directory.string());
  if (found.empty()) throw GgufError("target directory holds no GGUF: " + directory.string());
  return found;
}

GgufTargetLoader::GgufTargetLoader(metal::MetalBackend &backend, WeightImages &images,
                                   const std::filesystem::path &path, const QwenTargetDimensions &geometry)
    : backend_(backend), images_(images), planned_(std::make_shared<Planned>(path)),
      layerImages_(gguf::imagesPerLayer(geometry)) {
  const GgufFile file(planned_->source);
  rotation_ = file.rotation();
  planned_->images = gguf::planImages(file, geometry);
}

ImagePlan GgufTargetLoader::image(size_t index) const {
  const gguf::Image &plan = planned_->images.at(index);
  return {"target/" + plan.name, plan.magic, plan.layer, plan.type, plan.bytes,
          [&backend = backend_, planned = planned_, index](std::span<uint8_t>, const metal::MetalBuffer &buffer) {
            writeGgufImage(backend, buffer, planned->images[index]);
            planned->source.checkUnchanged();
          }};
}

WeightFile GgufTargetLoader::open(size_t index) { return images_.load(image(index)); }

WeightFile GgufTargetLoader::openLayer(uint32_t index, size_t part) {
  if (index >= (planned_->images.size() - 2) / layerImages_) throw GgufError("target layer is out of range");
  return open(index * layerImages_ + part);
}

WeightFile GgufTargetLoader::layer(uint32_t index) { return openLayer(index, 0); }

WeightFile GgufTargetLoader::experts(uint32_t index) {
  if (layerImages_ != 2) throw GgufError("a dense target has no experts image");
  return openLayer(index, 1);
}

WeightFile GgufTargetLoader::head() { return open(planned_->images.size() - 2); }

WeightFile GgufTargetLoader::embedding() { return open(planned_->images.size() - 1); }

} // namespace splash::model
