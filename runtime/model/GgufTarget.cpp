#include "model/GgufTarget.hpp"
#include "model/GgufPreparation.hpp"

#include <sstream>

namespace splash::model {
namespace {

// Describes a planned image for a shared image's key (ImagePlan): its fills'
// bytes, every field of its copies and repacks, and the rows they read, in
// plan order.
void describe(const WeightSource &source, const gguf::Image &image, ImagePlan &plan) {
  std::ostringstream text;
  const auto rows = [&](const gguf::TensorRows &rows) {
    text << " rows " << rows.name << ' ' << rows.type << ' ' << rows.rows << ' ' << rows.rowBytes << ' '
         << rows.order.from << ' ' << rows.order.headRows << ' ' << rows.order.keyHeads << ' '
         << rows.order.valueHeadsPerKey << '\n';
    plan.sources.push_back({&source, rows.offset, rows.rows * rows.rowBytes});
  };
  text << "gguf\n";
  for (const gguf::Fill &fill : image.fills) text << "fill " << fill.offset << ' ' << weightDigest(fill.bytes) << '\n';
  for (const gguf::Copy &copy : image.copies) {
    text << "copy " << copy.destination << ' ' << static_cast<int>(copy.conversion) << '\n';
    rows(copy.source);
  }
  for (const gguf::Repack &repack : image.repacks) {
    text << "repack " << repack.format << ' ' << repack.rows << ' ' << repack.columns << ' ' << repack.plane0 << ' '
         << repack.plane1 << ' ' << repack.meta << '\n';
    for (const gguf::TensorRows &source : repack.sources) rows(source);
  }
  plan.description = text.str();
}

} // namespace
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
                                   const std::filesystem::path &path, const gguf::TargetGeometry &geometry)
    : backend_(backend), images_(images), planned_(std::make_shared<Planned>(path)) {
  const GgufFile file(planned_->source);
  rotation_ = file.rotation();
  planned_->images = gguf::planImages(file, geometry);
}

WeightFile GgufTargetLoader::open(size_t index) {
  const gguf::Image &plan = planned_->images[index];
  ImagePlan image{"target/" + plan.name, plan.magic, plan.layer, plan.type, plan.bytes,
                  [&backend = backend_, planned = planned_, index](std::span<uint8_t>,
                                                                    const metal::MetalBuffer &buffer) {
                    writeGgufImage(backend, planned->source, buffer, planned->images[index]);
                    planned->source.checkUnchanged();
                  }};
  describe(planned_->source, plan, image);
  return images_.load(std::move(image));
}

WeightFile GgufTargetLoader::layer(uint32_t index) {
  if (index >= planned_->images.size() - 2) throw GgufError("target layer is out of range");
  return open(index);
}

WeightFile GgufTargetLoader::head() { return open(planned_->images.size() - 2); }

WeightFile GgufTargetLoader::embedding() { return open(planned_->images.size() - 1); }

} // namespace splash::model
