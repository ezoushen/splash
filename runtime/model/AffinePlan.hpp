#pragma once

// Plans the affine images (AffinePreparation.hpp) of a safetensors checkpoint:
// a header block, then 16 KiB-aligned sections, each bound to the checkpoint
// tensors it reads. AffineTarget.cpp plans an MLX target with it,
// DraftCheckpoint.cpp a DFlash2 draft.

#include "Checked.hpp"
#include "model/AffinePreparation.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/WeightImages.hpp"
#include "model/WeightLayout.hpp"
#include "model/WeightStore.hpp"

#include <algorithm>
#include <filesystem>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace splash::model::affine {

// An image of a header block; append places each section after it.
[[nodiscard]] inline Image image(std::string name, std::string_view magic, uint32_t layer, uint32_t type) {
  return {std::move(name), std::string(magic), layer, type, kWeightFileAlignment, {}, {}};
}

inline void append(Image &image, Section section) {
  section.offset = image.bytes;
  image.bytes = alignWeightOffset(image.bytes + section.bytes);
  image.sections.push_back(std::move(section));
}

// A tensor copied as stored, BF16 or U32.
inline void copy(Image &image, const std::string &name, std::vector<uint64_t> shape,
                 const std::string &dtype = "BF16") {
  Section section;
  section.bytes = dtype == "U32" ? 4 : kBFloat16Bytes;
  for (uint64_t dimension : shape)
    section.bytes = checkedMultiply<WeightStoreError>(section.bytes, dimension, "affine tensor");
  section.input = {name, {dtype}, std::move(shape)};
  append(image, std::move(section));
}

// Binds every input of image to its checkpoint tensor, which must have one of
// the input's dtypes and its shape, once the checkpoint states the
// quantization of every affine module the image reads.
inline void bind(Image &image, const SafetensorsCheckpoint &source) {
  const auto bindInput = [&](Input &input) {
    const SourceTensor &tensor = source.require(input.name);
    if (std::find(input.dtypes.begin(), input.dtypes.end(), tensor.dtype) == input.dtypes.end() ||
        tensor.shape != input.shape)
      throw WeightStoreError("source tensor type or shape does not match: " + input.name);
    input.tensor = &tensor;
  };
  for (const auto &[module, bits] : image.quantized) source.requireQuantization(module, bits);
  for (Section &section : image.sections) {
    if (section.parts.empty()) bindInput(section.input);
    for (ProjectionPart &part : section.parts)
      for (Input &field : part.fields) bindInput(field);
  }
}

// A checkpoint and the images planned from it, which their writers share:
// each image's inputs point into the checkpoint's tensors.
struct PlannedCheckpoint final {
  explicit PlannedCheckpoint(const std::filesystem::path &directory) : source(directory) {}
  SafetensorsCheckpoint source;
  std::vector<Image> images;
};

// Describes a bound image for a shared image's key (ImagePlan): every field
// of its sections and of the tensors they read, and those tensors' bytes in
// section order.
inline void describe(const Image &image, ImagePlan &plan) {
  std::ostringstream text;
  const auto input = [&](const Input &input) {
    text << " input " << input.name << ' ' << input.tensor->dtype << " [";
    for (const std::string &dtype : input.dtypes) text << ' ' << dtype;
    text << " ] (";
    for (uint64_t dimension : input.tensor->shape) text << ' ' << dimension;
    text << " )\n";
    plan.sources.push_back({input.tensor->file, input.tensor->offset, input.tensor->bytes});
  };
  text << "affine";
  for (const auto &[module, bits] : image.quantized) text << " quantized " << module << ' ' << bits;
  text << '\n';
  for (const Section &section : image.sections) {
    text << "section " << static_cast<int>(section.kind) << ' ' << section.offset << ' ' << section.bytes << ' '
         << section.rows << ' ' << section.columns << ' ' << section.experts << ' ' << section.bits << '\n';
    if (section.parts.empty()) input(section.input);
    for (const ProjectionPart &part : section.parts) {
      text << " part " << part.rows << '\n';
      for (const Input &field : part.fields) input(field);
    }
  }
  plan.description = text.str();
}

// Image `index` of planned, the component directory/name, written from the
// checkpoint.
[[nodiscard]] inline ImagePlan imagePlan(const std::shared_ptr<const PlannedCheckpoint> &planned, size_t index,
                                         std::string_view directory) {
  const Image &image = planned->images.at(index);
  ImagePlan plan{std::string(directory) + "/" + image.name, image.magic, image.layer, image.type, image.bytes,
                 [planned, index](std::span<uint8_t> bytes, const metal::MetalBuffer &) {
                   writeAffineImage(bytes, planned->images[index]);
                   planned->source.checkUnchanged();
                 }};
  describe(image, plan);
  return plan;
}

} // namespace splash::model::affine
