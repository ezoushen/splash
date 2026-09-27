#import <Foundation/Foundation.h>

#include "model/AffineTarget.hpp"
#include "model/ModelDescriptor.hpp"

#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cmath>
#include <iostream>
#include <vector>

using namespace splash;

// Where a package file holds the `bytes` bytes of a prepared file at
// `prepared`: at `packed`.
struct Span {
  uint64_t prepared, packed, bytes;
};

// The spans of a layer's prepared images in the package's layer file, which
// holds the layer image's sections in order and a MoE layer's routed experts
// after its router, where the reader reads them (Qwen3_6Moe.cpp readFfn).
// Each span runs to the next 16 KiB boundary, so padding is compared too.
std::vector<std::vector<Span>> layerSpans(const std::vector<model::affine::Image> &images) {
  std::vector<std::vector<Span>> spans(images.size());
  uint64_t packed = model::kWeightFileAlignment;
  const auto place = [&](size_t image, const model::affine::Section &section) {
    const uint64_t bytes = model::alignWeightOffset(section.offset + section.bytes) - section.offset;
    spans[image].push_back({section.offset, packed, bytes});
    packed += bytes;
  };
  for (const auto &section : images[0].sections) {
    place(0, section);
    if (images.size() > 1 && !section.parts.empty() &&
        section.parts.front().fields.front().name.ends_with(".mlp.gate.weight"))
      for (const auto &expert : images[1].sections) place(1, expert);
  }
  return spans;
}

// Throws unless the package file at path is its header and the spans.
void requireSize(const std::filesystem::path &path, const std::vector<std::vector<Span>> &spans) {
  uint64_t bytes = model::kWeightFileAlignment;
  for (const auto &image : spans)
    for (const Span &span : image) bytes += span.bytes;
  if (std::filesystem::file_size(path) != bytes)
    throw std::runtime_error("prepared size differs: " + path.filename().string());
}

// Compares the spans of a prepared file, which cover it after its header,
// with the package file at path, and with its header when the package file
// has it (withHeader).
void compare(model::WeightFile file, const std::filesystem::path &path, const std::vector<Span> &spans,
             bool withHeader, bool loadOnly, uint64_t decayOffset = 0, uint32_t decayHeads = 0) {
  const auto &record = file.record();
  const int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open packed oracle");
  try {
    std::array<uint8_t, 16> header;
    model::readWeightBytes(fd, 0, header);
    if (withHeader && (memcmp(header.data(), record.magic.data(), 8) || memcmp(header.data() + 8, &record.layer, 4) ||
                       memcmp(header.data() + 12, &record.type, 4)))
      throw std::runtime_error("prepared header differs");
    const auto data = file.section(record.declaredBytes - model::kWeightFileAlignment);
    const auto *prepared = static_cast<const uint8_t *>(data.contents());
    std::vector<uint8_t> bytes(1024 * 1024);
    uint32_t maximumDecayUlp = 0;
    for (const Span &span : spans) {
      for (uint64_t at = 0; !loadOnly && at < span.bytes; at += bytes.size()) {
        auto part = std::span(bytes).first(std::min<uint64_t>(bytes.size(), span.bytes - at));
        model::readWeightBytes(fd, span.packed + at, part);
        // Released dense packages used MLX's float exp for the small GDN
        // decay vector. The source adapter uses double exp rounded to float.
        // Only that named section permits a two-ULP difference; every packed
        // code, scale, bias, other tensor and padding byte remains exact.
        const uint64_t absolute = span.prepared + at;
        const uint8_t *actualBytes = prepared + (absolute - model::kWeightFileAlignment);
        for (uint32_t head = 0; head < decayHeads; ++head) {
          const uint64_t offset = decayOffset + head * sizeof(float);
          if (offset < absolute || offset + 4 > absolute + part.size()) continue;
          const size_t local = offset - absolute;
          uint32_t actual, expected;
          memcpy(&actual, actualBytes + local, 4);
          memcpy(&expected, part.data() + local, 4);
          float a, b;
          memcpy(&a, &actual, 4); memcpy(&b, &expected, 4);
          const uint32_t ulp = actual > expected ? actual - expected : expected - actual;
          if (!std::isfinite(a) || !std::isfinite(b) || !(a < 0 && b < 0) || ulp > 2)
            throw std::runtime_error("GDN decay differs by more than two ULP");
          maximumDecayUlp = std::max(maximumDecayUlp, ulp);
          memcpy(part.data() + local, &actual, 4);
        }
        if (memcmp(actualBytes, part.data(), part.size())) {
          const auto different = std::mismatch(part.begin(), part.end(), actualBytes).first;
          throw std::runtime_error("prepared bytes differ: " + record.relativePath + " offset=" +
                                   std::to_string(absolute + (different - part.begin())));
        }
      }
    }
    file.finish();
    std::cout << record.relativePath << " bytes=" << record.declaredBytes << (loadOnly ? " opened=true" : " packed_exact=true") << " decay_max_ulp=" << maximumDecayUlp << std::endl;
  } catch (...) { close(fd); throw; }
  close(fd);
}

// Compares a prepared file with the package file of its name, which holds it
// whole.
void compareWhole(model::WeightFile file, const std::filesystem::path &target, bool loadOnly) {
  const auto path = target / std::filesystem::path(file.record().relativePath).filename();
  const std::vector<Span> spans{{model::kWeightFileAlignment, model::kWeightFileAlignment,
                                 file.record().declaredBytes - model::kWeightFileAlignment}};
  requireSize(path, {spans});
  compare(std::move(file), path, spans, true, loadOnly);
}

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc < 4 || argc > 5) throw std::runtime_error("usage: affine-source-oracle METALLIB SOURCE PACKED_PACKAGE [LAYER|--load-only]");
      const bool loadOnly = argc == 5 && std::string_view(argv[4]) == "--load-only";
      const auto started = std::chrono::steady_clock::now();
      metal::MetalBackend backend(argv[1]);
      const std::filesystem::path package(argv[3]);
      const auto descriptor = model::inspectModelPackage(package);
      std::visit([&](const auto &layout) {
        model::AffineTargetLoader loader(backend, argv[2], layout);
        const uint32_t begin = argc == 5 && !loadOnly ? std::stoul(argv[4]) : 0;
        const uint32_t end = argc == 5 && !loadOnly ? begin + 1 : layout.layers;
        for (uint32_t layer = begin; layer < end; ++layer) {
          const auto images = model::affineLayerImages(layout, layer);
          const auto spans = layerSpans(images);
          const auto path = package / "target" / images[0].name;
          requireSize(path, spans);
          const auto &sections = images[0].sections;
          const auto decay = std::ranges::find(sections, model::affine::SectionKind::Decay,
                                               &model::affine::Section::kind);
          const bool gdn = decay != sections.end();
          compare(loader.layer(layer), path, spans[0], true, loadOnly, gdn ? decay->offset : 0,
                  gdn ? uint32_t(decay->bytes / sizeof(float)) : 0);
          if (images.size() > 1) compare(loader.experts(layer), path, spans[1], false, loadOnly);
        }
        if (argc != 5 || loadOnly) {
          compareWhole(loader.head(), package / "target", loadOnly);
          compareWhole(loader.embedding(), package / "target", loadOnly);
        }
      }, descriptor.target);
      std::cout << "affine source oracle PASS seconds="
                << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() << '\n';
    } catch (const std::exception &error) {
      std::cerr << error.what() << '\n';
      return 1;
    }
  }
}
