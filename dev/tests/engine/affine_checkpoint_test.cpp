#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include "model/SafetensorsCheckpoint.hpp"

#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace splash::model;
using splash::test::rejects;
using splash::test::require;
namespace {
void shard(const std::filesystem::path &path, std::string_view header, std::span<const uint8_t> data) {
  const uint64_t length = header.size();
  std::vector<uint8_t> file(sizeof length);
  std::memcpy(file.data(), &length, sizeof length);
  file.insert(file.end(), header.begin(), header.end());
  file.insert(file.end(), data.begin(), data.end());
  splash::test::writeFile(path, file);
}
void shard(const std::filesystem::path &path, std::string_view header, size_t bytes = 16, uint8_t first = 1) {
  std::vector<uint8_t> data(bytes);
  std::iota(data.begin(), data.end(), first);
  shard(path, header, data);
}
constexpr auto valid = R"({"a":{"dtype":"U32","shape":[2,2],"data_offsets":[0,16]}})";
}
int main() {
  try {
    const splash::test::TemporaryDirectory directory("splash-affine-checkpoint");
    const std::filesystem::path &root = directory.path();
    setenv("SPLASH_WEIGHT_CACHE", (root / "cache").c_str(), 1);
    splash::test::writeFile(root / "config.json", R"({"quantization":{"bits":4,"group_size":64,"router":{"bits":8}},"text_config":{"layers":2,"model_type":"fixture","layer_types":["linear_attention","full_attention"]}})");
    shard(root / "model.safetensors", valid);
    SafetensorsCheckpoint source(root);
    source.requireQuantization("projection", 4);
    source.requireQuantization("router", 8);
    source.requireConfigNumber("layers", 2);
    source.requireConfigString("model_type", "fixture");
    source.requireLayerTypes(2, 2);
    require(source.require("a").shape == std::vector<uint64_t>({2, 2}), "shape changed");
    std::array<uint8_t, 4> data{};
    source.require("a").read(7, data);
    require(data == std::array<uint8_t, 4>{8, 9, 10, 11}, "bounded slice differs");
    rejects([&] { source.require("a").read(14, data); }, "source tensor read is out of bounds",
            "out-of-bounds read accepted");
    rejects([&] { (void)source.require("missing"); }, "missing source tensor: missing", "missing tensor accepted");
    rejects([&] { source.requireQuantization("router", 4); }, "unsupported affine quantization for router",
            "wrong quantization accepted");
    rejects([&] { source.requireLayerTypes(2, 1); }, "source layer schedule does not match",
            "wrong layer schedule accepted");
    // A tensor's identity is its own bytes, dtype and shape, wherever a shard
    // stores them: rewriting the same content keeps it, a header-only edit,
    // moving it within its shard or to another and an edit to another tensor
    // keep it, a changed data byte changes it.
    const auto identity = [](const SafetensorsCheckpoint &checkpoint) {
      WeightIdentity identity("fixture");
      checkpoint.require("a").identify(identity);
      return identity.weight(16, "fixture", "").key;
    };
    const auto first = identity(source);
    shard(root / "model.safetensors", valid);
    rejects([&] { source.checkUnchanged(); }, "source weights changed", "changed source accepted");
    require(identity(SafetensorsCheckpoint(root)) == first, "same content changed identity");
    shard(root / "model.safetensors", R"({"__metadata__":{"format":"mlx"},"a":{"dtype":"U32","shape":[2,2],"data_offsets":[0,16]}})");
    require(identity(SafetensorsCheckpoint(root)) == first, "a header-only edit changed the tensor identity");
    // a after another tensor, b.
    constexpr auto moved = R"({"b":{"dtype":"U8","shape":[16],"data_offsets":[0,16]},"a":{"dtype":"U32","shape":[2,2],"data_offsets":[16,32]}})";
    std::vector<uint8_t> stored(32);
    std::iota(stored.begin() + 16, stored.end(), uint8_t{1});
    shard(root / "model.safetensors", moved, stored);
    require(identity(SafetensorsCheckpoint(root)) == first, "a tensor moved within its shard changed identity");
    stored[0] = 9;
    shard(root / "model.safetensors", moved, stored);
    require(identity(SafetensorsCheckpoint(root)) == first, "an edit to another tensor changed the identity");
    // a at the start of another shard, before a tensor c.
    shard(root / "model.safetensors", R"({"b":{"dtype":"U8","shape":[16],"data_offsets":[0,16]}})");
    shard(root / "other.safetensors",
          R"({"a":{"dtype":"U32","shape":[2,2],"data_offsets":[0,16]},"c":{"dtype":"U8","shape":[16],"data_offsets":[16,32]}})",
          32);
    require(identity(SafetensorsCheckpoint(root)) == first, "a tensor moved to another shard changed identity");
    std::filesystem::remove(root / "other.safetensors");
    shard(root / "model.safetensors", valid, 16, 9);
    require(identity(SafetensorsCheckpoint(root)) != first, "changed tensor data kept its identity");
    shard(root / "model.safetensors", valid);
    shard(root / "extra.safetensors", valid);
    rejects([&] { SafetensorsCheckpoint invalid(root); }, "duplicate source tensor: a", "duplicate tensor accepted");
    std::filesystem::remove(root / "extra.safetensors");
    for (const auto &[header, error] : std::initializer_list<std::pair<std::string_view, std::string_view>>{
             {R"({"a":{"dtype":"U32","shape":[2,2],"data_offsets":[0,15]}})", "safetensors data range is invalid"},
             {R"({"a":{"dtype":"U32","shape":[2,2],"data_offsets":[1,17]}})", "safetensors data range is invalid"},
             {R"({"a":{"dtype":"U32","shape":[true,4],"data_offsets":[0,16]}})",
              "safetensors metadata requires an integer"},
             {R"({"a":{"dtype":"U32","shape":[1,1,1,1,1,1,1,1,4],"data_offsets":[0,16]}})",
              "source tensor metadata exceeds bounds"},
             {R"({"a":{"dtype":"U32","shape":[4503599627370496,4096],"data_offsets":[0,16]}})",
              "source tensor size overflows"},
             {R"({"a":{"dtype":"U32","shape":[4],"data_offsets":[0,16]},"b":{"dtype":"U32","shape":[2],"data_offsets":[8,16]}})",
              "overlapping source tensors"},
             {R"({"a":{"dtype":"FP4","shape":[4],"data_offsets":[0,16]}})", "unsupported safetensors dtype: FP4"}}) {
      shard(root / "model.safetensors", header);
      rejects([&] { SafetensorsCheckpoint invalid(root); }, error,
              "malformed tensor accepted: " + std::string(header));
    }
    shard(root / "model.safetensors", valid, 8);
    rejects([&] { SafetensorsCheckpoint invalid(root); }, "safetensors data range is invalid",
            "truncated tensor accepted");
    std::cout << "affine checkpoint: bounded reads, metadata, quantization, identity and malformed sources PASS\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
