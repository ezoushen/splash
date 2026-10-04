// Loads every weight image of an installed model as the engine does and
// prints, one per line of a JSON array in load order, the component, size and
// SHA-256 of each: the release check records them (verify-models) and the
// regression benchmarks compare them between builds (dev/benchmarks/weights.py).
// With --share-weights the images are shared as `serve --share-weights` shares
// them, as the engine beside METALLIB writes them: an image another process
// of that build holds is received, which its digest then covers. This tool
// writes the images it holds with its own copy of the engine's writers, so it
// takes the engine's identity only when the engine is of its own build (its
// build identity, which covers the engine's sources and flags), and refuses
// otherwise.
//
//   weight-digests METALLIB MODEL_DIRECTORY [--share-weights]
#import <Foundation/Foundation.h>

#include "metal/MetalBackend.hpp"
#include "model/ModelDescriptor.hpp"
#include "model/ModelFactory.hpp"
#include "model/SharedImages.hpp"
#include "model/WeightImages.hpp"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#ifndef SPLASH_BUILD_ID
#error "weight-digests requires the generated BuildIdentity.hpp"
#endif

using namespace splash;

namespace {

// Refuses an engine of another build than this tool's: one that does not
// hold this build's identity.
void requireThisBuild(const std::filesystem::path &engine) {
  std::ifstream file(engine, std::ios::binary);
  const std::string bytes{std::istreambuf_iterator<char>(file), {}};
  if (!file) throw std::runtime_error("cannot read " + engine.string());
  if (bytes.find(SPLASH_BUILD_ID) == std::string::npos)
    throw std::runtime_error(engine.string() + " is not of this build (" SPLASH_BUILD_ID
                             "): --share-weights would key images this tool writes by another build's code");
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    const bool share = argc == 4 && std::string_view(argv[3]) == "--share-weights";
    if (argc != 3 && !share) {
      std::fprintf(stderr, "usage: weight-digests METALLIB MODEL_DIRECTORY [--share-weights]\n");
      return 2;
    }
    try {
      metal::MetalBackend backend(argv[1]);
      const std::filesystem::path root = std::filesystem::canonical(argv[2]);
      std::shared_ptr<model::SharedImages> shared;
      if (share) {
        const std::filesystem::path engine = std::filesystem::canonical(argv[1]).parent_path() / "splash";
        requireThisBuild(engine);
        shared = std::make_shared<model::SharedImages>(
            model::sharedImagesConfig(root, model::writerIdentity(engine, backend)));
      }
      const model::ModelPackage package =
          model::loadModelPackage(backend, root, model::inspectModelPackage(root), std::move(shared));
      const auto images = package.images->contents();
      std::vector<std::string> digests(images.size());
      model::parallelFor(images.size(), [&](size_t index, unsigned) {
        digests[index] = model::weightDigest(images[index].bytes);
      });
      std::cout << '[';
      for (size_t index = 0; index < images.size(); ++index)
        std::cout << (index ? ",\n" : "\n") << "{\"component\":\"" << images[index].component
                  << "\",\"bytes\":" << images[index].bytes.size() << ",\"sha256\":\"" << digests[index] << "\"}";
      std::cout << "\n]\n";
    } catch (const std::exception &error) {
      std::fprintf(stderr, "error: %s\n", error.what());
      return 1;
    }
  }
  return 0;
}
