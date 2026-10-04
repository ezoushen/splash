#include "model/SharedImages.hpp"
#include "model/ImageRegistry.hpp"
#include "model/TensorDigests.hpp"
#include "model/WeightStore.hpp"

#include <sstream>
#include <utility>

namespace splash::model {

struct SharedImages::Impl {
  explicit Impl(SharedImagesConfig settings)
      : config(std::move(settings)), registry(config.registry, config.writeWait),
        digests(config.digestTables, config.modelRoot, config.sourceDigests) {}
  SharedImagesConfig config;
  ImageRegistry registry;
  TensorDigests digests;
};

SharedImages::SharedImages(SharedImagesConfig config) : impl_(std::make_unique<Impl>(std::move(config))) {}

SharedImages::~SharedImages() = default;

std::string SharedImages::key(const ImagePlan &plan) {
  if (plan.description.empty()) return {};
  const auto sources = impl_->digests.digests(plan.sources);
  if (!sources) return {};
  std::ostringstream canonical;
  canonical << "splash-shared-image-v1\n"
            << impl_->config.writerIdentity << '\n'
            << plan.component << '\n'
            << plan.magic << '\n'
            << plan.layer << '\n'
            << plan.type << '\n'
            << plan.bytes << '\n'
            << plan.description << '\n';
  for (const std::string &digest : *sources) canonical << digest << '\n';
  return weightDigest(canonical.str());
}

bool SharedImages::acquire(const std::string &key, uint64_t bytes,
                           const std::function<void(metal::MappedMemory)> &attach,
                           const std::function<void()> &write) {
  return impl_->registry.acquire(key, bytes, attach, write);
}

void SharedImages::newPass() { impl_->registry.newPass(); }

} // namespace splash::model
