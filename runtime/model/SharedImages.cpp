#include "model/SharedImages.hpp"
#include "model/ImageRegistry.hpp"
#include "model/ModelDescriptor.hpp"
#include "model/TensorDigests.hpp"
#include "model/WeightStore.hpp"

#include <dlfcn.h>
#include <mach-o/loader.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

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

namespace {

// The digest of an engine's code: its __TEXT segment, from the Mach-O header
// at image, which a process maps as it is in the file: the header, the
// machine code and the constants. available bounds the bytes at image.
std::string codeDigest(const uint8_t *image, uint64_t available) {
  mach_header_64 header;
  if (available < sizeof(header)) throw std::runtime_error("an engine is not a Mach-O executable");
  std::memcpy(&header, image, sizeof(header));
  if (header.magic != MH_MAGIC_64 || header.sizeofcmds > available - sizeof(header))
    throw std::runtime_error("an engine is not a 64-bit Mach-O executable");
  uint64_t at = sizeof(header);
  for (uint32_t index = 0; index < header.ncmds; ++index) {
    load_command command;
    if (header.sizeofcmds - (at - sizeof(header)) < sizeof(command))
      throw std::runtime_error("an engine's load commands are malformed");
    std::memcpy(&command, image + at, sizeof(command));
    if (command.cmdsize < sizeof(command) || command.cmdsize > header.sizeofcmds - (at - sizeof(header)))
      throw std::runtime_error("an engine's load commands are malformed");
    if (command.cmd == LC_SEGMENT_64 && command.cmdsize >= sizeof(segment_command_64)) {
      segment_command_64 segment;
      std::memcpy(&segment, image + at, sizeof(segment));
      if (std::string_view(segment.segname, strnlen(segment.segname, sizeof(segment.segname))) == SEG_TEXT) {
        if (segment.fileoff != 0 || segment.filesize > available)
          throw std::runtime_error("an engine's code is malformed");
        return weightDigest(std::span(image, segment.filesize));
      }
    }
    at += command.cmdsize;
  }
  throw std::runtime_error("an engine has no code segment");
}

std::vector<uint8_t> readWholeFile(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  const std::streamoff size = file ? std::streamoff(file.tellg()) : -1;
  std::vector<uint8_t> bytes(size > 0 ? size_t(size) : 0);
  if (size < 0 || !file.seekg(0) || !file.read(reinterpret_cast<char *>(bytes.data()), size))
    throw std::runtime_error("cannot read " + path.string());
  return bytes;
}

std::string identityOf(std::string_view code, const metal::MetalBackend &backend) {
  return weightDigest(std::string(code) + '\n' + backend.libraryDigest() + '\n');
}

} // namespace

std::string writerIdentity(const metal::MetalBackend &backend) {
  // The executable or library that holds this code, as it is mapped.
  Dl_info image{};
  if (!dladdr(reinterpret_cast<const void *>(&codeDigest), &image) || !image.dli_fbase)
    throw std::runtime_error("cannot find the running engine's code");
  return identityOf(codeDigest(static_cast<const uint8_t *>(image.dli_fbase), UINT64_MAX), backend);
}

std::string writerIdentity(const std::filesystem::path &engine, const metal::MetalBackend &backend) {
  const std::vector<uint8_t> bytes = readWholeFile(engine);
  return identityOf(codeDigest(bytes.data(), bytes.size()), backend);
}

SharedImagesConfig sharedImagesConfig(const std::filesystem::path &root, std::string writerIdentity) {
  const char *home = std::getenv("HOME");
  if (!home || !*home) throw std::runtime_error("HOME is not set: shared weights keep digests in its caches");
  SharedImagesConfig config;
  config.registry = "/tmp/splash-" + std::to_string(geteuid());
  config.digestTables = std::filesystem::path(home) / "Library/Caches/Splash/tensor-digests";
  config.writerIdentity = std::move(writerIdentity);
  config.modelRoot = root;
  config.sourceDigests = recordedSourceDigests(root);
  return config;
}

} // namespace splash::model
