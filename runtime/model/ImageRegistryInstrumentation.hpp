#pragma once

#include "model/ImageRegistry.hpp"

namespace splash::model {

// What tests observe of an ImageRegistry beyond its serving API. Only the
// instrumented build of ImageRegistry.cpp (SPLASH_REGISTRY_INSTRUMENTATION)
// defines these; production binaries never link it.
class ImageRegistryInstrumentation final {
public:
  // hook runs on the thread of a lookup that has found a process ended,
  // before the lookup removes that process's files; a null hook clears it.
  // It applies to every registry of the process.
  static void setRemovalHook(void (*hook)());
};

} // namespace splash::model
