#pragma once

#include "QwenHybridLayout.hpp"
#include "QwenTarget.hpp"
#include "QwenTargetFiles.hpp"
#include "metal/abi/RoPE.h"
#include "ops/MoE.hpp"
#include "ops/Normalization.hpp"
#include "ops/Vision.hpp"

#include <array>
#include <cstdint>
#include <string_view>

namespace splash::model {

// What differs between the sparse-MoE Qwen targets Splash serves: the sizes
// of the mixer and the experts, and the block the draft they pair with was
// trained for. Everything else (window, vocabulary, key heads, KV heads,
// rotary, experts and top-k) is shared, and the descriptor checks each config
// against the geometry it names; the draft's own layout is its config's
// (ModelDescriptor.mm readDraftLayout).
struct QwenMoeGeometry final {
  // The family's name, as the installer pairs its draft (install/families.py).
  std::string_view family;
  uint32_t layers = 0;
  uint32_t hiddenSize = 0;
  uint32_t packedGdnWidth = 0;
  uint32_t packedFullWidth = 0;
  uint32_t convolutionDimension = 0;
  uint32_t gdnValueHeads = 0;
  uint32_t attentionWidth = 0;
  uint32_t attentionQueryHeads = 0;
  uint32_t expertIntermediateSize = 0;
  // The block its DFlash2 draft was trained for (DFlashDraftLayout::trainedBlockSize).
  uint32_t draftBlockSize = 8;
  // Whether a GGUF of the family has run on Splash.
  bool ggufServed = false;
};

inline constexpr QwenMoeGeometry kQwen3_6Moe35B{.family = "Qwen3.6-35B-A3B",
                                                .layers = 40,
                                                .hiddenSize = 2048,
                                                .packedGdnWidth = 12544,
                                                .packedFullWidth = 9216,
                                                .convolutionDimension = 8192,
                                                .gdnValueHeads = 32,
                                                .attentionWidth = 4096,
                                                .attentionQueryHeads = 16,
                                                .expertIntermediateSize = 512,
                                                .ggufServed = true};

// Qwen3.5-122B-A10B (Ornith-1.5-122B), whose DFlash2 draft was trained for
// blocks of 16.
inline constexpr QwenMoeGeometry kQwen3_5Moe122B{.family = "Qwen3.5-122B-A10B",
                                                 .layers = 48,
                                                 .hiddenSize = 3072,
                                                 .packedGdnWidth = 20736,
                                                 .packedFullWidth = 17408,
                                                 .convolutionDimension = 12288,
                                                 .gdnValueHeads = 64,
                                                 .attentionWidth = 8192,
                                                 .attentionQueryHeads = 32,
                                                 .expertIntermediateSize = 1024,
                                                 .draftBlockSize = 16};

// The layout of a sparse-MoE Qwen target: by default the 35B's (Qwen3.6-35B-A3B),
// or of another geometry on request.
struct Qwen3_6MoeLayout final : QwenHybridLayout {
  // The family's name, as the installer pairs its draft (install/families.py).
  std::string_view family;
  // Its config's text model type and its GGUFs' architecture; a safetensors
  // checkpoint of it loads too.
  static constexpr std::string_view modelType = "qwen3_5_moe_text";
  static constexpr std::string_view ggufArchitecture = "qwen35moe";
  static constexpr bool loadsSafetensors = true;

  constexpr explicit Qwen3_6MoeLayout(const QwenMoeGeometry &geometry)
      : QwenHybridLayout{{.nativeContextTokens = 262'144,
                          .yarnContextTokens = 1'010'000,
                          .layers = geometry.layers,
                          .hiddenSize = geometry.hiddenSize,
                          .vocabularySize = 248320,
                          .packedGdnWidth = geometry.packedGdnWidth,
                          .packedFullWidth = geometry.packedFullWidth,
                          .convolutionDimension = geometry.convolutionDimension,
                          .gdnKeyHeads = 16,
                          .gdnValueHeads = geometry.gdnValueHeads,
                          .gdnHeadDimension = 128,
                          .attentionWidth = geometry.attentionWidth,
                          .attentionQueryHeads = geometry.attentionQueryHeads,
                          .attentionKvHeads = 2,
                          .attentionHeadDimension = 256,
                          .rotaryPairs = SPLASH_TARGET_ROPE_PAIRS,
                          .rotaryTheta = 10'000'000.0F,
                          .fullAttentionPeriod = 4,
                          .maskToken = 248077,
                          .stopTokens = {248044, 248046},
                          .ffnKind = QwenFfnKind::SparseMoe,
                          .experts = 256,
                          .expertsPerToken = 8,
                          .expertIntermediateSize = geometry.expertIntermediateSize}},
        family(geometry.family) {}
  constexpr Qwen3_6MoeLayout() : Qwen3_6MoeLayout(kQwen3_6Moe35B) {}
  bool operator==(const Qwen3_6MoeLayout &) const = default;
};

struct Qwen3_6MoeLayerWeights final {
  ops::NormWeights inputNorm;
  QwenMixerWeights mixer;
  ops::NormWeights postAttentionNorm;
  ops::BlockMoeWeights ffn;
};

using Qwen3_6MoeWeights = QwenTargetWeights<Qwen3_6MoeLayout, Qwen3_6MoeLayerWeights>;

[[nodiscard]] Qwen3_6MoeWeights
loadQwen3_6MoeWeights(metal::MetalBackend &backend, Qwen3_6MoeLayout layout,
                      const QwenTargetFiles &files);

} // namespace splash::model
