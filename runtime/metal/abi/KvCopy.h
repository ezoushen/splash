#pragma once

// Parameters of the KV page copy shared by the host and the kernel.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#include "metal/abi/KvExtent.h"

// One dispatch moves whole pages of one attention layer between the pool's
// extents and host-visible staging. Threadgroup s serves staging slot s and
// reads its table entry: the page to move and the direction, none for an idle
// slot. A layer's bytes sit at layer_offset inside the slot as key data, key
// scales, value data, value scales, and in the page's extent where
// splash_kv_offset places them; every range is a multiple of 16 bytes.
#define SPLASH_KV_COPY_NONE 0u
#define SPLASH_KV_COPY_TO_STAGING 1u
#define SPLASH_KV_COPY_TO_PAGE 2u

struct SplashKvCopySlot {
  SplashKvPage page;
  uint32_t direction;
  uint32_t reserved;
};

struct SplashKvCopyParams {
  uint32_t data_bytes;
  uint32_t scale_bytes;
  uint32_t slot_bytes;
  uint32_t layer_offset;
  uint32_t extent_pages;
  uint32_t layer;
  uint32_t staging_slots;
  uint32_t reserved;
};

static_assert(sizeof(SplashKvCopySlot) == 16, "KV copy slots are 16 bytes on both sides");
static_assert(sizeof(SplashKvCopyParams) == 32,
              "KV copy parameters are 32 bytes on both sides");
