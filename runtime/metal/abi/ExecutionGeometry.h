#pragma once

// C/Metal ABI constants shared by host and shader compilation. Startup
// static assertions check the corresponding model and operator contracts.
#define SPLASH_DRAFT_QUERY_ROWS 8u
#define SPLASH_DRAFT_PROPOSAL_TOKENS 7u
#define SPLASH_TARGET_VERIFY_ROWS 8u
#define SPLASH_MAXIMUM_CONTEXT_TOKENS 262144u
#define SPLASH_SPECULATIVE_SCRATCH_TOKENS                                  \
  (SPLASH_TARGET_VERIFY_ROWS - 1u)
#define SPLASH_MAXIMUM_PHYSICAL_KV_TOKENS                                  \
  (SPLASH_MAXIMUM_CONTEXT_TOKENS + SPLASH_SPECULATIVE_SCRATCH_TOKENS)
#define SPLASH_MAXIMUM_BATCH_WIDTH 4u
#define SPLASH_PREFILL_TOKEN_BUDGET 2048u
#define SPLASH_DRAFT_SLIDING_WINDOW 2048u
#define SPLASH_TARGET_KV_BLOCK_TOKENS 32u
#define SPLASH_PREFILL_ATTENTION_TILE_ROWS 8u
#define SPLASH_PREFILL_ATTENTION_MAXIMUM_SPLITS 32u
#define SPLASH_VERIFY_ATTENTION_SPLITS 32u
// Verify attention runs one split per this many visible Page32 blocks, at
// least the configured split count and at most the maximum that sizes the
// partial workspace (ops::q8VerifyAttentionSplits).
#define SPLASH_VERIFY_ATTENTION_PAGES_PER_SPLIT 16u
#define SPLASH_VERIFY_ATTENTION_MAXIMUM_SPLITS 128u
#define SPLASH_TARGET_SAMPLING_SHARDS 16u
// Threads of each group that selects a sampled row over the whole
// vocabulary (decode_sample_vocabulary*); a bracket of at most this many
// tokens is ordered in threadgroup memory, one token per thread.
#define SPLASH_TARGET_VOCABULARY_THREADS 1024u
// Groups that share such a row's draw, each over its own slice of the
// vocabulary, so the eight rows of a lane spread over the GPU's cores.
#define SPLASH_TARGET_VOCABULARY_GROUPS 8u
#define SPLASH_DRAFT_SAMPLING_SHARDS 8u
// Rows of one value head's recurrent state a prefill GDN scan threadgroup
// carries through the chunk: four simdgroups whose lanes each own sixteen key
// columns of one row. The thread count follows from the rows: 128 columns /
// 16 per lane = 8 lanes per row, times the 16 rows; a static_assert in
// prefill/gdn.metal ties the two literals together.
#define SPLASH_GDN_SCAN_STATE_ROWS 16u
#define SPLASH_GDN_SCAN_THREADS 128u
// Plain norms of at most SPLASH_STAGED_NORM_ROWS rows of at most
// SPLASH_STAGED_NORM_WIDTH columns run norm_rms_staged, whose 1024-thread
// groups hold a row in threadgroup memory: the region where it measured
// faster than norm_rms (shared/normalization.metal), which covers every
// decode norm of a 2048-wide model and its short prefill chunks.
#define SPLASH_STAGED_NORM_WIDTH 2048u
#define SPLASH_STAGED_NORM_ROWS 64u
#define SPLASH_STAGED_NORM_THREADS 1024u
