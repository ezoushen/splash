# About this fork

This is [ezoushen](https://github.com/ezoushen)'s fork of [Inco AI's Splash](https://github.com/incoai/splash).
It exists to serve a fine-tuned model beside its base model on one Mac without loading their shared weights
twice, and to ask that fine-tune typed questions about images. We use it to run
[Umpire](https://huggingface.co/ezoushen/umpire-35b-a3b-mlx-4bit), a decision fine-tune of Ornith 1.5
35B-A3B, next to Ornith itself.

It adds two things to upstream Splash 1.3.2:

- **Shared weight images, behind `serve --share-weights` (off by default).** A MoE layer's routed experts get
  an image of their own, with or without the switch. With it, each image is keyed by the content of the source
  bytes it reads and the code that writes it, and processes of one user and one build hand images to each
  other through unnamed shared memory, so a second process on the same model, or a fine-tune that leaves the
  experts unchanged, maps the first one's memory instead of writing another copy. Every loader of 1.3.2
  shares: MLX, Model Optimizer and compressed-tensors targets, GGUF targets, the DFlash draft and the vision
  tower. An idle release frees only what no other process holds, and `/status` says how much of it others
  still hold. `DEVELOPMENT.md` describes the mechanism, its waits and its trust model. Proposed upstream in
  [incoai/splash#203](https://github.com/incoai/splash/issues/203).
- **Images on `POST /v1/systemone`.** A request may carry `"images"`: 1 to 64 `data:` URLs, which every
  question's prompt shows before the evidence. A server started with `--language-only` rejects them. The
  model needs a vision tower; Umpire's MLX repository carries Ornith's.

```bash
git clone -b integration-1.3.2 https://github.com/ezoushen/splash.git && cd splash && make -j4
./splash serve --model ezoushen/ornith-1.5-35b-a3b-mlx-4bit --share-weights --port 8000
./splash serve --model ezoushen/umpire-35b-a3b-mlx-4bit --share-weights --port 8001 --max-context 16K
```

## Branches, changes from upstream, and credit

`integration-1.3.2` merges two topic branches on upstream's `1.3.2` tag, each holding one feature:

| Branch | Content | Author | Origin |
| --- | --- | --- | --- |
| `shared-weight-images-1.3.2` | Write a MoE layer's routed experts into an image of their own; share every loader's weight images between processes by content (`--share-weights`), through idle release and restore; let the release check allow a named, intended change of weight images | ezoushen | this fork ([#203](https://github.com/incoai/splash/issues/203)) |
| `vision-systemone-1.3.2` | Allow score-only requests to carry images | [Audrey Tang](https://github.com/audreyt) | cherry-picked from [#159](https://github.com/incoai/splash/pull/159) and fitted to 1.3.2's protocol validation; the commit credits Devin as co-author, as the original does |
| | Accept images on `/v1/systemone` | ezoushen, co-authored by Audrey Tang | adapted from the `images` field of [#159](https://github.com/incoai/splash/pull/159), keeping its request shape and limits; #159's other extensions are not included |

Everything else is Splash by Inco AI, under the [Apache-2.0 licence](LICENSE), which also covers our changes.
Nothing here is merged upstream yet. The original README follows.

---

# Splash

[![CI](https://github.com/incoai/splash/actions/workflows/ci.yml/badge.svg)](https://github.com/incoai/splash/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Apple%20silicon-black.svg)](#quick-start)

**A local inference engine for Apple silicon, built around the model.**

Splash runs coding agents and OpenAI or Anthropic compatible applications on
one Mac. It combines [DFlash 2](https://inco.ai/blog/dflash2/) speculative
decoding, specialized Metal kernels, and automatic memory planning, with
vision, tool calling, and a built-in chat page.
It reuses cached prefixes and batches concurrent requests automatically.

## Quick start

Apple M3 or newer, macOS 26.4 or later, and [Homebrew](https://brew.sh).
The 4-bit examples need at least 36 GB of unified memory (48 GB recommended);
24 GB Macs can use [smaller GGUF variants](#models).

```bash
brew install incoai/tap/splash
splash serve --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
```

The first run downloads the model and its matching draft and starts serving
on `127.0.0.1:8000`. Later starts reuse the downloads; leave room on disk for
them ([storage requirements](DEVELOPMENT.md#model-storage)).

Once it prints `Ready`, leave this terminal open. Open <http://127.0.0.1:8000>
in your browser, or run an installed coding agent from another terminal:

```bash
splash opencode    # or: splash claude / splash codex / splash hermes / splash pi
```

For a server started with another `--port`, pass the same `--port` to the agent.

Press Ctrl+C in the server terminal to stop Splash.
For LM Studio Bionic, follow its [Splash setup guide](https://lmstudio.ai/blog/splash-engine).

## Use the API

OpenAI Chat Completions, Responses and Completions, and Anthropic Messages,
with streaming, tool calls, JSON Schema output, and base64 images and PDFs:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M",
    "messages": [{"role": "user", "content": "Explain speculative decoding in one sentence."}]
  }'
```

Reasoning follows the model default; `"reasoning_effort": "none"` turns it off
in Chat. Anthropic Messages requests reason only when they set `thinking`.
[Reasoning settings](DEVELOPMENT.md#default-reasoning-effort) ·
[Tool calls](DEVELOPMENT.md#tool-calls) ·
[API details](DEVELOPMENT.md#api)

## Models

Splash supports these model families, with a matching DFlash2 draft selected
automatically:

| Model | GGUF example | MLX 4-bit |
| --- | --- | --- |
| Qwen3.8-27B | `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M` | `mlx-community/Qwen3.8-27B-4bit` |
| Qwen3.6-35B-A3B | `unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M` | `mlx-community/Qwen3.6-35B-A3B-4bit` |

Unsloth GGUF variants span **1–8 bits**, including mixed-precision UD formats
up to `UD-Q8_K_XL`, whose BF16 tensors load too. Whole-BF16 and F16 models are
not supported: the installer refuses a GGUF whose projections and experts are
all BF16, F16 or F32.
[Prism ML Ternary Bonsai 2](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
is also supported in PQ2_0 (7.2 GB), including vision. Pass `OWNER/REPO:VARIANT`
to `--model`, as in the quick start. MLX targets can be affine 2–8-bit,
including mixed precision, mxfp4 or nvfp4. NVIDIA's and unsloth's NVFP4
checkpoints and NVFP4 GGUFs also load. Smaller variants run on
[24 GB Macs](docs/performance.md#smaller-ggufs-on-24-gb-macs).
[27B variants](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/main) ·
[35B variants](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/tree/main)

Vision and the tokenizer come from the target model's source.
[Model loading and compatibility](DEVELOPMENT.md#upstream-model-loading) ·
[Supported formats](DEVELOPMENT.md#gguf-targets)

## Settings

Memory and context are sized automatically, up to the model's native context
window. To set your own limits or cache options, add these to `splash serve`:

| Option | Purpose |
| --- | --- |
| `--max-memory 28G` | Cap Metal memory use. |
| `--idle-release off` | Keep the model in memory while idle (default: release after 10m). |
| `--max-context 100K` | Set the context limit. |
| `--language-only` | Skip vision; serve text only. |
| `--kv-format bf16` | Use BF16 KV cache. Default: 8-bit (INT8). |
| `--state-format bf16` | Use BF16 recurrent states, half the size. Default: FP32. |
| `--disable-ane` | Prefill on the GPU alone. Default: the 27B also uses the Neural Engine. |
| `--share-weights` | Share weight memory with other Splash processes of this build that load the same weights. Off by default. |
| `--max-cache-disk 16G` | Cap the SSD cache (default: auto, up to 32 GiB; `0` turns it off). |
| `--persistent-cache` | Keep the SSD cache across restarts. Off by default. |

Use `--max-memory` to leave room for other applications.
The server listens on localhost without authentication by default. For LAN
access, authentication, browser apps on other origins, and other options, see
[server configuration](DEVELOPMENT.md#server-configuration) or
`splash serve --help`.
[KV precision](DEVELOPMENT.md#kv-cache-precision) ·
[SSD cache](DEVELOPMENT.md#ssd-cache)

## Performance

Measured with Splash 1.0 on an M5 Pro (16-core GPU, 48 GB), using its model
packages and selected SPEED-Bench coding prompts over HTTP. Ratios compare
with the next-fastest engine measured in that benchmark.

| Metric | Qwen3.6-35B-A3B | Qwen3.8-27B |
| --- | ---: | ---: |
| Decode · short prompt | 210 tok/s (1.7×) | 74 tok/s (2.0×) |
| Prefill · 32K prompt | 2,011 tok/s (1.3×) | 363 tok/s (1.2×) |
| Cached time to first token · 32K replay | 123 ms (6.6×) | 282 ms (7.3×) |
| Aggregate decode · 4 concurrent short prompts | 357 tok/s (2.0×) | 170 tok/s (3.9×) |

[Launch benchmarks](https://inco.ai/blog/splash/) ·
[Measurement details](docs/performance.md#splash-10-launch-benchmarks) ·
[Run benchmarks locally](DEVELOPMENT.md#local-benchmarks)

### GGUF against llama.cpp

Same Unsloth UD-Q4_K_M weights on Metal. Decode speed in tok/s:

| Model | Engine | M5 Pro, 20-core GPU | M3 Max, 40-core GPU |
| --- | --- | ---: | ---: |
| 27B | llama.cpp | 16 | 17 |
| | llama.cpp with MTP | 27 | 20 |
| | **Splash** | **74** | **92** |
| 35B-A3B | llama.cpp | 69 | 66 |
| | **Splash** | **175** | **209** |

That is **2.5–3.2×** as fast on the 35B and **4.5–5.3×** on the 27B
(**2.7–4.6×** against MTP).

**Closely matches llama.cpp's predictions.**

| Next-token agreement ↑ | 27B | 35B-A3B |
| --- | ---: | ---: |
| llama.cpp: CPU vs. GPU | 97.8% | 96.5–96.9% |
| llama.cpp: single-token vs. batched | 99.65–99.75% | 97.95% |
| **Splash vs. llama.cpp** | **99.30–99.45%** | **97.83–98.14%** |

Splash uses BF16 KV and `--disable-ane` in this comparison.
[Benchmark details](docs/performance.md#gguf-against-llamacpp)

## Design

Each supported model pairs a trained DFlash2 draft with Metal kernels for its
shapes. The runtime, scheduler, cache, and API are shared. Weights are
converted to the kernels' layouts as they load, with no copy on disk; kernels
ship precompiled, with no Xcode or local tuning required.
[How Splash works](https://inco.ai/blog/splash/)

## More

- [Development](DEVELOPMENT.md): configuration, API, internals, building, and testing.
- [Issues and feedback](https://github.com/incoai/splash/issues)
- [Apache-2.0](LICENSE). GGUF kernels include MIT-licensed material from
  llama.cpp; see [third-party notices](THIRD_PARTY_NOTICES). Model weights keep their own licenses.
