# Strata features against Tetraeder (ADR 0085)

Target: Strata tag `v0.2.20` (commit a34979fc, 2026-11-22, MIT), read 2026-11-23 from source and docs without running it. Every number below is Strata's own; none is measured here. Paths are Strata repo paths at that tag.

## What Strata is

Four engines, one per model family, each with kernels compiled for one shape and no generic path: `model_family_from_arch` knows `qwen4exp` (Qwen3.8-Flash-Next), `qwen35moe` (Qwen3.6-35B-A3B), `hy_v4_flash` (Hy4-Flash) and `inkling2*` (Inkling 2 Mini) and refuses the rest (`engine/include/strata/artifact/model_family.hpp:17-23`; `q36_engine.cu:1391`). All four are hybrid: recurrent (GatedDeltaNet) or sliding-window layers with few full-attention layers. None is a Qwen3 model, so no model runs on both engines.

Closest model: Qwen3.6-35B-A3B (`qwen35moe`): 40 layers, 30 GatedDeltaNet and 10 full attention (16 query heads, 2 KV heads, head 256, gated), 256 experts top-8 plus a gated shared expert, one MTP block (`docs/QWEN36.md`, `q36_engine.cu:1459-1523`). The llama.cpp pin runs it (`src/models/qwen35moe.cpp`, MTP block loaded at :106) and its ggml has `GATED_DELTA_NET` and `SSM_CONV` on CUDA (`ggml-cuda.cu:3007,3034`), so `--engine auto` runs it today through libllama and an own builder needs no ggml patch for those ops.

Strata's claim on that model: decode 117.8 tok/s against llama.cpp 81.9 to 82.1 (1.44x), prompt 6,220 against 3,794 tok/s (R9700, HIP, Strata 0.1.45, llama.cpp 972b5fc; `docs/QWEN36.md`:121-127); at 8 requests 325.4 against 208.1 (`docs/QWEN36.md`, batching). Tetraeder sits at 0.94 to 1.09 of tuned llama-server on Qwen3 (`docs/features.md`).

## Class 1: applies to our Qwen3 models now

Judged by a paired run on a model we run (rule 3); no default changes on Strata's numbers.

| Strata feature | Mechanism (Strata) | Tetraeder today | Node |
|---|---|---|---|
| KV formats fp8, k8v8, k8v4, rotated | `--kv`; k8v4 = int8 K, rotated Q4_0 V (`generate.cpp:1227-1233`, `ink_main.cpp:414-420`) | **have**: F16, BF16, Q8_0, Q8_0/Q4_0, Q4_0 with Hadamard rotation (`crates/plan/src/kv.rs`, ADR 0050 b) | none |
| Automatic KV format | picks FP8 at 16+ slots or small VRAM (`ink_cuda_backend.cu:148-160`) | missing: the planner takes the format as given | `memory-kv-auto` |
| Elastic KV | allocate only reached cells, the expert cache gives slots back (`--kv-grow`, `profile_rules.hpp:176-183`) | partly: chunked pool reserves per sequence (ADR 0067), the pool size is fixed at load | `memory-kv-auto` |
| Host KV | position split over mapped pinned host memory, 8 global layers of Inkling only (`ink_runner.cuh:1882-1995`) | missing | `memory-kv-host` (ADR 0084) |
| Prompt-lookup drafts | suffix drafter (`engine/src/spec/suffix_drafter.cpp`) | **have**: n-gram drafter (`crates/scheduler/src/spec/ngram.rs`, ADR 0043) | none |
| Adaptive expert cache | VRAM cache learns routing, async swaps (`engine/src/core/expert_cache.cpp`, `--adapt-every`) | **have** in parts: pinned hot set, LRU, frequency tiers, prefetch (ADR 0050 c, 0054) | none; compare once a model is shared |
| GPU plus SSD only, CPU plus SSD only | fit helper sets `--ssd-experts --ram-experts-gb 0` (`inkling_fit.py:398-424`) | to check: the planner has RAM and NVMe tiers; whether it plans a zero-RAM expert tier is untested | `planner-low-ram-modes` |
| Expert-parallel split on 2 GPUs | each GPU holds a share of every layer's experts (`ep_dual_gpu.cu`, `--ep-split`) | missing: shapes are layer split and tensor parallel (ADR 0069) | `planner-ep-split` |
| Conversation spill to SSD | parked conversations in RAM with SSD spill (`docs/DETAILS.md` "Conversation cache") | **have** (2026-11-23): prefix snapshots in RAM spill to NVMe on eviction and return on the next turn, `--cache-disk` and `--cache-disk-mib` (ADR 0087); not measured | none |
| CUDA graph per token, CPU and GPU expert overlap | `--no-token-graph` A/B, hit poke (`generate.cpp:944,1288`) | **have**: graphs per segment, CPU hit group overlap (ADR 0030, 0054, 0064) | none |

## Class 2: needs a shared model

| Item | Why | Node |
|---|---|---|
| Own `qwen35moe` builder | the only route to a direct pairing and the first model with MTP heads. New builder, not an extension: GatedDeltaNet state is recurrent, while the KV pool, `seq_rm` and the prefix cache assume truncatable rows (ADR 0051 deferred exactly this) | `engine-qwen35moe-builder` (own ADR first) |
| MTP drafts | Qwen3.6 ships one MTP block; original Qwen3 has none | `serving-mtp-drafts` |
| Direct pairing | Strata q36 against the own builder, same box | `bench-strata-pairing` |

First measurement before any builder work: `bench-strata-q36`, one 3090-class session, Strata's q36 engine against llama-server at the pin and `tetraeder serve --engine auto` on the same Qwen3.6-35B-A3B GGUF, c1 and c4. It sizes what a builder must close and checks Strata's 1.44x on CUDA.

## Class 3: owner scope (own ADR each)

Inkling 2 Mini, Hy4-Flash and Qwen3.8-Flash-Next (new families with their own drafters: DSpark, DFly2, DFlash2); vision input (plan feature 34); HIP, SYCL and Metal backends (feature 14; rules 9 and 11 need that hardware in CI); NVFP4, Gyro and the SQ formats. Node `serving-strata-owner-scope`.

## Scoped out

| Row | Reason |
|---|---|
| Route tail skip, KVA, Q4_0-only KV defaults | change outputs (Strata: KL 0.0126; KVA approximates layers); rule 2 |
| Server API, tool calling, `/props`, `/slots`, `/metrics` | llama-server parity matrix (`docs/notes/llamacpp-parity.md`) |
| Per-model fused kernels (GDN megakernel, bit-plane LM head) | per-shape kernels are Strata's design, not ours (ggml with a patch queue, ADR 0011); a kernel enters only through the patch queue with a measured gap |
