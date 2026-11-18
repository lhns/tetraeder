# Session 20261101

- Status: development and diagnosis runs of Phase 2 (overflow passes, pair groups, CPU experts); not a verdict. Their findings led to ADR 0034, 0035, 0036 and 0039.
- Code and box: builder commits of 2026-11-01 (cd053a7 to cef79f2, see git log of this directory); several boxes that day, ids unknown; RTX 3090, driver 610.57.04 (env.json); `cpu-bench-real.json` on an EPYC 7K62, 11 threads, CPU only.
- Raw files: this directory.

## From ADR 0034 Decision (moved verbatim)

- The passes multiply by a 0/1 mask instead of the router weights and accumulate the unweighted expert outputs per (token, choice); the next segment applies the router weights and sums over experts with the same ops as a single pass. Each pair is nonzero in exactly one pass, and `x * 1` and `x + 0` are exact, so the result is bitwise equal to a single pass and streaming output does not depend on the slot count. The first real-model run (Qwen3-30B-A3B Q4_K_M, 32 slots) showed why exactness matters: CUDA fuses weighting and sum into one kernel (`moe_weighted_reduction`, which contracts to fused multiply-adds), passes that weighted and summed by themselves rounded differently in the last bit, a near-tied routing choice flipped at layer 2 and the difference cascaded through attention to logit differences above 1. The accumulator is `n_expert_used` times larger than the FFN output (about 270 MB at ubatch 4096 for Qwen3-30B-A3B, inside the phase workspace budget).

## From ADR 0034 Consequences (moved verbatim)

- **Masked pairs park outside the slots.** The same multi-pass prefill also differed on the CPU backend (Qwen3-30B-A3B Q4_K_M, 32 slots: first router weight difference at layer 1, including tokens whose experts were all in pass 0). Cause: the vendored CPU `mul_mat_id` sends an expert to a tiled K-quant kernel (`ggml-cpu/tiled`, Q2_K to Q6_K and IQ types, AVX builds, `GGML_CPU_TILED_MM`) when at least 8 rows route to it, else to `vec_dot`, and the two round differently. So a pair's output depends on how many rows share its expert. The masked pairs went to the lowest free slots, which hold the chunk's own experts (in a first-appearance chunk, the first tokens' experts), and pushed their row counts across 8. The F16 and Q8_0 fixtures have no tiled kernel, which is why they stayed exact. Now the slot tensors have `n_used` extra zero parking entries after the slots and masked pairs use only those, so every chunk expert sees exactly its single-pass rows in the same order. Cost: `n_used` experts per layer more memory (Qwen3-30B-A3B: about 1.1 GB at 32 slots). A pair-compacted pass gives each expert its real rows too, but must also keep their order and the input layout to stay exact; not yet verified. The tiled kernel needs AVX, so the K-quant exact test (`crates/model/tests/multipass.rs`) only discriminates on AVX machines; a unit test checks the row-count invariant everywhere.

## From ADR 0035 Context (moved verbatim)

The run of 2026-11-01 (bench/results/20261101, RTX 3090, Qwen3-30B-A3B Q4_K_M, the builder as of that run) measured:

| Setup | Prefill tok/s | Decode tok/s |
|---|---|---|
| llama.cpp, all on GPU | 2800 (ubatch 512) to 4300 (4096) | 190 |
| llama.cpp, all experts on CPU (`-ot`) | 460 to 2080 | 40 |
| llama.cpp, layers 24 to 47 experts on CPU | 880 to 2900 | 67 |
| Tetraeder, 32 VRAM slots, upload on miss | 9 | 6 |
| Tetraeder, 32 slots + 64 RAM slots | 6 | 18 |
| Tetraeder, 120 slots | 7 | 46 |

Uploading every missed expert loses to computing it where it already is: a decode step misses dozens of experts at a few MB each, and the read and upload path ran at 0.8 to 1.5 GB/s. The ADR 0033 overflow passes also compute every pair in every pass (P times the FFN work) and need `n_used` parking entries per layer to stay exact (ADR 0034).

## From ADR 0035 Consequences (moved verbatim)

- Exit numbers to beat: llama.cpp `-ot` at the same VRAM use (decode 40 to 67 tok/s, prefill 460 to 2900 by ubatch). A large prefill ubatch matters: llama.cpp's `-ot` prefill gains 4.5x from ubatch 512 to 4096.

## From ADR 0035 Implementation notes (first GPU run, after ADR 0036, paired with llama.cpp; moved verbatim)

- **First GPU run (2026-11-01, second box, RTX 3090, Qwen3-30B-A3B Q4_K_M, `bench/results/20261101/builder-real.json` at commit cef79f2).** Correctness: `cpu_slots32` (pair groups, AVX2 host) is bitwise equal to the resident CPU builder; on CUDA, 32 slots prefill reaches top-1 agreement 0.922 (masked passes: 0.859) and single-token streaming stays exact; `Cpu` policy with 32 VRAM and 128 RAM slots splits segments across both backends (max splits 3), uploads nothing during decode and agrees on top-1 at 0.859 (CPU and GPU kernels round differently). Speed is far off the target: `Cpu` policy decode 11.5 tok/s and prefill 2.2 tok/s against llama.cpp `-ot` 40 to 67 and 460 to 2900; the own builder all on CPU decodes at 4.1 tok/s. The CPU path's cost has to be found before placement is tuned.
- **After ADR 0036 (threadpool, one-graph resident step, topk-moe fusion fixed; 2026-11-01, third box).** All real-model checks pass with CUDA fusion on: resident equals libllama, `slots120` and token-wise 32 slots equal resident bitwise. Resident decode 103 tok/s against libllama 127. `Cpu` policy (32 VRAM, 128 RAM slots): cold 3.2 prefill and 17.5 decode tok/s (the RAM tier fills from disk); warm 144 prefill (64 tokens) and 30.5 decode tok/s. Warm decode step 33 ms: compute 23, alloc 5.3, streamer 2, build 1.4, readback 1.1 (49 graphs per step).
- **Paired with llama.cpp (2026-11-01, fourth box, 9 threads from the CPU quota, 64-token prompt, 16 decode tokens, `scripts/ot_pair2.sh`, commit after 4e1661e).** llama-bench: full GPU 1267 prefill and 180 decode tok/s; `-ot` all experts on CPU 48 and 30.7; `-ot` layers 24 to 47 on CPU 102 and 41.1. Builder `Cpu` policy warm (32 VRAM slots, 128 RAM slots): 165 and 34.6, so between the two `-ot` placements in decode, with less VRAM than the second, and ahead in short prefill. Exit regime (32 RAM slots, disk reads every step): 14.5 and 17.1 tok/s, decode step streamer time 29 ms of 59. The disk path is the next target.

## From ADR 0036 Context (moved verbatim)

With every expert resident the segments buy nothing: the host only copies the router's ids back into the next segment. They still cost a context, a graph build, a scheduler allocation and a compute launch per layer, and on a GPU a synchronization per layer (the resident GPU builder decoded 45 tok/s against libllama's 67 on the same box).

## From ADR 0036 (implementation note "CUDA graphs and the resident step (2026-11-01)", opening, moved verbatim)

On the RTX 3090 the resident builder decoded about 104 tok/s against libllama's 127 (`tests/real_moe.rs`, `resident` records routing, a confound handled on another branch). Reading ggml-cuda at the pin (b11976) shows why the builder never gets CUDA graphs:

## From ADR 0039 Context (moved verbatim)

llama.cpp allocates CPU-resident Q4_K weights in ggml's `CPU_REPACK` extra buffer type, which interleaves 8 rows (`q4_K_8x8_q8_K`, AVX2) on write; the builder's CPU buffers and RAM tier are plain. ADR 0036 left this open. Measured on the box (bench/results/20261101/cpu-bench-real.json, EPYC 7K62, 11 threads, Qwen3-30B-A3B Q4_K_M, CPU only): the builder against libllama (which repacks) is 0.82x on prefill and 1.30x on decode (`builder_cpupolicy` 0.78x and 1.22x).

## From ADR 0039 Decision (moved verbatim)

Keep the builder's CPU weights and RAM tier unpacked. In the GPU-plus-RAM regime that Phase 2 targets, repacking would turn off op offload for the RAM tier (prefill several times slower) to speed up batch-1 CPU groups, where the builder already decodes 1.3x libllama on CPU. The CPU-only prefill gap (0.82x) is real but outside the Phase 2 exit test.
