# 0045. Phase 2 exit verdict: Qwen3-235B-A22B larger than RAM, all baselines on the same box

Status: Accepted (Phase 2 exit). Builds on ADR 0035, 0037, 0038, 0044.

## Context

Phase 2 exit test (plan.md, roadmap row 2): logit parity with libllama, and Qwen3-235B-A22B Q4_K_M (142 GB) on a box with at most 128 GB RAM at least as fast as the best tuned baseline that runs it (Colibri, llama.cpp with expert overrides, ik_llama.cpp; the user's goal adds KTransformers), with the per-layer full-hit rate and decode right after a 32K prefill.

## Setup

- Two boxes, every engine paired on the same box, same threads (CPU quota 15), same frozen prompt (`bench/prompts/exit_prompt.txt`, token hash `c743c72b8ffceadd`), 32768 prompt tokens, 64 greedy tokens, `scripts/exit_run.sh`:
  - **A**, box 58900330: RTX 3090, Ryzen 9 5950X, 91.6 GB container RAM, NVMe (5 to 8 GB/s direct reads). `bench/results/20261103/*-013730|021753|030753|032306.json`.
  - **B**, box 58930011: RTX 3090, Xeon E5-2683 v4, 45.7 GB container RAM (model about 3x RAM), slower disk (2 GB/s). `*-070656|080404|085333|090438.json`.
- Expert VRAM is equal: `scripts/llama_fit.sh` finds the most layers' experts llama.cpp fits on the 24 GB card at llama-bench's 32K context and ubatch 2048 (7 layers, `blk.[0-6].ffn_.*_exps=CUDA0;exps=CPU`; 8 fit llama-server but not llama-bench), and the builder gets the same expert bytes as VRAM slots (10.97 GB, 10 slots per layer). The builder's RAM tier is the container RAM minus 8 GB.
- llama.cpp b11976 runs through libllama on the same token ids (`TETRAEDER_EXIT_ENGINE=libllama`, mmap, the fitted `-ot`), so it is also the logit reference. llama-bench (random token ids) is run as well on A.
- KTransformers 0.7.2 (kt-kernel LLAMAFILE on the GGUF experts, sglang-kt, `scripts/ktransformers_*.sh`): all experts on the CPU (its GPU experts need HF weights, which a GGUF-only setup lacks), bf16 attention dequantized from the GGUF, fp8 KV (bf16 KV does not fit at 32K).
- Colibri v2.1.0 (`scripts/colibri_exit.sh`): 2K profile only; its 32K prefill would take about 4 h (2046 tokens took 895 s on the 5-thread box).
- ik_llama.cpp: no number. It was OOM-killed at 128 GB on 2026-11-02 (it does not keep CPU-overridden experts mapped, so the 142 GB must fit in RAM; `--no-mmap` and run-time repack need that too). Its llama-bench also lacks `-d` and `;`-joined `-ot` (both fixed in the stage, not run again).
- FreeToken: not runnable here: every expert must sit in pinned host RAM, and it needs CUDA 13.

## Results

Decode and prefill in tok/s after a 32K prefill unless noted:

| Box | Engine | prefill | decode | builder decode ratio |
|---|---|---|---|---|
| A | **builder** (prefetch on) | **75.8** | **3.14** | |
| A | builder, prefetch off | 72.2 | 2.83 | |
| A | llama.cpp `-ot`, same tokens | 46.3 | 2.55 | 1.23 |
| A | KTransformers | 35.7 | 3.15 | 1.00 (0.998) |
| A | Colibri (2K depth) | 3.2 | 0.93 | |
| A | builder, random ids | 93.8 | 4.82 | |
| A | llama-bench, random ids (pp2048 / tg32 at depth 32K) | 49.0 | 3.53 | 1.37 |
| B | **builder** (prefetch on) | **26.4** | **0.69** | |
| B | llama.cpp `-ot`, same tokens | 21.9 | 0.43 | 1.60 |
| B | KTransformers, 2K depth | 8.7 | 0.19 | builder at 2K: 13.4 / 0.56 |

- KTransformers' 32K prefill on B was still in its first 4096-token chunk after 11 minutes and was stopped to stay within the 3 h box cap, so B pairs it at 2K (its 2K prefill rate, 8.7 tok/s, puts 32K near an hour).
- Random token ids route very differently from text (builder RAM hit 97.6% against 93.2%, 129 against 511 MB read per token), so the random-id pair is reported for comparison with llama-bench only; the text pairs are the verdict.
- Builder decode on A: 314 ms per token, compute 225, streamer 77; prefetch accuracy 0.85, 59% of the read time hidden. Per-layer full-hit rate 0.0 in every layer on both boxes (10 of 128 experts per layer in VRAM: every layer step routes at least one expert to the CPU); RAM-tier hit rate 0.93 (A) and 0.80 (B).
- **Parity.** The builder's 64 greedy tokens equal libllama's on both boxes (same token ids, 32K). The first token's top-2 ids agree; top-1 logit 39.49 (A) and 36.23 (B) against libllama's 38.72; top-1 margin 13.2 and 14.1 against 12.9, no near-tie. The 235B top-1 logit is off by 2% (A) and 6% (B), and the offset differs between boxes; the cause is open.

## Decision

- **Pass, with decode parity against KTransformers on box A.** The builder is faster than every baseline on prefill (1.6x to 3x) on both boxes and on decode against llama.cpp with tuned expert overrides (1.23x and 1.60x), Colibri and KTransformers on box B. On box A its decode equals KTransformers' (3.14 against 3.15, one pair each, inside run-to-run noise). ik_llama.cpp and FreeToken cannot run the model within 128 GB.
- The exit numbers stand on the paired text runs; llama-bench's random-id numbers are not used as the bar.

## Consequences

- Open, in order of expected gain: prefetch depth 2 (ADR 0038 Levers; implemented, not yet measured on the 235B: box B's disk made each 32K run 25 min and the window ran out), `cpu_poll`/`prefetch_threads`, a second paired KTransformers run on box A's class for a noise band.
- Open: the 235B top-1 logit offset against libllama (above). Greedy output is unaffected over 64 tokens on both boxes. The boxes differ in RAM slot count, which changes how the prefill's pair groups are split; that is the first thing to check.
- Not taken (the coordinator's FreeToken levers): splitting decode misses between CPU and PCIe gains little here because the CPU expert path already reads about 8 GB of weights per token in 225 ms, near the boxes' DRAM bandwidth, and PCIe DMA reads the same DRAM; prefill double buffering is not needed while prefill leads every baseline by 1.6x or more.
- Box cost of the exit work on 2026-11-03: about 2.9 USD (58894536 0.14, 58897870 and 58899229 about 0.05, 58900330 0.97 plus 0.92 traffic, 58930011 0.83, no traffic).

## Addendum (2026-11-05): llama.cpp's best setup was not measured

The llama.cpp rows above use `-ot`. The pinned build (b11976) also has a GPU cache of host experts (`--moe-cache-mib`, `llama_context_params.moe_cache_size`, ggml-org/llama.cpp#29887), which that pull request reports at 1.57x to 2.20x over `--fit` on other hardware and models. So "decode beats llama.cpp" here means beats `-ot`, not the best tuned llama.cpp the exit test asks for. The cache may help less at this regime's 8% of experts in VRAM, but only a paired run can say. `llama-bench` has no flag for it: run it through the libllama reference engine of `exit_regime` (needs the parameter set) or `llama-server`. Until then the llama.cpp part of this verdict is open. Since ggml-org/llama.cpp#30966 (merged 2026-11-07, after our pin) the cache can also compute a share of misses on the CPU (`--moe-cache-cpu-share`, tuned by `scripts/moe-cache-tune.py`); the baseline to beat is that tuned pair, built from a commit after the PR.

## Addendum (2026-11-03, box 58958565): the decode tie does not hold at a 26-core quota

Box C, 58958565: RTX 3090, Xeon Gold 6330N, 26-core quota, 91.4 GB container RAM, direct reads at about 5.2 GB/s. Same frozen prompt, 32K, 7-layer `-ot` and 10.97 GB of builder expert slots (`bench/results/20261103/*-120800.json`):

| Engine | prefill | decode |
|---|---|---|
| builder, prefetch on (two runs) | 72.4, 71.8 | 3.77, 3.73 |
| builder, prefetch off | 69.8 | 3.66 |
| KTransformers 0.7.2 | 50.1 | 5.04 |
| ik_llama.cpp | killed after 22 s (out of memory) | |

- **Decode is 1.34x behind KTransformers on this box.** Builder decode per token: 257 ms, of which compute 112 and the streamer 128. The streamer reads 511 MB per token: 414 MB by the prefetch reader at 3.2 GB/s, with only 28% of that time hidden; the rest are synchronous misses. KTransformers reads the same experts through the page cache with 26 faulting threads. With 15 threads (boxes A and B) the builder's compute dominated and the reads hid better; at 26 threads compute halves and the reads become the step.
- **Verdict, revised:** Phase 2 passes on prefill on every box (1.4x to 3x), and on decode against llama.cpp, Colibri and ik_llama.cpp. Against KTransformers it passes only on 15-thread boxes: a tie on A, ahead on B. At a 26-core quota KTransformers decodes 1.34x faster, so the decode clause is not met there. The prefetch levers below are the work to close it; they are measured when boxes are allowed again.
- The lever stages of this session (`d2`, `d2poll0`, `d2fewer`) ran with prefetch off: the harness defaulted `TETRAEDER_EXIT_PREFETCH` to off while `LoadOptions` defaults to on (fixed). From those runs: prefetch on is +3% here (3.77 against 3.66), and `cpu_poll = 0` costs 90 ms of compute per token (2.72 tok/s), so ggml's default spin stays.
- The speculative pair (Track E's `spec_run.sh main-builder`) did not start: the draft's context did not fit beside 10.97 GB of builder slots (fixed in the script).

### Logit offset against libllama (Rule 2)

2K prompt, same token ids (`exit-drift-*-120800.json`), top-1 logit:

| run | top-1 |
|---|---|
| libllama, ubatch 2048 / 512 | 27.710 / 27.778 |
| builder, prefetch off / on | 27.638 / 27.638 |
| builder, 30 GB RAM tier | 27.389 |
| builder, RAM groups on the CPU (no op offload), prefetch off / on | 28.468 / 28.468 |
| CUDA fusion off: builder / libllama | 27.915 / 27.904 |

- With CUDA fusion off the builder equals libllama to 4e-4. With fusion on, both engines move with how ops are batched: libllama by 0.07 between ubatches, the builder by 0.25 between RAM slot counts, which change how its pair groups split.
- The CPU paths are exact for any RAM slot count, with prefetch off and at depths 1 and 2 (`cpu_long_prefill_is_exact_for_every_ram_slot_count`). On a GPU the synthetic fixture is bit-identical over RAM slots and prefetch (`gpu_cpu_policy_logits_do_not_depend_on_ram_slots_or_prefetch`, `gpu-tests-120800.log`). Expert bytes do not depend on the configuration.
- At 32K the offsets grow over 16 ubatches but are deterministic per configuration across boxes: prefetch off gives 37.892 on boxes A and C, prefetch on 39.486 on both. Prefetch changes which experts are RAM hits, hence the hits-first chunk composition and the batches the fused CUDA kernels see.
- Greedy output equals libllama's in every pair. Conclusion: rounding of fused GPU kernels over a different op batching, not a defect. Open: the small-RAM run with fusion off was cancelled with the session hold; it should equal 27.915.

## Addendum (2026-11-09): Diagnosing the KTransformers decode gap

On the 26-core box KTransformers decodes 5.04 tok/s (198 ms per token) against our 3.77 (257 ms: compute 112, streamer 128). We only measured KTransformers end to end. From its source (v0.7.2): experts run in place from an `np.memmap` of the GGUF (no repack, one `MADV_WILLNEED` at load, no prefetch); 26 pinned, spinning workers take 32-row tasks (384 gate/up and 128 down per layer), so misses page-fault in parallel inside the compute; the kernels are iqk Q4_K x Q8_K, VNNI on AVX-512, chosen at runtime from `/proc/cpuinfo` (`KT_KERNEL_CPU_VARIANT` overrides). Our run used `--kt-num-gpu-experts 0` and no deferral, so GPU experts and deferred experts are ruled out. Our box builds use `TETRAEDER_GGML_NATIVE=OFF`, so our CPU experts run AVX2 kernels on an AVX-512 Xeon.

Hypotheses:

1. **Parallel faulting I/O.** 26 threads fault pages concurrently inside compute, so the disk queue is deep and I/O overlaps compute; we have one reader plus synchronous misses.
2. **Page cache against a fixed RAM tier.** KTransformers' cache grows to the container limit (peak RSS 100 GB against our 83 GB budget), so more experts are resident and fewer bytes are read per token.
3. **Kernel ISA.** AVX-512 VNNI iqk kernels against our AVX2 ggml kernels.

Minor, no stage: fp8 KV on the KT side. The sampler's GPU busy % shows whether the GPU part matters at all.

### What is measured

All stages are written, not run (box runs are on hold). `scripts/engine_sample.py` samples a process tree once a second and fills the same `host` block in the result JSON of the builder (every `exit_regime` stage, `exit-*.json`) and of KTransformers (`exit-ktransformers*.json`), for prefill and decode apart: read bytes, read ops, read bytes per token, read MB/s, disk queue depth (mean, from the time-weighted in-flight counter of `/proc/diskstats`) and in-flight max, major faults and faults per token, CPU busy % (cgroup CPU time over wall time times threads, and the mean of the busiest CPUs of `/proc/stat`), GPU busy % and memory, RSS max, page cache (cgroup `file`) and anon max. `env` has the CPU model and avx512 flags; the builder's `env.ggml_system_info` is the ISA ggml was compiled with. KTransformers has `kt_cpu_variant` (what kt-kernel picked) and `kt_cpu_variant_forced`. The builder reads with direct I/O, so its major faults are near zero by design; KT's are its reads.

Stages (`scripts/exit_session.sh`): `main` and `main-native` (same args, ggml built with `TETRAEDER_GGML_NATIVE=ON` in its own target dir), `main-repeat` and `main-native-repeat`, `ktransformers` and `ktransformers-avx2` (`KT_KERNEL_CPU_VARIANT=avx2`), and `kernels-cpu` (`scripts/kernels_cpu.sh`: CPU-only decode of Qwen3-30B-A3B Q4_K_M with mainline llama.cpp, ik_llama.cpp and the llama.cpp baseline build; all three are portable AVX2 builds). Before a box: `crates/ggml-sys/tests/expert_matmul.rs` (Q4_K `mul_mat_id`, Qwen3-235B shapes, one token), built with native OFF and ON by the `expert-bench` workflow. It bounds what AVX-512 buys in the expert matmul alone.

### How to read the numbers

Compare only pairs from one session. The gap is 59 ms per token.

| Observation | Verdict |
|---|---|
| `main-native` decode at least 10% above `main`, with the compute per token (`decode_profile_per_token`) down by a third of the gap or more | Hypothesis 3 holds on our side. If it closes most of the gap, the fix is runtime ISA dispatch (`GGML_CPU_ALL_VARIANTS` with `GGML_BACKEND_DL`), not a native build: the box image is built on a CI runner and must run on any host. |
| `main-native` within 3% of `main` | Hypothesis 3 rejected for the compute path as built. |
| `ktransformers-avx2` more than 15% slower than `ktransformers` (check `kt_cpu_variant` differs) | The ISA matters to KT too; KT on AVX2 against our AVX2 isolates the kernel code from the ISA. |
| `ktransformers-avx2` about as fast as `ktransformers` | The ISA is not why KT is fast; look at I/O. |
| `kernels-cpu`: ik beats mainline by a lot at the same AVX2 ISA | The kernel code (iqk) is a lever independent of the ISA. Check `zmm_instructions` to confirm none of the three has AVX-512 code. |
| `kernels-cpu`: ik within 10% of mainline | Kernel code is not the lever; the ISA (via `main-native`) is the only compute lever. |
| KT decode `read_bytes_per_token` about equal to ours (about 511 MB), KT `read_mb_s` higher, KT `disk_queue_depth_mean` much deeper than ours, `majflt_per_token` in the thousands, `cpu_busy_pct` near 100% | Hypothesis 1: same bytes, but faults overlap compute and fill the disk queue. The lever is concurrency of our misses (several readers, or reads issued from the compute threads), not more cache. |
| KT `read_bytes_per_token` well below ours (under 70%), KT `cache_file_gb_max` above our RAM tier, its `read_mb_s` not higher | Hypothesis 2: more resident experts. The lever is a larger RAM tier (smaller reserve, bounce buffers counted) or the page cache for the tail. |
| Our `main` decode has `disk_queue_depth_mean` near the reader depth (16) and `read_mb_s` at the device rate (5.2 GB/s) | The disk is saturated: only fewer bytes (hypothesis 2, ADR 0050 a to c) help, not deeper queues. |
| KT bytes per token similar to ours, KT disk throughput at or below ours, CPU busy lower than ours | Neither I/O hypothesis; the gap is in compute, back to 3. |

Order: `main-native` against `main` first (pre-box estimate: the microbench's native over portable ratio times the CPU-expert share of the 112 ms compute), then bytes per token and queue depth to separate 1 from 2. The effects can add up, so a partial result from each is still a result.

### Runtime ISA dispatch

`crates/ggml-sys/build.rs` builds ggml static (`BUILD_SHARED_LIBS=OFF`) and links `ggml-cpu` and `ggml-base` as static libraries. `GGML_CPU_ALL_VARIANTS=ON` needs `GGML_BACKEND_DL=ON`, which builds the backends as loadable `libggml-cpu-<isa>.so` plugins that ggml picks from at start; so it needs a shared build (and CUDA as a loadable backend too), a plugin directory next to the binary and a linking change in `build.rs`. That is a build and packaging change for its own ADR; it is not implemented here.

## Addendum (2026-11-10, box 60042169): verdict, met on the means

Session `b1-exit`: RTX 3090, Xeon Gold 6330N (Ice Lake, AVX-512 VNNI/VBMI), 91.4 GB cgroup RAM, quota 26.9 cores, 26 threads for every engine, ggml native, `MOE_CPU_SHARE=0.9` (B3 tuner). Qwen3-235B-A22B, 32K prefill, 64 decode, same prompt hash for builder and llama.cpp. Paired ratios on this box only:

| | Prefill tok/s | Decode tok/s |
|---|---|---|
| Builder (prefetch, repeat) | 69.5, 70.3 | 3.56, 3.31 |
| KTransformers (avx512_vbmi, repeat) | 36.6, 39.9 | 3.40, 2.70 |
| llama.cpp expert cache, share 0.9, mmap | 10.96 | 2.10 |
| llama.cpp expert cache, `--load-mode dio` | OOM-killed at load | |

- Prefill: 1.83x KTransformers, 6.4x the llama.cpp cache. Met.
- Decode: 1.13x KTransformers on the means (pairs 1.05 and 1.23) and 1.57x the llama.cpp cache. KTransformers spreads 23% between its two runs, the builder 7%, so the margin over KTransformers is at worst a tie within noise. Met on the means; more pairs in `b1b-levers` will firm it up.
- dio needs the CPU-share weights in anonymous memory, which cannot fit 142 GB into 91 GB, so mmap is llama.cpp's best here. Its mmap run thrashes (515k major faults in the timed run, prefill about 50 min).
- Host sampler, decode: the builder reads 570 MB per token at queue depth 34 with the CPU 96 to 99% busy; KTransformers reads 146 to 250 MB per token at depth 13 to 21 with 2 to 3 thousand major faults per token, CPU 55 to 70%.
- The 2026-11-03 box (KT 1.34x ahead) had another CPU and AVX2 builds of our ggml; since then boxes build ggml native with the iqk kernels (ADR 0052, 0053). Absolute numbers do not compare across boxes; this verdict is for this box class.
- Not run (deadline guard): `ktransformers-avx2`, the llama.cpp cache repeat. Results: `bench/results/20261110/exit-*.json`, `disk-curve-*.json`.

## Addendum (2026-11-10, box 60057461): an AVX2 box needs the ADR 0054 overlap

Session `b1b-levers`: AMD EPYC 7742 (AVX2 only, so no iqk VNNI kernels on either engine), 26 threads, 91 GB. KTransformers ran its `avx2` variant (spread 6.3%). Decode builder/KT: `main-prefetch` 0.92, `main-repeat` 0.89, so the b1-exit path is behind on this CPU class; with ADR 0054's I/O service and overlap 1.27 and 1.21 (`main-iosvc` 1.21). Prefill 2.1x to 2.6x. KT decodes with 171k major faults per run and reads 161 to 171 MB per token; the builder reads 550 to 565 MB.

- Not pooled with b1-exit: different CPU class.
- The verdict holds on both classes only with overlap, which `serve` now enables by default (ADR 0054 addendum). The builder's I/O workers used CPU beyond the 26 threads, which KTransformers did not; a quota-tight run (quota about threads plus one, as b1-exit) of `main-overlap` against KTransformers closes that question.

## Addendum (2026-11-10, box 60072027): met with the CPU matched

EPYC 7742 (AVX2), RTX 3090, 91 GB, both engines at 30 threads = floor(quota), decode CPU 27.5 to 29 cores on both. Decode `main-overlap`/KTransformers 1.143 and 1.181 (cross pairs 1.158, 1.166; overlap pair 1.3% apart, KT pair 2.0%, A/A 0.996). Prefill 2.3x. `main-prefetch` alone is 0.757 of KTransformers here: KTransformers gains more from the extra threads, so the overlap path (on by default in `serve`, ADR 0054) is what meets the bar on AVX2. Verdict so far: met against KTransformers on this AVX2 box with the builder at its `serve` defaults; on the AVX-512 box (b1-exit) the prefetch path ties KTransformers within noise (1.13x on the means) and overlap is unmeasured there. The llama.cpp expert cache ran one hand-picked CPU share (0.9; the tuner gave none) in mmap. A closing AVX-512 run (`20261110f`) pairs overlap with KTransformers and adds share 0. Results `bench/results/20261110e/`.

## Addendum (2026-11-10, box 60083507): closing run against the llama.cpp expert cache

No RTX 3090 offer had an AVX-512 CPU, so the run used the b1c host again: EPYC 7742 (AVX2), RTX 3090, 91 GB, 30 threads = floor(quota) for both engines, KTransformers not rerun. llama.cpp `-cmoe --moe-cache-mib 10458` (the builder's slot bytes), mmap, same prompt hash. Decode `main-overlap`/cache: 4.65, 4.59, 4.60 at share 0; 1.66 (adjacent pair), 1.68, 1.66 at share 0.9 (cache 1.22 and 3.36 tok/s). Prefill 2.8x to 2.9x. Three overlap runs 1.4% apart in decode, greedy tokens equal; overlap and `cpu_experts` GPU parity tests pass. Builder decode used 28 to 30 cores, hidden share 0.41 to 0.43. Verdict: Phase 2 met against KTransformers and the llama.cpp expert cache at `serve` defaults on AVX2; on AVX-512 the prefetch path ties KTransformers and overlap is unmeasured against KTransformers (see the next addendum). Results `bench/results/20261110f/`.

## Addendum (2026-11-10, box 60101282): AVX-512 attempt

Xeon Gold 6138 (Skylake AVX-512, no VNNI), RTX 3090, cgroup 137 GB ballooned to 91, 38 threads = floor(quota). KTransformers did not start: sglang hung at "Capture cuda graph bs [1]" for 40 min, no OOM, about 40% iowait. Its LLAMAFILE backend reads the GGUF experts in place through mmap (peak RSS 23.8 GB; the captured log tail shows MoE layer setup through layer 84 of 94), so the capture's forward passes page in experts with small faulting reads; on this disk that did not finish in the 40 min stage timeout. Likely cause: the disk, not a KTransformers bug, since the same command ran on the 3 GB/s boxes. A retry needs a fast disk under random reads, not only a good sequential number. The host was disk-starved (decode reads 675 to 863 MB/s, prefill about 410 MB/s; earlier boxes 2.8 to 3.0 GB/s), so absolute numbers do not compare across boxes. Builder only: overlap/prefetch decode 1.27 (1.574 vs 1.241 tok/s), prefill 1.03, greedy tokens equal, hidden share 0.41; overlap and `cpu_experts` GPU parity tests pass. Overlap against KTransformers on AVX-512 remains unmeasured. Results `bench/results/20261110h/`.

## Addendum (2026-11-11, box 60151711): AVX-512 at `serve` defaults

A10 24 GB, Xeon Gold 6342 (Ice Lake, AVX-512 VNNI), host 209278, ballooned to 91 GB, 30 threads for both engines; KTransformers ran its avx512_vbmi kernels. Decode `main-overlap`/KTransformers: adjacent 1.124 and 1.038, cross 1.043 and 1.119, means 4.283 vs 3.967 tok/s = 1.08. Overlap pair 0.4% apart, KTransformers pair 7.8%. Prefill 1.11 and 1.21. Greedy tokens equal across the builder runs; overlap and `cpu_experts` GPU parity tests pass. KTransformers read 160 to 180 MB per token through mmap at 84% CPU; the builder 585 MB per token at 2.4 GB/s at 99% CPU. Verdict: Phase 2 met at `serve` defaults on both CPU classes measured; on AVX-512 the margin is within the baseline's spread. Results `bench/results/20261111a/`.

## Addendum (2026-11-11, box 60190564): repeat on a slow disk

Same host (209278, A10, Xeon Gold 6342), new instance, 91 GB balloon, 30 threads, `VRAM_SLOT_BYTES` 8 layers (the 3090 default of 10 fails `graph tensors` on the A10; `session.sh` now picks 8 below 24000 MiB). The disk read 0.5 to 0.7 GB/s during decode (smoke 2.3 GB/s; `20261111a` 27.8 GB/s) with no contending process; cause unknown. Decode overlap 0.888 and 0.974 tok/s, KTransformers 1.600 and 1.559: ratios 0.555 and 0.624 adjacent, 0.569 and 0.609 cross. Prefill overlap 37.0 and 37.4, KTransformers 19.8 and 20.7 (1.8x). KTransformers read about 250 MB/s through mmap and kept hot experts in the page cache; the builder read 500 to 570 MB/s with direct I/O, `read_wait_ms` 810 to 890 per token. Greedy tokens equal across the builder runs; GPU parity tests pass. The llama.cpp tuner did not run (no time left on the box). Verdict: in this disk-bound regime KTransformers decodes about 1.6x faster. Bytes per token and read rates: the builder 594 MB per token at 515 MB/s with read waits 80% of its step; KTransformers 166 MB per token at 259 MB/s, half the rate the disk gave the builder, so KTransformers was not at the disk's bandwidth (were both, the ratio would be 0.28, not 0.56 to 0.62). The builder reads about 3.5x the bytes in the 64-token window, about 500 MB of its 582 as prefetch; KTransformers read 146 to 249 MB per token on every box since `20261110`, at the same 98 to 100 GB peak RSS and 83 GB file cache, so the gap is not new, only exposed. The disk rate at which the builder stops winning is not measured. Candidate lever, not designed: fewer decode bytes when the disk is the bound; the `20261110c` replay at 77 RAM slots gives 186 MB per token without prefetch against 262 with it (256 tokens). Results `bench/results/20261111b/`.

## Addendum (2026-11-11, box 60206988): llama.cpp cache at share 0.5, tuner OOM

RTX 3090, AMD EPYC 7642 (AVX2), host 706255, 46 threads, ballooned to 91 GB, disk 7.3 GB/s at depth 16 (gate passed). Stage `moe-tune` failed: llama.cpp's tuner sizes its first probe at 10% of the expert bytes (13048 MiB on the 235B) regardless of `--max-mib`, and with the f16 KV at 32K the CUDA compute buffer did not fit (rc 1, no report). `exit_run.sh` now runs the tuner at `TUNE_CTX` 8192. `llamacpp-cache` ran the session fallback, 10458 MiB at share 0.5: decode 1.440 and 1.562 tok/s against overlap 3.460 and 3.555, ratios 2.40 and 2.28 adjacent, 2.21 and 2.47 cross; prefill 65 vs 28 tok/s. With shares 0 and 0.9 (`20261110f`) the hand sweep now covers 0, 0.5 and 0.9 at the largest cache that fits; the builder leads each by 1.66x or more. KTransformers (kt-kernel 0.7.2, AVX2) decoded 0.523 and 0.529 tok/s with a 1453 s time to first token, against 4.5 to 4.7 on the EPYC 7742 at 30 threads; ratios 6.5 to 6.8. That is an outlier for KTransformers on this host, not a builder result, and does not enter the verdict. Its host block shows why: decode read 69 to 79 MB/s at queue depth 0.37 with 56% CPU and about 2,600 major faults per token, so it waited on page faults served one at a time (about 0.7 ms each), not on disk bandwidth or thread count; on the EPYC 7742 the same fault count cost about 0.08 ms each. Builder runs 2.7% apart, greedy tokens equal, same prompt hash as the cache runs. Results `bench/results/20261111c/`.

## Addendum (2026-11-11): owner decision on slow disks; correction

Correction to the box 60190564 addendum: the extra decode bytes are not mainly prefetch. On `20261110c` (same box and options) the 64-token runs read 549 MB per token and the 256-token trace run 262, so tokens 65 to 256 read about 167 MB per token, the same as KTransformers; without prefetch the first 64 tokens still read 443 (`20261103`). The builder's excess is warm-up of the RAM tier in the first tokens after the 32K prefill; KTransformers' page cache shows none. Cause under diagnosis; the replay sentence above stands as data, not as the lever.

Decision (owner, 2026-11-11, after the runs): slow disks are not calculated out of the result, they bound its scope. Phase 2 is met in the measured regime where read waits take at most 18% of the builder's decode step (`read_wait_ms / total_ms` of the overlap runs: 0.05 to 0.18 on every passing box, disks reading 1.8 to 3.0 GB/s during decode); at 0.80 (`20261111b`, 0.5 to 0.6 GB/s) KTransformers decodes 1.6x faster, recorded as a known limitation. Between the two there is no paired KTransformers run (`20261110h`, 0.21, has none). `decode_io.read_gb_s` is not a disk rate (its bytes include prefetch, its time only the waited reads) and is not used for scope. The exit run keeps the disk gate (loaded depth 16 at least 5 GB/s) as a box filter, not as a measured boundary. Follow-up after M1: the post-prefill warm-up, starting with an ADR and a failing test.

## Addendum (2026-11-11, box 60236780): tuner too slow for the 235B; thin margin at 23 threads

RTX 3090, AMD EPYC 7642, host 130210, CPU quota 23 (the B1 profile asks 24 to 26), 91 GB balloon, disk 6.4 GB/s at depth 16. llama.cpp's tuner at 8K context fit this time but timed out (rc 124 after 40 min) inside its first probe: that probe generates 2,048 tokens (warm-up at about 1.2 tok/s, 29 min) at a 13,048 MiB cache, above `--max-mib`; by estimate its six probes need over 2 hours (not measured beyond this 40-minute limit). The tuner fits per-machine rates, so its pick is only meaningful paired on the box it ran on; it always picks the largest cache that fits and the lowest share within 5% of the fastest of 0, 0.5, 0.8, 0.9, 0.95 and 0.98. `llamacpp-cache` at 10,458 MiB, share 0.5: decode 1.403 and 1.344 tok/s against overlap 1.578 and 1.504; ratios 1.125 and 1.119 adjacent, 1.174 and 1.072 cross; spreads 4.7% (builder) and 4.4% (cache); prefill 2.0x. The builder spent 533 of 627 ms per token computing and 11 waiting on reads: at 23 threads it is CPU-bound, while the cache runs most expert work on the GPU, so the margin shrinks with cores (2.3x at 46 threads, `20261111c`). Share 0.5 is probably not the cache's best: on `20261110f` it went from 1.22 tok/s at share 0 to 3.36 at 0.9, and the only pairing at that share is 1.66x (30 threads). Shares 0.8 to 0.98 have not been paired at 23 to 26 threads, where a similar lift would put the cache ahead. Greedy tokens equal across the builder runs; prompt hash equal in all four runs. Results `bench/results/20261111g/`.

`20261111h`, same host 130210 (RTX 3090, EPYC 7642, quota 23, 91 GB balloon, disk 5.75 GB/s at depth 16), shares 0.8 to 0.98 at 10,458 MiB: `main-overlap` decoded 2.664 tok/s; the cache 1.547 (0.8), 1.631 (0.9), 1.993 (0.95), 1.832 (0.98). The builder leads at every share, 1.34x at the cache's best (0.95), 1.63x to 1.72x elsewhere; prefill 3.3x. Together with 0.5 on `20261111g` this covers the tuner's candidates above 0, so the grid stands in for its pick. Caveats: one builder run first, then the four cache runs over about two hours (only share 0.9 ran next to it); the repeat did not fit the deadline, so no noise band on this box. The builder ran 1.69x faster than on `20261111g` on the same Vast host (a different container) with the same binary, options, threads and bytes per token: decode compute 292 against 533 ms per token, prefill compute 22.5 against 37.8 s per ubatch, at the same CPU busy share. The difference is CPU-side and its cause is unknown (the `20261111g` tuner was killed by `timeout`, which signals its process group, so a surviving llama-server is unlikely but not ruled out). The cache's prefill did not change (19 to 21 tok/s both nights), and its decode cannot be compared across nights because the share differs. Until a repeat settles it, both nights' margins (1.07x to 1.17x at 0.5, 1.34x at 0.95) stand as single-session readings.

`20261112a`, the interleaved repeat (host 24953, RTX 3090, EPYC 7642, quota 23.04, 24 threads for both engines, 91 GB balloon, disk 6.89 GB/s at depth 16): builder, cache 0.95, builder, cache 0.95. Decode 2.016, 2.724, 1.751, 2.697 tok/s; builder over cache 0.740 and 0.649 adjacent, 0.748 and 0.643 cross; spreads 13% (builder) and 1% (cache). **The cache leads by 1.35x to 1.54x; Phase 2 against the llama.cpp cache at its best share is not met at 24 threads on this box.** It does not hinge on warm-up: the builder's last decode window ran 2.05 and 2.16 tok/s, still below the cache. Nothing was I/O bound (`not_io_bound`, `disk_util` 0.16 to 0.21); the builder read 5.5 to 6.2e8 bytes per token against the cache's 3.9e8 and spent 406 and 465 of 489 and 564 ms per token computing on the CPU (GPU busy 6%; the cache's 14%). Prefill stays ours, 1.9x. Prompt hash equal in all four; the cache loaded with mmap, about 0.5 million major faults per run. This repeated pairing outweighs the single pass of `20261111h`, whose 1.34x lead came with a builder 1.69x faster than the same build on the same host a session earlier. Results `bench/results/20261112a/`. Nothing was I/O bound (`io_class` `not_io_bound` for every run, `disk_util` 0.11 to 0.25); the builder was CPU-bound (291.7 of 368.3 ms per token computing, 11.5 waiting on reads). The cache stages had about 1.2 million major faults per share, the builder none. Prompt hash equal in all runs; the cache results carry no token ids, so greedy equality is unchecked. Results `bench/results/20261111h/`.
