# 0045. Phase 2 exit verdict: Qwen3-235B-A22B larger than RAM, all baselines on the same box

Status: Accepted (Phase 2 exit, M1). Builds on ADR 0035, 0037, 0038, 0044. Phase 2 is met: against the llama.cpp expert cache in two clean sessions on two hosts (`20261113b`, `20261113k`), and against KTransformers with the free VRAM as expert slots (`20261114`, `20261114b`, owner decision 2026-11-14). Known limitation: on a slow disk KTransformers decodes faster (`20261111b`).

Measurements live in the session READMEs (`bench/results/<session>/README.md`), not here.

## Context

Phase 2 exit test (plan.md, roadmap row 2): logit parity with libllama, and Qwen3-235B-A22B Q4_K_M (142 GB) on a box with at most 128 GB RAM at least as fast as the best tuned baseline that runs it (Colibri, llama.cpp with expert overrides, ik_llama.cpp; the user's goal adds KTransformers), with the per-layer full-hit rate and decode right after a 32K prefill.

- The first verdict, on two 15-thread boxes A and B, had the builder ahead on prefill everywhere, ahead on decode against llama.cpp `-ot`, Colibri and KTransformers on box B, and level with KTransformers on box A (`20261103`).
- On box C (26-core quota, same session) KTransformers decoded faster: with more threads the builder's compute shrank and its expert reads stopped hiding (`20261103`). That started the diagnosis below and the levers of ADR 0050 and 0052 to 0054.
- llama.cpp's best setup is its expert cache, not `-ot` (2026-11-05). Pairings against it swung across sessions and hosts until co-tenant load, the RAM balloon, cgroup throttling and the cache's own polling were controlled (History).

## Setup

- Every engine paired on the same box, same threads, same frozen prompt (`bench/prompts/exit_prompt.txt`, token hash `c743c72b8ffceadd`), 32768 prompt tokens, 64 greedy tokens, `scripts/exit_run.sh`. The hardware of each box is in its session README (boxes A, B and C: `20261103`).
- Expert VRAM is equal against llama.cpp: `scripts/llama_fit.sh` finds the most layers' experts llama.cpp fits on the 24 GB card at llama-bench's 32K context and ubatch 2048 (7 layers, `blk.[0-6].ffn_.*_exps=CUDA0;exps=CPU`), and the builder gets the same expert bytes as VRAM slots (10.97 GB, 10 slots per layer; below 24000 MiB of VRAM `session.sh` sets `VRAM_SLOT_BYTES` to 8 layers, since 10 fail `graph tensors` on an A10). The builder's RAM tier is the container RAM minus 8 GB.
- llama.cpp b11976 runs through libllama on the same token ids (`TETRAEDER_EXIT_ENGINE=libllama`, mmap, the fitted `-ot`), so it is also the logit reference. llama-bench (random token ids) is run as well on A.
- KTransformers 0.7.2 (kt-kernel LLAMAFILE on the GGUF experts, sglang-kt, `scripts/ktransformers_*.sh`): all experts on the CPU (its GPU experts need HF weights, which a GGUF-only setup lacks), bf16 attention dequantized from the GGUF, fp8 KV (bf16 KV does not fit at 32K).
- Colibri v2.1.0 (`scripts/colibri_exit.sh`): 2K profile only; its 32K prefill does not fit a box session (`20261102`).
- ik_llama.cpp: no number. It cannot run the model within 128 GB: it does not keep CPU-overridden experts mapped, so the 142 GB must fit in RAM (`--no-mmap` and run-time repack need that too; OOM-killed, `20261102`, and on box C, `20261103`). Its llama-bench also lacks `-d` and `;`-joined `-ot` (both fixed in the stage, not run again).
- FreeToken: not runnable here: every expert must sit in pinned host RAM, and it needs CUDA 13.

## Decision

Current state; the addenda that led here are listed under History.

**Verdict.** Phase 2 is met (M1, `docs/m1-open.md`): point 1 on the sessions named below, point 3 by the owner's 2026-11-14 rule, point 4 by the owner's 2026-11-11 rule. The exit numbers stand on the paired text runs; llama-bench's random-id numbers are not used as the bar. Per point:

- **Point 1, llama.cpp.** The baseline is llama.cpp's GPU cache of host experts (`--moe-cache-mib` at the builder's slot bytes) with a CPU share of misses, at its best runnable configuration on the same box: mmap at `--poll 0`, share 0.95. The builder runs at its 10.97 GB default, matched to llama.cpp's `-ot`. Met: the builder is ahead in every pair in two clean sessions on two hosts (`20261113b`, `20261113k`).
- **Point 3, KTransformers.** The builder runs with the free VRAM as expert slots (`main-overlap-vramfill`). KTransformers 0.7.2 on a GGUF setup places no experts on the GPU (its GPU experts need HF weights, which do not fit the 230 GB disk next to the GGUF), so this pairs each engine at its best runnable configuration on the same box, the rule already used for the llama.cpp cache. Met: the builder leads in every pair over two sessions on one host (`20261114`, `20261114b`). At the 10.97 GB default the two are level, recorded as such.
- **Point 4, slow disks** (owner, 2026-11-11): they bound the scope, not the result. Phase 2 is met in the measured regime where read waits take at most 18% of the builder's decode step (`read_wait_ms / total_ms` of the overlap runs). In a disk-bound session KTransformers decodes faster (`20261111b`), recorded as a known limitation; between the two no paired KTransformers run exists (`20261110h` has none). `decode_io.read_gb_s` is not a disk rate (its bytes include prefetch, its time only the waited reads) and is not used for scope. The exit run keeps the disk gate (loaded depth 16 at least 5 GB/s) as a box filter, not as a measured boundary. Follow-up after M1: the post-prefill warm-up of the RAM tier, starting with an ADR and a failing test.

**Pairing rules.** A stage that breaks one gives no ratio.

- Paired ratios from one session on one box only; absolute rates are not compared across sessions or boxes.
- The builder runs at its `serve` defaults (overlap on, ADR 0054; polling from the quota, ADR 0061). Both engines get the same thread count: by default the CPU quota, or on an AVX-512 CPU whose stages include KTransformers `KT_HEADROOM` (default 4) below it (`scripts/kt_threads.sh`); an explicit `THREADS` overrides it.
- Each baseline runs at its best runnable configuration on the box. For the llama.cpp cache the share grid 0, 0.5, 0.8, 0.9, 0.95, 0.98 stands in for its tuner: the tuner (`scripts/moe-cache-tune.py`, run at `TUNE_CTX` 8192 by `exit_run.sh`) does not finish on the 235B within a box session (`20261111g`).
- The RAM balloon holds the budget for every stage; a stage that ran without it is flagged and does not count (`20261112f`). The cgroup limit is at least 150 GB, so the balloon sets the 91 GB budget (`20261113g`, docs/box-driver.md).
- A stage flagged by the co-tenant guard (`scripts/cotenant_guard.py`) does not count (`20261112g`).
- A thrashing cache gives no ratio (`20261112c`, `20261113g`).
- A baseline run that is an outlier for that baseline on the host does not enter the verdict (KTransformers on `20261111c`).

**Not rerun.** In this regime these do not run, so they are not paid for again:

- llama.cpp `--load-mode dio` under the 91 GB budget: OOM-killed at load, since dio reads the CPU-resident experts into anonymous memory (`20261110`, `20261113c`). mmap at poll 0 is the cache's best runnable configuration.
- KTransformers GPU experts on a GGUF-only setup (they need HF weights).
- kt-kernel on compute capability 12.0 (RTX 5090): no wheel (`20261113h`).
- ik_llama.cpp and FreeToken (Setup).
- Host 269862: downloads too slowly for a full pairing within the 3 h box limit (`20261113e`).

**The 235B logit offset against libllama (Rule 2).** Greedy output equals libllama's in every pair. The top-1 logit offset is rounding of fused GPU kernels over a different op batching, not a defect: with CUDA fusion off the builder equals libllama, the CPU paths are exact for any RAM slot count, and at 32K the offset is deterministic per configuration across boxes (prefetch changes which experts are RAM hits, hence the chunk composition the fused kernels see) (`20261103`, box C). Tests: `cpu_long_prefill_is_exact_for_every_ram_slot_count`, `gpu_cpu_policy_logits_do_not_depend_on_ram_slots_or_prefetch`. Open: the small-RAM run with fusion off was cancelled; it should equal the fusion-off builder value.

## Diagnosing the KTransformers decode gap

Written 2026-11-09, before any stage ran. On box C KTransformers decoded faster than the builder (`20261103`); we had only measured it end to end. From its source (v0.7.2): experts run in place from an `np.memmap` of the GGUF (no repack, one `MADV_WILLNEED` at load, no prefetch); 26 pinned, spinning workers take 32-row tasks (384 gate/up and 128 down per layer), so misses page-fault in parallel inside the compute; the kernels are iqk Q4_K x Q8_K, VNNI on AVX-512, chosen at runtime from `/proc/cpuinfo` (`KT_KERNEL_CPU_VARIANT` overrides). Our run used `--kt-num-gpu-experts 0` and no deferral, so GPU experts and deferred experts are ruled out. Our box builds then used `TETRAEDER_GGML_NATIVE=OFF`, so our CPU experts ran AVX2 kernels on an AVX-512 Xeon.

Hypotheses:

1. **Parallel faulting I/O.** 26 threads fault pages concurrently inside compute, so the disk queue is deep and I/O overlaps compute; we have one reader plus synchronous misses.
2. **Page cache against a fixed RAM tier.** KTransformers' cache grows to the container limit, above our RAM tier (`20261103`), so more experts are resident and fewer bytes are read per token.
3. **Kernel ISA.** AVX-512 VNNI iqk kernels against our AVX2 ggml kernels.

Minor, no stage: fp8 KV on the KT side. The sampler's GPU busy % shows whether the GPU part matters at all.

### What is measured

All stages are written, not run (box runs are on hold). `scripts/engine_sample.py` samples a process tree once a second and fills the same `host` block in the result JSON of the builder (every `exit_regime` stage, `exit-*.json`) and of KTransformers (`exit-ktransformers*.json`), for prefill and decode apart: read bytes, read ops, read bytes per token, read MB/s, disk queue depth (mean, from the time-weighted in-flight counter of `/proc/diskstats`) and in-flight max, major faults and faults per token, CPU busy % (cgroup CPU time over wall time times threads, and the mean of the busiest CPUs of `/proc/stat`), GPU busy % and memory, RSS max, page cache (cgroup `file`) and anon max. `env` has the CPU model and avx512 flags; the builder's `env.ggml_system_info` is the ISA ggml was compiled with. KTransformers has `kt_cpu_variant` (what kt-kernel picked) and `kt_cpu_variant_forced`. The builder reads with direct I/O, so its major faults are near zero by design; KT's are its reads.

Stages (`scripts/exit_session.sh`): `main` and `main-native` (same args, ggml built with `TETRAEDER_GGML_NATIVE=ON` in its own target dir), `main-repeat` and `main-native-repeat`, `ktransformers` and `ktransformers-avx2` (`KT_KERNEL_CPU_VARIANT=avx2`), and `kernels-cpu` (`scripts/kernels_cpu.sh`: CPU-only decode of Qwen3-30B-A3B Q4_K_M with mainline llama.cpp, ik_llama.cpp and the llama.cpp baseline build; all three are portable AVX2 builds). Before a box: `crates/ggml-sys/tests/expert_matmul.rs` (Q4_K `mul_mat_id`, Qwen3-235B shapes, one token), built with native OFF and ON by the `expert-bench` workflow. It bounds what AVX-512 buys in the expert matmul alone.

### How to read the numbers

Compare only pairs from one session.

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

Order: `main-native` against `main` first (pre-box estimate: the microbench's native over portable ratio times the CPU-expert share of the compute per token), then bytes per token and queue depth to separate 1 from 2. The effects can add up, so a partial result from each is still a result.

Hypothesis 3 was answered by the expert microbench (ADR 0052). Since `20261110` the boxes build ggml native with the iqk kernels (ADR 0052, 0053); the overlap path (ADR 0054) met the bar on the AVX2 and Ice Lake boxes, and on Zen 4 the free VRAM as expert slots did (History).

### Runtime ISA dispatch

`crates/ggml-sys/build.rs` builds ggml static (`BUILD_SHARED_LIBS=OFF`) and links `ggml-cpu` and `ggml-base` as static libraries. `GGML_CPU_ALL_VARIANTS=ON` needs `GGML_BACKEND_DL=ON`, which builds the backends as loadable `libggml-cpu-<isa>.so` plugins that ggml picks from at start; so it needs a shared build (and CUDA as a loadable backend too), a plugin directory next to the binary and a linking change in `build.rs`. That is a build and packaging change for its own ADR; it is not implemented here.

## Consequences

- The first verdict opened, in order of expected gain: prefetch depth 2 (ADR 0038 Levers), `cpu_poll`/`prefetch_threads`, a second paired KTransformers run on box A's class for a noise band. They were followed up in ADR 0050, 0054 and 0061 and the sessions under History.
- Not taken (the coordinator's FreeToken levers): splitting decode misses between CPU and PCIe gains little here because the CPU expert path already reads near the boxes' DRAM bandwidth and PCIe DMA reads the same DRAM; prefill double buffering is not needed while prefill leads every baseline (`20261103`).
- The builder's own run-to-run spread is the weak side of the verdict sessions. Host noise moves it: CPU quota throttling (`20261112e`), memory bandwidth (`20261112f`) and co-tenants (`20261112g`).

## History

One bullet per addendum, in file order; the numbers are in the linked session READMEs.

- 2026-11-03, first verdict, "Pass, with decode parity against KTransformers on box A": the builder leads every baseline on prefill and on decode against llama.cpp `-ot`, Colibri and KTransformers on box B (`20261103`).
- 2026-11-05, "llama.cpp's best setup was not measured": the llama.cpp rows used `-ot`, but the pinned build (b11976) also has a GPU cache of host experts (`--moe-cache-mib`, `llama_context_params.moe_cache_size`, ggml-org/llama.cpp#29887), so the llama.cpp part of the verdict was reopened. Since ggml-org/llama.cpp#30966 (merged 2026-11-07, after our pin) the cache can also compute a share of misses on the CPU (`--moe-cache-cpu-share`, tuned by `scripts/moe-cache-tune.py`); the baseline to beat is that tuned pair, built from a commit after the PR. `llama-bench` has no flag for it, so it runs through `llama-server`. The cache may help less at this regime's 8% of experts in VRAM, but only a paired run can say. The PR reports 1.57x to 2.20x over `--fit` on other hardware and models. (No session directory yet.)
- 2026-11-03, box 58958565 (box C), "the decode tie does not hold at a 26-core quota": verdict revised; at a 26-core quota KTransformers decodes faster, so the decode clause was not met there. The harness had defaulted `TETRAEDER_EXIT_PREFETCH` to off while `LoadOptions` defaults to on (fixed). ggml's default spin stayed then (`cpu_poll = 0` cost compute on that box); ADR 0061 later sets polling from the quota. The logit offset was diagnosed (Decision) (`20261103`).
- 2026-11-09, "Diagnosing the KTransformers decode gap": hypotheses, host sampler and stages (section above).
- 2026-11-10, box 60042169, "verdict, met on the means" (`b1-exit`, AVX-512): met against KTransformers on the means, a tie within its spread; boxes now build ggml native with the iqk kernels (`20261110`).
- 2026-11-10, box 60057461, "an AVX2 box needs the ADR 0054 overlap" (`b1b-levers`): on AVX2 the prefetch path is behind KTransformers and the overlap path ahead, so `serve` enables overlap by default (ADR 0054); not pooled with b1-exit (`20261110c`).
- 2026-11-10, box 60072027, "met with the CPU matched": met against KTransformers on AVX2 at `serve` defaults, both engines at floor(quota) threads (`20261110e`).
- 2026-11-10, box 60083507, "closing run against the llama.cpp expert cache": builder ahead of the cache at shares 0 and 0.9 on AVX2 (`20261110f`).
- 2026-11-10, box 60101282, "AVX-512 attempt": KTransformers did not start on a disk-starved host; a retry needs a fast disk under random reads (`20261110h`).
- 2026-11-11, box 60151711, "AVX-512 at `serve` defaults": met on AVX-512, margin within KTransformers' spread (`20261111a`).
- 2026-11-11, box 60190564, "repeat on a slow disk": KTransformers decodes faster when the disk is slow; `session.sh` now sets `VRAM_SLOT_BYTES` to 8 layers below 24000 MiB (`20261111b`).
- 2026-11-11, box 60206988, "llama.cpp cache at share 0.5, tuner OOM": builder ahead at share 0.5; the tuner's first probe did not fit, so `exit_run.sh` runs it at `TUNE_CTX` 8192; the KTransformers runs were a fault-bound outlier and stay out of the verdict (`20261111c`).
- 2026-11-11, "owner decision on slow disks; correction": slow disks bound the scope (Decision, point 4). Correction: the builder's extra decode bytes after the prefill are warm-up of the RAM tier, not mainly prefetch (`20261110c`, `20261111b`).
- 2026-11-11, box 60236780, "tuner too slow for the 235B; thin margin at 23 threads": the tuner does not finish on the 235B in a box session; at 23 threads the builder is CPU-bound and its margin over the cache shrinks (`20261111g`). The session paragraphs filed under this heading:
  - `20261111h`: shares 0.8 to 0.98 at the same cache size; with share 0.5 the grid stands in for the tuner's pick. One pass, no noise band.
  - `20261112a`: the interleaved repeat; the cache at share 0.95 leads, so point 1 was not met at 24 threads on that box.
  - `20261112e`: the CPU quota throttles the builder with ggml's default polling; `cpu_poll` 0 removes it (ADR 0061, 0062).
  - `20261112f`: invalid, the RAM balloon died; the harness keeps the balloon alive and flags any stage that ran without it.
  - Note 2026-11-12: every llama.cpp cache run before `20261112g` used baseline b12211 (7d7f028), not b12274; between the two the decode structure (scheduler, context, graph) is the same apart from scheduler reuse (ADR 0064).
  - `20261112g`: inconclusive; a co-tenant saturated the host during the builder repeat, the first direct evidence for the builder's swings; origin of the co-tenant guard.
  - `20261113a`: builder ahead, but against an untuned cache (default polling, throttled); not a verdict.
  - `20261113b`: builder ahead in every pair against the cache at poll 0 in mmap; the point 1 verdict pairing.
  - `20261113c`: no pairing; dio OOM-killed at load, so mmap at poll 0 is the cache's best runnable configuration; builder stages failed the co-tenant guard.
  - `20261113d`, `20261113e`, `20261113h`: no verdict for point 3 (one KTransformers run on its amx kernel; a host too slow to download; no kt-kernel wheel for the RTX 5090).
  - `20261113g`: no ratio, the cache thrashed mmap under a cgroup limit equal to the budget; pairings now require a cgroup limit of at least 150 GB.
  - `20261113k`: point 1 confirmed on a second host.
  - `20261113o`: KTransformers ahead on Zen 4 with its bf16 kernel; the builder stages failed the co-tenant guard, so no clean verdict, but the direction was against us.
  - `20261114`: at the 10.97 GB default the builder and KTransformers are level; with the free VRAM as expert slots the builder was ahead in one run.
  - `20261114b`: with the free VRAM as expert slots the builder leads in every pair.
- 2026-11-14, "owner decision on point 3": against KTransformers the builder runs with the free VRAM as expert slots, the 10.97 GB default stays the pairing against llama.cpp; point 3 is met (Decision) (`20261114`, `20261114b`).
