# 0084. KV cache partly in host memory

Status: Accepted (owner, 2026-11-23). Proposed (2026-11-23, owner asked for a plan after the Strata 0.2.20 release; planned with the advisor). Plan feature 3 (KV tiers), RAM half; NVMe and remote stay parked in `memory-kv-tiers`. Tree node `memory-kv-host`. Not started: the order against the parity groups (ADR 0083) is the owner's.

## Context

- Today every layer's KV lives on one backend (`alloc_kv_layers`, `crates/model/src/attn.rs`); a context that does not fit in VRAM is refused. libllama has `offload_kqv` in `ggml-sys` but `serve` does not expose it; llama-server has `-nkvo` (arg.cpp:2394 at the pin), a missing row of the parity matrix.
- Strata 0.2.20 (`--kv-offload auto|on|off`, read 2026-11-23 in `engine/src/inkling2/cuda/ink_runner.cuh`:1882-2040): KV split by position, the first 3/4 of what fits in VRAM there, the rest in mapped pinned host memory (`cudaHostAllocMapped`) that the CUDA attention kernel reads over PCIe. Only the 8 global-attention visits of Inkling 2 Mini are split; its 40 sliding-window visits stay in VRAM. Its 50.9 to 62.6 tok/s at 262K rest on that: few full-attention layers.
- Every layer of Qwen3 (8B, 32B, 30B-A3B, 235B) is full attention, so each decode step reads the whole KV of each sequence.

KV bytes per token, F16 (Q8_0 about 0.53 of it), and the time to read the host part once per decode step, at an assumed 25 GB/s over PCIe 4.0 x16 and 60 GB/s from host RAM (neither measured; step 1 measures both). The one CPU attention number in the repo is far lower: `bench/results/20261114/kv-layout/kv-gather.json` (local CPU, 8 threads, ggml SSE tier, not a box) takes 410 ms for one 8B layer over 32,768 rows, about 0.33 GB/s, so CPU attention may be compute-bound and slower than PCIe:

| Model | Layers x KV heads x 128 | Bytes per token | 32K | 131K | 1 GB on host per step: PCIe / CPU |
|---|---|---|---|---|---|
| Qwen3-8B | 36 x 8 | 147,456 | 4.8 GB | 19.3 GB | 40 ms / 17 ms |
| Qwen3-32B | 64 x 8 | 262,144 | 8.6 GB | 34.4 GB | same |
| Qwen3-30B-A3B | 48 x 4 | 98,304 | 3.2 GB | 12.9 GB | same |
| Qwen3-235B-A22B | 94 x 4 | 192,512 | 6.3 GB | 25.2 GB | same |

A decode step of the 8B on a 3090 is under 10 ms, so one GB of KV on host costs about 2 to 4 decode steps. Host KV is a way to fit a context, not a speed lever, on these models. The cheaper levers come first: Q8_0 KV (ADR 0050 b, 47% fewer bytes) and the planner's existing trade of KV rows against expert slots (about 1.1 GB per slot tier on the 235B).

## Decision

1. **Step 0, libllama.** `serve --no-kv-offload` passes `offload_kqv = false` on the libllama engine; parity row "KV offload (`-nkvo`)" in group D with a conformance case. No own-builder change.
2. **Step 1, own builders, split by layer.** First, before any builder code: `kv_gather` (ignored test, label `cpu`) on a box CPU at the AVX2 or AVX-512 tier and box threads, and a mapped-host read probe on its GPU, give both read rates; if CPU attention reads under the PCIe rate, step 1 is skipped and step 2 is designed instead. The KV of a chosen layer set lives in host memory and those layers' attention runs on the CPU backend, the way `-nkvo` does for all layers; Q, K and V of the step go to the CPU and the attention output comes back. The builders already mix a CPU backend into the graph (the MoE CPU hit group) and allocate KV per layer range (ADR 0070 ranks), so no ggml patch is expected; if one is needed, this ADR is amended first (rule 8). A plan option `kv_host_layers` (default none) sets the set; the planner chooses it only when the context does not fit otherwise.
3. **Step 2, split by position (Strata's way), if CPU attention is the slow route.** Hot rows in VRAM, the rest in mapped host memory read by the CUDA kernel, which needs a flash-attention patch (two passes merged by log-sum-exp, or a kernel that reads host pointers). Opened only if pair A measures CPU attention reading host KV at under 25 GB/s (slower than an assumed PCIe read), so the kernel route could win; then its own ADR with thresholds.
4. **Planner (rules 4 and 5).** Host KV gets a cost term: bytes on host per step over the measured host read rate, plus per-layer transfer latency, against the gain of what the freed VRAM holds (more sequences, more expert slots). Property tests on every hardware fixture: VRAM and host memory never exceeded with host KV counted, every layer's KV in exactly one tier.
5. **Parity (rule 2).** Logits with `kv_host_layers` set equal all-in-VRAM within `TOL_GPU` on the tiny fixtures, per split point; greedy golden sequence unchanged.
6. **Metrics (rule 6).** `tetraeder.kv.blocks` already has a `tier` attribute; it gets the value `ram`. Host-side attention time goes to the step histogram with a `kv_tier` attribute; KV bytes moved between tiers go to `tetraeder.tier.transfer.bytes` with purpose `kv`. Panel beside "KV pool"; alert when host KV is in use and decode per token is over its planned cost by 25% for 10 minutes.
7. **Interactions.** Chunked relocation copies rows within one tier only (a sequence's rows of one layer never span tiers in step 1). Prefix snapshots (ADR 0049) save and restore from either tier. Ranks (ADR 0070) each own their layers' KV, so placement is per rank.

## Verdict (fixed before measuring)

- **Pair A, cost at a context that fits:** own builder with `kv_host_layers` = k layers against k = 0, same box and context (8B, 32K, c1 and c4), interleaved, two clean pairs. Pass: measured slowdown per GB on host within 25% of the cost term's prediction, so the planner can be trusted with it.
- **Pair B, beyond VRAM:** own builder with the planner's choice against llama-server `-nkvo` (all KV on host, its best setting that fits) at a context that does not fit in VRAM, two clean pairs, ADR 0065 band rules. Pass: ratio at least 1 - band.
- Speed against all-in-VRAM beyond VRAM is not judged: that configuration cannot run.

## Consequences

- Long contexts that are refused today run, at a cost the planner states in advance.
- No box until step 1 is built and its CPU tests pass; pair A and B fit one 3090-class box session.
- The Strata pairing (`bench-strata-pairing`) is unchanged: still no model both engines run.

## History

- 2026-11-23: step 1 read rates measured on an A40 box with a 61-thread EPYC 7662 (`bench/results/strata-q36-20261123/README.md`, "Probes"): PCIe 4.0 x16 mapped read by a kernel 24.6 GB/s and `cudaMemcpy` 24.8 GB/s (the 25 GB/s assumption holds); CPU attention (`kv_gather` contiguous, AVX2 tier) 65 to 67 GB/s of K plus V at 32K per layer, cache-warm, so the step-1 skip rule (CPU attention under the PCIe rate) does not fire on a many-core server. The 0.33 GB/s figure was an 8-thread SSE-tier local CPU; the rate on a smaller box is open (no thread sweep yet).
