# Session 20261116pc

- Status: confirmation session; not a verdict.
- Code and box: 759ca1d (pin b12426, from pin-confirm summary.md), box 60983677, RTX 3090, driver 580.159.03.
- Raw files: this directory.

## Findings

From ADR 0074 Context (the measurement that motivated it):

- An admission that reuses an idle sequence first saves it to the host RAM prefix cache (ADR 0049, 0065), on the worker thread, with every live sequence stalled. On one RTX 3090 (`bench/results/20261116pc/pin-confirm/`, `save_ms` in the admission log) that takes 41 to 44 ms for Qwen3-8B and 30 to 33 ms for Qwen3-30B-A3B per sequence of about 533 tokens (78 and 52 MB of F16 KV, so about 1.8 GB/s); three saves in one pass serialize to 91 to 121 ms. It is the largest remaining loss at concurrency 4.

## From docs/m2-plan.md, section "CUDA confirmation on b12426" (moved verbatim)

- Holds: real-model parity (8B and 30B alone controls exactly 0 incl. (0,19), anchored gaps inside the bounds), patch 0004 GPU tests incl. id -1 above 16 tokens, gpu_dense, executor gpu tests. Step shapes: dense 72.8 ms against libllama 71.3 per 8 four-row steps; `decode1x8_resident` 63.3 against 59.0 alone for dense (libllama 59.5 both).
- Probe against llama-server b12426: 8B c1 0.999, c4 0.986 in the clean round; 30B c1 0.990, c4 0.904 and 0.927 (pass line 0.925).
- Broken: (1) the dense engine aborts with a CUDA illegal memory access at a fixed point of the c4 workload (4/4 with seed 94), right after 3 to 4 admissions in one pass, i.e. the first step carrying several prompts under the 2048 budget (`track-dense-crash`); (2) `gpu_batched_matches_the_cpu_builder` (MoE fixture) fails at 3x its bound at the new pin (`track-gpu-multiseq`).
- The remaining c4 loss is the admission save: 41 to 44 ms (8B) and 30 to 33 ms (30B) per reused idle sequence, serialized when several are admitted together (`track-fast-save`). Dense pays about 7% on decode with resident idle sequences, libllama nothing; a follow-up for the dense attention span.
