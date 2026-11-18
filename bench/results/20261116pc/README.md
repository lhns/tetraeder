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

## From ADR 0065 (addendum "one probe decides the pairs at 0.925", moved verbatim)

- The next D2a session runs with `PROBE_MIN=0.925`, the pass line, instead of 0.85: a model whose single probe c4 ratio is below it gets no pairs, and is not re-probed in that session. One probe round per model; the 30B sat at 0.904 and 0.927 around the line, and re-probing until one passes would select on noise.
- The session runs only after a confirmation on the same box: the dense crash workload (4 of 4 runs crashed before the mask padding fix, `bench/results/20261116pc/pin-confirm/repro/`) with zero CUDA errors, the `gpu_kv_save` and `gpu_multi_seq` tests passing, and the prefix save time measured (ADR 0074).

## From ADR 0071 (addendum "CUDA against the CPU builder at b12426 is libllama's own gap", `track-gpu-multiseq`, moved verbatim)

- `gpu_batched_matches_the_cpu_builder` failed at the new pin (`bench/results/20261116pc/pin-confirm/gpu-smoke5/`, RTX 3090; 2 of 2 in `repro/`): sequence 1 (the 21-token prompt) is off by 1.6e-2 to 6.4e-2 of the largest logit on every row, with a greedy flip at step 4; sequences 0 and 2 stay near 2e-3. The same run's anchor shows the builder on CUDA bit-equal to libllama on CUDA (batched and split, 22 rows) and libllama's own CUDA-against-CPU gap at the same values, digit for digit: (1,20) 1.7513514e-1, (0,6) 2.4983883e-2. Across pins the CPU side moved and CUDA did not: sequence 1's first-step largest logit is 10.795 to 10.882 on the CPU and 10.797 to 10.789 on CUDA. So the gap is libllama's own, not our builder, and the CPU move is consistent with the CPU numerics changes in the range (#31854, #29810); libllama on CUDA is built on the same patched ggml, so bit-equality alone does not clear patch 0004 or #32067, the cross-pin CUDA stability does. CUDA still has no flash-attention kernel for head 16 at b12426 (`fattn.cu` has no case for 16), so the fixture's attention runs on the CPU on both sides.
- The test now follows the anchored rule on its own tokens and steps: libllama on CUDA and on the CPU run the same script; the builder is asserted against libllama on the same backend and shapes (strict: `GPU_TOL` on CUDA, `TOL_CPU` on the CPU), and its CUDA-against-CPU gap (batched, and against each sequence alone on the CPU) within max(`GPU_TOL`, 1.5x libllama's own worst). Every table prints first; the parts run to completion. Same tokens matter: the existing anchor uses other decode tokens and tops out at 2.98e-2, below the 6.4e-2 row, so the pass is predicted from the bit-equality, not yet observed. The anchor is libllama on CUDA against libllama on the CPU, not libllama batched against split on CUDA (2.1e-3 here, kept in `gpu_libllama_batched_vs_split_anchor`).
- Open until the next box: why only sequence 1 moved, at about 10x the old gap. The hypothesis is a near-tied router that flips on one backend for a prompt token of sequence 1 and reaches every later row of it through attention. The hybrid run supports it: with expert misses on the CPU next to CUDA segments (router on CUDA), sequence 1's first-step largest logit stays at 10.789, so CPU expert matmuls alone do not move it. The test prints the routing difference of the two builders (`record_routing`, separate models, per step) and the CPU ISA. Identical routing with this gap would be a kernel divergence to escalate.
