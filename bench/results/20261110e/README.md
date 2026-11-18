# Session 20261110e

- Status: clean; Phase 2 met against KTransformers on AVX2 with CPU matched (ADR 0045); ADR 0054's overlap gain holds with no spare cores.
- Code and box: results commit 39e5e05; box 60072027, RTX 3090, AMD EPYC 7742 (AVX2), quota 30.72, 30 threads, 91 GB balloon.
- Raw files: this directory.

## From ADR 0045 (addendum "met with the CPU matched", moved verbatim)

## Addendum (2026-11-10, box 60072027): met with the CPU matched

EPYC 7742 (AVX2), RTX 3090, 91 GB, both engines at 30 threads = floor(quota), decode CPU 27.5 to 29 cores on both. Decode `main-overlap`/KTransformers 1.143 and 1.181 (cross pairs 1.158, 1.166; overlap pair 1.3% apart, KT pair 2.0%, A/A 0.996). Prefill 2.3x. `main-prefetch` alone is 0.757 of KTransformers here: KTransformers gains more from the extra threads, so the overlap path (on by default in `serve`, ADR 0054) is what meets the bar on AVX2. Verdict so far: met against KTransformers on this AVX2 box with the builder at its `serve` defaults; on the AVX-512 box (b1-exit) the prefetch path ties KTransformers within noise (1.13x on the means) and overlap is unmeasured there. The llama.cpp expert cache ran one hand-picked CPU share (0.9; the tuner gave none) in mmap. A closing AVX-512 run (`20261110f`) pairs overlap with KTransformers and adds share 0. Results `bench/results/20261110e/`.

## From ADR 0054 (addendum "starved case", moved verbatim)

Same CPU class as the previous addendum (EPYC 7742, AVX2), both engines at 30 threads = floor(quota 30.72). Decode CPU was matched: builder 27.7 to 29.0 cores, KTransformers 27.5 to 27.9. `main-overlap` decodes 1.51x and 1.53x `main-prefetch` (overlap pair 1.3% apart). Hidden share 0.42 and 0.38, still under the 0.5 bar. The decode gain holds without spare cores, so the confound of the previous addendum is closed. The `Reader` path now only serves `--io-service off`; deleting it is a cleanup, not a measurement question.
