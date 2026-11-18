# Session 20261101-local

- Status: local development runs on a Windows VM without a GPU; diagnosis and before/after checks for ADR 0036 and 0037, not a verdict. Local numbers are noisy.
- Code and box: commits of 2026-11-01 (see git log of this directory); local Windows VM, no box.
- Raw files: this directory.

## From ADR 0036 Context (moved verbatim)

The run of 2026-11-01 (ADR 0035, Consequences) measured the builder all on CPU at 4.1 tok/s decode and 9.8 tok/s prefill on Qwen3-30B-A3B, where llama.cpp on the same box is several times faster. `tests/cpu_bench.rs` reproduces it on a synthetic 48-layer top-8 Q4_K_M MoE (bench/results/20261101-local): at 1 thread the builder matches libllama (decode ratio 0.97); at 8 threads it decodes at 0.09x libllama, 10 times slower than its own 1-thread run, with only about 9 ms per token in graph building and allocation. The time is inside `sched.compute`.

## From ADR 0036 Consequences (moved verbatim)

- Local result (bench/results/20261101-local): the threadpool alone takes the builder at 8 threads from 0.25 to 2.5 tok/s decode; after both changes the 1-thread ratio to libllama is 1.14 decode and 1.08 prefill. The whole-step graph's gain is within this VM's noise; it removes 48 launches and synchronizations per token, which matters on a GPU and is measured on the box.

## From ADR 0037 Consequences (moved verbatim)

- Miss reads go through one bounce buffer per shard, reused across steps (at most 256 MiB, reported as `io`), and one copy into the slot. Local exit regime (`tests/exit_regime.rs`, synthetic 8-layer top-4 MoE, 4 of 16 experts in RAM, Windows VM, `bench/results/20261101-local/exit-regime-{before,after}.json`): decode streamer time 73 to 154 ms per token before, 33 to 35 after; read rate 2.4 to 2.6 GB/s after (device time only; the before figure of 0.4 to 0.9 also counted allocation, zeroing and two copies, so the streamer time is the comparable number); greedy tokens unchanged. Local prefill is CPU-compute bound (reads 1 to 5 % of it); on the box `op_offload` runs large CPU-group matmuls on the GPU, as llama.cpp does.
