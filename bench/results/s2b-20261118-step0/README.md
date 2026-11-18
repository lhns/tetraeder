# Session s2b-20261118-step0

- Status: diagnosis, no verdict: the gpu-parity gate failed on two new model tests at the absolute bound (ADR 0071); step 0 not run.
- Code and box: box 61138736, host 34742, 2x RTX 3090, driver 615.71.09.
- Raw files: this directory.

## From ADR 0071 (addendum "parity and slots use the relative rule", `track-s2b-fixes`, moved verbatim)

- On 2x RTX 3090 (`bench/results/s2b-20261118-step0/phase5/`) `gpu_matches_libllama` and `gpu_slot_counts_match_libllama` failed at the absolute 5e-3 (top-5 deltas 3.5e-3 to 1.16e-2, 3e-4 to 9e-4 of the largest logit, top-1 equal, same with fusion off and on one GPU). As the Consequences anticipated, both now use `GPU_TOL`; the CPU paths keep `TOL_CPU`.
