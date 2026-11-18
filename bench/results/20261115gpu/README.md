# Session 20261115gpu

- Status: diagnosis: every own-builder GPU parity test failed against the absolute `TOL_GPU`; led to ADR 0071.
- Code and box: results commit fde4066; box 60800908, RTX 3090.
- Raw files: this directory.

## From ADR 0071 Context (moved verbatim)

- The first CUDA run of the own builders (`bench/results/20261115gpu/gpu-smoke2/`, RTX 3090, cc 8.6) failed every own-builder parity test on the tiny fixtures against `TOL_GPU = 5e-3`, an absolute bound written for logits of 1 to 5. The fixtures' logits are 11 to 15. The diffs were 4e-4 (dense, one sequence, top 5) and 1.4e-3 to 2.4e-3 (batched, `seq_cp`) of the largest logit. The Qwen3-8B dense parity test against libllama passed.
