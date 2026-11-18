# Session 20261117c

- Status: diagnosis; `gpu_kv_save` failed its bounds.
- Code and box: main 1d09caa, box 61042290, RTX 3090.
- Raw files: this directory.

## Findings

From ADR 0074 (addendum "the host side dominates", moved verbatim):

- Staged save 47.5 ms (8B) and 28.6 ms (30B) against 55.4 and 35.6 ms straight: `gpu_kv_save` failed its 25 and 18 ms bounds; admission `save_ms` stayed at 43 to 49 ms (8B) and 29 to 35 ms (30B).
- Copying into fresh memory alone took 48.1 and 28.1 ms, and 15.7 and 9.0 ms with `MADV_HUGEPAGE`: the save is bound by page faults on the new snapshot, not by the device copy. Advising the snapshot before its first write (`track-save-thp`, 99988e3) was tried next.

## From docs/m2-plan.md, section "D2a throughput passes, provisional" (moved verbatim)

- Confirmation first (`20261117c`, box 61042290): the dense crash workload is clean 4/4 (it crashed 4/4 before the mask padding); the gpu_multi_seq shift is one router near-tie (ADR 0071). The pinned-staging save alone did not get faster: the save is bound by page faults on the fresh snapshot (ADR 0074 addendum).

## From ADR 0071 (addendum "the b12426 gpu_multi_seq shift is one router near-tie", moved verbatim)

- The routing printout of `gpu_batched_matches_the_cpu_builder` shows one differing (token, layer) choice over all three sequences: sequence 1, step 0, layer 0, position 12, where CUDA picks experts [3, 4] and the CPU [3, 0]. The first expert's weight agrees to 3e-4 (0.8701 against 0.8703); the second slot is a tie between experts 4 and 0 that kernel rounding decides. Sequences 0 and 2 route identically. This explains why only sequence 1 moved on the pin move and confirms the libllama anchor over a fixed bound. Crash repro on the same box: 4 of 4 dense runs clean (0 CUDA errors, steps up to 1216 prompt tokens), against 4 of 4 crashing before the mask padding.
