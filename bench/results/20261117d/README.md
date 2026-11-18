# Session 20261117d

- Status: diagnosis; D2a in this session ran on branch `track-save-thp` (99988e3, never merged): provisional, not the verdict of record.
- Code and box: 99988e3 (branch `track-save-thp`, never merged), box 61046208, RTX 3090, driver 595.84, THP and defrag `madvise`.
- Raw files: this directory.

## Findings

From ADR 0074 (addendum "the test passes, the server path stalls", moved verbatim):

- `gpu_kv_save` passes with the advice: staged 15.6 ms (8B) and 10.1 ms (30B). The 30B server probe saves take 10 to 13 ms.
- The 8B server probe saves, in admission order: 17, 17, 104, 432, 2041, 1979, 771, 234, then 157, 131, 127, 140 ms. Without advice (`20261117c`) they were 43 to 49 ms. The likely cause is direct compaction (defrag `madvise` stalls the faulting thread on advised memory), not yet proven.

## From docs/m2-plan.md, section "D2a throughput passes, provisional" (moved verbatim)

- D2a at `99988e3` (`track-save-thp`, huge pages on the snapshot; not on main), `PROBE_MIN=0.925`: probes 8B c1 1.045, c4 1.042; 30B c1 1.023, c4 0.976. Pairs, all clean: 8B c4 1.070 and 1.052, c1 1.011 and 1.011; 30B c4 0.998 and 0.999, c1 1.006 and 1.011. Both per-model passes true; `phase4_pass` null (no TTFT in D2a). Same workload and N as `20261116d2a2`; the libllama control moved with the own builder (30B 259 to 294 tok/s at c4), so most of the gain is the pin and the step budget.
- Not the verdict of record: the 8B server probe on the same box had admission saves of 104 ms to 2.0 s (10 of 12) with huge-page advice, against 43 to 49 ms without, while the pairs show no stall (8B c4 TTFT max 0.44 s). The verdict of record is a full `phase4_box.sh` session (pairs and TTFT) on the commit that merges, once the save stays off the admission path (`track-save-pool`).
