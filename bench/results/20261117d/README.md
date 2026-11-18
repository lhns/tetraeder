# Session 20261117d

- Status: diagnosis; D2a in this session ran on branch `track-save-thp` (99988e3, never merged): provisional, not the verdict of record.
- Code and box: 99988e3, box 61046208, THP and defrag `madvise`.
- Raw files: this directory.

## Findings

From ADR 0074 (addendum "the test passes, the server path stalls", moved verbatim):

- `gpu_kv_save` passes with the advice: staged 15.6 ms (8B) and 10.1 ms (30B). The 30B server probe saves take 10 to 13 ms.
- The 8B server probe saves, in admission order: 17, 17, 104, 432, 2041, 1979, 771, 234, then 157, 131, 127, 140 ms. Without advice (`20261117c`) they were 43 to 49 ms. The likely cause is direct compaction (defrag `madvise` stalls the faulting thread on advised memory), not yet proven.
