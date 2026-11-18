# Session 20261117c

- Status: diagnosis; `gpu_kv_save` failed its bounds.
- Code and box: main 1d09caa, box 61042290, RTX 3090.
- Raw files: this directory.

## Findings

From ADR 0074 (addendum "the host side dominates", moved verbatim):

- Staged save 47.5 ms (8B) and 28.6 ms (30B) against 55.4 and 35.6 ms straight: `gpu_kv_save` failed its 25 and 18 ms bounds; admission `save_ms` stayed at 43 to 49 ms (8B) and 29 to 35 ms (30B).
- Copying into fresh memory alone took 48.1 and 28.1 ms, and 15.7 and 9.0 ms with `MADV_HUGEPAGE`: the save is bound by page faults on the new snapshot, not by the device copy. Advising the snapshot before its first write (`track-save-thp`, 99988e3) was tried next.
