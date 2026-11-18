# Session 20261117e

- Status: kv-save2 measured; D2a in this session is unclean on throttle (null).
- Code and box: main afda512, box 61060914, THP and defrag `madvise`.
- Raw files: this directory.

## Findings

From ADR 0074 (addendum "the pool on the GPU", moved verbatim):

- 12 admission saves per arm. 30B: pool with advice median 10.4, p95 and max 13.2 ms (passes the bar); without advice 10.5 / 23.6; no pool 30.6 / 31.6. 8B: pool with advice median 11.2, max 48.4 ms; without advice 14.1 / 50.6; no pool 46.3 / 49.9.
- The 8B misses on one save, its first: a fallback. The dense backend did not report `kv_row_bytes` (the trait's 0), so the size hint never prepared a spare before the first save; the MoE backend reports it and had no fallback. Fixed (`DenseBackend::kv_row_bytes`, test `cpu_kv_row_bytes_matches_a_snapshot`).
- No compaction stall with the pool: `compact_stall`, `thp_fault_fallback` and `allocstall` deltas 0 in every arm, `memory.events` all 0; `thp_fault_alloc` 555 (8B) and 400 (30B) in the advised arms. The earlier 2 s stalls came with advice on the admission path, not in the preparer.
