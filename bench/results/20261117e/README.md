# Session 20261117e

- Status: kv-save2 measured; D2a in this session is unclean on throttle (null).
- Code and box: main afda512, box 61060914, host 87485, RTX 3090, driver 620.70.06, THP and defrag `madvise`.
- Raw files: this directory.

## Findings

From ADR 0074 (addendum "the pool on the GPU", moved verbatim):

- 12 admission saves per arm. 30B: pool with advice median 10.4, p95 and max 13.2 ms (passes the bar); without advice 10.5 / 23.6; no pool 30.6 / 31.6. 8B: pool with advice median 11.2, max 48.4 ms; without advice 14.1 / 50.6; no pool 46.3 / 49.9.
- The 8B misses on one save, its first: a fallback. The dense backend did not report `kv_row_bytes` (the trait's 0), so the size hint never prepared a spare before the first save; the MoE backend reports it and had no fallback. Fixed (`DenseBackend::kv_row_bytes`, test `cpu_kv_row_bytes_matches_a_snapshot`).
- No compaction stall with the pool: `compact_stall`, `thp_fault_fallback` and `allocstall` deltas 0 in every arm, `memory.events` all 0; `thp_fault_alloc` 555 (8B) and 400 (30B) in the advised arms. The earlier 2 s stalls came with advice on the admission path, not in the preparer.

## From ADR 0065 (addendum "D2a on afda512 unclean on throttle", moved verbatim)

- Probes 8B c4 1.044, 30B c4 0.957; pairs 8B c4 1.054 and 1.046, c1 1.006 and 1.009; 30B c4 0.968 and 0.963, c1 0.998 and 0.972. Every run had 0x400 or 0x600 before it (idle at SSH was 0x1), so every pair is unclean and both per-model passes are null, under the rule fixed before the session. Not rerun on the same host.
- Drivers: the bits appeared before runs on driver 615 (2026-11-16) and 620, not on 595 (host 264182: 0x0 before every run in `20261117d`). The next verdict box reads the throttle value after the prepass; outside 0x0-0x3 before a run, it is destroyed and another host taken.

## From ADR 0068 (addendum "throttle judged as in Phase 4", moved verbatim)

- Why: on drivers 615 and later the single samples read 0x4 at idle and 0x400 or 0x600 at partial load, which failed almost every Phase 4 host (`20261117e`, `g`, `i`, `j`) while the measured windows were clean.
