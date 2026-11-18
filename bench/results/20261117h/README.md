# Session 20261117h

- Status: kv-save2 passes the ADR 0074 bar; D2a probes stopped both models (null).
- Code and box: c56b184, box 61069127, host 264182, THP and defrag `madvise`.
- Raw files: this directory.

## Findings

From ADR 0074 (addendum "huge-page advice off by default", moved verbatim):

- The pool passes the bar on both models: 12 saves each, all from a spare. 8B median 11.6 / max 12.2 ms with advice, 11.3 / 12.2 without; 30B 8.6 / 9.2 and 8.3 / 8.8. No pool: 45.6 / 50.5 and 29.9 / 31.8.
- On this host memory was fragmented: a fresh copy with `MADV_HUGEPAGE` took 583 ms (8B) and 352 ms (30B) against 50 and 33 without, and the advised arms show 1619 and 1099 `compact_stall`, 531 and 365 `thp_fault_fallback`, 542 and 366 `allocstall_movable`; the arms without advice show none. The preparer then could not keep a spare ready (0 ready at the end with advice, 1 without).
- In the D2a probes on the same box (default: advice on), decode matched llama-server (30B c1 197.7 against 198.5 tok/s, 8B 138.4 against 137.9) but time to first token did not (30B median 0.532 s against 0.130, 8B 0.317 against 0.153; 0.118 for the 30B in `20261117d`), so both probes stopped (c4 0.819 and 0.922). The likely mechanism, not proven: page faults stalled in direct compaction hold the process's memory-map lock, so every other mapping change in the server (large allocations) waits for them.
