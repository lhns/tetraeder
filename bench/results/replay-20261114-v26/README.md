# Session replay-20261114-v26

- Status: clean for the replay: the pool across layers misses ADR 0050 (c)'s halving threshold at VRAM 26 too, so B1 is not built. Local replay of the `20261114` trace, no box run.
- Code and box: f826d15; local, no box.
- Raw files: this directory.

## From docs/m2-plan.md, section "Track B: decode levers" (moved verbatim)

- **B0 result (2026-11-14, ADR 0050 addendum): B1 is not built.** At 26 VRAM slots the best pool across layers reads 0.54 (all tokens) and 0.63 (first 64) of the best per-layer policy, short of ADR 0050's halving threshold; about 2 ms per token of synchronous reads is left. The VRAM hot-set arm (`main-overlap-poll0-hot`) goes on box D1 instead.
