# Session 20261109-local

- Status: clean local replay fidelity check on a synthetic model (ADR 0050 (c) fidelity gate); says nothing about the 235B.
- Code and box: cea1858; local run, no box.
- Raw files: this directory.

## From ADR 0050 Implementation notes ("(c) first half: recorder and replay", fidelity bullet moved verbatim)

- Fidelity (`crates/model/tests/replay_fidelity.rs`, label `cpu`): per layer, the replay's RAM hits and misses equal `ExpertStats` exactly, as do the prefetch issued, used and wasted counts and the decode hit rate, for depth 1, depth 2 and prefetch off. On the synthetic exit regime (8 layers, 16 experts, 2 VRAM and 4 RAM slots, 128 prompt tokens, 48 decode tokens) the replay gives the harness's decode `ram_hit_rate_mean` 0.3452 against 0.3452 measured and the decode disk bytes per token 88515072 against 88515072. Result: `bench/results/20261109-local/replay-synthetic.json`.
