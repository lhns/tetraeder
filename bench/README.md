# bench

- `spec/*.toml`: pinned benchmark specs. `phase0.toml` is the nightly gate, `smoke.toml` a CPU-only check for dev machines.
- `golden/`: greedy token sequences and top-k logits per position (format: `crates/bench/src/golden.rs`). Full logits are a nightly artifact, not in git.
- `results/<YYYYMMDD>/<spec>-<HHMMSS>.json`: output (UTC, one file per run) of `tetraeder-bench run <spec>`, plus `env.json` and the `DONE` marker from `scripts/nightly.sh`.

Run: `tetraeder-bench run bench/spec/phase0.toml`. Exit 0 ok, 1 regression beyond the noise band, 2 error (or a skipped pair under `--strict`), 3 inconclusive (a noise pair outside its band). Precedence: 2, then 1, then 3.
Comparisons are paired and interleaved (A, B, A, B ...) in one session; never compare absolute tok/s across nights.
