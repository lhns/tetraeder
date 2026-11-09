# bench

- `spec/*.toml`: pinned benchmark specs. `phase0.toml` is the nightly gate, `phase2-baseline.toml` the tuned llama.cpp and ik_llama.cpp baselines for Phase 2 (engine `kind = "ik-llama"`, binary `/opt/ik_llama/bin/llama-bench` or `$IK_LLAMA_BENCH`; `--llama-bench` only overrides the llama.cpp one), `smoke.toml` a CPU-only check for dev machines.
- `golden/`: greedy token sequences and top-k logits per position (format: `crates/bench/src/golden.rs`). Full logits are a nightly artifact, not in git.
- `noise/<class>.json`: measured noise bands per device and GPU class (`cpu`, `rtx-3090`, ...), written by `tetraeder-bench noise update`.
- `results/<YYYYMMDD>/<spec>-<HHMMSS>.json`: output (UTC, one file per run) of `tetraeder-bench run <spec>`, plus `env.json`, `parity.json` and the `DONE` marker from `scripts/nightly.sh`.

Run: `tetraeder-bench run bench/spec/phase0.toml`. Exit 0 ok, 1 regression beyond the noise band, 2 error (or a skipped pair under `--strict`), 3 inconclusive (a noise pair outside its band). Precedence: 2, then 1, then 3.
Comparisons are paired and interleaved in one session, with alternating order (A,B then B,A ...); never compare absolute tok/s across nights.
Bands: a run uses the measured noise band of its device and GPU class (`noise/<class>.json`, made by `tetraeder-bench noise update` from the A/A pairs of committed results, at least 3 sessions) and otherwise the spec's provisional `[[noise_band]]`; `pairs[].band_source` says which. See `docs/sessions.md`, "Noise bands".

## Result fields

Each sample (`pairs[].samples[]`) records:

- `order`: 0 if side a ran before b in this repetition, 1 if b ran first. Repetitions alternate so that a systematic advantage of running second (clocks, caches) cancels in the paired ratio. Warmup stays a,b.
- `settings`: what the engine reported it ran with, in llama-bench's field names: `flash_attn` (-1 auto, 0 off, 1 on), `n_batch`, `n_ubatch`, `n_threads`, `n_gpu_layers`, `tensor_buft_overrides` (`none` or `pattern=buffer;...`), `cuda_graphs`. llama-bench does not report `cuda_graphs`: the harness derives it (CUDA backend, layers offloaded, `GGML_CUDA_DISABLE_GRAPHS` unset); `tetraeder bench` reports its own expectation. Compare the settings of both sides of a pair before reading a ratio. Null means the engine did not report it.
- `n_eval`, `n_reused`: Tetraeder only. libllama's decode calls and graph reuses over the timed repetitions of the test (delta, warmup excluded). A high `n_reused` / `n_eval` is the evidence that the graph was reused; `cuda_graphs: true` with `n_reused` 0 means it was not. Null for llama-bench, which prints no counters.

Variant `args` may set `-b` and `-ub` (Tetraeder: `-b`, `--ubatch`) next to `-p`, `-n`, `-r`; they replace the workload defaults.

## Quality (parity)

A gate whose candidate is a Tetraeder build must also produce the reference output. `scripts/nightly.sh` runs `tetraeder-bench parity <spec>` after the tests. For every (model, device) of such a gate it runs the greedy golden check of ADR 0026 (the text of `llama-simple`, built from the pinned tree, is the reference, as in `cargo xtask golden`) for 2 short prompts of 16 tokens, on the backend of the gate (`-ngl 99` on CUDA, 0 on CPU). It writes `parity.json` next to the results: `{check, status, detail}` per model and device, status `ok`, `failed` (outputs differ) or `error` (could not run).

`run` reads `parity.json` from its results directory (`--parity` overrides) and attaches the entry to each such gate pair as `quality: {check, status, detail}`. A failed, errored or missing entry (also a missing file, or no `llama-simple` on the box) turns the pair into `error` (exit 2), whatever the speed ratio says. Gates that compare llama.cpp with llama.cpp have no `quality`. The box image must contain `llama-simple` (the Dockerfile in `docker/box` installs it; `LLAMA_SIMPLE` overrides the path).

## Phase 2 baseline

`phase2-baseline.toml`: llama.cpp on Qwen3-30B-A3B Q4_K_M with flash attention, ubatch 512 to 4096, full GPU and the two expert-offload placements of phase0, pp16384, pp32768 and tg128. Info pairs only, about 42 min estimated on a 3090 (not measured yet). Select it with `BENCH_SPEC=phase2-baseline scripts/nightly.sh`; it replaces phase0 for that night, and taxameter's nightly wait limit must allow for it.

Exit-run baseline from Colibri (pinned v2.1.0, `qwen36` engine reads the Qwen3 MoE GGUF in place): `scripts/colibri_build.sh`, then `scripts/colibri_exit.sh <first-shard> <threads> <ram-bytes> <out.json> [deep|short]`. It is not wired into `scripts/exit_run.sh`; its JSON has its own shape (prefill and decode tok/s from the engine's TTFT and total time).

llama.cpp expert cache with a CPU share of the misses (#30966; baseline build in `/opt/llama.cpp-baseline`, `baseline_commit` in `vendor/PIN`): `scripts/llamacpp_cache.sh` runs one llama-server and measures decode tok/s after a 32K prefill. `scripts/exit_run.sh` calls it in stages `llamacpp-cache` and `llamacpp-cache-repeat` (shares from `CPU_SHARES`). `scripts/moe_cache_tune.sh` runs llama.cpp's tuner on the 30B (`TUNE=1` in `scripts/c_session.sh`).

## Claude Code against both servers

`scripts/claude_code_proxy.sh` (`uv run taxameter run scripts/claude_code_proxy.sh`) runs real headless Claude Code sessions on 10 coding tasks against `tetraeder serve` and `llama-server`, ABBA over the reps, and scores them. Kit: `bench/claude-code-proxy/` (`fixture/` is the Python project the sessions work on, `tasks.json` the prompts, setup, hidden tests and checks, `solutions/` the reference fixes that `tests/test_claude_code_proxy.py` replays without Claude). Every request goes through `scripts/anthropic_proxy.py`, which logs tokens, TTFT and decode tok/s identically for both engines. Output: `bench/results/<date>/claude-code-proxy/` (`session-*.json`, `proxy-*.jsonl`, `summary.json`). The `*.reqlog` files and `diff-*.patch` stay on the box. Sessions are not deterministic: compare success rates over reps, not single runs.
