# Box sessions for the open M1 exit tests

Four launch-ready sessions, each a profile of `scripts/session.sh`. Written 2026-11-09 while box runs are on hold; nothing here has run yet. Rules: AGENTS.md "GPU work on Vast.ai" and the caps there (3 h and 0.45 USD/h per box, two boxes, 1.0 USD/h per account; B2 has the user's 48 GB exception, 6 h at 0.50 USD/h).

## How a profile keeps its cap

- **Budget.** 20% margin on the cap leaves 144 min of box time (288 for B2). Provisioning before the script (image pull, push, deps) and fetch, verify and destroy after take about 25 min, so the script gets `SESSION_MIN` = 120 min (260 for B2) from its first launch.
- **Deadline.** `session.sh` exports `DEADLINE` (first launch plus `SESSION_MIN`). `exit_run.sh` skips a stage whose estimate (`stage_min` in the script, the numbers below) would end after it; the planner fixture, the detector loop and the B2 hook stop starting work after it. Stages run in order of value, so a cut drops the diagnosis first.
- **Incremental and resumable.** Results go to `bench/results/<RESULTS_DATE>/`, fixed at the first launch (`runs/vast/session-<name>.start`). A relaunch of the same profile skips what has results: `exit_run.sh` with `RESUME=1` skips stages with an `exit-<stage>-<stamp>.json` holding numbers; `tetraeder-bench run --resume` runs only missing repeats and rewrites its JSON after every pair; the detector resumes after its last observation; the Claude Code runner skips sessions with output; the cache tuner reuses its finished runs.
- **Launch and wait** (all profiles; `T="uv run taxameter"`, `--id` always once a box exists):

```bash
git merge origin/main && uv sync && git push            # provision and run push committed HEAD only
$T status                                                # at most two boxes up, ask "main" for a slot
$T offers --min-cpu-cores <cores> --download-gb <gb> --min-disk <gb>   # pick by RAM, cores, inet_down_cost
$T create --offer <offer id> --min-cpu-cores <cores> --download-gb <gb>  # prints the box id
$T provision --id <id> [--models ...]                    # then check the provision log says cuda: ok
date +%T; $T run --id <id> scripts/session.sh --env SESSION=<name> [--env ...]
$T ssh --id <id> --timeout 540 -- 'for i in $(seq 100); do grep -q SESSION_DONE /root/tetraeder/runs/vast/session-<name>.log && break; sleep 5; done; tail -5 /root/tetraeder/runs/vast/session-<name>.log'
```

- **Fetch, verify, destroy:** `$T fetch --id <id>` (checks `bench/results/*/DONE`), then the profile's verify list below, then `$T destroy --id <id> --yes`. Report the box id, hours, `inet_down_billed` and `inet_down_cost`.
- Resume works on the same box only: a new box gets a new start file and date and starts over, as pairing needs (same-box runs only).
- If the session log stops advancing after `run`, relaunch with the same command (the flock makes a duplicate a no-op, `RESUME=1` skips finished work), or start it by hand: `ssh -- 'cd /root/tetraeder && (SESSION=<name> nohup setsid bash scripts/session.sh > /root/session.out 2>&1 < /dev/null &); sleep 30; tail runs/vast/session-<name>.log'`.

## Order and cost

B3 first: its cache tuner gives B1 the llama.cpp CPU share. B3-planner and B3-detector run in parallel on two boxes (0.90 USD/h together, under the account cap); B2 any time; B1 last, alone or beside B2 (0.95 USD/h).

| Session | Box | Script time (estimate / budget) | Box hours | Box cost | Download | Traffic cost | Total |
|---|---|---|---|---|---|---|---|
| B1 `b1-exit` | RTX 3090, about 91 GB RAM, 24 to 26 core quota, 220 GB disk | 127 / 120 min (the guard drops `ktransformers-avx2`) | 2.4 | at most 1.08 | 142 GB | 0.92 to 3.69 | 2.0 to 4.8 USD |
| B2 `b2-proxy` | 48 GB Ampere card (RTX A6000, A40), 64 GB RAM, 8+ cores, 80 GB disk | 82 min proxy + hook / 260 min | 1.8 to 4.8 | 0.90 to 2.40 | 19 GB | 0.25 to 0.50 | 1.2 to 2.9 USD |
| B3 `b3-planner` | RTX 3090, 32+ GB RAM, 9+ cores, 100 GB disk | 96 to 120 / 120 min | 2.0 to 2.4 | at most 1.08 | 38 GB | 0.50 to 1.00 | 1.3 to 2.1 USD |
| B3 `b3-detector` | RTX 3090, 32+ GB RAM, 9+ cores, 80 GB disk | 65 to 110 / 120 min | 1.5 to 2.3 | at most 1.04 | 19 GB | 0.25 to 0.50 | 0.9 to 1.5 USD |

Traffic is 0.013 to 0.026 USD/GB (Vast `inet_down_cost`, tracks seen so far); box 58900330 paid 0.92 USD for the same 142 GB (ADR 0045), so the low end exists. Rank offers with `--download-gb`.

## Sources of the estimates

- **Builder 32K run** (prefill 32768, 64 decode tokens): 8 to 9 min on the 91 GB class. `exit-main-prefetch-120800.json` load 6.6 s, prefill 453 s, decode 3.76 tok/s (17 s); `exit-d2-120800.json` prefill 469 s; `exit-main-021753.json` 454 s. On the 45 GB slow-disk class it is 21 min (`exit-main-prefetch-070656.json`: prefill 1239 s, 0.69 tok/s), so B1 needs the 91 GB class.
- **KTransformers 32K**: request 667 s (`exit-ktransformers-120800.json`, 26 cores) to 937 s (`-032306`, 15 threads). Load and the bf16 attention stub are in no committed artifact; `ktransformers_exit.sh` caps them at 40 min (stub 2400 s) and the stub is reused by later KT stages. Estimate 25 min for the first KT stage, 17 to 20 for the next. Least evidenced stage.
- **llama.cpp expert cache** (`llamacpp_cache.sh`, one share): never run. Proxy: the libllama `-ot` reference on the same model and mmap, `exit-reference-021753.json` load 52 s, prefill 707 s, decode 2.55 tok/s; plus the 256-token warmup and server start: 15 min.
- **Smoke**: prefill 8 tokens at 32K context, 37 to 61 s plus load (`exit-smoke-*.json`): 2 min.
- **Builds**: 8 min 51 s (`partial-planner-dense/planner_campaign-dense.log`), 11 min 11 s (`claude-code-proxy/claude_code_proxy.log`). B1 builds twice (portable and native ISA, own target dir), about 20 min, during the download.
- **Downloads**: no committed rate. `min_inet_mbps = 500` guarantees 62 MB/s: 142 GB in 9 to 38 min, 19 GB in 2 to 5 min.
- **Claude Code proxy**: 82 min for REPS=2 including the 11 min build (`claude_code_proxy.log`, 4 blocks of 17 to 19 min, streaming conformance 69 s).
- **Planner dense fixture**: measured wall per llama-bench run = predicted request time + prompt time + 5 s (llama-bench's warmup repeats the prompt; fit to the 14 runs of `partial-planner-dense/dense_rerun.log`, within 1 to 6 s each). Per repeat: 14 candidates with op offload 505 s, 10 without it 2450 s (pp 16 to 60 tok/s), GPU anchors 300 s (`anchor-gpu-small-opo-off` alone 264 s), CPU anchors at pp64 tg16 about 100 s, baselines 60 s: 57 min per repeat, 33 min for the anchors, baselines and the 20 cheapest candidates.
- **Detector campaign**: 3 + 40 + 9 = 52 runs of a 256-token prefill and 96 decode tokens on the 30B with misses from disk, plus the sampler. No box evidence; the 30B builder decodes 7 to 40 tok/s with slots and disk misses (`20261102/builder-real-202732.json`), so a run is 40 to 90 s with load and teardown: 35 to 78 min, plus a 10 min `exit_regime` build.
- **kernels-cpu**: 9 llama-bench runs of 64 tokens on the 30B (CPU only, about 15 tok/s at 9 threads from the planner CPU anchors) and up to 3 x 180 s of `objdump`: 10 to 15 min.
- **moe-cache tuner**: its script ships in the box image, not the repo; no run count known. Capped at `TUNE_MIN` = 25 min; a relaunch resumes from its finished runs.

## B1: Phase 2 exit verdict (`b1-exit`)

**Box.** RTX 3090 (the llama.cpp baseline build is Ampere only, `BASELINE_CUDA_ARCHS="86"`), container RAM about 91 GB (the 235B must not fit; `exit_run.sh` balloons any larger limit down to 128 GB), CPU quota 24 to 26 (the gap to KTransformers showed at 26 and was a tie at 15, ADR 0045 addendum), NVMe at 5 GB/s or more, 220 GB disk (142 GB model, the KT stub, two builds). `$T offers --min-cpu-cores 24 --download-gb 145 --min-disk 220`, pick an offer with 88 to 100 GB RAM.

**Launch.** No models at provision (the script fetches the 235B). `$T run --id <id> scripts/session.sh --env SESSION=b1-exit --env MOE_CPU_SHARE=<tune.json cpu_share from B3>`. Defaults: `VRAM_SLOT_BYTES=10966007808`, `OT_FIT='blk\.[0-6]\.ffn_.*_exps=CUDA0;exps=CPU'` (fitted for a 24 GB card at 32K), `LOAD_MODE=auto`. Without `MOE_CPU_SHARE` it takes the newest committed `moe-cache-tune/tune.json`, else 0.5 (logged).

**Stages** (code order of `exit_run.sh`; every builder stage has prefetch on at depth 2, the ADR 0038 levers; host sampling on every builder and KT stage):

| # | Stage | Estimate | Cumulative | Decides |
|---|---|---|---|---|
| 0 | download 142 GB, KT install, portable and native builds (parallel) | 20 (to 38) | 20 | |
| 1 | balloon, `smoke` | 3 | 23 | gate: KV fits, direct I/O, split |
| 2 | `main-prefetch` (builder, levers) | 9 | 32 | builder decode for the verdict |
| 3 | `llamacpp-cache` at the tuned share S | 15 | 47 | llama.cpp's best tuned setup (ADR 0045 addendum 2026-11-05) |
| 4 | `ktransformers` | 25 | 72 | KT decode for the verdict |
| 5 | `main-repeat` (builder, levers) | 9 | 81 | noise band of builder over KT |
| 6 | `ktransformers-repeat` | 17 | 98 | noise band |
| 7 | `main-native-repeat` (native ISA, levers) | 9 | 107 | ADR 0045 hypothesis 3 (paired with 5) |
| 8 | `ktransformers-avx2` | 20 | 127 | KT kernel ISA; skipped by the deadline unless the download was fast |

**Verify after fetch.** Every stage JSON has numbers (`decode_tok_s`, KT `ok: true`) and a `host` block. In `exit-llamacpp-cache-s<S>-*.json` read `warmup.majflt`, `run_stat.majflt` and `log_load_mode`: if major faults run into the millions and decode is far below the builder's, llama-server thrashed mmap; rerun that stage alone on the same box before calling the verdict: `ssh -- 'cd /root/tetraeder && (RESULTS_DATE=<date> RESUME=0 LOAD_MODE=dio STAGES=llamacpp-cache CPU_SHARES=<S> VRAM_SLOT_BYTES=10966007808 nohup setsid bash scripts/exit_run.sh > /root/dio.out 2>&1 < /dev/null &)'` (`RESUME=0`: the thrashing run has `ok: true` and would be skipped).

**Result.** Phase 2 decode clause: builder `main-prefetch` and `main-repeat` decode at least KTransformers' and the llama.cpp cache's, with the two builder/KT ratios as the noise band (the bar is paired ratios only, never absolute tok/s across boxes). Read the gap stages with the table in ADR 0045, "Diagnosing the KTransformers decode gap". Commit the JSONs and update ADR 0045 and `docs/m1-open.md` item 1.

## B2: Claude Code proxy rerun (`b2-proxy`)

**Box.** 48 GB card (CTX 131072 needs it), Ampere: RTX A6000 (the first run) or A40 (the box image's CUDA builds target sm_86; an Ada L40S is unverified), 64 GB RAM, 8+ cores, 80 GB disk, at most 0.50 USD/h (the user's exception, up to 6 h). `$T offers --gpu "RTX A6000,A40" --min-vram 48 --max-price 0.50 --download-gb 20`, then `create` with the same flags. 0.50 is above `max_price_per_hour = 0.45` in `taxameter.toml`; if taxameter refuses the flag over the config cap, use a local `.vast/config.json` override for this box only (untested).

**Launch.** `$T provision --id <id> --models unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf`, then `$T run --id <id> scripts/session.sh --env SESSION=b2-proxy`. Defaults: `REPS=2 BUDGET_MIN=150 STREAMING=1` (BUDGET_MIN is when the last block may start; the run took 82 min before).

**Stages.** Build and toolchain 12 min; 4 ABBA blocks of 10 sessions, 17 to 19 min each, 76 min; streaming conformance 1 min; then the hook: every script in `B2_STAGES` (default `scripts/a6000_gap.sh`, the A6000 resident-decode profiling of `docs/m1-open.md` item 4, being added on branch `track-r-a6000-gap`) runs if present, limited to the time left (about 170 min). Without it the session ends after the proxy (1.8 h of box time).

**Verify after fetch.** `claude-code-proxy/summary.json` and `cache_classes.txt` exist; fetch `*.reqlog.jsonl` but do not commit them: run `scripts/toolcall_fixtures.py <reqlog> <out.json>` on each and commit the fixtures.

**Result.** Phase 3b Claude Code clause: tasks solved at least llama-server's, prompt-cache reuse at least llama-server's in the same run (93.6% before; ADR 0051 expects 94.3%), zero HTTP 400s; per engine `tool_error_rate`, `leaked_markers_total`, `tool_names_unknown_total` (ADR 0051). Decode ratio against llama-server for item 4.

## B3: tuner, planner dense fixture, detector, kernels (`b3-planner` and `b3-detector`)

As one session B3 is 170 to 215 min of script time, more than one box allows, so it is two profiles on two boxes in parallel. Both need an RTX 3090 (the planner fixture balloons a 24 GB card to 18 GiB free; the detector's VRAM pressure fault leaves 10 GiB).

### `b3-planner`

**Box.** RTX 3090, 32+ GB RAM, 9+ cores (the dense fixture ran at 9), 100 GB disk. `$T offers --min-cpu-cores 9 --download-gb 40 --min-disk 100`.

**Launch.** `$T run --id <id> scripts/session.sh --env SESSION=b3-planner` (the script fetches the 30B and the 32B).

| # | Stage | Estimate | Cumulative | Decides |
|---|---|---|---|---|
| 0 | fetch 30B (32B in the background) | 2 to 5 | 5 | |
| 1 | `moe_cache_tune.sh` on the 30B | at most 25 | 30 | S for B1's `llamacpp-cache` |
| 2 | planner build | 9 | 39 | |
| 3 | dense pass 1: anchors, baselines, 24 candidates once, cheapest first | 57 (20 candidates by 33) | 96 | planner dense exit test |
| 4 | dense pass 2: second repeats while time is left | 24 | 120 | repeat spread |

**Verify after fetch.** `cargo run -p tetraeder-planner --bin tetraeder-plan -- score bench/results/<date>/planner-dense --out bench/results/<date>/planner-dense/score.json`: at least 20 candidates with `measured_s`, regret at most 10%, Spearman at least 0.8 (the scorer pools the samples of every `planner-dense-*.json` of the passes). Once committed, `cargo test -p tetraeder-planner committed_fixtures_meet_the_exit_test` is the CI verdict. `moe-cache-tune/tune.json` has a `cpu_share`.

**Result.** Phase 3a planner exit test on the dense fixture (ADR 0040); baselines `base-fit` and `base-ot-hand` are the "beat llama.cpp fit" clause. `tune.json` is B1's input.

### `b3-detector`

**Box.** RTX 3090, 32+ GB RAM, 9+ cores, writable `/sys/fs/cgroup/io.max` preferred (else the disk fault uses competing readers), 80 GB disk. `$T offers --min-cpu-cores 9 --download-gb 20 --min-disk 80`.

**Launch.** `$T run --id <id> scripts/session.sh --env SESSION=b3-detector`.

| # | Stage | Estimate | Cumulative | Decides |
|---|---|---|---|---|
| 0 | fetch 30B, `exit_regime` build | 12 | 12 | |
| 1 | detector campaign: 3 calibration, 10 rounds of 4, 3 rounds of the 3 pairs | 35 to 78 | 47 to 90 | detector exit test |
| 2 | `kernels_cpu.sh` (if 18 min are left) | 10 to 15 | 57 to 105 | ADR 0045 kernel-code question |

kernels-cpu is more telling on B1's CPU class; if B1 has time left it can run there instead.

**Verify after fetch.** `tetraeder detector-score bench/results/<date>/detector` writes `score.json`; commit the campaign directory (obs, manifest, calib, harness, score.json) and `cargo test -p tetraeder-analysis --test campaign` gives the verdict: single-fault top-1 at least 90%, at most 1 false positive in the no-fault runs, both faults of a pair in the top 3 in at least 90%.

**Result.** Phase 3a detector exit test on real hardware (ADR 0041). kernels-cpu answers whether ik's kernels beat mainline at the same AVX2 ISA (ADR 0045 table).
