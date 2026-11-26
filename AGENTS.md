# Tetraeder: agent rules

Rust LLM inference engine on a patched ggml. Design: `docs/plan.md`. Decisions: `docs/adr/`.
ADRs record decisions and chosen thresholds; measurements go in `bench/results/<session>/README.md`, linked from the ADR.
Plain, short docs and comments; no em dashes.

## Rules

1. Start every phase with an ADR and the interfaces it touches; failing tests before code.
2. Numerical parity per op and end to end against reference logits, per backend and per split point; greedy golden sequences for regressions.
3. No optimization without a benchmark run from the pinned spec; results committed as JSON; CI fails on a regression beyond the noise band.
4. Planner and detector changes are tested on every simulated hardware fixture and fault first; real hardware confirms nightly.
5. Property tests on plans: memory never exceeded, every unit computed exactly once, every migration DAG ends in its target plan.
6. Every new component ships with its metrics, a dashboard panel and an alert rule in the same PR.
7. `unsafe` only in `ggml-sys` and `memory`, each block with a safety comment.
8. A ggml change goes into the patch queue with an ADR and backend tests; nothing is upstreamed; never edit vendored ggml in place. Code or binaries taken from another project get a row in `docs/sources.md` (repo, path, commit, license, how to update); copied code keeps its copyright notice.
9. Tests are labelled by the hardware they need and skip cleanly without it.
10. Small PRs naming the phase and exit test they move.
11. Every change builds and tests on Windows and Linux (`cargo build --workspace`, `cargo test --workspace`, `cargo clippy --workspace -- -D warnings`, `cargo fmt --check`).

## Decisions

Design and rule calls are delegated to the advisor (owner, 2026-11-25): the coordinator asks the advisor, records each call as "advisor, delegated by the owner <date>" and does not ask the owner. The owner is asked only to raise a money cap (GPU work section).

## Tech tree

Project state lives in the tech tree (skill `tech-tree`, lhns/design-skills; config `tech-tree.yaml`): `docs/roadmap/nodes.yaml` is the one source of truth, `docs/roadmap.md` and its pages are generated. Update the node in the same push as each result, merge or owner decision (state, outcome with a quoted verdict, plain sentence, new or unblocked nodes), then run
`python <design-skills>/skills/tech-tree/scripts/build.py --config tech-tree.yaml` and `check.py` with the same config; CI runs the check. `docs/m2-plan.md` keeps the detail of each open item; the order and state are the tree's.

Measured results also update `docs/features.md` in the same push: the row of the plan feature they measure, with the paired ratio copied from the session README or `verdict.json`, its scope and evidence path, including results where Tetraeder is behind or tied. CI checks that every evidence path exists and every ratio appears in it.

Merges to main (rule 10; the coordinator merges branches directly, a PR is optional) carry this in the merge message or PR body: the phase, the plan feature row it moves, and the paired ratio with its session, or "not measured".

## Local files

The machine is shared with other projects; 2026-11-20 cleanup removed 156 stale worktrees (57 GB) and stray box logs in the parent `git` folder.

- Write only inside this repo's worktrees or the session scratchpad. Never into `Documents/git` or another sibling folder: no logs, scripts or worktrees there (`taxameter` output, `> file` redirects and `git worktree add` paths included).
- Worktrees live under `.claude/worktrees/` (agent isolation) or the scratchpad. The coordinator removes a worker's worktree when its branch is merged or abandoned, and runs `bash scripts/wt_gc.sh` (dry run first, then `--apply`) after each merge batch and at session end, when no worker is active.
- Local result files that must survive go in `runs/` (git-ignored), not in a worktree that will be removed.
- Workers run only `cargo check --workspace --all-targets`, `cargo clippy --workspace --all-targets -- -D warnings` and `cargo fmt --all --check` locally (owner, 2026-11-23), with `CARGO_TARGET_DIR` set to `C:/tt/<lane>` (one short folder per worker lane, outside the worktrees; MSBuild fails on long paths). No `cargo build` or `cargo test` locally: CI builds and tests. Worktrees carry no `target/`. Every lane uses the one shared `CARGO_TARGET_DIR=C:/tt` (only one lane builds locally at a time, so it is reused and stays bounded); never a new folder per worker. On 2026-11-26 thirteen per-worker folders (`C:/tt2` to `C:/tt14`) had filled the disk (271 GB). The coordinator checks free disk space before starting a local lane.

## Session prompt template

```
Phase <n>, item <x>. Read docs/plan.md (section <s>) and docs/adr/<nnnn>.
Write the failing test first, then implement, then run
`cargo test`, `bench run <spec>` and check the dashboard panel.
Update the ADR if a decision changed. Report exit-test numbers.
```

## GPU work on Vast.ai (ADR 0021)

All GPU work goes through taxameter (`uv run taxameter ...`, config in `taxameter.toml`, ADR 0029). There is no other GPU. Commit before provisioning: `provision` and `run` push committed HEAD only. Overview and Task Scheduler setup: README, section "Vast boxes".

- **Key hygiene:** never print, log, commit or pass the API key (`~/.config/vastai/vast_api_key`) on a command line. taxameter reads it from the file, sends it only as a header and redacts every output line.
- **Money:** the box bills every running minute. Never leave the GPU idle and keep independent jobs running in parallel. Print `date +%T` at launch and at checks, estimate durations, flag overruns. Destroy the box when done, after a verified fetch. Caps (raised by the user 2026-11-13): 0.90 USD/h per box all-in and two instances for this project, 1.80 USD/h for all projects on the account (`max_account_price_per_hour`), 3 h per box (`max_box_hours`). Exception (user, 2026-11-15): a box with 4 GPUs may cost up to 1.20 USD/h all-in (Phase 5 four-GPU sessions); its driver passes `--max-price 1.20` for that box only and the account cap stays. One-off (user, 2026-11-20): one 2-GPU box for V2 on host 87485 up to 0.92 USD/h all-in (its traffic puts it over 0.90); used (V2, 2026-11-20). One-off (user, 2026-11-21): one 4-GPU box on host 20210 for the cost model 5 rehearsal s3g up to 1.22 USD/h all-in; used (s3g, 2026-11-21). One-off (user, 2026-11-26): the two boxes of the next judged Case S session (ADR 0091) may run up to 4 h each; `taxameter.toml` `max_box_hours` is 4 for that session only (the watchdog stops a box at it, `scripts/phase6_clock.py` reads it) and goes back to 3 in the push that commits its results. Host 20210 (user, 2026-11-21): its 4-GPU verdict boxes may cost up to 1.22 USD/h all-in (1.192 plus traffic). Traffic counts toward the per-box cap (owner, 2026-11-24): a driver passes the real download size to `offers` and `create` (`--download-gb`, about 150 for the 235B) so the cap check prices it, and prefers hosts with cheap traffic. Ask before raising them.
- **Several boxes:** with more than one box up, pass `--id` to every taxameter command (`create` prints the new id; `current` stays with the first box). Each agent drives only its own box and destroys it after its verified fetch.
- **Detached:** long work runs detached on the box (`nohup setsid ... &`). `taxameter ssh` stops remote commands after 120 s unless `--timeout`.
- **Waiting:** run wait loops on the box (`ssh --timeout 540 -- 'for ...; sleep 5; done'`), not as local polling loops. Claude Code reaps local background shells when the machine is low on memory.
- **Idempotent launches:** taxameter re-runs a command after a dropped connection, which once started duplicate jobs. Guard every launch script with `exec 8> /root/x.lock; flock -n 8 || exit 0`.
- **No inline patterns:** never put a `pgrep -f` or `pkill -f` pattern inline in `ssh '...'`. It matches the ssh shell's own command line and kills it. Put the pattern in a script file on the box.
- **Scripts:** never overwrite a shell script while bash is executing it. Upload new versions under a new name.
- **VRAM:** budget every job before launching, including its final evaluation, which can peak far above the main run. Queue jobs with wait-for-file instead of packing them, and check `nvidia-smi` when several jobs share a card.
- **CPU quota:** a container shows all host cores but has a quota of about 11. Cap side jobs (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, cargo `-j`) at 2-3. One uncapped job once dropped GPU use from 98% to 20%.
- **Models:** store them in `.hf_cache/<org>__<model>/` and run offline (`HF_HUB_OFFLINE=1`). `models` probes ModelScope and Hugging Face and takes the faster. Never start a second download of the same model.
- **CUDA:** never launch on a box whose provision log does not say `cuda: ok` or `cuda: ok_preload`.
- **External stops:** a box can be stopped from outside (`intended_status=stopped`). Processes die and the disk survives; `start` may need a second request. Long scripts save results incrementally and resume.
- **Blocklisted hosts:** 483833 (network down about 28 of every 45 min) and 366851 (stopped from outside twice, GPU rented away while stopped). Both ship in taxameter's Vast provider seed blocklist.
- **Configs:** before comparing runs, diff their configs.
- **Windows:** set `MSYS_NO_PATHCONV=1` for Git Bash commands with `/root/...` arguments.

### Benchmark comparability on Vast

Clocks cannot be locked in a container and the host changes nightly. Compare only paired runs: baseline and Tetraeder (telemetry on and off) in the same session on the same box, repeats interleaved. Measure a noise band per GPU class and gate regressions on paired ratios. Never compare absolute tok/s across nights. Record host id, GPU, driver, clocks and throttle reasons with every run.

Thread counts and devices are part of the pairing: the harness passes one explicit `-t` (default `available_parallelism()`, which respects the container CPU quota; `threads = N` per variant overrides) to both engines, and runs `device = "cpu"` variants with `CUDA_VISIBLE_DEVICES` and `GGML_VK_VISIBLE_DEVICES` empty, so CPU numbers never use the GPU. Each sample records its thread count; `env.json` records the CPU quota. Repeats come from the harness (`-r 1` per call); a variant may set `-p` and `-r` in its args for both engines (CUDA variants: `-p 2048 -r 3`).
