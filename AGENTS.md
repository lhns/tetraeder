# Tetraeder: agent rules

Rust LLM inference engine on a patched ggml. Design: `docs/plan.md`. Decisions: `docs/adr/`.
Plain, short docs and comments; no em dashes.

## Rules

1. Start every phase with an ADR and the interfaces it touches; failing tests before code.
2. Numerical parity per op and end to end against reference logits, per backend and per split point; greedy golden sequences for regressions.
3. No optimization without a benchmark run from the pinned spec; results committed as JSON; CI fails on a regression beyond the noise band.
4. Planner and detector changes are tested on every simulated hardware fixture and fault first; real hardware confirms nightly.
5. Property tests on plans: memory never exceeded, every unit computed exactly once, every migration DAG ends in its target plan.
6. Every new component ships with its metrics, a dashboard panel and an alert rule in the same PR.
7. `unsafe` only in `ggml-sys` and `memory`, each block with a safety comment.
8. A ggml change goes into the patch queue with an ADR and backend tests; nothing is upstreamed; never edit vendored ggml in place.
9. Tests are labelled by the hardware they need and skip cleanly without it.
10. Small PRs naming the phase and exit test they move.
11. Every change builds and tests on Windows and Linux (`cargo build --workspace`, `cargo test --workspace`, `cargo clippy --workspace -- -D warnings`, `cargo fmt --check`).

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
- **Money:** the box bills every running minute. Never leave the GPU idle and keep independent jobs running in parallel. Print `date +%T` at launch and at checks, estimate durations, flag overruns. Destroy the box when done, after a verified fetch. Caps: 0.45 USD/h, one instance. Ask before raising them.
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
