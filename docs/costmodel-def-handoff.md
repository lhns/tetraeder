# Handoff: ADR 0056 parts D, E, F (branch track-ah-costmodel-def)

Code for D, E and F is written and committed; the last local runs were green. Nothing is left to implement. What is left is CI confirmation and the report items below.

## Status per part

- **D (virtual clock in memory): done.** Confirmed green on CI (Linux and Windows) at `0b9e2d0` (D 1/2: Queue, Disk, SimIo, `Io` trait, `io_factory`, overlap and fidelity tests on SimIo). D 2/2 (`replay/timed.rs`, `Timing` hook in `replay/sim.rs`, `bench replay --timed`, `planner/src/timemodel.rs`, `explain --time-model`) was compiled and tested locally (memory with `--no-default-features`, planner, bench, control) but only has CI through the later pushes.
- **E (Clock::sleep, drift loop): done, not compiled anywhere.** `crates/tetraeder` (`replan/{engine,control,source,start,mod,closed_loop}.rs`, `serve.rs`) needs the ggml build, which could not run locally. CI is the first compile of it. Planner side (`planner/src/serve.rs`, `lib.rs`, `tests/serve_replan_sim.rs`) and `control/src/replan.rs` were tested locally.
- **F (box stages): done, nothing run on a box.** `bench/src/disk_curve.rs`, `memory/src/io/probe.rs`, `scripts/{planner_builder_campaign,exit_run,session}.sh`, `scripts/{ledger_stages,disk_check}.py`, `planner/src/{cli.rs,score.rs}` (`ledger-dir`, `DiskCurve`), `docs/sessions.md`. Shell syntax checked with `bash -n` only.

## Last CI result known

- `0b9e2d0`: success (24 min).
- Run 42705138770 on `08bae73` (F fix-up, includes everything D to F plus merge of origin/main `cd434d0`): was `in_progress` at 23 min when last checked (23:06); not read to the end. The run on `7ed9056` was cancelled by the next push.

## Known failures

None known. Likely spots if CI is red: unused imports or the `Arc<Scripted>` to `Arc<dyn Clock + Send + Sync>` coercion in `crates/tetraeder/src/replan/engine.rs` tests; the `closed_loop.rs` test (its planner twin passes locally).

## Next concrete step

`gh run list --branch track-ah-costmodel-def --limit 1` (one call). If green: read the `overlap` and `replay_fidelity` test wall times from `gh run view <id> --log` for the D report, then hand to the coordinator. If red: fix by part, label commits `D:`/`E:`/`F:`, push once.

## Open items

- Merge conflicts to expect with track-ag-replan-b: `crates/tetraeder/src/replan/*`, `crates/tetraeder/src/serve.rs`, `crates/planner/src/serve.rs` (`ServeInput` gained `tier`).
- No fake-clock test for the cut wait inside `restart` (needs a scheduler that does not unwind).
- On the 91 GB class only two builder anchors fit the budget, so `anchors-builder-short` yields two anchors.
- `disk-curve` reruns on a `RESUME=1` relaunch (2 min).
- Session budgets: b1-exit 115/120, b1b-levers 112/120 (`main-overlap-ram` and `anchors-builder-short` cut by the guard), b3-planner 106/120.
