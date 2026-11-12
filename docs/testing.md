# Testing and benchmarking

State as of 2026-11-09 and the roadmap. Rules are in `AGENTS.md`; this file says what runs where.

## What runs where

| Layer | What it covers | Where |
|---|---|---|
| Unit and property tests | planner proptests, plan invariants, slot caches, gateway templates | CI, Linux and Windows |
| Cost model and DES simulator | analytic roofline plus FIFO devices and links, one op per layer, M=1 decode, no tiers, disk or experts; models time only | CI (planner search) |
| Fault simulator, routing replay | detector signals from nominal busy times; replay counts bytes, not time | CI |
| Synthetic models | `REALISH` MoE, a tiny f16 fixture, the slab-sized `exit_regime` model (ignored by default) | CI (CPU) |
| CPU parity | builder against libllama on the synthetic models; Qwen3-0.6B golden (`crates/engine/tests/golden.rs`), run pinned to the SSE4.2 tier against a per-OS file (`bench/golden/*-sse42-<os>.json`) | CI (Linux and Windows), box |
| Fakes | `FakeDevice`, fake llama-bench, `serve --fake` | CI |
| CUDA and Vulkan jobs | compile and link only, no GPU, not path-filtered | CI |
| GPU tests (`gpu_*`), real-model runs | logits against libllama on GPU, 235B decode, speculative, KV formats | Vast boxes only |
| Box sessions | `docs/sessions.md`; nothing has run yet (box runs on hold) | Vast |
| Bench gates | paired same-box ratios; bands are provisional | box |

Never tested end to end: the speedup of the prefetch levers, I/O overlap on a real disk, replanning against a real engine, the AVX-512 kernels (the SDE job is on branch `track-z-iqk-kernel`).

## Skips

A skipped test used to count as passed. Now:
- `TETRAEDER_REQUIRE_MODELS=1` (set in the CI test job) makes tests that need the Qwen3-0.6B GGUF fail when it is missing. CI fetches it with `cargo xtask fetch-model` and caches `.hf_cache`.
- `TETRAEDER_REQUIRE_GPU=1` (set in the box scripts that run GPU tests: `exit_run.sh` gpu-tests, `moe_check.sh`, `graphs_check.sh`, `nightly.sh`) makes a missing GPU fail the test. It is implemented in `Backend::gpu_available()` (branch `track-ad-gpu-smoke`), so every GPU test and `exit_regime` honors it.
- Skips print `skip: ...`; the CI test job lists those lines in its step summary ("Skipped tests").

## Build modes

| Where | ggml | Why |
|---|---|---|
| CI, release packaging | DL, 14 CPU variants (`TETRAEDER_GGML_NATIVE=OFF`, set in `ci.yml`) | exercises the dispatch; ships one binary for every x86 host |
| Local dev, boxes | native, one variant (`.cargo/config.toml` `[env]`, `docker/box/Dockerfile`) | build time; ADR 0052 addendum |

CI compiles ggml-cuda through sccache in the two CUDA jobs (`.github/actions/sccache`, a local disk cache that actions/cache saves on main only and every branch restores). rust-cache does not keep workspace crate outputs, so `tetraeder-ggml-sys` was rebuilt from scratch on every run. Measured on this branch (step `cargo build` of the CUDA job, cold then warm): Windows 1695 s then 425 s (99% hits; job 35 then 14 min), baseline before the change 22 min. On Linux sccache made the CPU-variant jobs about three times slower (clippy 68 s to 354 s) with no gain, so test and Vulkan jobs do not use it; the GitHub cache backend of sccache lost most writes on Linux, and branch-written caches were evicted within the hour by the 10 GB repository limit, hence main-only saves. rust-cache and the model cache are now saved on main only too (a branch entry of every job, about 2 GB per push, was what evicted the sccache entry); branches restore main's. Linux CUDA warm (branch `track-ai-ci-followup`, main's entry restored): `cargo build` 102 s against 1335 s cold, job 3 min against 25. The native-static job must not use it (`-march=native` objects must not move between runner CPUs).

## CI test job time

The Linux `test` job went from about 6 to 15 to 21 min with the DL build (ADR 0052): clippy (check), `cargo build` and `cargo test` are three build-script runs with separate `OUT_DIR`s, and each compiled the 14 CPU variants of ggml (4 min each on Linux); the tests themselves ran about 2 min. The job sets `TETRAEDER_GGML_BUILD_DIR` so all three share one cmake tree. Step times, Linux (before, run 42697629947, then after, run 42711806706): clippy 252 s then 251 s, `cargo build` 247 s then under 30 s, `cargo test` 382 s then 158 s, job 937 s then 495 s. Windows: build 216 s then 34 s, test 413 s then 347 s, job 975 s then 873 s (clippy 238 s then 332 s, noise on a shared runner). The multi-variant build is still exercised on every push, in the first step.

## Goldens

Float results depend on the CPU tier and on the OS, so `bench/golden/` holds one pair per `<variant>-<os>` (llama-simple text and our bench golden), and `golden.rs` picks `TETRAEDER_GGML_CPU_VARIANT` (default `sse42`) and the running OS. CI runs the golden pinned to `sse42` on both OSes. On Linux the greedy token at position 1 of the `capital` case flipped against the Windows file: top-2 logits 19.063 ('.') and 19.039 (',') on Linux, 19.136 (',') and 19.118 ('.') on Windows, so a gap of 0.02 in both and logits that differ by up to 0.07 between the OSes. That is a near-tie, not a bug; the cause of the 0.07 (libm in softmax and norms, or compiler codegen of the same C) is not isolated. The other two cases are identical on both OSes. To record a Linux file again, run the `ci` workflow with `record_golden` (the `golden-linux` job builds llama-simple pinned to SSE4.2, records both files and uploads the artifact `golden-linux`). A new tier or OS needs the same two files.

## Local builds for parallel agents

- One `target/` per worktree (nothing sets `CARGO_TARGET_DIR`); a shared local sccache for rustc and ggml objects.
- Use `bash scripts/dev-cargo.sh <cargo args>` instead of the single shared lock. It allows `DEV_CARGO_SLOTS` (default 2) concurrent cargo jobs on the machine, gives each `CARGO_BUILD_JOBS=4` (the 16 GB VM: raise only for a lone job, lower to 2 for ggml with CUDA), and turns sccache on when it is installed (no-op otherwise). Slots are directories in `~/.tetraeder-cargo-slots` holding the owner pid; a dead owner's slot is reclaimed.
- Measured on the VM (`cargo check` of model, engine, ggml-sys with tests, fresh target dir, 2 jobs, native ggml): 418 s plain, 496 s through `dev-cargo.sh` with a cold sccache, 657 s for a second fresh target dir (other agents were building; 0 hits). sccache gave no hit across target dirs for `cargo check`, and the Visual Studio generator ignores the CMake launchers, so on Windows the shared cache does not pay yet; the slot cap and per-worktree target dirs are the useful part. Treat the sccache hookup as unproven locally.
- To test the DL build locally set `TETRAEDER_GGML_NATIVE=OFF` (14 variants: slow on the VM; CI does it on every push).
- Prefer `-p <crate>` and targeted tests while iterating; CI runs the full Rule 11 set.

## Roadmap

1. **Planner verification by simulation is verification of the optimizer under the cost model, never of the model's accuracy.** Scoring the planner against its own model gives regret 0 by construction. The regret exit test needs box measurements; simulation covers enumeration, memory feasibility, the layer DP, expert greedy, quality knapsack, pins and forbids, `lower`/`recognize` round trips (ADR 0055), and the replan decision under a virtual clock.
2. **Virtual clock, disk curve and tier term.** A virtual clock in the I/O service and prefetch; a measured disk curve (bytes and latency at queue depth, from the `dio` tests) that turns replayed bytes into miss-wait time; CPU expert time from the calibrated thread curve; the missing RAM and disk tier term in the cost model (ADR 0040). Deterministic millisecond tests for overlap order, prefetch depth, RAM-tier policy and replan triggers. 4 to 6 days.
3. **Fidelity ledger (needs an ADR first).** Re-predict committed box results and fail outside the anchor band (9.2% MoE, 5.8% dense). Results must be compared like with like: a result carries the engine commit, so a re-prediction is checked only against measurements of the same engine build or against a versioned cost model. How versions and commits are recorded and which results stay comparable is design work, not a test.
4. **Fault testbed on CPU runners (nightly).** `exit_regime` on the synthetic model under cgroup faults (`IOReadBandwidthMax`, a thread-capped busy loop, `MemoryMax`, a `CPUQuota` below the thread count); four of nine fault classes, too noisy to gate PRs.
5. **GPU smoke on Vast** once the hold lifts: weekly or before merging GPU code, the `gpu_*` tests plus the 0.6B golden, about 40 min and at most 0.30 USD.
6. **Measured noise bands** per GPU class from the A/A pairs the sessions already run (gating code exists).
7. **AVX-512:** merge the SDE job (ADR 0053); pin goldens per tier with `TETRAEDER_GGML_CPU_VARIANT`.
8. **Nightly CI on GitHub:** everything, plus `exit_regime`, `cpu_bench`, the SDE job and the fault testbed.

## Rejected

- A fake GPU device that runs real ggml and charges simulated time: CPU parity already covers correctness, the timing would come from the same cost model, and it cannot model async CUDA overlap, the part worth testing. Item 2 gets most of the value for less.
- GitHub GPU runners (plan cost, T4 is sm_75 while we build sm_86, ADR 0021 puts GPU work on Vast) and a self-hosted runner on Vast.
- Per-PR box runs, wall-clock performance gates on shared runners, a model zoo, goldens for all 14 CPU variants.
- Path-filtering the CUDA and Vulkan jobs: a regression there must show on every push; sccache makes the unfiltered job cheap instead.

## Done on `track-aa-test-infra`

- sccache for ggml in CI (test, CUDA, Vulkan jobs); timings in the branch report.
- Qwen3-0.6B cached in CI; `TETRAEDER_REQUIRE_MODELS` and `TETRAEDER_REQUIRE_GPU`; skipped-test list.
- Native ggml for boxes and local dev, DL in CI (ADR 0052 addendum); `main-native` stages removed; session estimates redone.
- `scripts/dev-cargo.sh`: slot lock, per-job build cap, optional sccache.
