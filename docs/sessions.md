# Box sessions for the open M1 exit tests

Five exit sessions and a weekly GPU smoke, each a profile of `scripts/session.sh`. Written 2026-11-09 while box runs are on hold; nothing here has run yet. Rules: AGENTS.md "GPU work on Vast.ai" and the caps there (3 h and 0.45 USD/h per box, two boxes, 1.0 USD/h per account; B2 has the user's 48 GB exception, 6 h at 0.50 USD/h).

## How a profile keeps its cap

- **Budget.** 20% margin on the cap leaves 144 min of box time (288 for B2). Provisioning before the script (image pull, push, deps) and fetch, verify and destroy after take about 25 min, so the script gets `SESSION_MIN` = 120 min (260 for B2) from its first launch.
- **Deadline.** `session.sh` exports `DEADLINE` (first launch plus `SESSION_MIN`). `exit_run.sh` skips a stage whose estimate (`stage_min` in the script, the numbers below) would end after it; the planner fixture, the detector loop and the B2 hook stop starting work after it. Stages run in order of value, so a cut drops the diagnosis first.
- **Relaunch clock.** The deadline counts `SESSION_MIN` from the first launch (the time in `runs/vast/session-<name>.start`), so a relaunch with a larger `SESSION_MIN` still ends at the old start plus the new value. With `--env SESSION_RESTART_CLOCK=1` the relaunch counts `SESSION_MIN` from the relaunch time instead; the results date and session name still come from the first launch. `session.sh` logs which start it used (`deadline HH:MM:SS: SESSION_MIN n from ...`).
- **Measured stage minutes.** `stage_min` (`scripts/stage_budget.sh`) budgets builder 32K stages (`main-overlap*`, `main-prefetch*`, `main-repeat`) at 10 min and `llamacpp-cache*` at 30 min per share, but after each finished stage it records the wall time and later stages of that class use max(default, last measured * 1.1). On an EPYC 7642 at a 23-core quota the builder stages took 19.5 to 26 min, which the fixed 10 cut short.
- **RAM balloon guard.** `exit_run.sh` holds the difference to `EMULATE_RAM_GB` in a balloon process. Engines (builder, llama.cpp, llama-bench, KTransformers, Colibri, the cache tuner) start through `scripts/oom_engine.sh`, which sets their `oom_score_adj` to 1000; the balloon gets -1000 when the container allows it (the log says if not). Before and after every stage `scripts/balloon_guard.py` checks that the balloon is alive with RSS at least 90% of its size; if not, it is restarted and the stage's result JSON gets `balloon_ok: false` and `balloon_reason` (a `<json>.balloon.json` sidecar when the result is no object), and the script ends with rc 31. Stages with `balloon_ok: false` ran without memory pressure and do not count for the larger-than-RAM exit test (20261112f: the balloon died in the dio cache stage and later stages ran with the 142 GB model resident). Passing runs carry `balloon_ok: true`.
- **Incremental and resumable.** Results go to `bench/results/<RESULTS_DATE>/`, fixed at the first launch (`runs/vast/session-<name>.start`). A relaunch of the same profile skips what has results: `exit_run.sh` with `RESUME=1` skips stages with an `exit-<stage>-<stamp>.json` holding numbers; `tetraeder-bench run --resume` runs only missing repeats and rewrites its JSON after every pair; the detector resumes after its last observation; `kernels_cpu.sh` skips when its `kernels-cpu-*.json` exists; the Claude Code runner skips sessions with output; the cache tuner reuses its finished runs.
- **Launch and wait** (all profiles; `T="uv run taxameter"`, `--id` always once a box exists):

```bash
git merge origin/main && uv sync && git push            # provision and run push committed HEAD only
$T status                                                # at most two boxes up, ask "main" for a slot
$T offers --min-cpu-cores <cores> --download-gb <gb> --min-disk <gb>   # pick by RAM, cores, inet_down_cost
$T create --offer <offer id> --min-cpu-cores <cores> --download-gb <gb>  # prints the box id
$T provision --id <id> [--models ...]                    # then check the provision log says cuda: ok
date +%T; $T run --id <id> scripts/session.sh --env SESSION=<name> [--env ...]
$T ssh --id <id> --timeout 540 -- 'cd /root/tetraeder; for i in $(seq 100); do bash scripts/log_done.sh runs/vast/session-<name>.log SESSION_DONE && break; sleep 5; done; tail -5 runs/vast/session-<name>.log'
```

- **Logs append across relaunches.** Every launch writes a `=== RUN <time> <script> ===` header first (session, detector, kernels-cpu, model fetch); `scripts/log_done.sh <log> <marker>` is true only for a marker after the last header, so a `*_DONE` line of an earlier launch never ends a wait. Use it, not `grep -q`, in wait loops.
- **Submodule and line endings.** Nothing is needed locally: `scripts/ensure_patches.sh` / `patches/apply.sh` check `vendor/llama.cpp` on the box, and when it is not its own git repo (taxameter pushes no `.git`, or a dangling submodule `.git` file, and a Windows checkout with `core.autocrlf=true` pushes CRLF files) they wipe it and fetch the pin of `vendor/PIN` from the URL in `.gitmodules` into a fresh repo (a shallow fetch, about 100 MB; `PATCH_REMOTE` overrides the URL). Pushing the checkout is optional and wasted upload; if you do it, use `git -c core.autocrlf=false submodule update --init --depth 1 vendor/llama.cpp`. A `.gitattributes` rule cannot help: the submodule's files are not in the superproject's index, so only the submodule's own config (`core.autocrlf=false`) or the fresh box-side clone controls them.
- **ggml libraries.** Box scripts source `scripts/box_libs.sh`: it sets `TETRAEDER_GGML_NATIVE=ON` (static build, no `libllama.so`) and puts `target/release` and the ggml-sys `out/lib` dirs on `LD_LIBRARY_PATH`, so a binary run directly (the `exit_regime` test binary of the detector campaign) also finds the libraries of a DL build. The detector log prints the build mode (`ggml native` or `ggml dl`) after its build; `dl` means the env did not reach the build.
- **Fetch, verify, destroy:** `$T fetch --id <id>` (checks `bench/results/*/DONE`), then the profile's verify list below, then `$T destroy --id <id> --yes`. Report the box id, hours, `inet_down_billed` and `inet_down_cost`.
- Resume works on the same box only: a new box gets a new start file and date and starts over, as pairing needs (same-box runs only).
- If the session log stops advancing after `run`, relaunch with the same command (the flock makes a duplicate a no-op, `RESUME=1` skips finished work), or start it by hand: `ssh -- 'cd /root/tetraeder && (SESSION=<name> nohup setsid bash scripts/session.sh > /root/session.out 2>&1 < /dev/null &); sleep 30; tail runs/vast/session-<name>.log'`.

## Order and cost

B3 first: its cache tuner gives B1 the llama.cpp CPU share. B3-planner and B3-detector run in parallel on two boxes (0.90 USD/h together, under the account cap); B2 any time; B1 and B1b after B3, alone, together (0.90 USD/h) or beside B2 (0.95 USD/h). B1b is the second 235B box: it carries the lever stages (low-bit KV, routing trace) that do not fit next to the verdict.

| Session | Box | Script time (estimate / budget) | Box hours | Box cost | Download | Traffic cost | Total |
|---|---|---|---|---|---|---|---|
| B1 `b1-exit` | RTX 3090, about 91 GB RAM, 24 to 26 core quota, 220 GB disk | 115 / 120 min (the guard drops `ktransformers-avx2` and the cache repeat) | 2.4 | at most 1.08 | 142 GB | 0.92 to 3.69 | 2.0 to 4.8 USD |
| B1b `b1b-levers` | as B1 | 103 / 120 min core (through the trace); the optional tail `main-overlap-ram` (to 112) and `anchors-builder-short` (to 124) runs only while the deadline guard allows, cut last-first | 2.0 | at most 0.90 | 142 GB | 0.92 to 3.69 | 1.8 to 4.6 USD |
| B2 `b2-proxy` | 48 GB Ampere card (RTX A6000, A40), 64 GB RAM, 8+ cores, 100 GB disk | 82 min proxy, 40 kv-quality, hook / 260 min | 2.5 to 4.8 | 1.25 to 2.40 | 38 GB | 0.50 to 1.00 | 1.8 to 3.4 USD |
| B3 `b3-planner` | RTX 3090, 32+ GB RAM, 9+ cores, 100 GB disk | 106 to 120 / 120 min | 2.0 to 2.4 | at most 1.08 | 38 GB | 0.50 to 1.00 | 1.3 to 2.1 USD |
| B3 `b3-detector` | RTX 3090, 32+ GB RAM, 9+ cores, 80 GB disk | 65 to 110 / 120 min | 1.5 to 2.3 | at most 1.04 | 19 GB | 0.25 to 0.50 | 0.9 to 1.5 USD |

Every session except `b1-exit` (the verdict session, `NOISE_STAGE=1` opts in) now ends with the A/A pair of the "Noise bands" section when 12 minutes of its budget are left: it adds the 8B model (about 5 GB, 0.07 to 0.13 USD of traffic) and up to 12 minutes; `b1b-levers` and `b2-proxy` may skip it when the deadline is near.

Traffic is 0.013 to 0.026 USD/GB (Vast `inet_down_cost`, tracks seen so far); box 58900330 paid 0.92 USD for the same 142 GB (ADR 0045), so the low end exists. Rank offers with `--download-gb`.

## Sources of the estimates

- **Builder 32K run** (prefill 32768, 64 decode tokens): 8 to 9 min on the 91 GB class. `exit-main-prefetch-120800.json` load 6.6 s, prefill 453 s, decode 3.76 tok/s (17 s); `exit-d2-120800.json` prefill 469 s; `exit-main-021753.json` 454 s. On the 45 GB slow-disk class it is 21 min (`exit-main-prefetch-070656.json`: prefill 1239 s, 0.69 tok/s), so B1 needs the 91 GB class.
- **KTransformers 32K**: request 667 s (`exit-ktransformers-120800.json`, 26 cores) to 937 s (`-032306`, 15 threads). Load and the bf16 attention stub are in no committed artifact; `ktransformers_exit.sh` caps them at 40 min (stub 2400 s) and the stub is reused by later KT stages. Estimate 25 min for the first KT stage, 17 to 20 for the next. Least evidenced stage.
- **llama.cpp expert cache** (`llamacpp_cache.sh`, one share): never run. Proxy: the libllama `-ot` reference on the same model and mmap, `exit-reference-021753.json` load 52 s, prefill 707 s, decode 2.55 tok/s; plus the 256-token warmup and server start: 15 min.
- **Smoke**: prefill 8 tokens at 32K context, 37 to 61 s plus load (`exit-smoke-*.json`): 2 min.
- **Builds**: 8 min 51 s (`partial-planner-dense/planner_campaign-dense.log`), 11 min 11 s (`claude-code-proxy/claude_code_proxy.log`); both are single-variant (pre-dispatch) builds. Since ADR 0052 boxes build native (`TETRAEDER_GGML_NATIVE=ON`, addendum 2026-11-09), so the figures hold: one ggml CPU variant plus CUDA, during the download, in every profile. The DL default would add 13 CPU variants, an estimated 30 to 40 min of paid time on a fresh box (nothing is in sccache there). The old native-ISA second build of B1b (10 min, 15 GB) and its stages are gone.
- **Downloads**: no committed rate. `min_inet_mbps = 500` guarantees 62 MB/s: 142 GB in 9 to 38 min, 19 GB in 2 to 5 min.
- **Claude Code proxy**: 82 min for REPS=2 including the 11 min build (`claude_code_proxy.log`, 4 blocks of 17 to 19 min, streaming conformance 69 s).
- **Planner dense fixture**: measured wall per llama-bench run = predicted request time + prompt time + 5 s (llama-bench's warmup repeats the prompt; fit to the 14 runs of `partial-planner-dense/dense_rerun.log`, within 1 to 6 s each). Per repeat: 14 candidates with op offload 505 s, 10 without it 2450 s (pp 16 to 60 tok/s), GPU anchors 300 s (`anchor-gpu-small-opo-off` alone 264 s), CPU anchors at pp64 tg16 about 100 s, baselines 60 s: 57 min per repeat, 33 min for the anchors, baselines and the 20 cheapest candidates.
- **Detector campaign**: 3 + 40 + 9 = 52 runs of a 256-token prefill and 96 decode tokens on the 30B with misses from disk, plus the sampler. No box evidence; the 30B builder decodes 7 to 40 tok/s with slots and disk misses (`20261102/builder-real-202732.json`), so a run is 40 to 90 s with load and teardown: 35 to 78 min, plus a 10 min `exit_regime` build.
- **kernels-cpu**: 9 llama-bench runs of 64 tokens on the 30B (CPU only, about 15 tok/s at 9 threads from the planner CPU anchors) and up to 3 x 180 s of `objdump`: 10 to 15 min.
- **Low-bit KV and trace** (B1b): each `kv-*` stage is a builder 32K run, 9 min; the trace is one with 256 decode tokens plus the replay, about 15 min. **kv-quality** (B2): never run; 30B download, a model-crate build and 8K and 32K runs per format, capped at the time left (estimate 40 min).
- **ADR 0056 stages**: `disk-curve` is 6 depths x 4 s, twice (idle, loaded), plus the load of the file: 1 minute of reads, 2 with the bench build (cached after the first stage). `trace` is the box-C style run (`exit-trace` stage: 15 min on the 235B for 32K plus 256 decode tokens; the 30B at prefill 2048 and decode 128 with every expert in the RAM tier: 3 min) plus the replay of about 150 configurations (seconds). `anchors-builder-short`: two to four 235B runs at pp512 tg32 (`anchor-bld-ram-all` and `anchor-bld-opo-off` need every expert in the RAM tier and drop out on the 91 GB class, leaving `anchor-bld-vram-small` and `anchor-bld-vram-large`; prefill about 30 s at the box-C rate of 70 tok/s, decode 32 tokens at 3 to 4 tok/s, load 7 s: about 1.5 to 2 min each with teardown) plus the CUDA build of the `tetraeder` and `tetraeder-plan` binaries beside the cached test binary, 4 min the first time. None has run. `disk-curve` and `anchors-builder-short` write no `exit-<stage>-<stamp>.json`, so a relaunch with `RESUME=1` repeats the 2 minutes of `disk-curve`; the anchor loop skips finished `<id>.r1.json`.
- **moe-cache tuner**: its script ships in the box image, not the repo; no run count known. Capped at `TUNE_MIN` = 25 min; a relaunch resumes from its finished runs.

## B1: Phase 2 exit verdict (`b1-exit`)

**Box.** RTX 3090 (the llama.cpp baseline build is Ampere only, `BASELINE_CUDA_ARCHS="86"`), container RAM about 91 GB (the 235B must not fit; `exit_run.sh` balloons any larger limit down to 128 GB), CPU quota 24 to 26 (the gap to KTransformers showed at 26 and was a tie at 15, ADR 0045 addendum), NVMe at 5 GB/s or more, 220 GB disk (142 GB model, the KT stub, the build). `$T offers --min-cpu-cores 24 --download-gb 145 --min-disk 220`, pick an offer with 88 to 100 GB RAM. `--min-disk` only filters hosts: the box disk comes from `[policy] disk_gb` (80), so set `{"disk_gb": 230}` in the git-ignored `.vast/config.json` before `create` (also for `b1b-levers`; the 2026-11-10 run lost a box to this).

**Launch.** No models at provision (the script fetches the 235B): pass `--models ,`, since an empty `--models ""` falls back to the config's default models (about 24 GB, `20261111h`). `$T run --id <id> scripts/session.sh --env SESSION=b1-exit --env MOE_CPU_SHARE=<tune.json cpu_share from B3>`. Defaults: `VRAM_SLOT_BYTES=10966007808`, `OT_FIT='blk\.[0-6]\.ffn_.*_exps=CUDA0;exps=CPU'` (fitted for a 24 GB card at 32K). Without `MOE_CPU_SHARE` it takes the newest committed `moe-cache-tune/tune.json`, else 0.5 (logged).

**I/O attribution.** `disk-curve` is required in both B1 profiles (it is in both default stage lists; keep it in a `B1_STAGES` override): with the loaded curve every exit stage, builder or baseline, gets `host.decode.disk_util` and `io_class` (`disk_bandwidth`, `io_latency`, `io_under_issue`, `not_io_bound`, `unknown`) next to the I/O pressure shares (ADR 0056 addendum 2026-11-11).

**CPU attribution.** Each phase and decode window also carries `cpu_some_share`, `cpu_full_share`, `throttled_share`, `throttled_ms_per_s`, `cpu_steal_share` and `cpu_mhz_mean` (ADR 0058 addendum 2026-11-12); `session.sh` and `exit_run.sh` clamp `THREADS` to the whole-core CPU quota (`THREADS_ALLOW_OVER_QUOTA=1` keeps it).

**Co-tenant guard.** Each phase and decode window also carries `host_busy_cores` (all host CPUs), `container_cores` (cgroup CPU time over wall), `engine_cores` (engine tree), `outside_cores` (host minus container, at least 0), `smt_pairs_busy` (sibling pairs both above 50% busy, null without `/sys/.../topology/thread_siblings_list`) and `host_cpus`, in `host` and `host_otlp` alike. After every stage `scripts/cotenant_guard.py` sets `cotenant_ok` and `cotenant_reason` in the result JSON from the decode phase: false when `outside_cores` exceeds max(4, 15% of the host's physical cores (`host_cpus / 2`)) or, with at least 2 outside cores, `smt_pairs_busy` exceeds half the cores of the engine's threads; null when the counters are missing. It only logs a `cotenant ...` line in the session log and never fails the run. A pairing where either side has `cotenant_ok` false is not a verdict: rerun it or drop it (20261112g: a co-tenant on 62 host cores made the builder repeat 3.2x slower).

**Process probes.** Windows also carry `ctxsw_vol_per_s`, `ctxsw_invol_per_s` and `anon_huge_gb` (only in `*thp*` stages or with `SAMPLER_SMAPS=1`: reading smaps_rollup stalls the engine, ADR 0060 addendum 2026-11-14), `env.thp_mode` the THP setting; `exit_run.sh` runs a 10 s `engine_sample.py membw` before each builder and baseline stage (`env.membw`, or `membw-<stage>-<stamp>.json`; `MEMBW_PROBE=0` skips it). Windows also carry `cpus_used`, `numa_nodes_used` and `top_node_share` (the distinct CPUs and NUMA nodes the engine's threads last ran on in the window, and the share of thread samples on the busiest node; null without `/proc/<pid>/task/<tid>/stat` or the node files in `/sys/devices/system/cpu`). `StepProfile` adds `seg_pairs` (RAM pairs computed inside segment graphs, next to `cpu_pairs`). Stage `dispatch-floor` (STAGES only, about 3 min with the build, `stage_min` 3): the CPU dispatch floor of `cpu_bench.rs segment_overhead` at the box threads (1-node and 6-node graphs, spin gaps 0, 100 and 500 us, threads 1, n/2 and n, poll 0 and 50, pool pause on and off) to `dispatch-floor-<stamp>.json` (ADR 0064).

**Stages** (code order of `exit_run.sh`; every builder stage has prefetch on at depth 2, the ADR 0038 levers; host sampling on every builder, KT and llama.cpp cache stage). The verdict triple comes first and adjacent, then the noise pair; what does not fit is cut from the end:

| # | Stage | Estimate | Cumulative | Decides |
|---|---|---|---|---|
| 0 | download 142 GB, KT install, build (parallel) | 20 (to 38) | 20 | |
| 1 | balloon, `smoke` | 3 | 23 | gate: KV fits, direct I/O, split |
| 1b | `disk-curve` (idle and loaded) | 2 | 25 | the box's `nvme0` queue curve for the tier term (ADR 0056) |
| 2 | `main-prefetch` (builder, levers) | 9 | 34 | builder decode for the verdict |
| 3 | `ktransformers` | 25 | 59 | KT decode for the verdict |
| 4 | `llamacpp-cache` at the tuned share S, mmap | 15 | 74 | llama.cpp's best tuned setup (ADR 0045 addendum 2026-11-05) |
| 5 | `llamacpp-cache-dio`, same S, `--load-mode dio` | 15 | 89 | whether mmap streams or thrashes here; the faster of 4 and 5 is llama.cpp's number |
| 6 | `main-repeat` (builder, levers) | 9 | 98 | noise band of builder over KT |
| 7 | `ktransformers-repeat` | 17 | 115 | noise band |
| 8 | `ktransformers-avx2` | 20 | 135 | KT kernel ISA; runs only if the download was fast |
| 9 | `llamacpp-cache-repeat` | 16 | 151 | cache noise band; usually cut |

**Verify after fetch.** Every stage JSON has numbers (`decode_tok_s`, KT `ok: true`) and a `host` block. The llama-server log and host samples of each cache run are in `runs/vast/llamacpp_cache/<result name>.log` and `.samples.jsonl` (fetched; commit the samples, not the logs if large). In `exit-llamacpp-cache-s<S>-*.json` read `warmup.majflt`, `run_stat.majflt` and `log_load_mode`: if major faults run into the millions and decode is far below the builder's, llama-server thrashed mmap; compare with `exit-llamacpp-cache-s<S>-dio-*.json`. Both carry `prompt.fnv1a64`, which must equal the builder's (`c743c72b8ffceadd` for the frozen prompt; same token ids, BOS by the vocab's rule, special text not parsed).

**Result.** Phase 2 decode clause: builder `main-prefetch` and `main-repeat` decode at least KTransformers' and the llama.cpp cache's, with the two builder/KT ratios as the noise band (the bar is paired ratios only, never absolute tok/s across boxes). Read the gap stages with the table in ADR 0045, "Diagnosing the KTransformers decode gap". Commit the JSONs and update ADR 0045 and `docs/m1-open.md` item 1.

## B1b: lever stages on a second 235B box (`b1b-levers`)

**Box.** As B1. **Launch.** `$T run --id <id> scripts/session.sh --env SESSION=b1b-levers`. `--env B1_STAGES="..."` replaces the stage list (both B1 profiles; `exit_run.sh` runs them in the order of the list, once each; unset, in its code order; `DRY_RUN=1 STAGES="..." bash scripts/exit_run.sh` prints the sequence). After the 2026-11-10 verdict (low-bit KV failed the old absolute kv-quality bounds, since replaced by the excess-over-floor gate; the decode margin over KTransformers was within noise) the lever run swaps the kv stages for two more KTransformers pairs: `B1_STAGES="smoke disk-curve main-prefetch main-iosvc main-overlap trace ktransformers main-repeat main-overlap-repeat ktransformers-repeat"`. Stage `moe-tune` (about 45 min, `TUNE_MIN` 40) runs llama.cpp's cache tuner on the 235B under the balloon with the cache capped at the builder's VRAM slot bytes; a later `llamacpp-cache` in the same results directory runs the tuner's cache size and share instead of `CPU_SHARES`.

**KTransformers variant and threads.** `ktransformers-avx512` (and `-repeat`, opt in through `B1_STAGES`) forces `KT_KERNEL_CPU_VARIANT=avx512_vbmi` (`KT_AVX512_VARIANT` overrides); kt-kernel v0.7.2 accepts `amx`, `avx512_bf16`, `avx512_vbmi`, `avx512_vnni`, `avx512_base`, `avx2` and silently ignores other values, so the script rejects them up front and writes `ok=false` (`kt_cpu_variant_ok`) when the logged variant differs from the forced one. KT busy-spins its workers (50 ms before they sleep) and used 29.7 cores at 28 threads on a 30.7-core quota (20261113d, throttled 32% of decode periods), so on an AVX-512 CPU with a KT stage and no explicit `THREADS`, `scripts/kt_threads.sh` sets both engines to quota - 4 (`KT_HEADROOM`).

**Pairing stages** (opt in through `B1_STAGES`, 10 min each for the builder, 30 min per share for the cache): `main-overlap-tq4` runs main-overlap with `TETRAEDER_EXIT_THREADS` four below the quota (min 1), `main-overlap-poll0` with idle ggml threads sleeping; each has a `-repeat`. Combined with poll0 (builder, 10 min or the measured minutes): `main-overlap-poll0-thp` (`TETRAEDER_EXIT_RAM_HUGEPAGES=1`), `main-overlap-poll0-hot` (`TETRAEDER_EXIT_REFRESH_HOT=1`, hot set rebuilt from the whole prefill; the replay says VRAM share 0.253 against 0.178 at no upload cost) and `main-overlap-poll0-adapt` (`TETRAEDER_EXIT_ADMIT_SHARE=0.05`), each with `-repeat`. Cache variants: `llamacpp-cache-poll0` (`POLL=0` to `llamacpp_cache.sh`) and `llamacpp-cache-fp8` (`KV_TYPE=$LLAMACPP_FP8_KV_TYPE`, default `fp8` is a placeholder until the real type name is set), each with `-repeat`. `llamacpp-cache-dio-repeat` repeats the dio cache run, so one list can interleave every pair twice. `b1b-levers` now takes the llama.cpp CPU share like `b1-exit`: `MOE_CPU_SHARE`, else the newest `moe-cache-tune/tune.json`, else 0.5 (it was fixed at 0.5, so `MOE_CPU_SHARE` and `CPU_SHARES` had no effect there, 20261112c). `scripts/fetch_exit_model.sh` skips shards of the recorded size, probes both mirrors itself (0 KB/s or an error means Hugging Face) and switches mirror after 3 min under 20 MB/s.

`llamacpp_cache.sh` takes two optional env knobs, `KV_TYPE` (`f8_e4m3` passes `-ctk` and `-ctv`; llama.cpp FP8 KV, #28392, baseline pin b12274) and `POLL` (`0` passes `--poll 0`), both recorded in the result JSON with the server `--version`. FP8 KV frees about 3 GB of VRAM at 32K on the 235B (KV about 6 GB in f16, 3 GB in FP8), which can go to `--moe-cache-mib`: the caller adds it to the slot-bytes MiB it passes (`exit_run.sh` derives `mib` from the builder's VRAM slot bytes), and the builder must then get the same total VRAM budget (same extra bytes in its expert slots, or the same card budget), otherwise the pair is not fair. Without a raised `mib` the FP8 run only measures the conversion cost on Ampere (no FP8 math).

| # | Stage | Estimate | Cumulative | Decides |
|---|---|---|---|---|
| 0 | download, native build | 20 (to 38) | 20 | |
| 1 | `smoke` | 3 | 23 | |
| 1b | `disk-curve` (idle and loaded) | 2 | 25 | the box's `nvme0` queue curve for the tier term (ADR 0056) |
| 2 | `main-prefetch` (builder, levers) | 9 | 34 | the pair for every stage below |
| 3 | `kv-q8`, `kv-q8-slots`, `kv-q8q4`, `kv-q8q4-slots` | 36 | 70 | ADR 0050 b: decode with low-bit KV, and with the freed bytes as expert slots |
| 3b | `main-iosvc`, `main-overlap` | 18 | 88 | ADR 0054: the persistent I/O service alone, then with experts computed in arrival order; decode at least 15% above `main-prefetch` and hidden share at least 0.5 |
| 4 | `trace` and the replay | 15 | 103 | ADR 0050 c: frequency tiers against LRU (`replay-235b.json`) and, addendum 2026-11-14, the RAM policies across layers with 16-token decode windows (`replay-235b-ram.json`; calibration: `lru` windows within 5% of the run's own `decode_windows`). The trace stays in `runs/vast/`; its zstd copy `routing-235b-<stamp>.trace.zst` comes back with the results, git-ignored: after the fetch, move it to the main checkout's `runs/traces/` |
| 5 | `main-overlap-ram` (RAM tier with a 4 GB reserve) | 9 | 112 | ADR 0054 section 3; the deadline guard drops it unless the download was fast; peak RSS under 90 GB |
| 6 | `anchors-builder-short` (the builder anchors that fit at pp512 tg32, planner build) | 8 to 12 | 124 | ADR 0056: ledger coverage of the 235B session, `ledger-<session>/`; last, cut first |

**Polling and RAM-policy stages** (opt in through `B1_STAGES`, builder, 10 min or the measured minutes, each with a `-repeat`): `main-overlap-pause50` is main-overlap with explicit ggml polling `TETRAEDER_EXIT_CPU_POLL=50` and `TETRAEDER_EXIT_POOL_PAUSE=1`; `main-overlap-ramfreq` sets `TETRAEDER_EXIT_RAM_POLICY=freq:0.5`. Pair them with `main-overlap` and `main-overlap-poll50` on the same box. The engine switches land on other branches; until then the variables are ignored and the stages repeat main-overlap. The nightly fault testbed (`scripts/fault_testbed.sh`) now injects `cpu_quota_throttle` for real: a scope with `CPUQuota=` at half the threads and `TETRAEDER_EXIT_CPU_POLL=50` (manifest name `quota`, singles only; skipped with a logged reason when the runner enforces neither `CPUQuota` nor a writable cgroup v2 `cpu.max`). `detector_sample.py` reads `cpu.stat` and `cpu.pressure` of the engine's own cgroup first, so a scope quota is seen.

**Multi-arm stage `levers-snap` (opt in through `B1_STAGES`).** A pairing of builder stages spends about 26 min per arm prefilling the 32K prompt for 40 s of decode, so the arms sit an hour apart on a drifting host. `levers-snap` runs the arms in one process (`exit_regime_arms` in `crates/model/tests/exit_regime.rs`, env `TETRAEDER_EXIT_ARMS="name:ENV=V,ENV=V;name2:..."`): one prefill with the shared options, `kv_snapshot`, then for `SNAP_ROUNDS` rounds (3; odd rounds in arm order, even rounds reversed) each arm drops the model, loads it with its options, runs the first ubatch (the hot set is built from the first prefill call, as in a normal run), `kv_restore`s the snapshot, decodes 16 discarded warm-up tokens and times 64 in 4 windows. Default arms: `main-overlap` (auto poll), `poll50`, `pause50` (poll 50 and pool pause), `ramfreq` (`freq:0.5`), `thp`; `SNAP_ARMS` replaces them. Results: `exit-<arm>-r<k>-<stamp>.json` per arm and round, in the format of the other stages (the host sampler runs once over the whole process and `engine_sample.py summarize` cuts each result by its own decode windows, so `host`, the ratios and the summaries work unchanged), and `exit-levers-snap-<stamp>.json` with the prefill and snapshot numbers and, per result, whether its greedy tokens (warm-up plus timed) equal the first run's (`greedy_equal_all`; all five arms are meant to be token-exact, a flip is a near-tie to check against `margins`). In an arm's JSON `prefill_s` and `prefill_tokens` describe its preparation (load, first ubatch, restore, warm-up), not a prefill; `snapshot.shared` has the real one and `snapshot.restore_s`, `load_s` the reload cost.

- **Runtime or reload.** Every `LoadOptions` field is read once at load (`cpu_poll` builds the ggml threadpool, `pool_pause` is copied into the model, `admit_share` and `ram_policy` build the expert streamer, `ram_hugepages` advises the RAM tier's buffer at allocation), and `Qwen3Moe` has no setters. So all five arms reload; none switches at runtime. Every arm reloads in the same way, so the treatment is equal, and the RAM tier is cold after a reload for all of them (the warm-up window refills it only in part; `SNAP_REPLAY_LAST=1` recomputes the last ubatch after the restore, about 2 min per arm and round, which refills it like the prefill's end did). Runtime setters for `cpu_poll`, `pool_pause` and `admit_share` would remove the reload for those arms; not built.
- **KV on the streamed GPU path.** The snapshot is only the K and V buffers; the step inputs (tokens, positions, mask) are rebuilt from `n_past` every step, and the expert tiers, prefetch prediction, overlap state and hot set live outside the KV, so a restore changes timing, not values. `crates/model/tests/kv_snapshot_decode.rs` checks it on a streamed fixture with prefetch, the I/O service and overlap: a fresh model that ran the first ubatch and restored the snapshot decodes the same greedy tokens and logits as the model that prefilled (bitwise on the CPU backend; the `gpu_` variant uses the GPU tolerance and skips without a GPU).
- **Snapshot size and RAM.** At 32K the 235B's KV is about 6 GB in f16, over `TETRAEDER_EXIT_SNAP_RAM_MAX_GB` (2), so it goes to a file in `runs/vast/` (deleted at the end) and is read back whole for each restore (`kv_restore` takes one slice). That read is a 6 GB transient next to the RAM tier, so the stage sizes the RAM tier `SNAP_HEADROOM_GB` (7) below the other stages (`RAM_SLOT_BYTES`); compare these arms with each other, not with `main-overlap` from another stage.
- **Box time.** The prefill takes about 26 min (the host class that took 19.5 to 26 min per 32K builder stage); each of the 15 arm-rounds takes the model load (not yet measured on the 235B; `load_s` in the results) plus the first ubatch (under a minute), the restore (about 10 to 30 s from disk) and 80 tokens (about 50 s): 2.5 min plus the load. About 65 to 110 min in all, `stage_min` 90 (`SNAP_STAGE_MIN`), against about 6.5 h for the same 15 runs as separate stages. `SNAP_ROUNDS=2` cuts it by a third; the stage's own timeout is `SNAP_STAGE_RUN_MIN` (120), and results are written as each arm finishes. Example: `B1_STAGES="smoke disk-curve levers-snap"`.

**Adaptive compute tier (ADR 0059).** `main-overlap-adapt` and `main-overlap-adapt-repeat` run only when `B1_STAGES` names them: `main-overlap` plus `TETRAEDER_EXIT_ADMIT_SHARE=0.1` (decode uploads every tenth compute-tier miss into a VRAM slot, LRU seeded by the hot set). Interleave them with `main-overlap` and `main-overlap-repeat` on the same box, for example `B1_STAGES="smoke disk-curve main-overlap main-overlap-adapt main-overlap-repeat main-overlap-adapt-repeat"`, about 10 min each. Read `options.admit_share`, `decode_io.admitted`, `admitted_bytes` and `admit_ms_per_token` (the synchronous uploads) and `decode_layers.vram_pair_share` against the pair; other shares go through `VARIANTS="main-overlap-adapt-s05:TETRAEDER_EXIT_PREFETCH=1,TETRAEDER_EXIT_IO_SERVICE=1,TETRAEDER_EXIT_OVERLAP=1,TETRAEDER_EXIT_ADMIT_SHARE=0.05"`.

**Verify.** Every stage JSON has `decode_tok_s`; `replay-235b.json` exists. kv-quality (the KL bar for the kv-* formats) runs in B2.

## B2: Claude Code proxy rerun (`b2-proxy`)

**Box.** 48 GB card (CTX 131072 needs it), Ampere: RTX A6000 (the first run) or A40 (the box image's CUDA builds target sm_86; an Ada L40S is unverified), 64 GB RAM, 8+ cores, 80 GB disk, at most 0.50 USD/h (the user's exception, up to 6 h). `$T offers --gpu "RTX A6000,A40" --min-vram 48 --max-price 0.50 --download-gb 20`, then `create` with the same flags. 0.50 is above `max_price_per_hour = 0.45` in `taxameter.toml`, and the caps are not changed for it: renting above 0.45 USD/h needs the user's explicit go at launch time; without it, take an offer at or below 0.45.

**Launch.** `$T provision --id <id> --models unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` (kv-quality fetches Qwen3-30B-A3B itself), then `$T run --id <id> scripts/session.sh --env SESSION=b2-proxy`. Defaults: `REPS=2 BUDGET_MIN=150 STREAMING=1` (BUDGET_MIN is when the last block may start; the run took 82 min before).

**Stages.** Build and toolchain 12 min; 4 ABBA blocks of 10 sessions, 17 to 19 min each, 76 min; streaming conformance 1 min; `scripts/kv_quality.sh` (ADR 0050 b KL acceptance on Qwen3-30B-A3B, every expert resident on the 48 GB card; rc 30 when the test writes no JSON), about 40 min; then the hook: every script in `B2_STAGES` (default `scripts/decode_gap.sh`, the A6000 resident-decode profiling of `docs/m1-open.md` item 4 from branch `track-r-a6000-gap`; its microbench inside the proxy run is `MICROBENCH=1` on `claude_code_proxy.sh`, passed as `--env MICROBENCH=1` once that branch is merged) runs if present, limited to the time left (about 130 min). Without it the session ends after kv-quality (about 2.5 h of box time).

**Verify after fetch.** `claude-code-proxy/summary.json` and `cache_classes.txt` exist; fetch `*.reqlog.jsonl` but do not commit them: run `scripts/toolcall_fixtures.py <reqlog> <out.json>` on each and commit the fixtures.

**Result.** Phase 3b Claude Code clause: tasks solved at least llama-server's, prompt-cache reuse at least llama-server's in the same run (93.6% before; ADR 0051 expects 94.3%), zero HTTP 400s; per engine `tool_error_rate`, `leaked_markers_total`, `tool_names_unknown_total` (ADR 0051). Decode ratio against llama-server for item 4. `kv-quality.json`: Q8_0/Q8_0 mean KL at most 0.003 and top-1 at least 0.99, Q8_0/Q4_0 at most 0.015 and 0.97, at every depth.

## B3: tuner, planner dense fixture, detector, kernels (`b3-planner` and `b3-detector`)

As one session B3 is 170 to 215 min of script time, more than one box allows, so it is two profiles on two boxes in parallel. Both need an RTX 3090 (the planner fixture balloons a 24 GB card to 18 GiB free; the detector's VRAM pressure fault leaves 10 GiB).

### `b3-planner`

**Box.** RTX 3090, 32+ GB RAM, 9+ cores (the dense fixture ran at 9), 100 GB disk. `$T offers --min-cpu-cores 9 --download-gb 40 --min-disk 100`.

**Launch.** `$T run --id <id> scripts/session.sh --env SESSION=b3-planner` (the script fetches the 30B and the 32B).

| # | Stage | Estimate | Cumulative | Decides |
|---|---|---|---|---|
| 0 | fetch 30B (32B in the background) | 2 to 5 | 5 | |
| 1 | `moe_cache_tune.sh` on the 30B | at most 25 | 30 | S for B1's `llamacpp-cache` |
| 2 | planner build, then the builder fixture's session stages (`planner_builder_campaign.sh` with `STAGES="disk-curve trace"`): build 9 plus 3 for the `exit_regime` binary, `disk-curve` 2, `trace` and replay 5; skipped with fewer than 25 min left | 19 | 49 | ADR 0056: the 30B's disk curve and `tier-replay.json` for the builder fixture; the 3.7 h of candidate runs are not part of this session |
| 3 | dense pass 1: anchors, baselines, 24 candidates once, cheapest first (the build is cached) | 57 (20 candidates by 33) | 106 | planner dense exit test |
| 4 | dense pass 2: second repeats while time is left | 24 (14 fit) | 120 | repeat spread |

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

## GPU smoke (`gpu-smoke`, weekly)

A cheap RTX 3090-class box that checks the GPU code between exit sessions, so a CUDA regression shows up for under 0.30 USD and not on a 2 to 5 USD exit box. Driven from the local machine by `scripts/gpu_smoke.sh`; the box side is `SESSION=gpu-smoke` of `scripts/session.sh`.

**What it runs** (all with `TETRAEDER_REQUIRE_GPU=1`, results in `bench/results/<date>/gpu-smoke/`):

| # | Stage | Estimate | Output |
|---|---|---|---|
| 0 | provision (push, deps, 0.6B and 8B models), `cuda: ok` check | 12 (box 25 with fetch and destroy) | |
| 1 | CUDA test build (sm_86 only), the five `gpu_*` tests of `cpu_experts`, `quantized`, `slots` and `parity` (synthetic fixtures, no download) | 10 build, 1 run | `gpu-tests.log` |
| 2 | Qwen3-0.6B golden (`engine/tests/golden.rs`, CPU placement of the CUDA build) | 2 | `golden.log` |
| 3 | A/A pair `noise-cuda-dense` of `phase0.toml` (Qwen3-8B, 8 repeats): the noise-band sample | 4 | `phase0-<HHMMSS>.json` |
| 4 | `summary.json`, `DONE` | | `ok: true` only when every stage passed |

`gpu_cuda_graphs_decode` and `gpu_real_moe_streaming` need the 30B and stay in `nightly.sh` (`MOE_CHECK=1`). There is no GPU placement of the 0.6B golden yet (the golden is recorded for CPU).

**Skips fail.** `Backend::gpu_available()` panics when `TETRAEDER_REQUIRE_GPU` is set (not empty, not `0`) and there is no GPU, which covers every GPU test and `exit_regime`. A test that skips for a missing model prints `skip:`; `summary.json` is not `ok` if any such line is in the logs, or if fewer than `GPU_SMOKE_MIN_TESTS` (5) tests passed. Track `track-aa-test-infra` adds the same variable for models and GPU; when it merges, keep one definition (the panic in `gpu_available` is the GPU half).

**Cost** (script budget 30 min, `SESSION_MIN`, `GPU_SMOKE_SESSION_MIN` for a cold build on a new image tag; box life about 42 min = 0.70 h):

| Item | Value | Source |
|---|---|---|
| Box | price cap `(0.30 - traffic) / 0.75 h` = 0.19 USD/h with the 8B, 0.37 with `GPU_SMOKE_NOISE=0`; 0.13 USD for 0.70 h at the cap | `scripts/gpu_smoke.sh` computes it and passes `--max-price` to `create --auto` |
| Traffic | 6 GB (0.6 GB + about 5 GB 8B, `phase0.toml` note) at 0.013 to 0.026 USD/GB = 0.08 to 0.16 USD | `inet_down_cost` range above; the box image pull is not counted here (open) |
| Total | 0.21 to 0.29 USD per run, about 1.0 to 1.3 USD per month (4.3 runs) | |
| Time | build 10 min (8:51 and 11:11 in the two build logs cited above), tests about 1 min (`gpu-tests-120800.log`: two tests in 4.0 s), golden 2, A/A pair 4 (the whole `phase0` spec took 31 min), provisioning and teardown 25 (estimate above) | |

The box price itself has no committed sample (nobody measured the 3090 offers' price here); if no offer fits the cap, `create` fails and the wrapper exits 3 without renting. Raise `GPU_SMOKE_BUDGET_USD` only with the user's go (AGENTS.md: ask before raising caps).

**Run it** (dry run is the default and prints every command; `--run` rents a box):

```bash
git merge origin/main && uv sync && git push        # the box gets committed HEAD only
scripts/gpu_smoke.sh                                # dry run
scripts/gpu_smoke.sh --run                          # create, provision, cuda check, run, wait, fetch, verify, destroy
cargo run -p tetraeder-bench -- noise update        # fold the new A/A pair into bench/noise/, then commit both
```

It passes `--id` to every call after `create`, destroys the box on every exit path after a fetch (and with `--force` only when the fetch failed), writes `cost.json` next to the results (box id, minutes, price cap, taxameter's destroy output with `inet_down_billed` and `inet_down_cost`) and logs to `runs/vast/gpu-smoke-<stamp>.local.log`. Exit codes are at the top of the script. Verify: `summary.json` has `"ok": true`, `DONE` exists, an A/A results JSON is next to it.

**Weekly schedule (not registered).** Box runs are on hold; nothing here may be scheduled until the user lifts the hold and says so. Then, once, from a prompt (the time is an example; it needs the machine on, `uv`, Git Bash and the API key file as in the README section "Vast boxes"):

```bat
schtasks /Create /TN tetraeder-gpu-smoke /SC WEEKLY /D SUN /ST 03:00 /F /TR "cmd /c cd /d <repo> && bash scripts/gpu_smoke.sh --run >> .vast\gpu-smoke.out 2>&1"
```

Run it from a checkout on `main` that is pushed and clean (the wrapper refuses uncommitted changes). Also run it by hand before merging a branch that touches `ggml-sys`, `model`, `memory` or the patch queue.

**Relation to `nightly.sh`.** Not superseded as a script, superseded as a schedule. The nightly's job was "does the GPU code build and run, and is the Tetraeder engine within the noise band of llama.cpp"; the first half is the smoke at about a fifth of the cost. The second half (the Phase 1 gate pairs, `MOE_CHECK`, `GAP_PAIR`, the Phase 2 baseline spec) stays in `nightly.sh` for on-demand runs before a release or after a change to the bench or the engine's hot path; it should not come back as a daily job (`nightly_wait_min` 170, 1 to 2 USD a night).

## ADR 0056 stages (disk curve, trace, ledger directory)

Three stages feed the tier term and the calibration ledger. All are written into the scripts and the profiles above; none has run (box runs are on hold).

| Stage | Where | Minutes | Decides | Output |
|---|---|---|---|---|
| `disk-curve` | `exit_run.sh` (b1-exit, b1b-levers), `planner_builder_campaign.sh` (b3-planner) | 2 | the `nvme0` queue curve of the box, idle and beside `quota - 1` spinning threads (ADR 0054's starvation case): the tier term charges disk reads from it | `disk-curve-idle-<stamp>.json`, `disk-curve-loaded-<stamp>.json` next to the exit results; `planner-moe-builder/disk-curve-{idle,loaded}.json` |
| `trace` (builder fixture) | `planner_builder_campaign.sh` (b3-planner); `trace` of `exit_run.sh` for the 235B as before | 5 on the 30B | `tier-replay.json`: bytes per token for every slot pair of the fixture's anchors and candidates, so RAM options stop tying | `planner-moe-builder/tier-replay.json`; `routing.trace` when it compresses to under 20 MB |
| `anchors-builder-short` | `exit_run.sh` (last stage of b1b-levers; `ANCHORS_BUILDER_SHORT=1` elsewhere) | 8 to 12 (4 of them the first planner build) | ledger coverage of the session: the builder anchors that fit (two on the 91 GB class) at pp512 tg32 plus the session's builder stages as variants, with their read path | `ledger-<session>/` (`layout.json`, `candidates.json`, `<variant>.r1.json`) |

**What each session must export.** `session.sh` exports `TETRAEDER_GIT_COMMIT` and `TETRAEDER_SESSION` (part A); every result then carries them, and a result without them never enters the ledger. For the tier term the fetch must bring `disk-curve-loaded-*.json` (the scorer reads `disk-curve-loaded.json` of a fixture directory; copy the exit session's file there under that name) and, for the builder fixture, `tier-replay.json`.

**Verify after fetch.** `python3 scripts/disk_check.py bench/results/<date>/exit-<stage>-<stamp>.json bench/results/<date>/disk-curve-loaded-<stamp>.json`: the sampler's decode `read_mb_s` at `disk_queue_depth_mean` should lie within 20% of the loaded curve at that depth (reported, not gated). `cargo test -p tetraeder-bench committed_tier_replays_are_current` replays a committed `tier-replay.json` against its trace; `cargo run -p tetraeder-planner --bin tetraeder-plan -- ledger` shows the ledger rows of a new `ledger-<session>/` (add `--write` to snapshot, with `--accept-worse` for a reviewed regression).

## OTLP files of a box run (ADR 0060)

`exit_run.sh` starts a collector (`scripts/otelcol_box.sh`, config `deploy/otelcol/config-file.yaml`) before its first stage and stops it before `DONE`. It needs `/opt/otelcol/otelcol-contrib` (the provision hook installs it) and logs `otelcol: up, files in ...` to `runs/vast/exit_run.log`; without it the run goes on with the sampler's JSONL alone. `OTEL_COLLECTOR=0` skips it; `ENGINE_OTLP=0` keeps it but stops the builder's own export (the overhead pair of the ADR).

- **Where.** `runs/vast/otel/<date>-<stamp>/metrics.jsonl` (rotated at 100 MB into `metrics-<time>.jsonl`), one directory per launch, plus `otelcol.log`. About 120 MB per box hour plain. `otelcol_box.sh stop` is followed by `compress`, which turns every finished file into `*.jsonl.zst` (zstd -10, `--rm`), so a session's files are tens of MB; `taxameter fetch` brings them (`*.zst`, `*.jsonl` for a crashed run's plain ones). Readers take both. Commit the raw files under `bench/results/<date>/otel/` (copy the `<date>-<stamp>` directories without `otelcol.log` noise if large) only if the directory totals under 200 MB (`du -sk bench/results/<date>/otel`); over that, commit the results alone.
- **Format.** One OTLP JSON `ExportMetricsServiceRequest` per line: `resourceMetrics[].resource.attributes` (`service.name` is `tetraeder-host-sampler`, `tetraeder` or empty for hostmetrics; `tetraeder.stage` on the sampler and builder series), then `scopeMetrics[].metrics[]` with `name`, `unit` and `sum` or `gauge` `.dataPoints[]` (`timeUnixNano` and `asInt` are strings). Counters are cumulative.
- **Read them.** `python3 -c 'import sys; sys.path.insert(0, "scripts"); import otlp_host as oh; pts = oh.read_points(sys.argv[1:]); print(len(pts))' runs/vast/otel/*/metrics*.jsonl*` (plain or `.zst`; zstd needs Python 3.14 or the `zstd` CLI) gives the sampler and hostmetrics points as dicts (`name`, `attrs`, `resource`, `t_ns`, `value`); `oh.rows(pts)` turns them into sampler-shaped records. For any other series (the builder's `tetraeder.*`, `process.*`, `system.memory.usage`) use `oh.points_of(json.loads(line))` line by line. Into Grafana: point a collector with the `otlpjsonfile` receiver at the files and the compose stack's exporters.
- **In the results.** After the collector stops, `engine_sample.py otlp-merge --otel-dir runs/vast/otel --results bench/results/<date>` writes `host_otlp` (same fields as `host`, from the OTLP path) and `host_otlp_diff` (fields more than 5% apart or null on one side) into every result with windows. Rerun it locally after fetch the same way. Until one box run shows an empty or explained `host_otlp_diff`, `host` stays the sampler's.

## Noise bands

Gates compare the paired ratio of a candidate with `1 - band`. The band is per device and GPU class, from `tetraeder-bench`:

- **Sample.** One A/A pair (`role = "noise"`, the same variant twice, interleaved ABBA) per session. Every profile of `session.sh` except `b1-exit` ends with `noise-cuda-dense` (`noise_stage`, off with `NOISE_STAGE=0`, skipped with fewer than 12 minutes left); `gpu-smoke` always runs it, so a class gets about one sample a week. `nightly.sh` runs `noise-cuda-dense` and `noise-cpu-dense` as before.
- **Statistic.** A session's `d` is the largest `|median paired ratio - 1|` over its metrics (`pp_ts`, `tg_ts`): what the verdict compares with the band.
- **Band.** `band = clamp(1.25 * p95(d), 0.02, 0.25)` over the sessions of the class, p95 by nearest rank (the maximum below 20 sessions). The floor keeps a lucky quiet streak from gating on noise; the margin covers hosts not seen yet.
- **Class.** `cpu`, or the GPU name without vendor words: "NVIDIA GeForce RTX 3090" is `rtx-3090`, "NVIDIA RTX A6000" is `rtx-a6000`.
- **File.** `tetraeder-bench noise update [--results bench/results] [--out bench/noise]` scans all harness results, adds the A/A sessions it has not seen (by file and pair) and rewrites `bench/noise/<class>.json`: `n`, `n_hosts`, `d_p95`, `band`, the parameters, and per session the date, `host_id`, GPU, driver, threads, repeats and ratios. `noise show <class>` prints what a run would use. Commit the files with the results they came from.
- **Use.** `tetraeder-bench run` (default `--noise-dir bench/noise`) uses the class file when it has at least 3 sessions and otherwise the spec's provisional `[[noise_band]]`. Each pair result records `band`, `noise_class` and `band_source` (`measured rtx-3090 n=5` or `provisional rtx-3090 (measured n=2 < 3)`), and the summary text prints it. A corrupt file is an error for the pair, not a silent fallback.
- **Caveat.** Sessions on one box are correlated; `n_hosts` says how many distinct boxes the band rests on. Treat a band from fewer than 3 hosts as provisional in spirit.
