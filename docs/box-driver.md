# Box session driver brief (read fully before any taxameter call)

You drive ONE Vast box for ONE session profile of Tetraeder and bring back committed, verified results. Money is real: the box bills every minute.

## Read first
- AGENTS.md, sections "GPU work on Vast.ai" and "Benchmark comparability on Vast", in full. Every rule applies: key hygiene (never print or pass ~/.config/vastai/vast_api_key), --id on every command, detached flock-guarded launches, waits on the box with `ssh --timeout 540`, no inline pgrep/pkill patterns in ssh, never overwrite a running script, `cuda: ok` before launch, destroy after a verified fetch, `date +%T` at launch and checks.
- docs/sessions.md: your profile's box requirements, stages, time estimates, launch command, verify and fetch steps.
- scripts/session.sh (SESSION, DEADLINE, RESULTS_DATE, RESUME), taxameter.toml.

## Caps (do not raise; never edit taxameter.toml caps)
- Per box: the max price in your task; 3 h box life unless your task says otherwise; the script budget in docs/sessions.md.
- At most two boxes for the project; another driver runs the other box. Never act on a box you did not create. Run `uv run taxameter status` before `create`; if two boxes are up, wait and report.
- Check CPU model, cores, host RAM and disk bandwidth before `create`: see "Machine choice" below. Slow disks and low-quota hosts cut stages short.
- `taxameter adopt <id>` takes the id as a positional argument (there is no `--id`); `taxameter status` must then show the box as `managed:active` before you run anything on it.
- Sessions that fetch their own model (b1-exit does, docs/sessions.md) provision with `--models ,` so provisioning downloads nothing.
- Pick offers by cores and traffic price: `uv run taxameter offers --min-cpu-cores <n> --download-gb <gb>` with the GB your session fetches, then `create` with the same flags and your price cap.

## Machine choice (standing policy, owner 2026-11-13: drivers pick, nobody asks)
The driver picks the box by these rules and does not ask the owner. The coordinator asks the owner only for spend above the ceiling, credit below one session, or a verdict.

1. **Hard requirements, checked on the offer** (`uv run taxameter offers ...`, whose price is all-in with disk; `vast_offers.py` is for CPU facts only, its listed price has twice been under the real one):
   - all-in price (hourly plus the "+traffic" share `offers` prints) at most the ceiling in your task (default 0.90 USD/h, taxameter.toml);
   - disk at least 230 GB and 3.5 GB/s, at least 24 GB VRAM, host not blocklisted (also avoid 269862: 15 to 25 MB/s downloads on 13d and 13e);
   - `memory.max` at least 150 GB, so the 91 GB budget is set by the balloon as on `20261113b` (check after SSH; the offer's RAM column can understate it). With `memory.max` near 91 GB the llama.cpp cache thrashed mmap and gave no ratio (`20261113g`);
   - point 1 (llama.cpp cache): Zen 2/3 EPYC, quota at least 24, so both engines run 24 threads without going over quota;
   - point 3 (KTransformers): GPU compute capability 8.0, 8.6, 8.9 or 9.0 (the kt-kernel wheel has no other; a Blackwell RTX 5090, 12.0, cost 13h its KT stages); AVX-512 with VNNI and VBMI (/proc/cpuinfo flags `avx512_vnni` and `avx512vbmi`, no underscore in the second; 13i misread a Zen 4 as lacking it); prefer CPUs without AMX (Zen 4, Ice Lake), where KT's own pick is its avx512 kernel; threads follow `scripts/kt_threads.sh` (quota minus 4: KT busy-spins its workers and was throttled 32% at quota minus 2 on 13d). Stages: without AMX, `ktransformers` and `ktransformers-repeat` (its own pick is its best kernel); with AMX, `ktransformers-avx512` and its repeat interleaved with the builder, then `ktransformers` (amx); the faster KT kernel is the baseline.
2. **Ranking:** lowest session cost = hourly price x planned hours + traffic (`dl$/GB` x 145 GB for the model). At 0.013 USD/GB the model alone costs 1.9 USD.
3. **Fit before create:** download (at 100 MB/s, about 25 min) plus the measured stage budgets (`scripts/stage_budget.sh`) must end by box age 2h25m. Drop stages from B1_STAGES up front rather than letting the deadline guard cut the one that matters.
4. **No qualifying offer:** report and stop; the coordinator rechecks the offers every 30 minutes and starts a new driver when one appears. Do not relax a requirement to fill the slot.
5. **Abort gates** (destroy after fetching what exists, then try the next offer once):
   - at SSH: missing ISA flags, too little `memory.max`, or no `cuda: ok`;
   - status `loading` (image pull) or SSH refused for more than 10 minutes after create (13f lost 40 minutes to three such boxes: host 213498 rejected the key on direct and proxy ports, host 546840 never finished the pull; skip both; host 109790 stayed loading twice on 13j, skip it too);
   - model download under 100 MB/s after its first 5 minutes;
   - KT sessions: `runs/vast/ktransformers_build.log` must end ok within 15 minutes of launch; a failed KT build ends the point-3 session (destroy, do not run builder-only stages);
   - smoke with `cotenant_ok` false: recheck for 10 minutes, then destroy (13c paid 80 minutes for invalid stages).
6. **Before paying for a stage:** grep docs/m1-open.md and ADR 0045 for it; a configuration already recorded as impossible (llama.cpp dio under the 91 GB balloon OOMs at load) is not rerun. Run `STAGES="..." pytest tests/test_stage_names.py` once it exists.

## Known box pitfalls (from b3-detector, 2026-11-10; follow these until the scripts are fixed)
- vendor/llama.cpp and bench/results are not pushed (main bd9933e): the provision hook fetches llama.cpp at the pin and applies the patch queue (`scripts/ensure_patches.sh`). A push over about 5 MB means the config regressed; stop and report.
- Prebuilt test binaries run directly (e.g. by detector_campaign.sh) need `LD_LIBRARY_PATH` including the ggml-sys build's `out/lib` (find it under target/*/build/tetraeder-ggml-sys-*/out/lib on the box) when relaunching.
- Logs append across relaunches: wait loops must check the LAST line (`tail -n1`) for `*_DONE` markers, not grep the whole file.
- `kernels-cpu` has no resume: do not relaunch it if its results exist.

## Flow
1. You are in an isolated worktree. `git fetch origin && git checkout -B <your-branch> origin/main`. taxameter pushes the committed HEAD of this worktree to the box; keep HEAD = origin/main until the results commit.
2. `date +%T`; `uv run taxameter status`; `offers`; `create` (record the id); `provision --id <id>` with the models your profile needs (docs/sessions.md); the provision log must say `cuda: ok` or `cuda: ok_preload`, else destroy and report.
3. Launch: `uv run taxameter run --id <id> scripts/session.sh --env SESSION=<profile> [other --env from your task]`. Check the session log advances within a few minutes; if `run` died silently, use the detached ssh fallback from docs/sessions.md.
4. Wait on the box with `uv run taxameter ssh --id <id> --timeout 540 -- '<wait loop>'`; between waits compare progress with the estimates and flag overruns. No tight local polling loops (the local machine is low on memory and reaps background shells).
5. At the session's done marker (or the deadline): `fetch --id <id>`, verify the expected files per docs/sessions.md "Verify", then `destroy --id <id> --yes`. Never destroy before a verified fetch unless the box is already gone.
6. Commit the fetched results (bench/results/<date>/..., JSON and small logs; no GGUF; no traces over a few MB) on your branch, message naming the session and box id; push the branch. Never push to main.
7. Report: box id, host, GPU, CPU cores/quota, price, wall time, cost (box plus traffic from `inet_down_billed`/`inet_down_cost`), stages run and skipped, key numbers per exit test (paired ratios on this box, not absolute tok/s across boxes), failures and why, result file paths, branch.

## On surprises
- A stage fails: read its log on the box; fix only a trivial env or path issue, with a new script name (never overwrite a running script), and note it; otherwise let the session continue and report.
- Anything that risks money (box unreachable, GPU idle, provisioning loops, a cap about to be hit): fetch what exists, destroy, report.
- Use the advisor when stuck before improvising.
