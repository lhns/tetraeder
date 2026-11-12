# Box session driver brief (read fully before any taxameter call)

You drive ONE Vast box for ONE session profile of Tetraeder and bring back committed, verified results. Money is real: the box bills every minute.

## Read first
- AGENTS.md, sections "GPU work on Vast.ai" and "Benchmark comparability on Vast", in full. Every rule applies: key hygiene (never print or pass ~/.config/vastai/vast_api_key), --id on every command, detached flock-guarded launches, waits on the box with `ssh --timeout 540`, no inline pgrep/pkill patterns in ssh, never overwrite a running script, `cuda: ok` before launch, destroy after a verified fetch, `date +%T` at launch and checks.
- docs/sessions.md: your profile's box requirements, stages, time estimates, launch command, verify and fetch steps.
- scripts/session.sh (SESSION, DEADLINE, RESULTS_DATE, RESUME), taxameter.toml.

## Caps (do not raise; never edit taxameter.toml caps)
- Per box: the max price in your task; 3 h box life unless your task says otherwise; the script budget in docs/sessions.md.
- At most two boxes for the project; another driver runs the other box. Never act on a box you did not create. Run `uv run taxameter status` before `create`; if two boxes are up, wait and report.
- Check CPU model, cores, host RAM and disk bandwidth before `create`: from the main checkout run `uv run python -I scripts/vast_offers.py --min-ram-gb 128 --min-disk-mbps 1000 --cpu-regex EPYC --max-price 0.45 --gpu "RTX 3090"` (any filter is optional). Slow disks and low-quota hosts cut stages short.
- `taxameter adopt <id>` takes the id as a positional argument (there is no `--id`); `taxameter status` must then show the box as `managed:active` before you run anything on it.
- Sessions that fetch their own model (b1-exit does, docs/sessions.md) provision with `--models ,` so provisioning downloads nothing.
- Pick offers by cores and traffic price: `uv run taxameter offers --min-cpu-cores <n> --download-gb <gb>` with the GB your session fetches, then `create` with the same flags and your price cap.

## Known box pitfalls (from b3-detector, 2026-11-10; follow these until the scripts are fixed)
- Before provisioning: in your worktree run `git -C vendor/llama.cpp config core.autocrlf false 2>/dev/null; git submodule update --init --depth 1 vendor/llama.cpp` and make sure the submodule files have LF endings (`git -C vendor/llama.cpp config core.autocrlf false && git -C vendor/llama.cpp checkout -- .`). taxameter only pushes the submodule if it is checked out; CRLF files break the patch queue on the box.
- On the box, `scripts/ensure_patches.sh` needs a git repo at the pin: if it fails with "not a git repository", create one in `vendor/llama.cpp` at the pinned commit (`vendor/PIN`) and apply `patches/series` (bash patches/apply.sh), as a new script, then relaunch.
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
