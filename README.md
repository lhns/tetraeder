# Tetraeder

An LLM inference engine in Rust on a patched ggml: one placement plan expresses every parallel mode and memory tier, a measured-cost planner chooses it for the hardware at hand, and built-in OpenTelemetry observability finds the bottleneck and proposes the fix.

You set limits such as a target tok/s, a quality floor and which devices it may use; the planner picks formats per layer and expert (mixed quantization, measured per model) and placement together. See [Quantization and quality](docs/plan.md#quantization-and-quality).

- Design, phases and exit tests: [docs/plan.md](docs/plan.md)
- Decisions: [docs/adr/](docs/adr/README.md)
- Rules for contributors and agents: [AGENTS.md](AGENTS.md)
- Tests, benchmarks, build modes and the testing roadmap: [docs/testing.md](docs/testing.md)

Status (2026-11-19): M1 reached 2026-11-14; Phase 4 (own KV cache, continuous batching) passed its exit test 2026-11-17; Phase 5 (own executor, multi-GPU) is in progress. Where it stands: [docs/m2-plan.md](docs/m2-plan.md), section "Phase 5: where it stands".

Targets Windows and Linux. Build and test: `cargo build --workspace && cargo test --workspace`.

## Name

A tetrahedron (German: Tetraeder) has four faces, one for each of the four parallel modes the engine combines: tensor, pipeline, expert and data parallel. The core idea of the project is that all four are shapes of one placement plan, which the planner chooses together. Four also fits the four memory tiers (VRAM, RAM, NVMe, remote node) and the four milestones.

## Serving and Claude Code

`tetraeder serve` runs the OpenAI API (`/v1/chat/completions`, `/v1/completions`) and the Anthropic API (`/v1/messages`, `/v1/messages/count_tokens`) on one port. It prints `listening on http://host:port` to stderr when it is ready.

```
# dense model through libllama (add --engine dense for the own Qwen3 dense builder, ADR 0066)
tetraeder serve -m model.gguf --alias qwen --ctx 65536 --port 8080
# Qwen3 MoE through the builder on GPU 0, experts streamed from disk if slots are short
tetraeder serve -m Qwen3-30B-A3B-Q4_K_M.gguf --engine moe --alias qwen --ctx 65536 --expert-slots 64
# scripted engine, no model (SDK tests)
tetraeder serve --fake
# a plan chosen for this box (or a plan TOML file), replanned by drain and restart
tetraeder serve -m model.gguf --plan auto --hw calibrated-graph.toml
```

`--plan auto|<file>` replaces the placement flags: it picks the fastest predicted plan among the ones this box can run (`--hw` is a hardware graph, by default the box is discovered and GPU rates come from the priors; `--plan-workload P:N`, `--plan-vram`, `--plan-ram`), lowers it to the engine, compares the measured decode step with the prediction every `--replan-tick` seconds, and on drift, on `POST /admin/replan`, or on an SLO burn alert posted to `/api/v2/alerts` (vmalert's notifier) it stops admitting, drains, restarts the engine on the new plan and resumes (ADR 0055). The admin listener (`--admin`, default `127.0.0.1:8081`, no authentication) also serves `GET /admin/plan`. Replanning needs a graph the cost model can predict with; otherwise the first plan runs and replanning is off.

MoE builder with a RAM tier (`--ram-slots`): experts are read by the persistent I/O service (`--io-service on|off`, default on), and with `--miss cpu` the CPU experts of a decode step are computed as their reads land (`--overlap on|off`, default on; it needs the service) (ADR 0054). Prefetch of predicted experts is on too (`--prefetch on|off`, bare `--prefetch` means on; `--prefetch-depth 1|2`, default 2). The same defaults apply under `--plan`.

Up to `--parallel` requests (default 4) decode together (continuous batching, ADR 0065); more wait in a queue (`--max-queued`). Finished conversations stay resident in the KV cache, so an agent that resends its whole history only pays for the new tokens. Sequences that need to give way are saved to host RAM (`--cache-ram <MiB>`, default 8192, 0 off) and restored when their prefix comes back (ADR 0074).

Claude Code:

```
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_AUTH_TOKEN=local ANTHROPIC_MODEL=qwen ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen claude
```

Claude Code also asks for its own `claude-*` model ids; `--accept-any-model` (or `--default-model <id>`) maps unknown ids to the served model. It sends 15K to 30K tokens of system prompt and tools on each turn, so start with `--ctx 65536` or more. `--request-log <file>` appends one JSON line per request (prompt, response, token counts, timings) for replay; it contains your code, so keep it local. With `TETRAEDER_DECODE_PROFILE=1` each line also gets a `profile`: decode steps and the time per stage (eval, sampling, hand-off, gateway) and the engine's graph counters.

llama-server's own routes work too (ADR 0089): `/health`, `/props`, `/tokenize`, `/detokenize`, `/apply-template`, native `/completion`, `/infill`, `/v1/responses`, and `/metrics` with `--metrics`. `--api-key KEY` (or `--api-key-file`) asks every route but `/health` for a key in `Authorization` or `X-Api-Key`.

## Measured against other engines

Paired same-box ratios, Tetraeder over the other engine, worst case included. Each line holds only in its scope; the full table, with evidence per row, is [docs/features.md](docs/features.md). What works or not against llama-server (not speed) is the parity matrix, [docs/notes/llamacpp-parity.md](docs/notes/llamacpp-parity.md) (ADR 0083).

- Qwen3-235B-A22B beyond RAM, decode against the llama.cpp expert cache: 1.23 to 1.69 and 1.21 to 1.52 on two hosts (2026-11-13), but 0.643 to 0.748 on one EPYC 7642 host at 24 threads on a 23.04 quota (2026-11-12).
- Same model against KTransformers: level at the default expert slots (0.85 to 1.15, RTX 4090 48 GB); 1.28 to 1.65 with the free VRAM as expert slots; 0.555 to 0.624 on a 0.5 to 0.7 GB/s disk.
- Concurrency 4 and 1 against llama-server on one RTX 3090 (2026-11-17): Qwen3-8B 0.995 to 1.022, Qwen3-30B-A3B 0.943 to 0.990. The 30B is behind at concurrency 4 and passes only inside the noise band.
- Two to four RTX 3090 against llama.cpp `-sm tensor` and `-sm layer` (2026-11-21, one host): 1.011 and 1.022 for tensor split, 1.008 and 0.987 for layer split.
- Time to first token on the Claude Code replay against llama-server (2026-11-19): median 0.93 to 0.94, p90 0.84.

## Vast boxes

All GPU work runs on ephemeral Vast.ai boxes through [taxameter](https://github.com/lhns/taxameter), a git dependency pinned by tag in `pyproject.toml`, configured in `taxameter.toml` (ADR 0021, 0029). Needs `uv`, `ssh`, `scp`, `tar` (Git Bash on Windows), the API key in `~/.config/vastai/vast_api_key` and the registered SSH key `~/.ssh/lrn_vast_ed25519`. State lives in `.vast/` (git-ignored). Rules for agents are in [AGENTS.md](AGENTS.md).

```bash
T="uv run taxameter"
$T balance                   # credit and runway
$T offers                    # ranked RTX 3090 offers; the last-good host comes first
$T create --auto             # rent the best offer under max_price_per_hour (--image overrides the image, --num-gpus 2|4 rents a multi-GPU box)
$T provision                 # push committed HEAD, toolchain, models (--models org/name:file.gguf,...), on-box autostop
$T run scripts/nightly.sh    # start a repo script detached on the box
$T ssh --timeout 540 -- 'for i in $(seq 100); do grep -q ^NIGHTLY_DONE /root/tetraeder/runs/vast/nightly.log && break; sleep 5; done'
$T fetch                     # pack, scp back, verify bench/results/*/DONE, unpack
$T destroy --yes             # refuses while results are unfetched
$T nightly                   # all of the above in one go: create, provision, run, wait, fetch, destroy
$T watchdog --dry-run        # report what the cost-control loop would do
```

### ggml build modes (ADR 0052)

By default `crates/ggml-sys` builds libllama, ggml and ggml-base as shared libraries and every backend as a loadable module (`ggml-cpu-<tag>` per x86 level, `ggml-cuda`, `ggml-vulkan`). `llama::init()` loads them, so one build runs at the best CPU level of each host, including the box and the AVX-less dev VM. `tetraeder info [--json]` shows which variant was loaded. `TETRAEDER_GGML_NATIVE=ON` builds the static native-ISA version instead: it is the default in `.cargo/config.toml` `[env]` (local dev and boxes, for build time), while CI sets `OFF` explicitly and builds the DL version (`OFF` is also the build.rs default without the config). Run-time env: `TETRAEDER_GGML_BACKEND_DIR` (module directory; default: next to the executable, else the build dir) and `TETRAEDER_GGML_CPU_VARIANT=<tag>` (load that CPU module only, for paired ISA runs). Binaries need the libraries and modules next to them (`target/<profile>/`, copied by `build.rs`). The patch queue must be applied first (`patches/apply.sh` or `patches/apply.ps1`; box scripts call `scripts/ensure_patches.sh`).

`scripts/nightly.sh` builds ggml CUDA only for the box GPU (`TETRAEDER_CUDA_ARCHS` from `nvidia-smi`, unless set), tests with `--release` to reuse that build, then runs `bench/spec/phase0.toml`. Both engines get the same explicit thread count (default `available_parallelism()`, quota aware), CPU variants run with the GPU hidden, and `env.json` records the container CPU quota.

Optional nightly stages take env vars, so launch with `$T run scripts/nightly.sh --env MOE_CHECK=1 --env GAP_PAIR=gate-tetraeder-cuda-dense` (add `--env BENCH_SPEC=phase2-baseline` for the Phase 2 baseline). `MOE_CHECK=1` runs the builder on the real Qwen3-30B-A3B (`crates/model/tests/real_moe.rs`, writes `builder-real-<HHMMSS>.json`); `GAP_PAIR=<phase0 pair id>` repeats that pair 10 times into `bench/results/<date>/gap/`. Neither changes the verdict. `nightly_wait_min` in `taxameter.toml` covers all stage limits.

Commit before `provision` or `run`: only committed HEAD (plus the `vendor/llama.cpp` checkout) is pushed. The box image (`ghcr.io/lhns/tetraeder-box:<tag from vendor/PIN>`) is built by `.github/workflows/box-image.yml`; to use a plain CUDA image instead pass `--image nvidia/cuda:12.8.1-devel-ubuntu24.04` and `provision` installs the missing tools. `taxameter.toml` holds the policy (price cap 0.90 USD/h per box all-in, two instances per project, 1.80 USD/h across all projects on the account, 3 h per box, `auto_replace` off; a 4-GPU box may pass `--max-price 1.20`, see `AGENTS.md`); a `.vast/config.json` still overrides it locally. Project work on the box lives in the hooks `scripts/box_deps.sh` and `scripts/box_verify.sh`; taxameter's own helper files go to `/root/.taxameter` on the box.

Windows Task Scheduler, run once from a prompt (the nightly time is an example). `conhost.exe --headless` keeps the tasks from opening a console window every run. Not registered by the repo; check `watchdog --dry-run` first. Tasks registered earlier with `uv run python scripts/vastctl.py nightly|watchdog --once` keep working: that file is now a shim for taxameter.

```bat
schtasks /Create /TN tetraeder-vast-nightly /SC DAILY /ST 03:00 /F /TR "conhost.exe --headless cmd /c cd /d <repo> && uv run taxameter nightly >> .vast\nightly.out 2>&1"
schtasks /Create /TN tetraeder-vast-watchdog /SC MINUTE /MO 5 /F /TR "conhost.exe --headless cmd /c cd /d <repo> && uv run taxameter watchdog --once >> .vast\watchdog.out 2>&1"
```

Weekly GPU smoke (about 0.25 USD): `scripts/gpu_smoke.sh` prints the commands (dry run), `--run` does the cycle; see `docs/sessions.md`, "GPU smoke". Not scheduled; scheduling it needs the user's go.

Tests: `uv run pytest -q tests` (offline; checks `taxameter.toml` against the repo).
