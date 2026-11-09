# Tetraeder

An LLM inference engine in Rust on a patched ggml: one placement plan expresses every parallel mode and memory tier, a measured-cost planner chooses it for the hardware at hand, and built-in OpenTelemetry observability finds the bottleneck and proposes the fix.

You set limits such as a target tok/s, a quality floor and which devices it may use; the planner picks formats per layer and expert (mixed quantization, measured per model) and placement together. See [Quantization and quality](docs/plan.md#quantization-and-quality).

- Design, phases and exit tests: [docs/plan.md](docs/plan.md)
- Decisions: [docs/adr/](docs/adr/README.md)
- Rules for contributors and agents: [AGENTS.md](AGENTS.md)

Status: Phase 0 (foundations). Workspace stubs and Vast tooling only; no engine yet.

Targets Windows and Linux. Build and test: `cargo build --workspace && cargo test --workspace`.

## Name

A tetrahedron (German: Tetraeder) has four faces, one for each of the four parallel modes the engine combines: tensor, pipeline, expert and data parallel. The core idea of the project is that all four are shapes of one placement plan, which the planner chooses together. Four also fits the four memory tiers (VRAM, RAM, NVMe, remote node) and the four milestones.

## Serving and Claude Code

`tetraeder serve` runs the OpenAI API (`/v1/chat/completions`, `/v1/completions`) and the Anthropic API (`/v1/messages`, `/v1/messages/count_tokens`) on one port. It prints `listening on http://host:port` to stderr when it is ready.

```
# dense model through libllama
tetraeder serve -m model.gguf --alias qwen --ctx 65536 --port 8080
# Qwen3 MoE through the builder on GPU 0, experts streamed from disk if slots are short
tetraeder serve -m Qwen3-30B-A3B-Q4_K_M.gguf --engine moe --alias qwen --ctx 65536 --expert-slots 64
# scripted engine, no model (SDK tests)
tetraeder serve --fake
```

One request generates at a time; others queue (`--max-queued`). The last conversation's KV cache is kept, so an agent that resends its whole history only pays for the new tokens. Sequences that a different prompt evicts are kept in host RAM (`--cache-ram <MiB>`, default 8192, 0 off) and restored when their prefix comes back.

Claude Code:

```
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_AUTH_TOKEN=local ANTHROPIC_MODEL=qwen ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen claude
```

Claude Code also asks for its own `claude-*` model ids; `--accept-any-model` (or `--default-model <id>`) maps unknown ids to the served model. It sends 15K to 30K tokens of system prompt and tools on each turn, so start with `--ctx 65536` or more. `--request-log <file>` appends one JSON line per request (prompt, response, token counts, timings) for replay; it contains your code, so keep it local.

## Vast boxes

All GPU work runs on ephemeral Vast.ai boxes through [taxameter](https://github.com/lhns/taxameter), a git dependency pinned by tag in `pyproject.toml`, configured in `taxameter.toml` (ADR 0021, 0029). Needs `uv`, `ssh`, `scp`, `tar` (Git Bash on Windows), the API key in `~/.config/vastai/vast_api_key` and the registered SSH key `~/.ssh/lrn_vast_ed25519`. State lives in `.vast/` (git-ignored). Rules for agents are in [AGENTS.md](AGENTS.md).

```bash
T="uv run taxameter"
$T balance                   # credit and runway
$T offers                    # ranked RTX 3090 offers; the last-good host comes first
$T create --auto             # rent the best offer under max_price_per_hour (add --image to override the image)
$T provision                 # push committed HEAD, toolchain, models (--models org/name:file.gguf,...), on-box autostop
$T run scripts/nightly.sh    # start a repo script detached on the box
$T ssh --timeout 540 -- 'for i in $(seq 100); do grep -q ^NIGHTLY_DONE /root/tetraeder/runs/vast/nightly.log && break; sleep 5; done'
$T fetch                     # pack, scp back, verify bench/results/*/DONE, unpack
$T destroy --yes             # refuses while results are unfetched
$T nightly                   # all of the above in one go: create, provision, run, wait, fetch, destroy
$T watchdog --dry-run        # report what the cost-control loop would do
```

### ggml build modes (ADR 0052)

By default `crates/ggml-sys` builds libllama, ggml and ggml-base as shared libraries and every backend as a loadable module (`ggml-cpu-<tag>` per x86 level, `ggml-cuda`, `ggml-vulkan`). `llama::init()` loads them, so one build runs at the best CPU level of each host, including the box and the AVX-less dev VM. `tetraeder info [--json]` shows which variant was loaded. `TETRAEDER_GGML_NATIVE=ON` builds the static native-ISA version instead (comparison arm, fast local builds); unset or `OFF` is the default. Run-time env: `TETRAEDER_GGML_BACKEND_DIR` (module directory; default: next to the executable, else the build dir) and `TETRAEDER_GGML_CPU_VARIANT=<tag>` (load that CPU module only, for paired ISA runs). Binaries need the libraries and modules next to them (`target/<profile>/`, copied by `build.rs`). The patch queue must be applied first (`patches/apply.sh` or `patches/apply.ps1`; box scripts call `scripts/ensure_patches.sh`).

`scripts/nightly.sh` builds ggml CUDA only for the box GPU (`TETRAEDER_CUDA_ARCHS` from `nvidia-smi`, unless set), tests with `--release` to reuse that build, then runs `bench/spec/phase0.toml`. Both engines get the same explicit thread count (default `available_parallelism()`, quota aware), CPU variants run with the GPU hidden, and `env.json` records the container CPU quota.

Optional nightly stages take env vars, so launch with `$T run scripts/nightly.sh --env MOE_CHECK=1 --env GAP_PAIR=gate-tetraeder-cuda-dense` (add `--env BENCH_SPEC=phase2-baseline` for the Phase 2 baseline). `MOE_CHECK=1` runs the builder on the real Qwen3-30B-A3B (`crates/model/tests/real_moe.rs`, writes `builder-real-<HHMMSS>.json`); `GAP_PAIR=<phase0 pair id>` repeats that pair 10 times into `bench/results/<date>/gap/`. Neither changes the verdict. `nightly_wait_min` in `taxameter.toml` covers all stage limits.

Commit before `provision` or `run`: only committed HEAD (plus the `vendor/llama.cpp` checkout) is pushed. The box image (`ghcr.io/lhns/tetraeder-box:<tag from vendor/PIN>`) is built by `.github/workflows/box-image.yml`; to use a plain CUDA image instead pass `--image nvidia/cuda:12.8.1-devel-ubuntu24.04` and `provision` installs the missing tools. `taxameter.toml` holds the policy (price cap 0.45 USD/h, one instance per project, 1.0 USD/h across all projects on the account, 3 h per box, `auto_replace` off); a `.vast/config.json` still overrides it locally. Project work on the box lives in the hooks `scripts/box_deps.sh` and `scripts/box_verify.sh`; taxameter's own helper files go to `/root/.taxameter` on the box.

Windows Task Scheduler, run once from a prompt (the nightly time is an example). Not registered by the repo; check `watchdog --dry-run` first. Tasks registered earlier with `uv run python scripts/vastctl.py nightly|watchdog --once` keep working: that file is now a shim for taxameter.

```bat
schtasks /Create /TN tetraeder-vast-nightly /SC DAILY /ST 03:00 /F /TR "cmd /c cd /d <repo> && uv run taxameter nightly >> .vast\nightly.out 2>&1"
schtasks /Create /TN tetraeder-vast-watchdog /SC MINUTE /MO 5 /F /TR "cmd /c cd /d <repo> && uv run taxameter watchdog --once >> .vast\watchdog.out 2>&1"
```

Tests: `uv run pytest -q tests` (offline; checks `taxameter.toml` against the repo).
