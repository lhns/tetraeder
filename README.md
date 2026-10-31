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

`scripts/nightly.sh` builds ggml CUDA only for the box GPU (`TETRAEDER_CUDA_ARCHS` from `nvidia-smi`, unless set), tests with `--release` to reuse that build, then runs `bench/spec/phase0.toml`. Both engines get the same explicit thread count (default `available_parallelism()`, quota aware), CPU variants run with the GPU hidden, and `env.json` records the container CPU quota.

Commit before `provision` or `run`: only committed HEAD (plus the `vendor/llama.cpp` checkout) is pushed. The box image (`ghcr.io/lhns/tetraeder-box:<tag from vendor/PIN>`) is built by `.github/workflows/box-image.yml`; to use a plain CUDA image instead pass `--image nvidia/cuda:12.8.1-devel-ubuntu24.04` and `provision` installs the missing tools. `taxameter.toml` holds the policy (price cap 0.45 USD/h, one instance, `auto_replace` off); a `.vast/config.json` still overrides it locally. Project work on the box lives in the hooks `scripts/box_deps.sh` and `scripts/box_verify.sh`; taxameter's own helper files go to `/root/.taxameter` on the box.

Windows Task Scheduler, run once from a prompt (the nightly time is an example). Not registered by the repo; check `watchdog --dry-run` first. Tasks registered earlier with `uv run python scripts/vastctl.py nightly|watchdog --once` keep working: that file is now a shim for taxameter.

```bat
schtasks /Create /TN tetraeder-vast-nightly /SC DAILY /ST 03:00 /F /TR "cmd /c cd /d <repo> && uv run taxameter nightly >> .vast\nightly.out 2>&1"
schtasks /Create /TN tetraeder-vast-watchdog /SC MINUTE /MO 5 /F /TR "cmd /c cd /d <repo> && uv run taxameter watchdog --once >> .vast\watchdog.out 2>&1"
```

Tests: `uv run pytest -q tests` (offline; checks `taxameter.toml` against the repo).
