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

All GPU work runs on ephemeral Vast.ai boxes through `scripts/vastctl.py` (ADR 0021). Needs `uv`, `ssh`, `scp`, `tar` (Git Bash on Windows), the API key in `~/.config/vastai/vast_api_key` and the registered SSH key `~/.ssh/lrn_vast_ed25519`. State lives in `.vast/` (git-ignored). Rules for agents are in [AGENTS.md](AGENTS.md).

```bash
V="uv run python scripts/vastctl.py"
$V balance                   # credit and runway
$V offers                    # ranked RTX 3090 offers; the last-good host comes first
$V create --auto             # rent the best offer under max_price_per_hour (add --image to override the image)
$V provision                 # push committed HEAD, toolchain, models (--models org/name:file.gguf,...), on-box autostop
$V run scripts/nightly.sh    # start a repo script detached on the box
$V ssh --timeout 540 -- 'for i in $(seq 100); do grep -q ^NIGHTLY_DONE /root/tetraeder/runs/vast/nightly.log && break; sleep 5; done'
$V fetch                     # pack, scp back, verify bench/results/*/DONE, unpack
$V destroy --yes             # refuses while results are unfetched
$V nightly                   # all of the above in one go: create, provision, run, wait, fetch, destroy
$V watchdog --dry-run        # report what the cost-control loop would do
```

Commit before `provision` or `run`: only committed HEAD (plus the `vendor/llama.cpp` checkout) is pushed. The box image (`ghcr.io/lhns/tetraeder-box:<tag from vendor/PIN>`) is built by `.github/workflows/box-image.yml`; to use a plain CUDA image instead pass `--image nvidia/cuda:12.8.1-devel-ubuntu24.04` and `provision` installs the missing tools. `.vast/config.json` holds the thresholds (price cap 0.45 USD/h, one instance, `auto_replace` off).

Windows Task Scheduler, run once from a prompt (the nightly time is an example). Not registered by the repo; check `watchdog --dry-run` first.

```bat
schtasks /Create /TN tetraeder-vast-nightly /SC DAILY /ST 03:00 /F /TR "cmd /c cd /d <repo> && uv run python scripts/vastctl.py nightly >> .vast\nightly.out 2>&1"
schtasks /Create /TN tetraeder-vast-watchdog /SC MINUTE /MO 5 /F /TR "cmd /c cd /d <repo> && uv run python scripts/vastctl.py watchdog --once >> .vast\watchdog.out 2>&1"
```

Tests: `uv run pytest tests/test_vastctl.py` (offline, mocked API and ssh).
