# 0029. GPU box tooling comes from the shared package taxameter

Status: Accepted

## Context

ADR 0021 put the Vast box tooling in this repo (`scripts/vastctl.py`, `vast_box.sh`, `idle_autostop.sh`, `cuda_env.sh`, `pack_results.sh`). LRN carried a diverging copy. Fixes had to be made twice.

## Decision

The tool moved into the shared package [taxameter](https://github.com/lhns/taxameter), a git dependency in the `dev` group of `pyproject.toml`, pinned by tag (`v0.1.0`). Project values live in `taxameter.toml` (offers, image tag from `vendor/PIN`, push paths, results, policy). Rust dependencies and the cargo check run as box hooks: `scripts/box_deps.sh` (provision) and `scripts/box_verify.sh` (verify). `nightly.sh` sources `cuda_env.sh` from `/root/.taxameter` on the box. `scripts/vastctl.py` stays as a shim so the registered Task Scheduler commands keep working. The `.vast/` state is unchanged and carries over. The repo keeps tests only for its own config (`tests/test_taxameter_config.py`).

## Consequences

- Upgrading taxameter means changing the tag in `pyproject.toml` and running `uv lock`.
- The generic tests, helper scripts and the on-box watchdog now belong to taxameter; a box provisioned by the old tool keeps its old watchdog until it is destroyed.
- CI needs network access to GitHub to resolve the dependency.
- ADR 0021 stays valid for the workflow (ephemeral boxes, caps, nightly); only the tooling moved.
