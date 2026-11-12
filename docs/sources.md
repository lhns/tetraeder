# External sources

Everything in this repo or its builds that comes from another project: where it comes from, the exact version, its license, and how to update it. `tests/test_sources.py` checks that every patch in `patches/series` and every key in `vendor/PIN` has a row here. Rust crates and Python packages are pinned by `Cargo.lock` and `uv.lock` and are not listed.

## Vendored code (part of the engine)

| What | Source | Version | License | Update |
|---|---|---|---|---|
| ggml and libllama (`vendor/llama.cpp`, submodule) | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | `vendor/PIN` `tag`, `commit` | MIT | Weekly rebase job (`rebase-weekly.yml`) tries upstream master with the patch queue; bump `tag` and `commit`, rerun `patches/apply.sh`, parity tests, rebuild the box image |

## Patch queue (`patches/`)

One row per patch in `patches/series`. A patch with code from another project names that project, the file and commit it was copied from, and its license; copied code keeps its copyright notice.

| Patch | Origin | Copied from (repo, path, commit) | License | ADR |
|---|---|---|---|---|
| `0001-msvc-avx512-vbmi-score.patch` | Own fix (one line; mirrors the non-MSVC branch of the same file; upstream commit not checked) | none | MIT (llama.cpp) | 0052 |
| `0002-iqk-avx512-q4k-q6k-decode.patch` | Copied from KTransformers (kernel by Iwan Kawrakow, ik_llama.cpp) plus our wrapper and hook | [kvcache-ai/ktransformers](https://github.com/kvcache-ai/ktransformers) `third_party/llamafile/iqk_mul_mat.inc` at `a02efe36b5802e7ca79a6b29146e01cf3b97d305` (v0.7.2), sha256 `9bef4f7bdd6c4ce4703a72f99944703d47a9bbd789349e1865310e868d148264`. The same Q4_K/Q6_K AVX-512 kernel is in [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) `ggml/src/iqk/iqk_gemm_kquants.cpp` at `ed868d928cf98f83589798b2b73416d42f3e6e2e` (MIT) | Apache-2.0 (Copyright 2024 Iwan Kawrakow, Copyright 2024 KVCache.AI; header kept in the file) | 0053 |

Updating `0002`: the copied file is frozen (ik moved Q4_K/Q6_K to `Q8_2_X4` activations on 2025-06-13, b7768e20, so there is no newer `Q8_K` kernel and no ik upstream path). To re-copy: clone the new `KT_VERSION` tag of kvcache-ai/ktransformers (shallow, read only); diff its `third_party/llamafile/iqk_mul_mat.inc` against `vendor/llama.cpp/ggml/src/ggml-cpu/iqk/iqk_mul_mat.inc` after `patches/apply.sh`; copy it over byte for byte (it has CRLF line endings); update the sha256 in `ggml/src/ggml-cpu/iqk/README` and here; regenerate the patch with `git -C vendor/llama.cpp -c core.autocrlf=false diff` (run `git add -N` on new files first, and diff only the files 0002 touches, since 0001 is staged); rerun `iqk_kernels` under SDE (CI job `avx512-sde`) and `expert_matmul` on a box.

## Baselines built into the box image or on the box

| What | Source | Version | License | Update |
|---|---|---|---|---|
| llama.cpp at the engine pin (`/opt/llama.cpp`) | ggml-org/llama.cpp | `vendor/PIN` `commit` | MIT | Follows the engine pin |
| llama.cpp baseline (`/opt/llama.cpp-baseline`, expert cache with CPU share, #30966) | ggml-org/llama.cpp | `vendor/PIN` `baseline_tag`, `baseline_commit` | MIT | Bump to a newer release when upstream improves the expert cache; `image_tag` changes with it |
| ik_llama.cpp (`/opt/ik_llama`) | [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) | `vendor/PIN` `ik_commit`, `ik_date` | MIT | Bump `ik_commit`; `image_tag` changes with it |
| KTransformers (kt-kernel, sglang-kt) | [kvcache-ai/ktransformers](https://github.com/kvcache-ai/ktransformers) | `KT_VERSION` in `scripts/ktransformers_build.sh` (0.7.2 = a02efe36) | Apache-2.0 | Change `KT_VERSION` |
| Colibri | [JustVugg/colibri](https://github.com/JustVugg/colibri) | `COLIBRI_COMMIT` in `scripts/colibri_build.sh` (v2.1.0 = 704a3f68) | see its repo | Change `COLIBRI_COMMIT` |
| sccache (image build cache) | [mozilla/sccache](https://github.com/mozilla/sccache) | `SCCACHE_VERSION` in `docker/box/Dockerfile` | Apache-2.0 | Change the ARG |
| Claude Code (client in the Phase 3b proxy test) | npm `@anthropic-ai/claude-code` | `CLAUDE_CODE_VERSION` in `scripts/claude_code_proxy.sh` | proprietary, used as a client only | Change the variable |

## Tooling

| What | Source | Version | Update |
|---|---|---|---|
| taxameter (Vast box management) | [lhns/taxameter](https://github.com/lhns/taxameter) | tag in `pyproject.toml` | Bump the tag, `uv sync` |
| OpenTelemetry Collector contrib (`otelcol-contrib`, box collector and CI config check, ADR 0059; Apache-2.0) | [open-telemetry/opentelemetry-collector-releases](https://github.com/open-telemetry/opentelemetry-collector-releases), release tarball `otelcol-contrib_<v>_linux_amd64.tar.gz`, installed to `/opt/otelcol` by `scripts/otelcol_install.sh` | `OTELCOL_VERSION` 0.116.1 and its `OTELCOL_SHA256` in `scripts/otelcol_install.sh` (from the release's `opentelemetry-collector-releases_otelcol-contrib_checksums.txt`); the compose image `otel/opentelemetry-collector-contrib` in `deploy/docker-compose.yml` uses the same version | Change both, with the sha256 of the new tarball; check the hostmetrics names `scripts/otlp_host.py` reads against the receiver docs of that version |

## Ideas only (no code taken)

| Project | What we took | Why no code |
|---|---|---|
| [Strata](https://github.com/Niko1221/Strata) | Expert tiering, low-bit KV, checkpoints (plan.md levers table) | Different engine; designs only |
| [Unsloth](https://github.com/unslothai/unsloth) | Tool-call repair idea | `studio/` is AGPL-3.0: never copy |
| KTransformers | Parallel faulting I/O as the explanation of its decode speed (ADR 0045 addenda) | Read only |
