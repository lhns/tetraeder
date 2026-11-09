# Brief: runtime CPU ISA dispatch (ADR 0052)

For one implementation worker. Read AGENTS.md (Rules 6 to 11) and `docs/adr/0052-runtime-cpu-isa-dispatch.md` first; the
ADR has the evidence and the decisions. Everything below is decided; local choices are yours, list them in the report.
Phase: Phase 2 exit follow-up (ADR 0045 hypothesis 3). Exit test moved: "same binary, best CPU variant loaded on every
CI runner and on the box".

## Decisions (do not reopen)

- Default build: `BUILD_SHARED_LIBS=ON`, `GGML_BACKEND_DL=ON`, `GGML_CPU_ALL_VARIANTS=ON`, `GGML_NATIVE=OFF`, on Linux and
  Windows (MSVC). `TETRAEDER_GGML_NATIVE=ON` keeps today's static native build unchanged; `OFF` or unset means DL.
- Rust uses only the registry API for anything in `ggml-cpu.h`, in both modes. Only the module load is `#[cfg(ggml_dl)]`.
- Modules load in `llama::init()` before `llama_backend_init`. Directory order: `TETRAEDER_GGML_BACKEND_DIR`, exe dir
  (if it holds a `ggml-cpu` module), build dir baked in via `cargo:rustc-env=TETRAEDER_GGML_MODULE_DIR=<OUT_DIR>/bin`.
  No CMake `GGML_BACKEND_DIR`.
- `TETRAEDER_GGML_CPU_VARIANT=<tag>`: load `ggml-cpu-<tag>` with `ggml_backend_load(path)` plus the GPU modules this
  build has (`ggml-cuda`, `ggml-vulkan` by file name), instead of `ggml_backend_load_all_from_path`. A null result
  (score 0 on this CPU, or missing file) is a load error naming the file.
- One patch: `patches/0001-msvc-avx512-vbmi-score.patch`, and the queue gets applied everywhere (below).

## Files and steps (tests first)

### 1. Failing tests

- `crates/ggml-sys/tests/cpu_dispatch.rs` (hardware: cpu, any x86; non-x86 skips with a reason):
  - `cpu_backend_reports_variant`: after `llama::init()`, `cpu::backend_info()` has `dispatch == Dl` in a DL build
    (`cfg!(ggml_dl)`), `module_dir` is `Some`, `missing` is empty and `extra` is empty. `extra` non-empty is the
    VBMI bug (or any wrong pick); `missing` non-empty means the best module was not loaded.
  - `cpu_backend_computes`: `Backend::cpu(2)` runs the existing small add graph (copy from `tests/smoke.rs`).
- Unit tests in `src/ggml/cpu.rs`: tier from a feature list, `missing`/`extra` set arithmetic, module dir resolution
  as a pure function `resolve_module_dir(env: Option<&Path>, exe_dir: Option<&Path>, built: &Path, has_cpu_module:
  impl Fn(&Path) -> bool) -> PathBuf`, module file names per OS (`ggml-cpu-<tag>.dll`, `libggml-cpu-<tag>.so`).
- `crates/tetraeder/tests/info.rs`: spawn `env!("CARGO_BIN_EXE_tetraeder") info --json`; parse; `cpu.dispatch` is
  `"dl"`. With `TETRAEDER_GGML_CPU_VARIANT=x64` the tier is `"sse"` (x86 only). With `=nonexistent` the exit code is
  non-zero and stderr names the file.
- Telemetry: a test like the existing metric tests in `crates/telemetry` that `cpu_backend(...)` records both gauges.
- `tests/test_alert_rules.py` covers the new alert once the metric literal exists in `crates/*/src`.

### 2. `crates/ggml-sys/build.rs`

- Keep every existing define, including any `GGML_CUDA_*` define another branch adds (track-j adds
  `GGML_CUDA_FA_QUANTS`-style pass-through; merge, do not drop). Keep the MSVC flags block.
- `let native = env TETRAEDER_GGML_NATIVE == "ON"`. Native: today's code path unchanged (static, `link_cuda`).
- DL: `BUILD_SHARED_LIBS=ON`, `GGML_BACKEND_DL=ON`, `GGML_CPU_ALL_VARIANTS=ON`, `GGML_NATIVE=OFF`.
  Link `dylib=llama`, `dylib=ggml`, `dylib=ggml-base` only (no `ggml-cpu`, `ggml-cuda`, `ggml-vulkan`, no `link_cuda`,
  no `link_vulkan`). Emit `rustc-link-search=native` for `<out>/lib`, `<out>/lib64` and `<out>/bin` (Windows DLLs and
  all modules install to `bin`; Cargo puts these dirs on `PATH`/`LD_LIBRARY_PATH` for `cargo test`/`run`).
- Emit `cargo::rustc-check-cfg=cfg(ggml_dl)` always and `cargo::rustc-cfg=ggml_dl` in DL mode; `rustc-env
  TETRAEDER_GGML_MODULE_DIR`; `rerun-if-env-changed` for the new env vars.
- Copy into the profile dir (`OUT_DIR` ancestors: `target[/<triple>]/<profile>/build/<pkg>/out`, so `.nth(3)`; comment
  why): Windows `<out>/bin/*.dll`; Linux `<out>/lib*/lib{llama,ggml,ggml-base}.so*` (copy the real files under every
  name, including the SONAME names) and `<out>/bin/libggml-*.so`. Copy only when content changed.
- Patch check: for each entry in `patches/series`, `git -C vendor/llama.cpp apply --reverse --check <patch>`; on failure
  panic with "run patches/apply.sh (or apply.ps1)". Skip silently when `git` is missing.
- `.cargo/config.toml`: `[target.'cfg(target_os = "linux")'] rustflags = ["-C", "link-arg=-Wl,-rpath,$ORIGIN"]`.

### 3. Bindings and registry calls

- `wrapper.h`: remove `#include "ggml-cpu.h"`. Fix what stops compiling.
- New `src/ggml/cpu.rs`: `CpuBackendInfo`, `backend_info()`, the tracked feature list (ADR), host detection with
  `std::arch::is_x86_feature_detected!` (AVX512 means f, cd, vl, dq, bw all present; if `amx-int8` is not stable on
  1.95, read `amx_int8` from `/proc/cpuinfo` on Linux, else treat as absent). Features of the loaded backend:
  `ggml_backend_get_features` from the CPU reg (`ggml_backend_dev_backend_reg(ggml_backend_dev_by_type(CPU))`), keep
  entries with value `"1"`. Same code in native mode (`dispatch = Static`, `module_dir = None`).
- A private `cpu_procs()` in `src/ggml/backend.rs` (OnceLock) resolving `ggml_backend_set_n_threads`,
  `ggml_backend_cpu_set_threadpool`, `ggml_threadpool_new`, `ggml_threadpool_free` (names: `ggml-cpu.cpp:659-691`).
  Types: `ggml_backend_set_n_threads_t` from bindgen; declare the other three as `unsafe extern "C" fn` types.
  Missing proc is `GgmlError::NoBackend`.
- `backend.rs:54` becomes `ggml_backend_dev_init(dev_by_type(CPU), null)`; `:62,:75,:76,:89,:23` use the procs.
  `llama/context.rs:29,41` use the threadpool procs. `tests/smoke.rs:8` uses `Backend::cpu` or the device route.
- `llama/mod.rs` `init()`: under `cfg(ggml_dl)` resolve dir, load, store the outcome in a `OnceLock<Result<..>>`;
  then `llama_backend_init`. Constructors (`Backend::cpu*`, `gpu*`, model load) return the stored load error.
  Log at info: dir, forced variant, tier, loaded registry names.
- Under `#[cfg(feature = "cuda")]` (and `vulkan`), after loading, `ggml_backend_reg_by_name("CUDA")` (`"Vulkan"`) null
  is a load error naming the module file and dir. A machine without a GPU still loads the CUDA module (it registers
  zero devices); a module that fails to dlopen does not. The `gpu_*` skip message distinguishes "CUDA module not
  loaded" (an error, not a skip) from "no GPU device".
- Grep `ggml_backend_dev_(count|by_type|by_name|get)` and `ggml_backend_reg_` across `crates/`: every public entry that
  reaches one must call `llama::init()` first (today `llama/devices.rs` does; `memory` does not).
- `crates/memory/src/ggml.rs`: call `tetraeder_ggml_sys::llama::init()` at the top of `available`, `bind`, `bind_cpu`.
- `crates/scheduler/tests/spec_builder.rs:67`: base the AVX2 check on `cpu::backend_info().isa` (the loaded kernels),
  not the host.
- If `track-l-kt-diag` is merged into main before you finish: `crates/ggml-sys/tests/expert_matmul.rs:151-156` uses
  `ggml_cpu_has_*`; switch it to `cpu::backend_info().isa`. Otherwise list it as a follow-up.

### 4. Patch 0001

- In `vendor/llama.cpp` (after `patches/apply.sh`): in `ggml/src/ggml-cpu/CMakeLists.txt`, MSVC AVX512 branch, the
  `if (GGML_AVX512_VBMI)` block also appends `GGML_AVX512_VBMI` to `ARCH_DEFINITIONS` (as the non-MSVC branch does at
  about line 387). `git -C vendor/llama.cpp diff > patches/0001-msvc-avx512-vbmi-score.patch`, add to `series`, rerun
  the apply script from a clean pin. The ADR is 0052; the backend test is `cpu_backend_reports_variant` (`extra` empty).
- Apply the queue: in `.github/workflows/ci.yml` every job runs `patches/apply.sh` (Linux) or
  `pwsh patches/apply.ps1` (Windows) right after checkout; box build scripts (`scripts/exit_run.sh` and whatever else
  runs `cargo build` on the box; grep for `cargo build\|cargo test` in `scripts/`) run `patches/apply.sh` first.
  Do not run box scripts (box HOLD).

### 5. CLI, bench, telemetry (Rule 6)

- `tetraeder info [--json]` (new subcommand in `crates/tetraeder/src/info.rs`): `cpu` block from `backend_info()`
  plus `llama_system_info` if present on main. Add a direct dependency on `tetraeder-ggml-sys` if needed.
- `crates/bench/src/env.rs`: `EnvInfo.cpu_backend: Option<serde_json::Value>`; `run.rs` fills it once per run by
  calling the spec's tetraeder binary with `info --json` (absent or failing: `None`, no error).
- `crates/telemetry/src/metrics.rs`: `tetraeder.cpu.variant` (u64 gauge, 1; attrs `tetraeder.cpu.tier`,
  `tetraeder.cpu.isa` comma-joined, `tetraeder.cpu.dispatch`) and `tetraeder.cpu.isa.missing` (u64 gauge, unit
  `{feature}`; attr `tetraeder.cpu.isa` = the missing list). Add the attribute keys to the `attr` consts. Record once
  at engine start in `tetraeder serve` and `tetraeder run`.
- `deploy/grafana/dashboards/overview.json`: panel "CPU backend" (table or stat: tier, isa, dispatch, missing).
- `deploy/alerts/engine.yml`: `TetraederCpuVariantBelowHost`, `max(tetraeder_cpu_isa_missing) > 0` for 5m, warning,
  description: "modules missing next to the binary or TETRAEDER_GGML_CPU_VARIANT set; check `tetraeder info`".

### 6. CI and image

- `ci.yml`: remove `TETRAEDER_GGML_NATIVE: "OFF"` (the comment's reason is gone). After `cargo build -p tetraeder`,
  run `target/debug/tetraeder info --json` (logs which variant each runner gets; include the registry names). New job
  `native-static` (ubuntu-latest) WITHOUT rust-cache: a native build restored on a runner with an older CPU is a
  SIGILL (the reason for the removed env line). It runs `TETRAEDER_GGML_NATIVE=ON cargo test -p tetraeder-ggml-sys`
  and `cargo build -p tetraeder`, so the comparison mode keeps building. CUDA and Vulkan compile-only jobs: same commands, plus a step that checks
  `target/debug/libggml-cuda.so` / `target/debug/ggml-cuda.dll` (Vulkan: `ggml-vulkan`) exists.
- `docker/box/Dockerfile`: drop `TETRAEDER_GGML_NATIVE=OFF` from `ENV` and fix the comment (baselines keep their own
  `-DGGML_NATIVE=OFF`). Do not trigger the image build. `scripts/exit_run.sh` `main-native` stages stay as they are.
- `xtask/src/main.rs:95` comment ("same ggml options as build.rs") is stale: the reference tools stay static; fix the
  comment, keep the tools static.
- `crates/ggml-sys/build.rs` doc comment and README "Vast boxes"/build section: the two modes and the env vars.

## Done criteria

- Rule 11 on the final state, locally (cargo lock, one job) and on CI for Linux and Windows: `cargo build --workspace`,
  `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --check`, pytest.
  Locally the VM has no AVX: `cpu_backend_reports_variant` must pass there with the sse42 or x64 module.
- CI green: test (ubuntu, windows), native-static, cuda compile-only (ubuntu, windows), vulkan compile-only.
  At most one `gh` call per 15 minutes; no `gh run watch`.
- `patches/apply.sh` applies 0001 from a clean pin; `build.rs` refuses an unpatched tree.
- No `ggml_backend_cpu_*`, `ggml_threadpool_new/free` or `ggml_cpu_has_*` in `crates/` outside proc-address name strings.
- Commit on your branch and push it; do not merge to main.

## Report

Variant and tier each CI runner loaded (from the `info` step), CI build time before and after per job, local clean
build time of ggml-sys before and after, any test that changed numerics, the follow-ups (expert_matmul on track-l,
baseline image rebuild with DL llama.cpp, box measurement: same binary with `TETRAEDER_GGML_CPU_VARIANT=haswell`
against auto, interleaved, plus `main-native`).
