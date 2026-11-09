# 0052. Runtime CPU ISA dispatch: ggml backends as loadable modules, one build for every x86 CPU

Status: Proposed (Phase 2 exit follow-up, ADR 0045 hypothesis 3). Builds on ADR 0011, 0036, 0039.

## Context

- `crates/ggml-sys/build.rs:35` builds ggml and libllama static and links `ggml-cpu`, `ggml-base` (and `ggml-cuda`) into every binary. The CPU ISA is fixed at build time: native on a dev machine, the AVX2 baseline with `TETRAEDER_GGML_NATIVE=OFF` (CI `ci.yml:16`, box image `docker/box/Dockerfile:23`). On the AVX-512 Xeon boxes our CPU experts run AVX2 code; KTransformers picks VNNI kernels at runtime (ADR 0045 addendum).
- ggml's mechanism: `GGML_BACKEND_DL=ON` builds every backend as a MODULE (`ggml/src/CMakeLists.txt:381-397`), `GGML_CPU_ALL_VARIANTS=ON` builds one `ggml-cpu-<tag>` module per x86 level (`:489-520`: x64, sse42, sandybridge, haswell, skylakex, cannonlake, cascadelake, icelake, alderlake; with GCC/Clang also ivybridge, piledriver, cooperlake, zen4, sapphirerapids). `ggml_backend_load_best` (`ggml-backend-reg.cpp:480`) scores each module with its own cpuid check and loads the best one.
- Constraints found in the pin (b11976):
  - DL requires `BUILD_SHARED_LIBS=ON` (`src/CMakeLists.txt:188`): modules link `ggml-base` as a shared library (`:399`). `BUILD_SHARED_LIBS` is global, so `ggml`, `ggml-base` and `llama` become shared libraries too.
  - `GGML_NATIVE=ON` is rejected with DL on x86 (`ggml-cpu/CMakeLists.txt:413`). A native build stays a static build.
  - CUDA and Vulkan become modules too (`ggml_add_backend`, `:428-439`; `GGML_USE_CUDA` is no longer defined). The Rust binary then links no CUDA library; `libggml-cuda.so` / `ggml-cuda.dll` links cudart and cuBLAS itself.
  - libllama is DL-ready: `llama_backend_init` calls `ggml_backend_load_all()` only when no backend is registered (`src/llama.cpp:133`); its CPU-specific calls go through `ggml_backend_reg_get_proc_address` (`llama-context.cpp:2762`, `llama-model-loader.cpp:1433`, `llama.cpp:142`). `ggml_backend_cpu_buffer_type` lives in ggml-base.
  - MSVC builds 9 variants (no BF16, no AMX; configure and a `ggml-cpu-skylakex` build checked locally; `-fno-lto` only warns, D9002). Upstream ships Windows CPU builds with clang-cl, so MSVC with all variants is not tested upstream, and it has a scoring bug: in the MSVC branch (`ggml-cpu/CMakeLists.txt:301-305`) VBMI variants define `__AVX512VBMI__` but not `GGML_AVX512_VBMI`, so `cpu-feats.cpp` never checks VBMI. On a Skylake-X host `ggml-cpu-cannonlake` ties with `skylakex`, comes first in directory order and is loaded: illegal instruction on the first VBMI kernel.
- Rust calls into ggml-cpu directly at `crates/ggml-sys/src/ggml/backend.rs:23,54,62,75,76,89` (`ggml_backend_cpu_init`, `ggml_threadpool_new/free`, `ggml_backend_cpu_set_n_threads`, `ggml_backend_cpu_set_threadpool`), `src/llama/context.rs:29,41` (threadpool for `llama_attach_threadpool`) and `tests/smoke.rs:8`. Branch `track-l-kt-diag` adds `ggml_cpu_has_*` calls in `tests/expert_matmul.rs:151-156`. None of these link when ggml-cpu is a module. All have a registry route: the CPU registry's `get_proc_address` (`ggml-cpu.cpp:659-691`) returns `ggml_backend_set_n_threads`, `ggml_backend_cpu_set_threadpool`, `ggml_threadpool_new`, `ggml_threadpool_free`, `ggml_backend_get_features`; `ggml_backend_cpu_init` is `ggml_backend_dev_init(ggml_backend_dev_by_type(CPU), NULL)`. No crate calls a `ggml_backend_cuda_*` or `ggml_backend_vk_*` symbol; GPU devices already go through the registry (`backend.rs:95-143`, `llama/devices.rs`, `memory/src/ggml.rs`).
- `memory/src/ggml.rs:35-58` queries the registry without calling `llama::init()`. Static registration hides that today; with modules the registry is empty until something loads them.
- The patch queue is empty (`patches/series` is 0 bytes), and nothing outside `rebase-weekly.yml` applies it: not CI, not the box scripts, not `build.rs`.
- **What dispatch buys for the builder.** At this pin the x86 decode-path Q4_K and Q6_K dot products (`ggml-cpu/arch/x86/quants.c:2102-2280`, `:2490-2724`) and `quantize_row_q8_K` (`:505`) are AVX2 only: no `_mm512`, no `dpbusd`. ggml does have AVX-512 VNNI K-quant kernels (`ggml-cpu/tiled/tiled-kernel.cpp:83-207`, `_mm512_dpbusd_epi32`), but `mul_mat_id` uses them only when an expert gets 8 or more rows (`tiled.cpp:1202`), so they serve prefill, not decode. VNNI is otherwise used only through `mul_sum_us8_pairs_float` (Q4_0, Q8_0 and similar), AVX-512 in a few helpers, the repack GEMMs and llamafile sgemm (F32, F16, BF16, Q8_0, Q4_0, Q5_0, IQ4_NL). The repack Q4_K GEMV (`repack.cpp:1464`) has no `_mm512` either, and the builder does not repack (ADR 0039) nor use the AMX buffer type. So for the Qwen3 Q4_K_M experts an AVX-512 variant changes only compiler code generation (EVEX encodings, 32 vector registers, auto-vectorized float ops). So dispatch alone cannot deliver ADR 0045 hypothesis 3 for the builder. Whether KTransformers' 1.34x comes from its iqk VNNI kernels, parallel faulting I/O or the page cache is still open (none of the 0045 addendum stages have run).

## Decision

1. **Default build: `GGML_BACKEND_DL=ON`, `GGML_CPU_ALL_VARIANTS=ON`, `GGML_NATIVE=OFF`, `BUILD_SHARED_LIBS=ON`** on Linux and Windows (MSVC). CUDA and Vulkan, when their features are on, are modules too. Rust links `llama`, `ggml`, `ggml-base` as dylibs and nothing else from ggml.
2. **`TETRAEDER_GGML_NATIVE=ON` keeps today's static native build** (the box `main-native` stage, fast local builds). `OFF` or unset means the DL build. Rust code is the same in both modes: it uses only the registry API; only the load step is behind `cfg(ggml_dl)`, which `build.rs` sets.
3. **No direct ggml-cpu symbols.** `wrapper.h` drops `ggml-cpu.h`. CPU-only functions are resolved once through `ggml_backend_reg_get_proc_address` on the CPU registry and cached.
4. **Explicit load before libllama.** `ggml_sys::llama::init()` (the one choke point) loads modules, then calls `llama_backend_init`, which then skips its own load. Module directory, first match wins: `TETRAEDER_GGML_BACKEND_DIR`; the executable's directory if it holds a `ggml-cpu` module; the build's install dir (`<OUT_DIR>/bin`, baked in with `cargo:rustc-env`). No compile-time `GGML_BACKEND_DIR` (an absolute build path would leak into images). `TETRAEDER_GGML_CPU_VARIANT=<tag>` loads that one CPU module (plus the GPU modules of the build) instead of the best one: for paired ISA measurements and to pin golden runs. After loading, a missing CPU device is an error that names the directory searched (`load_best` is silent in Release, `ggml-backend-reg.cpp:572`). So is a missing `CUDA` (or `Vulkan`) registry in a build with that feature: a module that fails to dlopen (no driver, no cudart) would otherwise turn the engine CPU-only and the `gpu_*` tests into skips.
5. **Runtime files.** `build.rs` copies the shared libraries and modules into `target/<profile>/` so binaries run directly (pytest, box scripts call `target/release/tetraeder`). Linux binaries get `-Wl,-rpath,$ORIGIN` from `.cargo/config.toml` (a dependency's `cargo:rustc-link-arg` does not reach other packages' binaries). `cargo test` and `cargo run` find the libraries through the `-L native` dirs inside `target/`, which Cargo puts on `PATH` / `LD_LIBRARY_PATH`; test binaries in `deps/` find modules through the baked dir.
6. **Patch 0001 (Rule 8):** `patches/0001-msvc-avx512-vbmi-score.patch` adds `GGML_AVX512_VBMI` to the MSVC branch of `ggml-cpu/CMakeLists.txt`. With the first patch the queue must actually be applied: CI jobs run `patches/apply.sh` / `apply.ps1` after checkout, the box scripts run `apply.sh` before building, and `build.rs` fails with a hint when a patch in `series` is not applied (`git apply --reverse --check`).
7. **Telemetry (Rule 6).** At engine start: gauge `tetraeder.cpu.variant` (value 1, attributes `tetraeder.cpu.tier`, `tetraeder.cpu.isa`, `tetraeder.cpu.dispatch`) and gauge `tetraeder.cpu.isa.missing` (tracked ISA features the host has and the loaded backend lacks, among those this build can provide). Dashboard panel "CPU backend"; alert `TetraederCpuVariantBelowHost` when missing > 0 for 5 minutes. `tetraeder info --json` prints the same, and the bench harness records it as `env.cpu_backend` in every result.
8. **Box image:** drop `TETRAEDER_GGML_NATIVE=OFF` from the image env (the default DL build is portable by construction). The `main-native` stage stays as the comparison arm. Baselines stay as they are in this change (see Consequences).

## Interfaces

```rust
// crates/ggml-sys/src/ggml/cpu.rs, re-exported as tetraeder_ggml_sys::cpu
pub enum Dispatch { Dl, Static }                // serde: "dl" | "static"
pub struct CpuBackendInfo {
    pub dispatch: Dispatch,
    pub module_dir: Option<PathBuf>,             // DL only: where the modules came from
    pub forced: Option<String>,                  // TETRAEDER_GGML_CPU_VARIANT
    pub isa: Vec<String>,                        // ggml_backend_get_features of the CPU reg, value "1" only
    pub tier: &'static str,                      // amx_int8|avx512_bf16|avx512_vnni|avx512|avx_vnni|avx2|avx|sse
    pub host_isa: Vec<String>,                   // tracked features the host has (is_x86_feature_detected)
    pub missing: Vec<String>,                    // host_isa - isa, limited to features this build can provide
    pub extra: Vec<String>,                      // isa - host_isa: must be empty (a wrong variant was loaded)
}
pub fn backend_info() -> &'static CpuBackendInfo; // after llama::init()
// Tracked: AVX, AVX2, FMA, F16C, BMI2, AVX_VNNI, AVX512, AVX512_VBMI, AVX512_VNNI, AVX512_BF16, AMX_INT8
// (MSVC builds: without AVX512_BF16, AMX_INT8).
```

`Backend::cpu*`, `set_cpu_threads` and `llama::ThreadPool` keep their signatures. `llama::init()` keeps its signature; load errors surface as `GgmlError::NoBackend` from the constructors. Telemetry: `Metrics::cpu_backend(tier, isa, dispatch, missing)`. Env: `TETRAEDER_GGML_NATIVE` (ON: static native), `TETRAEDER_GGML_BACKEND_DIR`, `TETRAEDER_GGML_CPU_VARIANT`.

## Consequences

- One build runs on every x86 host at its best ggml level, including the local VM without AVX (sse42 or x64 module) and the boxes (skylakex to sapphirerapids). The VM no longer needs a native build to run tests.
- **Expected gain for the Q4_K expert path is small**, because ggml has no AVX-512 or VNNI Q4_K/Q6_K kernel at the pin. Dispatch is the delivery vehicle: any AVX-512 VNNI Q4_K kernel (a patch-queue item, or iqk-style kernels) reaches the boxes only through it. The measurement that settles the gain is a paired run of one binary with `TETRAEDER_GGML_CPU_VARIANT=haswell` against the auto-picked variant (exit regime decode and `expert_matmul`), interleaved, on one box; `main-native` is the third arm.
- Binaries are no longer self-contained: they need `libllama`, `libggml`, `libggml-base` and the modules next to them (or in `TETRAEDER_GGML_BACKEND_DIR`). Running with `RUSTFLAGS` set in the environment drops the `.cargo/config.toml` rpath on Linux; the binary then fails at start with a missing `libllama.so`.
- Build time: ggml-cpu compiles 9 (MSVC) or 14 (GCC) times. Cached in CI by rust-cache and on the box by sccache; a clean build on the local VM grows by several minutes per variant. `TETRAEDER_GGML_NATIVE=ON` stays the quick local option.
- Numerics follow the loaded variant. Within one process every backend uses one variant, so the bitwise checks of ADR 0034/0035 hold. Golden files and cross-machine logit comparisons depend on the variant: record `tier` with them and pin `TETRAEDER_GGML_CPU_VARIANT` when recording.
- Baselines: the image's llama.cpp (pin and baseline commits) and ik_llama.cpp stay static AVX2. Once our engine runs AVX-512 variants, a fair llama.cpp pairing needs llama.cpp built the way upstream ships it (`GGML_BACKEND_DL=ON`, `GGML_CPU_ALL_VARIANTS=ON`); that is an image rebuild and a separate change. ik_llama.cpp has its own ISA handling; it stays as is.
- The first patch makes patch application part of every build path (CI, box, `build.rs` check). The weekly rebase job already tests the queue on upstream master.

## Addendum 2026-11-09: build mode per environment

- **Boxes and local dev build native** (`TETRAEDER_GGML_NATIVE=ON`: one CPU variant, static). A fresh box with the DL default would compile 14 CPU variants plus CUDA, an estimated 30 to 40 paid minutes; the native build is the pre-dispatch 9 to 11 minutes. `.cargo/config.toml` `[env]` sets it for local builds and `docker/box/Dockerfile` for the image; `scripts/exit_run.sh` defaults it too. A process env var wins over `[env]`.
- **CI and release packaging build DL.** `ci.yml` sets `TETRAEDER_GGML_NATIVE=OFF` explicitly (the native-static job sets `ON`), so the config default never reaches it. The multi-variant build is exercised on every push; the box and the VM run the code paths of one variant of it.
- The `main-native` and `main-native-repeat` exit stages duplicated `main` and are removed. `ggml_system_info` in every exit result records the ISA of the native build. The ADR 0045 hypothesis 3 (native against portable AVX2) is answered by the expert microbench (`expert-bench.yml`), not by a box stage; a DL stage on the box would cost the 14-variant build again and is not worth it.
- ADR 0053's AVX-512 VNNI kernel is compiled into native builds on AVX-512 hosts (the box CPUs), so the box runs it without a variant selection.
- CI compiles ggml through sccache (GitHub Actions cache backend, as CMake compiler launcher) because rust-cache does not keep workspace crate outputs; it is not used for the native-static job, whose `-march=native` objects must not move between runner CPUs.
