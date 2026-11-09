# Brief: iqk AVX-512 VNNI decode kernels for Q4_K and Q6_K experts (ADR 0053, patch 0002)

For one implementation worker. Read AGENTS.md (Rules 2, 6 to 11), `patches/README.md`, `docs/sources.md` and
`docs/adr/0053-iqk-avx512-vnni-q4k-decode-kernel.md` first; the ADR has the evidence and the decisions. Everything
below is decided; local choices are yours, list them in the report. Phase: Phase 2 exit follow-up (ADR 0045
hypothesis 3). Exit test moved: "decode expert matmul on an AVX-512 VNNI host runs iqk kernels, at parity".

**Prerequisite:** the ADR 0052 implementation (track-o: DL build with CPU variants, `TETRAEDER_GGML_CPU_VARIANT`,
`cpu::backend_info()`, `patches/0001`, patch-apply in CI, box scripts and `build.rs`) is merged to main. Branch from
main after that merge. If it is not merged when you start, stop and report.

## Decisions (do not reopen)

- No new kernel code. Copy KTransformers v0.7.2 (a02efe36b5802e7ca79a6b29146e01cf3b97d305)
  `third_party/llamafile/iqk_mul_mat.inc` byte-identical (sha256
  `9bef4f7bdd6c4ce4703a72f99944703d47a9bbd789349e1865310e868d148264`). Get it with
  `git clone --depth 1 --branch v0.7.2 https://github.com/kvcache-ai/ktransformers` into a scratch dir; read only, never
  run its scripts. Its AVX-512 decode kernel (`mul_mat_qX_K_q8_K_AVX512_1`, `DequantizerQ4K`, `DequantizerQ6K`) is
  identical to ik_llama.cpp ed868d928cf98f83589798b2b73416d42f3e6e2e `ggml/src/iqk/iqk_gemm_kquants.cpp`.
- Only compiled where `__AVX512F__ && __AVX512VNNI__ && __AVX512VL__ && __AVX512BW__ && __AVX512DQ__`. Other
  variants are untouched.
- Hook in `mul_mat_id` only, one row per expert (`cne1 == 1`), types Q4_K and Q6_K only, after the tiled check.
- Use `iqk_mul_mat`, never the file's `iqk_mul_mat_moe` (broken in KT's copy, lines 1027-1040).
- `GGML_CPU_IQK_MM=0` disables; feature `IQK_MM` reports compiled-and-enabled.

## Files and steps (tests first)

### 1. Failing tests

- `crates/ggml-sys/tests/iqk_kernels.rs` (hardware label `avx512_vnni`). One gate for every test: skip with a printed
  reason when the loaded variant's `cpu::backend_info().tier` is below `avx512_vnni`; otherwise run, and assert
  `isa` contains `"IQK_MM"` unless `GGML_CPU_IQK_MM=0` is set. So on icelake the tests fail before the patch (feature
  missing) and pass after, and the env-off arm still runs the reference checks.
  - `q4k_mul_mat_id_decode_matches_reference` and `q6k_mul_mat_id_decode_matches_reference`: experts
    `[ne00, ne01, 16]` with ne00 in {256, 1536, 4096}, ne01 = 96, 4 of 16 experts used, one token; also the down shape
    (src1 `[ne00, 4, 1]`, one row per slot). Random weights through `quantize`, random activations. Reference in f64:
    dequantize weights with ggml-base `dequantize_row_q4_K`/`_q6_K`, activation through `quantize_row_q8_K_ref` and
    `dequantize_row_q8_K` (all `GGML_API` in ggml-base; declare them in the bindings, not in ggml-cpu). Assert
    |y - ref| <= 1e-5 * sum|w*x| + 1e-6 per output. Run the same graph with `use_ref` on (ggml's own vec_dot) and
    assert the same bound, and report max |iqk - vec_dot|.
  - `iqk_is_deterministic_across_threads`: 1, 3 and 8 threads give bitwise equal outputs.
  - `iqk_off_by_env`: run by CI with `GGML_CPU_IQK_MM=0`: asserts `IQK_MM` is absent; the reference tests above
    pass in the same run.
- `src/ggml/cpu.rs` unit test: `IQK_MM` is in the tracked list; host has it iff the five AVX-512 flags are detected.

### 2. Patch 0002 (in `vendor/llama.cpp`, then `git -C vendor/llama.cpp diff > patches/0002-iqk-avx512-q4k-q6k-decode.patch`)

Never commit changes inside `vendor/llama.cpp`; the patch file is the artifact. Run `patches/apply.sh` from a clean pin
to check it applies after 0001.

- `ggml/src/ggml-cpu/iqk/iqk_mul_mat.inc`: the copied file, unchanged (check the sha256).
- `ggml/src/ggml-cpu/iqk/README`: source repo, path, commit, sha256, license (Apache-2.0, Copyright 2024 Iwan
  Kawrakow, Copyright 2024 KVCache.AI), the identical ik commit, and "do not edit; re-copy".
- Forwarding headers in `ggml/src/ggml-cpu/iqk/`: `llama.cpp/ggml-impl.h`, `llama.cpp/ggml-quants.h`,
  `llama.cpp/ggml-common.h` (each one `#include` of the real header in `ggml/src/`), and `sgemm.h` (empty or the
  minimum the file needs; do not pull in llamafile's sgemm). If the file compiles without one of them, drop it.
- `ggml/src/ggml-cpu/iqk/iqk-mm.h` and `iqk-mm.cpp` (our code, a short header comment naming ADR 0053): the
  interface in the ADR. Linkage: `ggml-cpu.c` is C and the `.inc` defines its entry points with C++ linkage, so
  `iqk-mm.h` wraps its declarations in `#ifdef __cplusplus extern "C" {`. `iqk-mm.cpp` is the whole file under the
  five-flag `#if`; inside it `#define iqk_mul_mat iqk_mul_mat_impl` and
  `#define iqk_mul_mat_moe iqk_mul_mat_moe_unused` before `#include "iqk_mul_mat.inc"`, then
  `extern "C" bool ggml_iqk_mul_mat(...)` as a thin wrapper calling `iqk_mul_mat_impl`, and
  `extern "C" bool ggml_cpu_iqk_mm_enabled(void)` (`std::call_once`, `getenv("GGML_CPU_IQK_MM")`, on unless `"0"`),
  mirroring `tiled.cpp:572-581`. Compile warnings from the copied file: silence them per file in
  CMake (`set_source_files_properties`), never by editing it.
- `ggml/src/ggml-cpu/CMakeLists.txt`: add `iqk/iqk-mm.cpp` and `iqk/iqk-mm.h` to the source list next to `tiled`
  (line 44), and `ggml-cpu/iqk` to that target's include dirs so `"llama.cpp/ggml-impl.h"` and `"sgemm.h"` resolve.
- `ggml/src/ggml-cpu/ggml-cpu.c`, in `ggml_compute_forward_mul_mat_id`, after the tiled `continue` (line 1686) and
  after `src0_cur`, `wdata`, `row_size` are set:

  ```c
  #ifdef GGML_CPU_IQK_MM
          if (cne1 == 1 && !params->use_ref && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q6_K) &&
              vec_dot_type == GGML_TYPE_Q8_K && src1->type != vec_dot_type &&
              nb01 == ggml_row_size(type, ne00) && ggml_cpu_iqk_mm_enabled()) {
              const struct mmid_row_mapping rm = MMID_MATRIX_ROW(cur_a, 0);
              const char * src1_row = (const char *) wdata + ((rm.i1 % ne11) + rm.i2*ne11)*row_size;
              float * dst_row = (float *) ((char *) dst->data + rm.i1*nb1 + rm.i2*nb2);
              if (ggml_iqk_mul_mat(ne01, 1, ne00, type, src0_cur, nb01/ggml_type_size(type),
                                   GGML_TYPE_Q8_K, src1_row, row_size/ggml_type_size(GGML_TYPE_Q8_K),
                                   dst_row, ne0, ith, nth)) {
                  continue;
              }
          }
  #endif
  ```

  Strides are in blocks, not bytes (the file multiplies by `ggml_type_size`), like the `llamafile_sgemm` call at
  `:1393-1398`. The return value is the same on every thread for a given type, so no thread can skip an expert the
  others compute. Check `src1_row` against `ggml_compute_forward_mul_mat_id_one_chunk` (`:1519-1534`).
- `ggml/src/ggml-cpu/ggml-cpu.cpp` `ggml_backend_cpu_get_features` (`:541`): under `#ifdef GGML_CPU_IQK_MM`, push
  `{ "IQK_MM", "1" }` when `ggml_cpu_iqk_mm_enabled()`.
- Patch header (top of the `.patch` file, before the diff): one paragraph naming ADR 0053, the source, and any MSVC
  exclusion you had to make.
- MSVC: the file guards on `_M_X64` and should compile in MSVC's AVX-512 variants. If it does not and a small fix in
  our TU or CMake does not help, wrap `iqk-mm.cpp` and the hook in `&& !defined(_MSC_VER)`, say so in the patch
  header and the report. Windows must build and test green either way (Rule 11).

### 3. Rust side

- `src/ggml/cpu.rs` (from ADR 0052): add `IQK_MM` to the tracked features; host has it when
  `is_x86_feature_detected!` reports avx512f, avx512vnni, avx512vl, avx512bw, avx512dq. With `GGML_CPU_IQK_MM=0` set,
  do not count it as missing (the user turned it off on purpose).
- `Backend::set_cpu_use_ref(&self, bool)`: resolve `ggml_backend_cpu_set_use_ref` through the CPU registry's
  `get_proc_address`, cached like the other CPU functions in 0052. Safety comment on the call.
- `crates/ggml-sys/tests/expert_matmul.rs`: add a Q4_K_M variant (down projections Q6_K) next to the all-Q4_K run;
  replace the `ggml_cpu_has_*` block with `cpu::backend_info()` (`tier`, `isa`) if 0052 has not already; add
  `"iqk_mm"` (feature present) and `"tiled_force"` (`GGML_CPU_TILED_MM_FORCE`) to the `EXPERT_MATMUL` JSON.

### 4. CI

- `.github/workflows/ci.yml`, new Linux job `avx512-sde`: download Intel SDE (pinned version, sha256 check, cache the
  tarball), build the ggml-sys tests in release, run
  `sde64 -icx -- <iqk_kernels test binary> --nocapture` with `TETRAEDER_GGML_CPU_VARIANT=icelake`, once with
  `GGML_CPU_IQK_MM` unset and once `=0`, then the existing ggml-sys parity tests (`--test` list in the job) with the
  variant set. Fail if the iqk test skipped while the variant was icelake. If SDE cannot be fetched from a stable URL
  without a click-through, do not add the job; say so, and add the test to the box nightly list instead.
- `expert-bench.yml`: add `GGML_CPU_IQK_MM=0` and `=1` arms (runners with Ice Lake Xeons give a free pre-box number;
  print the CPU model).

### 5. Telemetry (Rule 6)

`IQK_MM` flows into `tetraeder.cpu.isa` and `missing` through ADR 0052's code. Dashboard "CPU backend" panel: make
`IQK_MM` visible (legend or a stat). Alert: `TetraederCpuVariantBelowHost` already covers it through `missing`; extend
its test in `tests/test_alert_rules.py` if the fixture lists features.

### 6. Sources and docs

- `docs/sources.md`, patch table, row for `` `0002-iqk-avx512-q4k-q6k-decode.patch` ``: Origin "copied from
  KTransformers (kernel by Iwan Kawrakow, ik_llama.cpp)"; Copied from: kvcache-ai/ktransformers
  `third_party/llamafile/iqk_mul_mat.inc` at a02efe36 (v0.7.2), sha256 above; identical kernel in
  ikawrakow/ik_llama.cpp `ggml/src/iqk/iqk_gemm_kquants.cpp` at ed868d92 (MIT); License Apache-2.0 (header kept);
  ADR 0053. Update procedure (in the row or a note under the table): clone the new `KT_VERSION` tag, diff its
  `third_party/llamafile/iqk_mul_mat.inc` against `vendor/llama.cpp/ggml/src/ggml-cpu/iqk/iqk_mul_mat.inc` after
  `patches/apply.sh`, copy it over, update sha256 in the README and here, regenerate the patch with
  `git -C vendor/llama.cpp diff`, rerun `iqk_kernels` under SDE and `expert_matmul` on a box. There is no ik upstream
  path: ik moved Q4_K/Q6_K to `Q8_2_X4` activations on 2025-06-13 (b7768e20).
- `patches/series`: add the patch after 0001.
- `docs/adr/0053`: status stays Proposed until the box run; add a "Measured" section only with box numbers.

## Done criteria

- `patches/apply.sh` applies 0001 and 0002 from a clean pin; `build.rs` patch check passes.
- Rule 11 set green locally and in CI on Linux and Windows: `cargo build --workspace`, `cargo test --workspace`,
  `cargo clippy --workspace -- -D warnings`, `cargo fmt --check`; `uv run pytest tests/test_sources.py
  tests/test_alert_rules.py` green.
- `avx512-sde` job green with the iqk tests actually run (not skipped), or the documented fallback.
- The copied file's sha256 in the vendor tree after apply equals the one above.
- No edit inside `iqk_mul_mat.inc`; no `unsafe` outside ggml-sys and memory, each block with a safety comment.

## Report

What you built, the patch size, MSVC outcome, max |iqk - vec_dot| and max error against the reference per shape,
SDE job status (or why not), the `expert-bench` numbers if a runner had AVX-512 VNNI, local choices, and the box
commands for the paired run (ADR 0053 decision 7) for when the hold lifts. Do not provision a box.
