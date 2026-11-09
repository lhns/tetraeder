# 0053. iqk AVX-512 VNNI kernels for Q4_K and Q6_K expert decode (patch 0002)

Status: Proposed (Phase 2 exit follow-up, ADR 0045 hypothesis 3). Builds on ADR 0011 (pin, patch queue), 0039 (no repack), 0052 (runtime ISA dispatch). Brief: `docs/q4k-kernel-brief.md`.

## Context

The builder's CPU experts (Qwen3-235B Q4_K_M, one token, `mul_mat_id` over 8 of 128 experts) run ggml's `ggml_vec_dot_q4_K_q8_K` and `ggml_vec_dot_q6_K_q8_K` one row at a time (`ggml-cpu.c:1483-1544`), which are AVX2 only at our pin (`arch/x86/quants.c:2102`, `:2490`). KTransformers decodes 1.34x faster on the 26-core Ice Lake box C (ADR 0045) with ik's iqk kernels on AVX-512 VNNI. The user asked whether this is already solved upstream.

**Mainline has no AVX-512 decode kernel for K-quants, at the pin or on master.** Checked against master a0ad695 (2026-11-09, after the b12211 baseline):
- `arch/x86/quants.c` and `arch/x86/repack.cpp` are byte-identical to the pin. The Q4_K/Q6_K dot products and `quantize_row_q8_K` (`= quantize_row_q8_K_ref`) are unchanged.
- Mainline does have AVX-512 VNNI K-quant code, but only for batches. `ggml-cpu/tiled/` (identical at pin and master) has `_mm512_dpbusd_epi32` microkernels for Q2_K to Q6_K and several IQ types (`tiled-kernel.cpp:83-207`), hooked into `mul_mat_id` per expert (`ggml-cpu.c:1684`). They run only when an expert gets 8 or more rows (`tiled.cpp:1202`, `ggml_tiled_min_batch`). They unpack a 256-row window of src0 to one byte per weight and run 16x16 microtiles, which is GEMM work and does not fit a 1-column GEMV. So prefill gets VNNI upstream; decode does not. ADR 0052 says the pin has no `dpbusd` for Q4_K; that is true for decode only.
- `llamafile/sgemm.cpp` has no K-quant types. `amx/mmq.cpp` has Q4_K/Q6_K kernels behind the AMX buffer type, which repacks and needs AMX hardware (Sapphire Rapids); the builder does neither.
- So a pin bump or backport brings nothing for decode.

**Why ik's kernels are not in mainline.** ik_llama.cpp forked in June 2024 and last synced with mainline in August 2024 (its README). iqk_mul_mat was first published in Mozilla's llamafile (0.8.8, Apache-2.0, Copyright 2024 Iwan Kawrakow), then grew inside ik_llama.cpp (MIT). Its shape differs from ggml's: one call computes a block of rows times up to 8 (16 on AVX-512) activation columns with its own dequantizer per type, instead of ggml's one-row `vec_dot`. Since 2025-06-13 (ik b7768e20) its Q4_K/Q5_K kernels take activations in ik's `GGML_TYPE_Q8_2_X4` (`ggml.c:965`, `iqk_gemm_kquants.cpp:3177`), a type mainline lacks, and the code lists more than 50 ik-only types (enum up to 399, `ggml.h:505-511`). Ports of ik work to mainline have stalled before (PR #19726, IQ*_K quants, closed unmerged 2026-02); that is a signal about ports in general, not about these kernels. The low-hanging fruit was picked in a fork.

**Where the decode kernel lives, and in which form it fits.** The one-column AVX-512 kernel `mul_mat_qX_K_q8_K_AVX512_1` with `DequantizerQ4K` and `DequantizerQ6K` (the `HAVE_FANCY_SIMD` instances) is identical, whitespace-normalized, in three places: ik 0ceeb117 (2024-07-27), ik ed868d92 (2025-06-12, the last commit where Q4_K/Q6_K take `Q8_K`), and KTransformers v0.7.2 (a02efe36) `third_party/llamafile/iqk_mul_mat.inc`, the file KT runs. Copy candidates:

| Source | Activations | Foreign deps against our ggml | Has the 1-column kernel |
|---|---|---|---|
| ik HEAD (83950a3) | `Q8_2_X4` (new type and quantizer) | ik type system | no Q8_K path |
| ik ed868d92 `iqk_gemm_kquants.cpp` + `iqk_common.h`, `iqk_config.h`, `iqk_utils.h`, `iqk_quantize.h` | `Q8_K` | 11 ik-only type ids, 7 ik-only block structs (`*_r4`, `*_r8`, `q8_K32`, `q8_KV`), `iqk_quantize.h` | yes |
| llamafile 0.8.8 / main | `Q8_K` | none | no (only the generic `<Dequantizer, 1>` instance) |
| KT a02efe36 `third_party/llamafile/iqk_mul_mat.inc` (one file, 4869 lines) | `Q8_K` | none: only mainline types plus two local constants (`Q8_0_X4 = 98`, `Q8_1_X4 = 99`) | yes |

**Fit with ggml's call path.** The file's `iqk_mul_mat(Nx, Ny, ne00, typeA, A, strideA, typeB, B, strideB, C, stride_C, ith, nth)` reads src0 rows in place (row stride in blocks), takes the Q8_K activation row that ggml already writes to `wdata` before the barrier (`ggml-cpu.c:1609-1644`), and splits the Nx rows statically over threads. ggml's per-expert loop at decode also splits statically (24 chunks < 4 x nth, so `nchunk0 = nth`, `:1706`), so the work split does not change. No repack, so in-place streaming of experts (ADR 0039) is unaffected. The file's `DataInfo` addresses B and C with the same `{i1, i2}` row mapping as ggml's `mmid_row_mapping` (`:1478`), but KT's `iqk_mul_mat_moe` is broken (`set_mul_mat` commented out, `row_size_q8` uninitialized, lines 1027-1040), so we call `iqk_mul_mat` per expert.

**Switching the base to ik_llama.cpp (option B) is a rewrite of ggml-sys, not a swap.** Blockers, checked against ik 83950a3:
- libllama API: our crates call functions ik's `include/llama.h` lacks: `llama_get_memory`, `llama_memory_seq_rm`/`seq_pos_max`/`clear`, the `llama_sampler_*` chain, `llama_model_free`, `llama_model_n_layer`, `llama_model_n_ctx_train`, `llama_attach_threadpool`, `llama_vocab_get_attr`, `llama_batch_ext_*`, `llama_perf_context`, `llama_max_tensor_buft_overrides` (`crates/ggml-sys/src/llama/*.rs`).
- Backend API: ik has the 2024 index-based registry (`ggml-backend.h:125-131`), no `ggml_backend_dev_*`, no `ggml-cpu.h`, no `ggml-backend-reg.cpp`. `crates/ggml-sys/src/ggml/backend.rs:95-143`, `llama/devices.rs` and `memory/src/ggml.rs` do not compile.
- ADR 0052 cannot work: ik has no `ggml/src/ggml-cpu/` split, no `GGML_BACKEND_DL`, no `GGML_CPU_ALL_VARIANTS`.
- Windows (Rule 11): ik's only workflow is `build-container.yml`; its CMake has MSVC exclusions (`ggml/src/CMakeLists.txt:18`, `:954`). Untested on MSVC.
- Types and parity: ik's enum diverges from mainline above 42; the parity reference (libllama at our pin, ADR 0034/0035) and the CUDA kernels would be replaced by a 2024 fork's, and every golden re-recorded.
- Rebase path: ik's last mainline sync is August 2024, so `rebase-weekly.yml` has nothing to follow; the moe-cache baseline (`baseline_commit`, #30966) and the pin's own llama.cpp baseline stop being "the same ggml".
- ik_llama.cpp is one of our Phase 2 baselines (`vendor/PIN` `ik_commit`); basing the engine on it removes an independent comparison.

## Decision

1. **Option A: copy, do not write, kernel code.** Patch `patches/0002-iqk-avx512-q4k-q6k-decode.patch` adds KT a02efe36 `third_party/llamafile/iqk_mul_mat.inc` byte-identical (sha256 `9bef4f7b...148264`) as `ggml/src/ggml-cpu/iqk/iqk_mul_mat.inc`, compiled through one translation unit of ours that renames the entry point (`#define iqk_mul_mat ggml_iqk_mul_mat`, as KT's `iqk_mul_mat_amd_zen4.cpp` does), and four forwarding headers so the file's own includes resolve unchanged (`llama.cpp/ggml-impl.h`, `llama.cpp/ggml-quants.h`, `llama.cpp/ggml-common.h`, `sgemm.h`). Deviation from "copy from ik_llama.cpp": the kernel is ik's, unchanged, but the packaging is KT's, because it compiles against mainline types with no shim. ik ed868d92 is the fallback source if the KT copy does not build; it needs a shim for the 11 types and 7 structs. Either copy is frozen: ik moved Q4_K/Q6_K to `Q8_2_X4` on 2025-06-13, so no newer Q8_K kernel exists.
2. **Only AVX-512 VNNI variants.** The TU and the hook compile only when `__AVX512F__`, `__AVX512VNNI__`, `__AVX512VL__`, `__AVX512BW__` and `__AVX512DQ__` are defined (the file's `HAVE_FANCY_SIMD`): ADR 0052 variants cascadelake, cooperlake, icelake, zen4, sapphirerapids, and native builds on such hosts. Other variants keep ggml's kernels, so AVX2 numerics, CI goldens and the boxes without AVX-512 (A: Zen 3, B: Broadwell) do not change. iqk on AVX2 is a follow-up, decided by the ADR 0045 `kernels-cpu` stage.
3. **Hook, `mul_mat_id` only, decode only.** In `ggml_compute_forward_mul_mat_id`, per expert, after the tiled check and before the chunk loop: if `!params->use_ref`, the env switch is on, `cne1 == 1`, src0 type is Q4_K or Q6_K, `vec_dot_type == Q8_K`, src1 was converted to `wdata`, and src0 rows are contiguous (`nb01 == ggml_row_size(type, ne00)`), call `ggml_iqk_mul_mat(ne01, 1, ne00, type, src0_cur, nb01/type_size, GGML_TYPE_Q8_K, src1_row, row_size/type_size(Q8_K), dst_row, ne0, ith, nth)` and `continue` on true. Order per expert: tiled (batch >= 8, unchanged upstream behavior), iqk, vec_dot. Dense `mul_mat`, other types and 2 to 7 rows per expert are follow-ups.
4. **Switch and visibility.** `GGML_CPU_IQK_MM=0` turns the hook off (read once, like `GGML_CPU_TILED_MM`), so one binary gives the paired A/B. The CPU backend reports feature `IQK_MM` = `1` when compiled in and enabled (`ggml_backend_cpu_get_features`). It reaches ADR 0052's `CpuBackendInfo.isa`, the `tetraeder.cpu.isa` attribute, `env.cpu_backend` in bench results, and the "CPU backend" panel. `IQK_MM` joins ADR 0052's tracked features (host has it when it has the five AVX-512 flags), so `TetraederCpuVariantBelowHost` fires when a VNNI host runs without it.
5. **Numerics: within tolerance, not bitwise, against the AVX2 path.** Integer block sums are exact in both; float accumulation order and FMA use differ. Each row is computed by one thread independent of thread count and chunking, so results are deterministic per variant and bitwise stable across thread counts, RAM slots and streaming (the ADR 0034/0035 in-process checks hold). Parity: random blocks against an f64 reference of dequantized weights times the exact Q8_K activation, |err| <= 1e-5 x sum|w x| + 1e-6, for iqk and for ggml's own path (`use_ref`). Goldens record the tier (ADR 0052) and are pinned with `TETRAEDER_GGML_CPU_VARIANT`.
6. **Tests without AVX-512 hardware.** CI runners mostly lack AVX-512, so `crates/ggml-sys/tests/iqk_kernels.rs` (label `avx512_vnni`) skips cleanly unless the loaded CPU backend reports `IQK_MM`. A Linux CI job runs it under Intel SDE (`sde64 -icx`), pinned version and sha256, with `TETRAEDER_GGML_CPU_VARIANT=icelake`, plus the ggml-sys parity tests under the same variant. If SDE cannot be fetched reliably in CI, the job is dropped and the box nightly runs the test natively; performance is box only either way.
7. **Benchmark.** `expert_matmul.rs` gains Q6_K down projections (the Q4_K_M mix), records `tier` and `iqk_mm` from `cpu::backend_info()`, and runs arms in one process where possible. Box protocol (one box, interleaved, ADR 0052 pairing): auto variant with `GGML_CPU_IQK_MM=1` and `=0`, `TETRAEDER_GGML_CPU_VARIANT=haswell`, and `GGML_CPU_TILED_MM_FORCE=1` as a zero-code arm (expected slower at one row); then exit-regime decode with `GGML_CPU_IQK_MM=1` and `=0`.
8. **Attribution.** The copied file keeps its header (Copyright 2024 Iwan Kawrakow, Copyright 2024 KVCache.AI, Apache-2.0; our repo is Apache-2.0). A `README` next to it names the source repo, path, commit, sha256 and the identical ik kernels. `docs/sources.md` gets the row for patch 0002 with both chains and the update procedure.

## Interfaces

```c
// ggml/src/ggml-cpu/iqk/iqk-mm.h (patch 0002, our code)
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
#define GGML_CPU_IQK_MM 1
bool ggml_iqk_mul_mat(long Nx, long Ny, long ne00, int typeA, const void * A, long strideA,
                      int typeB, const void * B, long strideB, float * C, long stride_C, int ith, int nth);
bool ggml_cpu_iqk_mm_enabled(void);   // GGML_CPU_IQK_MM env, read once, default on
#endif
```

Rust: `CpuBackendInfo.isa` may contain `"IQK_MM"`; tracked feature list gains `IQK_MM`; `Backend::set_cpu_use_ref(bool)` via the CPU registry's `ggml_backend_cpu_set_use_ref` (for the in-process A/B in tests). Env: `GGML_CPU_IQK_MM`.

## Consequences

- **Expected gain, box C:** about 8 GB of CPU expert weights per token (87 of 94 layers, gate/up Q4_K, down Q4_K/Q6_K) in 112 ms of compute is about 71 GB/s, against roughly 120 to 140 GB/s of DRAM on an 8-channel 6330N. Compute halved from 15 to 26 threads (ADR 0045), so it is instruction bound, not bandwidth bound. A dpbusd GEMV does about half the integer instructions per block; estimate compute 112 to 70-85 ms, floor about 60 ms from bandwidth. Decode 257 to about 220-230 ms (3.77 to 4.3-4.5 tok/s, +12 to 19%). That closes about half of the 59 ms gap to KTransformers; the streamer's 128 ms (ADR 0045 hypotheses 1 and 2) stays the other lever. No gain on boxes without AVX-512 VNNI. The box run decides; until then this is an estimate.
- Prefill is unchanged: experts with 8 or more rows take tiled; with 2 to 7 rows they take vec_dot as before.
- Patch 0002 depends on the ADR 0052 implementation (variants, `TETRAEDER_GGML_CPU_VARIANT`, `cpu::backend_info()`, patch-apply step) and applies after 0001.
- MSVC: the file's guard includes `_M_X64` and MSVC defines the VNNI macros for its AVX-512 variants (ADR 0052), so the Windows build compiles it; it is not run there (no Windows box). If MSVC cannot compile it, the TU is excluded on MSVC and Windows runs ggml's kernels; that is recorded in the patch header.
- A frozen third-party file in the queue: updates come only from a newer `KT_VERSION` (re-copy, diff, rerun tests). The weekly rebase job checks that 0002 still applies to upstream master; tiled or upstream kernels that make it redundant show up there.
- Correction to ADR 0052's context: the pin has AVX-512 VNNI K-quant kernels (tiled, batch >= 8), just not for decode.
