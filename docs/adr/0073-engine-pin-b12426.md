# 0073. Engine pin moves to llama.cpp b12426 before the Phase 4 verdict

Status: Proposed (2026-11-16). Supersedes the engine-pin part of ADR 0065's baseline note ("The engine pin stays at b11976").

## Context

- The engine (ggml-sys, libllama, the builders) was pinned at b11976 (2e0d86beb); the Phase 4 baseline llama-server is b12426 (fc504f91d, ADR 0065).
- On one RTX 3090 (`bench/results/20261116pinab/pin-ab/summary.md`) llama-server built at b11976 reaches 0.930 (Qwen3-8B) and 0.834 (Qwen3-30B-A3B) of llama-server b12426 at concurrency 4, and libllama's 4-sequence decode step takes 83.4 ms per 8 steps at b11976 against 71.8 at b12426; one-sequence steps are level. Part of the c4 gap is the pin, and the verdict compares against b12426.
- `track-pin-b12426` showed the move is cheap on the patch side: patches 0001 to 0003 apply unchanged, 0004 is rebased onto upstream's stream-k `tile_spec`.

## Decision

- The engine pin moves to b12426 (`vendor/PIN` `tag`, `commit`, `image_tag` `b12426-ikb25c3fd2-blfc504f91`), so engine and baseline run the same llama.cpp.
- CPU numerics change upstream in the range, mainly #31854 (F16 matmuls take F32 src1; flash attention keeps Q in F32 for an F16 K) and #29810 (flash attention accumulates an F16 V in F32). The goldens are re-recorded per OS by the `ci` workflow's `record_golden` jobs, and each shift is checked against llama-simple at the new pin (same greedy text) and against the dense builder (within its 0.05).
- DRIFT

## Consequences

- The CUDA parity validated on b11976 (ADR 0071 and its addenda) has to be confirmed on the new pin on one box before the verdict: the gpu-smoke5/6 suites, `gpu_real_batched` on both real models, and patch 0004's GPU tests.
- Old ADRs keep their b11976 readings of ggml; they describe the code they were written against.
- The box image is rebuilt for the new tag by `box-image.yml`.
