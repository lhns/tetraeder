# 0081. Prefill steps run without CUDA graph capture

Status: Proposed (2026-11-20). Phase 5. Adds ggml patch 0006 (rule 8). Amends ADR 0069 (own executor) and ADR 0075 (range graph views).

## Context

- ggml-cuda decides per graph by call count (`ggml-cuda.cu` at fc504f91): the first call of a graph runs its kernels directly, the second call with unchanged node properties captures a CUDA graph and launches it, and later calls replay it. A fresh range uid (patch 0005) does not change this, because properties are still compared and found equal. Only the `GGML_CUDA_DISABLE_GRAPHS` env var turns capture off, for the whole process.
- The judged run (one warm-up, then `-r 1` per process, ADR 0068) therefore times the capture call. In tp2-pp2 prefill the capture call blocks the single issuing thread on each stage's device while an NCCL all-reduce is pending, so the stages stop overlapping. Measured: 32B 1394 and 30B 5138 tok/s in the capture call; with graphs off, warm direct eval gives 2004 and 6072, within 3% and 7% of the predictions (2064, 6512) (`s3d-20261120`, `s3e-20261120`). Layer-4 prefill, with no all-reduce, overlaps in its capture call (s3c).
- Serving never replays a prefill graph: prompt lengths and KV positions change on every prefill, so each one is a first call. Replay (32B 2007, 30B 6633) is as unrepresentative as the capture call.
- Decode graphs repeat and replay; the decode toggles (ADR 0069 History) depend on it.

## Decision

- Patch 0006 adds a per-graph flag, `ggml_graph_set_no_capture(struct ggml_cgraph *, bool)`, default off. ggml-cuda runs a flagged graph directly and leaves the graph's warm-up state alone. Other backends ignore the flag. Upstream behaviour is unchanged for unflagged graphs.
- The executor flags the range views of every step that carries more than one input row per sequence (prefill, chunked prefill, pipelined micro-batches). Decode steps (one row per sequence) are not flagged and keep capture and replay.
- No change to the verdict rules or the harness: the judged prefill number becomes the direct-eval number serving pays.

## Consequences

- Judged prefill on every candidate is measured at what serving pays. Single-GPU and layer-split prefill lose the capture-call overhead too; their bench reps after the second no longer replay.
- One more patch to carry through rebases; it touches one branch in the CUDA graph path.
- Not addressed: why the capture call blocks with an NCCL kernel pending, and the 30B rank-3 ranges that still block in replay (`s3e-20261120`). If direct eval stalls under other loads, one issuing thread per stage is the next step (s3d analysis).
- Verification: a backend test that a flagged graph is never captured over several calls and an unflagged one is (CUDA, skipped without a GPU); greedy parity unchanged; one confirmation box on host 20210 with the judged protocol (warm-up, `-r 1`) and `-r 5`, tp2-pp2 and layer-4 at mb4, both models, against the s3e direct-eval numbers.
