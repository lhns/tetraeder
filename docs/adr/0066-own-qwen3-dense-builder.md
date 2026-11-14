# 0066. Own Qwen3 dense builder on the shared attention module, multi-sequence from the start

Status: Proposed (2026-11-14). Phase 4. Builds on ADR 0015 (own builders per family, libllama as reference), 0030 (MoE builder), 0036 (whole-step graph), 0050 b (KV formats, rotation) and 0065 (batch backend, attention over a shared pool). Brief: C5 in `docs/phase4-briefs.md`.

## Context

The Phase 4 verdict (ADR 0065) judges Qwen3-8B on an own builder, with libllama as the control. The own KV pool and scheduler need a graph Tetraeder builds itself; libllama's builders are internal C++ (ADR 0015). The MoE builder is 2848 lines of which most is expert machinery (slots, pair groups, streaming, overlap, prefetch); attention, norms, RoPE and the KV cache are shared with a dense Qwen3 and move into `crates/model/src/attn.rs` first (step C1.5).

Qwen3 dense against Qwen3 MoE, at the pin (`src/models/qwen3.cpp`, `qwen3moe.cpp`):

| | Dense (`qwen3`) | MoE (`qwen3moe`) |
|---|---|---|
| Attention | RMS norm, Q K V projections, Q and K RMS norm, NeoX RoPE, GQA flash attention, `wo` | same |
| FFN | `down(swiglu_split(gate(x), up(x)))`, one per layer (`build_ffn`, `LLM_FFN_SILU`, `LLM_FFN_PAR`) | router softmax, top-k, normalized weights, expert matmuls |
| Last layer | output rows selected after attention (`get_rows` with `inp_out_ids`) before the residual add and FFN | same in libllama; the builder selects in `head` |
| Output | `output.weight`, or the token embedding when absent (Qwen3-0.6B ties it) | same |
| GGUF keys | `qwen3.*`, `feed_forward_length` | `qwen3moe.*`, expert counts and lengths |

Qwen3-8B: 36 layers, n_embd 4096, 32 heads, 8 KV heads, head 128, n_ff 12288, context 40960. A tiny dense fixture does not exist; the tiny MoE (`tests/fixtures/tiny-qwen3moe-f16.gguf`) is written by `xtask/tiny_moe.py` through the vendored gguf-py (`cargo xtask tiny-moe`). A real dense model is already in CI: Qwen3-0.6B Q8_0 with greedy goldens (`crates/engine/tests/golden.rs`, `bench/golden/qwen3-0.6b-q8_0-cpu-sse42-<os>.json`).

## Decision

- **Module** `crates/model/src/qwen3.rs`: `pub struct Qwen3`, `pub struct DenseHparams` parsed from `qwen3.*` keys (refuses other architectures), `DenseOptions`. It does not use `Hparams` or `ModelFile` (both MoE-shaped); both hparam types give an `attn::AttnHparams` for the shared module.
- **Load.** `DenseOptions { backend: BackendChoice, n_ctx_pool: usize, n_seq_max: usize, kv: KvFormat, kv_rotation: Option<bool>, threads: Option<i32>, reuse_step_graph: bool }`. Every weight on the primary backend (CUDA, or CPU in file layout, no repack, as the MoE builder's resident path). No plan IR placement, partial offload or expert-style streaming in Phase 4: the 8B fits the verdict boxes; anything else is served by `--engine llama`.
- **Graph.** One whole-step graph per step (ADR 0036): `get_rows` embedding; per layer `attn::attention` (shared, unchanged), residual, FFN norm, `ffn_down(swiglu_split(ffn_gate(x), ffn_up(x)))`, residual; output norm and head. Op order and fusions follow libllama's `qwen3.cpp` so CPU logits match it: at the last layer the output rows are selected with `get_rows` after attention, as libllama does, so its FFN and the head run on the requested rows only (the last token of each sequence, every token of a sequence with `all_logits`). The kept graph is reused when (n_tokens, n_kv, outputs) repeat.
- **Multi-sequence from day one.** The KV is `kvcache`'s pool with the chunked layout (ADR 0067 step 1): per step `KvPool::plan_step` gives positions (RoPE), destination rows (`set_rows`), n_kv and the mask; attention is ADR 0065 (a). `Qwen3` implements the model side (`decode(&[SeqEval]) -> logits rows`, `seq_rm`, `seq_cp` as row copies on the device, `seq_snapshot`, `seq_restore`) and `scheduler::DenseBackend` wraps it as a `BatchBackend`. A single sequence is the case `n_seq_max = 1`; there is no separate one-sequence path.
- **KV formats and rotation** as ADR 0050 b, through the shared module: F16 default, Q8_0 and Q4_0 pairs, Hadamard rotation on exactly when libllama rotates.
- **serve.** `--engine dense` selects it; `auto` keeps libllama for `qwen3` until the verdict session has passed, then a one-line change makes `auto` pick the builder.
- **Telemetry (Rule 6).** The builder adds no new signal: it records `tetraeder.step.duration`, `tetraeder.memory.used{kind=weights|kv}`, `tetraeder.kv.tokens` and `.capacity` and `tetraeder.kv.blocks` through the shared module and pool, which the existing step and KV panels and `TetraederKvNearFull` cover, with the attribute `tetraeder.builder = dense`.

**Parity plan (Rule 2), tests before code.**
- **Tiny dense fixture.** `xtask/tiny_dense.py` (pattern of `tiny_moe.py`, same byte-level vocab and seed scheme) writes `tests/fixtures/tiny-qwen3-f16.gguf` via `cargo xtask tiny-dense`: arch `qwen3`, 2 layers, n_embd 128, 4 heads, 2 KV heads, head 64, n_ff 256, context 512, F16 matrices and F32 norms, untied output. Head 64 (not the MoE fixture's 16) so CUDA flash attention, quantized KV and the rotation run on it. About 1 MB, committed.
- **Per op.** The shared module is already covered by the MoE parity tests; per dense op, a graph test compares the builder's op sequence for one layer with libllama's names in order (pattern of `whole_step_graph_orders_ops_like_the_segments_and_libllama`).
- **End to end, CPU (CI).** Against libllama on the tiny dense fixture: prefill logits of every position and 12 greedy steps, top-5 within `TOL_CPU` (`crates/model/tests/common/mod.rs`), top-1 equal. `libllama_reference_at` already takes a path; `check_against` gets a small trait `Evaluator` (`reset`, `eval`, `eval_last`, `n_vocab`) implemented for `Qwen3Moe` and `Qwen3`, so both builders use one checker. KV formats and rotation as `kv_format.rs` and `kv_rotation.rs` (rotation in its own test binary).
- **Greedy golden.** On Qwen3-0.6B Q8_0 the builder's tokens equal `bench/golden/qwen3-0.6b-q8_0-cpu-sse42-<os>.json` exactly and its top-k logits are within that file's tolerance (the test of `crates/engine/tests/golden.rs`, run against the builder); skips without the model unless `TETRAEDER_REQUIRE_MODELS=1`.
- **Multi-sequence.** Three sequences of different lengths decoded together and with a prefill chunk mixed in, against each alone through the builder and against libllama with `n_seq_max = 3` and unified KV: logits within `TOL_CPU`, greedy equal except where the reference's top-2 margin is below ADR 0043's near-tie bound. Not bitwise: batched matmuls take other kernels.
- **GPU (box, `gpu_` prefix, `TETRAEDER_REQUIRE_GPU`).** The same against the CPU reference within `TOL_GPU`, one and three sequences; Qwen3-8B Q4_K_M: top-1 equal to libllama on the prompt of `parity.rs` and 64 greedy tokens equal or a reported near-tie divergence.

## Consequences

- A second builder exercises the shared module, so attention changes are tested by two families before Phase 8 adds more.
- Placement for dense models stays libllama's until a later phase gives the dense builder plan IR fields; the planner does not propose the dense builder in Phase 4.
- The builder is the judged path; if it misses the verdict while the libllama control passes, the gap is the builder's, and the control says so in the same session.
