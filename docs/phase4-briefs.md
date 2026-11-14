# Phase 4 implementation briefs

Briefs for ADR 0065 (verdict, batch backend, continuous batching), ADR 0066 (own Qwen3 dense builder) and ADR 0067 (KV layout). Every design decision is in those ADRs; read the ones your brief names, then AGENTS.md. A brief is executable without design decisions; if a premise looks wrong, stop and report with evidence instead of redesigning.

**For every brief.** Failing tests first, then code. Every local cargo command goes through the shared lock with one job: `L=/c/Users/pierr/AppData/Local/Temp/claude/cargo-host.lock; until mkdir "$L" 2>/dev/null; do sleep 20; done; CARGO_BUILD_JOBS=1 cargo <args>; rmdir "$L"`. Done means `cargo build --workspace`, `cargo test --workspace`, `cargo clippy --workspace -- -D warnings`, `cargo fmt --check` pass (Rule 11; use CI on your `track-*` branch when the lock starves local builds), plus the Python tests your brief names, and the ADR gets an "Implementation notes" paragraph for your step (what was built, any deviation). Tests that need a GPU or a real model are labelled and skip cleanly (Rule 9). No box: no brief provisions one. Commit only if your brief says so; push only your own branch.

## Order and parallelism

| Wave | Steps | Shared files |
|---|---|---|
| 1 | C1.5, C2, C3a, C6-1 in parallel | C2 and C6-1 both append to `telemetry/src/metrics.rs`, `overview.json` and `engine.yml`: append, do not reorder, merge textually |
| 2 | C3b, C4, C5 in parallel | C4 and C5 implement their backends in separate files and both append to `scheduler/src/{lib.rs,load.rs}` and `model/src/lib.rs`; C5 and C3b both add flags to `tetraeder/src/serve.rs`. Append only, merge textually. All three read, never change, `model/src/attn.rs` and `kvcache` |
| 3 | verdict session (ADR 0065), run by the coordinator | |

C3b needs C2. C4 needs C1.5, C2 and C6-1. C5 needs C1.5, C2 and C6-1.

---

## Brief C1.5: shared attention module, bit-equal

**Goal.** Move attention, norms, RoPE parameters, the KV tensors and the causal mask out of `crates/model/src/qwen3moe.rs` into `crates/model/src/attn.rs`, so ADR 0066's dense builder and ADR 0065's multi-sequence KV reuse them. Logits stay bit for bit what they are.

**Read first.** ADR 0066 (context, the table); ADR 0065 "Attention over sequences of unequal length"; `qwen3moe.rs`: `Kv` (line 410), `State` (450), `run` mask and pos (1095 to 1120), `kv_types`, `rot_size_k`, `ROT_SIZE_V`, `alloc_rot`, `alloc_kv` (2041 to 2147), `attention_and_router` (2345), the graph tests `whole_step_graph_orders_ops_like_the_segments_and_libllama` and `mask_has_one_row_per_token_like_libllama`; `crates/model/tests/parity.rs`, `kv_format.rs`, `kv_rotation.rs`.

**Decisions.**
- New `crates/model/src/attn.rs`, `pub` items re-exported from `lib.rs` only where listed:
  ```rust
  pub struct AttnHparams { pub n_layer: u32, pub n_embd: u32, pub n_head: u32, pub n_head_kv: u32,
      pub head_dim: u32, pub n_rot: u32, pub n_ctx_train: u32, pub n_vocab: u32,
      pub rms_eps: f32, pub rope_freq_base: f32 }            // re-exported
  impl Hparams { pub fn attn(&self) -> AttnHparams }          // in hparams.rs
  pub(crate) struct AttnWeights { pub attn_norm: OwnedTensor, pub wq: OwnedTensor, pub wk: OwnedTensor,
      pub wv: OwnedTensor, pub wo: OwnedTensor, pub q_norm: OwnedTensor, pub k_norm: OwnedTensor }
  pub(crate) struct Kv { pub k: Vec<OwnedTensor>, pub v: Vec<OwnedTensor>, pub n_rows: usize,
      pub fmt: KvFormat, pub rot_k: Option<OwnedTensor>, pub rot_v: Option<OwnedTensor> }
  pub(crate) fn alloc_kv(h: &AttnHparams, n_rows: usize, backend: &Backend, fmt: KvFormat,
      rotation: Option<bool>) -> Result<(Kv, usize)>;
  pub(crate) fn kv_types(h: &AttnHparams, fmt: KvFormat) -> Result<(Type, Type)>;
  /// Inputs of one step's attention. `rows` are the KV rows written (set_rows indices), `pos`
  /// the RoPE positions; the MoE builder passes its `pos` tensor for both until C4.
  pub(crate) struct AttnStep<'a> { pub pos: Tensor<'a>, pub rows: Tensor<'a>, pub mask: Tensor<'a>,
      pub n_tokens: usize, pub n_kv: usize }
  pub(crate) fn rope_params(h: &AttnHparams) -> RopeParams;
  /// RMS norm, attention with Q/K norm, RoPE, rotation, KV write and flash attention, `wo`,
  /// plus the residual: returns ffn_inp exactly as attention_and_router computes it today.
  pub(crate) fn attention<'a>(h: &AttnHparams, ctx: &'a Context, g: &mut Graph<'a>,
      lw: &'a AttnWeights, kv: &'a Kv, il: usize, st: &AttnStep<'a>, x: Tensor<'a>) -> Tensor<'a>;
  pub(crate) fn rms_norm_mul<'a>(ctx: &'a Context, x: Tensor<'a>, w: Tensor<'a>, eps: f32) -> Tensor<'a>;
  /// `[n_kv, n]` F16 bits, row i open on columns 0..=n_past+i (today's loop in `run`).
  pub fn causal_mask(n_past: usize, n: usize, n_kv: usize) -> Vec<u16>;
  pub const KV_PAD: usize = 256;
  pub const F16_NEG_INF: u16 = 0xFC00;
  /// One sequence's tokens in a multi-sequence step (C4 and C5 use it; re-exported).
  pub struct SeqEval<'a> { pub seq: u32, pub tokens: &'a [i32], pub all_logits: bool }
  /// Device copies of KV rows, every layer, K and V (one graph of `cpy` between views);
  /// used by C4 and C5 for seq_cp and chunked moves. Unused by the MoE builder in this step.
  pub(crate) fn copy_rows(kv: &Kv, sched: &Sched, copies: &[(usize, usize, usize)]) -> Result<()>;
  ```
- `qwen3moe::Layer` holds `attn: AttnWeights` instead of the seven fields; `attention_and_router` calls `attention` and keeps the router part. `Kv::n_ctx` is renamed `n_rows`; nothing else in the MoE builder changes meaning.
- The op sequence and node order stay identical (the graph tests are the check), and so do the CUDA graph key, `set_rows` with positions, tensor names (`k_cache_<il>`, `v_cache_<il>`) and the KV snapshot format.

**Tests first.**
- `crates/model/tests/extract_bits.rs` (label `cpu`): on the tiny fixture (resident) and on the synthetic Q8_0 MoE of `tests/common/synth.rs` with Q8_0 KV rotated, resident and streamed (`MissPolicy::Cpu`, setup as `kv_rotation.rs`): `eval(PROMPT)` logits of every position, 12 greedy `eval_last` steps' logits and the final `kv_snapshot()`. With `TETRAEDER_BITS_RECORD=<file>` it writes the f32 bit patterns and snapshot bytes; with `TETRAEDER_BITS_CHECK=<file>` it asserts exact equality; with neither it prints `skip:` and passes. Commit it first, record on the unchanged code, refactor, check. The record file is not committed (it depends on the CPU and OS).
- `attn::copy_rows` unit test on the CPU backend: copy three row ranges, read back with `get_bytes`, untouched rows unchanged.
- `causal_mask` unit test against the old loop for n_past in {0, 5, 255}, n in {1, 7}.

**Done.** Workspace checks; `extract_bits` check passes on your machine (report the CPU); every existing `crates/model` test passes unchanged.

**Must not change.** `Qwen3Moe`'s public API, `LoadOptions`, numerics, op order, any test expectation.

**Report.** Lines moved, the record/check run, any test that needed an edit (none expected).

---

## Brief C2: multi-sequence libllama, batch backend, continuous batching

**Goal.** `tetraeder serve --parallel 4` decodes up to four requests in one step on the libllama backend, with chunked prefill under a token budget, per ADR 0065. The MoE builder keeps working through a one-sequence adapter.

**Read first.** ADR 0065 (all of "Decision"); ADR 0046 and 0049 (what the current loop does); `crates/scheduler/src/{backend.rs,scheduler.rs,prefix_cache.rs,load.rs}`; `crates/engine/src/{engine.rs,config.rs}`; `crates/ggml-sys/src/llama/{context.rs,batch.rs}`; `crates/gateway/src/gateway.rs` (`max_concurrent`, `admit`); `crates/tetraeder/src/serve.rs`; `crates/scheduler/tests/scheduler.rs` (fixture and vocab setup); `vendor/llama.cpp/include/llama.h` 360 to 420 and 760 to 790.

**Decisions.**
- `ggml-sys`: `ContextParams { n_seq_max: u32, kv_unified: bool, .. }` from libllama's defaults; `Context::memory_seq_cp(&mut self, src: i32, dst: i32, p0: i32, p1: i32)`, `Context::n_seq_max(&self) -> u32`. Each `unsafe` block with a safety comment (Rule 7).
- `engine`: `ContextConfig { n_seq_max: u32 /* default 1 */, kv_unified: bool /* default true */, .. }`. `Engine` keeps a length per sequence (`n_past()` is sequence 0's). New:
  ```rust
  pub struct SeqEval<'a> { pub seq: i32, pub tokens: &'a [Token], pub all_logits: bool }
  /// One timed step (prefill phase if any input has more than one token). Fails with
  /// InvalidRequest when the total exceeds n_batch; on any error every input's sequence is
  /// cut back to its length before the call.
  pub fn decode_seqs(&mut self, inputs: &[SeqEval<'_>]) -> Result<()>;
  /// The k-th output of the last decode_seqs, in input order (rows of an input in token order).
  pub fn output_logits(&mut self, k: usize) -> Result<&[f32]>;
  pub fn seq_len(&self, seq: i32) -> usize;
  pub fn seq_rm(&mut self, seq: i32, keep: usize) -> Result<()>;
  pub fn seq_cp(&mut self, src: i32, dst: i32, n: usize) -> Result<()>;
  pub fn seq_snapshot(&mut self, seq: i32) -> Result<Vec<u8>>;
  pub fn seq_restore(&mut self, seq: i32, bytes: &[u8], n: usize) -> Result<()>;
  ```
  `eval`, `truncate`, `snapshot`, `restore`, `generate`, `bench_*` keep their behaviour on sequence 0.
- `scheduler/src/backend.rs`: `SeqId`, `SeqInput`, `BackendError`, `Outputs`, `BatchBackend` exactly as ADR 0065. `impl BatchBackend for LlamaBackend` (reservations are bookkeeping against `n_ctx`; `seq_cp` counts n rows for dst; `LlamaError::NoKvSlot` maps to `BackendError::KvFull`, every other error to `Failed`). `pub struct SingleSeq<B: Backend>` implements `BatchBackend` with `n_seq_max() == 1` over the old trait; `MoeBackend` is served through it. The old `Backend` trait stays until C4. `Loaded.backend` becomes `Box<dyn BatchBackend>`.
- `scheduler/src/batch.rs` (pure, no backend): the step planner.
  ```rust
  pub(crate) struct Live { pub seq: SeqId, pub pending: usize, pub decoding: bool,
      pub paused: bool, pub admitted: u64 }
  pub(crate) struct Take { pub seq: SeqId, pub n: usize }
  /// Decoding, unpaused sequences take their pending tokens (1, or 1 + drafts) even beyond the
  /// budget; prefilling ones take chunks in admission order from what is left.
  pub(crate) fn plan_step(budget: usize, live: &[Live]) -> Vec<Take>;
  /// The sequence to preempt on KvFull, after every idle sequence has been evicted (oldest
  /// first, saved to RAM per ADR 0049): the most recently admitted other live one.
  pub(crate) fn victim(live: &[Live], needing: SeqId) -> Option<SeqId>;
  ```
- `scheduler.rs`: the loop of ADR 0065 (admission, step, sampling, `try_send` with pause, cancellation, retirement to idle, preemption with requeue, errors per step). `SchedulerConfig` gains `parallel: usize` (default 4) and `step_tokens: usize` (default 512); `prefill_chunk` is removed in favour of `step_tokens`. Prefix reuse in this step: an admitted job takes the idle resident sequence with the longest common prefix (at most prompt - 1), or restores the best RAM entry into a free or evicted id when it beats that by `RESTORE_MIN_GAIN`; an evicted idle sequence of at least `SAVE_MIN_TOKENS` is saved to RAM first (ADR 0049 rules per sequence). Copying from a busy sequence is C3b.
- `serve`: `--parallel <N>` (default 4), `--step-tokens <N>` (default the `--ubatch` value, 512; with it, `-b` no longer applies to `serve` and its help says so); the gateway's `max_concurrent = parallel`; libllama gets `n_seq_max = parallel`, `kv_unified = true`, `n_batch = n_ubatch = step_tokens`, `n_ctx = --ctx`.
- Telemetry (`crates/telemetry/src/metrics.rs`): `scheduler_step_tokens(kind: &'static str, n: u32)` (`tetraeder.scheduler.step.tokens`, attribute `tetraeder.kind`), `scheduler_sequences(state: &'static str, n: u64)` (`tetraeder.scheduler.sequences`, `tetraeder.state`), `scheduler_preemption()` (`tetraeder.scheduler.preemptions`); `scheduler_batch` records sequences per step. Panel "Scheduler: sequences, step tokens, preemptions" beside "Scheduler batch size" in `deploy/grafana/dashboards/overview.json`. Alert in `deploy/alerts/engine.yml`:
  ```yaml
  # The KV pool is too small for the load: sequences are preempted and recomputed (ADR 0065).
  - alert: TetraederSchedulerPreempting
    expr: sum(rate(tetraeder_scheduler_preemptions_total[10m])) * 60 > 1
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "More than one preemption per minute for 10 minutes"
  ```

**Tests first.**
- `crates/scheduler/tests/fake/mod.rs`: `FakeKv: BatchBackend`, logits a deterministic function of a hash of the sequence's tokens (so P4 is checkable), disjoint row ranges per sequence (own accounting; C6-1's layouts are not needed), capacity, `fail_every: Option<u32>` and `kv_full_at: Option<usize>` injection; vocabulary from the tiny fixture as `tests/scheduler.rs` does.
- `crates/scheduler/tests/batching.rs`, proptest through the gateway `Engine` trait on `Scheduler` over `FakeKv`: P1 to P5 of ADR 0065 under random arrivals, prompt lengths (1 to 3000), `max_tokens`, cancellations, slow consumers (a receiver that waits), KvFull and failures.
- `batch.rs` unit tests: budget split, decoders beyond the budget, paused sequences skipped, victim choice.
- `crates/engine/tests/multi_seq.rs` (cpu): 4 sequences of different prompt lengths on the tiny fixture, decoded together with a prefill chunk mixed in, against each alone: greedy equal, last logits within `TOL_CPU` (1e-4); `seq_cp` then continue equals a fresh prefill within `TOL_CPU`; errors leave lengths unchanged.
- `crates/scheduler/tests/scheduler.rs` keeps passing (one conversation reuses its prefix as before); add: four concurrent requests on `--engine llama` with the tiny fixture all finish and `tetraeder.scheduler.batch.size` records a value above 1.
- `python -m pytest tests/test_alert_rules.py` passes with the new alert.

**Done.** Workspace checks; the tests above; locally, `scripts/gateway_bench.py` c1 and c4 against `serve --engine llama --parallel 4` on Qwen3-0.6B Q8_0 (CPU) report `throughput_tps` (informative, in the report).

**Must not change.** Gateway API behaviour and request log fields; sampling and its seeds; ADR 0049 semantics for a single conversation; the decode profile fields; anything in `crates/model`.

**Report.** c1 and c4 numbers, any libllama behaviour that differed from llama.h's comments, open issues.

---

## Brief C3a: verdict harness

**Goal.** Scripts that run ADR 0065's verdict session on a box and compute `verdict.json` exactly as the ADR states. No Rust.

**Read first.** ADR 0065 "Verdict" (all of it); `scripts/{gateway_bench.py,gateway_replay.py,gateway_box.sh,gateway_summary.py,cotenant_guard.py,engine_sample.py}`; how `scripts/exit_run.sh` runs the host sampler and the co-tenant guard; `tests/test_gateway_scripts.py`; `bench/noise/*.json` and `[[noise_band]]` in `bench/spec/phase0.toml`; AGENTS.md "GPU work on Vast.ai".

**Decisions.**
- `gateway_replay.py`: `summary.ttft_p90_s` (nearest rank, as `gateway_bench.percentile`); `--max-prompt-tokens N` skips entries whose logged `usage.prompt_tokens` exceeds N (count in `params.skipped`); `params.log_sha256` (sha256 of the log file's bytes) and `params.requests`. Results keep `index`, `status` and `ttft_s` per request.
- `scripts/phase4_box.sh` (pattern of `gateway_box.sh`; flock guard, `nohup setsid`, budget, results written as they finish, resumable by file presence): models `q8b` (Qwen3-8B Q4_K_M) and `q30b` (Qwen3-30B-A3B Q4_K_M); baseline `/opt/llama.cpp-baseline/bin/llama-server` with ADR 0065's flags; pre-pass `-kvu` and `-no-kvu` at c4 into `prepass-<model>-<kvu|nokvu>.json`; then per model (baseline, judged), (judged, baseline), control after each pair; judged engine `dense` for q8b and `moe` for q30b when `Q30B_ENGINE=moe`, else `llama` with `counts: false`; `nvidia-smi --query-gpu=clocks_throttle_reasons.active --format=csv` before and after each run into `gpu-<run>.txt`; host sampler and `cotenant_guard.py` per run; `--seed <rep>` for both engines. TTFT stage when `PHASE4_LOG=<path on the box>` is set: (on, off), (off, on) with ADR 0065's flags, files `replay-q8b-<on|off>-r<rep>.json`. Server logs kept (`kv_unified = '...'` is read from them).
- `scripts/phase4_summary.py <dir>` writes `verdict.json` with exactly ADR 0065's fields and rules (band choice and `band_source`, clean run and session clauses, `aa`, per-pair ratios, `worst`, pass as min r >= 1 - b, `counts`, TTFT ratios over requests with status 200 in both runs, `phase4_pass`). Missing inputs give `pass: null` with a reason, never true.
- `.gitignore`: `bench/private/`.

**Tests first.** `tests/test_phase4_summary.py` on synthetic result directories: a pass; a fail at r = 1 - b - 0.001; a pair with errors (unclean, `pass: null` when fewer than two clean pairs); throttle reason active; A/A outside the band; band from a class file with n >= 3 against the spec fallback; q30b on llama (`counts: false`, does not block `phase4_pass`); TTFT pass, TTFT p90 fail, a request failing in one run only (excluded). `tests/test_gateway_scripts.py`: p90, `--max-prompt-tokens`, hash.

**Done.** `python -m pytest tests/test_phase4_summary.py tests/test_gateway_scripts.py`; `bash -n scripts/phase4_box.sh`; a dry run of `phase4_box.sh` against `serve --fake` and a fake llama-server is not required.

**Must not change.** `gateway_box.sh` and `gateway_summary.py` behaviour (Phase 3b records use them).

**Report.** The `verdict.json` of one synthetic pass case.

---

## Brief C3b: prefix reuse across sequences

**Goal.** A new request reuses a prefix held by a busy sequence through `seq_cp`, and `--no-prefix-reuse` turns every form of reuse off, for ADR 0065's TTFT verdict.

**Read first.** ADR 0065 (scheduler loop, TTFT verdict), ADR 0049, C2's `scheduler.rs` and `fake/mod.rs`.

**Decisions.**
- Admission order of sources, longest usable prefix wins, ties to the cheaper source: idle resident sequence (C2), busy sequence via `seq_cp(busy, new, lcp)` when it beats the idle and RAM options by at least `RESTORE_MIN_GAIN`, RAM entry (C2). Reuse is at most prompt - 1 tokens.
- `SchedulerConfig::prefix_reuse: bool` (default true); `serve --no-prefix-reuse` sets it false: every admission starts an empty sequence (an idle one is cut to 0 first), no RAM restore, no `seq_cp`. `--cache-ram` is independent.
- Metrics: `tetraeder.kv.prefix.hits` gains source `seq`; each `seq_cp` records `tier_transfer(bytes, Some(took), Tier::Vram, Tier::Vram, Purpose::Kv)` (bytes from the backend's row size); the panel "Prompt tokens from the prefix cache" shows the new source (query change only). `PrefixCacheThrash` stays the alert.

**Tests first.** FakeKv: two concurrent requests sharing a 1000-token prefix, the second admitted while the first decodes: `cached_tokens >= 1000` and a `seq_cp` call recorded; with `prefix_reuse: false` `cached_tokens == 0` for every request of a five-turn conversation; batching.rs's generator gains shared prefixes and P1 to P5 hold. libllama tiny fixture: the copied sequence's continuation equals a fresh prefill within `TOL_CPU`.

**Done.** Workspace checks; `tests/test_alert_rules.py`.

**Must not change.** C2's loop structure; ADR 0049 save and restore thresholds.

---

## Brief C6-1: KV layouts, memory simulation, CPU gather benchmark

**Goal.** `crates/kvcache` provides the pool and layouts that C4 and C5 use (chunked), the paged layout for simulation, and the two measurements of ADR 0067 step 1 with their verdict.

**Read first.** ADR 0067 (all), ADR 0065 (attention, P2 and P3), `crates/kvcache/Cargo.toml`, the proxy logs `bench/results/20261111d/claude-code-proxy/proxy-tetraeder-r1-*.jsonl`, `crates/model/tests/cpu_bench.rs` (pattern of an ignored CPU benchmark), the safe ggml API in `crates/ggml-sys/src/ggml` (`get_rows`, `cpy`, `flash_attn_ext`, views).

**Decisions.** `kvcache` is pure Rust (no ggml), dependency `thiserror`; `model` and `scheduler` may depend on it, never the reverse.
```rust
pub type SeqId = u32;
pub const ROW_ALIGN: usize = 256;
pub const F16_NEG_INF: u16 = 0xFC00;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowCopy { pub src: usize, pub dst: usize, pub n: usize }
#[derive(Debug, thiserror::Error)]
pub enum KvError { Full { needed: usize, free: usize }, UnknownSeq(SeqId), Invalid(String) }

pub trait Layout: Send {
    fn name(&self) -> &'static str;
    fn capacity(&self) -> usize;
    /// Rows not reserved by any sequence.
    fn free(&self) -> usize;
    fn reserved(&self, seq: SeqId) -> usize;
    /// Rows for positions 0..n of `seq`; returns moves to apply before the next step. Full changes nothing.
    fn reserve(&mut self, seq: SeqId, n: usize) -> Result<Vec<RowCopy>, KvError>;
    /// Releases what positions 0..keep do not need, in ROW_ALIGN units; 0 frees the sequence.
    fn shrink(&mut self, seq: SeqId, keep: usize);
    fn row(&self, seq: SeqId, pos: usize) -> usize;
}
pub struct Chunked;  impl Chunked { pub fn new(capacity: usize) -> Self }
pub struct Paged;    impl Paged { pub fn new(capacity: usize, block: usize) -> Self }

pub struct StepInput { pub seq: SeqId, pub n_tokens: usize, pub all_logits: bool }
/// Tokens in input order, each input's tokens in position order.
pub struct StepPlan { pub pos: Vec<i32>, pub rows: Vec<i32>, pub n_kv: usize,
    /// F16 bits, `[n_kv, n_tokens]`, token-major: 0 where the token may read the row.
    pub mask: Vec<u16>,
    /// Token indices whose logits are returned.
    pub outputs: Vec<i32> }
pub struct BlockCounts { pub used: usize, pub reserved: usize, pub free: usize }

pub struct KvPool { /* Box<dyn Layout>, lengths */ }
impl KvPool {
    pub fn new(layout: Box<dyn Layout>) -> Self;
    pub fn layout_name(&self) -> &'static str;
    pub fn capacity(&self) -> usize;
    pub fn free(&self) -> usize;
    pub fn seq_len(&self, seq: SeqId) -> usize;
    pub fn reserve(&mut self, seq: SeqId, n: usize) -> Result<Vec<RowCopy>, KvError>;
    pub fn seq_rm(&mut self, seq: SeqId, keep: usize);
    /// `dst` must be empty: reserves n rows for it and returns the copies src -> dst.
    pub fn seq_cp(&mut self, src: SeqId, dst: SeqId, n: usize) -> Result<Vec<RowCopy>, KvError>;
    /// Sequences distinct; each reservation must cover seq_len + n_tokens.
    pub fn plan_step(&self, inputs: &[StepInput]) -> Result<StepPlan, KvError>;
    /// After a successful decode: lengths grow by n_tokens.
    pub fn commit(&mut self, inputs: &[StepInput]);
    /// Rows of positions 0..n of `seq` as (first row, length) runs, for snapshots.
    pub fn runs(&self, seq: SeqId, n: usize) -> Vec<(usize, usize)>;
    pub fn blocks(&self) -> BlockCounts;
}
```
- `Chunked` and `Paged` exactly as ADR 0067 (first fit from row 0, 256-row alignment, chunked moves to needed plus GROW, paged lowest free block). With one sequence at row 0, `plan_step` must give `rows == pos`, n_kv = pad(len + n, 256) and the mask of `model::attn::causal_mask`: this is what keeps C4 bit-equal for one sequence (copy the 12-line mask loop as a test oracle; do not depend on `model`).
- n_kv = pad(1 + highest row any input reads or writes, ROW_ALIGN).
- Telemetry (`metrics.rs`): `kv_blocks(device: &str, tier: &str, state: &'static str, n: u64)` (`tetraeder.kv.blocks`, `tetraeder.state` = used, reserved, idle, free) and `kv_relocation()` (`tetraeder.kv.relocations`); called by C4 and C5. Panel "KV pool: blocks by state, relocations" beside "KV cache"; alert in `deploy/alerts/engine.yml`:
  ```yaml
  # Preemptions while a quarter of the pool is free: holes, not size, limit the load (ADR 0067).
  - alert: TetraederKvFragmented
    expr: |
      sum(rate(tetraeder_scheduler_preemptions_total[10m])) > 0
        and sum(tetraeder_kv_blocks{tetraeder_state="free"}) / sum(tetraeder_kv_blocks) > 0.25
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "KV pool fragmented: preemptions with more than 25% of blocks free"
  ```
  The preemption counter is C2's; if C2 has not merged, add the counter name string to `metrics.rs` in the same form so `tests/test_alert_rules.py` passes, and note it.

**Tests first.**
- `crates/kvcache/tests/pool_props.rs` (proptest, both layouts, blocks 64 and 256): random reserve, append, `seq_rm`, `seq_cp`, moves: P2 (live sequences' rows disjoint; after `seq_cp` dst rows are its own), P3 (reserved never above capacity, `Full` changes nothing, every planned row lies inside its sequence's reservation), mask rows open exactly on the sequence's own positions up to the token's, moves preserve every position's content (simulate row contents in a Vec).
- Unit: the one-sequence equivalence above; c4 sanity pool 3072 rows on both layouts.
- `crates/kvcache/tests/layout_sim.rs` (CI, deterministic): workload W of ADR 0067 from the committed proxy JSONL (dev-dependency `serde_json`; conversations by the rule of `scripts/claude_code_cache_classes.py`; Wt and R are the constants in the ADR, no model file is read), bisection per layout, writes `sim.json` to `TETRAEDER_KV_SIM_OUT` when set; asserts only internal consistency (W completes at the found pool, not at one unit less).
- `crates/model/tests/kv_gather.rs` (ignored, label `cpu`): the three variants of ADR 0067 at the stated shapes, median of 20, `kv-gather.json` to `TETRAEDER_KV_GATHER_OUT`; first asserts on small sizes that (ii) and (iii) give the same attention output as (i) within 1e-3.

**Done.** Workspace checks, `tests/test_alert_rules.py`; run the simulation and the benchmark locally, commit both JSON files under `bench/results/<date>/kv-layout/` with the computed `saving`, `y_mask`, `y_gather`, `y_sparse` and `verdict` (`chunked`, `paged_unmodified` or `step2`, by ADR 0067's four outcome rules), and write ADR 0067's addendum with the numbers and which outcome rule applied. If the verdict is `step2`, stop there: step 2 gets its own ADR and brief.

**Must not change.** The thresholds X and Y or the workload definition (ADR 0067: fixed before measuring). Anything in `crates/model/src` or `crates/scheduler`.

**Report.** The three numbers, the verdict, the CPU and thread count.

---

## Brief C5: own Qwen3 dense builder

**Goal.** ADR 0066: a Qwen3 dense builder in Rust on `model::attn` and `kvcache`, multi-sequence, served by `serve --engine dense`, with parity against libllama.

**Read first.** ADR 0066 (all), ADR 0065 (backend trait, attention), C1.5's `attn.rs`, C6-1's `kvcache` API, `qwen3moe.rs` (`load`, `load_weights`, `build_step`, `head`, `finish_step`, `report_memory`), `vendor/llama.cpp/src/models/qwen3.cpp` and `build_ffn` in `src/llama-graph.cpp` (2040: `swiglu_split`), `xtask/tiny_moe.py` and `xtask/src/main.rs` (`tiny_moe`), `crates/model/tests/common/mod.rs`, `crates/engine/tests/golden.rs`.

**Decisions.**
- `crates/model/src/qwen3.rs`, re-exported as `model::{Qwen3, DenseHparams, DenseOptions}`:
  ```rust
  pub const ARCH_DENSE: &str = "qwen3";
  pub struct DenseHparams { pub attn: AttnHparams, pub n_ff: u32, pub tied_output: bool, pub name: String }
  impl DenseHparams { pub fn from_gguf(g: &Gguf) -> Result<Self, GgufError> }   // qwen3.* keys
  pub struct DenseOptions { pub backend: BackendChoice, pub n_ctx_pool: usize, pub n_seq_max: usize,
      pub kv: KvFormat, pub kv_rotation: Option<bool>, pub threads: Option<i32>, pub reuse_step_graph: bool }
  impl Qwen3 {   // SeqEval is model::SeqEval from C1.5
      pub fn load(path: impl AsRef<Path>, opts: &DenseOptions) -> Result<Qwen3>;
      pub fn n_vocab(&self) -> usize;
      pub fn n_seq_max(&self) -> usize;
      pub fn pool(&self) -> &KvPool;
      pub fn seq_reserve(&mut self, seq: u32, n: usize) -> Result<()>;   // applies moves
      /// One step; logits rows per input into `out`, position-major; returns each input's row range.
      pub fn decode(&mut self, inputs: &[SeqEval<'_>], out: &mut Vec<f32>) -> Result<Vec<std::ops::Range<usize>>>;
      pub fn seq_rm(&mut self, seq: u32, keep: usize) -> Result<()>;
      pub fn seq_cp(&mut self, src: u32, dst: u32, n: usize) -> Result<()>;   // attn::copy_rows
      pub fn seq_snapshot(&self, seq: u32) -> Result<Vec<u8>>;   // rows via KvPool::runs
      pub fn seq_restore(&mut self, seq: u32, bytes: &[u8], n: usize) -> Result<()>;
      pub fn kv_format(&self) -> KvFormat;
      pub fn memory(&self) -> Vec<MemoryUse>;
      // single-sequence conveniences on seq 0, for tests and spec: reset, eval, eval_last, rewind
  }
  ```
  Errors reuse `ModelError`. `Hparams` and `ModelFile` stay MoE-only.
- Graph exactly as ADR 0066 (whole step, `swiglu_split`, last layer `get_rows` of `StepPlan::outputs` after attention, then residual, FFN, output norm, head). If `ggml-sys` lacks `swiglu_split`, add a safe wrapper in `crates/ggml-sys/src/ggml` with its safety comment (the only ggml-sys change allowed).
- Chunked pool from `kvcache` (`Chunked::new(n_ctx_pool)`); per step: `plan_step`, write `pos`, `rows`, `mask`, `outputs` inputs, build or reuse the graph keyed by (n_tokens, n_kv, n_outputs), compute, read logits, `commit`.
- `crates/scheduler/src/dense_backend.rs`: `pub struct DenseBackend` implementing `BatchBackend` over `Qwen3`; `load.rs` gains `load_dense(path, opts) -> Result<Loaded, String>` (tokenizer from a `vocab_only` libllama load, as `load_moe`); `serve --engine dense` (auto unchanged).
- `xtask/tiny_dense.py` and `cargo xtask tiny-dense` writing `tests/fixtures/tiny-qwen3-f16.gguf` with ADR 0066's shape; commit the fixture.
- Metrics: the existing families through `finish_step`-style reporting with `tetraeder.builder = dense`, and C6-1's `kv_blocks`.

**Tests first.**
- `crates/model/tests/common/mod.rs`: `pub trait Evaluator { fn reset(&mut self); fn eval(&mut self, t: &[i32]) -> Vec<f32>; fn eval_last(&mut self, t: &[i32]) -> Vec<f32>; fn n_vocab(&self) -> usize; }` for `Qwen3Moe` and `Qwen3`; `check_against(m: &mut dyn Evaluator, ..)`; existing callers compile unchanged in behaviour.
- `crates/model/tests/dense_parity.rs` (cpu): tiny dense fixture against `libllama_reference_at` within `TOL_CPU`, top-1 equal; KV formats F16, Q8_0, Q8_0/Q4_0 with rotation off; multi-sequence (3 sequences, a prefill chunk mixed in) against each alone and against libllama (`n_seq_max = 3`, unified) within `TOL_CPU`, greedy equal or a near-tie per ADR 0043; `seq_cp` continuation; snapshot roundtrip.
- `crates/model/tests/dense_rotation.rs` (cpu, own binary): rotation on, against libllama's default.
- `crates/model/tests/dense_golden.rs` (cpu, model): Qwen3-0.6B Q8_0 against `bench/golden/qwen3-0.6b-q8_0-cpu-sse42-<os>.json` with `golden.rs`'s tolerance and token equality; skips without the model unless `TETRAEDER_REQUIRE_MODELS=1`.
- `crates/model/tests/gpu_dense.rs` (`gpu_` names, `TETRAEDER_REQUIRE_GPU`): tiny fixture on CUDA against the CPU reference within `TOL_GPU`, one and three sequences; Qwen3-8B Q4_K_M top-1 and 64 greedy tokens against libllama when `TETRAEDER_MODELS_DIR` has it.
- Graph test: one layer's op names in libllama's order.
- `crates/scheduler/tests/dense_serve.rs`: four concurrent requests through `Scheduler` on the tiny dense fixture finish with batch size above 1; P1 holds over a short proptest.

**Done.** Workspace checks; all CPU tests above pass; GPU tests compile and skip.

**Must not change.** `qwen3moe.rs`, `attn.rs` (report if it needs a change instead of making it), `kvcache`, `serve --engine auto`.

**Report.** Max logit deltas per test, the golden result, fixture size.

---

## Brief C4: MoE builder on the pool, multi-sequence

**Goal.** `Qwen3Moe` serves several sequences in one step on `kvcache`'s chunked pool, resident and streamed, so the 30B counts in the Phase 4 verdict (ADR 0065); one sequence stays bit-equal to today.

**Read first.** ADR 0065, ADR 0067 (adoption), C1.5's `attn.rs`, C6-1's API, `qwen3moe.rs` (`run`, `State`, `alloc_state`, `head`, `rewind`, `kv_snapshot`, `kv_restore`, `finish_step`), `crates/scheduler/src/{backend.rs,spec/target.rs}`, `crates/model/tests/{parity.rs,slots.rs,kv_format.rs,kv_snapshot_decode.rs,step_reuse.rs}`.

**Decisions.**
- `LoadOptions` gains `n_seq_max: usize` (default 1); `n_ctx` is the pool's rows. The KV is `attn::Kv` with `n_rows = n_ctx` plus a `KvPool` over `Chunked`.
- `State` gains `rows: OwnedTensor` (I32, n_tokens) and `outputs: OwnedTensor` (I32); `AttnStep.rows` is the rows tensor now; `head` selects `outputs` with `get_rows` at the point where it now takes the last-row view (after the output norm), so one sequence with one output is bit-equal.
- New methods with the same signatures as C5's `Qwen3` (`seq_reserve`, `decode(&[SeqEval], &mut Vec<f32>) -> Result<Vec<Range<usize>>>`, `seq_rm`, `seq_cp`, `seq_snapshot`, `seq_restore`, `n_seq_max`, `pool`); `SeqEval` is `model::SeqEval` from C1.5. `eval`, `eval_last`, `rewind`, `reset`, `kv_snapshot`, `kv_restore` become sequence-0 wrappers with unchanged behaviour (spec, exit harness and tests use them).
- Streaming, pair groups, overlap and prefetch see more tokens per step and nothing else; the routing log records tokens in step order.
- `scheduler::MoeBackend` implements `BatchBackend` natively; `SingleSeq` and the old `Backend` trait are deleted, with `LlamaBackend`'s old impl.

**Tests first.**
- C1.5's `extract_bits.rs` check passes against a record made before this step (one sequence, bit-equal).
- `crates/model/tests/multi_seq.rs` (cpu): 3 sequences batched with a prefill chunk mixed in, resident and streamed (`MissPolicy::Cpu`), against each alone within `TOL_CPU` and greedy equal or near-tie; against libllama (`n_seq_max = 3`, unified) on the tiny fixture; `seq_cp` continuation; per-sequence snapshot roundtrip; a chunked move mid-decode keeps logits within `TOL_CPU` of the unmoved run.
- `crates/model/tests/gpu_multi_seq.rs` (`gpu_`): the same on CUDA within `TOL_GPU`.
- `crates/scheduler/tests/spec_builder.rs` passes unchanged.

**Done.** Workspace checks; existing `crates/model` tests unchanged and passing.

**Must not change.** Single-sequence numerics; `LoadOptions` defaults; the exit harness's environment knobs; `attn.rs` and `kvcache` (report instead).

**Report.** Deltas, any streaming code path that needed more than token-count changes.
