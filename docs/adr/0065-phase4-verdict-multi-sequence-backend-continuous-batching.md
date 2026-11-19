# 0065. Phase 4: verdict rules, a multi-sequence backend and continuous batching

Status: Accepted (2026-11-17). The Phase 4 exit test passes: throughput for both counting models in `20261117k`, TTFT in `20261117l`. Earlier D2a sessions: `20261116d2a2` a c4 miss, `20261117d` a provisional pass on an unmerged branch, `20261117e` and `20261117g` null on throttle, `20261117h` probes stopped (ADR 0074). Proposed 2026-11-14. Phase 4. Supersedes, from ADR 0046, the one-sequence backend trait, the scheduler loop and "the gateway's semaphore stays at one request in generation"; ADR 0049's RAM prefix cache stays and gains resident sequences (step C3). Companions: ADR 0066 (own Qwen3 dense builder), ADR 0067 (KV layout experiment); the engine pin is ADR 0073's. Briefs: `docs/phase4-briefs.md`.

Measurements live in the session READMEs (`bench/results/<session>/README.md`), not here. Decisions made after the first draft are folded in below; "History" lists when each came and why.

## Verdict

**Throughput.**
- **Judged:** `tetraeder serve --engine dense` (own builder, ADR 0066) on Qwen3-8B Q4_K_M. **Control:** `--engine llama` (libllama with step C2's batching) on the same file, same session; reported, never judged.
- **Qwen3-30B-A3B Q4_K_M** counts only on the own MoE builder (`--engine moe`) after step C4 gives it multi-sequence KV; then it must pass too. Before C4 it runs `--engine llama`, is recorded with `counts: false` and does not count either way.
- **Baseline:** llama-server of the baseline build (`/opt/llama.cpp-baseline`; `vendor/PIN` `baseline_tag` b12426, `baseline_commit` fc504f91, since 2026-11-16; before that b12274, 202c1a22, after #30966): `-m F -ngl 99 -np 4 -c 32768 -fa on -ctk f16 -ctv f16 -t T`, continuous batching on (its default). The KV mode is tuned per session: a pre-pass runs c4 once with `-kvu` and once with `-no-kvu`; the higher `throughput_tps` is the session's baseline (`baseline_variant`). An explicit `-np` leaves `kv_unified` false (`common/common.h:582`; only auto `-np` sets it, `tools/server/server.cpp:127`; read at b11976, every flag and the log line rechecked at b12426); the result records the server's `kv_unified = '...'` log line, so the baseline's behaviour is read, not assumed. Results before and after a baseline move are separate baselines; pairs are formed within one session.
- **Tetraeder:** `tetraeder serve -m F --engine E --ngl -1 --parallel 4 --ctx 32768 -t T`, F16 KV, flash attention (the builders always use it; libllama with `--flash-attn on`). Same `T` for both engines (AGENTS.md pairing rules).
- **Workload:** `scripts/gateway_bench.py`, temperature 0, `--seed <rep>` equal for both engines of a pair (identical prompts). c4: `--prompt-tokens 512 --max-tokens 128 --concurrency 4 --requests 16`. c1: the same with `--concurrency 1 --requests 8`. A run is one server start: warm-up, c1, c4. (2026-11-15: each test of a server start has its own seed, c1 `rep*10+1` and c4 `rep*10+4`, equal across engines for the same test and rep; one seed for both tests made llama-server answer repeated prompts from its prompt cache, most of a 0.60 gap on CPU (No session directory yet.). Every gateway_bench call passes `--no-keepalive`.)
- **Gating fields:** c4 `summary.throughput_tps` (completion tokens over wall time, what ADR 0046 reported); c1 `summary.decode_tps_median`. Pair ratio r = Tetraeder / baseline.
- **Pairs:** two per model on one box in one session, interleaved: (baseline, Tetraeder), then (Tetraeder, baseline). The control runs after each pair. (2026-11-15: if a pair is unclean and the time budget allows, one more pair runs as rep 3; the verdict uses the first two clean pairs.)
- **Band b:** `band` of `bench/noise/<gpu-class>.json` if its `n` is at least 3, else the spec band (`bench/spec/phase0.toml`, class `cuda-default`, 0.08). Recorded as `band` and `band_source`.
- **Clean run:** `summary.errors` is 0; `scripts/cotenant_guard.py` does not say false; the throttle record passes (below). **Clean session:** the two baseline runs of a model agree within the band on both fields, |B2 / B1 - 1| <= b. A pair is clean when both runs are, the session is, and the engines' completion-token totals are equal.
- **Throttle record** (`scripts/phase4_summary.py`): nvidia-smi clock event reasons before and after each run, with power limit and draw; `verdict.json` keeps the raw values. Before a run only idle and the applications-clock setting (0x3) are accepted; after a run SW power cap (0x4) too. With session sampler data (`scripts/gpu_sampler.sh`, 1 Hz for the whole session: reasons, power draw, SM and memory clocks) the measured window of each run decides: any sample carrying a bit outside 0x1, 0x2, 0x4, 0x100, 0x200, 0x400 (thermal, hardware slowdown, power brake 0x8, 0x20, 0x40, 0x80, or unknown) makes the run unclean; BoardLimit and Reliability (0x200, 0x400) are judged by the share of window samples that carry them, and a pair is clean on them when its two sides' shares differ by at most 10 percentage points; the before sample may then also carry 0x200 and 0x400. A run without sampler data falls back to the before and after samples alone.
- **Pass,** per counting model and per field: both pairs clean and min over pairs of r >= 1 - b. "At least llama-server" means not slower beyond the noise band; a tie inside the band passes, as the owner ruled for Phase 3b (ADR 0046 addendum 2026-11-11). Today the A40 and A6000 class files hold one session each, so the spec band applies there and r >= 0.92 passes; the RTX 3090 file (11 sessions) gives 0.075. An unclean pair is rerun, never dropped to make a pass; fewer than two clean pairs per counting model is no verdict. (No session directory yet.)
- **Gates and probe,** before the pairs (`scripts/phase4_box.sh`): `gpu-parity` per model (including batched parity on the real models, `crates/model/tests/gpu_real_batched.rs`: three sequences with a prompt chunk mixed into decodes, at most 64 tokens per step, builder against libllama batched on CUDA within `GPU_TOL`, ADR 0071; `gpu_qwen3_8b_matches_libllama` checks one sequence only), `gpu-shapes` (`step_shapes.rs` on Qwen3-8B, engines llama, llama_streams, dense; records only, its data decides the unified pool against streams question under "Attention over sequences of unequal length"), then a probe: one short c1 and c4 (`--requests 8`) per judged engine against llama-server with the verdict flags. A failed parity or a probe c4 ratio below `PROBE_MIN` (default 0.85: the pass line is about 0.925, and below 0.85 the pairs cannot pass) stops that model: its pre-pass, probe and pairs are skipped and `verdict.json` records it with `pass: null`, `stopped: true` and the reasons; it still counts, so the overall verdict stays null. The other model's pairs and the TTFT pairs still run; the session ends (rc 3) only when every judged model is stopped. One probe round per model per session: re-probing until one passes would select on noise. Every D2a session from 2026-11-17 on ran with `PROBE_MIN=0.925`, the pass line (`threshold_c4` in `probe.json`), since below it the pairs cannot pass and the box time is better spent elsewhere.
- **Sessions:** each counting model is judged by its own two clean pairs on one box; the two models need not share a session. The throughput verdict may come from one session per model (each with its own gates, pre-pass and probe) and the TTFT pairs from a further session; `phase4_pass` combines the per-model results only when every counting model has a verdict.
- **Box choice:** idle gate at SSH (`scripts/idle_gate.sh`): 10 samples over 30 s; launch only if at least 9 carry no disqualifying bit (anything outside 0x1, 0x2, 0x4, 0x100, 0x200, 0x400). 0x4 is allowed because driver 615 reports it at 35 W idle, where it caps nothing (No session directory yet.); the sampler judges the measured windows.

**TTFT.**
- **Log:** one Claude Code request log (`serve --request-log`, ADR 0046) recorded once by `scripts/claude_code_proxy.sh` over 5 tasks, 1 rep. It holds prompts in clear text, so it stays out of git (`bench/private/`, gitignored) and is uploaded to each box; results carry its `log_sha256` and request count.
- **Server:** the judged engine, Qwen3-8B, `--parallel 4 --ctx 40960`. Reuse on: defaults (live, resident-sequence and RAM reuse, `--cache-ram 8192`). Reuse off: `--no-prefix-reuse` (new in C3: no prefix reuse of any kind) with `--cache-ram 0`.
- **Replay:** `scripts/gateway_replay.py --speed 0 --concurrency 1 --max-prompt-tokens 36000` (new flag: entries whose logged `usage.prompt_tokens` exceed it are skipped, so every request fits the 8B's context of 40960, its `qwen3.context_length`; the script reads the key from the GGUF and stops if it differs).
- **Fields:** `summary.ttft_median_s` and `summary.ttft_p90_s` (new), over the requests with status 200 in both runs of the pair; `summary.errors` equal in both runs.
- **Pairs:** two, interleaved (on, off) then (off, on), one box; clean as above, without the A/A clause; with sampler data the two sides (reuse on against off on one engine) are judged as a pair.
- **Pass:** in each pair median(on) / median(off) <= 1 - b and p90(on) / p90(off) <= 1 - b. Informative only: llama-server (`-np 4 -kvu --cache-ram 8192`) on the same log.

**Files** (`bench/results/<session>/phase4/`): `bench-<model>-<engine>-<test>-r<rep>.json` (gateway_bench.py), `prepass-<model>-<kvu|nokvu>.json`, `replay-<model>-<on|off>-r<rep>.json` (gateway_replay.py), `gpu-<run>.txt`, `server-*.log`, `env.json`, and `verdict.json` from `scripts/phase4_summary.py`: `{band, band_source, models: {<model>: {engine, counts, baseline_variant, baseline_kv_unified, aa: {c4, c1}, pairs: [{reps, c4_throughput_ratio, c1_decode_ratio, clean, reasons}], worst: {c4, c1}, pass}}, ttft: {log_sha256, requests, pairs: [{median_ratio, p90_ratio, clean, reasons}], pass}, phase4_pass}`. `phase4_pass` is true when every counting model and `ttft` pass. Engine names in files: `llama-server`, `tetraeder-dense`, `tetraeder-moe`, `tetraeder-llama`.

**Settled 2026-11-14.**
- Pass rule r >= 1 - b: the owner's choice ("not slower beyond noise").
- c4 gates on `throughput_tps`, the exit test's "throughput", which includes prefill as ADR 0046 measured; decode-only rates are reported, not gated.
- Workload size stays 16 c4 requests: a c4 run is short and its two repeats agree closely (`20261103`).
- c1 is not the risk: there Tetraeder already matched llama-server (`20261103`); the gap is c4 only, which is what batching addresses.
- Recording the Claude Code request log is the first stage of the verdict box session (`docs/m2-plan.md`, D2), before any timed run.
- To read on the box and record, not assume: b12274's `kv_unified` with `-np 4` (the pre-pass covers both) and Qwen3-8B's `qwen3.context_length` (40960; the harness stops if it differs).

## Context

The plan's Phase 4 exit test: at least llama.cpp server throughput at concurrency 4; prefix hits cut time to first token (TTFT) on a Claude Code replay. ADR 0046 measured one sequence at a time: at concurrency 4 Tetraeder reached well under llama-server `-np 4` throughput on both models, and TTFT was several times longer (`20261103`).

What exists: `scheduler::Backend` holds one sequence (`n_past`, `truncate`, `eval`, `snapshot`, `restore`). The libllama engine uses sequence 0 only (`memory_seq_rm(0, ..)`); `ggml_sys::llama::Batch` already carries a sequence id per token, but `ContextParams` has no `n_seq_max` or `kv_unified` and there is no `seq_cp` binding. The MoE builder (`crates/model/src/qwen3moe.rs`) has one contiguous KV per layer (`[n_embd_kv, n_ctx]`), a mask `[n_kv, n]` and writes K and V with `set_rows` at the positions.

The M1 milestone lost weeks to disputes over which configuration counts, whether a baseline was tuned and whether a session was clean (`docs/m1-open.md`). So the verdict above is fixed before any code.

## Decision

### Attention over sequences of unequal length

At the pin, `ggml_flash_attn_ext` takes q `[d, n_tok, n_head, ne3]`, k and v `[d, n_kv, n_head_kv, ne3]` and an F16 mask `[n_kv, n_tok, ne32, ne33]` (`ggml/include/ggml.h:2493`). Two ways to batch sequences:

- **(a) One pool with a mask** (llama.cpp unified KV): one K and one V tensor per layer `[n_embd_kv, n_rows]`; all tokens of a step are one Q (ne3 = 1); mask row t is 0 on rows of t's sequence at positions up to pos(t), minus infinity elsewhere. Any mix of token counts per sequence fits one op. The CPU kernel skips a masked row before its dot product (`ggml-cpu/ops.cpp:8944`, `:9209`), so it reads only rows a query may see. CUDA reads `[0, n_kv)` per query tile and trims only trailing fully masked blocks (`flash_attn_mask_to_KV_max`, `ggml-cuda/fattn-common.cuh:671`).
- **(b) A buffer per sequence** (llama.cpp non-unified): K `[n_embd_kv, kv_size, n_stream]`, ne3 = stream, one mask per stream. Each stream reads only its rows, but every stream has the same token count in one op (q's ne1 is shared; llama.cpp uses `split_equal`, `src/llama-kv-cache.cpp:705`), capacity per sequence is fixed at kv_size, and `seq_cp` across streams copies a whole stream buffer (`llama-kv-cache.cpp:820-845`).

What reaches llama-server's batched throughput is one graph per step for all sequences, so the weights are read once for four tokens. Attention is small at the exit workload: 36 layers x 640 positions x 4 KiB of K and V per row is 94 MB per step against 5.02 GB of Qwen3-8B Q4_K_M weights (5,021,827,072 bytes, `model_size` in `bench/results/20261103/gateway/overhead-q8b-llama-bench.json`). Both layouts batch decode; they differ when prefill and decode mix. (a) puts a 512-token prefill chunk and three decode tokens in one op and one graph. (b) needs equal counts per stream, so the step becomes several graphs (llama.cpp splits into ubatches) or padded queries. (a) pays on CUDA for rows below n_kv that the step's sequences do not own; when every live sequence decodes, every row is needed anyway.

**Decision: (a).** One pool per layer, one `flash_attn_ext` per layer per step, viewing rows `[0, n_kv)` with n_kv = pad(highest row the step reads + 1, 256). The graph key stays (n_tokens, n_kv), so CUDA graph reuse works as in ADR 0036. Which rows a sequence gets is the KV layout's job (`kvcache`, ADR 0067): chunked or paged changes the rows, the mask and the `set_rows` indices, never the op, so the trait and both builders hold either layout. The libllama backend runs llama.cpp's own cache with `kv_unified = true`, the same scheme. A row belongs to at most one sequence; `seq_cp` copies rows. llama.cpp's shared cells (one row visible to several sequences) save memory but make writes and `seq_rm` reference-counted; that waits for paged blocks (ADR 0067). To verify on the box: CUDA flash attention with mixed sequences in one query tile matches the CPU (gpu parity tests of C4 and C5).

Not adopted, so (a) stays the verdict path: attention per sequence (`DenseOptions::per_seq_attn`, `KvPool::plan_step_per_seq`, `attn::attention_seqs`: one `flash_attn_ext` per input over its own chunked run, `n_kv` rounded to 256 within its reservation) is built and bit-identical to each sequence alone, but it missed the threshold fixed before measuring (8% faster on both CPU decode shapes), so the flag stays off. One KV stream per sequence (llama-server's layout) is not adopted either: streams would cap each sequence at `n_ctx / parallel`.

### Backend trait (`crates/scheduler/src/backend.rs`)

```rust
pub type SeqId = u32;

pub struct SeqInput<'a> {
    pub seq: SeqId,
    /// Appended at positions seq_len(seq).. (never empty).
    pub tokens: &'a [i32],
    /// Logits of every token (speculative verification), else of the last one.
    pub all_logits: bool,
}

pub enum BackendError {
    /// Not enough KV rows; nothing changed.
    KvFull { seq: SeqId, needed: usize, free: usize },
    /// The step failed; every sequence in it is back at its length before the step.
    Failed(String),
}

/// Logits of the last decode, borrowed until the next call.
pub struct Outputs<'a> { /* n_vocab, data: &'a [f32], spans: &'a [std::ops::Range<usize>] */ }
impl<'a> Outputs<'a> {
    /// Rows of input i, position-major: 1 row, or tokens.len() rows with all_logits.
    pub fn rows(&self, i: usize) -> &'a [f32];
    pub fn last(&self, i: usize) -> &'a [f32];
}

pub trait BatchBackend {
    fn n_vocab(&self) -> usize;
    fn n_seq_max(&self) -> usize;
    /// Most positions one sequence may hold.
    fn n_ctx_seq(&self) -> usize;
    /// KV rows in the pool, and rows not reserved by any sequence.
    fn kv_capacity(&self) -> usize;
    fn kv_free(&self) -> usize;
    fn seq_len(&self, seq: SeqId) -> usize;
    /// Guarantees rows for `seq` to grow to `n` positions. KvFull changes nothing.
    fn seq_reserve(&mut self, seq: SeqId, n: usize) -> Result<(), BackendError>;
    /// Keeps the first `keep` positions; 0 frees the sequence and its reservation.
    fn seq_rm(&mut self, seq: SeqId, keep: usize) -> Result<(), BackendError>;
    /// `dst` (empty) becomes the first `n` positions of `src`, in rows of its own.
    fn seq_cp(&mut self, src: SeqId, dst: SeqId, n: usize) -> Result<(), BackendError>;
    /// One step: each input's tokens appended to its sequence (sequences distinct).
    fn decode(&mut self, inputs: &[SeqInput<'_>]) -> Result<Outputs<'_>, BackendError>;
    fn seq_snapshot(&mut self, seq: SeqId) -> Result<Vec<u8>, BackendError>;
    fn seq_restore(&mut self, seq: SeqId, bytes: &[u8], n: usize) -> Result<(), BackendError>;
    fn kv_format(&self) -> String;
    fn on_gpu(&self) -> bool;
    fn bytes_streamed_total(&self) -> Option<u64>;
    fn counters(&self) -> Vec<(&'static str, f64)>;
}
```

- Positions are implicit: a sequence only grows at its end or is cut by `seq_rm`, as ADR 0046's `truncate`.
- Added while building: `max_step_tokens() -> usize` (default `usize::MAX`), `kv_row_bytes() -> usize` (default 0; sizes the copy in the `kv` transfer metric and the snapshot size hint, so every backend reports it), `seq_reserved(seq)` (default the length; lets the scheduler resync its reservation after a prefix is restored or cut). A failed step or `KvFull` puts every input sequence back to its length and reservation.
- libllama: `n_seq_max` and `kv_unified = true` in `ContextParams`, `n_batch` = the step budget, `n_ubatch` = `--ubatch`, the compute shape (both were `step_tokens` until 2026-11-16); `seq_rm` is `llama_memory_seq_rm(seq, keep, -1)`, `seq_cp` is `llama_memory_seq_cp(src, dst, 0, n)` (in one stream llama.cpp only adds dst to the cells' sequence sets, `llama-kv-cache.cpp:470-497`; the adapter still counts n reserved rows for dst, which over-counts, never under-counts). Logits per output come from `llama_get_logits_ith(batch index)` into one kept buffer. Reservations are the adapter's bookkeeping against `n_ctx`: unified cells are found one by one (`find_slot(ubatch, false)`, `:761`, `:1011`), so a reservation under capacity cannot fail on fragmentation. libllama's `NoKvSlot` maps to `KvFull`, every other decode error to `Failed`.
- libllama as built: `n_ctx` is rounded up (to a multiple of 256 with the unified cache) and `n_batch` is clamped to the requested `n_ctx`; `max_step_tokens` is `n_batch`. A step whose cells do not fit fails with `NoKvSlot` and the engine cuts every input back with `seq_rm`, lengths unchanged; `KvFull` names the input that grew most past its reservation (libllama cannot say which one found no cells). `Engine::seq_rm(seq, 0)` always clears the sequence's cells, also when `seq_len` is 0 (a restore that fails partway leaves cells the length does not count), and rollback uses `Engine::seq_truncate_cells`, which cuts even when the length already equals the target. `kv_row_bytes` is learned from the first snapshot, so copies before it record 0 bytes.
- `MoeBackend` reached the loop through a `SingleSeq` adapter until C4; since C4 it implements the trait natively, and `SingleSeq` and the old `Backend` trait are gone.

### Scheduler loop (`crates/scheduler/src/scheduler.rs`, step planning in `batch.rs`)

- One worker thread owns the backend (as ADR 0046). Each iteration: take new jobs, admit, plan one step, decode, sample, emit, retire. It blocks on the job queue only when nothing is live.
- **Admission:** first in, first out. A job is admitted when a sequence id is free (`--parallel`, default 4) and `seq_reserve(seq, prompt + min(max_tokens, GROW))` succeeds, GROW = 1024. Prefix reuse picks where its first rows come from (C3). The gateway's `max_concurrent` becomes `--parallel` (was 1); `max_queued` and its 503 stay.
- **Prefix sources,** longest wins: an idle sequence, a copy of a busy one (`seq_cp`), a RAM entry. A busy copy is taken only when it beats the idle sequence and the RAM entry by `cache_restore_gain` (a tie goes to the cheaper idle or RAM source). Reuse is at most prompt - 1 tokens. Hits carry source `seq`; each copy records a VRAM to VRAM `kv` transfer. A failed busy copy releases the id and places the request again without the busy source (`Req.no_busy`). `SchedulerConfig::prefix_reuse` (`serve --no-prefix-reuse`, default on): off means no idle, busy or RAM reuse and no saves; `--cache-ram` is independent.
- **RAM entries:** an entry chosen for a restore is taken out of the cache before any idle sequence is evicted (the eviction may drop it) and put back when admission has to wait. An entry whose restore fails with `KvFull` goes back into the cache; one that fails otherwise is dropped, since its bytes would fail again.
- **Admission checks:** token ids are checked against `n_vocab`; an out-of-range one fails only that request. A requeued request that already holds `n_ctx_seq` tokens and has generated some ends with `Length`, as it would have live. The effective parallelism is `min(--parallel, backend.n_seq_max())` (`--plan` serves one sequence); `serve --engine moe` honours `--parallel`.
- **Step plan** under `min(--step-tokens, max_step_tokens)` (`--step-tokens` default 2048, llama-server's `n_batch`; libllama gets it as `n_batch`, `--ubatch` stays the compute shape), never below `--parallel` (decoders run beyond the budget; `--parallel` is capped by `n_seq_max` and `max_step_tokens`): every decoding, unpaused sequence gets 1 token (1 + k with drafts); the rest of the budget goes to prefilling sequences in admission order, each its next chunk up to what is left. One step is one `decode` call, one graph on the builders and one `n_batch` call on libllama (which runs it as ubatches of `--ubatch`). A wave of prompts that arrive together is prefilled in one step instead of one per step next to the decoders.
- **Growth and preemption:** a sequence about to pass its reservation calls `seq_reserve(len + GROW)` before the step. On KvFull idle resident sequences go first (oldest first, saved to RAM per ADR 0049); if that is not enough, the most recently admitted other live sequence is preempted: snapshot into the RAM prefix cache if it holds at least `SAVE_MIN_TOKENS` (ADR 0049), freed, and its job goes back to the queue front with prompt plus generated tokens as its prompt, keeping its sampler, constraint and channel (nothing is sent twice). With no other sequence the request finishes with `Length`, as a full context does today.
- If the most recently admitted other sequence is older than the one that needs rows, the needing sequence is requeued itself, so the oldest request always progresses.
- Not adopted: sequence ids beyond `--parallel` to keep finished sequences resident instead of saving them at admission. In the unified KV every resident row lengthens attention, so CPU decode fell as residents accumulated. Any replacement for the admission save must keep `decode1x8_resident` equal to `alone`.
- **Sampling:** per sequence its own `Sampler` and constraint, on the worker thread after the step, in sequence order.
- **Backpressure:** events go out with `try_send`; a sequence whose channel is full is paused (left out of steps, rows kept) until the channel has room. One slow client never stalls the batch.
- **Cancellation:** checked per sequence every step; a cancelled request or a closed receiver finishes `Cancelled`.
- **Retirement:** end of generation, `max_tokens`, full context or cancel send `Finished`; the sequence stays resident as an idle prefix (C3) until its rows or id are needed, then it is saved per ADR 0049's rules and freed.
- Events of a finished request that its channel does not take yet wait in a drain list; its sequence is idle at once.
- **Errors:** `Failed` fails the requests of that step with `GenEvent::Error` and frees their sequences; the others continue. `KvFull` never reaches a request.
- A step that fails with `Failed` is retried one sequence per step (lengths are unchanged by the rollback contract), so only the sequence the backend rejects fails.
- **Decode profile:** `eval_s` is the shared wall time of the steps the request took part in (concurrent requests each count a step in full) and `gpu_seconds` is a token share of those steps. Neither is the request's own cost; no gate may use per-request `gpu_seconds`.

### Speculative decoding (`spec/`)

`Target` and `Drafter` stay. `spec::SeqTarget { backend, seq }` implements `Target` over a `BatchBackend` (eval = one `SeqInput` with `all_logits`, rewind = `seq_rm`), so the existing spec tests run against batch backends; the loop does not call it. In the loop a sequence that drafts puts `[pending, drafts...]` into the shared step with `all_logits`, samples row i with its own `Sampler`, stops at the first draft it differs from, emits the chosen tokens and cuts the KV to `before + 1 + accepted` (less if the request ends inside the chain). A sequence drafts only when it is decoding, not paused, greedy and unconstrained, and at most `--spec-max-batch` (default 1) sequences are decoding: batching spends the bandwidth headroom speculation lives on, and ADR 0043's bitwise exactness holds only for one sequence. Drafts are capped by the remaining `max_tokens` and the context. `serve --spec ngram|draft:<gguf>`, `--spec-max-batch`, `--spec-max-draft` (default 6, adaptive); off by default until a paired box benchmark shows a gain (Rule 3, ADR 0043). Metrics are the existing `tetraeder.spec.*`. Exactness: greedy output equals plain on the tiny dense fixture for libllama and the dense builder, with n-gram and a draft model; on AVX-512 runners a verify step can differ in the last bits (ADR 0043, 0052), so the test allows a divergence only where the plain top-2 margin is below 1e-2 (1e-3 elsewhere).

### Backends as built

- **CPU steps:** on the CPU, `LlamaBackend`, `DenseBackend` and `MoeBackend` run a step's decoders (inputs under 64 tokens, ggml's flash-attention query tile) and prompt chunks as separate model calls, in input order; on a GPU a step stays one call.
- **MoE (C4):** `Qwen3Moe` has `LoadOptions::n_seq_max` (default 1) and the same `seq_*`, `decode`, `try_*` methods as `Qwen3`; `eval`, `eval_last`, `rewind`, `reset`, `kv_snapshot`, `kv_restore` are sequence-0 wrappers. One sequence stays bit-equal to before (`extract_bits`). Whole-step graph key and state key use the output count instead of `last_only`; `head` selects `outputs` with `get_rows` only when fewer tokens than the step has want logits. `MoeBackend::max_step_tokens` sets no cap (it was the 512-token chunk until 2026-11-16): a larger `decode` runs as model steps of the chunk, so the scheduler's budget decides how many waiting prompts share a step. Streaming code needed only the token count; the `n == 1` decode-only admission into the compute tier does not run for a batched decode step (a performance policy, not a correctness one). Batched, chunked and one-query steps take other kernel shapes, so a chunked move is checked against a run whose sequence starts where the move put it, and is bit-equal there.
- **MoE whole-step graph:** a resident MoE 4-token decode step is the whole-step graph (`build_step`, ADR 0036) on the first scheduler, keyed by (tokens, n_kv, outputs); `op_offload_min_tokens` (ADR 0044) applies only to streamed steps (segments). The graph cache holds two entries, each a step state plus its graph, most recent first; the second has its own scheduler (`sched_alt`, built on first use), because a graph cannot be re-allocated on another's allocation. Only on the CPU with all experts resident and graph reuse on, where the CPU split alternates decoders and a prompt chunk; a GPU runs one call per step and a second scheduler would double the compute workspace. The CPU hot set is built from the routing of a call that holds a prompt chunk (`Qwen3Moe::decode_group`: an input of at least `SHORT_INPUT` tokens, or more than one token without `all_logits`), not from a decoder group or a draft verify step.
- **Rollback:** `DenseBackend::decode` snapshots (length, reservation) of every input before the step and puts all of them back on any failure, also those of earlier groups that did not grow; `MoeBackend` does the same with `Qwen3Moe::rollback`. Both go through `KvPool::rollback`, which only shrinks, so a rollback cannot fail on a fragmented pool, and a sequence the step moved stays where it moved. A reservation that cannot be put back is an error, not silent. The dense backend shrinks a reservation to the rows it keeps.
- **Pool:** `KvPool::seq_cp` accepts a destination that holds no positions but already reserves rows (`Worker::place` reserves the target before the copy); the destination keeps at least its reservation. Chunked relocation asks for `need + GROW` rows when a gap holds them and `need` otherwise.
- **serve:** an explicit `--ngl` above 0 with no GPU is an error for the own builders; the default (`-1`) falls back to the CPU with a `tracing::warn!`. `--ctx` of the own builders is the shared pool's rows across all sequences, not a context per slot (llama.cpp's server splits it); one request can use all of it. The 30B counts on `--engine moe` by default (`Q30B_ENGINE`).
- **Harness:** the CPU rehearsal builds llama-server at `baseline_commit`, not the engine pin; `--engine moe` is rehearsed on the synthetic MoE of `moe_shapes.rs` (context 8192, a ChatML template). loop-bench passes `-fa on` to llama-server and measures `--engine moe` on the tiny synthetic MoE fixture, where only the loop is measured.

### Property tests, first

Over `FakeKv` (a `BatchBackend` with deterministic logits from a hash of the sequence's tokens, its own row accounting, a configurable capacity and injected failures; P2 and P3 over `kvcache`'s layouts are C6-1's pool tests), with proptest over arrivals, lengths, `max_tokens`, cancellations, slow consumers, KvFull and failures:

- **P1** every request ends in exactly one `Finished` or `Error`, and its sequence is freed.
- **P2** no row is shared without a copy: the row sets of live sequences are disjoint after every call, and after `seq_cp` the destination's rows are its own and hold equal contents.
- **P3** memory never exceeded: reserved rows never exceed capacity, and no decode writes outside its sequence's reservation.
- **P4** batching never mixes sequences: each request's tokens equal those of the same request run alone.
- **P5** progress: without failures every request finishes within a bound on steps (FIFO admission, prefill share of the budget).

Real backends, then: libllama with 4 sequences batched against each alone on the tiny fixture: greedy tokens equal and logits within `TOL_CPU`, not bitwise (batched matmul kernels differ from single-row ones, ADR 0043).

### Metrics, panel, alert (Rule 6)

- `tetraeder.scheduler.batch.size` now records sequences per step; `tetraeder.scheduler.queue.depth` (catalog name, gauge) waiting jobs.
- New: `tetraeder.scheduler.step.tokens` (histogram, attribute `tetraeder.kind` = decode, prefill, verify), `tetraeder.scheduler.sequences` (gauge, `tetraeder.state` = prefill, decode, paused, idle), `tetraeder.scheduler.preemptions` (counter).
- Panel "Scheduler: sequences, step tokens, preemptions" in `deploy/grafana/dashboards/overview.json` beside "Scheduler batch size".
- Alert `TetraederSchedulerPreempting` (`deploy/alerts/engine.yml`): more than 1 preemption per minute for 10 minutes (the KV pool is too small for the load).

### Steps, in order

| Step | What | Files | After |
|---|---|---|---|
| C1.5 | Extract attention, norms, RoPE and KV allocation from `qwen3moe.rs` into `crates/model/src/attn.rs`; logits bit-equal | `model/src/{attn.rs,qwen3moe.rs,lib.rs}`, `model/tests/extract_bits.rs` | nothing |
| C2 | Multi-sequence libllama, `BatchBackend`, the batching loop, gateway concurrency, metrics | `ggml-sys/src/llama/context.rs`, `engine/src/{engine.rs,config.rs}`, `scheduler/src/{backend.rs,batch.rs,scheduler.rs,lib.rs}`, `scheduler/tests/{fake/mod.rs,batching.rs}`, `tetraeder/src/serve.rs`, telemetry, dashboard, alerts | nothing |
| C3a | Verdict harness: replay p90, filters, log hash, `phase4_box.sh`, `phase4_summary.py` | `scripts/{gateway_replay.py,gateway_bench.py,phase4_box.sh,phase4_summary.py}`, `tests/test_phase4_summary.py`, `.gitignore` | nothing |
| C3b | Prefix reuse across sequences: resident idle sequences, `seq_cp`, `--no-prefix-reuse` | `scheduler/src/{scheduler.rs,prefix_cache.rs}`, `scheduler/tests/prefix_seq.rs`, `tetraeder/src/serve.rs` | C2 |
| C6-1 | `kvcache` layouts (chunked used, paged simulated), memory simulation, CPU gather benchmark (ADR 0067) | `kvcache/`, `model/tests/kv_gather.rs` | nothing |
| C5 | Own Qwen3 dense builder, multi-sequence (ADR 0066) | `model/src/qwen3.rs`, `xtask/tiny_dense.py`, `scheduler/src/dense_backend.rs`, tests | C1.5, C2, C6-1 |
| C4 | MoE builder multi-sequence KV on the pool | `model/src/qwen3moe.rs`, `scheduler/src/backend.rs` | C1.5, C2, C6-1 |

C1.5, C2, C3a and C6-1 share no files and run in parallel. C4, C5 and C3b run in parallel after them (C4 and C5 meet only in `scheduler`'s backend files, which are separate). The verdict session runs after C2, C3 and C5; the 30B counts once C4 is in.

## Consequences

- One trait and one loop for libllama and both builders; layouts and prefix sharing change behind `seq_*` without touching the loop.
- Batched output is not bitwise equal to unbatched output; greedy goldens are recorded at one sequence, batched checks use the tolerances and ADR 0043's near-tie rule.
- Mask attention on CUDA reads rows of idle sequences below n_kv. If that shows in decode at long contexts, the layout places live sequences low, or ADR 0067's step 2 makes attention read only finite mask entries.
- Sampling runs serially on the worker: about 0.1 ms per sequence greedy at 152K vocabulary, more with temperature (ADR 0046 addendum). If `tetraeder.scheduler.decode.stage{stage=sample}` shows it costing the c4 ratio, sampling moves to scoped threads per sequence. (No session directory yet.)
- The verdict reads only committed fields; a dispute about a session is settled by `verdict.json`, not by prose.

## History

One bullet per former addendum: its date and title, what changed, why.

- 2026-11-14, "C2b, the loop": as built: `max_step_tokens`, the needing sequence requeued when it is the youngest, effective parallelism, drain list, RAM entry taken out before eviction, decode profile, libllama `n_ctx` rounding. The decisions above stood.
- 2026-11-14, "C3b, prefix reuse across sequences": prefix sources and `cache_restore_gain`, `--no-prefix-reuse`, `kv_row_bytes`.
- 2026-11-15, "C4, the MoE builder on the pool": `MoeBackend` native, `SingleSeq` and the old trait removed; one sequence bit-equal; chunked moves checked against a run placed where the move put it.
- 2026-11-15, "review fixes after C3b and C4": token id check, one-sequence retry of a failed step, chunked relocation size, libllama `seq_rm(seq, 0)`, `seq_reserved`, RAM entry kept on `KvFull`, decode profile not a cost.
- 2026-11-15, "CPU steps and the loop-bench gap": decoders and prompt chunks as separate calls on the CPU; the loop itself was not the gap; one KV stream per sequence not adopted (it caps each sequence at `n_ctx / parallel`).
- 2026-11-15, "GPU gates before the verdict pairs": gpu-parity, gpu-shapes and a probe before the pairs, stop below 0.85, the 30B on `--engine moe`; the box is not kept for pairs that cannot pass.
- 2026-11-15, "review fixes after the c4gap changes": `KvPool::seq_cp` into a reserved destination (busy copies never worked on the own pools), retry without the busy source, rollback that only shrinks, `--ngl` and `--ctx` semantics.
- 2026-11-15, "third review follow-ups, MoE": two-entry MoE graph cache with `sched_alt`, hot set from prompt-chunk calls, MoE rollback over `KvPool::rollback`.
- 2026-11-16, "baseline moved to b12426": the baseline moved from b12274 (202c1a22) to b12426 (fc504f91) so the verdict uses the strongest current llama-server; it adds the server slot scheduler (#32154), the connection reuse fix (#30492) and CUDA FP8 (#28413, expected from the diff to remove the `cache_k_l0 (view)` abort of `20261115p1`); the moe-cache refactor (#32157) changed no flags and only the unused `-dt/--defrag-thold` is gone. It said "The engine pin stays at b11976"; `image_tag` became `b11976-ikb25c3fd2-blfc504f91`. ADR 0073 then moved the engine pin to b12426 as well (`image_tag` `b12426-ikb25c3fd2-blfc504f91`).
- 2026-11-16, "speculation in the loop, built": the in-loop draft and verify, `serve --spec`, off by default; the busy-reuse skip flag moved to `Req.no_busy`.
- 2026-11-16, "attention per sequence, prototyped and kept off": `per_seq_attn` built and kept off, it missed the threshold fixed before measuring; the CPU concurrency-4 gap on AMX runners was the weight placement, ADR 0072.
- 2026-11-16, "verdict harness gaps closed": CPU rehearsal at `baseline_commit`, batched parity on the real models in gpu-parity, unequal completion-token totals make a pair unclean, the stop rule per model with `pass: null` and `stopped: true`.
- 2026-11-16, "models judged in separate sessions": one session per model allowed; reason: the MoE gate was held by an open parity item (ADR 0071) that the 8B does not share.
- 2026-11-16, "D2a rerun misses at c4" (`20261116d2a2`): a c4 miss for both models, c1 level, the builders not the cause; it pointed at the loop and the pin (ADR 0073), and at the driver 615 throttle bits. The throttle rule was not loosened for that session.
- 2026-11-16, "step budget 2048, no spare sequence slots": step budget default 2048 and the MoE backend no longer capped at its chunk, because mixed prefill and decode steps cost more than a prompt alone; spare sequence ids not adopted (residents lengthen attention); the GPU cost of the admission save is measured first (ADR 0074; it cites this entry as "2026-11-16 addendum" for ruling out spare device room for a device-side copy).
- 2026-11-16, "SW power cap after a run is benign" (before the next verdict session): after a run 0x4 is accepted; thermal and hardware slowdown bits still make it unclean. Why: the after sample comes at the same point for every engine, so a cap left from the load is the same on both sides of a pair, and the RTX 3090 noise band already prices power capping (`20261116d2a2`, `20261116pinab`). Each run records power limit and draw. Bits before a run were then left to host choice (a box with an idle value outside 0x3 at SSH destroyed), replaced by the idle gate of 2026-11-17. Decided before the session it applies to; `20261116d2a2` stays a c4 miss.
- 2026-11-17, "one probe decides the pairs at 0.925" (before the next verdict session): the next D2a session ran with `PROBE_MIN=0.925` and one probe round per model, because the 30B sat around the line in `20261116pc`. That session ran only after a confirmation on the same box: the dense crash workload with zero CUDA errors (`20261116pc`, before the mask padding fix it crashed), `gpu_kv_save` and `gpu_multi_seq` passing, and the prefix save time measured (ADR 0074).
- 2026-11-17, "D2a throughput passes at 99988e3, provisional" (`20261117d`): both models passed, but on `track-save-thp`, never merged, whose huge-page advice stalled admission saves; the verdict of record had to come from a full session on the commit that merges.
- 2026-11-17, "D2a on afda512 unclean on throttle" (`20261117e`): null, every run had 0x400 or 0x600 before it, under the rule fixed before the session; not rerun on that host. Its driver note ("not on 595") and the procedure "outside 0x0-0x3 before a run, destroy the box" were replaced by the next entry.
- 2026-11-17, "BoardLimit and Reliability bits judged under load" (decided before the next verdict session; `20261117g`): since NVML 615, 0x200 is BoardLimit and 0x400 Reliability, operating policies that cap clocks below the application clocks (NVML API Reference, Release 615); driver 595 does not report them, so its 0x0 means "not reported", not "clean". The single before sample is not the measured run, so the 1 Hz sampler and the share rule above judge them, also for the TTFT pairs; the idle gate (amended the same day, before any session used it, to allow 0x4) replaces the single idle read. Scope: `20261117e` and `20261117g` stay null; the strict rule can be re-applied later from the sampler data.
- 2026-11-17, "D2a passes on main" (`20261117k`): both models pass on main 6b07795 under the sampler rule; the throughput verdict of record. The empty `git_commit` in its `env.json` came from `phase4_box_d2a.sh` ignoring `COMMIT`, fixed since.
- 2026-11-17, "the Phase 4 exit test passes" (`20261117l`, `20261117k`): TTFT pairs pass in `20261117l`; `phase4_pass` true, combining `20261117k` (q8b, q30b) and `20261117l` (ttft) as "models judged in separate sessions" allows. That session's own 8B throughput pairs are not used. The milestone M2 continues with Phase 5.
- 2026-11-19, "one-input steps attend over their own run" (ADR 0079): a narrowing of "attention per sequence, prototyped and kept off". Steps with several inputs keep the masked attention (a); a step with one input attends over its own run, since that is one attention op either way and the 2026-11-16 threshold judged several inputs.

## Measurements without a session directory

Kept here until they have a home (CI runs, local runs, smoke tests).

- From "C4, the MoE builder on the pool" (2026-11-15): Numerics: a batched step, a chunked prefill and a one-query decode over another `n_kv` use other kernel shapes, so logits differ at about 1e-3 of the largest logit (CPU flash attention for one query sums over `n_kv` in an order that depends on it once `n_kv` reaches 1024). (No session directory yet.)
- From "CPU steps and the loop-bench gap" (2026-11-15): Mixing 3 decoders into a 402-token chunk cost 195 ms over the chunk alone, as two calls 66 ms (Qwen3-0.6B, EPYC 9V74, `crates/scheduler/tests/step_shapes.rs`). loop-bench found the scheduler loop itself at under 3 ms per step (gap between steps 0.05 ms, sampling and emitting at most 2.7 ms at 4 sequences). What remains at c4 on the CPU is the unified KV pool: against one KV stream per sequence (llama-server's layout) a 4-sequence decode step took 74 against 61 ms and a prompt above three resident sequences 19% longer (Xeon 6975P). (No session directory yet.)
- From "third review follow-ups, MoE" (2026-11-15): Build plus alloc measured about 0.2 ms per graph on an 8-layer synthetic model. First CPU numbers (`moe-shapes` workflow, synthetic 8-layer 32-expert Q8_0 model, EPYC 9V45, 4 threads, one run, not interleaved): `--engine moe` against libllama on the same file takes 1.08x the time per decode step at 4 sequences and alone, 1.19x for one sequence with 3 resident, 1.10 to 1.14x on prompt shapes. (No session directory yet.)
- From "speculation in the loop, built" (2026-11-16): a CI runner flipped a token at margin 0.0044. (No session directory yet.)
- From "attention per sequence, prototyped and kept off" (2026-11-16): On CPU (EPYC 9V45, 4 threads, Qwen3-0.6B Q8_0) it gains 20% on `decode1x8_resident` and 8 to 15% on prompt shapes with resident sequences, but `decode4x8` is 6% slower (four split-KV ops against one over 64 rows), against the threshold fixed before measuring (8% faster on both decode shapes). On CUDA the shared pool costs only 5% for resident sequences (3090 smoke). (No session directory yet.)
- From "step budget 2048, no spare sequence slots" (2026-11-16): (the 30B GPU step log showed mixed steps at 143 to 149 ms against 97 ms for a prompt alone). CPU c4 is unchanged by it (libllama 0.944 against 0.933 on main, dense 0.850 against 0.866, c1 within 3%; run 43889556576); the GPU gain is measured on the next box. CPU c1 decode fell from 36 to 25 tok/s as residents accumulated (run 43876231491). (No session directory yet.)
