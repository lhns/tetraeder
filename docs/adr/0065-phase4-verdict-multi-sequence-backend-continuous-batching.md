# 0065. Phase 4: verdict rules, a multi-sequence backend and continuous batching

Status: Proposed (2026-11-14). Phase 4. Supersedes, from ADR 0046, the one-sequence backend trait, the scheduler loop and "the gateway's semaphore stays at one request in generation"; ADR 0049's RAM prefix cache stays and gains resident sequences (step C3). Companions: ADR 0066 (own Qwen3 dense builder), ADR 0067 (KV layout experiment). Briefs: `docs/phase4-briefs.md`.

## Verdict

**Throughput.**
- **Judged:** `tetraeder serve --engine dense` (own builder, ADR 0066) on Qwen3-8B Q4_K_M. **Control:** `--engine llama` (libllama with step C2's batching) on the same file, same session; reported, never judged.
- **Qwen3-30B-A3B Q4_K_M** counts only on the own MoE builder (`--engine moe`) after step C4 gives it multi-sequence KV; then it must pass too. Before C4 it runs `--engine llama`, is recorded with `counts: false` and does not count either way.
- **Baseline:** llama-server of the baseline build (`/opt/llama.cpp-baseline`; `vendor/PIN` `baseline_tag` b12274, `baseline_commit` 202c1a22, after #30966): `-m F -ngl 99 -np 4 -c 32768 -fa on -ctk f16 -ctv f16 -t T`, continuous batching on (its default). The KV mode is tuned per session: a pre-pass runs c4 once with `-kvu` and once with `-no-kvu`; the higher `throughput_tps` is the session's baseline (`baseline_variant`). At the engine pin an explicit `-np` leaves `kv_unified` false (`common/common.h:582`; only auto `-np` sets it, `tools/server/server.cpp:127`); the result records the server's `kv_unified = '...'` log line, so b12274's behaviour is read, not assumed.
- **Tetraeder:** `tetraeder serve -m F --engine E --ngl -1 --parallel 4 --ctx 32768 -t T`, F16 KV, flash attention (the builders always use it; libllama with `--flash-attn on`). Same `T` for both engines (AGENTS.md pairing rules).
- **Workload:** `scripts/gateway_bench.py`, temperature 0, `--seed <rep>` equal for both engines of a pair (identical prompts). c4: `--prompt-tokens 512 --max-tokens 128 --concurrency 4 --requests 16`. c1: the same with `--concurrency 1 --requests 8`. A run is one server start: warm-up, c1, c4.
- **Gating fields:** c4 `summary.throughput_tps` (completion tokens over wall time, what ADR 0046 reported); c1 `summary.decode_tps_median`. Pair ratio r = Tetraeder / baseline.
- **Pairs:** two per model on one box in one session, interleaved: (baseline, Tetraeder), then (Tetraeder, baseline). The control runs after each pair.
- **Band b:** `band` of `bench/noise/<gpu-class>.json` if its `n` is at least 3, else the spec band (`bench/spec/phase0.toml`, class `cuda-default`, 0.08). Recorded as `band` and `band_source`.
- **Clean run:** `summary.errors` is 0; nvidia-smi throttle reasons before and after the run show nothing active beyond idle and the applications-clock setting; `scripts/cotenant_guard.py` does not say false. **Clean session:** the two baseline runs of a model agree within the band on both fields, |B2 / B1 - 1| <= b. A pair is clean when both runs are and the session is.
- **Pass,** per counting model and per field: both pairs clean and min over pairs of r >= 1 - b. "At least llama-server" means not slower beyond the noise band; a tie inside the band passes, as the owner ruled for Phase 3b (ADR 0046 addendum 2026-11-11). Today the A40 and A6000 class files hold one session each, so the spec band applies there and r >= 0.92 passes; the RTX 3090 file (11 sessions) gives 0.075. An unclean pair is rerun, never dropped to make a pass; fewer than two clean pairs per counting model is no verdict.

**TTFT.**
- **Log:** one Claude Code request log (`serve --request-log`, ADR 0046) recorded once by `scripts/claude_code_proxy.sh` over 5 tasks, 1 rep. It holds prompts in clear text, so it stays out of git (`bench/private/`, gitignored) and is uploaded to each box; results carry its `log_sha256` and request count.
- **Server:** the judged engine, Qwen3-8B, `--parallel 4 --ctx 40960`. Reuse on: defaults (live, resident-sequence and RAM reuse, `--cache-ram 8192`). Reuse off: `--no-prefix-reuse` (new in C3: no prefix reuse of any kind) with `--cache-ram 0`.
- **Replay:** `scripts/gateway_replay.py --speed 0 --concurrency 1 --max-prompt-tokens 36000` (new flag: entries whose logged `usage.prompt_tokens` exceed it are skipped, so every request fits the 8B's context of 40960, its `qwen3.context_length`; the script reads the key from the GGUF and stops if it differs).
- **Fields:** `summary.ttft_median_s` and `summary.ttft_p90_s` (new), over the requests with status 200 in both runs of the pair; `summary.errors` equal in both runs.
- **Pairs:** two, interleaved (on, off) then (off, on), one box; clean as above, without the A/A clause.
- **Pass:** in each pair median(on) / median(off) <= 1 - b and p90(on) / p90(off) <= 1 - b. Informative only: llama-server (`-np 4 -kvu --cache-ram 8192`) on the same log.

**Files** (`bench/results/<session>/phase4/`): `bench-<model>-<engine>-<test>-r<rep>.json` (gateway_bench.py), `prepass-<model>-<kvu|nokvu>.json`, `replay-<model>-<on|off>-r<rep>.json` (gateway_replay.py), `gpu-<run>.txt`, `server-*.log`, `env.json`, and `verdict.json` from `scripts/phase4_summary.py`: `{band, band_source, models: {<model>: {engine, counts, baseline_variant, baseline_kv_unified, aa: {c4, c1}, pairs: [{reps, c4_throughput_ratio, c1_decode_ratio, clean, reasons}], worst: {c4, c1}, pass}}, ttft: {log_sha256, requests, pairs: [{median_ratio, p90_ratio, clean, reasons}], pass}, phase4_pass}`. `phase4_pass` is true when every counting model and `ttft` pass. Engine names in files: `llama-server`, `tetraeder-dense`, `tetraeder-moe`, `tetraeder-llama`.

**Settled 2026-11-14.**
- Pass rule r >= 1 - b: the owner's choice ("not slower beyond noise").
- c4 gates on `throughput_tps`, the exit test's "throughput", which includes prefill as ADR 0046 measured; decode-only rates are reported, not gated.
- Workload size stays 16 c4 requests: in `bench/results/20261103/gateway/` a c4 run takes 11 s (llama-server) to 27 s (Tetraeder) and the two repeats agree within 0.5%.
- c1 is not the risk: there Tetraeder already matched llama-server (Qwen3-8B 81.1 against 79.8 tok/s `decode_tps_median`, Qwen3-30B-A3B 129.9 against 129.4); the 0.41 to 0.61 gap is c4 only, which is what batching addresses.
- Recording the Claude Code request log is the first stage of the verdict box session (`docs/m2-plan.md`, D2), before any timed run.
- To read on the box and record, not assume: b12274's `kv_unified` with `-np 4` (the pre-pass covers both) and Qwen3-8B's `qwen3.context_length` (40960; the harness stops if it differs).

## Context

The plan's Phase 4 exit test: at least llama.cpp server throughput at concurrency 4; prefix hits cut time to first token (TTFT) on a Claude Code replay. ADR 0046 measured one sequence at a time: concurrency 4 gave 0.41 (Qwen3-8B) and 0.59 to 0.61 (Qwen3-30B-A3B) of llama-server `-np 4` throughput, TTFT 4.9 to 6.6x longer.

What exists: `scheduler::Backend` holds one sequence (`n_past`, `truncate`, `eval`, `snapshot`, `restore`). The libllama engine uses sequence 0 only (`memory_seq_rm(0, ..)`); `ggml_sys::llama::Batch` already carries a sequence id per token, but `ContextParams` has no `n_seq_max` or `kv_unified` and there is no `seq_cp` binding. The MoE builder (`crates/model/src/qwen3moe.rs`) has one contiguous KV per layer (`[n_embd_kv, n_ctx]`), a mask `[n_kv, n]` and writes K and V with `set_rows` at the positions.

The M1 milestone lost weeks to disputes over which configuration counts, whether a baseline was tuned and whether a session was clean (`docs/m1-open.md`). So the verdict above is fixed before any code.

## Decision

### Attention over sequences of unequal length

At the pin, `ggml_flash_attn_ext` takes q `[d, n_tok, n_head, ne3]`, k and v `[d, n_kv, n_head_kv, ne3]` and an F16 mask `[n_kv, n_tok, ne32, ne33]` (`ggml/include/ggml.h:2493`). Two ways to batch sequences:

- **(a) One pool with a mask** (llama.cpp unified KV): one K and one V tensor per layer `[n_embd_kv, n_rows]`; all tokens of a step are one Q (ne3 = 1); mask row t is 0 on rows of t's sequence at positions up to pos(t), minus infinity elsewhere. Any mix of token counts per sequence fits one op. The CPU kernel skips a masked row before its dot product (`ggml-cpu/ops.cpp:8944`, `:9209`), so it reads only rows a query may see. CUDA reads `[0, n_kv)` per query tile and trims only trailing fully masked blocks (`flash_attn_mask_to_KV_max`, `ggml-cuda/fattn-common.cuh:671`).
- **(b) A buffer per sequence** (llama.cpp non-unified): K `[n_embd_kv, kv_size, n_stream]`, ne3 = stream, one mask per stream. Each stream reads only its rows, but every stream has the same token count in one op (q's ne1 is shared; llama.cpp uses `split_equal`, `src/llama-kv-cache.cpp:705`), capacity per sequence is fixed at kv_size, and `seq_cp` across streams copies a whole stream buffer (`llama-kv-cache.cpp:820-845`).

What reaches llama-server's batched throughput is one graph per step for all sequences, so the weights are read once for four tokens. Attention is small at the exit workload: 36 layers x 640 positions x 4 KiB of K and V per row is 94 MB per step against 5.02 GB of Qwen3-8B Q4_K_M weights (5,021,827,072 bytes, `model_size` in `bench/results/20261103/gateway/overhead-q8b-llama-bench.json`). Both layouts batch decode; they differ when prefill and decode mix. (a) puts a 512-token prefill chunk and three decode tokens in one op and one graph. (b) needs equal counts per stream, so the step becomes several graphs (llama.cpp splits into ubatches) or padded queries. (a) pays on CUDA for rows below n_kv that the step's sequences do not own; when every live sequence decodes, every row is needed anyway.

**Decision: (a).** One pool per layer, one `flash_attn_ext` per layer per step, viewing rows `[0, n_kv)` with n_kv = pad(highest row the step reads + 1, 256). The graph key stays (n_tokens, n_kv), so CUDA graph reuse works as in ADR 0036. Which rows a sequence gets is the KV layout's job (`kvcache`, ADR 0067): chunked or paged changes the rows, the mask and the `set_rows` indices, never the op, so the trait and both builders hold either layout. The libllama backend runs llama.cpp's own cache with `kv_unified = true`, the same scheme. A row belongs to at most one sequence; `seq_cp` copies rows. llama.cpp's shared cells (one row visible to several sequences) save memory but make writes and `seq_rm` reference-counted; that waits for paged blocks (ADR 0067). To verify on the box: CUDA flash attention with mixed sequences in one query tile matches the CPU (gpu parity tests of C4 and C5).

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
- libllama: `n_seq_max` and `kv_unified = true` in `ContextParams`, `n_batch = n_ubatch = step_tokens`; `seq_rm` is `llama_memory_seq_rm(seq, keep, -1)`, `seq_cp` is `llama_memory_seq_cp(src, dst, 0, n)` (in one stream llama.cpp only adds dst to the cells' sequence sets, `llama-kv-cache.cpp:470-497`; the adapter still counts n reserved rows for dst, which over-counts, never under-counts). Logits per output come from `llama_get_logits_ith(batch index)` into one kept buffer. Reservations are the adapter's bookkeeping against `n_ctx`: unified cells are found one by one (`find_slot(ubatch, false)`, `:761`, `:1011`), so a reservation under capacity cannot fail on fragmentation. libllama's `NoKvSlot` maps to `KvFull`, every other decode error to `Failed`.
- Until C4, `MoeBackend` reaches the new loop through `SingleSeq<B: Backend>`, an adapter with `n_seq_max = 1` over the old trait; C4 implements the trait natively and removes the old trait.

### Scheduler loop (`crates/scheduler/src/scheduler.rs`, step planning in `batch.rs`)

- One worker thread owns the backend (as ADR 0046). Each iteration: take new jobs, admit, plan one step, decode, sample, emit, retire. It blocks on the job queue only when nothing is live.
- **Admission:** first in, first out. A job is admitted when a sequence id is free (`--parallel`, default 4) and `seq_reserve(seq, prompt + min(max_tokens, GROW))` succeeds, GROW = 1024. Prefix reuse picks where its first rows come from (C3). The gateway's `max_concurrent` becomes `--parallel` (was 1); `max_queued` and its 503 stay.
- **Step plan** under `step_tokens` (`--step-tokens`, default the `--ubatch` value, 512): every decoding, unpaused sequence gets 1 token (1 + k with drafts); the rest of the budget goes to prefilling sequences in admission order, each its next chunk up to what is left. One step is one `decode` call, one graph on the builders and one ubatch on libllama.
- **Growth and preemption:** a sequence about to pass its reservation calls `seq_reserve(len + GROW)` before the step. On KvFull idle resident sequences go first (oldest first, saved to RAM per ADR 0049); if that is not enough, the most recently admitted other live sequence is preempted: snapshot into the RAM prefix cache if it holds at least `SAVE_MIN_TOKENS` (ADR 0049), freed, and its job goes back to the queue front with prompt plus generated tokens as its prompt, keeping its sampler, constraint and channel (nothing is sent twice). With no other sequence the request finishes with `Length`, as a full context does today.
- **Sampling:** per sequence its own `Sampler` and constraint, on the worker thread after the step, in sequence order.
- **Backpressure:** events go out with `try_send`; a sequence whose channel is full is paused (left out of steps, rows kept) until the channel has room. One slow client never stalls the batch.
- **Cancellation:** checked per sequence every step; a cancelled request or a closed receiver finishes `Cancelled`.
- **Retirement:** end of generation, `max_tokens`, full context or cancel send `Finished`; the sequence stays resident as an idle prefix (C3) until its rows or id are needed, then it is saved per ADR 0049's rules and freed.
- **Errors:** `Failed` fails the requests of that step with `GenEvent::Error` and frees their sequences; the others continue. `KvFull` never reaches a request.

### Speculative decoding (`spec/`)

`Target` and `Drafter` stay. An adapter `SeqTarget { backend, seq }` implements `Target` (eval = one `SeqInput` with `all_logits`, rewind = `seq_rm`), so the existing spec tests run against batch backends. In the loop a sequence drafts only when it is the only decoding sequence (`--spec-max-batch`, default 1): batching spends the bandwidth headroom speculation lives on, and ADR 0043's bitwise exactness holds only for one sequence. Its verify rows are its entry in the step; acceptance is `seq_rm(seq, before + 1 + accepted)`. Speculation stays off by default (ADR 0043 results); wiring it into `serve` is a later item, not one of the steps below.

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
- Sampling runs serially on the worker: about 0.1 ms per sequence greedy at 152K vocabulary, more with temperature (ADR 0046 addendum). If `tetraeder.scheduler.decode.stage{stage=sample}` shows it costing the c4 ratio, sampling moves to scoped threads per sequence.
- The verdict reads only committed fields; a dispute about a session is settled by `verdict.json`, not by prose.

## Addendum: C2b, the loop (2026-11-14)

Differences found while building the loop; the decisions above stand.

- `BatchBackend::max_step_tokens() -> usize` (default `usize::MAX`; libllama: `n_batch`). The loop keeps its step plan under `min(--step-tokens, max_step_tokens)`, never below `--parallel` (decoders run beyond the budget), and caps `--parallel` by `n_seq_max` and `max_step_tokens`.
- Preemption: if the most recently admitted other sequence is older than the one that needs rows, the needing sequence is requeued itself, so the oldest request always progresses. With no other sequence the request ends with `Length`.
- The effective parallelism is `min(--parallel, backend.n_seq_max())`: the MoE builder serves one sequence until C4, `--plan` serves one.
- Events of a finished request that its channel does not take yet wait in a drain list; its sequence is idle at once. A request that finishes with a full channel is not held back by it.
- A RAM entry chosen for a restore is taken out of the cache before any idle sequence is evicted (the eviction may drop it) and put back when admission has to wait.
- The decode profile's `eval_s` is the wall time of the steps the request took part in; `gpu_seconds` is its token share of those steps.
- libllama: `n_ctx` is rounded up (to a multiple of 256 with the unified cache) and `n_batch` is clamped to the requested `n_ctx`; a step whose cells do not fit fails with `NoKvSlot` (the engine cuts every input back with `seq_rm`, lengths unchanged).
