//! The serving scheduler: a worker thread owning a [`BatchBackend`], a job queue, continuous
//! batching of up to `parallel` sequences with chunked prefill under a step token budget, prefix
//! reuse of idle resident sequences and of saved ones in host RAM, and Rust sampling over the
//! logits (ADR 0046, ADR 0065).

use crate::backend::{BackendError, BatchBackend, SeqId, SeqInput};
use crate::batch::{Live, plan_step, victim};
use crate::load::{Loaded, model_id};
use crate::prefix_cache::{PrefixCache, RESTORE_MIN_GAIN, SAVE_MIN_TOKENS, common_prefix};
use crate::sample::{SampleParams, Sampler, SamplerEnv};
use crate::spec::{Drafter, Sampling, SpecConfig, Speculator};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc as std_mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tetraeder_gateway::engine::{
    DecodeProfile, Engine, EngineError, EngineStats, FinishReason, GenEvent, GenerateRequest,
    ModelInfo, SamplingParams, TokenConstraint, TokenId, Vocab,
};
use tetraeder_ggml_sys::llama::Model;
use tetraeder_telemetry::{Metrics, Purpose, Tier};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// Default tokens per step (`--step-tokens`): the unit of cancellation during prefill. llama-server's
/// `n_batch`, so prompts that arrive together are prefilled in one step.
pub const STEP_TOKENS: usize = 2048;
/// Default sequences decoded together (`--parallel`).
pub const PARALLEL: usize = 4;
/// Rows reserved beyond the prompt, and the growth step of a sequence (ADR 0065).
const GROW: usize = 1024;

/// Events buffered per request before the worker waits for the gateway.
const EVENT_BUFFER: usize = 32;

pub struct SchedulerConfig {
    /// Served model id; default is the model file name without extension and shard suffix.
    pub alias: Option<String>,
    /// The model file, for the default id and the `created` time.
    pub model_path: PathBuf,
    /// Jinja chat template; `None` serves a base model (completions only).
    pub chat_template: Option<String>,
    /// Sampling defaults for requests that send none (`SampleParams::default` is llama-server's).
    pub sampling: SampleParams,
    /// Seed of requests that send none; `None` draws a random one per request, like llama-server.
    pub seed: Option<u32>,
    /// Scheduler metrics are recorded here when set.
    pub metrics: Option<Metrics>,
    /// Host bytes for saved sequences (the prefix cache); 0 turns it off.
    pub cache_ram_bytes: usize,
    /// A live sequence is saved before it loses this many tokens.
    pub cache_save_min: usize,
    /// A saved sequence is restored when it beats the live one by this many tokens.
    pub cache_restore_gain: usize,
    /// Report where each request's decode time went ([`DecodeProfile`]); default from
    /// `TETRAEDER_DECODE_PROFILE=1`. Off, it costs nothing beyond the timers that exist anyway.
    pub decode_profile: bool,
    /// Sequences decoded together; default [`PARALLEL`]. At most the backend's `n_seq_max`.
    pub parallel: usize,
    /// Tokens per step over all sequences; default [`STEP_TOKENS`]. Decoding sequences take
    /// their token even beyond it; the rest goes to prompt chunks. The builder's plan sets it
    /// (ADR 0055). Capped by the backend's `max_step_tokens`.
    pub step_tokens: usize,
    /// Reuse prefixes: idle and busy resident sequences and the RAM cache. Off, every request
    /// starts an empty sequence (`--no-prefix-reuse`); `cache_ram_bytes` is independent.
    pub prefix_reuse: bool,
    /// Speculative decoding of a lone greedy sequence (ADR 0043, 0065); `None` is off.
    pub spec: Option<SpecConfig>,
}

/// Generated tokens past the prompt that a snapshot size hint counts (ADR 0074).
const EXPECT_AHEAD: usize = 512;

/// An idle sequence reused in place by its own continuation is saved only when the rows the
/// request cuts off exceed this fraction of its length: recomputing a small cut costs less than
/// the multi-GB snapshot (prefill about 0.36 ms per token against a save of about 0.06 ms per
/// token on the 8B, ADR 0074).
const LIVE_SAVE_CUT_FRACTION: f64 = 0.25;

/// Default prefix cache budget, like llama-server's `--cache-ram`.
pub const DEFAULT_CACHE_RAM_BYTES: usize = 8192 << 20;

impl SchedulerConfig {
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            alias: None,
            model_path: model_path.into(),
            chat_template: None,
            sampling: SampleParams::default(),
            seed: None,
            metrics: None,
            cache_ram_bytes: DEFAULT_CACHE_RAM_BYTES,
            cache_save_min: SAVE_MIN_TOKENS,
            cache_restore_gain: RESTORE_MIN_GAIN,
            decode_profile: std::env::var_os("TETRAEDER_DECODE_PROFILE").is_some_and(|v| v == "1"),
            parallel: PARALLEL,
            step_tokens: STEP_TOKENS,
            prefix_reuse: true,
            spec: None,
        }
    }
}

/// The gateway's [`Engine`] over one backend.
pub struct Scheduler {
    info: ModelInfo,
    vocab: Arc<Vocab>,
    tokenizer: Arc<Model>,
    queue: Mutex<std_mpsc::Sender<Req>>,
    worker: std::thread::JoinHandle<()>,
}

impl Scheduler {
    /// Starts the worker, which runs `load` on its own thread (the MoE builder may not be
    /// `Send`) and owns the backend from then on. Returns when the model is loaded, or with
    /// the load error.
    pub fn start(
        cfg: SchedulerConfig,
        load: impl FnOnce() -> Result<Loaded, String> + Send + 'static,
    ) -> Result<Self, String> {
        let (job_tx, job_rx) = std_mpsc::channel::<Req>();
        let (ready_tx, ready_rx) = std_mpsc::channel::<Result<(Arc<Model>, usize), String>>();
        let defaults = cfg.sampling.clone();
        let default_seed = cfg.seed;
        let metrics = cfg.metrics.clone();
        let decode_profile = cfg.decode_profile;
        let (parallel, step_tokens) = (cfg.parallel.max(1), cfg.step_tokens.max(1));
        let prefix_reuse = cfg.prefix_reuse;
        let spec_cfg = cfg.spec.clone();
        let (cache_ram, save_min, restore_gain) = (
            cfg.cache_ram_bytes,
            cfg.cache_save_min,
            cfg.cache_restore_gain,
        );
        let vocab_cell: Arc<std::sync::OnceLock<Arc<Vocab>>> = Arc::default();
        let worker_vocab = Arc::clone(&vocab_cell);
        let worker = std::thread::Builder::new()
            .name("tetraeder-scheduler".into())
            .spawn(move || {
                let loaded = match load() {
                    Ok(l) => l,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                let vocab = Arc::new(build_vocab(&loaded.tokenizer));
                let _ = worker_vocab.set(Arc::clone(&vocab));
                let n_ctx = loaded.backend.n_ctx_seq();
                let spec = match spec_cfg
                    .map(|c| c.build_drafter(loaded.backend.n_vocab()).map(|d| (c, d)))
                    .transpose()
                {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = ready_tx.send(Err(format!("cannot set up speculation: {e}")));
                        return;
                    }
                };
                let _ = ready_tx.send(Ok((Arc::clone(&loaded.tokenizer), n_ctx)));
                let pool = tetraeder_model::snapshot_pool::global();
                pool.set_max_bytes(cache_ram);
                pool.set_default_spares(parallel.max(2));
                let mut w = Worker::new(
                    loaded.backend,
                    PrefixCache::new(cache_ram),
                    save_min,
                    restore_gain,
                    prefix_reuse,
                    vocab,
                    defaults,
                    metrics,
                    decode_profile,
                    parallel,
                    step_tokens,
                );
                w.model = Some(Arc::clone(&loaded.tokenizer));
                w.default_seed = default_seed;
                if let Some((c, d)) = spec {
                    w.set_spec(&c, d);
                }
                w.run(&job_rx);
            })
            .map_err(|e| format!("cannot start the scheduler thread: {e}"))?;
        let (tokenizer, n_ctx) = ready_rx
            .recv()
            .map_err(|_| "the scheduler thread died while loading".to_owned())??;
        let vocab = vocab_cell
            .get()
            .cloned()
            .ok_or("the vocabulary was not built")?;
        let piece = |t: i32| {
            (t >= 0 && (t as usize) < vocab.len())
                .then(|| String::from_utf8_lossy(vocab.bytes(t as u32)).into_owned())
        };
        let created = std::fs::metadata(&cfg.model_path)
            .and_then(|m| m.modified())
            .unwrap_or_else(|_| SystemTime::now())
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let info = ModelInfo {
            id: cfg
                .alias
                .clone()
                .unwrap_or_else(|| model_id(&cfg.model_path)),
            context_length: u32::try_from(n_ctx).unwrap_or(u32::MAX),
            chat_template: cfg.chat_template.clone(),
            bos_token: piece(tokenizer.bos()),
            eos_token: piece(tokenizer.eos()),
            created,
        };
        Ok(Self {
            info,
            vocab,
            tokenizer,
            queue: Mutex::new(job_tx),
            worker,
        })
    }

    /// Stops accepting jobs, lets the running ones finish and joins the worker, so the
    /// backend (and its memory) is gone when this returns.
    pub fn shutdown(self) -> Result<(), String> {
        let Self { queue, worker, .. } = self;
        drop(queue);
        worker
            .join()
            .map_err(|_| "the scheduler thread panicked".to_owned())
    }

    fn check(&self, model: &str) -> Result<(), EngineError> {
        if model == self.info.id {
            Ok(())
        } else {
            Err(EngineError::ModelNotFound(model.into()))
        }
    }
}

impl Engine for Scheduler {
    fn models(&self) -> Vec<ModelInfo> {
        vec![self.info.clone()]
    }

    fn vocab(&self, model: &str) -> Result<Arc<Vocab>, EngineError> {
        self.check(model)?;
        Ok(Arc::clone(&self.vocab))
    }

    fn tokenize(&self, model: &str, text: &str) -> Result<Vec<TokenId>, EngineError> {
        self.check(model)?;
        let toks = self
            .tokenizer
            .tokenize(text, false, true)
            .map_err(|e| EngineError::Other(e.to_string()))?;
        Ok(toks.into_iter().map(|t| t as TokenId).collect())
    }

    fn generate(&self, req: GenerateRequest) -> Result<mpsc::Receiver<GenEvent>, EngineError> {
        self.check(&req.model)?;
        let (tx, rx) = mpsc::channel(EVENT_BUFFER);
        let job = Req::new(req.prompt, req.params, req.constraint, req.cancel, tx);
        self.queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .send(job)
            .map_err(|_| EngineError::Other("the engine worker has stopped".into()))?;
        Ok(rx)
    }
}

/// Token texts, control flags and end-of-generation ids from a libllama vocabulary.
pub fn build_vocab(model: &Model) -> Vocab {
    let n = model.n_vocab();
    let mut tokens = Vec::with_capacity(n as usize);
    let mut special = Vec::new();
    let mut eog = Vec::new();
    for id in 0..n {
        tokens.push(model.token_to_piece(id, true).unwrap_or_default());
        if model.is_control(id) {
            special.push(id as TokenId);
        }
        if model.is_eog(id) {
            eog.push(id as TokenId);
        }
    }
    // The primary EOS comes first.
    let eos = model.eos();
    if let Some(i) = eog.iter().position(|&t| i64::from(t) == i64::from(eos)) {
        eog.swap(0, i);
    }
    Vocab {
        tokens,
        special,
        eog,
    }
}

/// Time of the steps of one request so far.
#[derive(Debug, Clone, Copy, Default)]
struct StepTimes {
    /// Steps that sampled a token.
    steps: u32,
    /// Wall time of every backend `decode` the request took part in: a step shared with other
    /// requests counts in full for each of them, so this is not the request's own cost.
    eval: Duration,
    /// Constraint mask, sampling and commit.
    sample: Duration,
}

/// A request from arrival to its last event. It goes back to the queue front on preemption with
/// its sampler, constraint and channel: `toks` then holds the prompt plus what was generated, and
/// nothing is sent twice.
struct Req {
    /// The prompt, then the generated tokens (the last one may not be in the KV yet).
    toks: Vec<i32>,
    params: SamplingParams,
    constraint: Option<Box<dyn TokenConstraint>>,
    cancel: CancellationToken,
    tx: mpsc::Sender<GenEvent>,
    /// Events not yet accepted by the channel; while there are any the request is paused.
    outbox: VecDeque<GenEvent>,
    /// Created at the first admission.
    sampler: Option<Sampler>,
    stats: EngineStats,
    times: StepTimes,
    /// This request's share of the step times, by tokens. Shares of concurrent requests add up
    /// to the step, so it depends on who else ran; no gate may use per-request `gpu_seconds`.
    gpu: Duration,
    emit: Duration,
    /// With the decode profile: the times and backend counters when its first token was sampled.
    after_prefill: Option<(StepTimes, Vec<(&'static str, f64)>)>,
    bytes0: Option<u64>,
    /// Admitted at least once.
    started: bool,
    /// A busy copy for this request failed: its next placement does not offer a busy source, so
    /// it falls back to the idle sequence or the RAM entry. Cleared when it is placed.
    no_busy: bool,
    /// When the gateway handed it to the scheduler, for the admission log.
    queued: Instant,
    /// Placement attempts that put it back in the queue, for the admission log.
    placing: Placing,
}

/// What earlier placement attempts of a request cost before it went back to the queue (a
/// failed copy, a sequence freed to grow, no rows yet). The admission log adds them to the
/// attempt that places it, so a save is reported on the admission it was made for, not as
/// queueing (ttft4-20261119).
#[derive(Debug, Default, Clone)]
struct Placing {
    attempts: u32,
    took: Duration,
    save: Duration,
    restore: Duration,
    /// Sequences saved to RAM, as (sequence, positions).
    saved: Vec<(usize, usize)>,
}

/// One admission, as logged (`tetraeder::step` at debug).
#[derive(Debug, Clone)]
struct AdmitRecord {
    seq: usize,
    /// From arrival to the last attempt, less the earlier attempts.
    queued: Duration,
    /// Every attempt.
    place: Duration,
    save: Duration,
    restore: Duration,
    attempts: u32,
    /// Sequences saved to RAM while placing it, as (sequence, positions): the one it reuses
    /// in place, or others evicted for its rows.
    saved: Vec<(usize, usize)>,
    prompt: usize,
    common: usize,
    source: &'static str,
    skipped: Option<&'static str>,
}

impl Req {
    fn new(
        prompt: Vec<TokenId>,
        params: SamplingParams,
        constraint: Option<Box<dyn TokenConstraint>>,
        cancel: CancellationToken,
        tx: mpsc::Sender<GenEvent>,
    ) -> Self {
        let stats = EngineStats {
            prompt_tokens: u32::try_from(prompt.len()).unwrap_or(u32::MAX),
            ..EngineStats::default()
        };
        Self {
            toks: prompt.iter().map(|&t| t as i32).collect(),
            params,
            constraint,
            cancel,
            tx,
            outbox: VecDeque::new(),
            sampler: None,
            stats,
            times: StepTimes::default(),
            gpu: Duration::ZERO,
            emit: Duration::ZERO,
            after_prefill: None,
            bytes0: None,
            started: false,
            no_busy: false,
            queued: Instant::now(),
            placing: Placing::default(),
        }
    }

    /// Sends what the channel takes, in order. False when the receiver is gone.
    fn flush(&mut self) -> bool {
        while let Some(ev) = self.outbox.pop_front() {
            match self.tx.try_send(ev) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Full(ev)) => {
                    self.outbox.push_front(ev);
                    break;
                }
                Err(mpsc::error::TrySendError::Closed(_)) => return false,
            }
        }
        true
    }
}

/// The events a retired request still has to deliver; the sequence is already free.
struct Drain {
    tx: mpsc::Sender<GenEvent>,
    events: VecDeque<GenEvent>,
}

/// A running request on a sequence.
struct Seq {
    req: Req,
    /// Positions of `req.toks` in the KV cache.
    fed: usize,
    /// Admission order, smaller is older.
    admitted: u64,
    /// Its first token was sampled (else it is prefilling).
    decoding: bool,
    /// Rows asked of the backend for this sequence.
    reserved_to: usize,
    /// Draft tokens that ride in this step after the pending token. `None`: the sequence does
    /// not speculate in this step; `Some(empty)`: it does, with no draft.
    drafts: Option<Vec<i32>>,
}

impl Seq {
    fn paused(&self) -> bool {
        !self.req.outbox.is_empty()
    }
}

/// A finished request's sequence, kept as a prefix until its rows or id are needed.
struct Idle {
    toks: Vec<i32>,
    /// Length of the request's prompt; the rest of `toks` is what it generated.
    prompt_len: usize,
    /// Retirement order, smaller is older.
    order: u64,
}

/// What reusing an idle sequence in place does to the rows a request cuts off (ADR 0074).
enum LiveCut {
    /// Too few rows to bother.
    Nothing,
    /// Recomputing is cheaper than a save: `small_cut` or `own_output`.
    Skip(&'static str),
    /// The sequence is saved first.
    Save,
}

impl Idle {
    /// The fate of this sequence's cut when a request shares its first `common` tokens.
    /// A cut that is more than `save_min` rows is saved unless it is a quarter of the
    /// sequence at most, or lies wholly in the tokens the sequence generated itself, which a
    /// client may never resend.
    fn live_cut(&self, common: usize, save_min: usize) -> LiveCut {
        let cut = self.toks.len() - common.min(self.toks.len());
        if cut < save_min {
            LiveCut::Nothing
        } else if common >= self.prompt_len {
            LiveCut::Skip("own_output")
        } else if cut as f64 <= LIVE_SAVE_CUT_FRACTION * self.toks.len() as f64 {
            LiveCut::Skip("small_cut")
        } else {
            LiveCut::Save
        }
    }
}

enum Slot {
    Free,
    Idle(Idle),
    Live(Box<Seq>),
}

enum Admission {
    Admitted,
    /// Not enough KV rows now; the request stays at the queue front.
    Wait,
}

struct Worker {
    backend: Box<dyn BatchBackend>,
    /// One slot per sequence id in use.
    slots: Vec<Slot>,
    waiting: VecDeque<Req>,
    drains: Vec<Drain>,
    next_admitted: u64,
    next_order: u64,
    /// Sequences evicted from the KV cache, kept in host RAM.
    cache: PrefixCache,
    save_min: usize,
    restore_gain: usize,
    prefix_reuse: bool,
    vocab: Arc<Vocab>,
    defaults: SampleParams,
    /// Seed of requests that send none.
    default_seed: Option<u32>,
    /// The tokenizer model, whose vocabulary DRY needs; `None` rejects DRY requests.
    model: Option<Arc<Model>>,
    metrics: Option<Metrics>,
    decode_profile: bool,
    /// Tokens per step.
    budget: usize,
    n_ctx_seq: usize,
    /// End of the last step, for the step log's idle gap.
    last_step: Option<Instant>,
    n_vocab: usize,
    /// After a failed multi-input step: the sequences to retry one per step, so only the one
    /// whose input the backend rejects fails.
    solo: VecDeque<SeqId>,
    spec: Option<SpecState>,
    /// Time spent saving sequences to the prefix cache, for the admission log.
    save_time: Duration,
    /// Time spent restoring saved sequences, for the admission log.
    restore_time: Duration,
    /// Snapshot pool counters (spare, fallback, touched, freed) already reported to the metrics.
    pool_seen: (u64, u64, u64, u64),
    /// Saves made inside a step by `make_room` (not part of an admission's `save_ms`) and their
    /// time; each is also traced as a `step save` line.
    step_saves: u64,
    step_save_time: Duration,
    /// Sequences saved since the current placement attempt began, as (sequence, positions).
    saved: Vec<(usize, usize)>,
    /// The last admission, as logged.
    last_admit: Option<AdmitRecord>,
}

/// Speculation in the loop: drafts for a lone greedy decoding sequence (ADR 0065).
struct SpecState {
    speculator: Speculator,
    drafter: Box<dyn Drafter>,
    max_batch: usize,
}

/// Mask, sample and commit one token from `logits`.
fn sample_one(req: &mut Req, logits: &[f32]) -> Result<u32, String> {
    let mask = match req.constraint.as_mut() {
        Some(c) => c.mask()?,
        None => None,
    };
    let sampler = req.sampler.as_mut().ok_or("no sampler")?;
    let tok = sampler
        .sample(logits, mask.as_ref())
        .ok_or("the constraint allows no token")?;
    if let Some(c) = req.constraint.as_mut() {
        c.commit(tok)?;
    }
    Ok(tok)
}

impl Worker {
    #[allow(clippy::too_many_arguments)]
    fn new(
        backend: Box<dyn BatchBackend>,
        cache: PrefixCache,
        save_min: usize,
        restore_gain: usize,
        prefix_reuse: bool,
        vocab: Arc<Vocab>,
        defaults: SampleParams,
        metrics: Option<Metrics>,
        decode_profile: bool,
        parallel: usize,
        step_tokens: usize,
    ) -> Self {
        let max_step = backend.max_step_tokens().max(1);
        let parallel = parallel.min(backend.n_seq_max()).min(max_step).max(1);
        // Decoding sequences take their token even beyond the budget, so it covers them all.
        let budget = step_tokens.min(max_step).max(parallel);
        let n_ctx_seq = backend.n_ctx_seq();
        let n_vocab = backend.n_vocab();
        let pool_seen = {
            let s = cache.pool().stats();
            (s.from_spare, s.fallback, s.touched_bytes, s.freed_bytes)
        };
        Self {
            backend,
            slots: (0..parallel).map(|_| Slot::Free).collect(),
            waiting: VecDeque::new(),
            drains: Vec::new(),
            next_admitted: 0,
            next_order: 0,
            cache,
            save_min,
            restore_gain,
            prefix_reuse,
            vocab,
            defaults,
            default_seed: None,
            model: None,
            metrics,
            decode_profile,
            budget,
            n_ctx_seq,
            last_step: None,
            n_vocab,
            solo: VecDeque::new(),
            spec: None,
            save_time: Duration::ZERO,
            restore_time: Duration::ZERO,
            pool_seen,
            step_saves: 0,
            step_save_time: Duration::ZERO,
            saved: Vec::new(),
            last_admit: None,
        }
    }

    fn set_spec(&mut self, cfg: &SpecConfig, drafter: Box<dyn Drafter>) {
        self.spec = Some(SpecState {
            speculator: Speculator::new(cfg.len, Sampling::GREEDY, None),
            drafter,
            max_batch: cfg.max_batch.max(1),
        });
    }

    fn live_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| matches!(s, Slot::Live(_)))
            .count()
    }

    fn has_work(&self) -> bool {
        !self.waiting.is_empty() || !self.drains.is_empty() || self.live_count() > 0
    }

    /// The loop of ADR 0065: take new jobs, admit, plan one step, decode, sample, emit, retire.
    /// Blocks on the job queue only when nothing is live.
    fn run(&mut self, rx: &std_mpsc::Receiver<Req>) {
        let mut closed = false;
        loop {
            loop {
                match rx.try_recv() {
                    Ok(j) => self.waiting.push_back(j),
                    Err(std_mpsc::TryRecvError::Empty) => break,
                    Err(std_mpsc::TryRecvError::Disconnected) => {
                        closed = true;
                        break;
                    }
                }
            }
            self.flush_drains();
            if !self.has_work() {
                if closed {
                    return;
                }
                match rx.recv() {
                    Ok(j) => self.waiting.push_back(j),
                    Err(_) => return,
                }
                continue;
            }
            self.admit();
            self.sweep();
            let ran = self.step();
            self.record_sequences();
            if !ran && self.has_work() {
                // Everything live is paused (or only events are left to deliver): wait for the
                // consumers, but wake for new jobs.
                if closed {
                    std::thread::sleep(Duration::from_millis(1));
                } else {
                    match rx.recv_timeout(Duration::from_millis(1)) {
                        Ok(j) => self.waiting.push_back(j),
                        Err(std_mpsc::RecvTimeoutError::Timeout) => {}
                        Err(std_mpsc::RecvTimeoutError::Disconnected) => closed = true,
                    }
                }
            }
        }
    }

    fn flush_drains(&mut self) {
        self.drains.retain_mut(|d| {
            while let Some(ev) = d.events.pop_front() {
                match d.tx.try_send(ev) {
                    Ok(()) => {}
                    Err(mpsc::error::TrySendError::Full(ev)) => {
                        d.events.push_front(ev);
                        return true;
                    }
                    Err(mpsc::error::TrySendError::Closed(_)) => return false,
                }
            }
            false
        });
    }

    fn record_sequences(&self) {
        let Some(m) = &self.metrics else { return };
        let (mut prefill, mut decode, mut paused, mut idle) = (0, 0, 0, 0);
        for s in &self.slots {
            match s {
                Slot::Free => {}
                Slot::Idle(_) => idle += 1,
                Slot::Live(s) if s.paused() => paused += 1,
                Slot::Live(s) if s.decoding => decode += 1,
                Slot::Live(_) => prefill += 1,
            }
        }
        m.scheduler_sequences("prefill", prefill);
        m.scheduler_sequences("decode", decode);
        m.scheduler_sequences("paused", paused);
        m.scheduler_sequences("idle", idle);
    }

    // ---- ending requests ----

    /// Fills in the stats that need the backend.
    fn finalize(&self, req: &mut Req) {
        if self.backend.on_gpu() {
            req.stats.gpu_seconds = Some(req.gpu.as_secs_f64());
        }
        if let Some((t0, c0)) = req.after_prefill.take() {
            req.stats.profile = Some(self.profile_of(req, &t0, &c0));
        }
        req.stats.bytes_streamed = match (req.bytes0, self.backend.bytes_streamed_total()) {
            (Some(a), Some(b)) => Some(b.saturating_sub(a)),
            _ => None,
        };
    }

    /// The decode profile of a request: everything after the step that ended its prefill.
    fn profile_of(
        &self,
        req: &Req,
        prefill: &StepTimes,
        counters0: &[(&'static str, f64)],
    ) -> DecodeProfile {
        let end = &req.times;
        let steps = end.steps - prefill.steps;
        let p = DecodeProfile {
            prefill_s: (prefill.eval + prefill.sample).as_secs_f64(),
            decode_steps: steps,
            // Shared step wall time (see `StepTimes::eval`), not this request's own cost.
            eval_s: (end.eval - prefill.eval).as_secs_f64(),
            sample_s: (end.sample - prefill.sample).as_secs_f64(),
            emit_s: req.emit.as_secs_f64(),
            gateway_s: 0.0,
            backend: self
                .backend
                .counters()
                .into_iter()
                .zip(counters0)
                .map(|((k, v), (_, v0))| (k.to_owned(), v - v0))
                .collect(),
        };
        if let Some(m) = self.metrics.as_ref().filter(|_| steps > 0) {
            let per = |s: f64| s / f64::from(steps);
            m.decode_stage("eval", per(p.eval_s));
            m.decode_stage("sample", per(p.sample_s));
            m.decode_stage("emit", per(p.emit_s));
        }
        p
    }

    /// Hands the request's remaining events, then `last`, to the drains.
    fn hand_off(&mut self, mut req: Req, last: GenEvent) {
        req.outbox.push_back(last);
        let mut d = Drain {
            tx: req.tx,
            events: req.outbox,
        };
        // Try at once; what the channel does not take now waits in the drain.
        while let Some(ev) = d.events.pop_front() {
            match d.tx.try_send(ev) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Full(ev)) => {
                    d.events.push_front(ev);
                    self.drains.push(d);
                    return;
                }
                Err(mpsc::error::TrySendError::Closed(_)) => return,
            }
        }
    }

    /// A request that never got a sequence (or lost it) ends.
    fn end_unplaced(&mut self, mut req: Req, ev: Result<FinishReason, String>) {
        let last = match ev {
            Ok(reason) => {
                self.finalize(&mut req);
                GenEvent::Finished {
                    reason,
                    stats: req.stats.clone(),
                }
            }
            Err(e) => GenEvent::Error(e),
        };
        self.hand_off(req, last);
    }

    /// The request on `id` ends with `reason`; its sequence stays resident as an idle prefix.
    fn finish(&mut self, id: usize, reason: FinishReason) {
        let Slot::Live(seq) = std::mem::replace(&mut self.slots[id], Slot::Free) else {
            return;
        };
        let Seq { mut req, fed, .. } = *seq;
        self.finalize(&mut req);
        let toks = req.toks[..fed.min(req.toks.len())].to_vec();
        let prompt_len = req.stats.prompt_tokens as usize;
        let stats = req.stats.clone();
        self.hand_off(req, GenEvent::Finished { reason, stats });
        if toks.is_empty() {
            let _ = self.backend.seq_rm(id as SeqId, 0);
        } else {
            self.next_order += 1;
            self.slots[id] = Slot::Idle(Idle {
                toks,
                prompt_len,
                order: self.next_order,
            });
        }
    }

    /// The request on `id` fails; its sequence is freed (the cache may be half-written).
    fn fail(&mut self, id: usize, msg: String) {
        let Slot::Live(seq) = std::mem::replace(&mut self.slots[id], Slot::Free) else {
            return;
        };
        tracing::warn!("generation failed: {msg}");
        let _ = self.backend.seq_rm(id as SeqId, 0);
        self.hand_off(seq.req, GenEvent::Error(msg));
    }

    // ---- saving and evicting ----

    /// Copies sequence `id`, holding `toks`, into the prefix cache if it is worth it.
    fn save(&mut self, id: usize, toks: &[i32]) {
        if !self.prefix_reuse
            || !self.cache.enabled()
            || toks.len() < self.save_min
            || self.cache.holds(&self.backend.kv_format(), toks)
        {
            return;
        }
        let t0 = Instant::now();
        let snap = self.backend.seq_snapshot(id as SeqId);
        let took = t0.elapsed();
        self.save_time += took;
        self.saved.push((id, toks.len()));
        match snap {
            Ok(bytes) => {
                let evicted = self
                    .cache
                    .insert(&self.backend.kv_format(), toks.to_vec(), bytes);
                if let Some(m) = &self.metrics {
                    m.kv_prefix_save(took);
                    m.kv_prefix_evicted(evicted as u64);
                    m.kv_prefix_bytes(self.cache.bytes() as u64);
                }
            }
            Err(e) => tracing::warn!("cannot save a sequence: {e}"),
        }
        self.report_pool();
    }

    /// A sequence of `positions` may be saved later: tell the snapshot pool its size.
    fn expect_snapshot(&self, positions: usize) {
        if self.prefix_reuse && self.cache.enabled() && positions >= self.save_min {
            self.cache
                .pool()
                .expect(positions * self.backend.kv_row_bytes());
        }
    }

    /// Reports what the snapshot pool did since the last call (ADR 0074).
    fn report_pool(&mut self) {
        let s = self.cache.pool().stats();
        if let Some(m) = &self.metrics {
            m.kv_prefix_save_buffers(
                s.from_spare.saturating_sub(self.pool_seen.0),
                s.fallback.saturating_sub(self.pool_seen.1),
            );
            m.kv_prefix_spares(s.ready as u64, s.ready_bytes as u64);
            m.kv_prefix_spare_bytes_worked(
                s.touched_bytes.saturating_sub(self.pool_seen.2),
                s.freed_bytes.saturating_sub(self.pool_seen.3),
            );
        }
        self.pool_seen = (s.from_spare, s.fallback, s.touched_bytes, s.freed_bytes);
    }

    /// Saves the idle sequence on `id` (ADR 0049) and frees its id and rows.
    fn evict(&mut self, id: usize) {
        if let Slot::Idle(idle) = std::mem::replace(&mut self.slots[id], Slot::Free) {
            self.save(id, &idle.toks);
            let _ = self.backend.seq_rm(id as SeqId, 0);
        }
    }

    fn oldest_idle(&self, except: &[usize]) -> Option<usize> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| match s {
                Slot::Idle(x) if !except.contains(&i) => Some((x.order, i)),
                _ => None,
            })
            .min()
            .map(|(_, i)| i)
    }

    // ---- admission ----

    /// Idle sequences whose rows the backend lost are free: nothing is reused from them, and no
    /// request fails for them. Running ones fail on their next use (the backend's guard).
    fn drop_lost_idle(&mut self) {
        for id in 0..self.slots.len() {
            if matches!(self.slots[id], Slot::Idle(_)) && self.backend.seq_lost(id as SeqId) {
                self.slots[id] = Slot::Free;
                let _ = self.backend.seq_rm(id as SeqId, 0);
            }
        }
    }

    fn admit(&mut self) {
        self.drop_lost_idle();
        while let Some(req) = self.waiting.front() {
            if req.cancel.is_cancelled() || req.tx.is_closed() {
                let req = self.waiting.pop_front().expect("front");
                self.end_unplaced(req, Ok(FinishReason::Cancelled));
                continue;
            }
            if req.stats.generated_tokens >= req.params.max_tokens {
                let req = self.waiting.pop_front().expect("front");
                self.end_unplaced(req, Ok(FinishReason::Length));
                continue;
            }
            if !req.toks.is_empty()
                && req.toks.len() >= self.n_ctx_seq
                && req.stats.generated_tokens > 0
            {
                // A preempted request that filled the context: it ends as it would have live.
                let req = self.waiting.pop_front().expect("front");
                self.end_unplaced(req, Ok(FinishReason::Length));
                continue;
            }
            if let Some(&t) = req
                .toks
                .iter()
                .find(|&&t| usize::try_from(t).map_or(true, |t| t >= self.n_vocab))
            {
                let msg = format!("token {t} outside the vocabulary of {}", self.n_vocab);
                let req = self.waiting.pop_front().expect("front");
                self.end_unplaced(req, Err(msg));
                continue;
            }
            if req.toks.is_empty() {
                let req = self.waiting.pop_front().expect("front");
                self.end_unplaced(req, Err("an empty prompt".into()));
                continue;
            }
            if req.toks.len() >= self.n_ctx_seq {
                let n = req.toks.len();
                let msg = format!(
                    "a prompt of {n} tokens does not fit the context of {}",
                    self.n_ctx_seq
                );
                let req = self.waiting.pop_front().expect("front");
                self.hand_off(req, GenEvent::ContextExceeded(msg));
                continue;
            }
            if self.live_count() >= self.slots.len() {
                break;
            }
            match self.place() {
                Admission::Admitted => {}
                Admission::Wait => break,
            }
        }
    }

    /// Gives the request at the queue front a sequence: the idle one sharing the longest prefix,
    /// or a free or evicted id that gets either a copy of a busy sequence's prefix (`seq_cp`) or
    /// the best RAM entry, whichever beats the idle one by `restore_gain` (ADR 0049 per
    /// sequence); ties go to the cheaper source (idle, busy copy, RAM). With `prefix_reuse` off
    /// the id starts empty.
    fn place(&mut self) -> Admission {
        let Some(req) = self.waiting.pop_front() else {
            return Admission::Wait;
        };
        let (t_place, save0, restore0) = (Instant::now(), self.save_time, self.restore_time);
        self.saved.clear();
        let toks = req.toks.clone();
        let last = toks.len() - 1;
        // About what this sequence holds when it is saved later, so a spare is ready by then. Only
        // a stretch of generation counts: clients send max_tokens of tens of thousands.
        let ahead =
            ((req.params.max_tokens - req.stats.generated_tokens) as usize).min(EXPECT_AHEAD);
        self.expect_snapshot((toks.len() + ahead).min(self.n_ctx_seq));
        let fmt = self.backend.kv_format();
        // The idle sequence sharing the longest prefix (at most prompt - 1).
        let best_idle = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| match s {
                Slot::Idle(x) => Some((i, common_prefix(&x.toks, &toks).min(last))),
                _ => None,
            })
            .filter(|&(_, n)| n > 0 && self.prefix_reuse)
            .max_by_key(|&(_, n)| n);
        let live_n = best_idle.map_or(0, |(_, n)| n);
        let ram = if self.prefix_reuse && self.cache.enabled() {
            self.cache
                .best(&fmt, &toks)
                .map(|(i, n)| (i, n.min(last)))
                .filter(|&(_, n)| n >= live_n + self.restore_gain)
        } else {
            None
        };
        // A busy sequence holding the longest prefix, copied when it beats the idle and RAM
        // options by `restore_gain` (a copy costs less than a restore, but more than nothing).
        let ram_n = ram.map_or(0, |(_, n)| n);
        let busy_ok = !req.no_busy;
        let busy = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| match s {
                Slot::Live(x) if self.prefix_reuse && busy_ok => {
                    let held = x.fed.min(self.backend.seq_len(i as SeqId));
                    Some((
                        i,
                        common_prefix(&x.req.toks[..held.min(x.req.toks.len())], &toks).min(last),
                    ))
                }
                _ => None,
            })
            .filter(|&(_, n)| n > 0 && n >= live_n.max(ram_n) + self.restore_gain)
            .max_by_key(|&(_, n)| n);
        // A new conversation sharing little with an idle one: reusing that sequence in place
        // would save it first (seconds at long lengths), so with a free slot the prefix is
        // copied there and the idle sequence stays resident (ADR 0074).
        let free_slot = self.slots.iter().any(|s| matches!(s, Slot::Free));
        let copy_idle = match best_idle {
            Some((i, n))
                if free_slot
                    && ram.is_none()
                    && busy.is_none()
                    && busy_ok
                    && self.cache.enabled() =>
            {
                match &self.slots[i] {
                    Slot::Idle(idle)
                        if matches!(idle.live_cut(n, self.save_min), LiveCut::Save)
                            && !self.cache.holds(&fmt, &idle.toks) =>
                    {
                        Some((i, n))
                    }
                    _ => None,
                }
            }
            _ => None,
        };
        let from_idle = copy_idle.is_some();
        let busy = busy.or(copy_idle);
        let ram = if busy.is_some() { None } else { ram };
        // Take the entry out first: saving a sequence below may evict it.
        let mut picked = ram.map(|(i, _)| self.cache.take(i));
        if picked.is_some()
            && let Some(m) = &self.metrics
        {
            m.kv_prefix_bytes(self.cache.bytes() as u64);
        }
        // Where the request goes.
        let id = match best_idle {
            Some((i, _)) if ram.is_none() && busy.is_none() => i,
            _ => match self.slots.iter().position(|s| matches!(s, Slot::Free)) {
                Some(i) => i,
                None => {
                    let i = self.oldest_idle(&[]).expect("a free or idle id");
                    self.evict(i);
                    i
                }
            },
        };
        // Rows for the prompt and the first stretch of generation.
        let remaining = (req.params.max_tokens - req.stats.generated_tokens) as usize;
        let mut target = (toks.len() + remaining.min(GROW)).min(self.n_ctx_seq);
        // Never evicted to make room here: the chosen id and the idle sequence a free-slot copy
        // reads from.
        let keep: Vec<usize> = std::iter::once(id)
            .chain(copy_idle.map(|(i, _)| i))
            .collect();
        // The least admission takes: the prompt and its first token, and all of a RAM entry it
        // restores (cut to the common prefix afterwards).
        let least = (toks.len() + 1)
            .max(picked.as_ref().map_or(0, |(t, _)| t.len()))
            .min(self.n_ctx_seq);
        loop {
            match self.backend.seq_reserve(id as SeqId, target) {
                Ok(()) => break,
                Err(BackendError::KvFull { needed, free, .. }) => {
                    let idle = self.oldest_idle(&keep);
                    if target > least && (idle.is_some() || self.live_count() == 0) {
                        // The least before an idle sequence is evicted: the steps grow the run as
                        // it generates, evicting then if they must, and may never need the
                        // stretch (ADR 0074, 2026-11-19). With only live ones left the request
                        // waits for the stretch as before, so it is not admitted just to be
                        // preempted at its next row.
                        target = least;
                    } else if let Some(o) = idle {
                        self.evict(o);
                    } else if copy_idle.is_some() {
                        // Only evicting the copy's own source would make room: reuse that sequence
                        // in place instead, which saves it once and restores nothing
                        // (ttft4-20261119 #42 and #102 saved it, failed the copy, restored it).
                        let mut req = req;
                        req.no_busy = true;
                        self.requeue(req, t_place, save0, restore0);
                        return Admission::Admitted;
                    } else if self.live_count() > 0 {
                        // Others will free rows; keep the order of arrival and the entry.
                        self.put_back(picked);
                        self.requeue(req, t_place, save0, restore0);
                        return Admission::Wait;
                    } else if free >= needed && matches!(self.slots[id], Slot::Idle(_)) {
                        // Alone with the rows free: a layout of one run per sequence (ADR 0067)
                        // cannot grow this idle prefix's run where it lies. Free it (saved when
                        // worth it) and place the request again, from row 0 of an empty pool.
                        // The chunked layout slides a lone run down instead (ADR 0074,
                        // 2026-11-19), so this is left to layouts that cannot.
                        tracing::info!(
                            "sequence {id} cannot grow in place ({needed} rows needed, {free} \
                             free): placing its request again"
                        );
                        self.evict(id);
                        self.put_back(picked);
                        self.requeue(req, t_place, save0, restore0);
                        return Admission::Admitted;
                    } else {
                        let n = toks.len();
                        let msg = format!(
                            "a prompt of {n} tokens does not fit the KV cache ({needed} rows \
                             needed, {free} free)"
                        );
                        self.hand_off(req, GenEvent::ContextExceeded(msg));
                        self.put_back(picked);
                        return Admission::Admitted;
                    }
                }
                Err(BackendError::Failed(e)) => {
                    self.end_unplaced(req, Err(e));
                    self.put_back(picked);
                    return Admission::Admitted;
                }
            }
        }
        let mut req = req;
        let mut skipped: Option<&'static str> = None;
        let mut source = "live";
        let mut common = 0;
        if let Some((src, n)) = busy {
            // The chosen id is free here (an idle one was evicted above) and `src` is live.
            let t0 = Instant::now();
            match self.backend.seq_cp(src as SeqId, id as SeqId, n) {
                Ok(()) => {
                    common = n;
                    source = if from_idle { "copy" } else { "seq" };
                    if from_idle && let Some(m) = &self.metrics {
                        m.kv_prefix_placed_free();
                    }
                    if let Some(m) = &self.metrics {
                        let bytes = (n * self.backend.kv_row_bytes()) as u64;
                        m.tier_transfer(
                            bytes,
                            Some(t0.elapsed()),
                            Tier::Vram,
                            Tier::Vram,
                            Purpose::Kv,
                        );
                    }
                }
                Err(e) => {
                    // The copy left `id` empty: release it and place the request again without
                    // the busy source, so it can still use the idle sequence or the RAM entry.
                    tracing::warn!("cannot copy a busy sequence: {e}");
                    let _ = self.backend.seq_rm(id as SeqId, 0);
                    req.no_busy = true;
                    self.requeue(req, t_place, save0, restore0);
                    return Admission::Admitted;
                }
            }
        } else if let Some((etoks, bytes)) = picked.take() {
            // The chosen id is free here (an idle one was evicted above).
            let t0 = Instant::now();
            match self.backend.seq_restore(id as SeqId, &bytes, etoks.len()) {
                Ok(()) => {
                    common = common_prefix(&etoks, &toks).min(last);
                    source = "ram";
                    self.restore_time += t0.elapsed();
                    if let Some(m) = &self.metrics {
                        let took = t0.elapsed();
                        m.kv_prefix_restore(took);
                        m.tier_transfer(
                            bytes.len() as u64,
                            Some(took),
                            Tier::Ram,
                            Tier::Vram,
                            Purpose::Kv,
                        );
                    }
                    self.cache.release(bytes);
                }
                Err(e) => {
                    tracing::warn!("cannot restore a saved sequence: {e}");
                    // Clears the cells even when the sequence looks empty.
                    let _ = self.backend.seq_rm(id as SeqId, 0);
                    // Kept when the pool had no room (it may fit later); a `Failed` restore means
                    // the bytes are unusable, so retrying them would only fail again.
                    if matches!(e, BackendError::KvFull { .. }) {
                        self.put_back(Some((etoks, bytes)));
                    } else {
                        self.cache.release(bytes);
                    }
                }
            }
        } else if let Slot::Idle(idle) = std::mem::replace(&mut self.slots[id], Slot::Free) {
            common = best_idle.map_or(0, |(_, n)| n);
            // The sequence is saved before this request drops much of it, unless the cut is
            // small enough to recompute for less than the save costs, or lies wholly in the
            // tokens the sequence generated itself, which a client may never resend (ADR 0074).
            match idle.live_cut(common, self.save_min) {
                LiveCut::Nothing => {}
                LiveCut::Save => self.save(id, &idle.toks),
                LiveCut::Skip(reason) => {
                    skipped = Some(reason);
                    if self.prefix_reuse
                        && self.cache.enabled()
                        && let Some(m) = &self.metrics
                    {
                        m.kv_prefix_save_skipped(reason);
                    }
                }
            }
        }
        let have = self.backend.seq_len(id as SeqId);
        if common < have
            && let Err(e) = self.backend.seq_rm(id as SeqId, common)
        {
            self.end_unplaced(req, Err(e.to_string()));
            let _ = self.backend.seq_rm(id as SeqId, 0);
            return Admission::Admitted;
        }
        // The restore and the cut may have changed the reservation (the dense backend shrinks it
        // to the rows it keeps), so ask instead of assuming `target`.
        let reserved_to = self.backend.seq_reserved(id as SeqId);
        if !req.started {
            req.started = true;
            req.stats.cached_tokens = common as u32;
            req.bytes0 = self.backend.bytes_streamed_total();
            let params = self.defaults.resolve(&req.params);
            let env = SamplerEnv {
                n_vocab: self.n_vocab,
                eog: &self.vocab.eog,
                model: self.model.as_deref(),
            };
            let seed = req.params.seed.map(|s| s as u32).or(self.default_seed);
            match Sampler::new(&params, seed, &env) {
                Ok(mut s) => {
                    s.accept_all(&req.toks);
                    req.sampler = Some(s);
                }
                Err(e) => {
                    self.end_unplaced(req, Err(e));
                    let _ = self.backend.seq_rm(id as SeqId, 0);
                    return Admission::Admitted;
                }
            }
        }
        if common > 0
            && let Some(m) = &self.metrics
        {
            m.kv_prefix_hit(source);
        }
        req.no_busy = false;
        self.report_pool();
        let earlier = std::mem::take(&mut req.placing);
        let mut saved = earlier.saved;
        saved.append(&mut self.saved);
        let rec = AdmitRecord {
            seq: id,
            queued: t_place
                .saturating_duration_since(req.queued)
                .saturating_sub(earlier.took),
            place: earlier.took + t_place.elapsed(),
            save: earlier.save + (self.save_time - save0),
            restore: earlier.restore + (self.restore_time - restore0),
            attempts: earlier.attempts + 1,
            saved,
            prompt: toks.len(),
            common,
            source,
            skipped,
        };
        self.last_admit = Some(rec);
        if let Some(rec) = &self.last_admit
            && tracing::enabled!(target: "tetraeder::step", tracing::Level::DEBUG)
        {
            let ms = |d: Duration| d.as_secs_f64() * 1e3;
            let saved = rec
                .saved
                .iter()
                .map(|(s, n)| format!("{s}:{n}"))
                .collect::<Vec<_>>()
                .join(",");
            tracing::debug!(
                target: "tetraeder::step",
                "admit seq={} queued_ms={:.2} place_ms={:.2} save_ms={:.2} restore_ms={:.2} prompt={} common={} source={} live={} pool_spare={} pool_fallback={} pool_ready={} attempts={}{}{}",
                rec.seq,
                ms(rec.queued),
                ms(rec.place),
                ms(rec.save),
                ms(rec.restore),
                rec.prompt,
                rec.common,
                rec.source,
                self.live_count(),
                self.pool_seen.0,
                self.pool_seen.1,
                self.cache.pool().stats().ready,
                rec.attempts,
                if saved.is_empty() {
                    String::new()
                } else {
                    format!(" saved={saved}")
                },
                rec.skipped.map_or(String::new(), |r| format!(" skipped={r}")),
            );
        }
        self.next_admitted += 1;
        self.slots[id] = Slot::Live(Box::new(Seq {
            req,
            fed: common,
            admitted: self.next_admitted,
            decoding: false,
            reserved_to,
            drafts: None,
        }));
        Admission::Admitted
    }

    /// Puts a request back at the queue front after a placement attempt that began at `t0` with
    /// the worker's save and restore totals at `save0` and `restore0`; the attempt's cost goes
    /// to the request's admission log.
    fn requeue(&mut self, mut req: Req, t0: Instant, save0: Duration, restore0: Duration) {
        let p = &mut req.placing;
        p.attempts += 1;
        p.took += t0.elapsed();
        p.save += self.save_time - save0;
        p.restore += self.restore_time - restore0;
        p.saved.append(&mut self.saved);
        self.waiting.push_front(req);
    }

    /// Returns an entry taken for a restore that did not happen.
    fn put_back(&mut self, entry: Option<(Vec<i32>, Vec<u8>)>) {
        if let Some((toks, bytes)) = entry {
            self.cache.insert(&self.backend.kv_format(), toks, bytes);
        }
    }

    // ---- per iteration ----

    /// Delivers pending events and ends cancelled requests.
    fn sweep(&mut self) {
        for id in 0..self.slots.len() {
            let Slot::Live(seq) = &mut self.slots[id] else {
                continue;
            };
            let open = seq.req.flush();
            if !open || seq.req.cancel.is_cancelled() || seq.req.tx.is_closed() {
                seq.req.outbox.clear();
                self.finish(id, FinishReason::Cancelled);
            }
        }
    }

    fn live_view(&self) -> Vec<Live> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| match s {
                Slot::Live(s) => Some(Live {
                    seq: i as SeqId,
                    pending: s.req.toks.len() - s.fed + s.drafts.as_ref().map_or(0, Vec::len),
                    decoding: s.decoding,
                    paused: s.paused(),
                    admitted: s.admitted,
                }),
                _ => None,
            })
            .collect()
    }

    /// Plans and runs one step. False when nothing could run (everything paused).
    fn step(&mut self) -> bool {
        'plan: loop {
            self.draft();
            let live = self.live_view();
            let mut takes = plan_step(self.budget, &live);
            if takes.is_empty() {
                return false;
            }
            // Retrying one sequence at a time after a failed step: the next one still live.
            while let Some(s) = self.solo.pop_front() {
                if let Some(&t) = takes.iter().find(|t| t.seq == s) {
                    takes = vec![t];
                    break;
                }
            }
            // Rows for the growth of each sequence.
            for t in &takes {
                let id = t.seq as usize;
                let len = self.backend.seq_len(t.seq);
                let want = len + t.n;
                let Slot::Live(seq) = &mut self.slots[id] else {
                    continue;
                };
                if want > self.n_ctx_seq {
                    self.finish(id, FinishReason::Length);
                    continue 'plan;
                }
                if want <= seq.reserved_to {
                    continue;
                }
                let grown = (want + GROW).min(self.n_ctx_seq);
                let mut r = self.backend.seq_reserve(t.seq, grown);
                let mut got = grown;
                if matches!(r, Err(BackendError::KvFull { .. })) {
                    r = self.backend.seq_reserve(t.seq, want);
                    got = want;
                }
                match r {
                    Ok(()) => seq.reserved_to = got,
                    Err(BackendError::KvFull { needed, free, .. }) => {
                        self.make_room(id, free >= needed);
                        continue 'plan;
                    }
                    Err(BackendError::Failed(e)) => {
                        self.fail(id, e);
                        continue 'plan;
                    }
                }
            }
            // The step.
            // A sequence with drafts sends its pending token and the drafts, with the logits of
            // every row to verify them.
            let bufs: Vec<Option<Vec<i32>>> = takes
                .iter()
                .map(|t| {
                    let Slot::Live(seq) = &self.slots[t.seq as usize] else {
                        unreachable!("planned sequences are live");
                    };
                    seq.drafts.as_ref().filter(|d| !d.is_empty()).map(|d| {
                        let mut b = vec![seq.req.toks[seq.fed]];
                        b.extend_from_slice(d);
                        b
                    })
                })
                .collect();
            let mut inputs = Vec::with_capacity(takes.len());
            for (t, buf) in takes.iter().zip(&bufs) {
                let Slot::Live(seq) = &self.slots[t.seq as usize] else {
                    unreachable!("planned sequences are live");
                };
                inputs.push(SeqInput {
                    seq: t.seq,
                    tokens: buf
                        .as_deref()
                        .unwrap_or_else(|| &seq.req.toks[seq.fed..seq.fed + t.n]),
                    all_logits: buf.is_some(),
                });
            }
            let total: usize = takes.iter().map(|t| t.n).sum();
            let t0 = Instant::now();
            let res = self.backend.decode(&inputs);
            let took = t0.elapsed();
            let out = match res {
                Ok(out) => out,
                Err(BackendError::KvFull { seq, .. }) => {
                    self.make_room(seq as usize, false);
                    continue 'plan;
                }
                Err(BackendError::Failed(e)) if takes.len() > 1 => {
                    // Lengths are unchanged, so each sequence runs alone in the next steps and
                    // only the one the backend rejects fails.
                    tracing::warn!(
                        "step of {} sequences failed, retrying alone: {e}",
                        takes.len()
                    );
                    self.solo = takes.iter().map(|t| t.seq).collect();
                    continue 'plan;
                }
                Err(BackendError::Failed(e)) => {
                    self.fail(takes[0].seq as usize, e);
                    return true;
                }
            };
            // Sample the sequences whose prompt is complete, in sequence order.
            let mut sampled: Vec<Option<Result<Vec<u32>, String>>> =
                Vec::with_capacity(takes.len());
            let mut sample_times = Vec::with_capacity(takes.len());
            let nv = self.n_vocab;
            for (k, t) in takes.iter().enumerate() {
                let Slot::Live(seq) = &mut self.slots[t.seq as usize] else {
                    unreachable!("planned sequences are live");
                };
                if let Some(d) = bufs[k].as_ref().map(|b| &b[1..]) {
                    // Row i holds the logits after the pending token and i drafts. The token
                    // chosen there is the one plain decoding would emit if the drafts before it
                    // were right; the chain ends at the first draft it differs from.
                    let t1 = Instant::now();
                    let rows = out.rows(k);
                    let mut chain = Ok(Vec::with_capacity(d.len() + 1));
                    for i in 0..=d.len() {
                        match sample_one(&mut seq.req, &rows[i * nv..(i + 1) * nv]) {
                            Ok(tok) => {
                                let more = d.get(i).is_some_and(|&x| x == tok as i32);
                                chain.as_mut().expect("ok").push(tok);
                                if !more {
                                    break;
                                }
                            }
                            Err(e) => {
                                chain = Err(e);
                                break;
                            }
                        }
                    }
                    sampled.push(Some(chain));
                    sample_times.push(t1.elapsed());
                } else if seq.fed + t.n == seq.req.toks.len() {
                    let t1 = Instant::now();
                    sampled.push(Some(sample_one(&mut seq.req, out.last(k)).map(|t| vec![t])));
                    sample_times.push(t1.elapsed());
                } else {
                    sampled.push(None);
                    sample_times.push(Duration::ZERO);
                }
            }
            let (mut dec, mut pre, mut dec_seqs) = (0u32, 0u32, 0usize);
            for t in &takes {
                let Slot::Live(seq) = &self.slots[t.seq as usize] else {
                    continue;
                };
                if seq.decoding {
                    dec += t.n as u32;
                    dec_seqs += 1;
                } else {
                    pre += t.n as u32;
                }
            }
            if let Some(m) = &self.metrics {
                m.scheduler_batch(takes.len() as u32);
                if dec > 0 {
                    m.scheduler_step_tokens("decode", dec);
                }
                if pre > 0 {
                    m.scheduler_step_tokens("prefill", pre);
                }
            }
            for (k, t) in takes.iter().enumerate() {
                let id = t.seq as usize;
                let Slot::Live(seq) = &mut self.slots[id] else {
                    continue;
                };
                let before = seq.fed;
                seq.fed += t.n;
                seq.req.times.eval += took;
                seq.req.gpu += took.mul_f64(t.n as f64 / total as f64);
                let drafts = seq.drafts.take();
                match sampled[k].take() {
                    None => {}
                    Some(Err(e)) => self.fail(id, e),
                    Some(Ok(chain)) => {
                        seq.req.times.sample += sample_times[k];
                        seq.req.times.steps += 1;
                        match drafts {
                            Some(d) => {
                                self.on_chain(id, before, &chain, d.len(), took, takes.len() == 1);
                            }
                            None => self.on_token(id, chain[0]),
                        }
                    }
                }
            }
            self.log_step(t0, took, &sample_times, dec_seqs, dec, takes.len(), pre);
            return true;
        }
    }

    /// Proposes drafts for the sequences that speculate in the next step: decoding, not paused,
    /// greedy and unconstrained (so the chosen token is the argmax, with no sampler state), and
    /// at most `max_batch` of them decoding. They are kept until the step has run.
    fn draft(&mut self) {
        let Some(spec) = self.spec.as_mut() else {
            return;
        };
        let decoding = self
            .slots
            .iter()
            .filter(|s| matches!(s, Slot::Live(s) if s.decoding && !s.paused()))
            .count();
        for slot in &mut self.slots {
            let Slot::Live(seq) = slot else { continue };
            let greedy = self.defaults.resolve(&seq.req.params).is_plain_greedy();
            let eligible = decoding <= spec.max_batch
                && seq.decoding
                && !seq.paused()
                && greedy
                && seq.req.constraint.is_none()
                && seq.req.toks.len() == seq.fed + 1;
            if !eligible {
                seq.drafts = None;
            } else if seq.drafts.is_none() {
                // Rows past the context or the token limit would only be cut again.
                let left = seq
                    .req
                    .params
                    .max_tokens
                    .saturating_sub(seq.req.stats.generated_tokens)
                    as usize;
                let cap = left
                    .saturating_sub(1)
                    .min(self.n_ctx_seq.saturating_sub(seq.fed + 1));
                let drafts = spec
                    .speculator
                    .propose(spec.drafter.as_mut(), &seq.req.toks, cap)
                    .unwrap_or_else(|e| {
                        tracing::warn!("drafting failed: {e}");
                        Vec::new()
                    });
                seq.drafts = Some(drafts);
            }
        }
    }

    /// Emits the tokens a verify step chose for `id` and keeps in its KV the pending token and
    /// the drafts that were accepted. `before` is the sequence's length before the step, `chain`
    /// the chosen tokens (one more than the accepted drafts). Ends early when a token ends the
    /// request; what was fed beyond that is cut like a rejected draft.
    fn on_chain(
        &mut self,
        id: usize,
        before: usize,
        chain: &[u32],
        drafted: usize,
        took: Duration,
        alone: bool,
    ) {
        let mut last = 0;
        for (i, &tok) in chain.iter().enumerate() {
            let Slot::Live(seq) = &mut self.slots[id] else {
                break;
            };
            last = i;
            seq.fed = before + 1 + i;
            self.on_token(id, tok);
        }
        let accepted = chain.len() - 1;
        if drafted > 0
            && let (Some(m), Some(spec)) = (&self.metrics, &self.spec)
        {
            m.spec_step(spec.drafter.method(), drafted as u64, accepted as u64);
        }
        if let Some(spec) = &mut self.spec
            && alone
        {
            spec.speculator.record(1 + drafted, drafted, accepted, took);
        }
        let keep = before + 1 + last;
        if self.backend.seq_len(id as SeqId) > keep
            && let Err(e) = self.backend.seq_rm(id as SeqId, keep)
        {
            if matches!(self.slots[id], Slot::Live(_)) {
                self.fail(id, e.to_string());
            } else {
                let _ = self.backend.seq_rm(id as SeqId, 0);
                self.slots[id] = Slot::Free;
            }
            return;
        }
        if let Slot::Live(seq) = &mut self.slots[id] {
            seq.reserved_to = self.backend.seq_reserved(id as SeqId);
        }
    }

    /// One line per step at `RUST_LOG=tetraeder::step=debug`: its shape, decode wall time, the
    /// time before it (idle gap since the last step ended) and after it (sampling and emitting).
    #[allow(clippy::too_many_arguments)]
    fn log_step(
        &mut self,
        t0: Instant,
        took: Duration,
        sample_times: &[Duration],
        dec_seqs: usize,
        dec: u32,
        seqs: usize,
        pre: u32,
    ) {
        let end = Instant::now();
        let gap = self.last_step.map(|l| t0.saturating_duration_since(l));
        self.last_step = Some(end);
        if !tracing::enabled!(target: "tetraeder::step", tracing::Level::DEBUG) {
            return;
        }
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        let paused = self
            .slots
            .iter()
            .filter(|s| matches!(s, Slot::Live(s) if s.paused()))
            .count();
        let idle = self
            .slots
            .iter()
            .filter(|s| matches!(s, Slot::Idle(_)))
            .count();
        tracing::debug!(
            target: "tetraeder::step",
            "step dec_seqs={dec_seqs} dec_toks={dec} pre_seqs={} pre_toks={pre} decode_ms={:.2}              gap_ms={:.2} sample_ms={:.2} post_ms={:.2} live={} paused={paused} idle={idle} waiting={}",
            seqs - dec_seqs,
            ms(took),
            gap.map_or(-1.0, ms),
            ms(sample_times.iter().sum()),
            ms(end - t0 - took),
            self.live_count(),
            self.waiting.len(),
        );
    }

    /// A token was sampled on `id`: emit it, or end the request.
    fn on_token(&mut self, id: usize, tok: u32) {
        if self.decode_profile
            && let Slot::Live(seq) = &mut self.slots[id]
            && seq.req.after_prefill.is_none()
        {
            seq.req.after_prefill = Some((seq.req.times, self.backend.counters()));
        }
        let Slot::Live(seq) = &mut self.slots[id] else {
            return;
        };
        seq.decoding = true;
        if self.vocab.is_eog(tok) {
            self.finish(id, FinishReason::EndOfGeneration);
            return;
        }
        seq.req.stats.generated_tokens += 1;
        seq.req.toks.push(tok as i32);
        let t0 = Instant::now();
        seq.req.outbox.push_back(GenEvent::Token(tok));
        let open = seq.req.flush();
        seq.req.emit += t0.elapsed();
        if !open {
            seq.req.outbox.clear();
            self.finish(id, FinishReason::Cancelled);
        } else if seq.req.stats.generated_tokens >= seq.req.params.max_tokens
            || seq.fed >= self.n_ctx_seq
        {
            self.finish(id, FinishReason::Length);
        }
    }

    /// Frees rows for `needing` after `KvFull`: idle sequences first (oldest first, saved to
    /// RAM), then the most recently admitted other live sequence is preempted and requeued. If
    /// that one is older than `needing`, `needing` is requeued itself, so the oldest request
    /// always progresses. With no other sequence the request ends with `Length`, unless the
    /// rows are free and only the layout is `fragmented`: then it is requeued itself.
    fn make_room(&mut self, needing: usize, fragmented: bool) {
        let (saves0, save0) = (self.saved.len(), self.save_time);
        // Slots whose first token this step's saves delay (ttft_join.py reads them).
        let prefilling: Vec<String> = (0..self.slots.len())
            .filter(|&i| matches!(&self.slots[i], Slot::Live(s) if !s.decoding))
            .map(|i| i.to_string())
            .collect();
        let reason = self.make_room_by(needing, fragmented);
        if self.saved.len() == saves0 {
            return;
        }
        let took = self.save_time - save0;
        self.step_saves += 1;
        self.step_save_time += took;
        if tracing::enabled!(target: "tetraeder::step", tracing::Level::DEBUG) {
            tracing::debug!(
                target: "tetraeder::step",
                "step save reason={reason} seq={} save_ms={:.2} prefilling={}",
                self.saved[saves0].0,
                took.as_secs_f64() * 1e3,
                prefilling.join(","),
            );
        }
    }

    /// Frees rows for `needing`; returns how, for the trace.
    fn make_room_by(&mut self, needing: usize, fragmented: bool) -> &'static str {
        if let Some(i) = self.oldest_idle(&[]) {
            self.evict(i);
            return "evict";
        }
        let live = self.live_view();
        let admitted = |id: SeqId| live.iter().find(|l| l.seq == id).map_or(0, |l| l.admitted);
        match victim(&live, needing as SeqId) {
            Some(v) if admitted(v) > admitted(needing as SeqId) => self.preempt(v as usize),
            Some(_) => self.preempt(needing),
            // Alone, and the rows are free but the layout cannot grow the run where it lies
            // (ADR 0067): placed again, the sequence starts at row 0 of an empty pool.
            None if fragmented => {
                tracing::info!("sequence {needing} cannot grow in place: requeued");
                self.preempt(needing);
            }
            None => self.finish(needing, FinishReason::Length),
        }
        "preempt"
    }

    /// Saves the sequence on `id` if it is worth it, frees it and puts its request back at the
    /// queue front with its generated tokens as part of the prompt.
    fn preempt(&mut self, id: usize) {
        let Slot::Live(seq) = std::mem::replace(&mut self.slots[id], Slot::Free) else {
            return;
        };
        let Seq { req, fed, .. } = *seq;
        let toks = req.toks[..fed].to_vec();
        self.save(id, &toks);
        let _ = self.backend.seq_rm(id as SeqId, 0);
        if let Some(m) = &self.metrics {
            m.scheduler_preemption();
        }
        self.waiting.push_front(req);
    }
}

#[cfg(test)]
mod tests;
