//! The serving scheduler: a worker thread owning a [`BatchBackend`], a job queue, continuous
//! batching of up to `parallel` sequences with chunked prefill under a step token budget, prefix
//! reuse of idle resident sequences and of saved ones in host RAM, and Rust sampling over the
//! logits (ADR 0046, ADR 0065).

use crate::backend::{BackendError, BatchBackend, SeqId, SeqInput};
use crate::batch::{Live, plan_step, victim};
use crate::load::{Loaded, model_id};
use crate::prefix_cache::{PrefixCache, RESTORE_MIN_GAIN, SAVE_MIN_TOKENS, common_prefix};
use crate::sample::{SampleParams, Sampler, counter_seed};
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

/// Default tokens per step (`--step-tokens`): the unit of cancellation during prefill.
pub const STEP_TOKENS: usize = 512;
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
    /// Defaults for requests that send none (llama-server's: 0.8, 40, 0.95).
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
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
}

/// Default prefix cache budget, like llama-server's `--cache-ram`.
pub const DEFAULT_CACHE_RAM_BYTES: usize = 8192 << 20;

impl SchedulerConfig {
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            alias: None,
            model_path: model_path.into(),
            chat_template: None,
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            metrics: None,
            cache_ram_bytes: DEFAULT_CACHE_RAM_BYTES,
            cache_save_min: SAVE_MIN_TOKENS,
            cache_restore_gain: RESTORE_MIN_GAIN,
            decode_profile: std::env::var_os("TETRAEDER_DECODE_PROFILE").is_some_and(|v| v == "1"),
            parallel: PARALLEL,
            step_tokens: STEP_TOKENS,
            prefix_reuse: true,
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
        let defaults = SampleParams {
            temperature: cfg.temperature,
            top_k: cfg.top_k,
            top_p: cfg.top_p,
        };
        let metrics = cfg.metrics.clone();
        let decode_profile = cfg.decode_profile;
        let (parallel, step_tokens) = (cfg.parallel.max(1), cfg.step_tokens.max(1));
        let prefix_reuse = cfg.prefix_reuse;
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
                let _ = ready_tx.send(Ok((Arc::clone(&loaded.tokenizer), n_ctx)));
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
    /// Created at the first admission, in arrival order of admissions (`counter_seed`).
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
}

impl Seq {
    fn paused(&self) -> bool {
        !self.req.outbox.is_empty()
    }
}

/// A finished request's sequence, kept as a prefix until its rows or id are needed.
struct Idle {
    toks: Vec<i32>,
    /// Retirement order, smaller is older.
    order: u64,
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
            metrics,
            decode_profile,
            budget,
            n_ctx_seq,
            last_step: None,
            n_vocab,
            solo: VecDeque::new(),
        }
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
        let stats = req.stats.clone();
        self.hand_off(req, GenEvent::Finished { reason, stats });
        if toks.is_empty() {
            let _ = self.backend.seq_rm(id as SeqId, 0);
        } else {
            self.next_order += 1;
            self.slots[id] = Slot::Idle(Idle {
                toks,
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
        match self.backend.seq_snapshot(id as SeqId) {
            Ok(bytes) => {
                let evicted = self
                    .cache
                    .insert(&self.backend.kv_format(), toks.to_vec(), bytes);
                if let Some(m) = &self.metrics {
                    m.kv_prefix_evicted(evicted as u64);
                    m.kv_prefix_bytes(self.cache.bytes() as u64);
                }
            }
            Err(e) => tracing::warn!("cannot save a sequence: {e}"),
        }
    }

    /// Saves the idle sequence on `id` (ADR 0049) and frees its id and rows.
    fn evict(&mut self, id: usize) {
        if let Slot::Idle(idle) = std::mem::replace(&mut self.slots[id], Slot::Free) {
            self.save(id, &idle.toks);
            let _ = self.backend.seq_rm(id as SeqId, 0);
        }
    }

    fn oldest_idle(&self, except: Option<usize>) -> Option<usize> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| match s {
                Slot::Idle(x) if Some(i) != except => Some((x.order, i)),
                _ => None,
            })
            .min()
            .map(|(_, i)| i)
    }

    // ---- admission ----

    fn admit(&mut self) {
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
            if req.toks.is_empty() || req.toks.len() >= self.n_ctx_seq {
                let n = req.toks.len();
                let msg = format!(
                    "a prompt of {n} tokens does not fit the context of {}",
                    self.n_ctx_seq
                );
                let req = self.waiting.pop_front().expect("front");
                self.end_unplaced(req, Err(msg));
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
        let toks = req.toks.clone();
        let last = toks.len() - 1;
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
        let busy = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| match s {
                Slot::Live(x) if self.prefix_reuse => {
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
                    let i = self.oldest_idle(None).expect("a free or idle id");
                    self.evict(i);
                    i
                }
            },
        };
        // Rows for the prompt and the first stretch of generation.
        let remaining = (req.params.max_tokens - req.stats.generated_tokens) as usize;
        let mut target = (toks.len() + remaining.min(GROW)).min(self.n_ctx_seq);
        loop {
            match self.backend.seq_reserve(id as SeqId, target) {
                Ok(()) => break,
                Err(BackendError::KvFull { .. }) => {
                    if let Some(o) = self.oldest_idle(Some(id)) {
                        self.evict(o);
                    } else if self.live_count() > 0 {
                        // Others will free rows; keep the order of arrival and the entry.
                        self.waiting.push_front(req);
                        self.put_back(picked);
                        return Admission::Wait;
                    } else if target > toks.len() + 1 {
                        // Alone: what the pool still holds, the context ends the request.
                        target = toks.len() + 1;
                    } else {
                        let msg = format!(
                            "a prompt of {} tokens does not fit the KV cache",
                            toks.len()
                        );
                        self.end_unplaced(req, Err(msg));
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
        let mut source = "live";
        let mut common = 0;
        if let Some((src, n)) = busy {
            // The chosen id is free here (an idle one was evicted above) and `src` is live.
            let t0 = Instant::now();
            match self.backend.seq_cp(src as SeqId, id as SeqId, n) {
                Ok(()) => {
                    common = n;
                    source = "seq";
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
                    // A failed copy leaves `id` empty with its reservation.
                    tracing::warn!("cannot copy a busy sequence: {e}");
                }
            }
        } else if let Some((etoks, bytes)) = picked.take() {
            // The chosen id is free here (an idle one was evicted above).
            let t0 = Instant::now();
            match self.backend.seq_restore(id as SeqId, &bytes, etoks.len()) {
                Ok(()) => {
                    common = common_prefix(&etoks, &toks).min(last);
                    source = "ram";
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
                }
                Err(e) => {
                    tracing::warn!("cannot restore a saved sequence: {e}");
                    // Clears the cells even when the sequence looks empty.
                    let _ = self.backend.seq_rm(id as SeqId, 0);
                    // Kept when the pool had no room (it may fit later); a `Failed` restore means
                    // the bytes are unusable, so retrying them would only fail again.
                    if matches!(e, BackendError::KvFull { .. }) {
                        self.put_back(Some((etoks, bytes)));
                    }
                }
            }
        } else if let Slot::Idle(idle) = std::mem::replace(&mut self.slots[id], Slot::Free) {
            common = best_idle.map_or(0, |(_, n)| n);
            // The sequence is saved before this request drops much of it.
            if idle.toks.len() - common.min(idle.toks.len()) >= self.save_min {
                self.save(id, &idle.toks);
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
            req.sampler = Some(Sampler::new(
                SampleParams {
                    temperature: req.params.temperature.unwrap_or(self.defaults.temperature),
                    top_k: req.params.top_k.unwrap_or(self.defaults.top_k),
                    top_p: req.params.top_p.unwrap_or(self.defaults.top_p),
                },
                req.params.seed.unwrap_or_else(counter_seed),
            ));
        }
        if common > 0
            && let Some(m) = &self.metrics
        {
            m.kv_prefix_hit(source);
        }
        self.next_admitted += 1;
        self.slots[id] = Slot::Live(Box::new(Seq {
            req,
            fed: common,
            admitted: self.next_admitted,
            decoding: false,
            reserved_to,
        }));
        Admission::Admitted
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
                    pending: s.req.toks.len() - s.fed,
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
                    Err(BackendError::KvFull { .. }) => {
                        self.make_room(id);
                        continue 'plan;
                    }
                    Err(BackendError::Failed(e)) => {
                        self.fail(id, e);
                        continue 'plan;
                    }
                }
            }
            // The step.
            let mut inputs = Vec::with_capacity(takes.len());
            for t in &takes {
                let Slot::Live(seq) = &self.slots[t.seq as usize] else {
                    unreachable!("planned sequences are live");
                };
                inputs.push(SeqInput {
                    seq: t.seq,
                    tokens: &seq.req.toks[seq.fed..seq.fed + t.n],
                    all_logits: false,
                });
            }
            let total: usize = takes.iter().map(|t| t.n).sum();
            let t0 = Instant::now();
            let res = self.backend.decode(&inputs);
            let took = t0.elapsed();
            let out = match res {
                Ok(out) => out,
                Err(BackendError::KvFull { seq, .. }) => {
                    self.make_room(seq as usize);
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
            let mut sampled: Vec<Option<Result<u32, String>>> = Vec::with_capacity(takes.len());
            let mut sample_times = Vec::with_capacity(takes.len());
            for (k, t) in takes.iter().enumerate() {
                let Slot::Live(seq) = &mut self.slots[t.seq as usize] else {
                    unreachable!("planned sequences are live");
                };
                if seq.fed + t.n == seq.req.toks.len() {
                    let t1 = Instant::now();
                    sampled.push(Some(sample_one(&mut seq.req, out.last(k))));
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
                seq.fed += t.n;
                seq.req.times.eval += took;
                seq.req.gpu += took.mul_f64(t.n as f64 / total as f64);
                match sampled[k].take() {
                    None => {}
                    Some(Err(e)) => self.fail(id, e),
                    Some(Ok(tok)) => {
                        seq.req.times.sample += sample_times[k];
                        seq.req.times.steps += 1;
                        self.on_token(id, tok);
                    }
                }
            }
            self.log_step(t0, took, &sample_times, dec_seqs, dec, takes.len(), pre);
            return true;
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
        tracing::debug!(
            target: "tetraeder::step",
            "step dec_seqs={dec_seqs} dec_toks={dec} pre_seqs={} pre_toks={pre} decode_ms={:.2}              gap_ms={:.2} sample_ms={:.2} post_ms={:.2} live={} paused={paused} waiting={}",
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
    /// always progresses. With no other sequence the request ends with `Length`.
    fn make_room(&mut self, needing: usize) {
        if let Some(i) = self.oldest_idle(None) {
            self.evict(i);
            return;
        }
        let live = self.live_view();
        let admitted = |id: SeqId| live.iter().find(|l| l.seq == id).map_or(0, |l| l.admitted);
        match victim(&live, needing as SeqId) {
            Some(v) if admitted(v) > admitted(needing as SeqId) => self.preempt(v as usize),
            Some(_) => self.preempt(needing),
            None => self.finish(needing, FinishReason::Length),
        }
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
