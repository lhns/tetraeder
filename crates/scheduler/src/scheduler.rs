//! The one-sequence scheduler: a worker thread owning the backend, a job queue, prefix reuse
//! of the last sequence, Rust sampling over the logits (ADR 0046).

use crate::backend::Backend;
use crate::load::{Loaded, model_id};
use crate::prefix_cache::{PrefixCache, RESTORE_MIN_GAIN, SAVE_MIN_TOKENS, common_prefix};
use crate::sample::{SampleParams, Sampler, counter_seed};
use std::path::PathBuf;
use std::sync::mpsc as std_mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tetraeder_gateway::engine::{
    DecodeProfile, Engine, EngineError, EngineStats, FinishReason, GenEvent, GenerateRequest,
    ModelInfo, SamplingParams, TokenConstraint, TokenId, Vocab,
};
use tetraeder_ggml_sys::llama::Model;
use tetraeder_telemetry::Metrics;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// Prompt tokens per backend call: the unit of cancellation during prefill.
pub const PREFILL_CHUNK: usize = 512;

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
        }
    }
}

struct Job {
    prompt: Vec<TokenId>,
    params: SamplingParams,
    constraint: Option<Box<dyn TokenConstraint>>,
    cancel: CancellationToken,
    tx: mpsc::Sender<GenEvent>,
}

/// The gateway's [`Engine`] over one backend.
pub struct Scheduler {
    info: ModelInfo,
    vocab: Arc<Vocab>,
    tokenizer: Arc<Model>,
    queue: Mutex<std_mpsc::Sender<Job>>,
}

impl Scheduler {
    /// Starts the worker, which runs `load` on its own thread (the MoE builder may not be
    /// `Send`) and owns the backend from then on. Returns when the model is loaded, or with
    /// the load error.
    pub fn start(
        cfg: SchedulerConfig,
        load: impl FnOnce() -> Result<Loaded, String> + Send + 'static,
    ) -> Result<Self, String> {
        let (job_tx, job_rx) = std_mpsc::channel::<Job>();
        let (ready_tx, ready_rx) = std_mpsc::channel::<Result<(Arc<Model>, usize), String>>();
        let defaults = SampleParams {
            temperature: cfg.temperature,
            top_k: cfg.top_k,
            top_p: cfg.top_p,
        };
        let metrics = cfg.metrics.clone();
        let decode_profile = cfg.decode_profile;
        let (cache_ram, save_min, restore_gain) = (
            cfg.cache_ram_bytes,
            cfg.cache_save_min,
            cfg.cache_restore_gain,
        );
        let vocab_cell: Arc<std::sync::OnceLock<Arc<Vocab>>> = Arc::default();
        let worker_vocab = Arc::clone(&vocab_cell);
        std::thread::Builder::new()
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
                let n_ctx = loaded.backend.n_ctx();
                let _ = ready_tx.send(Ok((Arc::clone(&loaded.tokenizer), n_ctx)));
                let mut w = Worker {
                    backend: loaded.backend,
                    cached: Vec::new(),
                    cache: PrefixCache::new(cache_ram),
                    save_min,
                    restore_gain,
                    vocab,
                    defaults,
                    metrics,
                    decode_profile,
                };
                for job in job_rx {
                    w.run(job);
                }
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
        })
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
        let job = Job {
            prompt: req.prompt,
            params: req.params,
            constraint: req.constraint,
            cancel: req.cancel,
            tx,
        };
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

struct Worker {
    backend: Box<dyn Backend>,
    /// The tokens in the backend's KV cache, in order.
    cached: Vec<i32>,
    /// Sequences evicted from the KV cache, kept in host RAM.
    cache: PrefixCache,
    save_min: usize,
    restore_gain: usize,
    vocab: Arc<Vocab>,
    defaults: SampleParams,
    metrics: Option<Metrics>,
    decode_profile: bool,
}

/// Time of the steps of one request so far.
#[derive(Debug, Clone, Copy, Default)]
struct StepTimes {
    /// Steps that sampled a token.
    steps: u32,
    /// Backend `eval` calls (device time on a GPU backend).
    eval: Duration,
    /// Constraint mask, sampling and commit.
    sample: Duration,
}

enum Step {
    Token(u32),
    Cancelled,
}

impl Worker {
    fn run(&mut self, job: Job) {
        let Job {
            prompt,
            params,
            mut constraint,
            cancel,
            tx,
        } = job;
        let mut stats = EngineStats {
            prompt_tokens: u32::try_from(prompt.len()).unwrap_or(u32::MAX),
            ..EngineStats::default()
        };
        if cancel.is_cancelled() || tx.is_closed() {
            let _ = tx.blocking_send(GenEvent::Finished {
                reason: FinishReason::Cancelled,
                stats,
            });
            return;
        }
        let ev = match self.generate(&prompt, &params, &mut constraint, &cancel, &tx, &mut stats) {
            Ok(reason) => GenEvent::Finished { reason, stats },
            Err(e) => {
                // The cache may be half-written: start the next request cold.
                let _ = self.backend.truncate(0);
                self.cached.clear();
                tracing::warn!("generation failed: {e}");
                GenEvent::Error(e)
            }
        };
        let _ = tx.blocking_send(ev);
    }

    fn generate(
        &mut self,
        prompt: &[TokenId],
        params: &SamplingParams,
        constraint: &mut Option<Box<dyn TokenConstraint>>,
        cancel: &CancellationToken,
        tx: &mpsc::Sender<GenEvent>,
        stats: &mut EngineStats,
    ) -> Result<FinishReason, String> {
        let n_ctx = self.backend.n_ctx();
        if prompt.is_empty() || prompt.len() >= n_ctx {
            return Err(format!(
                "a prompt of {} tokens does not fit the context of {n_ctx}",
                prompt.len()
            ));
        }
        let toks: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
        // Keep the shared prefix, but evaluate at least the last prompt token for logits.
        let last = toks.len() - 1;
        let live = common_prefix(&self.cached, &toks).min(last);
        let ram = if self.cache.enabled() {
            self.cache
                .best(&toks)
                .map(|(i, n)| (i, n.min(last)))
                .filter(|&(_, n)| n >= live + self.restore_gain)
        } else {
            None
        };
        // Take the entry out first: saving the live sequence may evict it.
        let picked = ram.map(|(i, _)| self.cache.take(i));
        if picked.is_some()
            && let Some(m) = &self.metrics
        {
            m.kv_prefix_bytes(self.cache.bytes() as u64);
        }
        // The live sequence is saved before truncation or a restore drops much of it.
        let lost = if picked.is_some() {
            self.cached.len()
        } else {
            self.cached.len() - live.min(self.cached.len())
        };
        if self.cache.enabled() && lost >= self.save_min && !self.cache.holds(&self.cached) {
            self.save_live();
        }
        let mut restored = false;
        if let Some((etoks, bytes)) = picked {
            let t0 = Instant::now();
            match self.backend.restore(&bytes, etoks.len()) {
                Ok(()) => {
                    self.cached = etoks;
                    restored = true;
                    if let Some(m) = &self.metrics {
                        m.kv_prefix_restore(t0.elapsed());
                    }
                }
                Err(e) => {
                    tracing::warn!("cannot restore a saved sequence: {e}");
                    self.backend.truncate(0)?;
                    self.cached.clear();
                }
            }
        }
        let common = common_prefix(&self.cached, &toks).min(last);
        if common < self.cached.len() {
            self.backend.truncate(common)?;
            self.cached.truncate(common);
        }
        stats.cached_tokens = common as u32;
        if common > 0
            && let Some(m) = &self.metrics
        {
            m.kv_prefix_hit(if restored { "ram" } else { "live" });
        }

        let mut sampler = Sampler::new(
            SampleParams {
                temperature: params.temperature.unwrap_or(self.defaults.temperature),
                top_k: params.top_k.unwrap_or(self.defaults.top_k),
                top_p: params.top_p.unwrap_or(self.defaults.top_p),
            },
            params.seed.unwrap_or_else(counter_seed),
        );
        let bytes0 = self.backend.bytes_streamed_total();
        let mut times = StepTimes::default();
        // With the decode profile: the times and backend counters when the prefill step ended.
        let mut after_prefill: Option<(StepTimes, Vec<(&'static str, f64)>)> = None;
        let mut emit = Duration::ZERO;
        let mut pending = toks[common..].to_vec();
        let reason = loop {
            if stats.generated_tokens >= params.max_tokens {
                break FinishReason::Length;
            }
            let tok = match self.step(&pending, &mut sampler, constraint, cancel, &mut times)? {
                Step::Cancelled => break FinishReason::Cancelled,
                Step::Token(t) => t,
            };
            if self.decode_profile && after_prefill.is_none() {
                after_prefill = Some((times, self.backend.counters()));
            }
            if self.vocab.is_eog(tok) {
                break FinishReason::EndOfGeneration;
            }
            stats.generated_tokens += 1;
            let t_emit = Instant::now();
            let sent = tx.blocking_send(GenEvent::Token(tok));
            emit += t_emit.elapsed();
            if sent.is_err() {
                break FinishReason::Cancelled;
            }
            if cancel.is_cancelled() {
                break FinishReason::Cancelled;
            }
            if self.cached.len() >= n_ctx {
                break FinishReason::Length;
            }
            pending.clear();
            pending.push(tok as i32);
        };
        if self.backend.on_gpu() {
            stats.gpu_seconds = Some(times.eval.as_secs_f64());
        }
        if let Some((t0, c0)) = after_prefill {
            stats.profile = Some(self.profile_of(&times, &t0, &c0, emit));
        }
        stats.bytes_streamed = match (bytes0, self.backend.bytes_streamed_total()) {
            (Some(a), Some(b)) => Some(b.saturating_sub(a)),
            _ => None,
        };
        Ok(reason)
    }

    /// The decode profile of a request: everything after the step that ended its prefill.
    fn profile_of(
        &self,
        end: &StepTimes,
        prefill: &StepTimes,
        counters0: &[(&'static str, f64)],
        emit: Duration,
    ) -> DecodeProfile {
        let steps = end.steps - prefill.steps;
        let p = DecodeProfile {
            prefill_s: (prefill.eval + prefill.sample).as_secs_f64(),
            decode_steps: steps,
            eval_s: (end.eval - prefill.eval).as_secs_f64(),
            sample_s: (end.sample - prefill.sample).as_secs_f64(),
            emit_s: emit.as_secs_f64(),
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

    /// Copies the live sequence into the prefix cache.
    fn save_live(&mut self) {
        match self.backend.snapshot() {
            Ok(bytes) => {
                let evicted = self.cache.insert(self.cached.clone(), bytes);
                if let Some(m) = &self.metrics {
                    m.kv_prefix_evicted(evicted as u64);
                    m.kv_prefix_bytes(self.cache.bytes() as u64);
                }
            }
            Err(e) => tracing::warn!("cannot save the live sequence: {e}"),
        }
    }

    /// Evaluates `tokens` in chunks (checking for cancellation between them) and samples the
    /// next token from the last logits.
    fn step(
        &mut self,
        tokens: &[i32],
        sampler: &mut Sampler,
        constraint: &mut Option<Box<dyn TokenConstraint>>,
        cancel: &CancellationToken,
        times: &mut StepTimes,
    ) -> Result<Step, String> {
        let n_chunks = tokens.chunks(PREFILL_CHUNK).count();
        for (i, chunk) in tokens.chunks(PREFILL_CHUNK).enumerate() {
            if cancel.is_cancelled() {
                return Ok(Step::Cancelled);
            }
            let t0 = Instant::now();
            let logits = self.backend.eval(chunk)?;
            times.eval += t0.elapsed();
            self.cached.extend_from_slice(chunk);
            if let Some(m) = &self.metrics {
                m.scheduler_batch(1);
            }
            if i + 1 < n_chunks {
                continue;
            }
            let t0 = Instant::now();
            let mask = match constraint.as_mut() {
                Some(c) => c.mask()?,
                None => None,
            };
            let tok = sampler
                .sample(logits, mask.as_ref())
                .ok_or("the constraint allows no token")?;
            if let Some(c) = constraint.as_mut() {
                c.commit(tok)?;
            }
            times.sample += t0.elapsed();
            times.steps += 1;
            return Ok(Step::Token(tok));
        }
        Err("no tokens to evaluate".into())
    }
}
