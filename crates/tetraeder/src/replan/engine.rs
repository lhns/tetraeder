//! The gateway's engine behind a gate: it can stop admitting, drain, and swap the scheduler
//! under a running server (ADR 0055).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tetraeder_control::EngineControl;
use tetraeder_gateway::engine::{
    Engine, EngineError, GenEvent, GenerateRequest, ModelInfo, SamplingParams, TokenId, Vocab,
};
use tetraeder_plan::Plan;
use tetraeder_planner::Lowered;
use tetraeder_scheduler::Scheduler;
use tetraeder_telemetry::Metrics;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// Events buffered between the forwarding task and the gateway.
const FORWARD_BUFFER: usize = 32;
/// How long a cut request waits for the gateway to take its error event.
const CUT_NOTICE: Duration = Duration::from_secs(1);
/// How long a restart waits for cut requests to unwind.
const CUT_WAIT: Duration = Duration::from_secs(10);
/// Deadline of the one-token health generation.
const HEALTH_DEADLINE: Duration = Duration::from_secs(60);

pub type Loader = Box<dyn Fn(&Lowered) -> Result<Scheduler, String> + Send + Sync>;
pub type Lowerer = Box<dyn Fn(&Plan) -> Result<Lowered, String> + Send + Sync>;

/// Requests in flight: how many, and the token that cuts each one.
#[derive(Default)]
struct Flight {
    count: AtomicUsize,
    cuts: Mutex<HashMap<u64, CancellationToken>>,
}

/// Leaves the books when a request ends, however it ends.
struct Ticket {
    flight: Arc<Flight>,
    id: u64,
}

impl Drop for Ticket {
    fn drop(&mut self) {
        self.flight
            .cuts
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&self.id);
        self.flight.count.fetch_sub(1, Ordering::SeqCst);
    }
}

pub struct SwitchableEngine {
    sched: RwLock<Option<Arc<Scheduler>>>,
    info: ModelInfo,
    vocab: Arc<Vocab>,
    admitting: AtomicBool,
    flight: Arc<Flight>,
    next_id: AtomicU64,
    handle: tokio::runtime::Handle,
    loader: Loader,
    lower: Lowerer,
    metrics: Option<Metrics>,
    /// Seconds the last successful load took (the switch cost the planner uses).
    last_load_s: Arc<Mutex<f64>>,
}

impl SwitchableEngine {
    /// Loads the first scheduler (timed, recorded as an engine load) and caches the model facts
    /// every later scheduler has to repeat.
    pub fn new(
        handle: tokio::runtime::Handle,
        loader: Loader,
        lower: Lowerer,
        first: &Lowered,
        metrics: Option<Metrics>,
    ) -> Result<Self, String> {
        let t0 = Instant::now();
        let loaded = loader(first);
        record_load(metrics.as_ref(), t0.elapsed(), loaded.is_ok());
        let s = loaded?;
        let info = s
            .models()
            .into_iter()
            .next()
            .ok_or("the engine serves no model")?;
        let vocab = s.vocab(&info.id).map_err(|e| e.to_string())?;
        if let Some(m) = &metrics {
            m.replan_admitting(true);
        }
        Ok(Self {
            sched: RwLock::new(Some(Arc::new(s))),
            info,
            vocab,
            admitting: AtomicBool::new(true),
            flight: Arc::default(),
            next_id: AtomicU64::new(0),
            handle,
            loader,
            lower,
            metrics,
            last_load_s: Arc::new(Mutex::new(t0.elapsed().as_secs_f64())),
        })
    }

    /// Shared with the planner source, which adds it to the switch cost.
    pub fn last_load_s(&self) -> Arc<Mutex<f64>> {
        Arc::clone(&self.last_load_s)
    }

    pub fn admitting(&self) -> bool {
        self.admitting.load(Ordering::SeqCst)
    }

    pub fn in_flight(&self) -> usize {
        self.flight.count.load(Ordering::SeqCst)
    }

    fn scheduler(&self) -> Option<Arc<Scheduler>> {
        self.sched.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn cut_all(&self) {
        for t in self
            .flight
            .cuts
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
        {
            t.cancel();
        }
    }
}

fn record_load(m: Option<&Metrics>, d: Duration, ok: bool) {
    if let Some(m) = m {
        m.engine_load(d, if ok { "ok" } else { "failed" });
    }
}

impl Engine for SwitchableEngine {
    fn models(&self) -> Vec<ModelInfo> {
        vec![self.info.clone()]
    }

    fn vocab(&self, model: &str) -> Result<Arc<Vocab>, EngineError> {
        if model == self.info.id {
            Ok(Arc::clone(&self.vocab))
        } else {
            Err(EngineError::ModelNotFound(model.into()))
        }
    }

    fn tokenize(&self, model: &str, text: &str) -> Result<Vec<TokenId>, EngineError> {
        if !self.admitting() {
            return Err(EngineError::Overloaded);
        }
        self.scheduler()
            .ok_or(EngineError::Overloaded)?
            .tokenize(model, text)
    }

    fn generate(&self, req: GenerateRequest) -> Result<mpsc::Receiver<GenEvent>, EngineError> {
        // Count first, then look at the gate: `stop_admitting` followed by a read of the count
        // cannot miss a request that got through.
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.flight.count.fetch_add(1, Ordering::SeqCst);
        let ticket = Ticket {
            flight: Arc::clone(&self.flight),
            id,
        };
        if !self.admitting() {
            return Err(EngineError::Overloaded);
        }
        let sched = self.scheduler().ok_or(EngineError::Overloaded)?;
        let cut = CancellationToken::new();
        self.flight
            .cuts
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id, cut.clone());
        let inner = sched.generate(req)?;
        let (tx, rx) = mpsc::channel(FORWARD_BUFFER);
        self.handle.spawn(forward(inner, tx, cut, ticket));
        Ok(rx)
    }
}

/// Passes the scheduler's events on until the request ends, the client leaves or a switch cuts it.
async fn forward(
    mut inner: mpsc::Receiver<GenEvent>,
    tx: mpsc::Sender<GenEvent>,
    cut: CancellationToken,
    _ticket: Ticket,
) {
    loop {
        let ev = tokio::select! {
            ev = inner.recv() => match ev {
                Some(ev) => ev,
                // Closed without Finished or Error: the gateway reports it as an error.
                None => return,
            },
            () = cut.cancelled() => break,
        };
        let end = matches!(ev, GenEvent::Finished { .. } | GenEvent::Error(_));
        tokio::select! {
            sent = tx.send(ev) => if sent.is_err() { return },
            () = cut.cancelled() => break,
        }
        if end {
            return;
        }
    }
    // Cut by a plan switch. Dropping the scheduler side first makes the worker's send fail, so
    // its job ends; then the client gets an error if it is listening.
    drop(inner);
    let err = GenEvent::Error("request cut by a plan switch".into());
    let _ = tokio::time::timeout(CUT_NOTICE, tx.send(err)).await;
}

/// How long a restart waits for other holders of the old scheduler to let go.
const SHUTDOWN_WAIT: Duration = Duration::from_secs(10);

/// Joins the scheduler's worker once `old` is the last `Arc`; waits up to `wait` for the others.
fn shutdown_scheduler(mut old: Arc<Scheduler>, wait: Duration) -> Result<(), String> {
    let deadline = Instant::now() + wait;
    loop {
        match Arc::try_unwrap(old) {
            Ok(s) => return s.shutdown(),
            Err(a) if Instant::now() >= deadline => {
                // Still held: the worker ends when the last holder drops it.
                drop(a);
                return Ok(());
            }
            Err(a) => old = a,
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// `EngineControl` over the shared engine; the replanner owns one.
pub struct Control(pub Arc<SwitchableEngine>);

impl EngineControl for Control {
    fn stop_admitting(&mut self) {
        self.0.admitting.store(false, Ordering::SeqCst);
        if let Some(m) = &self.0.metrics {
            m.replan_admitting(false);
        }
    }

    fn in_flight(&self) -> usize {
        self.0.in_flight()
    }

    fn wait_drained(&mut self, timeout_s: f64) -> bool {
        let deadline = Instant::now() + Duration::from_secs_f64(timeout_s.max(0.0));
        loop {
            if self.0.in_flight() == 0 {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    fn restart(&mut self, plan: &Plan) -> Result<(), String> {
        let e = &self.0;
        let lowered = (e.lower)(plan)?;
        if e.in_flight() > 0 {
            e.cut_all();
            let deadline = Instant::now() + CUT_WAIT;
            while e.in_flight() > 0 && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(10));
            }
        }
        // Stop the old scheduler before loading: two models would not fit one box. The join
        // returns once its backend is released (ADR 0055).
        let old = e.sched.write().unwrap_or_else(|x| x.into_inner()).take();
        if let Some(old) = old {
            shutdown_scheduler(old, SHUTDOWN_WAIT)?;
        }
        let t0 = Instant::now();
        let loaded = (e.loader)(&lowered);
        let took = t0.elapsed();
        record_load(e.metrics.as_ref(), took, loaded.is_ok());
        let new = loaded?;
        let same = new
            .models()
            .first()
            .is_some_and(|m| m.id == e.info.id && m.context_length == e.info.context_length);
        if !same {
            return Err("the new engine serves another model or context length".into());
        }
        *e.sched.write().unwrap_or_else(|x| x.into_inner()) = Some(Arc::new(new));
        *e.last_load_s.lock().unwrap_or_else(|x| x.into_inner()) = took.as_secs_f64();
        Ok(())
    }

    /// One greedy token through the scheduler itself, not the gate.
    fn healthy(&mut self) -> bool {
        let Some(s) = self.0.scheduler() else {
            return false;
        };
        let id = self.0.info.id.clone();
        let text = self
            .0
            .info
            .bos_token
            .clone()
            .filter(|t| !t.is_empty())
            .unwrap_or_else(|| "Hello".into());
        let Ok(prompt) = s.tokenize(&id, &text) else {
            return false;
        };
        if prompt.is_empty() {
            return false;
        }
        let cancel = CancellationToken::new();
        let req = GenerateRequest {
            model: id,
            prompt,
            params: SamplingParams {
                max_tokens: 1,
                temperature: Some(0.0),
                ..Default::default()
            },
            constraint: None,
            cancel: cancel.clone(),
        };
        let Ok(mut rx) = s.generate(req) else {
            return false;
        };
        // try_recv, not blocking_recv: this may run on a runtime thread (tests).
        let deadline = Instant::now() + HEALTH_DEADLINE;
        let ok = loop {
            match rx.try_recv() {
                Ok(GenEvent::Finished { .. }) => break true,
                Ok(GenEvent::Error(_)) | Err(mpsc::error::TryRecvError::Disconnected) => {
                    break false;
                }
                Ok(GenEvent::Token(_)) => {}
                Err(mpsc::error::TryRecvError::Empty) => {
                    if Instant::now() >= deadline {
                        break false;
                    }
                    std::thread::sleep(Duration::from_millis(2));
                }
            }
        };
        cancel.cancel();
        ok
    }

    fn resume(&mut self) {
        self.0.admitting.store(true, Ordering::SeqCst);
        if let Some(m) = &self.0.metrics {
            m.replan_admitting(true);
        }
    }

    /// Every switch drains and restarts.
    fn supports_knobs(&self) -> bool {
        false
    }

    fn apply_knobs(&mut self, _plan: &Plan) -> Result<(), String> {
        Err("knobs cannot change without a reload".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine_args::PlacementArgs;
    use crate::replan::glue::{builder_chunk, builder_options, llama_options};
    use clap::Parser;
    use std::path::PathBuf;
    use tetraeder_control::{Config, FakeClock, PlanSource, Proposal, Replanner, Trigger};
    use tetraeder_gateway::engine::FinishReason;
    use tetraeder_model::MissPolicy;
    use tetraeder_plan::{Knobs, KvFormat};
    use tetraeder_planner::{ModelLayout, Placement, ServeEngine, Workload, build_plan, lower};
    use tetraeder_scheduler::{SchedulerConfig, load_llama, load_moe_chunked};
    use tetraeder_telemetry::{Telemetry, TelemetryConfig};

    const CTX: u32 = 256;

    #[derive(Parser)]
    struct P {
        #[command(flatten)]
        place: PlacementArgs,
    }

    fn fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-qwen3moe-f16.gguf")
    }

    fn layout() -> ModelLayout {
        crate::plan::layout(&fixture()).unwrap()
    }

    /// The CPU plan with `threads`; ubatch 64 keeps the tiny model's batches small.
    fn plan(l: &ModelLayout, threads: u32) -> Plan {
        let knobs = Knobs {
            threads: Some(threads),
            ubatch: Some(64),
            op_offload: Some(false),
            ..Knobs::default()
        };
        let w = Workload {
            prompt: 128,
            r#gen: 128,
        };
        build_plan(l, Placement::Cpu, knobs, 64, &w, KvFormat::default())
    }

    /// A loader that fails for plans with `fail_threads` threads, like a plan that cannot load.
    fn loader(engine: ServeEngine, fail_threads: Option<u32>) -> Loader {
        let base = P::parse_from(["x"]).place;
        let tel = Arc::new(Telemetry::new(&TelemetryConfig::from_env()).unwrap());
        Box::new(move |l: &Lowered| {
            let threads = match l {
                Lowered::Llama { threads, .. } | Lowered::Builder { threads, .. } => *threads,
            };
            if Some(threads) == fail_threads {
                return Err("scripted load failure".into());
            }
            let path = fixture();
            let mut cfg = SchedulerConfig::new(&path);
            match engine {
                ServeEngine::Llama => {
                    let (pl, cc) = llama_options(l, &base, CTX)?;
                    Scheduler::start(cfg, move || load_llama(&path, &pl, &cc, None))
                }
                ServeEngine::Builder => {
                    let o = builder_options(l, CTX, MissPolicy::Upload, false, Arc::clone(&tel))?;
                    let chunk = builder_chunk(l)?;
                    cfg.prefill_chunk = chunk;
                    Scheduler::start(cfg, move || load_moe_chunked(&path, &o, chunk))
                }
            }
        })
    }

    fn rig(engine: ServeEngine, fail_threads: Option<u32>) -> (Arc<SwitchableEngine>, ModelLayout) {
        let l = layout();
        let lay = l.clone();
        let lower_fn: Lowerer = Box::new(move |p| lower(p, &lay, engine));
        let first = lower(&plan(&l, 2), &l, engine).unwrap();
        let e = SwitchableEngine::new(
            tokio::runtime::Handle::current(),
            loader(engine, fail_threads),
            lower_fn,
            &first,
            None,
        )
        .unwrap();
        (Arc::new(e), l)
    }

    fn request(e: &SwitchableEngine, max: u32) -> Result<mpsc::Receiver<GenEvent>, EngineError> {
        let id = e.models()[0].id.clone();
        // through the scheduler: the gate refuses tokenizing while it is closed
        let prompt = e.scheduler().unwrap().tokenize(&id, "Hello there").unwrap();
        e.generate(GenerateRequest {
            model: id,
            prompt,
            params: SamplingParams {
                max_tokens: max,
                temperature: Some(0.0),
                ..Default::default()
            },
            constraint: None,
            cancel: CancellationToken::new(),
        })
    }

    async fn collect(mut rx: mpsc::Receiver<GenEvent>) -> (Vec<u32>, Option<GenEvent>) {
        let mut tokens = Vec::new();
        while let Some(ev) = rx.recv().await {
            match ev {
                GenEvent::Token(t) => tokens.push(t),
                end => return (tokens, Some(end)),
            }
        }
        (tokens, None)
    }

    async fn greedy16(e: &SwitchableEngine) -> Vec<u32> {
        let (tokens, end) = collect(request(e, 16).unwrap()).await;
        assert!(matches!(end, Some(GenEvent::Finished { .. })), "{end:?}");
        tokens
    }

    const ENGINES: [ServeEngine; 2] = [ServeEngine::Llama, ServeEngine::Builder];

    // hardware: cpu
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn in_flight_counts_until_the_receiver_finishes() {
        let (e, _) = rig(ServeEngine::Llama, None);
        let rx = request(&e, 200).unwrap();
        assert_eq!(e.in_flight(), 1);
        let (tokens, end) = collect(rx).await;
        assert!(
            matches!(
                end,
                Some(GenEvent::Finished {
                    reason: FinishReason::Length,
                    ..
                })
            ),
            "{} tokens, {end:?}",
            tokens.len()
        );
        let mut c = Control(Arc::clone(&e));
        assert!(c.wait_drained(5.0));
        assert_eq!(c.in_flight(), 0);
    }

    // hardware: cpu
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn stop_admitting_refuses_and_resume_admits() {
        let (e, _) = rig(ServeEngine::Llama, None);
        let mut c = Control(Arc::clone(&e));
        c.stop_admitting();
        assert!(matches!(request(&e, 4), Err(EngineError::Overloaded)));
        assert!(matches!(e.tokenize("x", "y"), Err(EngineError::Overloaded)));
        assert_eq!(e.in_flight(), 0, "a refused request leaves no trace");
        c.resume();
        assert_eq!(greedy16(&e).await.len(), 16);
    }

    // hardware: cpu
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn restart_keeps_greedy_output() {
        for engine in ENGINES {
            let (e, l) = rig(engine, None);
            let before = greedy16(&e).await;
            let models = e.models();
            let mut c = Control(Arc::clone(&e));
            c.stop_admitting();
            assert!(c.wait_drained(1.0));
            c.restart(&plan(&l, 1)).unwrap();
            assert!(c.healthy());
            c.resume();
            assert_eq!(e.models(), models, "{engine:?}");
            assert_eq!(greedy16(&e).await, before, "{engine:?}");
        }
    }

    // hardware: cpu
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn drain_timeout_cuts_requests() {
        let (e, l) = rig(ServeEngine::Llama, None);
        let rx = request(&e, 200).unwrap();
        let mut c = Control(Arc::clone(&e));
        c.stop_admitting();
        assert!(!c.wait_drained(0.2));
        let t0 = Instant::now();
        c.restart(&plan(&l, 1)).unwrap();
        assert!(t0.elapsed() < Duration::from_secs(15));
        let (_, end) = collect(rx).await;
        assert!(
            matches!(end, Some(GenEvent::Error(_)) | None),
            "cut request ended with {end:?}"
        );
        assert_eq!(e.in_flight(), 0);
        assert!(c.healthy());
    }

    struct Scripted(Option<Plan>);

    impl PlanSource for Scripted {
        fn propose(&mut self, _: &Plan, _: &Trigger) -> Option<Proposal> {
            self.0.clone().map(|plan| Proposal {
                plan,
                predicted_gain_s_per_s: 1.0,
                switch_cost_s: 1.0,
                knobs_only: false,
            })
        }
    }

    // hardware: cpu
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn failed_restart_rolls_back_through_the_replanner() {
        // plans with 3 threads fail to load
        let (e, l) = rig(ServeEngine::Llama, Some(3));
        let before = greedy16(&e).await;
        let old = plan(&l, 2);
        let mut r = Replanner::new(
            Config::default(),
            old.clone(),
            Scripted(Some(plan(&l, 3))),
            Control(Arc::clone(&e)),
            FakeClock::at(12, 0),
        );
        match r.on_trigger(Trigger::DeviceChange) {
            tetraeder_control::Outcome::RolledBack { error } => {
                assert!(error.contains("scripted load failure"), "{error}");
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(r.current(), &old);
        assert!(e.admitting(), "the rollback resumes admission");
        assert_eq!(greedy16(&e).await, before);
    }
}
