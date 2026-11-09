//! The gateway's engine behind a gate: it can stop admitting, drain, and swap the scheduler
//! under a running server (ADR 0055).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tetraeder_control::{Clock, EngineControl, SystemClock};
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
/// Polling period of the drain and cut waits, and of the health wait.
const DRAIN_POLL_S: f64 = 0.01;
const HEALTH_POLL_S: f64 = 0.002;

/// Polls `done` every `step_s` of `clock` time until it holds (true) or `timeout_s` has passed
/// (false). The clock sleeps, so a fake clock makes this instant (ADR 0056).
fn wait_until(
    clock: &dyn Clock,
    timeout_s: f64,
    step_s: f64,
    mut done: impl FnMut() -> bool,
) -> bool {
    let deadline = clock.now_s() + timeout_s.max(0.0);
    loop {
        if done() {
            return true;
        }
        if clock.now_s() >= deadline {
            return false;
        }
        clock.sleep(step_s);
    }
}

/// Waits for the one-token health generation to finish: true on `Finished`, false on an error,
/// a closed channel or `deadline_s` of `clock` time.
fn await_finish(rx: &mut mpsc::Receiver<GenEvent>, clock: &dyn Clock, deadline_s: f64) -> bool {
    let deadline = clock.now_s() + deadline_s;
    loop {
        match rx.try_recv() {
            Ok(GenEvent::Finished { .. }) => return true,
            Ok(GenEvent::Error(_)) | Err(mpsc::error::TryRecvError::Disconnected) => {
                return false;
            }
            Ok(GenEvent::Token(_)) => {}
            Err(mpsc::error::TryRecvError::Empty) => {
                if clock.now_s() >= deadline {
                    return false;
                }
                clock.sleep(HEALTH_POLL_S);
            }
        }
    }
}

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
    /// Time for the drain, cut and health waits: the replanner's clock.
    clock: Arc<dyn Clock + Send + Sync>,
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
            clock: Arc::new(SystemClock::new()),
        })
    }

    /// Takes the drain, cut and health waits from `clock` (the replanner's), so a fake clock
    /// runs them in microseconds.
    #[cfg(test)]
    pub fn with_clock(mut self, clock: Arc<dyn Clock + Send + Sync>) -> Self {
        self.clock = clock;
        self
    }

    /// The clock of the waits, for the replanner to share.
    pub fn clock(&self) -> Arc<dyn Clock + Send + Sync> {
        Arc::clone(&self.clock)
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
        wait_until(self.0.clock.as_ref(), timeout_s, DRAIN_POLL_S, || {
            self.0.in_flight() == 0
        })
    }

    fn restart(&mut self, plan: &Plan) -> Result<(), String> {
        let e = &self.0;
        let lowered = (e.lower)(plan)?;
        if e.in_flight() > 0 {
            e.cut_all();
            wait_until(
                e.clock.as_ref(),
                CUT_WAIT.as_secs_f64(),
                DRAIN_POLL_S,
                || e.in_flight() == 0,
            );
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
        let ok = await_finish(
            &mut rx,
            self.0.clock.as_ref(),
            HEALTH_DEADLINE.as_secs_f64(),
        );
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
    use tetraeder_control::{Clock, Config, FakeClock, PlanSource, Proposal, Replanner, Trigger};
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

    /// A fake clock that runs `on_sleep(n)` at the n-th sleep, to script what happens meanwhile.
    struct Scripted {
        clock: FakeClock,
        sleeps: std::sync::atomic::AtomicU32,
        on_sleep: Box<dyn Fn(u32) + Send + Sync>,
    }

    impl Clock for Scripted {
        fn now_s(&self) -> f64 {
            self.clock.now_s()
        }
        fn local_hhmm(&self) -> (u8, u8) {
            self.clock.local_hhmm()
        }
        fn sleep(&self, s: f64) {
            let n = self.sleeps.fetch_add(1, Ordering::SeqCst) + 1;
            (self.on_sleep)(n);
            self.clock.sleep(s);
        }
    }

    fn scripted(on_sleep: impl Fn(u32) + Send + Sync + 'static) -> (Arc<Scripted>, FakeClock) {
        let clock = FakeClock::at(12, 0);
        let s = Scripted {
            clock: clock.clone(),
            sleeps: Default::default(),
            on_sleep: Box::new(on_sleep),
        };
        (Arc::new(s), clock)
    }

    // hardware: none
    #[test]
    fn wait_until_runs_on_the_clock_without_sleeping() {
        let t0 = Instant::now();
        let clock = FakeClock::at(0, 0);
        let start = clock.now_s();
        let mut polls = 0;
        // never holds: gives up after the timeout in fake time
        assert!(!wait_until(&clock, 30.0, 0.01, || {
            polls += 1;
            false
        }));
        assert!(clock.now_s() - start >= 30.0 && clock.now_s() - start < 30.1);
        assert!(polls > 2900 && polls < 3100, "{polls}");
        // holds at the fifth poll
        let clock = FakeClock::at(0, 0);
        let mut n = 0;
        assert!(wait_until(&clock, 30.0, 0.01, || {
            n += 1;
            n == 5
        }));
        assert!((clock.now_s() - 0.04).abs() < 1e-9);
        // a zero timeout still looks once
        assert!(wait_until(&clock, 0.0, 0.01, || true));
        assert!(t0.elapsed() < Duration::from_secs(2));
    }

    // hardware: none
    #[test]
    fn the_health_wait_ends_on_finish_error_or_the_deadline() {
        let clock = FakeClock::at(0, 0);
        let (tx, mut rx) = mpsc::channel(4);
        // nothing arrives: false after the deadline, in fake time
        assert!(!await_finish(&mut rx, &clock, 60.0));
        assert!(clock.now_s() >= 60.0);
        tx.try_send(GenEvent::Token(1)).unwrap();
        tx.try_send(GenEvent::Finished {
            reason: FinishReason::Length,
            stats: Default::default(),
        })
        .unwrap();
        assert!(await_finish(&mut rx, &clock, 60.0));
        tx.try_send(GenEvent::Error("scripted".into())).unwrap();
        assert!(!await_finish(&mut rx, &clock, 60.0));
        drop(tx);
        assert!(!await_finish(&mut rx, &clock, 60.0));
    }

    // hardware: cpu
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn drain_waits_follow_the_engine_clock() {
        let (e, _) = rig(ServeEngine::Llama, None);
        let flight = Arc::clone(&e.flight);
        // one request that stays in flight until the third sleep
        flight.count.fetch_add(1, Ordering::SeqCst);
        let release = Arc::clone(&flight);
        let (clock, fake) = scripted(move |n| {
            if n == 3 {
                release.count.fetch_sub(1, Ordering::SeqCst);
            }
        });
        let Ok(owned) = Arc::try_unwrap(e) else {
            panic!("the rig has one owner");
        };
        let e = Arc::new(owned.with_clock(clock));
        let mut c = Control(Arc::clone(&e));
        let t0 = Instant::now();
        let before = fake.now_s();
        assert!(c.wait_drained(30.0));
        assert!((fake.now_s() - before - 0.03).abs() < 1e-9);
        // never released: gives up at the timeout, in fake time
        flight.count.fetch_add(1, Ordering::SeqCst);
        assert!(!c.wait_drained(5.0));
        assert!(fake.now_s() - before >= 5.0);
        flight.count.fetch_sub(1, Ordering::SeqCst);
        assert!(t0.elapsed() < Duration::from_secs(5));
    }

    type Events = Arc<Mutex<Vec<String>>>;

    fn note(events: &Events, what: impl Into<String>) {
        events.lock().unwrap().push(what.into());
    }

    /// A backend that answers every step at once with the same token, so a request with a large
    /// budget fills its buffers and then waits for its reader. Notes when it is dropped.
    struct Endless {
        logits: Vec<f32>,
        n_past: usize,
        events: Events,
        name: &'static str,
    }

    impl Drop for Endless {
        fn drop(&mut self) {
            note(&self.events, format!("{} backend dropped", self.name));
        }
    }

    impl tetraeder_scheduler::Backend for Endless {
        fn n_ctx(&self) -> usize {
            CTX as usize
        }
        fn n_past(&self) -> usize {
            self.n_past
        }
        fn truncate(&mut self, keep: usize) -> Result<(), String> {
            self.n_past = keep;
            Ok(())
        }
        fn eval(&mut self, tokens: &[i32]) -> Result<&[f32], String> {
            self.n_past += tokens.len();
            Ok(&self.logits)
        }
        fn snapshot(&mut self) -> Result<Vec<u8>, String> {
            Ok(Vec::new())
        }
        fn restore(&mut self, _: &[u8], n: usize) -> Result<(), String> {
            self.n_past = n;
            Ok(())
        }
        fn on_gpu(&self) -> bool {
            false
        }
        fn bytes_streamed_total(&self) -> Option<u64> {
            None
        }
    }

    /// The tiny model's tokenizer over an `Endless` backend: no weights, no compute.
    fn endless(events: &Events, name: &'static str) -> Result<Scheduler, String> {
        let path = fixture();
        let events = Arc::clone(events);
        Scheduler::start(SchedulerConfig::new(&path), move || {
            let tokenizer = Arc::new(
                tetraeder_ggml_sys::llama::Model::load(
                    &path,
                    &tetraeder_ggml_sys::llama::ModelParams {
                        vocab_only: true,
                        ..Default::default()
                    },
                )
                .map_err(|e| e.to_string())?,
            );
            let mut logits = vec![0.0; tokenizer.n_vocab() as usize];
            logits[65] = 1.0;
            Ok(tetraeder_scheduler::Loaded {
                backend: Box::new(Endless {
                    logits,
                    n_past: 0,
                    events,
                    name,
                }),
                tokenizer,
            })
        })
    }

    // hardware: none (the tokenizer comes from the tiny fixture; no weights load)
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn restart_cuts_in_flight_requests_on_the_fake_clock() {
        let events: Events = Arc::default();
        let l = layout();
        let first = Lowered::Llama {
            n_gpu_layers: 0,
            cpu_patterns: Vec::new(),
            threads: 2,
            batch: 64,
            ubatch: 64,
            op_offload: false,
        };
        let loads = Arc::new(AtomicUsize::new(0));
        let loader: Loader = {
            let (events, loads) = (Arc::clone(&events), Arc::clone(&loads));
            Box::new(move |_| {
                let n = loads.fetch_add(1, Ordering::SeqCst);
                if n > 0 {
                    note(&events, "load");
                }
                endless(&events, if n == 0 { "old" } else { "new" })
            })
        };
        let lower_fn: Lowerer = {
            let first = first.clone();
            Box::new(move |_| Ok(first.clone()))
        };
        let e = SwitchableEngine::new(
            tokio::runtime::Handle::current(),
            loader,
            lower_fn,
            &first,
            None,
        )
        .unwrap();
        // The hook sees the first cut-wait sleep: it counts the cut tokens, then lets the
        // runtime unwind the requests (a yield loop, no sleep), so the cut wait ends at its
        // next poll of fake time.
        let flight = Arc::clone(&e.flight);
        let hook_events = Arc::clone(&events);
        let (clock, fake) = scripted(move |_| {
            let cut = flight
                .cuts
                .lock()
                .unwrap()
                .values()
                .filter(|t| t.is_cancelled())
                .count();
            let seen = hook_events
                .lock()
                .unwrap()
                .iter()
                .any(|x| x.starts_with("cut"));
            if cut > 0 && !seen {
                note(&hook_events, format!("cut {cut}"));
                let giveup = Instant::now() + Duration::from_secs(10);
                while flight.count.load(Ordering::SeqCst) > 0 && Instant::now() < giveup {
                    std::thread::yield_now();
                }
            }
        });
        let e = Arc::new(e.with_clock(clock));
        let rx1 = request(&e, 100_000).unwrap();
        let rx2 = request(&e, 100_000).unwrap();
        assert_eq!(e.in_flight(), 2);
        let mut c = Control(Arc::clone(&e));
        c.stop_admitting();
        let t0 = Instant::now();
        let start = fake.now_s();
        assert!(
            !c.wait_drained(0.2),
            "the requests never finish by themselves"
        );
        note(&events, "drain timed out");
        let drained_at = fake.now_s() - start;
        assert!((0.2..0.25).contains(&drained_at), "{drained_at}");
        c.restart(&plan(&l, 1)).unwrap();
        // Unwound within one poll of the cut wait, not after CUT_WAIT.
        let cut_took = fake.now_s() - start - drained_at;
        assert!(cut_took < 1.0, "{cut_took} fake seconds");
        assert_eq!(e.in_flight(), 0);
        assert_eq!(
            *events.lock().unwrap(),
            ["drain timed out", "cut 2", "old backend dropped", "load"]
        );
        for rx in [rx1, rx2] {
            let (_, end) = collect(rx).await;
            assert!(
                matches!(end, Some(GenEvent::Error(_)) | None),
                "cut request ended with {end:?}"
            );
        }
        // The new scheduler serves, and no wall-clock wait happened.
        c.resume();
        let (tokens, end) = collect(request(&e, 4).unwrap()).await;
        assert_eq!(tokens.len(), 4);
        assert!(matches!(end, Some(GenEvent::Finished { .. })), "{end:?}");
        assert!(t0.elapsed() < Duration::from_secs(2), "{:?}", t0.elapsed());
    }

    struct ScriptedPlan(Option<Plan>);

    impl PlanSource for ScriptedPlan {
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
            ScriptedPlan(Some(plan(&l, 3))),
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
