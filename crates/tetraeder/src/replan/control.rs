//! The replan thread: triggers from the admin route and from the residual monitor go through one
//! `Replanner` (ADR 0042, ADR 0055).

use super::engine::{Control, SwitchableEngine};
use super::monitor::ResidualMonitor;
use super::source::{PlannerSource, Shared};
use serde::Serialize;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tetraeder_control::{Config, Outcome, Replanner, SystemClock, Trigger};
use tetraeder_costmodel::predict::Prediction;
use tetraeder_plan::{Plan, Stability};
use tetraeder_planner::{Placement, recognize, regret};
use tetraeder_telemetry::{Metrics, StepRing};
use tokio::sync::oneshot;

/// A short, stable name of a plan for metric labels: the first 8 hex digits of the FNV-1a hash
/// of its TOML.
pub fn plan_label(plan: &Plan) -> String {
    let text = plan_toml(plan);
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{h:016x}")[..8].to_string()
}

pub fn plan_toml(plan: &Plan) -> String {
    plan.to_toml().unwrap_or_else(|_| format!("{plan:?}"))
}

/// What `GET /admin/plan` reports.
#[derive(Debug, Clone, Serialize)]
pub struct Status {
    pub label: String,
    pub description: String,
    pub plan: String,
    pub threads: Option<u32>,
    pub ubatch: Option<u32>,
    pub op_offload: Option<bool>,
    pub predicted_step_s: Option<f64>,
    pub last_trigger: Option<String>,
    pub last_outcome: Option<String>,
    pub last_reason: Option<String>,
    pub switches: u32,
    pub admitting: bool,
}

impl Status {
    pub fn new(plan: &Plan, placement: Option<Placement>, predicted_step_s: Option<f64>) -> Self {
        Self {
            label: plan_label(plan),
            description: placement.map_or_else(|| "unknown".into(), |p| p.describe()),
            plan: plan_toml(plan),
            threads: plan.knobs.threads,
            ubatch: plan.knobs.ubatch,
            op_offload: plan.knobs.op_offload,
            predicted_step_s,
            last_trigger: None,
            last_outcome: None,
            last_reason: None,
            switches: 0,
            admitting: true,
        }
    }
}

pub enum Cmd {
    Trigger(Trigger, Option<oneshot::Sender<Outcome>>),
}

/// The way into the replan thread.
#[derive(Clone)]
pub struct ReplanHandle {
    tx: mpsc::Sender<Cmd>,
    pub status: Arc<Mutex<Status>>,
}

impl ReplanHandle {
    pub fn new(tx: mpsc::Sender<Cmd>, status: Arc<Mutex<Status>>) -> Self {
        Self { tx, status }
    }

    /// Queues a trigger; with `wait` the receiver gets the outcome.
    pub fn trigger(
        &self,
        t: Trigger,
        wait: bool,
    ) -> Result<Option<oneshot::Receiver<Outcome>>, String> {
        let (reply, rx) = if wait {
            let (a, b) = oneshot::channel();
            (Some(a), Some(b))
        } else {
            (None, None)
        };
        self.tx
            .send(Cmd::Trigger(t, reply))
            .map_err(|_| "the replan thread has stopped".to_string())?;
        Ok(rx)
    }
}

pub struct Params {
    pub tick: Duration,
    pub dwell_s: f64,
    pub drain_timeout_s: f64,
    pub stability: Stability,
    pub plan: Plan,
    pub placement: Option<Placement>,
    pub prediction: Option<Prediction>,
    pub band: f64,
    pub engine: Arc<SwitchableEngine>,
    pub source: PlannerSource,
    pub shared: Arc<Mutex<Shared>>,
    pub ring: Arc<StepRing>,
    pub metrics: Option<Metrics>,
}

/// Starts the thread `tetraeder-replan`.
pub fn spawn(p: Params) -> std::io::Result<ReplanHandle> {
    let (tx, rx) = mpsc::channel();
    let step = p.prediction.as_ref().map(|x| 1.0 / x.tg_tok_s);
    let status = Arc::new(Mutex::new(Status::new(&p.plan, p.placement, step)));
    let handle = ReplanHandle::new(tx, Arc::clone(&status));
    std::thread::Builder::new()
        .name("tetraeder-replan".into())
        .spawn(move || run(p, &rx, &status))?;
    Ok(handle)
}

fn set_status(status: &Mutex<Status>, f: impl FnOnce(&mut Status)) {
    f(&mut status.lock().unwrap_or_else(|e| e.into_inner()));
}

fn run(p: Params, rx: &mpsc::Receiver<Cmd>, status: &Mutex<Status>) {
    let m = p.metrics.clone();
    let cfg = Config {
        min_dwell_s: p.dwell_s,
        drain_timeout_s: p.drain_timeout_s,
        stability: p.stability,
        ..Config::default()
    };
    let engine = Arc::clone(&p.engine);
    let mut monitor = ResidualMonitor::new(p.band);
    monitor.set_prediction(p.prediction.as_ref().map(|x| 1.0 / x.tg_tok_s), m.as_ref());
    let mut r = Replanner::new(
        cfg,
        p.plan,
        p.source,
        Control(Arc::clone(&engine)),
        SystemClock::new(),
    );
    if let Some(m) = m.clone() {
        r.on_outcome(Box::new(move |t, o, took| {
            m.replan(t.label(), o.label());
            m.replan_duration(Duration::from_secs_f64(took.max(0.0)));
        }));
    }
    let decide = |r: &mut Replanner<_, _, _>, monitor: &mut ResidualMonitor, t: Trigger| {
        let outcome = r.on_trigger(t.clone());
        let switched = outcome == Outcome::Switched;
        if switched {
            let last = p
                .shared
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .last
                .take();
            // The proposal that was just served; a stale one (another plan) is not used.
            let step = last
                .filter(|l| l.plan == *r.current())
                .map(|l| 1.0 / l.prediction.tg_tok_s);
            monitor.set_prediction(step, m.as_ref());
            let placement = recognize(r.current(), &p.shared.lock().unwrap().inp.layout)
                .ok()
                .map(|x| x.0);
            let fresh = Status::new(r.current(), placement, step);
            set_status(status, |s| {
                s.label = fresh.label;
                s.description = fresh.description;
                s.plan = fresh.plan;
                s.threads = fresh.threads;
                s.ubatch = fresh.ubatch;
                s.op_offload = fresh.op_offload;
                s.predicted_step_s = fresh.predicted_step_s;
                s.switches += 1;
            });
        }
        set_status(status, |s| {
            s.last_trigger = Some(t.label().to_string());
            s.last_outcome = Some(outcome.label().to_string());
            s.last_reason = match &outcome {
                Outcome::Rejected { reason: why } => Some(why.clone()),
                Outcome::RolledBack { error } => Some(error.clone()),
                _ => None,
            };
            s.admitting = engine.admitting();
        });
        outcome
    };
    loop {
        match rx.recv_timeout(p.tick) {
            Ok(Cmd::Trigger(t, reply)) => {
                let outcome = decide(&mut r, &mut monitor, t);
                if let Some(reply) = reply {
                    let _ = reply.send(outcome);
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let trigger = monitor.tick(&p.ring, m.as_ref());
                p.shared
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .measured_step_s = monitor.last_median_s;
                if let Some(m) = &m {
                    let inp = p
                        .shared
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .inp
                        .clone();
                    if let Some(g) = regret(&inp, r.current()) {
                        m.plan_regret(&plan_label(r.current()), g);
                    }
                }
                if let Some(t) = trigger {
                    decide(&mut r, &mut monitor, t);
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => return,
        }
    }
}
