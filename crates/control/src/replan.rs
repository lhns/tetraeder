//! Replanning by drain and restart (ADR 0042, ADR 0009). Synchronous; the clock, the planner and the
//! engine are traits, so the decision logic is tested with fakes.

use std::collections::VecDeque;
use tetraeder_plan::{Plan, Stability};

/// What asked for a new plan.
#[derive(Debug, Clone, PartialEq)]
pub enum Trigger {
    /// Measured step time drifted from the prediction (relative residual).
    Drift {
        residual: f64,
    },
    /// An SLO burns its error budget (`ttft`, `tpot`, `availability`).
    SloBurn {
        sli: String,
    },
    LoadShift,
    DeviceChange,
}

impl Trigger {
    /// Metric label (`tetraeder.replan.trigger`).
    pub fn label(&self) -> &'static str {
        match self {
            Trigger::Drift { .. } => "drift",
            Trigger::SloBurn { .. } => "slo_burn",
            Trigger::LoadShift => "load_shift",
            Trigger::DeviceChange => "device_change",
        }
    }
}

/// A candidate plan with the planner's estimate of what the switch buys and costs.
#[derive(Debug, Clone, PartialEq)]
pub struct Proposal {
    pub plan: Plan,
    /// Seconds saved per second of operation (fraction of time saved).
    pub predicted_gain_s_per_s: f64,
    /// Seconds of lost service for the switch.
    pub switch_cost_s: f64,
    /// Only knobs differ: no reload needed.
    pub knobs_only: bool,
}

pub trait PlanSource {
    fn propose(&mut self, current: &Plan, trigger: &Trigger) -> Option<Proposal>;
}

pub trait EngineControl {
    fn stop_admitting(&mut self);
    fn in_flight(&self) -> usize;
    /// True when no request is left in flight within `timeout_s`.
    fn wait_drained(&mut self, timeout_s: f64) -> bool;
    fn restart(&mut self, plan: &Plan) -> Result<(), String>;
    fn healthy(&mut self) -> bool;
    fn resume(&mut self);
    /// Whether knobs can change without a reload; if not, knob changes take the drain path.
    fn supports_knobs(&self) -> bool {
        true
    }
    fn apply_knobs(&mut self, plan: &Plan) -> Result<(), String>;
}

pub trait Clock {
    /// Seconds since an arbitrary fixed origin; only differences matter.
    fn now_s(&self) -> f64;
    /// Local wall clock for the freeze windows.
    fn local_hhmm(&self) -> (u8, u8);
    /// Waits `s` seconds of this clock's time. The default blocks the thread; a test clock
    /// advances its time instead, so a polling loop that sleeps runs in microseconds
    /// (ADR 0056).
    fn sleep(&self, s: f64) {
        std::thread::sleep(std::time::Duration::from_secs_f64(s.max(0.0)));
    }
}

/// A shared clock is a clock: the engine wrapper and the replanner hold the same one.
impl<C: Clock + ?Sized> Clock for std::sync::Arc<C> {
    fn now_s(&self) -> f64 {
        (**self).now_s()
    }

    fn local_hhmm(&self) -> (u8, u8) {
        (**self).local_hhmm()
    }

    fn sleep(&self, s: f64) {
        (**self).sleep(s);
    }
}

/// Test clock: `now_s` counts from midnight of day 0 and local time equals it. Clones share the
/// time, also across threads; `sleep` advances it.
#[derive(Debug, Clone, Default)]
pub struct FakeClock(std::sync::Arc<std::sync::Mutex<f64>>);

impl FakeClock {
    pub fn at(hh: u8, mm: u8) -> Self {
        Self(std::sync::Arc::new(std::sync::Mutex::new(
            f64::from(hh) * 3600.0 + f64::from(mm) * 60.0,
        )))
    }

    pub fn advance(&self, s: f64) {
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) += s;
    }

    fn get(&self) -> f64 {
        *self.0.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Clock for FakeClock {
    fn now_s(&self) -> f64 {
        self.get()
    }

    fn local_hhmm(&self) -> (u8, u8) {
        let m = (self.get() / 60.0) as u64 % 1440;
        ((m / 60) as u8, (m % 60) as u8)
    }

    fn sleep(&self, s: f64) {
        self.advance(s.max(0.0));
    }
}

/// The process clock: monotonic seconds since creation, and the wall clock for freeze windows.
/// Freeze windows are read in UTC (ADR 0055): the process knows no time zone.
#[derive(Debug, Clone, Copy)]
pub struct SystemClock {
    origin: std::time::Instant,
}

impl SystemClock {
    pub fn new() -> Self {
        Self {
            origin: std::time::Instant::now(),
        }
    }
}

impl Default for SystemClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for SystemClock {
    fn now_s(&self) -> f64 {
        self.origin.elapsed().as_secs_f64()
    }

    fn local_hhmm(&self) -> (u8, u8) {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let minute = secs / 60 % 1440;
        ((minute / 60) as u8, (minute % 60) as u8)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    Switched,
    KnobsApplied,
    /// `reason`: `no proposal`, `frozen`, `dwell`, `rate limited`, `gain too small`.
    Rejected {
        reason: String,
    },
    /// The new plan failed to start or to become healthy; the old plan runs again.
    RolledBack {
        error: String,
    },
}

impl Outcome {
    /// Metric label (`tetraeder.replan.outcome`).
    pub fn label(&self) -> &'static str {
        match self {
            Outcome::Switched => "switched",
            Outcome::KnobsApplied => "knobs_applied",
            Outcome::Rejected { .. } => "rejected",
            Outcome::RolledBack { .. } => "rolled_back",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    /// Switch only if gain * horizon > cost * safety.
    pub horizon_s: f64,
    pub safety: f64,
    pub min_dwell_s: f64,
    pub drain_timeout_s: f64,
    pub stability: Stability,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            horizon_s: 3600.0,
            safety: 2.0,
            min_dwell_s: 600.0,
            drain_timeout_s: 120.0,
            stability: Stability::default(),
        }
    }
}

/// Called after every decision with the trigger, the outcome and the seconds it took.
pub type OutcomeHook = Box<dyn FnMut(&Trigger, &Outcome, f64)>;

pub struct Replanner<P, E, C> {
    cfg: Config,
    current: Plan,
    planner: P,
    engine: E,
    clock: C,
    switches: VecDeque<f64>,
    last_switch: Option<f64>,
    hook: Option<OutcomeHook>,
}

impl<P: PlanSource, E: EngineControl, C: Clock> Replanner<P, E, C> {
    pub fn new(cfg: Config, current: Plan, planner: P, engine: E, clock: C) -> Self {
        Self {
            cfg,
            current,
            planner,
            engine,
            clock,
            switches: VecDeque::new(),
            last_switch: None,
            hook: None,
        }
    }

    /// Metrics hook, e.g. `|t, o, s| { m.replan(t.label(), o.label()); m.replan_duration(..) }`.
    pub fn on_outcome(&mut self, hook: OutcomeHook) {
        self.hook = Some(hook);
    }

    pub fn current(&self) -> &Plan {
        &self.current
    }

    pub fn engine(&self) -> &E {
        &self.engine
    }

    pub fn on_trigger(&mut self, t: Trigger) -> Outcome {
        let start = self.clock.now_s();
        let outcome = self.decide(&t);
        let took = self.clock.now_s() - start;
        if let Some(h) = self.hook.as_mut() {
            h(&t, &outcome, took);
        }
        outcome
    }

    fn reject(reason: &str) -> Outcome {
        Outcome::Rejected {
            reason: reason.to_owned(),
        }
    }

    fn decide(&mut self, t: &Trigger) -> Outcome {
        let now = self.clock.now_s();
        if self.frozen() {
            return Self::reject("frozen");
        }
        if self
            .last_switch
            .is_some_and(|l| now - l < self.cfg.min_dwell_s)
        {
            return Self::reject("dwell");
        }
        while self.switches.front().is_some_and(|&s| now - s >= 3600.0) {
            self.switches.pop_front();
        }
        if let Some(max) = self.cfg.stability.max_switches_per_hour
            && self.switches.len() >= max as usize
        {
            return Self::reject("rate limited");
        }
        let Some(p) = self.planner.propose(&self.current, t) else {
            return Self::reject("no proposal");
        };
        if p.predicted_gain_s_per_s * self.cfg.horizon_s <= p.switch_cost_s * self.cfg.safety {
            return Self::reject("gain too small");
        }
        // From here on the engine is touched, so the attempt counts against dwell and rate.
        self.last_switch = Some(now);
        self.switches.push_back(now);
        if p.knobs_only && self.engine.supports_knobs() {
            self.apply_knobs(p.plan)
        } else {
            self.drain_and_restart(p.plan)
        }
    }

    fn apply_knobs(&mut self, plan: Plan) -> Outcome {
        match self.engine.apply_knobs(&plan) {
            Ok(()) => {
                self.current = plan;
                Outcome::KnobsApplied
            }
            Err(error) => {
                // Best effort: put the old knobs back.
                let _ = self.engine.apply_knobs(&self.current);
                Outcome::RolledBack { error }
            }
        }
    }

    fn drain_and_restart(&mut self, plan: Plan) -> Outcome {
        self.engine.stop_admitting();
        // On timeout the remaining requests are cut by the restart.
        self.engine.wait_drained(self.cfg.drain_timeout_s);
        let error = match self.engine.restart(&plan) {
            Ok(()) if self.engine.healthy() => {
                self.current = plan;
                self.engine.resume();
                return Outcome::Switched;
            }
            Ok(()) => "new plan not healthy".to_owned(),
            Err(e) => e,
        };
        let back = self
            .engine
            .restart(&self.current)
            .map(|()| self.engine.healthy())
            .map_err(|e| format!("rollback failed: {e}"));
        let error = match back {
            Ok(true) => error,
            Ok(false) => format!("{error}; old plan not healthy after rollback"),
            Err(e) => format!("{error}; {e}"),
        };
        self.engine.resume();
        Outcome::RolledBack { error }
    }

    fn frozen(&self) -> bool {
        let (h, m) = self.clock.local_hhmm();
        let now = u32::from(h) * 60 + u32::from(m);
        self.cfg.stability.freeze.iter().any(|f| {
            let (Some(from), Some(to)) = (parse_hhmm(&f.from), parse_hhmm(&f.to)) else {
                return false;
            };
            if from <= to {
                (from..to).contains(&now)
            } else {
                now >= from || now < to
            }
        })
    }
}

fn parse_hhmm(s: &str) -> Option<u32> {
    let (h, m) = s.split_once(':')?;
    let (h, m): (u32, u32) = (h.parse().ok()?, m.parse().ok()?);
    (h < 24 && m < 60).then_some(h * 60 + m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;
    use tetraeder_plan::{Freeze, Knobs};

    fn plan(name: &str) -> Plan {
        Plan {
            model: name.to_owned(),
            knobs: Knobs::default(),
            replicas: vec![],
        }
    }

    type Log = Rc<RefCell<Vec<String>>>;

    struct Engine {
        log: Log,
        in_flight: usize,
        drained: bool,
        /// One entry per restart call; empty string is success.
        restart_results: Vec<String>,
        /// One entry per health call; empty means healthy.
        healthy: Vec<bool>,
        knobs_ok: bool,
        supports_knobs: bool,
    }

    impl Engine {
        fn new(log: &Log) -> Self {
            Self {
                log: log.clone(),
                in_flight: 3,
                drained: true,
                restart_results: vec![],
                healthy: vec![],
                knobs_ok: true,
                supports_knobs: true,
            }
        }
    }

    impl EngineControl for Engine {
        fn stop_admitting(&mut self) {
            self.log.borrow_mut().push("stop".into());
        }
        fn in_flight(&self) -> usize {
            self.in_flight
        }
        fn wait_drained(&mut self, timeout_s: f64) -> bool {
            let n = self.in_flight;
            self.log.borrow_mut().push(format!("drain {n} {timeout_s}"));
            if self.drained {
                self.in_flight = 0;
            }
            self.drained
        }
        fn restart(&mut self, plan: &Plan) -> Result<(), String> {
            self.log
                .borrow_mut()
                .push(format!("restart {}", plan.model));
            if self.restart_results.is_empty() {
                return Ok(());
            }
            match self.restart_results.remove(0) {
                e if e.is_empty() => Ok(()),
                e => Err(e),
            }
        }
        fn healthy(&mut self) -> bool {
            self.log.borrow_mut().push("health".into());
            if self.healthy.is_empty() {
                true
            } else {
                self.healthy.remove(0)
            }
        }
        fn resume(&mut self) {
            self.log.borrow_mut().push("resume".into());
        }
        fn supports_knobs(&self) -> bool {
            self.supports_knobs
        }
        fn apply_knobs(&mut self, plan: &Plan) -> Result<(), String> {
            self.log.borrow_mut().push(format!("knobs {}", plan.model));
            if self.knobs_ok {
                Ok(())
            } else {
                Err("bad knob".into())
            }
        }
    }

    struct Planner(Option<Proposal>);

    impl PlanSource for Planner {
        fn propose(&mut self, _: &Plan, _: &Trigger) -> Option<Proposal> {
            self.0.clone()
        }
    }

    fn proposal(gain: f64, cost: f64, knobs_only: bool) -> Proposal {
        Proposal {
            plan: plan("new"),
            predicted_gain_s_per_s: gain,
            switch_cost_s: cost,
            knobs_only,
        }
    }

    type R = Replanner<Planner, Engine, FakeClock>;

    fn setup(p: Option<Proposal>, cfg: Config, hh: u8, mm: u8) -> (R, Log, FakeClock) {
        let log: Log = Rc::default();
        let clock = FakeClock::at(hh, mm);
        let r = Replanner::new(
            cfg,
            plan("old"),
            Planner(p),
            Engine::new(&log),
            clock.clone(),
        );
        (r, log, clock)
    }

    fn good() -> Option<Proposal> {
        // 0.01 * 3600 = 36 s gained vs 10 * 2 = 20 s required.
        Some(proposal(0.01, 10.0, false))
    }

    fn rejected(reason: &str) -> Outcome {
        Outcome::Rejected {
            reason: reason.to_owned(),
        }
    }

    #[test]
    fn switch_drains_before_restart_and_resumes() {
        let (mut r, log, _) = setup(good(), Config::default(), 12, 0);
        assert_eq!(r.engine().in_flight(), 3);
        assert_eq!(r.on_trigger(Trigger::LoadShift), Outcome::Switched);
        assert_eq!(
            *log.borrow(),
            ["stop", "drain 3 120", "restart new", "health", "resume"]
        );
        assert_eq!(r.current().model, "new");
        assert_eq!(r.engine().in_flight(), 0);
    }

    #[test]
    fn drain_timeout_still_restarts() {
        let (mut r, log, _) = setup(good(), Config::default(), 12, 0);
        r.engine.drained = false;
        assert_eq!(r.on_trigger(Trigger::DeviceChange), Outcome::Switched);
        assert!(log.borrow().contains(&"restart new".to_owned()));
    }

    #[test]
    fn no_proposal_is_rejected() {
        let (mut r, log, _) = setup(None, Config::default(), 12, 0);
        assert_eq!(r.on_trigger(Trigger::LoadShift), rejected("no proposal"));
        assert!(log.borrow().is_empty());
    }

    #[test]
    fn gain_must_beat_cost_times_safety() {
        // 0.005 * 3600 = 18 s < 20 s.
        let (mut r, log, _) = setup(Some(proposal(0.005, 10.0, false)), Config::default(), 12, 0);
        assert_eq!(r.on_trigger(Trigger::LoadShift), rejected("gain too small"));
        assert!(log.borrow().is_empty());
        // Exactly equal does not switch either.
        let (mut r, _, _) = setup(
            Some(proposal(20.0 / 3600.0, 10.0, false)),
            Config::default(),
            12,
            0,
        );
        assert_eq!(r.on_trigger(Trigger::LoadShift), rejected("gain too small"));
    }

    #[test]
    fn dwell_blocks_until_elapsed() {
        let (mut r, _, clock) = setup(good(), Config::default(), 12, 0);
        assert_eq!(r.on_trigger(Trigger::LoadShift), Outcome::Switched);
        clock.advance(599.0);
        assert_eq!(r.on_trigger(Trigger::LoadShift), rejected("dwell"));
        clock.advance(1.0);
        assert_eq!(r.on_trigger(Trigger::LoadShift), Outcome::Switched);
    }

    fn cfg_rate(max: u32) -> Config {
        Config {
            min_dwell_s: 0.0,
            stability: Stability {
                max_switches_per_hour: Some(max),
                freeze: vec![],
            },
            ..Config::default()
        }
    }

    #[test]
    fn rate_limit_uses_a_sliding_hour() {
        let (mut r, _, clock) = setup(good(), cfg_rate(2), 12, 0);
        assert_eq!(r.on_trigger(Trigger::LoadShift), Outcome::Switched); // t=0
        clock.advance(1800.0);
        assert_eq!(r.on_trigger(Trigger::LoadShift), Outcome::Switched); // t=1800
        clock.advance(1700.0); // t=3500: both in the window
        assert_eq!(r.on_trigger(Trigger::LoadShift), rejected("rate limited"));
        clock.advance(100.0); // t=3600: the first one slid out
        assert_eq!(r.on_trigger(Trigger::LoadShift), Outcome::Switched);
        assert_eq!(r.on_trigger(Trigger::LoadShift), rejected("rate limited"));
    }

    #[test]
    fn zero_switches_per_hour_never_switches() {
        let (mut r, _, _) = setup(good(), cfg_rate(0), 12, 0);
        assert_eq!(r.on_trigger(Trigger::LoadShift), rejected("rate limited"));
    }

    fn cfg_freeze(from: &str, to: &str) -> Config {
        Config {
            stability: Stability {
                max_switches_per_hour: None,
                freeze: vec![Freeze {
                    from: from.into(),
                    to: to.into(),
                }],
            },
            ..Config::default()
        }
    }

    fn check_freeze(from: &str, to: &str, cases: &[(u8, u8, bool)]) {
        for &(hh, mm, frozen) in cases {
            let (mut r, _, _) = setup(good(), cfg_freeze(from, to), hh, mm);
            let want = if frozen {
                rejected("frozen")
            } else {
                Outcome::Switched
            };
            assert_eq!(r.on_trigger(Trigger::LoadShift), want, "{hh}:{mm}");
        }
    }

    #[test]
    fn freeze_window_same_day() {
        check_freeze(
            "09:00",
            "18:00",
            &[(8, 59, false), (9, 0, true), (17, 59, true), (18, 0, false)],
        );
    }

    #[test]
    fn freeze_window_wraps_midnight() {
        check_freeze(
            "22:00",
            "06:00",
            &[
                (21, 59, false),
                (22, 0, true),
                (23, 59, true),
                (0, 0, true),
                (5, 59, true),
                (6, 0, false),
            ],
        );
    }

    #[test]
    fn unparsable_freeze_is_ignored() {
        check_freeze("25:00", "xx", &[(12, 0, false)]);
    }

    #[test]
    fn restart_failure_rolls_back_to_the_old_plan() {
        let (mut r, log, _) = setup(good(), Config::default(), 12, 0);
        r.engine.restart_results = vec!["oom".into(), String::new()];
        assert_eq!(
            r.on_trigger(Trigger::LoadShift),
            Outcome::RolledBack {
                error: "oom".into()
            }
        );
        assert_eq!(
            *log.borrow(),
            [
                "stop",
                "drain 3 120",
                "restart new",
                "restart old",
                "health",
                "resume"
            ]
        );
        assert_eq!(r.current().model, "old");
    }

    #[test]
    fn unhealthy_new_plan_rolls_back() {
        let (mut r, log, _) = setup(good(), Config::default(), 12, 0);
        r.engine.healthy = vec![false, true];
        assert_eq!(
            r.on_trigger(Trigger::LoadShift),
            Outcome::RolledBack {
                error: "new plan not healthy".into()
            }
        );
        assert_eq!(
            *log.borrow(),
            [
                "stop",
                "drain 3 120",
                "restart new",
                "health",
                "restart old",
                "health",
                "resume"
            ]
        );
        assert_eq!(r.current().model, "old");
    }

    #[test]
    fn failed_rollback_is_reported_and_resumes() {
        let (mut r, log, _) = setup(good(), Config::default(), 12, 0);
        r.engine.restart_results = vec!["oom".into(), "gone".into()];
        assert_eq!(
            r.on_trigger(Trigger::LoadShift),
            Outcome::RolledBack {
                error: "oom; rollback failed: gone".into()
            }
        );
        assert_eq!(log.borrow().last().unwrap(), "resume");
    }

    #[test]
    fn rolled_back_attempt_counts_against_dwell() {
        let (mut r, _, _) = setup(good(), Config::default(), 12, 0);
        r.engine.restart_results = vec!["oom".into(), String::new()];
        assert!(matches!(
            r.on_trigger(Trigger::LoadShift),
            Outcome::RolledBack { .. }
        ));
        assert_eq!(r.on_trigger(Trigger::LoadShift), rejected("dwell"));
    }

    #[test]
    fn knobs_only_skips_the_drain() {
        let (mut r, log, _) = setup(Some(proposal(0.01, 0.0, true)), Config::default(), 12, 0);
        assert_eq!(
            r.on_trigger(Trigger::Drift { residual: 0.3 }),
            Outcome::KnobsApplied
        );
        assert_eq!(*log.borrow(), ["knobs new"]);
        assert_eq!(r.current().model, "new");
    }

    #[test]
    fn knobs_without_support_take_the_reload_path() {
        let (mut r, log, _) = setup(Some(proposal(0.01, 10.0, true)), Config::default(), 12, 0);
        r.engine.supports_knobs = false;
        assert_eq!(r.on_trigger(Trigger::LoadShift), Outcome::Switched);
        assert!(log.borrow().contains(&"stop".to_owned()));
    }

    #[test]
    fn knob_failure_restores_old_knobs() {
        let (mut r, log, _) = setup(Some(proposal(0.01, 0.0, true)), Config::default(), 12, 0);
        r.engine.knobs_ok = false;
        assert_eq!(
            r.on_trigger(Trigger::LoadShift),
            Outcome::RolledBack {
                error: "bad knob".into()
            }
        );
        assert_eq!(*log.borrow(), ["knobs new", "knobs old"]);
        assert_eq!(r.current().model, "old");
    }

    #[test]
    fn hook_sees_every_outcome_with_labels() {
        let (mut r, _, _) = setup(good(), Config::default(), 12, 0);
        let seen: Rc<RefCell<Vec<(String, String)>>> = Rc::default();
        let s = seen.clone();
        r.on_outcome(Box::new(move |t, o, _| {
            s.borrow_mut().push((t.label().into(), o.label().into()));
        }));
        r.on_trigger(Trigger::SloBurn { sli: "ttft".into() });
        r.on_trigger(Trigger::DeviceChange); // dwell
        assert_eq!(
            *seen.borrow(),
            [
                ("slo_burn".to_owned(), "switched".to_owned()),
                ("device_change".to_owned(), "rejected".to_owned())
            ]
        );
    }

    #[test]
    fn fake_clock_sleep_advances_time_and_crosses_threads() {
        let c = FakeClock::at(12, 0);
        let t0 = c.now_s();
        c.sleep(2.5);
        assert!((c.now_s() - t0 - 2.5).abs() < 1e-9);
        // Negative sleeps do not run time backwards; clones share the time.
        c.sleep(-1.0);
        let shared = std::sync::Arc::new(c.clone());
        let c2 = std::sync::Arc::clone(&shared);
        std::thread::spawn(move || c2.sleep(60.0)).join().unwrap();
        assert!((c.now_s() - t0 - 62.5).abs() < 1e-9);
        assert_eq!(c.local_hhmm(), (12, 1));
        // An `Arc<dyn Clock>` is a clock too.
        let dynamic: std::sync::Arc<dyn Clock + Send + Sync> = shared;
        dynamic.sleep(1.0);
        assert!((c.now_s() - t0 - 63.5).abs() < 1e-9);
    }

    #[test]
    fn system_clock_is_monotonic_and_utc_hhmm_is_in_range() {
        let c = SystemClock::new();
        let a = c.now_s();
        std::thread::sleep(std::time::Duration::from_millis(5));
        let b = c.now_s();
        assert!(a >= 0.0 && b > a);
        let (h, m) = c.local_hhmm();
        assert!(h < 24 && m < 60);
    }
}
