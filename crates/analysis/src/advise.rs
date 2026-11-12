//! The improvement advisor, report only (ADR 0041): faults and the top limit become candidate
//! actions, ranked by predicted gain over a horizon minus switch cost. Nothing is applied.

use serde::{Deserialize, Serialize};

use crate::detect::{Fault, FaultClass, Regime, Report, Verdict};
use crate::observation::Observation;

/// Planning horizon in seconds: a gain share is worth `share * HORIZON_S` seconds of step time.
pub const HORIZON_S: f64 = 600.0;
/// Cost of an environment fix (someone has to act): 30 s of horizon.
pub const ENV_FIX_COST_S: f64 = 30.0;
/// Cost of reloading hot-set slots or experts.
pub const RELOAD_COST_S: f64 = 5.0;
/// Cost of a replan with migration.
pub const REPLAN_COST_S: f64 = 20.0;
/// Cost of a recalibration run.
pub const RECALIBRATE_COST_S: f64 = 60.0;
/// Cost of a setting that applies at the next step.
pub const SETTING_COST_S: f64 = 0.5;
/// Share of a fault's cost that a mitigation inside the engine can win back.
pub const MITIGATION_FRAC: f64 = 0.4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionKind {
    /// Bigger hot set or more VRAM expert slots.
    GrowHotSet,
    PrefetchDepth,
    Threads,
    Ubatch,
    OpOffload,
    SpeculativeDecoding,
    /// Move layers, experts or stages.
    Replan,
    Recalibrate,
    /// Something outside the engine: stop a co-tenant, fix link width, cooling, power limit.
    FixEnvironment,
    /// Nothing stands out: add capacity instead of tuning.
    ScaleOut,
}

impl ActionKind {
    /// Name used in `hints.auto_apply`.
    pub fn name(self) -> String {
        serde_json::to_value(self)
            .ok()
            .and_then(|v| v.as_str().map(str::to_owned))
            .unwrap_or_default()
    }

    /// Class from docs/plan.md: batch limits, hot set and prefetch may be auto-applied if the
    /// hints allow it; replanning and moving experts ask first; the environment is report only.
    fn base_class(self) -> PolicyClass {
        match self {
            ActionKind::GrowHotSet | ActionKind::PrefetchDepth | ActionKind::Ubatch => {
                PolicyClass::Auto
            }
            ActionKind::SpeculativeDecoding => PolicyClass::Auto,
            ActionKind::Threads
            | ActionKind::OpOffload
            | ActionKind::Replan
            | ActionKind::Recalibrate => PolicyClass::Ask,
            ActionKind::FixEnvironment | ActionKind::ScaleOut => PolicyClass::Report,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyClass {
    Auto,
    Ask,
    Report,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Action {
    pub kind: ActionKind,
    /// Resource or device the action is about.
    pub target: String,
    /// Predicted share of step time won.
    pub predicted_gain_share: f64,
    pub switch_cost_s: f64,
    pub policy: PolicyClass,
    pub why: String,
}

impl Action {
    /// Gain over the horizon minus switch cost, in seconds.
    pub fn score(&self) -> f64 {
        self.predicted_gain_share * HORIZON_S - self.switch_cost_s
    }
}

/// Candidate actions, best first. `auto_apply` is `hints.auto_apply`: kind names that may be
/// applied without asking.
pub fn advise(report: &Report, obs: &Observation, auto_apply: &[String]) -> Vec<Action> {
    let mut out: Vec<Action> = Vec::new();
    for f in &report.faults {
        out.extend(fault_actions(f));
    }
    out.extend(limit_actions(report, obs));
    out.retain(|a| a.predicted_gain_share > 0.0 || a.kind == ActionKind::ScaleOut);
    for a in &mut out {
        a.policy = match a.kind.base_class() {
            PolicyClass::Auto if !auto_apply.contains(&a.kind.name()) => PolicyClass::Ask,
            c => c,
        };
    }
    // One entry per (kind, target): keep the best.
    out.sort_by(|a, b| b.score().total_cmp(&a.score()));
    let mut seen = Vec::new();
    out.retain(|a| {
        let key = (a.kind, a.target.clone());
        let fresh = !seen.contains(&key);
        seen.push(key);
        fresh
    });
    out
}

fn act(kind: ActionKind, target: &str, gain: f64, cost: f64, why: String) -> Action {
    Action {
        kind,
        target: target.into(),
        predicted_gain_share: gain,
        switch_cost_s: cost,
        policy: PolicyClass::Report,
        why,
    }
}

fn fault_actions(f: &Fault) -> Vec<Action> {
    let g = f.cost_share;
    let r = f.resource.as_str();
    let mut v = Vec::new();
    let env = |fix: &str| {
        act(
            ActionKind::FixEnvironment,
            r,
            g,
            ENV_FIX_COST_S,
            format!("{fix}; costs {:.0}% of the step", g * 100.0),
        )
    };
    match f.class {
        FaultClass::GpuThrottle => v.push(env("fix cooling or the power limit")),
        FaultClass::CpuCotenant => {
            v.push(env("stop or move the co-tenant process"));
            v.push(act(
                ActionKind::Threads,
                r,
                g * MITIGATION_FRAC,
                SETTING_COST_S,
                "fewer CPU threads avoid contention with the co-tenant".into(),
            ));
        }
        FaultClass::CpuQuotaThrottle => {
            v.push(env("raise the CPU quota or move to a box with more cores"));
            v.push(act(
                ActionKind::Threads,
                r,
                g * MITIGATION_FRAC,
                SETTING_COST_S,
                "lower ggml polling (cpu_poll 0) and keep threads below the quota".into(),
            ));
        }
        FaultClass::DiskSlow => {
            v.push(env("remove the io.max limit or the competing reader"));
            v.push(act(
                ActionKind::GrowHotSet,
                r,
                g * MITIGATION_FRAC,
                RELOAD_COST_S,
                "a bigger hot set reads less from the slow disk".into(),
            ));
        }
        FaultClass::VramPressure => {
            v.push(env("free the VRAM held by other processes"));
            v.push(act(
                ActionKind::Replan,
                r,
                g * MITIGATION_FRAC,
                REPLAN_COST_S,
                "replan for the VRAM that is actually free".into(),
            ));
        }
        FaultClass::PcieDegraded => {
            v.push(env(
                "reseat or reconfigure the slot to full width and generation",
            ));
            v.push(act(
                ActionKind::Replan,
                r,
                g * MITIGATION_FRAC,
                REPLAN_COST_S,
                "move traffic off the degraded link".into(),
            ));
        }
        FaultClass::LinkSlow => {
            v.push(env("repair or replace the slow link"));
            v.push(act(
                ActionKind::Replan,
                r,
                g * MITIGATION_FRAC,
                REPLAN_COST_S,
                "fewer layers across the slow link".into(),
            ));
        }
        FaultClass::Swapping => {
            v.push(env("free RAM or add RAM so nothing swaps"));
            v.push(act(
                ActionKind::GrowHotSet,
                r,
                g * MITIGATION_FRAC,
                RELOAD_COST_S,
                "shrink the RAM slots to stop paging; the hot set moves to VRAM".into(),
            ));
        }
        FaultClass::ModelMismatch => v.push(act(
            ActionKind::Recalibrate,
            r,
            g,
            RECALIBRATE_COST_S,
            format!("{r} runs below its calibrated rate with no known cause; recalibrate"),
        )),
    }
    v
}

fn limit_actions(report: &Report, obs: &Observation) -> Vec<Action> {
    let Verdict::Limited(top) = &report.verdict else {
        return vec![act(
            ActionKind::ScaleOut,
            "all",
            0.0,
            0.0,
            "no resource stands out: scale instead of tuning".into(),
        )];
    };
    let top_share = report.limits.first().map_or(0.0, |l| l.share);
    let second = report.limits.get(1).map_or(0.0, |l| l.share);
    // Roofline: relieving the top resource down to the next one is the most to win. What a fault
    // on that resource already costs is counted by the fault's own actions.
    let fault_cost: f64 = report
        .faults
        .iter()
        .filter(|f| &f.resource == top)
        .map(|f| f.cost_share)
        .sum();
    let gain = (top_share - second - fault_cost).max(0.0);
    let hit = obs
        .env
        .expert_hit_rate
        .map(|h| format!(" (expert hit rate {h:.2})"))
        .unwrap_or_default();
    let pct = format!("{:.0}%", top_share * 100.0);
    let mut v = Vec::new();
    if top.starts_with("disk:") || top == "ram" {
        v.push(act(
            ActionKind::GrowHotSet,
            top,
            gain,
            RELOAD_COST_S,
            format!("{top} holds {pct} of the step{hit}: bigger hot set, more slots"),
        ));
        v.push(act(
            ActionKind::PrefetchDepth,
            top,
            gain * 0.5,
            SETTING_COST_S,
            format!("{top} holds {pct} of the step{hit}: deeper prefetch"),
        ));
    } else if top == "cpu" || top == "host" {
        v.push(act(
            ActionKind::Threads,
            top,
            gain * 0.5,
            SETTING_COST_S,
            format!("{top} holds {pct} of the step: tune and pin threads"),
        ));
        v.push(act(
            ActionKind::OpOffload,
            top,
            gain,
            REPLAN_COST_S,
            format!("{top} holds {pct} of the step: offload ops to a GPU"),
        ));
    } else if top.starts_with("gpu:") {
        let (kind, why) = match report.regime {
            Regime::Serialized => (
                ActionKind::SpeculativeDecoding,
                "bandwidth bound at batch 1: speculative decoding reuses each weight read",
            ),
            Regime::Load => (
                ActionKind::Ubatch,
                "compute bound under load: tune the ubatch",
            ),
        };
        v.push(act(
            kind,
            top,
            gain * 0.5,
            SETTING_COST_S,
            format!("{top} holds {pct} of the step: {why}"),
        ));
    } else if top.starts_with("link:") {
        v.push(act(
            ActionKind::Replan,
            top,
            gain,
            REPLAN_COST_S,
            format!("{top} holds {pct} of the step: fewer layers or ranks across it"),
        ));
    }
    v
}
