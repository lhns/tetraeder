//! What `tetraeder serve --plan` needs from the planner (ADR 0055): the executable candidates of
//! a box, the pick, a drift correction of the hardware graph and the replanner's proposal.
//! Pure: no I/O, no engine types.

use crate::candidate::{
    Budget, CPU, GPU, Placement, Workload, enumerate, enumerate_cpu, footprint,
};
use crate::layout::ModelLayout;
use crate::lower::{Applies, ServeEngine, llama_args, llama_lowered, recognize};
use crate::score::{builder_features, features};
use crate::{Enumeration, enumerate_builder};
use std::collections::BTreeMap;
use tetraeder_costmodel::predict::{Prediction, predict};
use tetraeder_plan::{KvFormat, Plan};
use tetraeder_profiler::{DeviceKind, HardwareGraph, Source};

/// Physical batch of a plan that does not say (`-ub` default).
const DEFAULT_UBATCH: u32 = 512;

#[derive(Debug, Clone)]
pub struct ServeInput {
    pub hw: HardwareGraph,
    pub layout: ModelLayout,
    pub budget: Budget,
    /// Context length the server allocates; candidates are sized for it.
    pub ctx: u32,
    /// The request shape predictions are made for.
    pub workload: Workload,
    pub engine: ServeEngine,
    pub applies: Applies,
}

#[derive(Debug, Clone)]
pub struct Picked {
    pub plan: Plan,
    pub placement: Placement,
    /// `None` when the graph cannot predict (then the first candidate runs).
    pub prediction: Option<Prediction>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Proposed {
    pub plan: Plan,
    pub prediction: Prediction,
    /// `1 - t_new / t_cur` in request time; 1 when the current plan no longer fits.
    pub gain: f64,
}

/// Capacities of `gpu0` (0 without a GPU) and `cpu0` and the thread quota; the overrides win.
pub fn budget_of(hw: &HardwareGraph, quota: u32, vram: Option<u64>, ram: Option<u64>) -> Budget {
    let cap = |id: &str| hw.device(id).map_or(0, |d| d.capacity_bytes);
    let cpu_threads = hw.device(CPU).and_then(|d| d.threads).unwrap_or(quota);
    Budget {
        gpu_bytes: vram.unwrap_or_else(|| cap(GPU)),
        ram_bytes: ram.unwrap_or_else(|| cap(CPU)),
        threads: cpu_threads.min(quota).max(1),
    }
}

/// Fills the knobs a hand-written plan leaves out: the thread quota, ubatch 512, op offload on.
/// Returns a message per knob it set.
pub fn fill_knobs(plan: &mut Plan, budget: &Budget) -> Vec<String> {
    let mut notes = Vec::new();
    let k = &mut plan.knobs;
    if k.threads.is_none() {
        k.threads = Some(budget.threads);
        notes.push(format!("threads: not set, using {}", budget.threads));
    }
    if k.ubatch.is_none() {
        k.ubatch = Some(DEFAULT_UBATCH);
        notes.push(format!("ubatch: not set, using {DEFAULT_UBATCH}"));
    }
    if k.op_offload.is_none() {
        k.op_offload = Some(true);
        notes.push("op_offload: not set, using true".into());
    }
    notes
}

fn has_gpu(hw: &HardwareGraph) -> bool {
    hw.device(GPU).is_some_and(|d| d.kind == DeviceKind::Gpu)
}

/// The workload candidates are sized with: the whole context, generation as asked.
fn sizing(inp: &ServeInput) -> Workload {
    Workload {
        prompt: inp.ctx.saturating_sub(inp.workload.r#gen).max(1),
        r#gen: inp.workload.r#gen,
    }
}

fn enumeration(inp: &ServeInput, w: &Workload) -> Enumeration {
    if has_gpu(&inp.hw) {
        let en = match inp.engine {
            ServeEngine::Llama => enumerate(&inp.budget, &inp.layout, w),
            ServeEngine::Builder => enumerate_builder(&inp.budget, &inp.layout, w),
        };
        if !en.candidates.is_empty() {
            return en;
        }
    }
    // No GPU, or nothing fits on it: the CPU is always an executable placement.
    enumerate_cpu(&inp.budget, &inp.layout, w)
}

fn fits(inp: &ServeInput, plan: &Plan, w: &Workload) -> bool {
    let Ok((placement, knobs)) = recognize(plan, &inp.layout) else {
        return false;
    };
    let f = footprint(
        &inp.layout,
        placement,
        knobs.ubatch.unwrap_or(DEFAULT_UBATCH),
        w,
        KvFormat::default(),
    );
    f.gpu <= inp.budget.gpu_bytes && f.host <= inp.budget.ram_bytes
}

/// The enumerator's candidates for the engine and the GPU presence, each projected with
/// `applies`, deduplicated in order, and dropped when the projected footprint no longer fits.
pub fn candidates(inp: &ServeInput) -> Vec<Plan> {
    let w = sizing(inp);
    let mut out: Vec<Plan> = Vec::new();
    for c in enumeration(inp, &w).candidates {
        let (plan, _) = inp.applies.project(&c.plan, inp.engine);
        if !out.contains(&plan) && fits(inp, &plan, &w) {
            out.push(plan);
        }
    }
    out
}

/// The cost model's prediction for a plan on a graph.
pub fn predict_plan(
    hw: &HardwareGraph,
    layout: &ModelLayout,
    plan: &Plan,
    w: &Workload,
) -> Result<Prediction, String> {
    let (placement, knobs) = recognize(plan, layout)?;
    let threads = knobs.threads.unwrap_or(1);
    let ubatch = knobs.ubatch.unwrap_or(DEFAULT_UBATCH);
    let op = knobs.op_offload.unwrap_or(true);
    let f = match placement {
        Placement::Builder { vram_slots, .. } => {
            // Uniform routing prior: the hit rate equals the slot share.
            let hit = f64::from(vram_slots) / f64::from(layout.n_expert.max(1));
            builder_features(layout, vram_slots, hit, (threads, ubatch, op))
        }
        p => {
            let l = llama_lowered(p, &knobs, layout).expect("not a builder placement");
            features(layout, Some(p), threads, &llama_args(&l), &BTreeMap::new())
        }
    };
    predict(hw, &f, w.prompt, w.r#gen).map_err(|e| e.to_string())
}

/// The candidate with the lowest predicted request time (ties: enumeration order). When any
/// prediction fails the first candidate runs, without a prediction. `Err` without candidates.
pub fn pick(inp: &ServeInput) -> Result<Picked, String> {
    let cands = candidates(inp);
    let Some(first) = cands.first() else {
        let why = enumeration(inp, &sizing(inp))
            .dropped
            .first()
            .map(|(d, e)| format!(": {d}: {e}"))
            .unwrap_or_default();
        return Err(format!("no executable plan fits the budget{why}"));
    };
    let placement_of = |p: &Plan| recognize(p, &inp.layout).map(|r| r.0);
    let mut best: Option<(&Plan, Prediction)> = None;
    for p in &cands {
        match predict_plan(&inp.hw, &inp.layout, p, &inp.workload) {
            Ok(pred) => {
                if best.as_ref().is_none_or(|b| pred.request_s < b.1.request_s) {
                    best = Some((p, pred));
                }
            }
            Err(e) => {
                return Ok(Picked {
                    plan: first.clone(),
                    placement: placement_of(first)?,
                    prediction: None,
                    notes: vec![format!(
                        "no prediction ({e}): running the first candidate, replanning needs a predictable graph"
                    )],
                });
            }
        }
    }
    let (plan, prediction) = best.expect("at least one candidate");
    Ok(Picked {
        plan: plan.clone(),
        placement: placement_of(plan)?,
        prediction: Some(prediction),
        notes: vec![],
    })
}

/// A graph whose rates for the device the prediction's binding term names are scaled so that
/// the plan's predicted decode step equals `measured_step_s` (one factor, the other terms kept).
/// `None` when the binding term is not a device or unknown.
pub fn correct_for_drift(
    hw: &HardwareGraph,
    pred: &Prediction,
    measured_step_s: f64,
) -> Option<HardwareGraph> {
    let dev = pred.binding.split_once(':')?.1;
    let t_b = pred.tg_breakdown.iter().find(|e| e.0 == pred.binding)?.1;
    let t = 1.0 / pred.tg_tok_s;
    if !(t_b > 0.0 && measured_step_s.is_finite() && measured_step_s > 0.0) {
        return None;
    }
    let k = ((measured_step_s - (t - t_b)) / t_b).clamp(0.1, 10.0);
    let mut g = hw.clone();
    let d = g.devices.iter_mut().find(|d| d.id == dev)?;
    if let Some(c) = d.compute.as_mut() {
        c.flops_per_s /= k;
    }
    if let Some(b) = d.bandwidth_bytes_per_s.as_mut() {
        b.value /= k;
        b.source = Source::Measured;
    }
    for p in &mut d.thread_curve {
        p.flops_per_s /= k;
        p.bytes_per_s /= k;
    }
    Some(g)
}

/// Whether the running plan still fits: memory, thread quota and the devices it uses.
fn still_fits(inp: &ServeInput, current: &Plan) -> bool {
    let Ok((_, knobs)) = recognize(current, &inp.layout) else {
        return false;
    };
    let uses_gpu = current
        .replicas
        .iter()
        .flat_map(|r| &r.stages)
        .any(|s| s.devices.iter().any(|d| d == GPU));
    fits(inp, current, &sizing(inp))
        && knobs.threads.unwrap_or(1) <= inp.budget.threads
        && (!uses_gpu || has_gpu(&inp.hw))
}

/// A switch away from `current` when the pick differs and is predicted faster, or when `current`
/// no longer fits (gain 1). `None` when the pick equals `current`, has no prediction, or the
/// prediction of a still fitting `current` fails.
pub fn propose(inp: &ServeInput, current: &Plan) -> Option<Proposed> {
    let picked = pick(inp).ok()?;
    if picked.plan == *current {
        return None;
    }
    let prediction = picked.prediction?;
    let gain = if still_fits(inp, current) {
        let cur = predict_plan(&inp.hw, &inp.layout, current, &inp.workload).ok()?;
        1.0 - prediction.request_s / cur.request_s
    } else {
        1.0
    };
    (gain > 0.0).then_some(Proposed {
        plan: picked.plan,
        prediction,
        gain,
    })
}

/// The running plan's predicted request time over the best candidate's, minus 1.
pub fn regret(inp: &ServeInput, current: &Plan) -> Option<f64> {
    let best = pick(inp).ok()?.prediction?;
    let cur = predict_plan(&inp.hw, &inp.layout, current, &inp.workload).ok()?;
    Some(cur.request_s / best.request_s - 1.0)
}
