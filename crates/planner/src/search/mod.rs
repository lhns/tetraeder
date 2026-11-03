//! The planner's search (docs/plan.md "Planner and optimizer / Search", ADR 0025, 0040).
//!
//! 1. **Skeletons:** ordered pipeline stages over the usable GPUs and CPUs (at most
//!    [`MAX_STAGES`]); where the phase allows tensor and expert parallel, groups of GPUs of one
//!    vendor joined by direct peer links, with degrees that divide heads, KV heads and experts
//!    (Linux only, [`SearchInput::linux`]). For MoE models, cold experts either compute on the
//!    CPU or stream to the GPU.
//! 2. **Prune** by memory lower bounds (smallest allowed formats against the capacity left after
//!    the reserve), hard hints (`pin`, `forbid`), missing links and what the phase cannot run.
//! 3. **Solve** each skeleton: layer split over uneven stages by dynamic programming
//!    ([`split_layers`]), experts of a stage by routing frequency into the compute tier, then
//!    CPU RAM, then NVMe, formats by a greedy knapsack over the [`QualityModel`] (least loss per
//!    byte saved first) until the memory fits, and further compression levels
//!    ([`LEVELS`]) that trade quality for speed so the front has more than one point.
//! 4. **Score** all survivors analytically (decode speed from the costmodel roofline and
//!    [`pipeline_token_time`](tetraeder_costmodel::pipeline_token_time)), run the best
//!    [`SIMULATED`] through the simulator for TTFT and TPOT, then [`choose`].
//!
//! Every returned plan passes `tetraeder_plan::validate` for the phase and hints. The search is
//! deterministic.

mod cost;
mod dp;
mod prep;
mod solve;

pub use dp::{Split, split_layers};

use crate::objective::{Choice, Scored, StageTime, choose, resource_cost};
use crate::quality::QualityModel;
use cost::{hop, rates, stage_time};
use prep::Prep;
use serde::{Deserialize, Serialize};
use solve::{Group, Mode, Skel, Solved, StageSol, StageTiers};
use std::collections::{BTreeMap, HashSet};
use tetraeder_costmodel::sim::{
    Arrival, LengthDist, ModelSpec, PlanStub, StageSpec, WorkloadProfile, simulate,
};
use tetraeder_costmodel::{GraphMode, Stage as CmStage, pipeline_token_time};
use tetraeder_plan::{Context, Hints, ModelDesc, Phase, Span, UnitId, validate};
use tetraeder_profiler::{DeviceKind, HardwareGraph, LinkKind, Measurement};

/// Most pipeline stages of a skeleton.
pub const MAX_STAGES: usize = 3;
/// Plans run through the simulator.
pub const SIMULATED: usize = 20;
/// Fractions of the way from the memory-fit formats to the smallest allowed ones.
pub const LEVELS: [f64; 4] = [0.25, 0.5, 0.75, 1.0];

pub struct SearchInput<'a> {
    pub hw: &'a HardwareGraph,
    pub model: &'a ModelDesc,
    pub hints: &'a Hints,
    pub phase: Phase,
    /// `(prompt tokens, generated tokens, concurrency)`. `hints.workload` overrides prompt and
    /// concurrency.
    pub workload: (u32, u32, u32),
    pub quality: &'a dyn QualityModel,
    /// Measured reserve per device id.
    pub reserve: BTreeMap<String, u64>,
    /// Relative selection frequency per expert; uniform when `None` or of the wrong length.
    pub routing: Option<&'a [f64]>,
    /// The machines run Linux: tensor and expert parallel need NCCL (docs/plan.md "Platform
    /// constraints").
    pub linux: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchResult {
    /// Every plan found, in search order.
    pub scored: Vec<Scored>,
    pub choice: Choice,
    /// Skeletons and devices left out, with the reason.
    pub pruned: Vec<(String, String)>,
}

fn vendor(name: &str) -> u8 {
    let n = name.to_lowercase();
    if ["amd", "radeon", "instinct"].iter().any(|k| n.contains(k)) {
        1
    } else if ["intel", "arc "].iter().any(|k| n.contains(k)) {
        2
    } else if n.contains("apple") {
        3
    } else {
        0
    }
}

fn skeletons(p: &Prep, pruned: &mut Vec<(String, String)>) -> Vec<Skel> {
    let moe = p.model.is_moe();
    let mut units: Vec<Group> = p.compute.iter().map(|d| Group::single(d)).collect();
    let gpus: Vec<&String> = p
        .compute
        .iter()
        .filter(|d| p.kind(d) == Some(DeviceKind::Gpu))
        .take(6)
        .collect();
    for mask in 1u32..(1 << gpus.len()) {
        let devs: Vec<String> = (0..gpus.len())
            .filter(|i| (mask >> i) & 1 == 1)
            .map(|i| gpus[i].clone())
            .collect();
        let g = devs.len() as u32;
        if !(2..=4).contains(&g) {
            continue;
        }
        let name = |d: &String| p.hw.device(d).map_or("", |x| x.name.as_str()).to_string();
        let same_vendor = devs
            .iter()
            .all(|d| vendor(&name(d)) == vendor(&name(&devs[0])));
        let linked = devs.iter().enumerate().all(|(i, a)| {
            devs[i + 1..].iter().all(|b| {
                p.hw.link_between(a, b)
                    .is_some_and(|l| l.kind == LinkKind::P2p)
            })
        });
        if !same_vendor || !linked {
            continue;
        }
        let label = format!("tensor/expert parallel over {}", devs.join("+"));
        if !p.caps.tp_ep {
            pruned.push((label, format!("phase {} cannot execute it", p.phase.0)));
            continue;
        }
        if !p.linux {
            pruned.push((label, "needs NCCL, Linux only".into()));
            continue;
        }
        let m = p.model;
        let experts_ok = !moe || m.experts.is_multiple_of(g);
        if m.heads.is_multiple_of(g) && m.kv_heads.is_multiple_of(g) && experts_ok {
            units.push(Group {
                devs: devs.clone(),
                tp: g,
                ep: 1,
            });
        } else {
            pruned.push((
                label.clone(),
                format!("{g} does not divide heads, KV heads and experts"),
            ));
        }
        if moe && m.experts.is_multiple_of(g) {
            units.push(Group { devs, tp: 1, ep: g });
        }
    }

    fn walk(units: &[Group], max: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        for i in 0..units.len() {
            let clash = cur
                .iter()
                .any(|&j| units[j].devs.iter().any(|d| units[i].devs.contains(d)));
            if clash {
                continue;
            }
            cur.push(i);
            out.push(cur.clone());
            if cur.len() < max {
                walk(units, max, cur, out);
            }
            cur.pop();
        }
    }
    let mut orders = Vec::new();
    walk(
        &units,
        MAX_STAGES.min(p.layers as usize),
        &mut Vec::new(),
        &mut orders,
    );
    let mut out = Vec::new();
    for o in orders {
        let groups: Vec<Group> = o.iter().map(|&i| units[i].clone()).collect();
        let gpu_stage = groups
            .iter()
            .any(|g| p.kind(g.primary()) == Some(DeviceKind::Gpu));
        let mut modes = vec![Mode::Cpu];
        if moe && gpu_stage && p.caps.compute_override != tetraeder_plan::ComputeOverride::None {
            modes.push(Mode::Stream);
        }
        for mode in modes {
            out.push(Skel {
                groups: groups.clone(),
                mode,
            });
        }
    }
    out
}

/// A scored plan with what the simulator needs.
struct Cand {
    scored: Scored,
    spec: ModelSpec,
    graphs: bool,
}

fn rank_by_front(c: &[Cand]) -> Vec<usize> {
    // Non-dominated layers in (tok/s up, loss down), cost-blind so the pick does not depend on
    // `cost` hints.
    let mut rank = vec![usize::MAX; c.len()];
    let mut left: Vec<usize> = (0..c.len()).collect();
    let mut layer = 0;
    while !left.is_empty() {
        let mut order = left.clone();
        order.sort_by(|&a, &b| {
            let (x, y) = (&c[a].scored, &c[b].scored);
            y.tok_s
                .total_cmp(&x.tok_s)
                .then(x.quality_loss.total_cmp(&y.quality_loss))
                .then(a.cmp(&b))
        });
        let mut best = f64::INFINITY;
        let mut front = Vec::new();
        for i in order {
            if c[i].scored.quality_loss < best {
                best = c[i].scored.quality_loss;
                front.push(i);
            }
        }
        for &i in &front {
            rank[i] = layer;
        }
        left.retain(|i| !front.contains(i));
        layer += 1;
    }
    let mut idx: Vec<usize> = (0..c.len()).collect();
    idx.sort_by(|&a, &b| {
        rank[a]
            .cmp(&rank[b])
            .then(c[b].scored.tok_s.total_cmp(&c[a].scored.tok_s))
            .then(a.cmp(&b))
    });
    idx
}

pub fn search(inp: &SearchInput) -> SearchResult {
    let mut pruned: Vec<(String, String)> = Vec::new();
    let p = match Prep::new(inp) {
        Ok(p) => p,
        Err(e) => {
            pruned.push(("model".into(), e));
            return finish(Vec::new(), inp, pruned);
        }
    };
    pruned.extend(p.notes.iter().cloned());
    if p.compute.is_empty() {
        pruned.push(("hardware".into(), "no usable compute device".into()));
        return finish(Vec::new(), inp, pruned);
    }
    let sks = skeletons(&p, &mut pruned);
    let mut seen: HashSet<String> = HashSet::new();
    let mut cands: Vec<Cand> = Vec::new();
    for sk in &sks {
        solve_skeleton(&p, sk, &mut seen, &mut cands, &mut pruned);
    }
    // Simulate the best few.
    for i in rank_by_front(&cands).into_iter().take(SIMULATED) {
        run_sim(&p, &mut cands[i]);
    }
    let scored = cands.into_iter().map(|c| c.scored).collect();
    finish(scored, inp, pruned)
}

fn finish(scored: Vec<Scored>, inp: &SearchInput, pruned: Vec<(String, String)>) -> SearchResult {
    let choice = choose(&scored, inp.hints);
    SearchResult {
        scored,
        choice,
        pruned,
    }
}

fn graph_mode(graphs: bool) -> GraphMode {
    if graphs {
        GraphMode::Graphs
    } else {
        GraphMode::Eager
    }
}

fn hop_out(p: &Prep, sk: &Skel, s: usize) -> Option<(f64, f64)> {
    let n = sk.groups.len();
    hop(
        p.hw,
        sk.groups[s].primary(),
        sk.groups[(s + 1) % n].primary(),
    )
}

fn solve_skeleton(
    p: &Prep,
    sk: &Skel,
    seen: &mut HashSet<String>,
    out: &mut Vec<Cand>,
    pruned: &mut Vec<(String, String)>,
) {
    let moe = p.model.is_moe();
    let desc = sk.describe(moe);
    let n = sk.groups.len();
    let mut skip = |why: String| pruned.push((desc.clone(), why));

    if n > 1 {
        for s in 0..n {
            let (a, b) = (sk.groups[s].primary(), sk.groups[(s + 1) % n].primary());
            if hop(p.hw, a, b).is_none() {
                return skip(format!("no direct link between {a} and {b}"));
            }
        }
    }
    let (first, last) = (sk.groups[0].primary(), sk.groups[n - 1].primary());
    if !p.rules.ok(UnitId::Embed, first, first) || !p.rules.ok(UnitId::Head, last, last) {
        return skip("hints do not allow the embedding or head there".into());
    }

    // Memory lower bound: smallest formats against the capacity after the reserve.
    let min_dense: u64 = p.pre_dense_min[p.layers as usize]
        + p.items[p.embed_item()].opts.last().map_or(0, |o| o.bytes)
        + p.items[p.head_item()].opts.last().map_or(0, |o| o.bytes);
    let min_exp = p.pre_exp_min[p.layers as usize];
    let kv_total = p.kv_layer * u64::from(p.layers);
    let devs: Vec<&String> = sk.groups.iter().flat_map(|g| &g.devs).collect();
    let cap_sk: u64 = devs.iter().map(|d| p.cap(d)).sum();
    let ws_total = p.ws * devs.len() as u64;
    if min_dense + kv_total + ws_total > cap_sk {
        return skip(format!(
            "dense weights {min_dense} B + KV {kv_total} B + workspace exceed {cap_sk} B"
        ));
    }
    let mut cap_all = cap_sk;
    if moe {
        for h in [p.cpu.as_deref(), p.nvme.as_deref()].into_iter().flatten() {
            let usable = !devs.iter().any(|d| d.as_str() == h)
                && (p.kind(h) == Some(DeviceKind::Cpu) || p.caps.expert_tiers);
            if usable {
                cap_all += p.cap(h);
            }
        }
    }
    if min_dense + min_exp + kv_total + ws_total > cap_all {
        return skip(format!(
            "weights at the smallest formats ({} B) exceed the memory of {} ({cap_all} B)",
            min_dense + min_exp,
            devs.iter()
                .map(|d| d.as_str())
                .collect::<Vec<_>>()
                .join("+")
        ));
    }

    let tiers: Vec<StageTiers> = (0..n).map(|s| solve::stage_tiers(p, sk, s)).collect();
    let mask: Vec<Vec<bool>> = sk
        .groups
        .iter()
        .map(|g| {
            let shards = if g.tp > 1 { &g.devs[..] } else { &g.devs[..1] };
            let e = p.experts;
            (0..p.layers)
                .map(|l| {
                    let dense = [UnitId::Attn(l), UnitId::Mlp(l)]
                        .iter()
                        .all(|&u| shards.iter().all(|d| p.rules.ok(u, d, d)));
                    // A parallel group spreads the experts evenly over its devices.
                    let spread = e == 0
                        || g.devs.len() == 1
                        || (0..e).all(|x| {
                            let d = &g.devs[x / (e / g.devs.len())];
                            p.rules.ok(UnitId::Expert(l, x as u32), d, d)
                        });
                    dense && spread
                })
                .collect()
        })
        .collect();

    // Layer split: minimize the slowest stage at the most precise formats.
    let est_mode = if moe
        || sk
            .groups
            .iter()
            .any(|g| p.kind(g.primary()) == Some(DeviceKind::Cpu))
    {
        GraphMode::Eager
    } else {
        GraphMode::Graphs
    };
    let est = |s: usize, a: u32, b: u32| -> f64 {
        stage_estimate(p, sk, &tiers, &mask, est_mode, s, a, b).unwrap_or(f64::INFINITY)
    };
    let Some(split) = split_layers(p.layers, n, est) else {
        return skip("no feasible layer split (memory, hints or links)".into());
    };
    let bounds = split.bounds;

    // Formats: fit the memory, then the compression levels.
    let mut sel = vec![0usize; p.items.len()];
    if let Err(e) = solve::fit(p, sk, &bounds, &mut sel, &tiers) {
        return skip(e);
    }
    let fit_total = solve::total_bytes(p, &sel) as f64;
    let min_total = solve::min_total_bytes(p) as f64;
    let mut variants: Vec<(String, Vec<usize>)> = vec![("fit".to_string(), sel.clone())];
    for frac in LEVELS {
        let s2 = solve::compress_to(p, &sel, fit_total - frac * (fit_total - min_total));
        if variants.iter().all(|v| v.1 != s2) {
            variants.push((format!("level {frac}"), s2));
        }
    }
    for (level, sel) in variants {
        let Ok(stages) = solve::place(p, sk, &bounds, &sel, &tiers) else {
            skip(format!("{level}: placement failed after compression"));
            continue;
        };
        let sv = Solved {
            bounds: bounds.clone(),
            sel,
            stages,
        };
        let graphs = !solve::uses_host(p, sk, &sv.stages);
        let plan = solve::build_plan(p, sk, &sv, graphs, &tiers);
        let key = plan.to_json().unwrap_or_default();
        if !seen.insert(key) {
            continue;
        }
        let ctx = Context {
            hw: p.hw,
            model: p.model,
            phase: p.phase,
            reserve: p.reserve,
            hints: Some(p.hints),
        };
        let expanded = match validate(&plan, &ctx) {
            Ok(e) => e,
            Err(issues) => {
                skip(format!("invalid plan ({level}): {:?}", issues.first()));
                continue;
            }
        };
        let Some(an) = analytic(p, sk, &sv, graphs) else {
            skip(format!("{level}: a stage has no usable time"));
            continue;
        };
        let headroom = expanded
            .memory_per_device()
            .iter()
            .map(|(d, used)| p.cap(d).saturating_sub(*used))
            .min()
            .unwrap_or(0);
        let tok_s = 1.0 / an.token_s;
        out.push(Cand {
            scored: Scored {
                desc: format!("{desc} [{level}]"),
                resource_cost: resource_cost(p.hw, p.hints, &plan),
                plan,
                tok_s,
                ttft_s: an.ttft_s,
                tpot_s: None,
                simulated: false,
                quality_loss: solve::loss_of(p, &sv.sel),
                headroom_bytes: headroom,
                stage_times: an.stage_times,
            },
            spec: model_spec(p, &sv.sel),
            graphs,
        });
    }
}

/// Time of the span `[a, b)` on stage `s` at the most precise formats plus the hop out of it;
/// `None` when the hints, memory lower bound or links forbid it.
#[allow(clippy::too_many_arguments)]
fn stage_estimate(
    p: &Prep,
    sk: &Skel,
    tiers: &[StageTiers],
    mask: &[Vec<bool>],
    mode: GraphMode,
    s: usize,
    a: u32,
    b: u32,
) -> Option<f64> {
    let g = &sk.groups[s];
    let n = sk.groups.len();
    if (a..b).any(|l| !mask[s][l as usize]) {
        return None;
    }
    let (la, lb) = (a as usize, b as usize);
    let len = u64::from(b - a);
    let of = if g.tp > 1 { g.devs.len() as u64 } else { 1 };
    let end_min = |first: bool, k: usize| -> u64 {
        if first {
            p.items[k].opts.last().map_or(0, |o| o.bytes)
        } else {
            0
        }
    };
    let end_max =
        |first: bool, k: usize| -> u64 { if first { p.items[k].opts[0].bytes } else { 0 } };
    let ends_min = end_min(s == 0, p.embed_item()) + end_min(s + 1 == n, p.head_item());
    let ends_max = end_max(s == 0, p.embed_item()) + end_max(s + 1 == n, p.head_item());
    let kv = (p.kv_layer * len).div_ceil(of);
    let mut need = (p.pre_dense_min[lb] - p.pre_dense_min[la]).div_ceil(of) + kv + p.ws + ends_min;
    let group_experts = p.experts > 0 && g.devs.len() > 1;
    if group_experts {
        need += (p.pre_exp_min[lb] - p.pre_exp_min[la]) / g.devs.len() as u64;
    }
    if g.devs.iter().any(|d| need > p.cap(d)) {
        return None;
    }
    let dense_max = (p.pre_dense_max[lb] - p.pre_dense_max[la]).div_ceil(of) + kv + p.ws + ends_max;
    let dense_min = (p.pre_dense_min[lb] - p.pre_dense_min[la]).div_ceil(of) + kv + p.ws + ends_min;
    let layers = f64::from(b - a);
    let m = p.model;
    let e = p.experts.max(1) as u64;
    let exp_max = (p.pre_exp_max[lb] - p.pre_exp_max[la]) / e;
    let exp_min = (p.pre_exp_min[lb] - p.pre_exp_min[la]) / e;
    let mut counts = [0usize; 3];
    if group_experts {
        counts = [p.experts, 0, 0];
    } else if p.experts > 0 {
        let d = g.primary();
        let t = tiers[s].for_span(Span(a, b));
        let gpu = p.kind(d) == Some(DeviceKind::Gpu) && p.caps.expert_tiers;
        let streams = t[2].is_some() || t[1].as_ref().is_some_and(|x| x.1 == d && x.0 != d);
        let helper = t[1].as_ref().is_some_and(|x| x.1 != d);
        let slots = u64::from(solve::slots_for(p));
        let try_counts = |dense: u64, layer_bytes: u64| {
            let slot_bytes = if gpu && streams {
                slots * layer_bytes
            } else {
                0
            };
            let mut avail = [None, None, None];
            if t[0].is_some() {
                avail[0] = Some(p.cap(d).saturating_sub(dense + slot_bytes));
            }
            if let Some((h, _)) = &t[1] {
                avail[1] = Some(p.cap(h).saturating_sub(if helper { p.ws } else { 0 }));
            }
            if let Some((h, _)) = &t[2] {
                avail[2] = Some(p.cap(h));
            }
            solve::fill(avail, layer_bytes, p.experts)
        };
        // At the smallest formats when they do not fit at the most precise ones: the knapsack
        // compresses later, this only has to keep the skeleton alive and cost it.
        counts = try_counts(dense_max, exp_max).or_else(|| try_counts(dense_min, exp_min))?;
    }
    let bytes_per_expert = if p.experts > 0 {
        exp_max as f64 / layers
    } else {
        0.0
    };
    let shape = cost::StageShape {
        group: g.devs.clone(),
        tp: g.tp,
        ep: g.ep,
        layers: b - a,
        nb: (p.pre_dense_max[lb] - p.pre_dense_max[la]) as f64 / layers / of as f64,
        nf: (m.attn.flops_per_token + m.mlp.flops_per_token) / of as f64,
        kv_read: p.read_ctx * m.kv_bytes_per_token_per_layer as f64 / of as f64,
        eb: bytes_per_expert,
        ef: m.expert.as_ref().map_or(0.0, |e| e.flops_per_token),
        counts,
        stream: sk.mode == Mode::Stream,
        cpu: p.cpu.clone(),
        nvme: p.nvme.clone(),
        head_bytes: if s + 1 == n {
            p.items[p.head_item()].opts[0].bytes as f64
        } else {
            0.0
        },
        head_flops: if s + 1 == n {
            m.head.flops_per_token
        } else {
            0.0
        },
    };
    let t = stage_time(p, &shape, 1.0, mode);
    let (alpha, beta) = if n > 1 {
        hop_out(p, sk, s)?
    } else {
        (0.0, f64::INFINITY)
    };
    let total = t + alpha + p.act / beta;
    total.is_finite().then_some(total)
}

struct Analytic {
    token_s: f64,
    ttft_s: f64,
    stage_times: Vec<StageTime>,
}

fn analytic(p: &Prep, sk: &Skel, sv: &Solved, graphs: bool) -> Option<Analytic> {
    let mode = graph_mode(graphs);
    let n = sk.groups.len();
    let mut cm = Vec::with_capacity(n);
    let mut times = Vec::with_capacity(n);
    let mut prefill = 0.0;
    for s in 0..n {
        let sol: &StageSol = &sv.stages[s];
        let shape = solve::shape(p, sk, &sv.bounds, s, &sv.sel, sol.counts);
        let t1 = stage_time(p, &shape, 1.0, mode);
        let (alpha, beta) = hop_out(p, sk, s)?;
        let hop_s = alpha + p.act / beta;
        if !t1.is_finite() || !hop_s.is_finite() {
            return None;
        }
        cm.push(CmStage {
            t_s: t1,
            alpha_s: alpha,
            beta_bytes_per_s: beta,
        });
        let mut devices = sk.groups[s].devs.clone();
        if sol.counts[1] > 0 && sk.mode == Mode::Cpu {
            let extra: Vec<String> = p
                .cpu
                .iter()
                .filter(|c| !devices.contains(c))
                .cloned()
                .collect();
            devices.extend(extra);
        }
        times.push(StageTime {
            stage: s as u32,
            layers: sol.span,
            devices,
            compute_s: t1,
            hop_s,
        });
        prefill += stage_time(p, &shape, f64::from(p.prompt), mode);
        if s + 1 < n {
            prefill += alpha + p.act * f64::from(p.prompt) / beta;
        }
    }
    let token_s = pipeline_token_time(&cm, p.act);
    let (alpha, beta) = hop_out(p, sk, n - 1)?;
    let ttft_s = prefill + alpha + p.act / beta;
    (token_s.is_finite() && token_s > 0.0 && ttft_s.is_finite()).then_some(Analytic {
        token_s,
        ttft_s,
        stage_times: times,
    })
}

/// Uniform per-layer view of the plan for the simulator: what one decode token reads and computes
/// per layer at the plan's formats.
fn model_spec(p: &Prep, sel: &[usize]) -> ModelSpec {
    let m = p.model;
    let l = f64::from(p.layers);
    let bytes = |i: usize| p.items[i].opts[sel[i]].bytes as f64;
    let k = f64::from(m.experts_used.min(m.experts));
    let mut weights = 0.0;
    for layer in 0..p.layers {
        weights += bytes(p.attn_item(layer)) + bytes(p.mlp_item(layer));
        if p.experts > 0 {
            weights += k * bytes(p.expert_item(layer)) / p.experts as f64;
        }
    }
    let exp_flops = m.expert.as_ref().map_or(0.0, |e| k * e.flops_per_token);
    ModelSpec {
        layers: p.layers,
        flops_per_token_per_layer: m.attn.flops_per_token + m.mlp.flops_per_token + exp_flops,
        weight_bytes_per_layer: (weights / l) as u64,
        kv_bytes_per_token_per_layer: m.kv_bytes_per_token_per_layer,
        activation_bytes_per_token: p.act,
    }
}

/// Run the simulator on a hardware copy whose stage devices are scaled so that the simulator's
/// uniform layer costs the same as the analytic stage time at M = 1 (the simulator knows one
/// device per stage and one cost per layer, not tiers, tensor parallel or mixed formats).
fn run_sim(p: &Prep, c: &mut Cand) {
    let mut hw = p.hw.clone();
    let mode = graph_mode(c.graphs);
    let spec = c.spec.clone();
    let mut stages = Vec::new();
    for (st, time) in c.scored.plan.replicas[0]
        .stages
        .iter()
        .zip(&c.scored.stage_times)
    {
        let name = st.devices[0].clone();
        let Some(dev) = hw.devices.iter_mut().find(|d| d.id == name) else {
            return;
        };
        let (Some((f, b)), Some(cs)) = (rates(dev), dev.compute) else {
            return;
        };
        let layers = f64::from(st.layers.len());
        let c0 = match mode {
            GraphMode::Graphs => cs.c0_graph_s,
            GraphMode::Eager => cs.c0_eager_s,
        };
        let roof = (spec.flops_per_token_per_layer / f).max(spec.weight_bytes_per_layer as f64 / b);
        let target = time.compute_s / layers - c0;
        let r = if target > roof * 1e-6 {
            roof / target
        } else {
            1e6
        };
        if let Some(cs) = dev.compute.as_mut() {
            cs.flops_per_s = f * r;
        }
        dev.bandwidth_bytes_per_s = Some(Measurement::prior(b * r));
        dev.thread_curve.clear();
        stages.push(StageSpec {
            device: name,
            layers: st.layers.len(),
        });
    }
    let stub = PlanStub {
        stages,
        prefill_micro_batches: 1,
        graphs: c.graphs,
    };
    let conc = p.conc;
    let wl = WorkloadProfile {
        arrival: Arrival::Closed { concurrency: conc },
        prompt: LengthDist::Fixed { tokens: p.prompt },
        output: LengthDist::Fixed {
            tokens: p.gen_tokens,
        },
        requests: (4 * conc).max(8),
        seed: 1,
    };
    if let Ok(r) = simulate(&hw, &spec, &stub, &wl) {
        c.scored.ttft_s = c.scored.ttft_s.max(r.ttft_s.p95);
        c.scored.tpot_s = Some(r.tpot_s.mean);
        c.scored.simulated = true;
    }
}
