//! Multi-GPU shapes of Phase 5 (ADR 0068, 0070): the fixed candidate set of a box, and the
//! features that price each shape with the multi-rank cost model.
//!
//! - [`enumerate_multi`]: `single`, `layer-<g>`, `tp-<g>`, `ep-<g>` (MoE) and `tp2-pp2` on four
//!   GPUs, each a complete plan that passes the Phase 5 validator and
//!   [`Topology::from_plan`]; shapes the memory check or the block rules prune are listed with the
//!   reason. It uses no calibration (ADR 0040: the set is fixed before scoring).
//! - [`features_multi`]: the [`RankFeatures`] of a topology, `single` included, with shares from
//!   [`block_slice`].
//! - [`expected_max_local`]: the expert-parallel imbalance by a fixed-seed Monte Carlo.

use crate::candidate::CUDA_RESERVE;
use crate::layout::ModelLayout;
use crate::search::{split_layers, vendor};
use crate::serve::budget_of;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::ops::Range;
use tetraeder_costmodel::predict::{
    Coll, DevicePart, Phase as RankPhase, Prediction, RankFeatures, RunFeatures, predict,
};
use tetraeder_plan::{
    Assign, Context, DeviceBytes, Format, Hints, Issue, Knobs, KvFormat, ModelDesc, Phase, Plan,
    Replica, Span, Stage, StageSpec, Topology, UnitKind, UnitSel, block_slice, validate,
};
use tetraeder_profiler::{DeviceKind, HardwareGraph, LinkKind};

/// Context length of the judged runs (ADR 0068).
pub const CONTEXT: u64 = 4096;
/// Physical batch of the judged runs.
pub const UBATCH: u32 = 512;
/// Seed of the expert-parallel Monte Carlo (ADR 0070).
pub const EP_SEED: u64 = 20_261_115;
/// Steps of the expert-parallel Monte Carlo.
pub const EP_STEPS: u32 = 1000;

/// Graph nodes of an attention range and of the dense FFN, as `score::features` counts them
/// (24 per dense layer); the expert ops and the output are the same constants.
const OPS_ATTN: f64 = 14.0;
const OPS_FFN: f64 = 10.0;
const OPS_EXPERT: f64 = 10.0;
const OPS_OUTPUT: f64 = 4.0;
/// The host the logits are read back to.
const HOST: &str = "cpu0";

/// One shape of ADR 0068's table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    Single,
    Layer(u32),
    Tp(u32),
    Ep(u32),
    /// `pp` stages of `tp` ranks each.
    TpPp {
        tp: u32,
        pp: u32,
    },
}

impl Shape {
    pub fn id(self) -> String {
        match self {
            Self::Single => "single".into(),
            Self::Layer(n) => format!("layer-{n}"),
            Self::Tp(n) => format!("tp-{n}"),
            Self::Ep(n) => format!("ep-{n}"),
            Self::TpPp { tp, pp } => format!("tp{tp}-pp{pp}"),
        }
    }

    pub fn parse(id: &str) -> Option<Self> {
        if id == "single" {
            return Some(Self::Single);
        }
        if let Some((a, b)) = id.split_once('-') {
            if let (Some(tp), Some(pp)) = (a.strip_prefix("tp"), b.strip_prefix("pp")) {
                return Some(Self::TpPp {
                    tp: tp.parse().ok()?,
                    pp: pp.parse().ok()?,
                });
            }
            let n: u32 = b.parse().ok()?;
            return match a {
                "layer" => Some(Self::Layer(n)),
                "tp" => Some(Self::Tp(n)),
                "ep" => Some(Self::Ep(n)),
                _ => None,
            };
        }
        None
    }

    fn gpus(self) -> u32 {
        match self {
            Self::Single => 1,
            Self::Layer(n) | Self::Tp(n) | Self::Ep(n) => n,
            Self::TpPp { tp, pp } => tp * pp,
        }
    }
}

/// A shape as a complete plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultiCandidate {
    /// `single`, `layer-2`, `tp-2`, `ep-2`, `tp2-pp2`, ...
    pub id: String,
    pub gpus: Vec<String>,
    pub plan: Plan,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultiCandidates {
    pub candidates: Vec<MultiCandidate>,
    /// `(id, reason)` of the shapes left out.
    pub pruned: Vec<(String, String)>,
}

struct Ctx<'a> {
    hw: &'a HardwareGraph,
    layout: &'a ModelLayout,
    desc: ModelDesc,
    hints: &'a Hints,
    phase: Phase,
    gpus: Vec<String>,
    threads: u32,
    fmt: Format,
    reserve: BTreeMap<String, u64>,
}

/// `(layers, devices, tp, ep)` of a stage.
type StageDef = (Span, Vec<String>, u32, u32);

fn gpu_ids(hw: &HardwareGraph) -> Vec<String> {
    let all: Vec<&tetraeder_profiler::Device> = hw
        .devices
        .iter()
        .filter(|d| d.kind == DeviceKind::Gpu)
        .collect();
    let Some(first) = all.first() else {
        return Vec::new();
    };
    all.iter()
        .filter(|d| vendor(&d.name) == vendor(&first.name))
        .map(|d| d.id.clone())
        .collect()
}

/// The candidate set of the GPUs in `hw`, with the thread quota the process runs under
/// (`available_parallelism`, which follows the container quota).
pub fn enumerate_multi(
    hw: &HardwareGraph,
    layout: &ModelLayout,
    hints: &Hints,
    phase: Phase,
) -> MultiCandidates {
    let quota = std::thread::available_parallelism().map_or(1, |n| n.get() as u32);
    enumerate_multi_with(hw, layout, hints, phase, quota)
}

/// [`enumerate_multi`] with an explicit thread quota.
pub fn enumerate_multi_with(
    hw: &HardwareGraph,
    layout: &ModelLayout,
    hints: &Hints,
    phase: Phase,
    quota: u32,
) -> MultiCandidates {
    let gpus = gpu_ids(hw);
    let mut out = MultiCandidates {
        candidates: Vec::new(),
        pruned: Vec::new(),
    };
    if gpus.is_empty() {
        out.pruned.push(("all".into(), "no GPU".into()));
        return out;
    }
    let ctx = Ctx {
        hw,
        layout,
        desc: layout.to_model_desc(),
        hints,
        phase,
        threads: budget_of(hw, quota, None, None).threads,
        fmt: layout.main_format(),
        reserve: gpus
            .iter()
            .map(|g| (g.clone(), CUDA_RESERVE))
            .collect::<BTreeMap<_, _>>(),
        gpus,
    };
    let g = ctx.gpus.len() as u32;
    let mut shapes = vec![Shape::Single];
    if g >= 2 {
        shapes.push(Shape::Layer(g));
        shapes.push(Shape::Tp(g));
        if layout.is_moe() {
            shapes.push(Shape::Ep(g));
        }
    }
    if g == 4 {
        shapes.push(Shape::TpPp { tp: 2, pp: 2 });
    }
    for s in shapes {
        match build_shape(&ctx, s) {
            Ok(c) => out.candidates.push(c),
            Err(why) => out.pruned.push((s.id(), why)),
        }
    }
    out
}

/// The candidate for shape id `id` on the first GPUs of `hw` (calibration anchors name shapes
/// that are not in the box's fixed set, such as `tp-2` on four GPUs).
pub fn candidate_for(
    hw: &HardwareGraph,
    layout: &ModelLayout,
    hints: &Hints,
    phase: Phase,
    id: &str,
) -> Result<MultiCandidate, String> {
    let shape = Shape::parse(id).ok_or_else(|| format!("unknown shape id {id:?}"))?;
    let gpus = gpu_ids(hw);
    let quota = std::thread::available_parallelism().map_or(1, |n| n.get() as u32);
    let ctx = Ctx {
        hw,
        layout,
        desc: layout.to_model_desc(),
        hints,
        phase,
        threads: budget_of(hw, quota, None, None).threads,
        fmt: layout.main_format(),
        reserve: gpus
            .iter()
            .map(|g| (g.clone(), CUDA_RESERVE))
            .collect::<BTreeMap<_, _>>(),
        gpus,
    };
    build_shape(&ctx, shape)
}

fn kv_per_layer(layout: &ModelLayout) -> u64 {
    layout.kv_bytes_per_token_per_layer_as(KvFormat::default()) * CONTEXT
}

fn workspace(layout: &ModelLayout, last: bool) -> u64 {
    let logits = if last {
        4 * u64::from(layout.n_vocab)
    } else {
        0
    };
    u64::from(UBATCH) * (logits + 64 * u64::from(layout.n_embd))
}

/// Why `tp` ranks cannot cut the model, in whole quantization blocks (ADR 0070).
fn check_cut(layout: &ModelLayout, fmt: Format, tp: u32, ep: u32) -> Result<(), String> {
    let (h, kv) = (layout.n_head, layout.n_head_kv);
    if !h.is_multiple_of(tp) || !kv.is_multiple_of(tp) {
        return Err(format!(
            "{tp} does not divide the {h} heads and {kv} KV heads"
        ));
    }
    let block = fmt.block().0;
    let blocks = |len: u64| len / block;
    let rows = [(
        "attention output rows",
        u64::from(layout.n_head) * u64::from(layout.head_dim),
    )];
    let ffn = if layout.is_moe() {
        if ep > 1 {
            if !layout.n_expert.is_multiple_of(ep) {
                return Err(format!(
                    "{ep} does not divide the {} experts",
                    layout.n_expert
                ));
            }
            None
        } else {
            Some(("expert down rows", u64::from(layout.ff_exp_dim())))
        }
    } else {
        Some(("FFN down rows", u64::from(layout.ff_dim())))
    };
    for (what, len) in rows.into_iter().chain(ffn) {
        if len == 0 {
            return Err(format!("{what}: width unknown"));
        }
        block_slice(len, block, tp, 0, 0)
            .map_err(|_| format!("{} blocks over {tp} ranks ({what})", blocks(len)))?;
    }
    Ok(())
}

fn build_shape(ctx: &Ctx, shape: Shape) -> Result<MultiCandidate, String> {
    let need = shape.gpus() as usize;
    if ctx.gpus.len() < need {
        return Err(format!("needs {need} GPUs, the box has {}", ctx.gpus.len()));
    }
    let gpus: Vec<String> = ctx.gpus[..need].to_vec();
    for (i, a) in gpus.iter().enumerate() {
        for b in &gpus[i + 1..] {
            let ok = ctx
                .hw
                .link_between(a, b)
                .is_some_and(|l| matches!(l.kind, LinkKind::P2p | LinkKind::HostStaged));
            if !ok {
                return Err(format!("no P2P or host-staged link between {a} and {b}"));
            }
        }
    }
    let l = ctx.layout.n_layer;
    let stages: Vec<StageDef> = match shape {
        Shape::Single => vec![(Span(0, l), gpus.clone(), 1, 1)],
        Shape::Tp(n) => {
            check_cut(ctx.layout, ctx.fmt, n, 1)?;
            vec![(Span(0, l), gpus.clone(), n, 1)]
        }
        Shape::Ep(n) => {
            check_cut(ctx.layout, ctx.fmt, n, n)?;
            vec![(Span(0, l), gpus.clone(), n, n)]
        }
        Shape::Layer(n) => spans(ctx, &gpus, n as usize, 1)?,
        Shape::TpPp { tp, pp } => {
            check_cut(ctx.layout, ctx.fmt, tp, 1)?;
            spans(ctx, &gpus, pp as usize, tp as usize)?
        }
    };
    let plan = build_plan(ctx, &stages);
    let rcx = Context {
        hw: ctx.hw,
        model: &ctx.desc,
        phase: ctx.phase,
        reserve: &ctx.reserve,
        hints: Some(ctx.hints),
    };
    let x = validate(&plan, &rcx).map_err(|issues| reason(&issues))?;
    Topology::from_plan(&plan, &x, ctx.hw).map_err(|e| e.join("; "))?;
    let notes = stages
        .iter()
        .map(|s| {
            format!(
                "layers [{}, {}) on {} tp {} ep {}",
                s.0.0,
                s.0.1,
                s.1.join("+"),
                s.2,
                s.3
            )
        })
        .collect();
    Ok(MultiCandidate {
        id: shape.id(),
        gpus,
        plan,
        notes,
    })
}

fn reason(issues: &[Issue]) -> String {
    let first = issues.first();
    match first {
        Some(Issue::MemoryExceeded {
            device,
            used,
            capacity,
            reserve,
        }) => format!(
            "memory: {device} needs {used} B, has {} B",
            capacity.saturating_sub(*reserve)
        ),
        Some(i) => format!("{i}"),
        None => "invalid".into(),
    }
}

/// `n` stages of `tp` GPUs each over `gpus` (stage s takes `gpus[s * tp..]`), spans from the
/// layer DP: the time of a stage is its decode bytes over its bandwidth, infinite when its
/// weights, KV and workspace do not fit.
fn spans(ctx: &Ctx, gpus: &[String], n: usize, tp: usize) -> Result<Vec<StageDef>, String> {
    let lay = ctx.layout;
    let l = lay.n_layer;
    let r = if lay.is_moe() {
        f64::from(lay.n_expert_used) / f64::from(lay.n_expert)
    } else {
        0.0
    };
    let est = |s: usize, a: u32, b: u32| -> f64 {
        let dev = &gpus[s * tp];
        let Some(d) = ctx.hw.device(dev) else {
            return f64::INFINITY;
        };
        let Some(bw) = d.bandwidth_bytes_per_s.map(|m| m.value) else {
            return f64::INFINITY;
        };
        let (mut decode, mut weights) = (0.0f64, 0.0f64);
        for x in &lay.layers[a as usize..b as usize] {
            decode += (x.attn_bytes + x.ffn_bytes) as f64 + r * x.expert_bytes as f64;
            weights += (x.attn_bytes + x.ffn_bytes + x.expert_bytes) as f64;
        }
        let first = s == 0;
        let last = b == l;
        if first {
            weights += lay.embed_bytes as f64;
        }
        if last {
            decode += lay.output_bytes as f64;
            weights += lay.output_bytes as f64;
        }
        let kv = f64::from(b - a) * kv_per_layer(lay) as f64;
        let per_dev = (weights + kv) / tp as f64 + workspace(lay, last) as f64;
        let cap = d.capacity_bytes.saturating_sub(CUDA_RESERVE) as f64;
        if per_dev > cap {
            return f64::INFINITY;
        }
        decode / tp as f64 / bw
    };
    let split = split_layers(l, n, est).ok_or_else(|| {
        "memory: no layer split fits the GPUs (weights, KV and workspace)".to_string()
    })?;
    Ok((0..n)
        .map(|s| {
            (
                Span(split.bounds[s], split.bounds[s + 1]),
                gpus[s * tp..(s + 1) * tp].to_vec(),
                tp as u32,
                1,
            )
        })
        .collect())
}

fn build_plan(ctx: &Ctx, stages: &[StageDef]) -> Plan {
    let lay = ctx.layout;
    let e = lay.n_expert;
    let n = stages.len();
    let rule =
        |kinds: &[UnitKind], layers: Option<Span>, experts: Option<Span>, home: Vec<String>| {
            Assign {
                units: UnitSel {
                    kinds: kinds.to_vec(),
                    layers,
                    experts,
                },
                home: Some(home),
                compute: None,
                format: Some(ctx.fmt),
                role_formats: BTreeMap::new(),
                kv_format: None,
            }
        };
    let mut out = Vec::new();
    for (s, (span, devs, tp, ep)) in stages.iter().enumerate() {
        let (first, last) = (s == 0, s + 1 == n);
        let cut: Vec<String> = if *tp > 1 {
            devs.clone()
        } else {
            vec![devs[0].clone()]
        };
        let mut assign = vec![rule(
            &[UnitKind::Attn, UnitKind::Mlp],
            Some(*span),
            None,
            cut.clone(),
        )];
        if first {
            assign.push(rule(&[UnitKind::Embed], None, None, vec![devs[0].clone()]));
        }
        if last {
            assign.push(rule(&[UnitKind::Head], None, None, cut.clone()));
        }
        if lay.is_moe() {
            if *ep > 1 {
                let per = e / ep;
                for (i, d) in devs.iter().enumerate() {
                    let sp = Span(i as u32 * per, (i as u32 + 1) * per);
                    assign.push(rule(
                        &[UnitKind::Expert],
                        Some(*span),
                        Some(sp),
                        vec![d.clone()],
                    ));
                }
            } else {
                assign.push(rule(&[UnitKind::Expert], Some(*span), None, cut.clone()));
            }
        }
        let kv = (kv_per_layer(lay) * u64::from(span.len())).div_ceil(devs.len() as u64);
        out.push(Stage {
            layers: *span,
            devices: devs.clone(),
            tp: Some(*tp),
            ep: Some(*ep),
            assign,
            kv: Some(
                devs.iter()
                    .map(|d| DeviceBytes {
                        device: d.clone(),
                        bytes: kv,
                    })
                    .collect(),
            ),
            workspace: devs
                .iter()
                .map(|d| DeviceBytes {
                    device: d.clone(),
                    bytes: workspace(lay, last),
                })
                .collect(),
            ..Default::default()
        });
    }
    Plan {
        model: lay.model.clone(),
        knobs: Knobs {
            prefill_micro_batches: 1,
            cuda_graphs: true,
            max_batch: None,
            threads: Some(ctx.threads),
            ubatch: Some(UBATCH),
            op_offload: Some(false),
        },
        replicas: vec![Replica { stages: out }],
    }
}

/// The topology of a plan on `hw` (validates it at `phase`).
pub fn topology_of(
    hw: &HardwareGraph,
    layout: &ModelLayout,
    plan: &Plan,
    phase: Phase,
) -> Result<Topology, String> {
    let desc = layout.to_model_desc();
    let reserve: BTreeMap<String, u64> = hw
        .devices
        .iter()
        .filter(|d| d.kind == DeviceKind::Gpu)
        .map(|d| (d.id.clone(), CUDA_RESERVE))
        .collect();
    let ctx = Context {
        hw,
        model: &desc,
        phase,
        reserve: &reserve,
        hints: None,
    };
    let x = validate(plan, &ctx).map_err(|i| reason(&i))?;
    Topology::from_plan(plan, &x, hw).map_err(|e| e.join("; "))
}

/// The cost model's prediction for a plan of this module on `hw`.
pub fn predict_candidate(
    hw: &HardwareGraph,
    layout: &ModelLayout,
    plan: &Plan,
    phase: Phase,
    workload: (u32, u32),
    routing: Option<&[Vec<f64>]>,
) -> Result<Prediction, String> {
    let topo = topology_of(hw, layout, plan, phase)?;
    let f = features_multi(layout, &topo, &plan.knobs, None, routing)?;
    predict(hw, &f, workload.0, workload.1).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------------------------
// Expert-parallel imbalance

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `E[max over ranks of the experts a token routes to that rank]` for `top_k` experts drawn
/// without replacement with probability proportional to `freq`, experts cut into `ranks`
/// contiguous spans. `steps` draws from the SplitMix64 stream of `seed`; only additions and
/// multiplications on f64, so the value is the same on every platform.
pub fn expected_max_local(freq: &[f64], top_k: usize, ranks: usize, steps: u32, seed: u64) -> f64 {
    let e = freq.len();
    if e == 0 || ranks == 0 || steps == 0 {
        return 0.0;
    }
    let per = (e / ranks).max(1);
    let mut state = seed;
    let mut sum = 0.0;
    for _ in 0..steps {
        let mut w = freq.to_vec();
        let mut total: f64 = w.iter().sum();
        let mut count = vec![0u32; ranks];
        for _ in 0..top_k.min(e) {
            let u = (splitmix(&mut state) >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0);
            let mut target = u * total;
            let mut pick = None;
            for (i, &x) in w.iter().enumerate() {
                if x <= 0.0 {
                    continue;
                }
                if target < x {
                    pick = Some(i);
                    break;
                }
                target -= x;
            }
            let Some(i) = pick.or_else(|| w.iter().rposition(|&x| x > 0.0)) else {
                break;
            };
            total -= w[i];
            w[i] = 0.0;
            count[(i / per).min(ranks - 1)] += 1;
        }
        sum += f64::from(count.iter().copied().max().unwrap_or(0));
    }
    sum / f64::from(steps)
}

/// Largest share of the selections that lands on one rank.
fn max_mass(freq: &[f64], ranks: usize) -> f64 {
    let per = (freq.len() / ranks.max(1)).max(1);
    let total: f64 = freq.iter().sum();
    if total <= 0.0 {
        return 1.0 / ranks.max(1) as f64;
    }
    (0..ranks)
        .map(|r| {
            freq[(r * per).min(freq.len())..((r + 1) * per).min(freq.len())]
                .iter()
                .sum::<f64>()
        })
        .fold(0.0, f64::max)
        / total
}

// ---------------------------------------------------------------------------------------------
// Features

struct Dims {
    bpp: f64,
    kv_layer: f64,
    attn_layer: f64,
    r: f64,
    block: u64,
    act: f64,
}

fn part(dev: &str, bytes: f64, ops: f64, kv: f64, attn: f64, bpp: f64) -> DevicePart {
    DevicePart {
        device: dev.to_string(),
        weight_bytes_per_token: bytes,
        params_per_token: bytes / bpp,
        resident_matmul_bytes: bytes,
        ops_per_token: ops,
        kv_bytes_per_ctx_token: kv,
        attn_flops_per_ctx_token: attn,
        expert_bytes_per_token: 0.0,
        expert_params_per_token: 0.0,
    }
}

/// `p` with routed-expert matmuls: `bytes` read per step, `params` used per token.
fn with_experts(mut p: DevicePart, bytes: f64, params: f64) -> DevicePart {
    p.expert_bytes_per_token = bytes;
    p.expert_params_per_token = params;
    p
}

fn push_phase(v: &mut Vec<RankPhase>, p: RankPhase) {
    match v.iter_mut().find(|q| {
        q.parts == p.parts
            && q.prefill_parts == p.prefill_parts
            && q.ranges_per_rank == p.ranges_per_rank
            && q.after == p.after
            && q.out_rows == p.out_rows
    }) {
        Some(q) => q.repeat += p.repeat,
        None => v.push(p),
    }
}

/// The features of `topo` running `layout` as phases between collectives (ADR 0070), a one-rank
/// `single` included. `layer_limit` keeps the first N layers and the head (an anchor, ADR 0068).
/// `routing` holds per layer per expert selection frequencies for expert parallel; uniform when
/// `None`.
///
/// What the builders compute, as MODEL_VERSION 2 prices it: expert matmuls apart from the dense
/// ones (their own rate), a prefill step reads every local expert (`prefill_parts`), and the head
/// runs on the output rows only (`out_rows`); the dense builder also runs the last layer's FFN on
/// them (`qwen3/ranked.rs`), the MoE shard builder does not (`qwen3moe_shard.rs`).
pub fn features_multi(
    layout: &ModelLayout,
    topo: &Topology,
    knobs: &Knobs,
    layer_limit: Option<u32>,
    routing: Option<&[Vec<f64>]>,
) -> Result<RunFeatures, String> {
    let lim = layer_limit.unwrap_or(layout.n_layer).min(layout.n_layer);
    if lim == 0 {
        return Err("layer limit 0".into());
    }
    let threads = knobs.threads.unwrap_or(1);
    let ubatch = knobs.ubatch.unwrap_or(UBATCH);
    let last_ffn_on_outputs = !layout.is_moe();

    let d = Dims {
        bpp: layout.main_format().bits_per_weight() / 8.0,
        kv_layer: 2.0 * f64::from(layout.n_head_kv) * f64::from(layout.head_dim) * 2.0,
        attn_layer: 4.0 * f64::from(layout.n_head) * f64::from(layout.head_dim),
        r: if layout.is_moe() {
            f64::from(layout.n_expert_used) / f64::from(layout.n_expert)
        } else {
            0.0
        },
        block: layout.main_format().block().0,
        act: 4.0 * f64::from(layout.n_embd),
    };
    let mut stages: Vec<(&StageSpec, Range<u32>)> = Vec::new();
    for st in &topo.stages {
        let (a, b) = (st.layers.0, st.layers.1.min(lim));
        if a < b {
            stages.push((st, a..b));
        }
    }
    if stages.is_empty() {
        return Err("no stage within the layer limit".into());
    }
    let dev = |r: usize| topo.ranks[r].device.clone();
    let mut phases: Vec<RankPhase> = Vec::new();
    let mut uniform_ep: Option<(f64, f64)> = None;
    let n = stages.len();
    for (s, (st, span)) in stages.iter().enumerate() {
        let last = s + 1 == n;
        let devs: Vec<String> = st.ranks.iter().map(|&r| dev(r)).collect();
        let tp = st.tp as usize;
        let hop = (!last).then(|| Coll::Copy {
            pairs: devs
                .iter()
                .cloned()
                .zip(stages[s + 1].0.ranks.iter().map(|&r| dev(r)))
                .collect(),
            bytes_per_token: d.act,
            per_step: false,
        });
        let readback = Coll::Copy {
            pairs: devs.iter().map(|x| (x.clone(), HOST.to_string())).collect(),
            bytes_per_token: 4.0 * f64::from(layout.n_vocab) / tp as f64,
            per_step: true,
        };
        if tp == 1 {
            // A stage of one rank is one range per step: its layers summed, then (last stage)
            // the work on the output rows in the same range.
            let (mut w, mut ex, mut ops, mut kv, mut attn) = (0.0, 0.0, 0.0, 0.0, 0.0);
            let (mut out_w, mut out_ops) = (0.0, 0.0);
            for l in span.clone() {
                let lb = &layout.layers[l as usize];
                if last && last_ffn_on_outputs && l + 1 == span.end {
                    w += lb.attn_bytes as f64;
                    out_w += lb.ffn_bytes as f64;
                    ops += OPS_ATTN;
                    out_ops += OPS_FFN;
                } else {
                    w += (lb.attn_bytes + lb.ffn_bytes) as f64;
                    ops += OPS_ATTN + OPS_FFN;
                }
                ex += lb.expert_bytes as f64;
                ops += if lb.expert_bytes > 0 { OPS_EXPERT } else { 0.0 };
                kv += d.kv_layer;
                attn += d.attn_layer;
            }
            let layers = part(&devs[0], w, ops, kv, attn, d.bpp);
            // decode reads the chosen experts, a prefill every expert
            let pre = (ex > 0.0).then(|| vec![with_experts(layers.clone(), ex, d.r * ex / d.bpp)]);
            phases.push(RankPhase {
                repeat: 1,
                parts: vec![with_experts(layers, d.r * ex, d.r * ex / d.bpp)],
                prefill_parts: pre,
                ranges_per_rank: 1,
                after: hop,
                out_rows: false,
            });
            if last {
                out_w += layout.output_bytes as f64;
                out_ops += OPS_OUTPUT;
                phases.push(RankPhase {
                    repeat: 1,
                    parts: vec![part(&devs[0], out_w, out_ops, 0.0, 0.0, d.bpp)],
                    prefill_parts: None,
                    ranges_per_rank: 0,
                    after: Some(readback),
                    out_rows: true,
                });
            }
            continue;
        }
        if !layout.n_head.is_multiple_of(st.tp) || !layout.n_head_kv.is_multiple_of(st.tp) {
            return Err(format!("tp {tp} does not divide the heads"));
        }
        let group = Coll::AllReduce {
            group: devs.clone(),
            bytes_per_token: d.act,
        };
        for l in span.clone() {
            let lb = &layout.layers[l as usize];
            let rot = l % st.tp;
            // Attention: heads split evenly (checked above); `wo` rows in whole blocks.
            let wo = u64::from(layout.n_head) * u64::from(layout.head_dim);
            for i in 0..st.tp {
                block_slice(wo, d.block, st.tp, i, rot)?;
            }
            let attn_parts = devs
                .iter()
                .map(|x| {
                    part(
                        x,
                        lb.attn_bytes as f64 / tp as f64,
                        OPS_ATTN,
                        d.kv_layer / tp as f64,
                        d.attn_layer / tp as f64,
                        d.bpp,
                    )
                })
                .collect();
            push_phase(
                &mut phases,
                RankPhase {
                    repeat: 1,
                    parts: attn_parts,
                    prefill_parts: None,
                    ranges_per_rank: 1,
                    after: Some(group.clone()),
                    out_rows: false,
                },
            );
            let ffn = ffn_phase(layout, st, &devs, l, rot, &d, routing, &mut uniform_ep)?;
            push_phase(
                &mut phases,
                RankPhase {
                    after: Some(group.clone()),
                    out_rows: last && last_ffn_on_outputs && l + 1 == span.end,
                    ..ffn
                },
            );
        }
        if last {
            let parts = devs
                .iter()
                .map(|x| {
                    part(
                        x,
                        layout.output_bytes as f64 / tp as f64,
                        OPS_OUTPUT,
                        0.0,
                        0.0,
                        d.bpp,
                    )
                })
                .collect();
            phases.push(RankPhase {
                repeat: 1,
                parts,
                prefill_parts: None,
                ranges_per_rank: 1,
                after: Some(readback),
                out_rows: true,
            });
        } else if let Some(h) = hop {
            // The hop after the last all-reduce: nothing computes, the pairs copy at once.
            let parts = devs
                .iter()
                .map(|x| part(x, 0.0, 0.0, 0.0, 0.0, d.bpp))
                .collect();
            phases.push(RankPhase {
                repeat: 1,
                parts,
                prefill_parts: None,
                ranges_per_rank: 0,
                after: Some(h),
                out_rows: false,
            });
        }
    }

    // Per-rank totals, for the record; the step time comes from the phases.
    let mut totals: BTreeMap<String, DevicePart> = BTreeMap::new();
    for ph in &phases {
        for p in &ph.parts {
            let t = totals
                .entry(p.device.clone())
                .or_insert_with(|| part(&p.device, 0.0, 0.0, 0.0, 0.0, d.bpp));
            let k = f64::from(ph.repeat);
            t.weight_bytes_per_token += k * p.weight_bytes_per_token;
            t.params_per_token += k * p.params_per_token;
            t.resident_matmul_bytes += k * p.resident_matmul_bytes;
            t.ops_per_token += k * p.ops_per_token;
            t.kv_bytes_per_ctx_token += k * p.kv_bytes_per_ctx_token;
            t.attn_flops_per_ctx_token += k * p.attn_flops_per_ctx_token;
            t.expert_bytes_per_token += k * p.expert_bytes_per_token;
            t.expert_params_per_token += k * p.expert_params_per_token;
        }
    }
    Ok(RunFeatures {
        parts: totals.into_values().collect(),
        switches_per_token: 0.0,
        activation_bytes_per_token: d.act,
        threads,
        ubatch,
        op_offload: false,
        graphs: knobs.cuda_graphs,
        tier: None,
        ranks: Some(RankFeatures {
            phases,
            step_fixed: true,
        }),
    })
}

/// The FFN (dense or MoE) phase of layer `l` in a tensor-parallel stage, without its collective.
#[allow(clippy::too_many_arguments)]
fn ffn_phase(
    layout: &ModelLayout,
    st: &StageSpec,
    devs: &[String],
    l: u32,
    rot: u32,
    d: &Dims,
    routing: Option<&[Vec<f64>]>,
    uniform_ep: &mut Option<(f64, f64)>,
) -> Result<RankPhase, String> {
    let lb = &layout.layers[l as usize];
    let tp = st.tp;
    let width = |len: u32| -> Result<Vec<f64>, String> {
        (0..tp)
            .map(|i| {
                block_slice(u64::from(len), d.block, tp, i, rot)
                    .map(|r| (r.end - r.start) as f64 / f64::from(len))
            })
            .collect()
    };
    let mk = |parts: Vec<DevicePart>, pre: Option<Vec<DevicePart>>| RankPhase {
        repeat: 1,
        parts,
        prefill_parts: pre,
        ranges_per_rank: 1,
        after: None,
        out_rows: false,
    };
    if !layout.is_moe() {
        let shares = width(layout.ff_dim())?;
        let parts = devs
            .iter()
            .zip(&shares)
            .map(|(x, s)| part(x, lb.ffn_bytes as f64 * s, OPS_FFN, 0.0, 0.0, d.bpp))
            .collect();
        return Ok(mk(parts, None));
    }
    let ops = OPS_FFN + OPS_EXPERT;
    let dense = lb.ffn_bytes as f64;
    let ex = lb.expert_bytes as f64;
    if st.ep == 1 {
        // Experts cut like the rest: every rank reads its column share of the chosen experts at
        // decode and of every expert at prefill.
        let shares = width(layout.ff_exp_dim())?;
        let ranks = || devs.iter().zip(&shares);
        let exp = |x: &str, bytes: f64, s: f64| {
            with_experts(
                part(x, dense, ops, 0.0, 0.0, d.bpp),
                bytes,
                d.r * ex * s / d.bpp,
            )
        };
        let decode = ranks().map(|(x, s)| exp(x, d.r * ex * s, *s)).collect();
        let prefill = ranks().map(|(x, s)| exp(x, ex * s, *s)).collect();
        return Ok(mk(decode, Some(prefill)));
    }
    // Whole experts in contiguous spans: the busiest rank sets the time.
    let e = layout.n_expert as usize;
    let g = st.ep as usize;
    let uniform = vec![1.0; e];
    let given = routing
        .and_then(|r| r.get(l as usize))
        .filter(|f| f.len() == e);
    let is_uniform = given.is_none();
    let freq: &[f64] = given.map_or(&uniform[..], |f| &f[..]);
    let (dec_max, mass) = match (is_uniform, *uniform_ep) {
        (true, Some(v)) => v,
        _ => {
            let v = (
                expected_max_local(freq, layout.n_expert_used as usize, g, EP_STEPS, EP_SEED),
                max_mass(freq, g),
            );
            if is_uniform {
                *uniform_ep = Some(v);
            }
            v
        }
    };
    let eb = ex / e as f64;
    let k = f64::from(layout.n_expert_used);
    let decode: Vec<DevicePart> = devs
        .iter()
        .map(|x| {
            let p = part(x, dense, ops, 0.0, 0.0, d.bpp);
            with_experts(p, dec_max * eb, dec_max * eb / d.bpp)
        })
        .collect();
    // A prefill reads every local expert once; its FLOPs follow the busiest rank's pair share.
    let local = (e / g) as f64 * eb;
    let prefill: Vec<DevicePart> = devs
        .iter()
        .map(|x| {
            let p = part(x, dense, ops, 0.0, 0.0, d.bpp);
            with_experts(p, local, k * mass * eb / d.bpp)
        })
        .collect();
    Ok(mk(decode, Some(prefill)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_ids_roundtrip() {
        for s in [
            Shape::Single,
            Shape::Layer(4),
            Shape::Tp(2),
            Shape::Ep(2),
            Shape::TpPp { tp: 2, pp: 2 },
        ] {
            assert_eq!(Shape::parse(&s.id()), Some(s));
        }
        assert_eq!(Shape::parse("tp-x"), None);
        assert_eq!(Shape::parse("bogus"), None);
    }

    #[test]
    fn ep_monte_carlo_is_deterministic_and_between_the_mean_and_top_k() {
        let freq = vec![1.0; 128];
        let a = expected_max_local(&freq, 8, 2, EP_STEPS, EP_SEED);
        let b = expected_max_local(&freq, 8, 2, EP_STEPS, EP_SEED);
        assert_eq!(a.to_bits(), b.to_bits());
        assert!((4.0..8.0).contains(&a), "{a}");
        // more ranks: each holds fewer of the 8, the max falls but stays above top_k / ranks
        let four = expected_max_local(&freq, 8, 4, EP_STEPS, EP_SEED);
        assert!((2.0..a).contains(&four), "{four} {a}");
        // all weight on rank 0's experts: every pick lands there
        let mut skew = vec![0.0; 128];
        skew[..64].fill(1.0);
        assert_eq!(expected_max_local(&skew, 8, 2, 50, EP_SEED), 8.0);
        assert!((max_mass(&skew, 2) - 1.0).abs() < 1e-12);
        assert!((max_mass(&freq, 2) - 0.5).abs() < 1e-12);
    }
}
