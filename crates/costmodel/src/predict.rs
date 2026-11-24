//! Analytic prediction of a llama.cpp style run: prefill rate `pp`, decode rate `tg` and the
//! request time `P / pp + N / tg` (ADR 0040, "Cost model"). It needs only a `HardwareGraph` and
//! the run's features, never a planner type.
//!
//! Notation: per device part `ops`, `W` (weight bytes read per decode token), `p` (matmul
//! params per token, FLOPs = `2 p M`), `kv` and `a` (KV bytes and attention FLOPs per context
//! token), `R` (resident matmul bytes). `F`, `B`, `c0` are the part's device rates: a GPU uses
//! its `ComputeSpec` and bandwidth with `c0_graph_s` when `graphs` else `c0_eager_s`; a CPU takes
//! `F` and `B` from its thread curve at `threads` ([`Device::cpu_rates`]) and `c0_eager_s`
//! (the CPU has no graphs). The hop uses the link between the parts' devices: `alpha`, and
//! `beta = min(link beta, lane group capacity)`.
//!
//! Decode (depth `ctx = gen / 2`, the average over the run), seconds per token:
//!
//! ```text
//! t_part = c0 * ops + max((2 p + a ctx) / F, (W + kv ctx) / B)
//! t_tok  = sum_parts t_part + switches * (alpha + act / beta)
//! tg     = 1 / t_tok
//! ```
//!
//! Prefill: `P` tokens in ubatches of `M = min(ubatch, remaining)`; `ctx` is the mean context of
//! the ubatch (tokens done before it plus `M / 2`):
//!
//! ```text
//! t_part = c0 * ops + max((2 p M + a M ctx) / F, (W + kv ctx) / B)
//! t_hop  = switches * (alpha + M act / beta)
//! ```
//!
//! With `op_offload` and `M >= 32` a CPU part's matmuls run on the first GPU part's device
//! (needs one) after their weights cross the link once per ubatch. The part then costs
//!
//! ```text
//! t_copy = alpha + R / beta                         (breakdown "offload_copy:<dev>")
//! t_gpu  = max(2 p M / F_gpu, W / B_gpu)            (breakdown "offload_compute:<dev>")
//! t_rest = c0 * ops + max(a M ctx / F, kv ctx / B)  (non-matmul ops and attention stay on the CPU)
//! ```
//!
//! and the hop term is still charged (the CPU part's other ops exchange activations).
//! `pp = P / sum_ubatches (...)`.
//!
//! Disk reads (ADR 0056) are charged only when `tier` is set and moves bytes; without it every
//! number above is unchanged. With `sync` and `prefetch` the decode bytes per token that the
//! RAM tier could not serve, `B = disk_rate(depth)` of the tier's disk, `t_s = sync / B`,
//! `t_p = prefetch / B`, `D = t_s + t_p`, `C_exp` the compute terms of the CPU parts and `C_rest`
//! every other term (the switches included), the decode time per token becomes
//!
//! ```text
//! Reader:          t_tok = C_rest + C_exp + (sync + prefetch) / B_eff,  1/B_eff = 1/B + 1/B_copy
//! Service:         t_tok = max(C_rest + C_exp + t_s, D)
//! ServiceOverlap:  t_tok = max(C_rest + max(C_exp', t_s), D)
//! ```
//!
//! `B_copy` is the CPU's byte rate at one thread (the reader copies serially) and `C_exp'` is
//! `C_exp` at `threads - 1` (at least 1) when `sync > 0`: the overlapped service runs compute one
//! thread short. The service formulas are lower bounds: the critical chain or the busy disk,
//! whichever is longer. Prefill reads its misses synchronously: each ubatch of `M` tokens adds
//! `M * prefill_bytes / disk_rate(sync_depth)`. The added time is the breakdown term
//! `read:<disk>`. Breakdown labels are `compute:<dev>`, `switches`, `offload_copy:<dev>`,
//! `offload_compute:<dev>`, `read:<disk>`; `tg_breakdown` is seconds per token, `pp_breakdown`
//! is seconds over the whole prompt.

use crate::{allreduce_time, transfer_time};
use serde::{Deserialize, Serialize};
use std::fmt;
use tetraeder_profiler::{CollectiveOp, Device, DeviceKind, HardwareGraph, curve_time};

/// One device's share of the model for a run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DevicePart {
    /// Id in the `HardwareGraph`.
    pub device: String,
    /// Weights read per decode token: dense weights fully, experts at top-k over E.
    pub weight_bytes_per_token: f64,
    /// Matmul params used per token: FLOPs = 2 * params * M.
    pub params_per_token: f64,
    /// Matmul weight bytes resident here (what op offload copies per ubatch from the CPU).
    pub resident_matmul_bytes: f64,
    /// ggml graph nodes per decode step on this device.
    pub ops_per_token: f64,
    pub kv_bytes_per_ctx_token: f64,
    pub attn_flops_per_ctx_token: f64,
    /// Routed-expert matmuls, kept apart from the fields above and priced at the expert rate
    /// (ranks path only, ADR 0070): bytes read per step and params used per token.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub expert_bytes_per_token: f64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub expert_params_per_token: f64,
    /// Activation bytes per token of the routed slots outside the matmuls (gate and up outputs,
    /// swiglu, input quantization, the weighting and the sum over slots; ADR 0080). Priced at
    /// `B` only on graphs with `RangeCosts::expert_slot_traffic`.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub expert_slot_bytes_per_token: f64,
}

fn is_zero(x: &f64) -> bool {
    *x == 0.0
}

/// How the bytes of a RAM-tier miss are read (ADR 0056, section 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadPath {
    /// The single reader thread: reads, then copies serially.
    Reader,
    /// The I/O service (ADR 0054): reads in parallel, compute waits for them.
    Service,
    /// The service with CPU experts computed in arrival order.
    ServiceOverlap,
}

/// Bytes the disk serves and how, for a run with a RAM tier (from the routing replay).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TierFeatures {
    /// Device id of the disk in the `HardwareGraph`.
    pub disk: String,
    pub path: ReadPath,
    /// Queue depth of the reads of `path`.
    pub depth: u32,
    /// Direct-I/O queue depth of the synchronous reads (prefill misses).
    pub sync_depth: u32,
    /// Decode bytes per token read synchronously (the compute waits for them).
    pub decode_sync_bytes_per_token: f64,
    /// Decode bytes per token read by the prefetch.
    pub decode_prefetch_bytes_per_token: f64,
    /// Prefill bytes per prompt token (read synchronously).
    pub prefill_bytes_per_token: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunFeatures {
    pub parts: Vec<DevicePart>,
    /// Backend switches per decode token; each is a sync plus an activation hop.
    pub switches_per_token: f64,
    pub activation_bytes_per_token: f64,
    pub threads: u32,
    pub ubatch: u32,
    pub op_offload: bool,
    /// CUDA graphs on (the default).
    pub graphs: bool,
    /// The RAM tier's disk reads; `None` charges none (ADR 0056).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier: Option<TierFeatures>,
    /// Several ranks (ADR 0070): the step time comes from these phases and `parts` only names
    /// the devices; `None` keeps every number of the single-run model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ranks: Option<RankFeatures>,
}

/// What runs on several ranks, as phases between collectives (ADR 0070).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankFeatures {
    pub phases: Vec<Phase>,
    /// Charge `c_step` once per step.
    #[serde(default)]
    pub step_fixed: bool,
    /// Prefill micro-batches pipelined over the stages (ADR 0077); 1 runs the stages in turn.
    #[serde(default = "one_u32", skip_serializing_if = "is_one")]
    pub prefill_micro_batches: u32,
    /// `layout.model` of the run (ADR 0082): the key into `HardwareGraph::decode_clocks`.
    /// `None` prices decode as cost model 4.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

fn one_u32() -> u32 {
    1
}

fn is_one(x: &u32) -> bool {
    *x == 1
}

/// What runs concurrently on the ranks between two collectives, `repeat` times per step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Phase {
    pub repeat: u32,
    /// One part per rank with its share per layer (or per span) of bytes, params, ops and KV.
    pub parts: Vec<DevicePart>,
    /// Parts for a prefill step when they differ from the decode `parts` (experts: a prefill
    /// reads every local expert).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_parts: Option<Vec<DevicePart>>,
    pub ranges_per_rank: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after: Option<Coll>,
    /// The phase computes only the step's output rows (the head, and the dense builder's last
    /// FFN): a prefill step prices it, and its collective, at [`OUTPUTS_PER_STEP`] rows.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub out_rows: bool,
}

/// Output rows of a step: the bench and the judged request take the last row's logits.
pub const OUTPUTS_PER_STEP: f64 = 1.0;

/// A collective that ends a phase.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Coll {
    AllReduce {
        /// Device ids of the group.
        group: Vec<String>,
        bytes_per_token: f64,
    },
    Copy {
        /// `(src, dst)` pairs that copy at once.
        pairs: Vec<(String, String)>,
        bytes_per_token: f64,
        /// The bytes are per step, not per token (the logit readback).
        #[serde(default)]
        per_step: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Prediction {
    pub pp_tok_s: f64,
    pub tg_tok_s: f64,
    /// `prompt / pp + gen / tg`.
    pub request_s: f64,
    /// Seconds per decode token per term.
    pub tg_breakdown: Vec<(String, f64)>,
    /// Seconds over the whole prompt per term.
    pub pp_breakdown: Vec<(String, f64)>,
    /// Label of the largest `tg_breakdown` term.
    pub binding: String,
    /// The decode clock assumed for each GPU of a ranks plan (ADR 0082); empty without
    /// `HardwareGraph::decode_clocks`. Not part of the step sum.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub decode_clocks: Vec<DecodeClock>,
}

/// One GPU's decode state in a prediction (ADR 0082).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodeClock {
    pub device: String,
    /// Busy share of the unscaled decode step.
    pub duty: f64,
    /// Factor on the clock-bound terms of the GPU's decode part (1: busy state or no gap).
    pub factor: f64,
    /// The SM clock the prediction assumes.
    pub mhz: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PredictError {
    Invalid(String),
    UnknownDevice(String),
    MissingSpec { device: String, what: &'static str },
    NoLink { a: String, b: String },
}

impl fmt::Display for PredictError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(s) => write!(f, "invalid run features: {s}"),
            Self::UnknownDevice(d) => write!(f, "device {d} is not in the hardware graph"),
            Self::MissingSpec { device, what } => write!(f, "device {device} has no {what}"),
            Self::NoLink { a, b } => write!(f, "no link between {a} and {b}"),
        }
    }
}

impl std::error::Error for PredictError {}

#[derive(Clone, Copy)]
struct Rates {
    f: f64,
    /// Routed-expert matmuls (the ranks path only).
    f_exp: f64,
    b: f64,
    c0: f64,
}

fn rates(dev: &Device, threads: u32, graphs: bool) -> Result<Rates, PredictError> {
    let missing = |what| PredictError::MissingSpec {
        device: dev.id.clone(),
        what,
    };
    let c = dev.compute.ok_or_else(|| missing("compute spec"))?;
    let (f, f_exp, b, c0) = if dev.kind == DeviceKind::Cpu {
        let (f, b) = dev
            .cpu_rates(threads)
            .ok_or_else(|| missing("flops or bandwidth"))?;
        (f, f, b, c.c0_eager_s)
    } else {
        let b = dev
            .bandwidth_bytes_per_s
            .ok_or_else(|| missing("bandwidth"))?
            .value;
        let c0 = if graphs { c.c0_graph_s } else { c.c0_eager_s };
        (c.flops_per_s, c.expert_flops(), b, c0)
    };
    let ok = |x: f64| x > 0.0 && x.is_finite();
    if !(ok(f) && ok(f_exp) && ok(b) && c0 >= 0.0 && c0.is_finite()) {
        return Err(PredictError::Invalid(format!(
            "device {} has non-positive rates",
            dev.id
        )));
    }
    Ok(Rates { f, f_exp, b, c0 })
}

#[derive(Clone, Copy)]
struct Hop {
    alpha: f64,
    beta: f64,
}

/// The first link between two parts' devices, with beta capped by its lane group.
fn hop(hw: &HardwareGraph, f: &RunFeatures) -> Option<Hop> {
    for (i, a) in f.parts.iter().enumerate() {
        for b in &f.parts[i + 1..] {
            if let Some(l) = hw.link_between(&a.device, &b.device) {
                let mut beta = l.beta_bytes_per_s.value;
                if let Some(g) = l.lane_group.as_deref().and_then(|g| hw.lane_group(g)) {
                    beta = beta.min(g.capacity_bytes_per_s);
                }
                return Some(Hop {
                    alpha: l.alpha_s.value,
                    beta,
                });
            }
        }
    }
    None
}

fn add(v: &mut Vec<(String, f64)>, label: String, s: f64) {
    match v.iter_mut().find(|e| e.0 == label) {
        Some(e) => e.1 += s,
        None => v.push((label, s)),
    }
}

/// Decode seconds per token of one part at average context `ctx`.
fn decode_part_s(p: &DevicePart, r: &Rates, ctx: f64) -> f64 {
    r.c0 * p.ops_per_token
        + ((2.0 * p.params_per_token + p.attn_flops_per_ctx_token * ctx) / r.f)
            .max((p.weight_bytes_per_token + p.kv_bytes_per_ctx_token * ctx) / r.b)
}

/// The disk of a tier and its rate at `depth`.
fn disk_rate(hw: &HardwareGraph, t: &TierFeatures, depth: u32) -> Result<f64, PredictError> {
    let rate = hw
        .device(&t.disk)
        .filter(|d| d.kind == DeviceKind::Disk)
        .and_then(|d| d.disk_rate(depth))
        .filter(|r| *r > 0.0 && r.is_finite());
    rate.ok_or_else(|| PredictError::MissingSpec {
        device: t.disk.clone(),
        what: "disk rate",
    })
}

/// Seconds per decode token that the disk adds to `t0` (the sum of the terms without it).
fn decode_read_s(
    hw: &HardwareGraph,
    f: &RunFeatures,
    t: &TierFeatures,
    devs: &[(&Device, Rates)],
    part_s: &[f64],
    ctx: f64,
    t0: f64,
) -> Result<f64, PredictError> {
    let (sync, prefetch) = (
        t.decode_sync_bytes_per_token,
        t.decode_prefetch_bytes_per_token,
    );
    let b = disk_rate(hw, t, t.depth)?;
    let cpu_s = |threads: u32| -> Result<f64, PredictError> {
        let mut sum = 0.0;
        for ((p, (d, _)), s) in f.parts.iter().zip(devs).zip(part_s) {
            if d.kind != DeviceKind::Cpu {
                continue;
            }
            sum += if threads == f.threads {
                *s
            } else {
                decode_part_s(p, &rates(d, threads, f.graphs)?, ctx)
            };
        }
        Ok(sum)
    };
    let c_exp = cpu_s(f.threads)?;
    let c_rest = t0 - c_exp;
    let (t_s, t_p) = (sync / b, prefetch / b);
    let t_tok = match t.path {
        ReadPath::Reader => {
            let copy = devs
                .iter()
                .find(|(d, _)| d.kind == DeviceKind::Cpu)
                .map(|(d, _)| *d)
                .or_else(|| hw.devices.iter().find(|d| d.kind == DeviceKind::Cpu))
                .and_then(|d| d.cpu_rates(1))
                .map(|r| r.1)
                .ok_or_else(|| PredictError::MissingSpec {
                    device: "cpu".into(),
                    what: "copy rate",
                })?;
            let b_eff = 1.0 / (1.0 / b + 1.0 / copy);
            c_rest + c_exp + (sync + prefetch) / b_eff
        }
        ReadPath::Service => (c_rest + c_exp + t_s).max(t_s + t_p),
        ReadPath::ServiceOverlap => {
            let c_exp = if sync > 0.0 {
                cpu_s(f.threads.saturating_sub(1).max(1))?
            } else {
                c_exp
            };
            (c_rest + c_exp.max(t_s)).max(t_s + t_p)
        }
    };
    Ok(t_tok - t0)
}

/// See the module doc for the formulas.
pub fn predict(
    hw: &HardwareGraph,
    f: &RunFeatures,
    prompt: u32,
    gen_tokens: u32,
) -> Result<Prediction, PredictError> {
    if let Some(r) = &f.ranks {
        return predict_ranks(hw, f, r, prompt, gen_tokens);
    }
    if f.parts.is_empty() || prompt == 0 || f.ubatch == 0 || f.threads == 0 {
        return Err(PredictError::Invalid(
            "need parts, a prompt, ubatch and threads".into(),
        ));
    }
    let mut devs = Vec::with_capacity(f.parts.len());
    for p in &f.parts {
        let d = hw
            .device(&p.device)
            .ok_or_else(|| PredictError::UnknownDevice(p.device.clone()))?;
        devs.push((d, rates(d, f.threads, f.graphs)?));
    }
    let link = hop(hw, f);
    let need_hop = || {
        link.ok_or_else(|| PredictError::NoLink {
            a: f.parts[0].device.clone(),
            b: f.parts.get(1).map_or(String::new(), |p| p.device.clone()),
        })
    };
    // A plan on one device has no link to cross, whatever `switches_per_token` says.
    let has_hops = f.switches_per_token > 0.0 && f.parts.len() > 1;
    let gpu = devs
        .iter()
        .position(|(d, _)| d.kind == DeviceKind::Gpu)
        .filter(|_| f.op_offload);

    // Decode.
    let ctx = f64::from(gen_tokens) / 2.0;
    let mut tg: Vec<(String, f64)> = Vec::new();
    let mut part_s = Vec::with_capacity(f.parts.len());
    for (p, (d, r)) in f.parts.iter().zip(&devs) {
        let t = decode_part_s(p, r, ctx);
        part_s.push(t);
        add(&mut tg, format!("compute:{}", d.id), t);
    }
    if has_hops {
        let h = need_hop()?;
        let t = f.switches_per_token * (h.alpha + f.activation_bytes_per_token / h.beta);
        add(&mut tg, "switches".into(), t);
    }
    let mut t_tok: f64 = tg.iter().map(|e| e.1).sum();
    // The tier term, only when bytes move: otherwise the numbers above stand untouched.
    let tier = f.tier.as_ref();
    let moves =
        |t: &TierFeatures| t.decode_sync_bytes_per_token + t.decode_prefetch_bytes_per_token > 0.0;
    if let Some(t) = tier.filter(|t| moves(t)) {
        let read = decode_read_s(hw, f, t, &devs, &part_s, ctx, t_tok)?;
        add(&mut tg, format!("read:{}", t.disk), read);
        t_tok = tg.iter().map(|e| e.1).sum();
    }

    // Prefill.
    let mut pp: Vec<(String, f64)> = Vec::new();
    let mut done = 0u32;
    while done < prompt {
        let m_tok = f.ubatch.min(prompt - done);
        let m = f64::from(m_tok);
        let ctx = f64::from(done) + m / 2.0;
        let offload = gpu.is_some() && m_tok >= 32;
        for (p, (d, r)) in f.parts.iter().zip(&devs) {
            let flops = 2.0 * p.params_per_token * m;
            let attn = p.attn_flops_per_ctx_token * m * ctx;
            let kv = p.kv_bytes_per_ctx_token * ctx;
            if let (true, Some(g), DeviceKind::Cpu) = (offload, gpu, d.kind) {
                let rg = devs[g].1;
                if p.resident_matmul_bytes > 0.0 {
                    let h = need_hop()?;
                    add(
                        &mut pp,
                        format!("offload_copy:{}", d.id),
                        h.alpha + p.resident_matmul_bytes / h.beta,
                    );
                }
                add(
                    &mut pp,
                    format!("offload_compute:{}", d.id),
                    (flops / rg.f).max(p.weight_bytes_per_token / rg.b),
                );
                add(
                    &mut pp,
                    format!("compute:{}", d.id),
                    r.c0 * p.ops_per_token + (attn / r.f).max(kv / r.b),
                );
            } else {
                add(
                    &mut pp,
                    format!("compute:{}", d.id),
                    r.c0 * p.ops_per_token
                        + ((flops + attn) / r.f).max((p.weight_bytes_per_token + kv) / r.b),
                );
            }
        }
        if has_hops {
            let h = need_hop()?;
            add(
                &mut pp,
                "switches".into(),
                f.switches_per_token * (h.alpha + m * f.activation_bytes_per_token / h.beta),
            );
        }
        if let Some(t) = tier.filter(|t| t.prefill_bytes_per_token > 0.0) {
            let rate = disk_rate(hw, t, t.sync_depth)?;
            add(
                &mut pp,
                format!("read:{}", t.disk),
                m * t.prefill_bytes_per_token / rate,
            );
        }
        done += m_tok;
    }
    let t_pp: f64 = pp.iter().map(|e| e.1).sum();

    let binding = tg
        .iter()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|e| e.0.clone())
        .unwrap_or_default();
    Ok(Prediction {
        pp_tok_s: f64::from(prompt) / t_pp,
        tg_tok_s: 1.0 / t_tok,
        request_s: t_pp + f64::from(gen_tokens) * t_tok,
        tg_breakdown: tg,
        pp_breakdown: pp,
        binding,
        decode_clocks: Vec::new(),
    })
}

/// Time of one rank's part for `m` tokens at mean context `ctx` (decode is `m = 1`): `c0` per
/// op plus one roofline per op class, since a class bound by bytes does not hide behind one
/// bound by compute (ADR 0070, MODEL_VERSION 2):
///
/// ```text
/// t = c0 ops + max(2 p m / F, W / B) + max(2 p_exp m / F_exp, W_exp / B) + max(a m ctx / F, kv ctx / B)
///     + S m / B
/// ```
///
/// `S` (the routed slots' activation bytes, ADR 0080) counts only when `slots` (the graph's
/// `RangeCosts::expert_slot_traffic`); its kernels run apart from the matmuls, so it adds.
///
/// `k` (ADR 0082, decode of a GPU in the low clock state) multiplies the clock-bound terms: `c0 ops`,
/// the routed-expert roofline and `S m / B`. The dense and KV rooflines stay. `k = 1` changes no bit.
fn part_time_s(p: &DevicePart, r: &Rates, m: f64, ctx: f64, slots: bool, k: f64) -> f64 {
    let slot = if slots {
        p.expert_slot_bytes_per_token * m / r.b
    } else {
        0.0
    };
    r.c0 * p.ops_per_token * k
        + (2.0 * p.params_per_token * m / r.f).max(p.weight_bytes_per_token / r.b)
        + (2.0 * p.expert_params_per_token * m / r.f_exp).max(p.expert_bytes_per_token / r.b) * k
        + (p.attn_flops_per_ctx_token * m * ctx / r.f).max(p.kv_bytes_per_ctx_token * ctx / r.b)
        + slot * k
}

/// `(alpha, beta)` of the link between two devices, beta capped by its lane group.
fn link_between(hw: &HardwareGraph, a: &str, b: &str) -> Result<Hop, PredictError> {
    let l = hw.link_between(a, b).ok_or_else(|| PredictError::NoLink {
        a: a.to_string(),
        b: b.to_string(),
    })?;
    let mut beta = l.beta_bytes_per_s.value;
    if let Some(g) = l.lane_group.as_deref().and_then(|g| hw.lane_group(g)) {
        beta = beta.min(g.capacity_bytes_per_s);
    }
    Ok(Hop {
        alpha: l.alpha_s.value,
        beta,
    })
}

/// All-reduce time of `bytes` over `group` as measured: the probe curve, else the ring prior.
fn allreduce_curve_s(
    hw: &HardwareGraph,
    group: &[String],
    bytes: f64,
) -> Result<f64, PredictError> {
    if let Some(t) = hw.collective_time(CollectiveOp::AllReduce, group, bytes) {
        return Ok(t);
    }
    let (mut alpha, mut beta) = (0.0f64, f64::INFINITY);
    for d in group.iter().skip(1) {
        let h = link_between(hw, &group[0], d)?;
        alpha = alpha.max(h.alpha);
        beta = beta.min(h.beta);
    }
    Ok(allreduce_time(group.len() as u32, alpha, beta, bytes))
}

/// The all-reduce curve as cost model 4 reads it inside a step (ADR 0080 History): a head point
/// not below its successor is the probe's floor (its first timed op), not the op's bytes, so it
/// takes the next segment's time extended back to its size. Rising heads are unchanged.
fn allreduce_step_curve_s(
    hw: &HardwareGraph,
    group: &[String],
    bytes: f64,
) -> Result<f64, PredictError> {
    let curve = hw.collectives.iter().find(|c| {
        c.op == CollectiveOp::AllReduce
            && c.ranks.len() == group.len()
            && group.iter().all(|r| c.ranks.contains(r))
    });
    let Some(curve) = curve else {
        return allreduce_curve_s(hw, group, bytes);
    };
    let mut pts = curve.points.clone();
    pts.sort_by(|a, b| a.0.total_cmp(&b.0));
    if pts.len() > 2 && pts[0].1 >= pts[1].1 {
        let ((x1, y1), (x2, y2)) = (pts[1], pts[2]);
        let slope = ((y2 - y1) / (x2 - x1)).max(0.0);
        pts[0].1 = (y1 - slope * (x1 - pts[0].0)).max(0.0);
    }
    curve_time(&pts, bytes).map_or_else(|| allreduce_curve_s(hw, group, bytes), Ok)
}

/// Seconds of a collective for a step of `m` tokens and its breakdown label: the measured curve
/// when the graph has one, else the link prior (ring all-reduce, `alpha + n / beta` copy).
/// An all-reduce inside a step keeps the curve's byte part, the BF16 switch included, and pays
/// its own latency instead of the probe's zero-byte time `t(0)`: with the fitted
/// `RangeCosts::allreduce_latency_s` = `a` and `allreduce_step_s` = `b` (0 when not fitted,
/// ADR 0078) it costs `max(t(n) - t(0), 0) + a + 2 (p - 1) b` over `p` ranks, on a curve
/// without a floor head on graphs of cost model 4 (`allreduce_step_curve_s`); graphs fitted
/// before that pay the share `s` of `t(0)`, `t(n) - (1 - s) t(0)` (ADR 0070). The one helper the
/// planner's search shares.
pub fn coll_time(hw: &HardwareGraph, c: &Coll, m: f64) -> Result<(String, f64), PredictError> {
    match c {
        Coll::AllReduce {
            group,
            bytes_per_token,
        } => {
            let label = format!("allreduce:{}", group.join("+"));
            let t = allreduce_curve_s(hw, group, bytes_per_token * m)?;
            let costs = hw.range_costs_or_prior();
            if let Some(a) = costs.allreduce_latency_s {
                let (t, t0) = if costs.expert_slot_traffic {
                    (
                        allreduce_step_curve_s(hw, group, bytes_per_token * m)?,
                        allreduce_step_curve_s(hw, group, 0.0)?,
                    )
                } else {
                    (t, allreduce_curve_s(hw, group, 0.0)?)
                };
                let steps = 2.0 * (group.len().max(1) - 1) as f64;
                let b = costs.allreduce_step_s.unwrap_or(0.0);
                return Ok((label, (t - t0).max(0.0) + a + steps * b));
            }
            let s = costs.allreduce_latency_share;
            if s == 1.0 {
                return Ok((label, t));
            }
            let t0 = allreduce_curve_s(hw, group, 0.0)?;
            Ok((label, (t - (1.0 - s) * t0).max(0.0)))
        }
        Coll::Copy {
            pairs,
            bytes_per_token,
            per_step,
        } => {
            let bytes = bytes_per_token * if *per_step { 1.0 } else { m };
            let label = pairs
                .first()
                .map_or_else(|| "copy".to_string(), |p| format!("copy:{}-{}", p.0, p.1));
            let mut t = 0.0f64;
            for (a, b) in pairs {
                let pair = [a.clone(), b.clone()];
                let one = match hw.collective_time(CollectiveOp::Copy, &pair, bytes) {
                    Some(t) => t,
                    None => {
                        let h = link_between(hw, a, b)?;
                        transfer_time(h.alpha, h.beta, bytes)
                    }
                };
                t = t.max(one);
            }
            Ok((label, t))
        }
    }
}

/// The pipeline stages of a ranks model: indices of its phases without output rows, cut after
/// each stage hop (a per-token copy).
fn pipeline_stages(phases: &[Phase]) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new()];
    for (i, ph) in phases.iter().enumerate().filter(|(_, p)| !p.out_rows) {
        out.last_mut().expect("one stage").push(i);
        if matches!(
            ph.after,
            Some(Coll::Copy {
                per_step: false,
                ..
            })
        ) {
            out.push(Vec::new());
        }
    }
    out.retain(|s| !s.is_empty());
    out
}

/// Step time with ranks (ADR 0070): `T(M) = sum_p repeat_p (max_r t(r, M) + ranges_p c_range +
/// t_coll(after_p, M)) + c_step`, summed over the decode tokens and the prefill ubatches as in
/// [`predict`]. `t(r, M)` is [`part_time_s`] (a roofline per op class); a phase with `out_rows`
/// runs [`OUTPUTS_PER_STEP`] rows in a prefill step, and a phase with `prefill_parts` uses them
/// there (every local expert is read). Breakdown labels: `compute:<dev>` (the slowest rank of each
/// phase), `ranges`, `allreduce:<group>`, `copy:<a>-<b>` and `step`.
fn predict_ranks(
    hw: &HardwareGraph,
    f: &RunFeatures,
    r: &RankFeatures,
    prompt: u32,
    gen_tokens: u32,
) -> Result<Prediction, PredictError> {
    if r.phases.is_empty() || prompt == 0 || f.ubatch == 0 || f.threads == 0 {
        return Err(PredictError::Invalid(
            "need phases, a prompt, ubatch and threads".into(),
        ));
    }
    let mut rates_of: Vec<(String, Rates)> = Vec::new();
    for ph in &r.phases {
        if ph.parts.is_empty() || ph.prefill_parts.as_ref().is_some_and(Vec::is_empty) {
            return Err(PredictError::Invalid("a phase has no parts".into()));
        }
        for p in ph.parts.iter().chain(ph.prefill_parts.iter().flatten()) {
            if rates_of.iter().any(|e| e.0 == p.device) {
                continue;
            }
            let d = hw
                .device(&p.device)
                .ok_or_else(|| PredictError::UnknownDevice(p.device.clone()))?;
            rates_of.push((p.device.clone(), rates(d, f.threads, f.graphs)?));
        }
    }
    let costs = hw.range_costs_or_prior();
    let c_range = if f.graphs {
        costs.c_range_graph_s
    } else {
        costs.c_range_eager_s
    };
    // One phase for `m` tokens at mean context `ctx`, as breakdown terms; `with_after` charges
    // the collective that ends it.
    // `ks`: the decode factor of each GPU in the low clock state (ADR 0082); empty elsewhere.
    let phase = |ph: &Phase,
                 m: f64,
                 ctx: f64,
                 prefill: bool,
                 with_after: bool,
                 ks: &[(String, f64)],
                 out: &mut Vec<(String, f64)>|
     -> Result<(), PredictError> {
        {
            // An output-row phase prices a prefill step at its output rows, with the decode parts.
            let (m, parts) = match (prefill, ph.out_rows) {
                (true, true) => (OUTPUTS_PER_STEP.min(m), &ph.parts),
                (true, false) => (m, ph.prefill_parts.as_ref().unwrap_or(&ph.parts)),
                (false, _) => (m, &ph.parts),
            };
            let rep = f64::from(ph.repeat);
            let mut slowest = (String::new(), 0.0f64);
            for p in parts {
                let rt = rates_of
                    .iter()
                    .find(|e| e.0 == p.device)
                    .map(|e| e.1)
                    .ok_or_else(|| PredictError::UnknownDevice(p.device.clone()))?;
                let k = ks.iter().find(|e| e.0 == p.device).map_or(1.0, |e| e.1);
                let t = part_time_s(p, &rt, m, ctx, costs.expert_slot_traffic, k);
                if t > slowest.1 || slowest.0.is_empty() {
                    slowest = (p.device.clone(), t);
                }
            }
            add(out, format!("compute:{}", slowest.0), rep * slowest.1);
            if ph.ranges_per_rank > 0 {
                add(
                    out,
                    "ranges".into(),
                    rep * f64::from(ph.ranges_per_rank) * c_range,
                );
            }
            if let Some(c) = ph.after.as_ref().filter(|_| with_after) {
                let (label, t) = coll_time(hw, c, m)?;
                add(out, label, rep * t);
            }
        }
        Ok(())
    };
    // One step of `m` tokens at mean context `ctx`, as breakdown terms.
    let step = |m: f64,
                ctx: f64,
                prefill: bool,
                ks: &[(String, f64)],
                out: &mut Vec<(String, f64)>|
     -> Result<(), PredictError> {
        for ph in &r.phases {
            phase(ph, m, ctx, prefill, true, ks, out)?;
        }
        if r.step_fixed {
            add(out, "step".into(), costs.c_step_s);
        }
        Ok(())
    };

    let dec_ctx = f64::from(gen_tokens) / 2.0;
    let mut tg: Vec<(String, f64)> = Vec::new();
    step(1.0, dec_ctx, false, &[], &mut tg)?;
    let mut t_tok: f64 = tg.iter().map(|e| e.1).sum();
    // Cost model 5 (ADR 0082): a GPU busy under the duty threshold of the unscaled step runs at the
    // low clock; the step is recomputed once with the factor on its clock-bound terms.
    let mut clocks: Vec<DecodeClock> = Vec::new();
    if let (Some(dc), Some(model)) = (&hw.decode_clocks, r.model.as_deref()) {
        let k_model = dc.factor(model);
        let mut busy: Vec<(String, f64)> = Vec::new();
        for ph in &r.phases {
            let rep = f64::from(ph.repeat);
            let coll = match &ph.after {
                Some(c @ Coll::AllReduce { group, .. }) => Some((group, coll_time(hw, c, 1.0)?.1)),
                _ => None,
            };
            for p in &ph.parts {
                let rt = rates_of
                    .iter()
                    .find(|e| e.0 == p.device)
                    .map(|e| e.1)
                    .ok_or_else(|| PredictError::UnknownDevice(p.device.clone()))?;
                let mut t = part_time_s(p, &rt, 1.0, dec_ctx, costs.expert_slot_traffic, 1.0);
                if let Some((group, tc)) = &coll
                    && group.contains(&p.device)
                {
                    t += tc;
                }
                add(&mut busy, p.device.clone(), rep * t);
            }
        }
        let mut ks: Vec<(String, f64)> = Vec::new();
        for (device, b) in busy {
            let duty = b / t_tok;
            let low = duty < dc.duty_threshold;
            let factor = if low { k_model } else { 1.0 };
            if factor > 1.0 {
                ks.push((device.clone(), factor));
            }
            let mhz = match dc.busy_mhz.get(model) {
                Some(&busy_mhz) if !low => busy_mhz,
                _ => dc.low_mhz,
            };
            clocks.push(DecodeClock {
                device,
                duty,
                factor,
                mhz,
            });
        }
        if !ks.is_empty() {
            tg.clear();
            step(1.0, dec_ctx, false, &ks, &mut tg)?;
            t_tok = tg.iter().map(|e| e.1).sum();
        }
    }

    let mut pp: Vec<(String, f64)> = Vec::new();
    let mut done = 0u32;
    let stages = pipeline_stages(&r.phases);
    let micro = r.prefill_micro_batches.max(1);
    if micro > 1 && stages.len() > 1 {
        // Pipelined prefill (ADR 0077): a step of `micro x ubatch` tokens cut into chunks as
        // `plan::micro_batch_sizes`; stage s computes chunk c after chunk c - 1 and after the
        // chunk's hop arrived, hops serialized per boundary (GPipe, as `sim.rs`). The output-row
        // phases and `c_step` follow once per step.
        let n_st = stages.len();
        while done < prompt {
            let n = (f.ubatch * micro).min(prompt - done);
            let (mut free, mut link) = (vec![0.0f64; n_st], vec![0.0f64; n_st]);
            let mut at = done;
            for c in tetraeder_plan::micro_batch_sizes(n as usize, micro) {
                let m = c as f64;
                let ctx = f64::from(at) + m / 2.0;
                let mut arrive = 0.0f64;
                for (s, st) in stages.iter().enumerate() {
                    let mut terms = Vec::new();
                    let last = st[st.len() - 1];
                    for &i in st {
                        let hop = s + 1 < n_st && i == last;
                        phase(&r.phases[i], m, ctx, true, !hop, &[], &mut terms)?;
                    }
                    let t: f64 = terms.iter().map(|e| e.1).sum();
                    free[s] = arrive.max(free[s]) + t;
                    if s + 1 < n_st
                        && let Some(h) = &r.phases[last].after
                    {
                        let (_, th) = coll_time(hw, h, m)?;
                        link[s] = free[s].max(link[s]) + th;
                        arrive = link[s];
                    }
                }
                at += c as u32;
            }
            add(&mut pp, "pipeline".into(), free[n_st - 1]);
            let m = f64::from(n);
            for ph in r.phases.iter().filter(|p| p.out_rows) {
                phase(ph, m, f64::from(done) + m / 2.0, true, true, &[], &mut pp)?;
            }
            if r.step_fixed {
                add(&mut pp, "step".into(), costs.c_step_s);
            }
            done += n;
        }
    }
    while done < prompt {
        let m_tok = f.ubatch.min(prompt - done);
        let m = f64::from(m_tok);
        step(m, f64::from(done) + m / 2.0, true, &[], &mut pp)?;
        done += m_tok;
    }
    let t_pp: f64 = pp.iter().map(|e| e.1).sum();
    let binding = tg
        .iter()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|e| e.0.clone())
        .unwrap_or_default();
    Ok(Prediction {
        pp_tok_s: f64::from(prompt) / t_pp,
        tg_tok_s: 1.0 / t_tok,
        request_s: t_pp + f64::from(gen_tokens) * t_tok,
        tg_breakdown: tg,
        pp_breakdown: pp,
        binding,
        decode_clocks: clocks,
    })
}

#[cfg(test)]
pub(crate) mod fixture {
    use super::*;
    use tetraeder_profiler::{
        ComputeSpec, LaneGroup, Link, LinkKind, Measurement, ThreadPoint, Tier,
    };

    /// A GPU (id "gpu0"), a CPU ("cpu0") and a PCIe link with a lane group.
    pub fn hw() -> HardwareGraph {
        let mut gpu = Device::new("gpu0", DeviceKind::Gpu, "g", Tier::Vram, 24 << 30);
        gpu.compute = Some(ComputeSpec {
            flops_per_s: 60e12,
            c0_graph_s: 2e-6,
            c0_eager_s: 9e-6,
            expert_flops_per_s: None,
        });
        gpu.bandwidth_bytes_per_s = Some(Measurement::prior(800e9));
        let mut cpu = Device::new("cpu0", DeviceKind::Cpu, "c", Tier::Ram, 64 << 30);
        cpu.threads = Some(16);
        cpu.compute = Some(ComputeSpec {
            flops_per_s: 1.2e12,
            c0_graph_s: 3e-6,
            c0_eager_s: 3e-6,
            expert_flops_per_s: None,
        });
        cpu.bandwidth_bytes_per_s = Some(Measurement::prior(60e9));
        cpu.thread_curve = vec![
            ThreadPoint {
                threads: 4,
                flops_per_s: 0.4e12,
                bytes_per_s: 35e9,
            },
            ThreadPoint {
                threads: 8,
                flops_per_s: 0.7e12,
                bytes_per_s: 52e9,
            },
            ThreadPoint {
                threads: 16,
                flops_per_s: 1.2e12,
                bytes_per_s: 60e9,
            },
        ];
        HardwareGraph {
            devices: vec![gpu, cpu],
            links: vec![Link {
                id: "pcie".into(),
                a: "gpu0".into(),
                b: "cpu0".into(),
                kind: LinkKind::Pcie,
                alpha_s: Measurement::prior(12e-6),
                beta_bytes_per_s: Measurement::prior(24e9),
                lane_group: Some("lanes".into()),
            }],
            lane_groups: vec![LaneGroup {
                id: "lanes".into(),
                capacity_bytes_per_s: 20e9,
            }],
            notes: Vec::new(),
            ..Default::default()
        }
    }

    pub fn part(dev: &str, gb: f64, ops: f64) -> DevicePart {
        DevicePart {
            device: dev.into(),
            weight_bytes_per_token: gb * 1e9,
            params_per_token: gb * 1e9 / 0.5,
            resident_matmul_bytes: gb * 1e9,
            ops_per_token: ops,
            kv_bytes_per_ctx_token: 40e3 * gb / 10.0,
            attn_flops_per_ctx_token: 1e5 * gb / 10.0,
            expert_bytes_per_token: 0.0,
            expert_params_per_token: 0.0,
            expert_slot_bytes_per_token: 0.0,
        }
    }

    pub fn features(gpu_gb: f64, cpu_gb: f64) -> RunFeatures {
        let mut parts = Vec::new();
        if gpu_gb > 0.0 {
            parts.push(part("gpu0", gpu_gb, 12.0 * gpu_gb * 10.0));
        }
        if cpu_gb > 0.0 {
            parts.push(part("cpu0", cpu_gb, 12.0 * cpu_gb * 10.0));
        }
        RunFeatures {
            switches_per_token: if parts.len() > 1 { 2.0 } else { 0.0 },
            activation_bytes_per_token: 8192.0 * 4.0,
            parts,
            threads: 8,
            ubatch: 512,
            op_offload: true,
            graphs: true,
            tier: None,
            ranks: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fixture::*;
    use super::*;

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() <= 1e-9 * b.abs().max(1e-12), "{a} vs {b}");
    }

    #[test]
    fn gpu_only_decode_is_bandwidth_bound_plus_c0() {
        let hw = hw();
        let f = features(10.0, 0.0);
        let p = predict(&hw, &f, 512, 128).unwrap();
        let part = &f.parts[0];
        let ctx = 64.0;
        let bytes = part.weight_bytes_per_token + part.kv_bytes_per_ctx_token * ctx;
        close(
            p.tg_tok_s,
            1.0 / (2e-6 * part.ops_per_token + bytes / 800e9),
        );
        assert_eq!(p.binding, "compute:gpu0");
        let eager = predict(&hw, &RunFeatures { graphs: false, ..f }, 512, 128).unwrap();
        assert!(eager.tg_tok_s < p.tg_tok_s);
    }

    #[test]
    fn switches_use_the_lane_capped_link() {
        let hw = hw();
        let f = features(6.0, 4.0);
        let p = predict(&hw, &f, 512, 128).unwrap();
        let sw = p.tg_breakdown.iter().find(|e| e.0 == "switches").unwrap().1;
        // beta is min(24e9, 20e9)
        close(sw, 2.0 * (12e-6 + 32768.0 / 20e9));
        let t: f64 = p.tg_breakdown.iter().map(|e| e.1).sum();
        close(p.tg_tok_s, 1.0 / t);
    }

    #[test]
    fn cpu_threads_follow_the_curve() {
        let hw = hw();
        let at = |t| {
            let f = RunFeatures {
                threads: t,
                ..features(0.0, 4.0)
            };
            predict(&hw, &f, 512, 128).unwrap().tg_tok_s
        };
        assert!(at(4) < at(8) && at(8) < at(16));
        close(at(32), at(16));
    }

    #[test]
    fn op_offload_replaces_cpu_prefill_with_a_copy() {
        let hw = hw();
        let on = features(6.0, 4.0);
        let off = RunFeatures {
            op_offload: false,
            ..on.clone()
        };
        let a = predict(&hw, &on, 2048, 64).unwrap();
        let b = predict(&hw, &off, 2048, 64).unwrap();
        assert!(a.pp_breakdown.iter().any(|e| e.0 == "offload_copy:cpu0"));
        assert!(a.pp_breakdown.iter().any(|e| e.0 == "switches"));
        assert!(b.pp_breakdown.iter().any(|e| e.0 == "switches"));
        assert!(
            a.pp_tok_s > b.pp_tok_s,
            "4 GB copy beats 4 GB of CPU matmul"
        );
        close(a.tg_tok_s, b.tg_tok_s);
        // 4 ubatches of 512, each copying 4e9 B at 20e9 B/s plus alpha
        let copy = a.pp_breakdown.iter().find(|e| e.0 == "offload_copy:cpu0");
        close(copy.unwrap().1, 4.0 * (12e-6 + 4e9 / 20e9));
    }

    #[test]
    fn small_ubatches_do_not_offload() {
        let hw = hw();
        let f = RunFeatures {
            ubatch: 16,
            ..features(6.0, 4.0)
        };
        let p = predict(&hw, &f, 64, 8).unwrap();
        assert!(!p.pp_breakdown.iter().any(|e| e.0.starts_with("offload")));
    }

    #[test]
    fn request_time_and_errors() {
        let hw = hw();
        let f = features(6.0, 4.0);
        let p = predict(&hw, &f, 1000, 200).unwrap();
        close(p.request_s, 1000.0 / p.pp_tok_s + 200.0 / p.tg_tok_s);
        let mut bad = f.clone();
        bad.parts[0].device = "nope".into();
        assert!(matches!(
            predict(&hw, &bad, 10, 10),
            Err(PredictError::UnknownDevice(_))
        ));
        let mut nolink = hw.clone();
        nolink.links.clear();
        assert!(matches!(
            predict(&nolink, &f, 10, 10),
            Err(PredictError::NoLink { .. })
        ));
        assert!(predict(&hw, &f, 0, 10).is_err());
    }

    #[test]
    fn a_single_part_plan_needs_no_link() {
        let mut f = features(6.0, 0.0);
        assert_eq!(f.parts.len(), 1);
        f.switches_per_token = 1.0;
        let mut nolink = hw();
        nolink.links.clear();
        for g in [hw(), nolink] {
            let p = predict(&g, &f, 100, 20).unwrap();
            assert!(!p.tg_breakdown.iter().any(|e| e.0 == "switches"));
            assert!(!p.pp_breakdown.iter().any(|e| e.0 == "switches"));
        }
    }

    #[test]
    fn features_roundtrip_json() {
        let f = features(6.0, 4.0);
        let s = serde_json::to_string(&f).unwrap();
        assert_eq!(serde_json::from_str::<RunFeatures>(&s).unwrap(), f);
        // an old feature file without `tier` still parses, and None is not written
        assert!(!s.contains("tier"));
    }

    // ---- ADR 0056: the tier term

    use tetraeder_profiler::{Measurement, QueuePoint, Tier};

    fn hw_disk() -> HardwareGraph {
        let mut h = hw();
        let mut d = Device::new("nvme0", DeviceKind::Disk, "n", Tier::Nvme, 1 << 40);
        d.bandwidth_bytes_per_s = Some(Measurement::prior(5e9));
        d.queue_curve = vec![
            QueuePoint {
                depth: 1,
                bytes_per_s: 1e9,
            },
            QueuePoint {
                depth: 4,
                bytes_per_s: 3e9,
            },
            QueuePoint {
                depth: 16,
                bytes_per_s: 5e9,
            },
        ];
        h.devices.push(d);
        h
    }

    fn tier(path: ReadPath, sync: f64, prefetch: f64, prefill: f64) -> TierFeatures {
        TierFeatures {
            disk: "nvme0".into(),
            path,
            depth: 16,
            sync_depth: 4,
            decode_sync_bytes_per_token: sync,
            decode_prefetch_bytes_per_token: prefetch,
            prefill_bytes_per_token: prefill,
        }
    }

    fn with_tier(t: TierFeatures) -> RunFeatures {
        RunFeatures {
            tier: Some(t),
            ..features(6.0, 4.0)
        }
    }

    fn term(v: &[(String, f64)], label: &str) -> f64 {
        v.iter().find(|e| e.0 == label).map_or(0.0, |e| e.1)
    }

    #[test]
    fn no_tier_and_a_zero_byte_tier_are_bit_identical() {
        let hw = hw_disk();
        for (g, c) in [(10.0, 0.0), (6.0, 4.0), (0.0, 4.0)] {
            let none = features(g, c);
            let base = predict(&hw, &none, 2048, 128).unwrap();
            for path in [
                ReadPath::Reader,
                ReadPath::Service,
                ReadPath::ServiceOverlap,
            ] {
                let zero = RunFeatures {
                    tier: Some(tier(path, 0.0, 0.0, 0.0)),
                    ..none.clone()
                };
                assert_eq!(predict(&hw, &zero, 2048, 128).unwrap(), base);
            }
        }
    }

    #[test]
    fn reader_adds_the_serial_sum_of_read_and_copy() {
        let hw = hw_disk();
        let base = predict(&hw, &features(6.0, 4.0), 512, 128).unwrap();
        let t0: f64 = base.tg_breakdown.iter().map(|e| e.1).sum();
        let (sync, pf) = (30e6, 60e6);
        let p = predict(
            &hw,
            &with_tier(tier(ReadPath::Reader, sync, pf, 0.0)),
            512,
            128,
        )
        .unwrap();
        // depth 16 reads 5e9 B/s; the CPU copies at its one-thread rate (35e9 / 4 below the curve)
        let b_eff = 1.0 / (1.0 / 5e9 + 1.0 / 8.75e9);
        close(term(&p.tg_breakdown, "read:nvme0"), (sync + pf) / b_eff);
        close(p.tg_tok_s, 1.0 / (t0 + (sync + pf) / b_eff));
        assert_eq!(p.pp_tok_s, base.pp_tok_s);
    }

    /// One thread's copy rate does not shrink when the container gets more cores: the reader's
    /// read term must be the same on a CPU with 12 and with 61 threads and no thread curve
    /// (`docs/notes/stream-predict-diagnosis.md` section 1; measured 2.8 to 3.3 GB/s on 61).
    #[test]
    #[ignore = "fails until ADR 0092 (planner-costmodel-stream-terms)"]
    fn the_reader_copy_rate_does_not_depend_on_the_thread_quota() {
        let f = with_tier(tier(ReadPath::Reader, 30e6, 60e6, 0.0));
        let read = |threads: u32| {
            let mut hw = hw_disk();
            let cpu = hw.devices.iter_mut().find(|d| d.id == "cpu0").unwrap();
            cpu.thread_curve.clear();
            cpu.threads = Some(threads);
            let p = predict(&hw, &f, 512, 128).unwrap();
            term(&p.tg_breakdown, "read:nvme0")
        };
        close(read(61), read(12));
    }

    #[test]
    fn service_is_the_longer_of_the_chain_and_the_busy_disk() {
        let hw = hw_disk();
        let base = predict(&hw, &features(6.0, 4.0), 512, 128).unwrap();
        let t0: f64 = base.tg_breakdown.iter().map(|e| e.1).sum();
        // small reads: the chain (compute plus the synchronous reads) is longer
        let (sync, pf) = (10e6, 5e6);
        let p = predict(
            &hw,
            &with_tier(tier(ReadPath::Service, sync, pf, 0.0)),
            512,
            128,
        )
        .unwrap();
        assert!(t0 + sync / 5e9 > (sync + pf) / 5e9);
        close(p.tg_tok_s, 1.0 / (t0 + sync / 5e9));
        // large prefetch: the disk is busy longer than the chain
        let (sync, pf) = (10e6, 900e6);
        let p = predict(
            &hw,
            &with_tier(tier(ReadPath::Service, sync, pf, 0.0)),
            512,
            128,
        )
        .unwrap();
        assert!((sync + pf) / 5e9 > t0 + sync / 5e9);
        close(p.tg_tok_s, 1.0 / ((sync + pf) / 5e9));
        close(term(&p.tg_breakdown, "read:nvme0"), (sync + pf) / 5e9 - t0);
        assert_eq!(p.binding, "read:nvme0");
    }

    #[test]
    fn overlap_hides_the_sync_read_behind_the_expert_compute() {
        let hw = hw_disk();
        let f = features(6.0, 4.0);
        let base = predict(&hw, &f, 512, 128).unwrap();
        let t0: f64 = base.tg_breakdown.iter().map(|e| e.1).sum();
        let c_exp = term(&base.tg_breakdown, "compute:cpu0");
        // compute one thread short
        let short = RunFeatures {
            threads: f.threads - 1,
            ..f.clone()
        };
        let c_exp7 = term(
            &predict(&hw, &short, 512, 128).unwrap().tg_breakdown,
            "compute:cpu0",
        );
        assert!(c_exp7 > c_exp);
        let (sync, pf) = (20e6, 10e6);
        let t = tier(ReadPath::ServiceOverlap, sync, pf, 0.0);
        let p = predict(&hw, &with_tier(t), 512, 128).unwrap();
        let t_s = sync / 5e9;
        let want = (t0 - c_exp + c_exp7.max(t_s)).max((sync + pf) / 5e9);
        close(p.tg_tok_s, 1.0 / want);
        // no synchronous reads: compute keeps all its threads
        let t = tier(ReadPath::ServiceOverlap, 0.0, 10e6, 0.0);
        let p = predict(&hw, &with_tier(t), 512, 128).unwrap();
        close(p.tg_tok_s, 1.0 / t0.max(10e6 / 5e9));
    }

    #[test]
    fn prefill_misses_are_read_synchronously_per_ubatch() {
        let hw = hw_disk();
        let base = predict(&hw, &features(6.0, 4.0), 2048, 64).unwrap();
        let p = predict(
            &hw,
            &with_tier(tier(ReadPath::Service, 0.0, 0.0, 2e6)),
            2048,
            64,
        )
        .unwrap();
        // 4 ubatches of 512 tokens, each 512 * 2e6 B at the depth-4 rate of 3e9 B/s
        close(term(&p.pp_breakdown, "read:nvme0"), 4.0 * 512.0 * 2e6 / 3e9);
        let t0: f64 = base.pp_breakdown.iter().map(|e| e.1).sum();
        close(2048.0 / p.pp_tok_s, t0 + 2048.0 * 2e6 / 3e9);
        // decode has no bytes, so it is untouched
        assert_eq!(p.tg_tok_s, base.tg_tok_s);
        // a partial last ubatch reads its own tokens only
        let p = predict(
            &hw,
            &with_tier(tier(ReadPath::Service, 0.0, 0.0, 2e6)),
            1000,
            8,
        )
        .unwrap();
        close(term(&p.pp_breakdown, "read:nvme0"), 1000.0 * 2e6 / 3e9);
    }

    #[test]
    fn a_tier_without_a_disk_rate_is_an_error() {
        let f = with_tier(tier(ReadPath::Reader, 1e6, 0.0, 0.0));
        // the plain graph has no disk at all
        assert!(matches!(
            predict(&hw(), &f, 512, 8),
            Err(PredictError::MissingSpec { .. })
        ));
        let mut bare = hw_disk();
        let d = bare.devices.iter_mut().find(|d| d.id == "nvme0").unwrap();
        d.queue_curve.clear();
        d.bandwidth_bytes_per_s = None;
        assert!(matches!(
            predict(&bare, &f, 512, 8),
            Err(PredictError::MissingSpec { .. })
        ));
        // zero bytes need no disk
        let z = with_tier(tier(ReadPath::Reader, 0.0, 0.0, 0.0));
        assert!(predict(&hw(), &z, 512, 8).is_ok());
    }

    #[test]
    fn tier_features_roundtrip_json() {
        let f = with_tier(tier(ReadPath::ServiceOverlap, 1.0, 2.0, 3.0));
        let s = serde_json::to_string(&f).unwrap();
        assert!(s.contains("service_overlap"));
        assert_eq!(serde_json::from_str::<RunFeatures>(&s).unwrap(), f);
    }

    mod props {
        use super::*;
        use proptest::prelude::*;

        fn path(i: u8) -> ReadPath {
            [
                ReadPath::Reader,
                ReadPath::Service,
                ReadPath::ServiceOverlap,
            ][usize::from(i % 3)]
        }

        fn tg(hw: &HardwareGraph, t: TierFeatures) -> f64 {
            predict(hw, &with_tier(t), 512, 128).unwrap().tg_tok_s
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(64))]

            /// Fewer disk bytes never predict slower (decode and prefill).
            #[test]
            fn fewer_bytes_are_never_slower(
                p in 0u8..3,
                sync in 0.0f64..1e9, pf in 0.0f64..1e9, pre in 0.0f64..1e7,
                ks in 0.0f64..=1.0, kp in 0.0f64..=1.0, kq in 0.0f64..=1.0,
            ) {
                let hw = hw_disk();
                let more = tier(path(p), sync, pf, pre);
                let less = tier(path(p), sync * ks, pf * kp, pre * kq);
                prop_assert!(tg(&hw, less.clone()) >= tg(&hw, more.clone()) * (1.0 - 1e-12));
                let pp = |t: TierFeatures| predict(&hw, &with_tier(t), 512, 128).unwrap().pp_tok_s;
                prop_assert!(pp(less) >= pp(more) * (1.0 - 1e-12));
            }

            /// A faster disk curve never predicts slower.
            #[test]
            fn a_faster_curve_is_never_slower(
                p in 0u8..3,
                sync in 0.0f64..1e9, pf in 0.0f64..1e9, pre in 0.0f64..1e7,
                k in 1.0f64..8.0,
            ) {
                let slow = hw_disk();
                let mut fast = slow.clone();
                for d in &mut fast.devices {
                    for q in &mut d.queue_curve {
                        q.bytes_per_s *= k;
                    }
                }
                let t = tier(path(p), sync, pf, pre);
                prop_assert!(tg(&fast, t.clone()) >= tg(&slow, t.clone()) * (1.0 - 1e-12));
                let pp = |h: &HardwareGraph| predict(h, &with_tier(t.clone()), 512, 128).unwrap().pp_tok_s;
                prop_assert!(pp(&fast) >= pp(&slow) * (1.0 - 1e-12));
            }

            /// On a non-decreasing curve the interpolated rate is non-decreasing in depth.
            #[test]
            fn disk_rate_is_monotone_in_depth(
                steps in proptest::collection::vec((1u32..16, 0.0f64..2e9), 1..6),
                a in 0u32..80, b in 0u32..80,
            ) {
                let mut d = Device::new("n", DeviceKind::Disk, "n", Tier::Nvme, 1);
                let (mut depth, mut rate) = (0, 1e8);
                for (dd, dr) in steps {
                    depth += dd;
                    rate += dr;
                    d.queue_curve.push(QueuePoint { depth, bytes_per_s: rate });
                }
                let (lo, hi) = (a.min(b), a.max(b));
                prop_assert!(d.disk_rate(lo).unwrap() <= d.disk_rate(hi).unwrap() * (1.0 + 1e-12));
            }
        }
    }
}
