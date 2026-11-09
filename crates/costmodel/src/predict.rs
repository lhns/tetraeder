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

use serde::{Deserialize, Serialize};
use std::fmt;
use tetraeder_profiler::{Device, DeviceKind, HardwareGraph};

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
    b: f64,
    c0: f64,
}

fn rates(dev: &Device, threads: u32, graphs: bool) -> Result<Rates, PredictError> {
    let missing = |what| PredictError::MissingSpec {
        device: dev.id.clone(),
        what,
    };
    let c = dev.compute.ok_or_else(|| missing("compute spec"))?;
    let (f, b, c0) = if dev.kind == DeviceKind::Cpu {
        let (f, b) = dev
            .cpu_rates(threads)
            .ok_or_else(|| missing("flops or bandwidth"))?;
        (f, b, c.c0_eager_s)
    } else {
        let b = dev
            .bandwidth_bytes_per_s
            .ok_or_else(|| missing("bandwidth"))?
            .value;
        let c0 = if graphs { c.c0_graph_s } else { c.c0_eager_s };
        (c.flops_per_s, b, c0)
    };
    if !(f > 0.0 && b > 0.0 && c0 >= 0.0 && f.is_finite() && b.is_finite() && c0.is_finite()) {
        return Err(PredictError::Invalid(format!(
            "device {} has non-positive rates",
            dev.id
        )));
    }
    Ok(Rates { f, b, c0 })
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
        });
        gpu.bandwidth_bytes_per_s = Some(Measurement::prior(800e9));
        let mut cpu = Device::new("cpu0", DeviceKind::Cpu, "c", Tier::Ram, 64 << 30);
        cpu.threads = Some(16);
        cpu.compute = Some(ComputeSpec {
            flops_per_s: 1.2e12,
            c0_graph_s: 3e-6,
            c0_eager_s: 3e-6,
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
