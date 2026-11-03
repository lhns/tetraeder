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
//! and the hop term is still charged (the CPU part's other ops exchange activations). `pp = P / sum_ubatches (...)`. Breakdown labels are `compute:<dev>`, `switches`,
//! `offload_copy:<dev>`, `offload_compute:<dev>`; `tg_breakdown` is seconds per token,
//! `pp_breakdown` is seconds over the whole prompt.

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
    let gpu = devs
        .iter()
        .position(|(d, _)| d.kind == DeviceKind::Gpu)
        .filter(|_| f.op_offload);

    // Decode.
    let ctx = f64::from(gen_tokens) / 2.0;
    let mut tg: Vec<(String, f64)> = Vec::new();
    for (p, (d, r)) in f.parts.iter().zip(&devs) {
        let t = r.c0 * p.ops_per_token
            + ((2.0 * p.params_per_token + p.attn_flops_per_ctx_token * ctx) / r.f)
                .max((p.weight_bytes_per_token + p.kv_bytes_per_ctx_token * ctx) / r.b);
        add(&mut tg, format!("compute:{}", d.id), t);
    }
    if f.switches_per_token > 0.0 {
        let h = need_hop()?;
        let t = f.switches_per_token * (h.alpha + f.activation_bytes_per_token / h.beta);
        add(&mut tg, "switches".into(), t);
    }
    let t_tok: f64 = tg.iter().map(|e| e.1).sum();

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
        if f.switches_per_token > 0.0 {
            let h = need_hop()?;
            add(
                &mut pp,
                "switches".into(),
                f.switches_per_token * (h.alpha + m * f.activation_bytes_per_token / h.beta),
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
    fn features_roundtrip_json() {
        let f = features(6.0, 4.0);
        let s = serde_json::to_string(&f).unwrap();
        assert_eq!(serde_json::from_str::<RunFeatures>(&s).unwrap(), f);
    }
}
