//! Analytic stage times: the roofline of each device (costmodel), hops of `alpha + a / beta`,
//! ring all-reduce for tensor parallel, tier streaming for experts.

use super::prep::Prep;
use tetraeder_costmodel::{GraphMode, OpCost, OpShape, Roofline, allreduce_time};
use tetraeder_profiler::{Device, DeviceKind, HardwareGraph};

/// `(flops/s, bytes/s)` of a device; a CPU at all its threads.
pub(crate) fn rates(dev: &Device) -> Option<(f64, f64)> {
    match dev.kind {
        DeviceKind::Cpu => dev.cpu_rates(dev.threads.unwrap_or(u32::MAX)),
        _ => Some((dev.compute?.flops_per_s, dev.bandwidth_bytes_per_s?.value)),
    }
}

/// Time of one op on `dev` with `m` tokens in the step; infinite when the device lacks numbers.
pub(crate) fn op_time(
    hw: &HardwareGraph,
    dev: &str,
    flops_per_token: f64,
    fixed_bytes: f64,
    m: f64,
    mode: GraphMode,
) -> f64 {
    let Some(d) = hw.device(dev) else {
        return f64::INFINITY;
    };
    let (Some((f, b)), Some(c)) = (rates(d), d.compute) else {
        return f64::INFINITY;
    };
    OpCost {
        c0_graph_s: c.c0_graph_s,
        c0_eager_s: c.c0_eager_s,
        prior: Roofline {
            shape: OpShape {
                flops_per_token,
                fixed_bytes,
                bytes_per_token: 0.0,
            },
            flops_per_s: f,
            bytes_per_s: b,
        },
        measured: Vec::new(),
    }
    .time(m, mode)
}

/// `(alpha, beta)` of the direct link between two devices, beta capped by its lane group; the
/// same device is a free hop.
pub(crate) fn hop(hw: &HardwareGraph, a: &str, b: &str) -> Option<(f64, f64)> {
    if a == b {
        return Some((0.0, f64::INFINITY));
    }
    let l = hw.link_between(a, b)?;
    let mut beta = l.beta_bytes_per_s.value;
    if let Some(g) = l.lane_group.as_deref().and_then(|g| hw.lane_group(g)) {
        beta = beta.min(g.capacity_bytes_per_s);
    }
    Some((l.alpha_s.value, beta))
}

/// Everything the time of a stage depends on, with per-layer numbers averaged over its layers.
#[derive(Debug, Clone)]
pub(crate) struct StageShape {
    pub group: Vec<String>,
    pub tp: u32,
    pub ep: u32,
    pub layers: u32,
    /// Attention and MLP weight bytes and FLOPs per token per layer on the primary device.
    pub nb: f64,
    pub nf: f64,
    /// KV bytes a decode token reads per layer on the primary device.
    pub kv_read: f64,
    /// Bytes and FLOPs per token of one expert.
    pub eb: f64,
    pub ef: f64,
    /// Experts per layer in the hot, RAM and NVMe tiers (single-device stages).
    pub counts: [usize; 3],
    /// The RAM tier computes on the stage device (streamed) instead of on the CPU.
    pub stream: bool,
    pub cpu: Option<String>,
    pub nvme: Option<String>,
    pub head_bytes: f64,
    pub head_flops: f64,
}

/// Time for `m` tokens through the stage. Infinite when a link or device is missing.
pub(crate) fn stage_time(p: &Prep, s: &StageShape, m: f64, mode: GraphMode) -> f64 {
    let hw = p.hw;
    let dev = s.group[0].as_str();
    let e = p.experts;
    let decode = m <= 1.0;
    let cum_d = if decode { &p.cum_d1 } else { &p.cum_dm };
    let touched = |lo: usize, hi: usize| cum_d[hi] - cum_d[lo];
    let flops_mass = |lo: usize, hi: usize| p.cum_p[hi] - p.cum_p[lo];
    let kv = if decode { s.kv_read } else { 0.0 };
    let mut flops = s.nf;
    let mut bytes = s.nb + kv;
    let mut extra = 0.0;
    let [h0, h1, h2] = s.counts;
    let g = s.group.len();
    if e > 0 && g > 1 {
        let g = g as f64;
        flops += flops_mass(0, e) * s.ef / g;
        bytes += touched(0, e) * s.eb / g;
    } else if e > 0 {
        flops += flops_mass(0, h0) * s.ef;
        bytes += touched(0, h0) * s.eb;
        if h1 > 0 {
            let Some(cpu) = s.cpu.as_deref() else {
                return f64::INFINITY;
            };
            let (tb, tf) = (touched(h0, h0 + h1) * s.eb, flops_mass(h0, h0 + h1) * s.ef);
            let Some((a, b)) = hop(hw, cpu, dev) else {
                return f64::INFINITY;
            };
            if s.stream {
                flops += tf;
                bytes += tb;
                extra += a + tb / b;
            } else {
                extra += op_time(hw, cpu, tf, tb, m, mode) + 2.0 * (a + p.act * m / b);
            }
        }
        if h2 > 0 {
            let (Some(cpu), Some(nvme)) = (s.cpu.as_deref(), s.nvme.as_deref()) else {
                return f64::INFINITY;
            };
            let (hi, lo) = (h0 + h1 + h2, h0 + h1);
            let (tb, tf) = (touched(lo, hi) * s.eb, flops_mass(lo, hi) * s.ef);
            let (Some((a1, b1)), Some((a2, b2))) = (hop(hw, nvme, cpu), hop(hw, cpu, dev)) else {
                return f64::INFINITY;
            };
            let tier = hw
                .device(nvme)
                .and_then(|d| d.bandwidth_bytes_per_s)
                .map_or(f64::INFINITY, |b| b.value);
            let beta = b1.min(b2).min(tier);
            flops += tf;
            bytes += tb;
            extra += a1 + a2 + tb / beta;
        }
    }
    // Compute on every group member; the slowest sets the layer time.
    let mut layer = s
        .group
        .iter()
        .map(|d| op_time(hw, d, flops, bytes, m, mode))
        .fold(0.0, f64::max);
    if g > 1 {
        let (mut alpha, mut beta) = (0.0f64, f64::INFINITY);
        for d in &s.group[1..] {
            let Some((a, b)) = hop(hw, dev, d) else {
                return f64::INFINITY;
            };
            alpha = alpha.max(a);
            beta = beta.min(b);
        }
        if s.tp > 1 {
            layer += 2.0 * allreduce_time(g as u32, alpha, beta, p.act * m);
        }
        if s.ep > 1 {
            layer += 2.0 * (alpha + p.act * m / beta);
        }
    }
    let head = if s.head_bytes > 0.0 {
        op_time(hw, dev, s.head_flops, s.head_bytes, m, mode)
    } else {
        0.0
    };
    f64::from(s.layers) * (layer + extra) + head
}
