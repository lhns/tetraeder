//! Step-time cost model with measured per-op curves; `sim` holds the discrete-event simulator.
//!
//! Formulas follow docs/plan.md, "Planner and optimizer / Cost model" and ADR 0007:
//!
//! - `t_op(M) = c0 + f_op,dev(M)`, with c0 kept per graph mode ([`OpCost`]);
//! - roofline `max(FLOPs / F, bytes / B)` as the prior until a curve is measured ([`Roofline`]);
//! - transfer `alpha + size / beta` ([`transfer_time`]);
//! - ring all-reduce `2(p-1) alpha + 2(p-1)/p * n / beta` ([`allreduce_time`]);
//! - pipeline `t_token = sum_s (t_s + alpha_s + a / beta_s)` ([`pipeline_token_time`]);
//! - memory `W + KV(batch x context) + workspace <= capacity` ([`check_memory`]).

pub mod sim;

use tetraeder_profiler::Device;

/// Whether a step runs as a captured CUDA graph (small c0) or op by op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphMode {
    Graphs,
    Eager,
}

/// FLOPs and bytes of one op as a function of M, the tokens in the step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OpShape {
    pub flops_per_token: f64,
    /// Bytes read regardless of M (the weights).
    pub fixed_bytes: f64,
    pub bytes_per_token: f64,
}

/// Roofline prior: `max(FLOPs / F, bytes / B)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Roofline {
    pub shape: OpShape,
    pub flops_per_s: f64,
    pub bytes_per_s: f64,
}

impl Roofline {
    pub fn time(&self, m: f64) -> f64 {
        let flops = self.shape.flops_per_token * m;
        let bytes = self.shape.fixed_bytes + self.shape.bytes_per_token * m;
        (flops / self.flops_per_s).max(bytes / self.bytes_per_s)
    }
}

/// Cost of one op on one device: `c0` per graph mode plus `f(M)`.
///
/// `f` is the measured curve when there is one and the roofline otherwise. Measured points are
/// `(M, seconds)` with c0 already subtracted.
#[derive(Debug, Clone, PartialEq)]
pub struct OpCost {
    pub c0_graph_s: f64,
    pub c0_eager_s: f64,
    pub prior: Roofline,
    /// Sorted by M, may be empty.
    pub measured: Vec<(f64, f64)>,
}

impl OpCost {
    /// Roofline-only cost from a profiler device (needs its compute spec and bandwidth).
    pub fn from_prior(dev: &Device, shape: OpShape) -> Option<Self> {
        let c = dev.compute?;
        let b = dev.bandwidth_bytes_per_s?;
        Some(Self {
            c0_graph_s: c.c0_graph_s,
            c0_eager_s: c.c0_eager_s,
            prior: Roofline {
                shape,
                flops_per_s: c.flops_per_s,
                bytes_per_s: b.value,
            },
            measured: Vec::new(),
        })
    }

    pub fn with_measured(mut self, mut points: Vec<(f64, f64)>) -> Self {
        points.sort_by(|a, b| a.0.total_cmp(&b.0));
        self.measured = points;
        self
    }

    pub fn c0(&self, mode: GraphMode) -> f64 {
        match mode {
            GraphMode::Graphs => self.c0_graph_s,
            GraphMode::Eager => self.c0_eager_s,
        }
    }

    /// `f(M)`: piecewise linear through the measured points, proportional below the first,
    /// extending the last segment above the last; roofline when nothing is measured.
    pub fn f(&self, m: f64) -> f64 {
        let p = &self.measured;
        match p.len() {
            0 => self.prior.time(m),
            1 => p[0].1 * m / p[0].0,
            n => {
                if m <= p[0].0 {
                    return p[0].1 * m / p[0].0;
                }
                let i = p.iter().position(|q| m <= q.0).unwrap_or(n - 1).max(1);
                let (x0, y0) = p[i - 1];
                let (x1, y1) = p[i];
                y0 + (y1 - y0) * (m - x0) / (x1 - x0)
            }
        }
    }

    /// `t_op(M) = c0 + f(M)`.
    pub fn time(&self, m: f64, mode: GraphMode) -> f64 {
        self.c0(mode) + self.f(m)
    }
}

/// `alpha + size / beta`.
pub fn transfer_time(alpha_s: f64, beta_bytes_per_s: f64, bytes: f64) -> f64 {
    alpha_s + bytes / beta_bytes_per_s
}

/// Ring all-reduce over `p` devices of `n` bytes: `2(p-1) alpha + 2(p-1)/p * n / beta`.
pub fn allreduce_time(p: u32, alpha_s: f64, beta_bytes_per_s: f64, n_bytes: f64) -> f64 {
    let steps = 2.0 * f64::from(p.saturating_sub(1));
    steps * alpha_s + steps / f64::from(p.max(1)) * n_bytes / beta_bytes_per_s
}

/// One pipeline stage: compute time and the hop to the next stage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stage {
    pub t_s: f64,
    pub alpha_s: f64,
    pub beta_bytes_per_s: f64,
}

/// Per-token time through S stages: `sum_s (t_s + alpha_s + a / beta_s)`, `a` = activation bytes
/// per hop. A hop within one device is a stage with alpha 0 and beta infinite.
pub fn pipeline_token_time(stages: &[Stage], activation_bytes: f64) -> f64 {
    stages
        .iter()
        .map(|s| s.t_s + s.alpha_s + activation_bytes / s.beta_bytes_per_s)
        .sum()
}

/// Upper bound on tokens per second when each tier serves its share:
/// `min_tier B_tier / bytes_per_token_tier`. Input is `(B, bytes per token)` per tier.
pub fn tier_bound_tok_s(tiers: &[(f64, f64)]) -> f64 {
    tiers
        .iter()
        .filter(|t| t.1 > 0.0)
        .map(|t| t.0 / t.1)
        .fold(f64::INFINITY, f64::min)
}

/// KV bytes for `batch` sequences of `context` tokens.
pub fn kv_bytes(bytes_per_token: u64, batch: u64, context: u64) -> u64 {
    bytes_per_token * batch * context
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryOverflow {
    pub needed: u64,
    pub capacity: u64,
}

/// `W_shards + KV + workspace <= capacity`. Returns the headroom in bytes.
pub fn check_memory(
    weights: u64,
    kv: u64,
    workspace: u64,
    capacity: u64,
) -> Result<u64, MemoryOverflow> {
    let needed = weights + kv + workspace;
    capacity
        .checked_sub(needed)
        .ok_or(MemoryOverflow { needed, capacity })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() <= 1e-12 * b.abs().max(1e-12), "{a} vs {b}");
    }

    fn cost() -> OpCost {
        OpCost {
            c0_graph_s: 5e-6,
            c0_eager_s: 20e-6,
            prior: Roofline {
                shape: OpShape {
                    flops_per_token: 2e9,
                    fixed_bytes: 1e9,
                    bytes_per_token: 0.0,
                },
                flops_per_s: 1e12,
                bytes_per_s: 1e11,
            },
            measured: Vec::new(),
        }
    }

    #[test]
    fn roofline_is_memory_bound_at_m1_and_compute_bound_at_m8() {
        // M=1: flops 2e9/1e12 = 2 ms, bytes 1e9/1e11 = 10 ms -> 10 ms.
        close(cost().prior.time(1.0), 1e-2);
        // M=8: flops 16e9/1e12 = 16 ms, bytes 10 ms -> 16 ms.
        close(cost().prior.time(8.0), 1.6e-2);
    }

    #[test]
    fn op_time_adds_c0_per_graph_mode() {
        let c = cost();
        close(c.time(1.0, GraphMode::Graphs), 1e-2 + 5e-6);
        close(c.time(1.0, GraphMode::Eager), 1e-2 + 20e-6);
    }

    #[test]
    fn measured_curve_overrides_roofline() {
        let c = cost().with_measured(vec![(16.0, 4e-3), (1.0, 1e-3), (4.0, 2e-3)]);
        close(c.f(1.0), 1e-3);
        // between (1,1ms) and (4,2ms): at 2.5 -> 1.5 ms
        close(c.f(2.5), 1.5e-3);
        // between (4,2ms) and (16,4ms): at 10 -> 3 ms
        close(c.f(10.0), 3e-3);
        // below the first point: proportional; above the last: last slope (2ms per 12 tokens)
        close(c.f(0.5), 0.5e-3);
        close(c.f(28.0), 6e-3);
        close(c.time(10.0, GraphMode::Graphs), 3e-3 + 5e-6);
    }

    #[test]
    fn from_prior_needs_compute_and_bandwidth() {
        use tetraeder_profiler::{ComputeSpec, DeviceKind, Measurement, Tier};
        let mut d = Device::new("g", DeviceKind::Gpu, "g", Tier::Vram, 1 << 30);
        let shape = cost().prior.shape;
        assert!(OpCost::from_prior(&d, shape).is_none());
        d.compute = Some(ComputeSpec {
            flops_per_s: 1e12,
            c0_graph_s: 5e-6,
            c0_eager_s: 20e-6,
        });
        d.bandwidth_bytes_per_s = Some(Measurement::prior(1e11));
        assert_eq!(OpCost::from_prior(&d, shape).unwrap(), cost());
    }

    #[test]
    fn transfer_is_alpha_plus_size_over_beta() {
        // 10 us + 1e6 / 1e10 = 10 us + 100 us
        close(transfer_time(10e-6, 1e10, 1e6), 110e-6);
    }

    #[test]
    fn allreduce_matches_ring_formula() {
        // p=4: 2*3*5us = 30 us; 6/4 * 1e6/1e10 = 150 us
        close(allreduce_time(4, 5e-6, 1e10, 1e6), 180e-6);
        // p=2: 2*1*5us + 2/2 * 100 us = 110 us
        close(allreduce_time(2, 5e-6, 1e10, 1e6), 110e-6);
        assert_eq!(allreduce_time(1, 5e-6, 1e10, 1e6), 0.0);
    }

    #[test]
    fn pipeline_token_time_sums_stage_and_hop() {
        let stages = [
            Stage {
                t_s: 1e-3,
                alpha_s: 10e-6,
                beta_bytes_per_s: 1e10,
            },
            Stage {
                t_s: 2e-3,
                alpha_s: 50e-6,
                beta_bytes_per_s: 5e9,
            },
        ];
        // a = 1e5: stage 1 = 1 ms + 10 us + 10 us; stage 2 = 2 ms + 50 us + 20 us
        close(pipeline_token_time(&stages, 1e5), 3.09e-3);
    }

    #[test]
    fn tier_bound_is_the_slowest_tier() {
        // VRAM 1e12 B/s with 1e9 B/token -> 1000; NVMe 7e9 with 11e9 B/token -> 0.636...
        let v = tier_bound_tok_s(&[(1e12, 1e9), (7e9, 11e9), (1e11, 0.0)]);
        close(v, 7.0 / 11.0);
    }

    #[test]
    fn memory_check_boundaries() {
        // KV: 100 KB/token * 4 * 8192 = 3_276_800_000 B
        let kv = kv_bytes(100_000, 4, 8192);
        assert_eq!(kv, 3_276_800_000);
        // 10e9 + 3.2768e9 + 1e9 = 14_276_800_000
        assert_eq!(
            check_memory(10_000_000_000, kv, 1_000_000_000, 16_000_000_000),
            Ok(1_723_200_000)
        );
        assert_eq!(
            check_memory(10_000_000_000, kv, 1_000_000_000, 14_276_800_000),
            Ok(0)
        );
        assert_eq!(
            check_memory(10_000_000_000, kv, 1_000_000_000, 14_276_799_999),
            Err(MemoryOverflow {
                needed: 14_276_800_000,
                capacity: 14_276_799_999
            })
        );
    }
}
