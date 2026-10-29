//! Discrete-event simulator for load: replays a workload against a candidate plan.
//!
//! Skeleton scope: devices and links are single-server FIFO resources. A plan stub splits the
//! layers across pipeline stages, one stage per device. Prefill is cut into micro-batches that
//! pipeline through the stages; each decode token is its own M = 1 step (no batching across
//! requests yet). A layer is one op, costed by [`OpCost`] with the roofline prior. A stage hands
//! its activations to the next over the link between the two devices; the last stage returns the
//! sampled token to the first, so a token pays S hops, as in `pipeline_token_time`. Links are
//! undirected: both directions share one resource. Same input and seed give the same output.

use crate::{GraphMode, MemoryOverflow, OpCost, OpShape, Stage, check_memory, kv_bytes};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};
use tetraeder_profiler::HardwareGraph;

// ---------------------------------------------------------------- model, plan, workload

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelSpec {
    pub layers: u32,
    pub flops_per_token_per_layer: f64,
    pub weight_bytes_per_layer: u64,
    pub kv_bytes_per_token_per_layer: u64,
    /// Activation bytes per token crossing a stage boundary.
    pub activation_bytes_per_token: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageSpec {
    pub device: String,
    pub layers: u32,
}

/// Layer split across stages with prefill micro-batches.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanStub {
    pub stages: Vec<StageSpec>,
    /// A prompt is cut into at most this many chunks that pipeline through the stages.
    pub prefill_micro_batches: u32,
    pub graphs: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LengthDist {
    Fixed {
        tokens: u32,
    },
    Uniform {
        min: u32,
        max: u32,
    },
    /// Log-normal with the given median and sigma of the underlying normal.
    LogNormal {
        median: f64,
        sigma: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Arrival {
    /// Open loop: exponential inter-arrival times.
    Poisson { rate_per_s: f64 },
    /// Closed loop: this many requests in flight; a finished request is replaced at once.
    Closed { concurrency: u32 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkloadProfile {
    pub arrival: Arrival,
    pub prompt: LengthDist,
    pub output: LengthDist,
    pub requests: u32,
    pub seed: u64,
}

// ---------------------------------------------------------------- results

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Dist {
    pub n: usize,
    pub mean: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
}

impl Dist {
    pub fn from_samples(mut v: Vec<f64>) -> Self {
        if v.is_empty() {
            return Self::default();
        }
        v.sort_by(f64::total_cmp);
        let n = v.len();
        let pct = |p: f64| v[(((p * n as f64).ceil() as usize).max(1) - 1).min(n - 1)];
        Self {
            n,
            mean: v.iter().sum::<f64>() / n as f64,
            p50: pct(0.50),
            p95: pct(0.95),
            p99: pct(0.99),
            max: v[n - 1],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimReport {
    pub requests: usize,
    pub output_tokens: u64,
    pub makespan_s: f64,
    /// Output tokens per second over the whole run.
    pub throughput_tok_s: f64,
    /// Arrival to first token.
    pub ttft_s: Dist,
    /// Mean time per token after the first, per request (requests with one output token skipped).
    pub tpot_s: Dist,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SimError {
    UnknownDevice(String),
    MissingCompute(String),
    MissingLink(String, String),
    LayerMismatch { plan: u32, model: u32 },
    EmptyPlan,
    BadWorkload(&'static str),
}

impl std::fmt::Display for SimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownDevice(d) => write!(f, "unknown device {d}"),
            Self::MissingCompute(d) => write!(f, "device {d} lacks compute or bandwidth"),
            Self::MissingLink(a, b) => write!(f, "no link between {a} and {b}"),
            Self::LayerMismatch { plan, model } => {
                write!(f, "plan has {plan} layers, model has {model}")
            }
            Self::EmptyPlan => write!(f, "plan has no stages"),
            Self::BadWorkload(w) => write!(f, "bad workload: {w}"),
        }
    }
}

impl std::error::Error for SimError {}

// ---------------------------------------------------------------- rng

/// xoshiro256** seeded through splitmix64: small, fast, reproducible on every platform.
struct Rng([u64; 4]);

impl Rng {
    fn new(seed: u64) -> Self {
        let mut x = seed;
        let mut next = || {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        Self([next(), next(), next(), next()])
    }

    fn next_u64(&mut self) -> u64 {
        let s = &mut self.0;
        let r = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        r
    }

    /// Uniform in (0, 1].
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) + 1) as f64 / (1u64 << 53) as f64
    }

    fn exp(&mut self, rate: f64) -> f64 {
        -self.unit().ln() / rate
    }

    fn normal(&mut self) -> f64 {
        let (a, b) = (self.unit(), self.unit());
        (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
    }

    fn length(&mut self, d: &LengthDist) -> u32 {
        match *d {
            LengthDist::Fixed { tokens } => tokens,
            LengthDist::Uniform { min, max } => {
                let span = u64::from(max.saturating_sub(min)) + 1;
                min + (self.next_u64() % span) as u32
            }
            LengthDist::LogNormal { median, sigma } => {
                (median * (sigma * self.normal()).exp()).round().min(1e7) as u32
            }
        }
        .max(1)
    }
}

// ---------------------------------------------------------------- plan evaluation

/// A hop between consecutive stages (the last one wraps to the first).
struct Hop {
    /// Resource that serializes transfers: the lane group if the link has one, else the link.
    resource: Option<String>,
    alpha_s: f64,
    beta_bytes_per_s: f64,
}

struct Resolved {
    /// Layer-op cost per stage.
    ops: Vec<OpCost>,
    layers: Vec<u32>,
    devices: Vec<String>,
    hops: Vec<Hop>,
    mode: GraphMode,
    activation_bytes: f64,
    prefill_micro_batches: u32,
}

impl Resolved {
    fn new(hw: &HardwareGraph, model: &ModelSpec, plan: &PlanStub) -> Result<Self, SimError> {
        if plan.stages.is_empty() {
            return Err(SimError::EmptyPlan);
        }
        let total: u32 = plan.stages.iter().map(|s| s.layers).sum();
        if total != model.layers {
            return Err(SimError::LayerMismatch {
                plan: total,
                model: model.layers,
            });
        }
        let shape = OpShape {
            flops_per_token: model.flops_per_token_per_layer,
            fixed_bytes: model.weight_bytes_per_layer as f64,
            bytes_per_token: 0.0,
        };
        let mut ops = Vec::new();
        for s in &plan.stages {
            let dev = hw
                .device(&s.device)
                .ok_or_else(|| SimError::UnknownDevice(s.device.clone()))?;
            ops.push(
                OpCost::from_prior(dev, shape)
                    .ok_or_else(|| SimError::MissingCompute(s.device.clone()))?,
            );
        }
        let n = plan.stages.len();
        let mut hops = Vec::new();
        for i in 0..n {
            let (a, b) = (&plan.stages[i].device, &plan.stages[(i + 1) % n].device);
            if a == b {
                hops.push(Hop {
                    resource: None,
                    alpha_s: 0.0,
                    beta_bytes_per_s: f64::INFINITY,
                });
                continue;
            }
            let l = hw
                .link_between(a, b)
                .ok_or_else(|| SimError::MissingLink(a.clone(), b.clone()))?;
            let mut beta = l.beta_bytes_per_s.value;
            if let Some(g) = l.lane_group.as_deref().and_then(|g| hw.lane_group(g)) {
                beta = beta.min(g.capacity_bytes_per_s);
            }
            hops.push(Hop {
                resource: Some(format!(
                    "link:{}",
                    l.lane_group.as_deref().unwrap_or(l.id.as_str())
                )),
                alpha_s: l.alpha_s.value,
                beta_bytes_per_s: beta,
            });
        }
        Ok(Self {
            ops,
            layers: plan.stages.iter().map(|s| s.layers).collect(),
            devices: plan.stages.iter().map(|s| s.device.clone()).collect(),
            hops,
            mode: if plan.graphs {
                GraphMode::Graphs
            } else {
                GraphMode::Eager
            },
            activation_bytes: model.activation_bytes_per_token,
            prefill_micro_batches: plan.prefill_micro_batches.max(1),
        })
    }

    fn stage_time(&self, s: usize, m: f64) -> f64 {
        f64::from(self.layers[s]) * self.ops[s].time(m, self.mode)
    }
}

/// The plan's stages for one decode token, ready for [`crate::pipeline_token_time`].
pub fn decode_stages(
    hw: &HardwareGraph,
    model: &ModelSpec,
    plan: &PlanStub,
) -> Result<Vec<Stage>, SimError> {
    let r = Resolved::new(hw, model, plan)?;
    Ok((0..r.layers.len())
        .map(|s| Stage {
            t_s: r.stage_time(s, 1.0),
            alpha_s: r.hops[s].alpha_s,
            beta_bytes_per_s: r.hops[s].beta_bytes_per_s,
        })
        .collect())
}

/// Decode throughput ceiling in tokens per second: the busiest resource (device or link) per
/// token bounds it, however many requests are in flight.
pub fn decode_throughput_bound(
    hw: &HardwareGraph,
    model: &ModelSpec,
    plan: &PlanStub,
) -> Result<f64, SimError> {
    let r = Resolved::new(hw, model, plan)?;
    let mut busy: BTreeMap<String, f64> = BTreeMap::new();
    for s in 0..r.layers.len() {
        *busy.entry(r.devices[s].clone()).or_default() += r.stage_time(s, 1.0);
        if let Some(res) = &r.hops[s].resource {
            *busy.entry(res.clone()).or_default() +=
                r.activation_bytes / r.hops[s].beta_bytes_per_s;
        }
    }
    Ok(1.0 / busy.values().copied().fold(0.0, f64::max))
}

/// Why a plan fails the memory check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanMemoryError {
    /// A stage names a device the hardware graph does not have.
    UnknownDevice(String),
    Overflow(String, MemoryOverflow),
}

/// Per-stage memory check: weights + KV(batch x context) + workspace against device capacity.
pub fn check_plan_memory(
    hw: &HardwareGraph,
    model: &ModelSpec,
    plan: &PlanStub,
    batch: u64,
    context: u64,
    workspace_bytes: u64,
) -> Result<(), PlanMemoryError> {
    for s in &plan.stages {
        let Some(dev) = hw.device(&s.device) else {
            return Err(PlanMemoryError::UnknownDevice(s.device.clone()));
        };
        let l = u64::from(s.layers);
        let kv = kv_bytes(model.kv_bytes_per_token_per_layer * l, batch, context);
        check_memory(
            model.weight_bytes_per_layer * l,
            kv,
            workspace_bytes,
            dev.capacity_bytes,
        )
        .map_err(|e| PlanMemoryError::Overflow(s.device.clone(), e))?;
    }
    Ok(())
}

// ---------------------------------------------------------------- event loop

#[derive(Debug, Clone, Copy)]
enum Kind {
    Arrive {
        req: usize,
    },
    /// Activations of `tokens` tokens are at the input of `stage`.
    Ready {
        req: usize,
        stage: usize,
        tokens: u32,
        prefill: bool,
    },
    /// `stage` finished computing them.
    Done {
        req: usize,
        stage: usize,
        tokens: u32,
        prefill: bool,
    },
}

struct Event {
    t: f64,
    seq: u64,
    kind: Kind,
}

impl PartialEq for Event {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Event {}
impl PartialOrd for Event {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Event {
    /// Reversed so the max-heap pops the earliest event; ties break by insertion order.
    fn cmp(&self, o: &Self) -> Ordering {
        o.t.total_cmp(&self.t).then(o.seq.cmp(&self.seq))
    }
}

#[derive(Default, Clone)]
struct Req {
    arrival: f64,
    prompt: u32,
    output: u32,
    chunks_total: u32,
    chunks_done: u32,
    tokens_done: u32,
    first_token: f64,
    finish: f64,
}

struct Sim<'a> {
    r: &'a Resolved,
    heap: BinaryHeap<Event>,
    seq: u64,
    free_at: BTreeMap<String, f64>,
    reqs: Vec<Req>,
    rng: Rng,
    workload: &'a WorkloadProfile,
    issued: u32,
}

impl Sim<'_> {
    fn push(&mut self, t: f64, kind: Kind) {
        self.seq += 1;
        self.heap.push(Event {
            t,
            seq: self.seq,
            kind,
        });
    }

    fn new_request(&mut self, at: f64) -> usize {
        let prompt = self.rng.length(&self.workload.prompt);
        let output = self.rng.length(&self.workload.output);
        let chunks_total = self.r.prefill_micro_batches.min(prompt);
        self.reqs.push(Req {
            arrival: at,
            prompt,
            output,
            chunks_total,
            ..Req::default()
        });
        self.issued += 1;
        self.reqs.len() - 1
    }

    /// Claims a resource FIFO for `duration` from `t`; returns the start time.
    fn claim(&mut self, key: &str, t: f64, duration: f64) -> f64 {
        let free = self.free_at.entry(key.to_string()).or_insert(0.0);
        let start = t.max(*free);
        *free = start + duration;
        start
    }

    /// Time the activations sent at `t` over the hop after `stage` arrive.
    fn send(&mut self, stage: usize, t: f64, tokens: u32) -> f64 {
        let hop = &self.r.hops[stage];
        let Some(res) = hop.resource.clone() else {
            return t;
        };
        let ser = self.r.activation_bytes * f64::from(tokens) / hop.beta_bytes_per_s;
        let alpha = hop.alpha_s;
        self.claim(&res, t, ser) + ser + alpha
    }

    fn run(&mut self) -> SimReport {
        let n_stages = self.r.layers.len();
        let w = self.workload;
        match w.arrival {
            Arrival::Poisson { rate_per_s } => {
                let mut t = 0.0;
                for _ in 0..w.requests {
                    t += self.rng.exp(rate_per_s);
                    let req = self.new_request(t);
                    self.push(t, Kind::Arrive { req });
                }
            }
            Arrival::Closed { concurrency } => {
                for _ in 0..concurrency.min(w.requests) {
                    let req = self.new_request(0.0);
                    self.push(0.0, Kind::Arrive { req });
                }
            }
        }
        while let Some(Event { t, kind, .. }) = self.heap.pop() {
            match kind {
                Kind::Arrive { req } => {
                    let r = &self.reqs[req];
                    let (chunks, prompt) = (r.chunks_total, r.prompt);
                    // Chunk sizes differ by at most one token; all enter stage 0 in order.
                    for c in 0..chunks {
                        let tokens = prompt / chunks + u32::from(c < prompt % chunks);
                        self.push(
                            t,
                            Kind::Ready {
                                req,
                                stage: 0,
                                tokens,
                                prefill: true,
                            },
                        );
                    }
                }
                Kind::Ready {
                    req,
                    stage,
                    tokens,
                    prefill,
                } => {
                    let dur = self.r.stage_time(stage, f64::from(tokens));
                    let key = format!("dev:{}", self.r.devices[stage]);
                    let start = self.claim(&key, t, dur);
                    self.push(
                        start + dur,
                        Kind::Done {
                            req,
                            stage,
                            tokens,
                            prefill,
                        },
                    );
                }
                Kind::Done {
                    req,
                    stage,
                    tokens,
                    prefill,
                } => {
                    if stage + 1 < n_stages {
                        let at = self.send(stage, t, tokens);
                        self.push(
                            at,
                            Kind::Ready {
                                req,
                                stage: stage + 1,
                                tokens,
                                prefill,
                            },
                        );
                        continue;
                    }
                    // Last stage: a prefill chunk completes, or a decode token is emitted.
                    let emitted = if prefill {
                        let r = &mut self.reqs[req];
                        r.chunks_done += 1;
                        r.chunks_done == r.chunks_total
                    } else {
                        true
                    };
                    if !emitted {
                        continue;
                    }
                    let r = &mut self.reqs[req];
                    r.tokens_done += 1;
                    if r.tokens_done == 1 {
                        r.first_token = t;
                    }
                    if r.tokens_done >= r.output {
                        r.finish = t;
                        if matches!(w.arrival, Arrival::Closed { .. }) && self.issued < w.requests {
                            let nr = self.new_request(t);
                            self.push(t, Kind::Arrive { req: nr });
                        }
                    } else {
                        // Return the sampled token to stage 0, charged as one activation (an upper bound for a token id).
                        let at = self.send(n_stages - 1, t, 1);
                        self.push(
                            at,
                            Kind::Ready {
                                req,
                                stage: 0,
                                tokens: 1,
                                prefill: false,
                            },
                        );
                    }
                }
            }
        }
        self.report()
    }

    fn report(&self) -> SimReport {
        let done: Vec<&Req> = self.reqs.iter().filter(|r| r.finish > 0.0).collect();
        let makespan = done.iter().map(|r| r.finish).fold(0.0, f64::max);
        let output_tokens: u64 = done.iter().map(|r| u64::from(r.output)).sum();
        SimReport {
            requests: done.len(),
            output_tokens,
            makespan_s: makespan,
            throughput_tok_s: if makespan > 0.0 {
                output_tokens as f64 / makespan
            } else {
                0.0
            },
            ttft_s: Dist::from_samples(done.iter().map(|r| r.first_token - r.arrival).collect()),
            tpot_s: Dist::from_samples(
                done.iter()
                    .filter(|r| r.output > 1)
                    .map(|r| (r.finish - r.first_token) / f64::from(r.output - 1))
                    .collect(),
            ),
        }
    }
}

/// Replays `workload` against `plan` on `hw`.
pub fn simulate(
    hw: &HardwareGraph,
    model: &ModelSpec,
    plan: &PlanStub,
    workload: &WorkloadProfile,
) -> Result<SimReport, SimError> {
    match workload.arrival {
        Arrival::Poisson { rate_per_s } if rate_per_s <= 0.0 => {
            return Err(SimError::BadWorkload("arrival rate must be positive"));
        }
        Arrival::Closed { concurrency: 0 } => {
            return Err(SimError::BadWorkload("concurrency must be positive"));
        }
        _ => {}
    }
    if workload.requests == 0 {
        return Err(SimError::BadWorkload("no requests"));
    }
    let r = Resolved::new(hw, model, plan)?;
    let mut sim = Sim {
        r: &r,
        heap: BinaryHeap::new(),
        seq: 0,
        free_at: BTreeMap::new(),
        reqs: Vec::new(),
        rng: Rng::new(workload.seed),
        workload,
        issued: 0,
    };
    Ok(sim.run())
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline_token_time;
    use proptest::prelude::*;
    use tetraeder_profiler::{ComputeSpec, Device, DeviceKind, Link, LinkKind, Measurement, Tier};

    const FIXTURE: &str = include_str!("../fixtures/two_gpu_cpu_nvme.toml");
    const WAN_FIXTURE: &str = include_str!("../fixtures/wan_two_node.toml");

    fn model() -> ModelSpec {
        ModelSpec {
            layers: 32,
            flops_per_token_per_layer: 2.0e8,
            weight_bytes_per_layer: 100_000_000,
            kv_bytes_per_token_per_layer: 4096,
            activation_bytes_per_token: 8192.0,
        }
    }

    fn plan(stages: &[(&str, u32)]) -> PlanStub {
        PlanStub {
            stages: stages
                .iter()
                .map(|(d, l)| StageSpec {
                    device: (*d).into(),
                    layers: *l,
                })
                .collect(),
            prefill_micro_batches: 4,
            graphs: true,
        }
    }

    fn closed(concurrency: u32, requests: u32, prompt: u32, output: u32) -> WorkloadProfile {
        WorkloadProfile {
            arrival: Arrival::Closed { concurrency },
            prompt: LengthDist::Fixed { tokens: prompt },
            output: LengthDist::Fixed { tokens: output },
            requests,
            seed: 1,
        }
    }

    fn rel(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs()
    }

    /// `n` identical GPUs joined in a ring (a single link for n = 2).
    fn ring(n: usize, flops: &[f64], alpha: f64, beta: f64) -> HardwareGraph {
        let mut g = HardwareGraph::default();
        for i in 0..n {
            let mut d = Device::new(&format!("d{i}"), DeviceKind::Gpu, "g", Tier::Vram, 1 << 36);
            d.compute = Some(ComputeSpec {
                flops_per_s: flops[i % flops.len()],
                c0_graph_s: 10e-6,
                c0_eager_s: 40e-6,
            });
            d.bandwidth_bytes_per_s = Some(Measurement::prior(5.0e11));
            g.devices.push(d);
        }
        let edges = match n {
            1 => 0,
            2 => 1,
            _ => n,
        };
        for i in 0..edges {
            g.links.push(Link {
                id: format!("l{i}"),
                a: format!("d{i}"),
                b: format!("d{}", (i + 1) % n),
                kind: LinkKind::Network,
                alpha_s: Measurement::prior(alpha),
                beta_bytes_per_s: Measurement::prior(beta),
                lane_group: None,
            });
        }
        g
    }

    fn even_plan(n: usize, layers_each: u32) -> (ModelSpec, PlanStub) {
        let mut m = model();
        m.layers = layers_each * n as u32;
        let p = plan(
            &(0..n)
                .map(|i| (format!("d{i}"), layers_each))
                .collect::<Vec<_>>()
                .iter()
                .map(|(d, l)| (d.as_str(), *l))
                .collect::<Vec<_>>(),
        );
        (m, p)
    }

    #[test]
    fn fixture_loads_and_runs_deterministically() {
        let hw = HardwareGraph::from_toml(FIXTURE).unwrap();
        hw.validate().unwrap();
        let p = plan(&[("gpu0", 22), ("gpu1", 10)]);
        let w = WorkloadProfile {
            arrival: Arrival::Poisson { rate_per_s: 2.0 },
            prompt: LengthDist::LogNormal {
                median: 200.0,
                sigma: 0.6,
            },
            output: LengthDist::Uniform { min: 20, max: 120 },
            requests: 200,
            seed: 42,
        };
        let a = simulate(&hw, &model(), &p, &w).unwrap();
        let b = simulate(&hw, &model(), &p, &w).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.requests, 200);
        assert!(a.ttft_s.p50 > 0.0 && a.ttft_s.p95 >= a.ttft_s.p50);
        assert!(a.tpot_s.p50 > 0.0 && a.throughput_tok_s > 0.0);
        let other = simulate(&hw, &model(), &p, &WorkloadProfile { seed: 43, ..w }).unwrap();
        assert_ne!(a, other);
    }

    #[test]
    fn plan_errors_are_reported() {
        let hw = HardwareGraph::from_toml(FIXTURE).unwrap();
        let w = closed(1, 1, 1, 2);
        let e = simulate(&hw, &model(), &plan(&[("gpu0", 31)]), &w).unwrap_err();
        assert_eq!(
            e,
            SimError::LayerMismatch {
                plan: 31,
                model: 32
            }
        );
        let e = simulate(&hw, &model(), &plan(&[("nvme0", 32)]), &w).unwrap_err();
        assert_eq!(e, SimError::MissingCompute("nvme0".into()));
        let e = simulate(&hw, &model(), &plan(&[("gpu9", 32)]), &w).unwrap_err();
        assert_eq!(e, SimError::UnknownDevice("gpu9".into()));
    }

    #[test]
    fn memory_check_flags_the_small_gpu() {
        let hw = HardwareGraph::from_toml(FIXTURE).unwrap();
        // 100 MB/layer: 20 layers = 2 GB fits 8 GB gpu1; 32 layers = 3.2 GB fits too.
        let ok = plan(&[("gpu0", 12), ("gpu1", 20)]);
        assert!(check_plan_memory(&hw, &model(), &ok, 1, 4096, 0).is_ok());
        // 20 layers x 4096 B x 100_000 x 100 = 819 GB of KV: gpu1 overflows first.
        let err = check_plan_memory(&hw, &model(), &ok, 100, 100_000, 0).unwrap_err();
        assert!(matches!(err, PlanMemoryError::Overflow(ref d, _) if d == "gpu0"));
        let e = check_plan_memory(&hw, &model(), &plan(&[("gpu9", 32)]), 1, 1, 0).unwrap_err();
        assert_eq!(e, PlanMemoryError::UnknownDevice("gpu9".into()));
    }

    #[test]
    fn single_stream_token_time_is_hand_computed() {
        // Two stages, 16 layers each, one link: alpha 20 us, beta 1e9, 8192 B per hop.
        // Layer: c0 10 us + max(2e8 / 1e12 = 0.2 ms, 1e8 / 5e11 = 0.2 ms) = 210 us.
        // Stage: 16 x 210 us = 3.36 ms. Hop: 20 us + 8.192 us. Two hops per token.
        let hw = ring(2, &[1e12], 20e-6, 1e9);
        let (m, p) = even_plan(2, 16);
        let r = simulate(&hw, &m, &p, &closed(1, 1, 1, 11)).unwrap();
        let expect = 2.0 * 3.36e-3 + 2.0 * (20e-6 + 8.192e-6);
        assert!(
            rel(r.tpot_s.mean, expect) < 1e-9,
            "{} vs {expect}",
            r.tpot_s.mean
        );
    }

    #[test]
    fn prefill_micro_batches_pipeline() {
        let hw = ring(2, &[1e12], 0.0, 1e12);
        let (m, mut p) = even_plan(2, 16);
        let w = closed(1, 1, 256, 1);
        p.prefill_micro_batches = 1;
        let whole = simulate(&hw, &m, &p, &w).unwrap().ttft_s.mean;
        p.prefill_micro_batches = 8;
        let chunked = simulate(&hw, &m, &p, &w).unwrap().ttft_s.mean;
        assert!(chunked < whole * 0.8, "{chunked} vs {whole}");
    }

    #[test]
    fn high_latency_link_costs_latency_not_throughput() {
        let wan = HardwareGraph::from_toml(WAN_FIXTURE).unwrap();
        wan.validate().unwrap();
        let rtt = 2.0 * wan.link_between("gpu0", "gpu1").unwrap().alpha_s.value;
        assert!((rtt - 0.050).abs() < 1e-12, "fixture is a 50 ms RTT link");
        let mut lan = wan.clone();
        lan.links[0].alpha_s = Measurement::prior(20e-6);
        let lan_alpha = 20e-6;
        let p = plan(&[("gpu0", 16), ("gpu1", 16)]);
        let m = model();
        let s = 2.0;

        // Single stream: each of the S hops adds (alpha_wan - alpha_lan) = RTT/2 per token.
        let one = closed(1, 1, 1, 21);
        let t_wan = simulate(&wan, &m, &p, &one).unwrap().tpot_s.mean;
        let t_lan = simulate(&lan, &m, &p, &one).unwrap().tpot_s.mean;
        assert!(
            rel(t_wan - t_lan, s * (rtt / 2.0 - lan_alpha)) < 1e-6,
            "delta {}",
            t_wan - t_lan
        );
        // Both match the closed-form pipeline time.
        let f = |hw: &HardwareGraph| {
            pipeline_token_time(
                &decode_stages(hw, &m, &p).unwrap(),
                m.activation_bytes_per_token,
            )
        };
        assert!(rel(t_wan, f(&wan)) < 1e-9 && rel(t_lan, f(&lan)) < 1e-9);

        // Pipelined: enough requests in flight to cover the longer token time.
        let bound = decode_throughput_bound(&wan, &m, &p).unwrap();
        let inflight = (f(&wan) * bound).ceil() as u32 + 2;
        let many = closed(inflight, inflight * 4, 1, 200);
        let th_wan = simulate(&wan, &m, &p, &many).unwrap().throughput_tok_s;
        let th_lan = simulate(&lan, &m, &p, &many).unwrap().throughput_tok_s;
        assert!(th_wan > 0.9 * bound, "{th_wan} vs bound {bound}");
        assert!(th_wan > 0.9 * th_lan, "{th_wan} vs lan {th_lan}");

        // Without enough requests in flight the long link does cost throughput.
        let few = simulate(&wan, &m, &p, &closed(2, 8, 1, 50))
            .unwrap()
            .throughput_tok_s;
        assert!(few < 0.2 * bound);
    }

    #[test]
    fn four_site_ring_adds_s_times_half_rtt() {
        let m_alpha = 0.025;
        let wan = ring(4, &[1e12], m_alpha, 1.25e8);
        let lan = ring(4, &[1e12], 20e-6, 1.25e8);
        let (m, p) = even_plan(4, 8);
        let one = closed(1, 1, 1, 11);
        let d = simulate(&wan, &m, &p, &one).unwrap().tpot_s.mean
            - simulate(&lan, &m, &p, &one).unwrap().tpot_s.mean;
        assert!(rel(d, 4.0 * (m_alpha - 20e-6)) < 1e-6, "{d}");
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(48))]

        /// One request at a time: TPOT is exactly the sum of stage times and hops.
        #[test]
        fn single_stream_latency_is_stage_sum_plus_hops(
            n in 1usize..=6,
            layers_each in 1u32..=6,
            speeds in proptest::collection::vec(2.0e11f64..4.0e12, 1..=6),
            alpha in 0.0f64..5e-3,
            beta in 1.0e8f64..1.0e11,
        ) {
            let hw = ring(n, &speeds, alpha, beta);
            let (m, p) = even_plan(n, layers_each);
            let r = simulate(&hw, &m, &p, &closed(1, 1, 3, 6)).unwrap();
            let expect = pipeline_token_time(
                &decode_stages(&hw, &m, &p).unwrap(),
                m.activation_bytes_per_token,
            );
            prop_assert!(rel(r.tpot_s.mean, expect) < 1e-9, "{} vs {}", r.tpot_s.mean, expect);
        }

        /// Enough requests in flight: throughput reaches the slowest-resource bound, never beyond.
        #[test]
        fn throughput_approaches_slowest_stage_bound(
            n in 1usize..=6,
            layers_each in 1u32..=6,
            speeds in proptest::collection::vec(2.0e11f64..4.0e12, 1..=6),
            alpha in 0.0f64..2e-3,
        ) {
            let hw = ring(n, &speeds, alpha, 1.0e10);
            let (m, p) = even_plan(n, layers_each);
            let bound = decode_throughput_bound(&hw, &m, &p).unwrap();
            let t_token = pipeline_token_time(
                &decode_stages(&hw, &m, &p).unwrap(),
                m.activation_bytes_per_token,
            );
            let inflight = (t_token * bound).ceil() as u32 + 2;
            let r = simulate(&hw, &m, &p, &closed(inflight, inflight * 3, 1, 150)).unwrap();
            prop_assert!(r.throughput_tok_s <= bound * (1.0 + 1e-9), "{} > {}", r.throughput_tok_s, bound);
            prop_assert!(r.throughput_tok_s >= 0.85 * bound, "{} vs {}", r.throughput_tok_s, bound);
        }
    }
}
