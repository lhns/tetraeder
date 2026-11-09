//! The replay on a virtual clock (ADR 0056).
//!
//! [`replay_timed`] runs the same replay as [`super::sim::replay`], so its byte counters are
//! equal, and adds time: the simulator reports its calls through [`Timing`], and [`Driver`]
//! turns them into reads on a [`Queue`] and a [`Disk`] plus compute time from a [`TimeModel`].
//! It checks a plan's read path and prefetch depth in milliseconds of virtual time; it is not an
//! input to the analytic cost model.
//!
//! One layer step, in the engine's order:
//! 1. The prefetch reads of the layer are waited for (`Reader`, `Service`; also at a step
//!    boundary, whatever the path).
//! 2. Per CPU group (the experts of one chunk of RAM slots): its misses are read now, urgent.
//!    Without overlap the group computes once all its reads are in. With `ServiceOverlap` the
//!    resident group computes first, then the groups whose reads have landed, in landing order,
//!    each costing `cpu_group_s` plus `cpu_expert_s` per routed pair.
//! 3. The GPU part of the layer, `gpu_layer_s`.
//!
//! Reads are chunks of `chunk_bytes`; up to `workers` are in service at once (`sync_depth` for
//! `Reader`, whose reads are one thread's queue) and share the disk's rate at that depth.
//! Deadlines are the `(step, layer)` order, as in the service (the reader serves prefetches in
//! submit order, and synchronous reads first).

use super::sim::{Config, Outcome, Timing, Trace, replay_timing};
use super::trace::TraceHeader;
use crate::dio::Range;
use crate::io::{Chunk, Deadline, Disk, Part, Queue, UnitId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tetraeder_profiler::QueuePoint;

/// How the bytes of a RAM-tier miss are read (the cost model's `ReadPath`, ADR 0056).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadPath {
    /// The single reader thread (ADR 0038).
    Reader,
    /// The I/O service; compute waits for the layer's reads (ADR 0054).
    Service,
    /// The service with CPU experts computed in arrival order.
    ServiceOverlap,
}

fn two() -> u32 {
    2
}

/// Plain numbers for the virtual clock. The planner writes them from a calibrated prediction
/// (`tetraeder-plan explain --time-model`); it keeps its own copy of this struct and
/// `tests/data/time-model.json` is the contract both sides' tests read.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimeModel {
    /// GPU compute of one layer of one decode token, seconds.
    pub gpu_layer_s: f64,
    /// CPU time of one routed (token, expert) pair, seconds.
    pub cpu_expert_s: f64,
    /// CPU time per group of experts, whatever its size (the dispatch cost, `c0`).
    pub cpu_group_s: f64,
    /// The disk's bytes per second by queue depth.
    pub curve: Vec<QueuePoint>,
    pub chunk_bytes: u64,
    /// Reads in service at once on the service paths.
    pub workers: u32,
    pub path: ReadPath,
    /// Direct-I/O queue depth of the synchronous reads; the reads in service at once on the
    /// `Reader` path.
    pub sync_depth: u32,
    /// 1 or 2 layers ahead (the recorded depth-2 predictions are ignored at 1).
    #[serde(default = "two")]
    pub prefetch_depth: u32,
}

/// One decode token.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StepTiming {
    pub step: u32,
    pub time_s: f64,
    /// CPU compute of groups that did not wait for a read.
    pub cpu_ready_s: f64,
    /// The main thread blocked on reads.
    pub read_wait_s: f64,
    /// CPU compute of groups that ran after their reads landed.
    pub cpu_landed_s: f64,
    /// From the start of the layer steps that read, to the end of their last wait.
    pub read_span_s: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DecodeTiming {
    pub tokens: u64,
    pub time_s: f64,
    pub tokens_per_s: f64,
    pub step_s_mean: f64,
    pub cpu_ready_s_mean: f64,
    pub read_wait_s_mean: f64,
    pub cpu_landed_s_mean: f64,
    pub read_span_s_mean: f64,
    /// `1 - wait / span` over the decode steps (`tetraeder.moe.read.hidden`); 1 without reads.
    pub hidden_share: f64,
    pub steps: Vec<StepTiming>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TimedOutcome {
    /// Equal to [`super::sim::replay`]'s outcome (with prefetch depth 2). Not serialized: the
    /// byte outcome is written beside it.
    #[serde(skip_serializing)]
    pub bytes: Outcome,
    pub path: ReadPath,
    pub prefetch_depth: u32,
    pub decode: DecodeTiming,
    pub prefill_time_s: f64,
    /// Reads queued and landed; equal after a replay.
    pub units_submitted: u64,
    pub units_landed: u64,
    /// Bytes the disk served.
    pub disk_bytes: u64,
}

#[derive(Default)]
struct StepAcc {
    cpu_ready: f64,
    read_wait: f64,
    cpu_landed: f64,
    span: f64,
}

struct Driver<'m> {
    m: &'m TimeModel,
    n_layer: u64,
    t: f64,
    queue: Queue<()>,
    disk: Disk<Chunk<()>>,
    landed: HashMap<UnitId, f64>,
    submitted: u64,
    served: u64,
    seq: u64,
    /// Prefetch units per (layer, expert), until the expert is routed.
    pf: HashMap<(usize, u32), UnitId>,
    /// Misses of the group being read.
    miss: Vec<(u32, UnitId)>,
    step: u32,
    started: bool,
    decode: bool,
    step_start: f64,
    step_end: f64,
    acc: StepAcc,
    layer_start: f64,
    layer_wait: f64,
    layer_reads: bool,
    layer_span_end: f64,
    steps: Vec<StepTiming>,
    prefill_s: f64,
}

impl<'m> Driver<'m> {
    fn new(h: &TraceHeader, m: &'m TimeModel) -> Result<Self, String> {
        if m.curve.is_empty() {
            return Err("time model: the disk curve is empty".into());
        }
        if m.chunk_bytes == 0 || m.workers == 0 || m.sync_depth == 0 {
            return Err("time model: chunk_bytes, workers and sync_depth must be positive".into());
        }
        let in_service = match m.path {
            ReadPath::Reader => m.sync_depth,
            _ => m.workers,
        };
        Ok(Self {
            m,
            n_layer: u64::from(h.n_layer),
            t: 0.0,
            queue: Queue::new(m.chunk_bytes as usize),
            disk: Disk::new(m.curve.clone(), in_service as usize),
            landed: HashMap::new(),
            submitted: 0,
            served: 0,
            seq: 0,
            pf: HashMap::new(),
            miss: Vec::new(),
            step: 0,
            started: false,
            decode: false,
            step_start: 0.0,
            step_end: 0.0,
            acc: StepAcc::default(),
            layer_start: 0.0,
            layer_wait: 0.0,
            layer_reads: false,
            layer_span_end: 0.0,
            steps: Vec::new(),
            prefill_s: 0.0,
        })
    }

    fn pump(&mut self) {
        while self.disk.free() > 0 {
            let Some(c) = self.queue.pop() else { break };
            let bytes = c.range.len as u64;
            if self.disk.submit(bytes, c).is_err() {
                unreachable!("a worker was free");
            }
        }
    }

    /// Moves the disk to `to`, landing every chunk that finishes on the way and refilling the
    /// workers at each finish.
    fn advance(&mut self, to: f64) {
        loop {
            self.pump();
            match self.disk.next_completion() {
                Some(tc) if tc <= to => {
                    for (at, chunk) in self.disk.advance_to(tc) {
                        let bytes = chunk.range.len as u64;
                        self.served += bytes;
                        if let Some(u) = self.queue.complete(at, chunk, Ok(bytes)) {
                            self.landed.insert(u, at);
                            self.queue.collect(&[u]);
                        }
                    }
                }
                _ => {
                    self.disk.advance_to(to);
                    return;
                }
            }
        }
    }

    fn order(&mut self, layer: usize, sync: bool) -> u64 {
        match self.m.path {
            ReadPath::Reader if sync => 0,
            ReadPath::Reader => {
                self.seq += 1;
                self.seq
            }
            _ => u64::from(self.step) * self.n_layer + layer as u64,
        }
    }

    fn submit_unit(&mut self, bytes: u64, urgent: bool, order: u64) -> UnitId {
        self.advance(self.t);
        let id = self.queue.submit(
            self.t,
            Deadline { order, urgent },
            vec![Part {
                shard: 0,
                range: Range {
                    offset: 0,
                    len: bytes as usize,
                },
                dst: (),
            }],
        );
        self.submitted += 1;
        if !self.queue.any_open(&[id]) {
            // Nothing to read.
            self.landed.insert(id, self.t);
            self.queue.collect(&[id]);
        }
        self.pump();
        id
    }

    /// The main thread blocks until all (`any` false) or one of `units` has landed.
    fn wait(&mut self, units: &[UnitId], any: bool) {
        let landed = |d: &Self| {
            let mut it = units.iter().map(|u| d.landed.contains_key(u));
            if any { it.any(|x| x) } else { it.all(|x| x) }
        };
        let before = self.t;
        while !units.is_empty() && !landed(self) {
            self.pump();
            let Some(tc) = self.disk.next_completion() else {
                break;
            };
            self.advance(tc);
            self.t = self.t.max(tc);
        }
        let waited = self.t - before;
        if waited > 0.0 {
            self.layer_wait += waited;
            self.acc.read_wait += waited;
        }
        self.layer_span_end = self.layer_span_end.max(self.t);
    }

    fn group_cost(&self, pairs: u32) -> f64 {
        self.m.cpu_group_s + self.m.cpu_expert_s * f64::from(pairs)
    }

    fn finish_step(&mut self) {
        if !self.started {
            return;
        }
        let time = (self.step_end - self.step_start).max(0.0);
        if self.decode {
            self.steps.push(StepTiming {
                step: self.step,
                time_s: time,
                cpu_ready_s: self.acc.cpu_ready,
                read_wait_s: self.acc.read_wait,
                cpu_landed_s: self.acc.cpu_landed,
                read_span_s: self.acc.span,
            });
        } else {
            self.prefill_s += time;
        }
        self.acc = StepAcc::default();
    }

    /// Runs the reads still queued to the end, so every unit lands.
    fn drain(&mut self) {
        loop {
            self.pump();
            let Some(tc) = self.disk.next_completion() else {
                break;
            };
            self.advance(tc);
        }
    }
}

impl Timing for Driver<'_> {
    fn layer_begin(&mut self, step: u32, _layer: usize, decode: bool) {
        if !self.started || step != self.step {
            self.finish_step();
            self.started = true;
            self.step = step;
            self.decode = decode;
            self.step_start = self.t;
        }
        self.layer_start = self.t;
        self.layer_wait = 0.0;
        self.layer_reads = false;
        self.layer_span_end = self.t;
    }

    fn prefetch_submit(&mut self, layer: usize, experts: &[u32], bytes: u64) {
        for &e in experts {
            let order = self.order(layer, false);
            let id = self.submit_unit(bytes, false, order);
            self.pf.insert((layer, e), id);
        }
    }

    fn prefetch_land(&mut self, layer: usize, settle: bool) {
        if !settle && self.m.path == ReadPath::ServiceOverlap {
            return;
        }
        let units: Vec<UnitId> = self
            .pf
            .iter()
            .filter(|((l, _), _)| *l == layer)
            .map(|(_, &u)| u)
            .collect();
        if units.iter().any(|u| !self.landed.contains_key(u)) {
            self.layer_reads = true;
        }
        self.wait(&units, false);
    }

    fn miss_submit(&mut self, layer: usize, expert: u32, bytes: u64) {
        let order = self.order(layer, true);
        let id = self.submit_unit(bytes, true, order);
        self.miss.push((expert, id));
        self.pf.remove(&(layer, expert));
        self.layer_reads = true;
    }

    fn cpu_group(&mut self, layer: usize, experts: &[(u32, u32)], compute: bool) {
        self.advance(self.t);
        let mut ready_pairs = 0;
        let mut ready_n = 0;
        let mut pending: Vec<(UnitId, u32)> = Vec::new();
        for &(e, pairs) in experts {
            let unit = match self.miss.iter().position(|m| m.0 == e) {
                Some(i) => Some(self.miss.remove(i).1),
                None => self.pf.remove(&(layer, e)),
            };
            match unit {
                Some(u) if !self.landed.contains_key(&u) => pending.push((u, pairs)),
                _ => {
                    ready_pairs += pairs;
                    ready_n += 1;
                }
            }
        }
        if !pending.is_empty() {
            self.layer_reads = true;
        }
        if self.m.path != ReadPath::ServiceOverlap || !compute {
            let units: Vec<UnitId> = pending.iter().map(|p| p.0).collect();
            self.wait(&units, false);
            if compute {
                let pairs = ready_pairs + pending.iter().map(|p| p.1).sum::<u32>();
                let cost = self.group_cost(pairs);
                self.t += cost;
                self.acc.cpu_ready += cost;
            }
            return;
        }
        if ready_n > 0 {
            let cost = self.group_cost(ready_pairs);
            self.t += cost;
            self.acc.cpu_ready += cost;
        }
        while !pending.is_empty() {
            let units: Vec<UnitId> = pending.iter().map(|p| p.0).collect();
            self.wait(&units, true);
            let (now, rest): (Vec<_>, Vec<_>) = pending
                .into_iter()
                .partition(|p| self.landed.contains_key(&p.0));
            pending = rest;
            if now.is_empty() {
                // The disk ran dry with a unit open: cannot happen; do not hang.
                break;
            }
            let cost = self.group_cost(now.iter().map(|p| p.1).sum());
            self.t += cost;
            self.acc.cpu_landed += cost;
            self.advance(self.t);
        }
    }

    fn layer_end(&mut self, _layer: usize) {
        self.t += self.m.gpu_layer_s;
        self.step_end = self.t;
        if self.layer_reads || self.layer_wait > 0.0 {
            self.acc.span += (self.layer_span_end - self.layer_start).max(self.layer_wait);
        }
    }
}

/// Replays `trace` with `cfg` on the virtual clock of `model`. Errors on a model that cannot
/// run (no curve, a zero chunk, workers or sync depth).
pub fn replay_timed(
    trace: &Trace,
    cfg: &Config,
    model: &TimeModel,
) -> Result<TimedOutcome, String> {
    let mut d = Driver::new(&trace.header, model)?;
    let bytes = replay_timing(trace, cfg, &mut d, model.prefetch_depth != 1);
    d.finish_step();
    d.drain();
    let n = d.steps.len() as f64;
    let mean = |f: fn(&StepTiming) -> f64| d.steps.iter().map(f).sum::<f64>() / n.max(1.0);
    let time: f64 = d.steps.iter().map(|s| s.time_s).sum();
    let wait: f64 = d.steps.iter().map(|s| s.read_wait_s).sum();
    let span: f64 = d.steps.iter().map(|s| s.read_span_s).sum();
    let decode = DecodeTiming {
        tokens: d.steps.len() as u64,
        time_s: time,
        tokens_per_s: if time > 0.0 { n / time } else { 0.0 },
        step_s_mean: mean(|s| s.time_s),
        cpu_ready_s_mean: mean(|s| s.cpu_ready_s),
        read_wait_s_mean: mean(|s| s.read_wait_s),
        cpu_landed_s_mean: mean(|s| s.cpu_landed_s),
        read_span_s_mean: mean(|s| s.read_span_s),
        hidden_share: if span > 0.0 {
            (1.0 - wait / span).clamp(0.0, 1.0)
        } else {
            1.0
        },
        steps: std::mem::take(&mut d.steps),
    };
    Ok(TimedOutcome {
        bytes,
        path: model.path,
        prefetch_depth: model.prefetch_depth,
        decode,
        prefill_time_s: d.prefill_s,
        units_submitted: d.submitted,
        units_landed: d.landed.len() as u64,
        disk_bytes: d.served,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = include_str!("../../tests/data/time-model.json");

    /// The contract with the planner: its `TimeModel` parses and writes this same file.
    #[test]
    fn the_shared_sample_round_trips() {
        let m: TimeModel = serde_json::from_str(SAMPLE).unwrap();
        assert_eq!(m.path, ReadPath::ServiceOverlap);
        assert_eq!(m.curve.len(), 4);
        let again: serde_json::Value = serde_json::to_value(&m).unwrap();
        let want: serde_json::Value = serde_json::from_str(SAMPLE).unwrap();
        assert_eq!(again, want);
    }
}
