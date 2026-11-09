//! Offline replay of a routing trace through the engine's own slot caches (ADR 0050 (c)).
//!
//! Per layer step the simulator makes the engine's calls in the engine's order
//! (`Qwen3Moe` run loop under `MissPolicy::Cpu`): complete the layer's prefetch (unpin), VRAM
//! lookups of the routed experts that are resident (`plan_cpu`; the others get no VRAM slot),
//! the prefetch top-up of layer `il + 1` from this layer's prediction and of `il + 2` from its
//! depth-2 prediction (`prefetch_ahead`), then the RAM lookups of the CPU experts, hits first,
//! in chunks of the RAM slot count (`stage_ram`, `fill_ram`). Prefetched reads land before the
//! target layer is planned, and their bytes count as prefetch bytes. A hot-set marker rebuilds
//! the VRAM hot set from the routing counts and makes the new experts resident through the RAM
//! tier (`refresh_hot_set`, `make_resident`, `load_via_ram`).
//!
//! Policies for the RAM tier: [`Policy::Lru`] is the engine as built; [`Policy::Freq`] pins the
//! most routed experts of the layer on top of LRU; [`Policy::Belady`] is the offline optimum
//! without prefetch, an upper bound.

use super::prefetch_plan;
use super::trace::{LayerStep, Record, TraceHeader, TraceReader};
use crate::slots::{Lookup, SlotCache, Tier, hot_from_counts};
use serde::Serialize;
use std::collections::{HashMap, HashSet, VecDeque};
use std::io;
use std::path::Path;

/// Frequency pins are refreshed every this many decode steps (and at every hot-set marker).
pub const FREQ_REFRESH_STEPS: u64 = 16;

/// The RAM-tier policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Policy {
    /// As built: LRU, prefetch reads pinned while in flight.
    Lru,
    /// LRU plus frequency pins: the top `ceil(share * slots)` experts of the layer by routing
    /// count so far, leaving out those pinned in the VRAM tier (they never reach the RAM tier),
    /// refreshed every [`FREQ_REFRESH_STEPS`] decode steps and at each marker. The simulator
    /// unions them into the pins it gives `LayerSlots::set_hot_set`.
    Freq { share: f64 },
    /// Evict the resident expert whose next use is farthest, from the trace. No prefetch.
    Belady,
}

impl Policy {
    /// `lru`, `belady` or `freq:<share>`.
    pub fn parse(s: &str) -> Result<Policy, String> {
        match s {
            "lru" => Ok(Policy::Lru),
            "belady" => Ok(Policy::Belady),
            _ => {
                let share = s
                    .strip_prefix("freq:")
                    .and_then(|v| v.parse::<f64>().ok())
                    .filter(|v| *v > 0.0 && *v <= 1.0)
                    .ok_or_else(|| {
                        format!("policy {s:?}: expected lru, belady or freq:<share in (0, 1]>")
                    })?;
                Ok(Policy::Freq { share })
            }
        }
    }

    pub fn name(&self) -> String {
        match self {
            Policy::Lru => "lru".into(),
            Policy::Freq { share } => format!("freq:{share}"),
            Policy::Belady => "belady".into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    pub vram_slots: u32,
    pub ram_slots: u32,
    pub policy: Policy,
    /// Replay the recorded predictions as prefetch reads. Ignored by [`Policy::Belady`].
    pub prefetch: bool,
}

/// A trace held in memory, so one load serves many configurations.
pub struct Trace {
    pub header: TraceHeader,
    pub records: Vec<Record>,
}

impl Trace {
    pub fn load(path: &Path) -> io::Result<Trace> {
        let reader = TraceReader::open(path)?;
        let header = reader.header().clone();
        let records = reader.collect::<io::Result<Vec<_>>>()?;
        Ok(Trace { header, records })
    }
}

/// Counters of one phase (prefill or decode). The refresh at a marker counts in the phase of the
/// step before it.
#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct PhaseOutcome {
    pub steps: u64,
    pub tokens: u64,
    /// Routed (token, expert) pairs and the share that found their expert in VRAM.
    pub pairs: u64,
    pub vram_share: f64,
    /// RAM lookups of the CPU experts (and of the refresh), hits and misses.
    pub ram_lookups: u64,
    pub ram_hits: u64,
    pub ram_hit_rate: f64,
    /// Mean over layers of hits / (hits + misses) with the prefetch's own lookups counted as
    /// misses, as the exit harness reports `ram_hit_rate_mean`.
    pub ram_hit_rate_engine_mean: f64,
    pub disk_bytes_per_token: f64,
    pub sync_miss_bytes_per_token: f64,
    pub prefetch_bytes_per_token: f64,
    pub wasted_bytes_per_token: f64,
    pub prefetch_issued: u64,
    pub prefetch_used: u64,
    pub prefetch_wasted: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Outcome {
    pub vram_slots: u32,
    pub ram_slots: u32,
    pub policy: String,
    pub prefetch: bool,
    pub prefill: PhaseOutcome,
    pub decode: PhaseOutcome,
    /// Per layer, RAM-cache `(hits, misses)` over the whole replay including the prefetch's
    /// lookups: what `ExpertStats::layers[i].ram` reports.
    #[serde(skip)]
    pub layer_stats: Vec<(u64, u64)>,
}

#[derive(Default, Clone)]
struct Acc {
    steps: u64,
    tokens: u64,
    pairs: u64,
    vram_pairs: u64,
    hits: u64,
    misses: u64,
    sync_bytes: u64,
    prefetch_bytes: u64,
    wasted_bytes: u64,
    issued: u64,
    used: u64,
    wasted: u64,
    layer_hits: Vec<u64>,
    layer_misses: Vec<u64>,
}

const PREFILL: usize = 0;
const DECODE: usize = 1;

/// What the RAM lookup of one expert did.
enum Look {
    Hit,
    Miss {
        evicted: Option<u32>,
    },
    /// Every slot is pinned: the expert is read without being cached.
    Bypass,
}

/// Expert accesses per layer: the events at which each expert is looked up in the RAM tier.
type AccessLog = Vec<HashMap<u32, Vec<u32>>>;

/// Offline optimum for one tier: per layer a set of residents, evicting the one used farthest
/// ahead. Time is the event counter (one per layer step or marker).
struct Belady {
    cap: usize,
    layers: Vec<BeladyLayer>,
    times: AccessLog,
    ev: u32,
}

#[derive(Default)]
struct BeladyLayer {
    resident: Vec<u32>,
    /// Experts looked up in the current event: not needed again until their next access.
    served: HashSet<u32>,
    served_ev: u32,
}

impl Belady {
    fn next_use(&self, il: usize, e: u32) -> u32 {
        let l = &self.layers[il];
        let Some(ts) = self.times[il].get(&e) else {
            return u32::MAX;
        };
        let served = l.served_ev == self.ev && l.served.contains(&e);
        if !served && ts.binary_search(&self.ev).is_ok() {
            return self.ev;
        }
        let i = ts.partition_point(|&t| t <= self.ev);
        ts.get(i).copied().unwrap_or(u32::MAX)
    }

    fn lookup(&mut self, il: usize, e: u32, chunk: &[u32]) -> Look {
        let ev = self.ev;
        let l = &mut self.layers[il];
        if l.served_ev != ev {
            l.served.clear();
            l.served_ev = ev;
        }
        l.served.insert(e);
        if l.resident.contains(&e) {
            return Look::Hit;
        }
        if l.resident.len() < self.cap {
            l.resident.push(e);
            return Look::Miss { evicted: None };
        }
        let victim = (0..self.layers[il].resident.len())
            .filter(|&i| !chunk.contains(&self.layers[il].resident[i]))
            .max_by_key(|&i| {
                (
                    self.next_use(il, self.layers[il].resident[i]),
                    std::cmp::Reverse(i),
                )
            });
        match victim {
            Some(i) => {
                let old = std::mem::replace(&mut self.layers[il].resident[i], e);
                Look::Miss { evicted: Some(old) }
            }
            None => Look::Bypass,
        }
    }
}

enum Ram {
    Slots(SlotCache),
    Belady(Belady),
}

struct Sim<'a> {
    h: &'a TraceHeader,
    cfg: Config,
    prefetch_on: bool,
    ram_slots: usize,
    vram: SlotCache,
    ram: Ram,
    counts: Vec<Vec<u64>>,
    freq_pins: Vec<Vec<u32>>,
    /// Per layer and expert: read by a prefetch and not routed since.
    prefetched: Vec<Vec<bool>>,
    inflight: VecDeque<(usize, Vec<u32>)>,
    phase: usize,
    acc: [Acc; 2],
    cur_step: Option<u32>,
    decode_steps: u64,
    ev: u32,
    log: Option<AccessLog>,
    /// Count invariant violations ([`replay_checked`]).
    check: bool,
    violations: u64,
}

/// `items` with those `resident` holds first (stable): the engine's `hits_first`.
fn hits_first(items: &[u32], resident: impl Fn(u32) -> bool) -> Vec<u32> {
    let (hits, misses): (Vec<u32>, Vec<u32>) = items.iter().partition(|&&e| resident(e));
    hits.into_iter().chain(misses).collect()
}

impl<'a> Sim<'a> {
    fn new(trace: &'a Trace, cfg: Config, belady_times: Option<AccessLog>, log: bool) -> Self {
        let h = &trace.header;
        let layers = h.n_layer as usize;
        let ne = h.n_expert as usize;
        let ram = match (cfg.policy, belady_times) {
            (Policy::Belady, Some(times)) => Ram::Belady(Belady {
                cap: cfg.ram_slots as usize,
                layers: (0..layers).map(|_| BeladyLayer::default()).collect(),
                times,
                ev: 0,
            }),
            _ => Ram::Slots(SlotCache::new(
                Tier::Ram,
                h.n_layer,
                h.n_expert,
                cfg.ram_slots,
            )),
        };
        let acc = Acc {
            layer_hits: vec![0; layers],
            layer_misses: vec![0; layers],
            ..Acc::default()
        };
        Sim {
            h,
            cfg,
            prefetch_on: cfg.prefetch && cfg.policy != Policy::Belady,
            ram_slots: (cfg.ram_slots as usize).max(1),
            vram: SlotCache::new(Tier::Vram, h.n_layer, h.n_expert, cfg.vram_slots),
            ram,
            counts: h
                .profile
                .clone()
                .unwrap_or_else(|| vec![vec![0; ne]; layers]),
            freq_pins: vec![Vec::new(); layers],
            prefetched: vec![vec![false; ne]; layers],
            inflight: VecDeque::new(),
            phase: PREFILL,
            acc: [acc.clone(), acc],
            cur_step: None,
            decode_steps: 0,
            ev: 0,
            log: log.then(|| vec![HashMap::new(); layers]),
            check: false,
            violations: 0,
        }
    }

    fn ram_has(&self, il: usize, e: u32) -> bool {
        match &self.ram {
            Ram::Slots(c) => c.layer(il).slot_of(e).is_some(),
            Ram::Belady(b) => b.layers[il].resident.contains(&e),
        }
    }

    fn ram_lookup(&mut self, il: usize, e: u32, chunk: &[u32]) -> Look {
        let look = match &mut self.ram {
            Ram::Slots(c) => match c.layer_mut(il).lookup(e) {
                Lookup::Hit(_) => Look::Hit,
                Lookup::Miss { evicted, .. } => {
                    if self.check && evicted.is_some_and(|old| c.layer(il).is_pinned(old)) {
                        self.violations += 1;
                    }
                    Look::Miss { evicted }
                }
                Lookup::Bypass => Look::Bypass,
            },
            Ram::Belady(b) => b.lookup(il, e, chunk),
        };
        let acc = &mut self.acc[self.phase];
        match look {
            Look::Hit => acc.layer_hits[il] += 1,
            _ => acc.layer_misses[il] += 1,
        }
        look
    }

    fn set_ram_pins(&mut self, il: usize, pins: &[u32]) {
        if let Ram::Slots(c) = &mut self.ram {
            c.layer_mut(il).set_hot_set(pins);
        }
    }

    fn expert_bytes(&self, il: usize) -> u64 {
        self.h.expert_bytes[il]
    }

    /// A prefetched expert left its slot before it was routed.
    fn note_wasted(&mut self, il: usize, e: u32) {
        if std::mem::take(&mut self.prefetched[il][e as usize]) {
            let bytes = self.expert_bytes(il);
            let acc = &mut self.acc[self.phase];
            acc.wasted += 1;
            acc.wasted_bytes += bytes;
        }
    }

    /// `Streamer::fill_ram`: RAM hits first, then the misses; each miss is a synchronous read.
    fn fill_ram(&mut self, il: usize, experts: &[u32]) {
        let mut order: Vec<usize> = (0..experts.len()).collect();
        order.sort_by_key(|&i| !self.ram_has(il, experts[i]));
        for i in order {
            let e = experts[i];
            if let Some(log) = &mut self.log {
                let ts = log[il].entry(e).or_default();
                if ts.last() != Some(&self.ev) {
                    ts.push(self.ev);
                }
            }
            match self.ram_lookup(il, e, experts) {
                Look::Hit => {
                    if std::mem::take(&mut self.prefetched[il][e as usize]) {
                        self.acc[self.phase].used += 1;
                    }
                    self.acc[self.phase].hits += 1;
                }
                Look::Miss { evicted } => {
                    if let Some(old) = evicted {
                        self.note_wasted(il, old);
                    }
                    self.acc[self.phase].misses += 1;
                    self.acc[self.phase].sync_bytes += self.expert_bytes(il);
                }
                Look::Bypass => {
                    self.acc[self.phase].misses += 1;
                    self.acc[self.phase].sync_bytes += self.expert_bytes(il);
                }
            }
        }
    }

    /// `Streamer::start_prefetch` on the shared selection.
    fn start_prefetch(&mut self, il: usize, predicted: &[u16]) {
        if il >= self.h.n_layer as usize {
            return;
        }
        let Ram::Slots(cache) = &self.ram else {
            return;
        };
        let mut inflight: Vec<u32> = self
            .inflight
            .iter()
            .filter(|f| f.0 == il)
            .flat_map(|f| f.1.iter().copied())
            .collect();
        // Frequency pins hold slots like reads in flight do.
        inflight.extend(&self.freq_pins[il]);
        let predicted: Vec<i32> = predicted.iter().map(|&e| i32::from(e)).collect();
        let (mut pins, missing) = prefetch_plan(
            &predicted,
            self.h.n_expert,
            self.vram.layer(il),
            cache.layer(il),
            &inflight,
            self.ram_slots,
        );
        if missing.is_empty() {
            return;
        }
        pins.extend(&missing);
        // Pins beyond the slots would be dropped silently by `set_hot_set`.
        if self.check && pins.len() > self.ram_slots {
            self.violations += 1;
        }
        self.set_ram_pins(il, &pins);
        let mut read = Vec::new();
        for &e in &missing {
            match self.ram_lookup(il, e, &[]) {
                Look::Miss { evicted } => {
                    if let Some(old) = evicted {
                        self.note_wasted(il, old);
                    }
                    self.prefetched[il][e as usize] = true;
                    read.push(e);
                }
                // The engine fails the step here; the replay skips the read.
                Look::Hit | Look::Bypass => break,
            }
        }
        self.acc[self.phase].issued += read.len() as u64;
        if !read.is_empty() {
            self.inflight.push_back((il, read));
        }
    }

    /// `Streamer::finish_prefetch`: the reads of layer `il` land; their pins are released (the
    /// frequency pins stay).
    fn finish_prefetch(&mut self, il: usize) {
        let mut mine = 0u64;
        self.inflight.retain(|f| {
            if f.0 == il {
                mine += f.1.len() as u64;
                false
            } else {
                true
            }
        });
        if mine == 0 {
            return;
        }
        self.acc[self.phase].prefetch_bytes += mine * self.expert_bytes(il);
        let pins = self.freq_pins[il].clone();
        self.set_ram_pins(il, &pins);
    }

    fn settle(&mut self) {
        while let Some(il) = self.inflight.front().map(|f| f.0) {
            self.finish_prefetch(il);
        }
    }

    fn refresh_freq_pins(&mut self) {
        let Policy::Freq { share } = self.cfg.policy else {
            return;
        };
        let k = (share * f64::from(self.cfg.ram_slots)).ceil() as u32;
        for il in 0..self.h.n_layer as usize {
            let mut c = self.counts[il].clone();
            for (e, n) in c.iter_mut().enumerate() {
                if self.vram.layer(il).is_pinned(e as u32) {
                    *n = 0;
                }
            }
            self.freq_pins[il] = hot_from_counts(&c, k);
            let pins = self.freq_pins[il].clone();
            self.set_ram_pins(il, &pins);
        }
    }

    /// `Qwen3Moe::refresh_hot_set`.
    fn hot_refresh(&mut self) {
        self.settle();
        self.ev += 1;
        self.set_event();
        for il in 0..self.h.n_layer as usize {
            let active = self.vram.layer(il).active();
            let hot = hot_from_counts(&self.counts[il], active);
            let change = self.vram.layer_mut(il).set_hot_set(&hot);
            if change.missing.is_empty() {
                continue;
            }
            // make_resident: VRAM lookups (all misses), then the loads through the RAM tier in
            // hits-first groups of the RAM slot count.
            let mut loads = Vec::new();
            for &e in &change.missing {
                if matches!(self.vram.layer_mut(il).lookup(e), Lookup::Miss { .. }) {
                    loads.push(e);
                }
            }
            let ordered = hits_first(&loads, |e| self.ram_has(il, e));
            for group in ordered.chunks(self.ram_slots) {
                self.fill_ram(il, group);
            }
        }
        self.refresh_freq_pins();
    }

    fn set_event(&mut self) {
        if let Ram::Belady(b) = &mut self.ram {
            b.ev = self.ev;
        }
    }

    /// One layer step of the run loop.
    fn layer_step(&mut self, s: &LayerStep) {
        if self.cur_step != Some(s.step) {
            self.settle();
            self.cur_step = Some(s.step);
            self.phase = if s.n_tokens == 1 { DECODE } else { PREFILL };
            self.acc[self.phase].steps += 1;
            if self.phase == DECODE {
                if self.decode_steps.is_multiple_of(FREQ_REFRESH_STEPS) {
                    self.refresh_freq_pins();
                }
                self.decode_steps += 1;
            }
        }
        let il = s.layer as usize;
        self.ev += 1;
        self.set_event();
        self.finish_prefetch(il);
        if il == 0 {
            self.acc[self.phase].tokens += u64::from(s.n_tokens);
        }
        // plan_cpu: count the pairs, partition by VRAM residency, look up the resident ones.
        let mut seen = vec![false; self.h.n_expert as usize];
        let mut distinct = Vec::new();
        let mut vram_pairs = 0u64;
        for &e in &s.ids {
            self.counts[il][e as usize] += 1;
            if self.vram.layer(il).slot_of(u32::from(e)).is_some() {
                vram_pairs += 1;
            }
            if !std::mem::replace(&mut seen[e as usize], true) {
                distinct.push(u32::from(e));
            }
        }
        let acc = &mut self.acc[self.phase];
        acc.pairs += s.ids.len() as u64;
        acc.vram_pairs += vram_pairs;
        let (compute, rest): (Vec<u32>, Vec<u32>) = distinct
            .into_iter()
            .partition(|&e| self.vram.layer(il).slot_of(e).is_some());
        for e in compute {
            self.vram.layer_mut(il).lookup(e);
        }
        let order = hits_first(&rest, |e| self.ram_has(il, e));
        if self.prefetch_on {
            if let Some(p) = &s.predicted {
                self.start_prefetch(il + 1, p);
            }
            if let Some(p) = &s.predicted2 {
                self.start_prefetch(il + 2, p);
            }
        }
        for chunk in order.chunks(self.ram_slots) {
            self.fill_ram(il, chunk);
        }
        self.check_pins();
    }

    fn check_pins(&mut self) {
        if let (true, Ram::Slots(c)) = (self.check, &self.ram) {
            let bad = (0..c.layers())
                .filter(|&il| c.layer(il).pinned_count() > c.layer(il).active())
                .count();
            self.violations += bad as u64;
        }
    }

    fn outcome(mut self) -> (Outcome, Option<AccessLog>) {
        self.settle();
        let phase = |a: &Acc| {
            let tok = a.tokens.max(1) as f64;
            let rate = |h: u64, m: u64| h as f64 / (h + m).max(1) as f64;
            let layers = a.layer_hits.len().max(1) as f64;
            PhaseOutcome {
                steps: a.steps,
                tokens: a.tokens,
                pairs: a.pairs,
                vram_share: a.vram_pairs as f64 / a.pairs.max(1) as f64,
                ram_lookups: a.hits + a.misses,
                ram_hits: a.hits,
                ram_hit_rate: rate(a.hits, a.misses),
                ram_hit_rate_engine_mean: a
                    .layer_hits
                    .iter()
                    .zip(&a.layer_misses)
                    .map(|(&h, &m)| rate(h, m))
                    .sum::<f64>()
                    / layers,
                disk_bytes_per_token: (a.sync_bytes + a.prefetch_bytes) as f64 / tok,
                sync_miss_bytes_per_token: a.sync_bytes as f64 / tok,
                prefetch_bytes_per_token: a.prefetch_bytes as f64 / tok,
                wasted_bytes_per_token: a.wasted_bytes as f64 / tok,
                prefetch_issued: a.issued,
                prefetch_used: a.used,
                prefetch_wasted: a.wasted,
            }
        };
        let layer_stats = (0..self.h.n_layer as usize)
            .map(|il| {
                (
                    self.acc[PREFILL].layer_hits[il] + self.acc[DECODE].layer_hits[il],
                    self.acc[PREFILL].layer_misses[il] + self.acc[DECODE].layer_misses[il],
                )
            })
            .collect();
        let out = Outcome {
            vram_slots: self.cfg.vram_slots,
            ram_slots: self.cfg.ram_slots,
            policy: self.cfg.policy.name(),
            prefetch: self.prefetch_on,
            prefill: phase(&self.acc[PREFILL]),
            decode: phase(&self.acc[DECODE]),
            layer_stats,
        };
        (out, self.log)
    }
}

fn run(
    trace: &Trace,
    cfg: Config,
    times: Option<AccessLog>,
    log: bool,
    check: bool,
) -> (Outcome, Option<AccessLog>, u64) {
    let mut sim = Sim::new(trace, cfg, times, log);
    sim.check = check;
    for r in &trace.records {
        match r {
            Record::Step(s) => sim.layer_step(s),
            Record::HotRefresh { .. } => sim.hot_refresh(),
        }
    }
    let violations = sim.violations;
    let (out, log) = sim.outcome();
    (out, log, violations)
}

/// Replays `trace` with one configuration. [`Policy::Belady`] first replays LRU without
/// prefetch to learn when each expert is looked up (the lookup set of a step does not depend on
/// the RAM policy).
pub fn replay(trace: &Trace, cfg: &Config) -> Outcome {
    if cfg.policy == Policy::Belady {
        let probe = Config {
            policy: Policy::Lru,
            prefetch: false,
            ..*cfg
        };
        let (_, log, _) = run(trace, probe, None, true, false);
        return run(trace, *cfg, log, false, false).0;
    }
    run(trace, *cfg, None, false, false).0
}

/// [`replay`] for an LRU or frequency policy that also counts violations of the pin invariants:
/// more pins than active slots, pins the slot table had to drop, a pinned expert evicted.
pub fn replay_checked(trace: &Trace, cfg: &Config) -> (Outcome, u64) {
    let (out, _, violations) = run(trace, *cfg, None, false, true);
    (out, violations)
}

/// Every combination of the lists; prefetch on and off for the policies that have both.
pub fn replay_matrix(
    trace: &Trace,
    vram: &[u32],
    ram: &[u32],
    policies: &[Policy],
    prefetch: &[bool],
) -> Vec<Outcome> {
    let mut out = Vec::new();
    for &vram_slots in vram {
        for &ram_slots in ram {
            for &policy in policies {
                for &pf in prefetch {
                    if policy == Policy::Belady && pf {
                        continue;
                    }
                    let cfg = Config {
                        vram_slots,
                        ram_slots,
                        policy,
                        prefetch: pf,
                    };
                    out.push(replay(trace, &cfg));
                }
            }
        }
    }
    out
}
