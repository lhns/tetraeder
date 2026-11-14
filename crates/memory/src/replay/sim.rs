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
//! without prefetch, an upper bound. The `Global*` policies share one byte budget across layers
//! (ADR 0050 (c) addendum 2026-11-14).

use super::prefetch_plan_with;
use super::trace::{LayerStep, Record, TraceHeader, TraceReader};
pub use crate::slots::FREQ_REFRESH_STEPS;
use crate::slots::{
    GlobalLookup, GlobalSlots, Lookup, RamPolicy, SlotCache, Tier, freq_pins, hot_from_counts,
};
use serde::Serialize;
use std::collections::{HashMap, HashSet, VecDeque};
use std::io;
use std::path::Path;

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
    /// One budget of `ram_slots x sum(expert_bytes)` across layers, LRU over every
    /// (layer, expert); pinned experts and the current chunk are never evicted.
    GlobalLru,
    /// [`Policy::GlobalLru`] plus, in decode, the most routed experts across all layers (outside
    /// the VRAM hot set) pinned up to `share` of the budget, always leaving room for one decode
    /// step and its reads in flight (`4 x n_used` experts); refreshed as [`Policy::Freq`].
    GlobalFreq { share: f64 },
    /// [`Policy::Belady`] over the shared budget. No prefetch.
    GlobalBelady,
}

fn share_of(s: &str, prefix: &str) -> Option<f64> {
    s.strip_prefix(prefix)
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| *v > 0.0 && *v <= 1.0)
}

impl Policy {
    /// `lru`, `belady`, `freq:<share>`, `glru`, `gbelady` or `gfreq:<share>`.
    pub fn parse(s: &str) -> Result<Policy, String> {
        match s {
            "lru" => Ok(Policy::Lru),
            "belady" => Ok(Policy::Belady),
            "glru" => Ok(Policy::GlobalLru),
            "gbelady" => Ok(Policy::GlobalBelady),
            _ => share_of(s, "freq:")
                .map(|share| Policy::Freq { share })
                .or_else(|| share_of(s, "gfreq:").map(|share| Policy::GlobalFreq { share }))
                .ok_or_else(|| {
                    format!(
                        "policy {s:?}: expected lru, belady, freq:<share in (0, 1]>, glru, gbelady or gfreq:<share in (0, 1]>"
                    )
                }),
        }
    }

    pub fn name(&self) -> String {
        match self {
            Policy::Lru => "lru".into(),
            Policy::Freq { share } => format!("freq:{share}"),
            Policy::Belady => "belady".into(),
            Policy::GlobalLru => "glru".into(),
            Policy::GlobalFreq { share } => format!("gfreq:{share}"),
            Policy::GlobalBelady => "gbelady".into(),
        }
    }

    /// An offline optimum: replayed without prefetch.
    pub fn is_belady(&self) -> bool {
        matches!(self, Policy::Belady | Policy::GlobalBelady)
    }

    /// One budget across layers.
    pub fn is_global(&self) -> bool {
        matches!(
            self,
            Policy::GlobalLru | Policy::GlobalFreq { .. } | Policy::GlobalBelady
        )
    }
}

/// The VRAM (compute) tier policy, ADR 0050 (c) addendum 2026-11-12. The engine as built is
/// [`VramPolicy::Static`]; the others ask whether admitting experts during decode would raise
/// the share of pairs served from VRAM enough to matter. Admissions copy an expert from the
/// host to VRAM; the replay counts those bytes (`vram_admit_bytes_per_token`) and models no
/// RAM-tier traffic or time for them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VramPolicy {
    /// As built: the hot set is pinned at the marker and never changes during decode.
    Static,
    /// Every decode miss is admitted to the compute tier, evicting the least recently used
    /// unpinned expert. `pin` is the share of the slots that keep the marker's hot set pinned
    /// (floor of `pin * slots`); the rest of the hot set is loaded unpinned as a warm start.
    Lru { pin: f64 },
    /// The per-layer top-N by routing counts (N = the slots) is pinned again every
    /// [`FREQ_REFRESH_STEPS`] decode steps; newcomers are copied in.
    Freq,
    /// A static hot set like [`VramPolicy::Static`], but picked from the routing counts of the
    /// whole prefill (all ubatches) when decode starts; the marker after the first ubatch is
    /// ignored.
    Prefill,
}

impl VramPolicy {
    /// `static`, `prefill`, `lru`, `lru:<pin share in [0, 1)>` or `freq`.
    pub fn parse(s: &str) -> Result<VramPolicy, String> {
        match s {
            "static" => Ok(VramPolicy::Static),
            "lru" => Ok(VramPolicy::Lru { pin: 0.0 }),
            "freq" => Ok(VramPolicy::Freq),
            "prefill" => Ok(VramPolicy::Prefill),
            _ => {
                let pin = s
                    .strip_prefix("lru:")
                    .and_then(|v| v.parse::<f64>().ok())
                    .filter(|v| (0.0..1.0).contains(v))
                    .ok_or_else(|| {
                        format!(
                            "vram policy {s:?}: expected static, prefill, lru, lru:<pin share in [0, 1)> or freq"
                        )
                    })?;
                Ok(VramPolicy::Lru { pin })
            }
        }
    }

    pub fn name(&self) -> String {
        match self {
            VramPolicy::Static => "static".into(),
            VramPolicy::Lru { pin } if *pin == 0.0 => "lru".into(),
            VramPolicy::Lru { pin } => format!("lru:{pin}"),
            VramPolicy::Freq => "freq".into(),
            VramPolicy::Prefill => "prefill".into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    pub vram_slots: u32,
    pub ram_slots: u32,
    pub policy: Policy,
    /// The VRAM-tier policy; [`VramPolicy::Static`] is the engine as built.
    pub vram_policy: VramPolicy,
    /// Replay the recorded predictions as prefetch reads. Ignored by the Belady policies.
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
    /// Layer steps in which every routed expert of every token was in VRAM, over layer steps
    /// (what a step without a CPU hop is; the planner's switch count, ADR 0056).
    pub full_hit_share: f64,
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
    /// Bytes copied to VRAM per token by admissions during decode (`lru`, `freq`); absent for
    /// `static`, which never admits.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_admit_bytes_per_token: Option<f64>,
    pub prefetch_issued: u64,
    pub prefetch_used: u64,
    pub prefetch_wasted: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Outcome {
    pub vram_slots: u32,
    pub ram_slots: u32,
    pub policy: String,
    /// The VRAM-tier policy name; absent for `static`, so those outcomes stay as before.
    #[serde(skip_serializing_if = "is_static_name")]
    pub vram_policy: String,
    pub prefetch: bool,
    pub prefill: PhaseOutcome,
    pub decode: PhaseOutcome,
    /// Global policies: experts per layer in the RAM tier at the end of the replay.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ram_layer_residents: Option<Spread>,
    /// Global policies: RAM lookups that found no evictable room and read without caching, both
    /// phases; absent at zero.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ram_bypasses: Option<u64>,
    /// Per layer, RAM-cache `(hits, misses)` over the whole replay including the prefetch's
    /// lookups: what `ExpertStats::layers[i].ram` reports.
    #[serde(skip)]
    pub layer_stats: Vec<(u64, u64)>,
    /// Disk bytes (synchronous misses and landed prefetches) of each decode step, in order; the
    /// bench tool folds them into `decode_windows`.
    #[serde(skip)]
    pub decode_step_bytes: Vec<u64>,
}

/// Min, mean and max over layers.
#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub struct Spread {
    pub min: u32,
    pub mean: f64,
    pub max: u32,
}

impl Spread {
    fn of(v: impl IntoIterator<Item = u32>) -> Option<Spread> {
        let v: Vec<u32> = v.into_iter().collect();
        Some(Spread {
            min: *v.iter().min()?,
            mean: f64::from(v.iter().sum::<u32>()) / v.len() as f64,
            max: *v.iter().max()?,
        })
    }
}

fn is_static_name(s: &str) -> bool {
    s == "static"
}

#[derive(Default, Clone)]
struct Acc {
    admit_bytes: u64,
    bypasses: u64,
    steps: u64,
    tokens: u64,
    pairs: u64,
    vram_pairs: u64,
    layer_steps: u64,
    full_steps: u64,
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
    /// `evicted`: the `(layer, expert)` that left (another layer only under a global policy).
    Miss {
        evicted: Vec<(usize, u32)>,
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
    /// [`Policy::GlobalBelady`]: one byte budget instead of `cap` per layer.
    global: Option<Bytes>,
}

struct Bytes {
    per_layer: Vec<u64>,
    budget: u64,
    used: u64,
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
        if self.global.is_some() {
            return self.lookup_global(il, e, chunk);
        }
        if l.resident.len() < self.cap {
            l.resident.push(e);
            return Look::Miss {
                evicted: Vec::new(),
            };
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
                Look::Miss {
                    evicted: vec![(il, old)],
                }
            }
            None => Look::Bypass,
        }
    }

    /// A miss under the shared budget: evict the residents of any layer used farthest ahead
    /// (never the chunk) until the expert fits; a bypass evicts nothing.
    fn lookup_global(&mut self, il: usize, e: u32, chunk: &[u32]) -> Look {
        let g = self.global.as_ref().expect("global budget");
        let need = g.per_layer[il];
        let protected = self.layers[il]
            .resident
            .iter()
            .filter(|x| chunk.contains(x))
            .count() as u64
            * need;
        // Evicting everything but the chunk leaves `protected` bytes.
        if protected + need > g.budget {
            return Look::Bypass;
        }
        let mut evicted = Vec::new();
        while {
            let g = self.global.as_ref().expect("global budget");
            g.used + need > g.budget
        } {
            let (l, i) = (0..self.layers.len())
                .flat_map(|l| (0..self.layers[l].resident.len()).map(move |i| (l, i)))
                .filter(|&(l, i)| !(l == il && chunk.contains(&self.layers[l].resident[i])))
                .max_by_key(|&(l, i)| {
                    (
                        self.next_use(l, self.layers[l].resident[i]),
                        std::cmp::Reverse((l, i)),
                    )
                })
                .expect("enough evictable bytes");
            let old = self.layers[l].resident.swap_remove(i);
            let g = self.global.as_mut().expect("global budget");
            g.used -= g.per_layer[l];
            evicted.push((l, old));
        }
        self.layers[il].resident.push(e);
        let g = self.global.as_mut().expect("global budget");
        g.used += need;
        Look::Miss { evicted }
    }
}

enum Ram {
    Slots(SlotCache),
    Belady(Belady),
    Global(GlobalSlots),
}

/// Observer of the replay's calls, in the engine's order and with their sizes, for the virtual
/// clock of [`super::timed`] (ADR 0056). The replay's counters never depend on it: a replay
/// without one ([`replay`]) counts bytes only.
pub trait Timing {
    /// A layer step begins (before the previous step's stranded reads are settled).
    fn layer_begin(&mut self, step: u32, layer: usize, decode: bool);
    /// Reads of `experts` of `layer` (`bytes` each) are queued as prefetches.
    fn prefetch_submit(&mut self, layer: usize, experts: &[u32], bytes: u64);
    /// The prefetch reads of `layer` are completed before the layer is planned (`settle`: also
    /// at a step boundary and before a hot-set refresh, whatever the read path).
    fn prefetch_land(&mut self, layer: usize, settle: bool);
    /// One expert of `layer` is read synchronously because the RAM tier missed it.
    fn miss_submit(&mut self, layer: usize, expert: u32, bytes: u64);
    /// A CPU group of `layer`: `(expert, routed pairs)`. `compute` is false for the loads of a
    /// hot-set refresh, which read but compute nothing.
    fn cpu_group(&mut self, layer: usize, experts: &[(u32, u32)], compute: bool);
    /// The layer step is done (the GPU part runs here).
    fn layer_end(&mut self, layer: usize);
}

struct Sim<'a, 't> {
    h: &'a TraceHeader,
    cfg: Config,
    prefetch_on: bool,
    ram_slots: usize,
    vram: SlotCache,
    ram: Ram,
    counts: Vec<Vec<u64>>,
    freq_pins: Vec<Vec<u32>>,
    /// The frequency pins are held in decode steps only: a prefill step may route more distinct
    /// experts than the unpinned slots hold, and the engine cannot serve a miss without a slot.
    pins_on: bool,
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
    timing: Option<&'t mut dyn Timing>,
    /// Also prefetch two layers ahead from the recorded depth-2 predictions.
    depth2: bool,
    /// Decode disk bytes so far at the start of every decode step.
    decode_marks: Vec<u64>,
}

/// `items` with those `resident` holds first (stable): the engine's `hits_first`.
fn hits_first(items: &[u32], resident: impl Fn(u32) -> bool) -> Vec<u32> {
    let (hits, misses): (Vec<u32>, Vec<u32>) = items.iter().partition(|&&e| resident(e));
    hits.into_iter().chain(misses).collect()
}

impl<'a, 't> Sim<'a, 't> {
    fn new(trace: &'a Trace, cfg: Config, belady_times: Option<AccessLog>, log: bool) -> Self {
        let h = &trace.header;
        let layers = h.n_layer as usize;
        let ne = h.n_expert as usize;
        let budget = u64::from(cfg.ram_slots) * h.expert_bytes.iter().sum::<u64>();
        let ram = match (cfg.policy, belady_times) {
            (Policy::Belady | Policy::GlobalBelady, Some(times)) => Ram::Belady(Belady {
                cap: cfg.ram_slots as usize,
                layers: (0..layers).map(|_| BeladyLayer::default()).collect(),
                times,
                ev: 0,
                global: (cfg.policy == Policy::GlobalBelady).then(|| Bytes {
                    per_layer: h.expert_bytes.clone(),
                    budget,
                    used: 0,
                }),
            }),
            (Policy::GlobalLru | Policy::GlobalFreq { .. }, _) => {
                Ram::Global(GlobalSlots::new(&h.expert_bytes, h.n_expert, budget))
            }
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
            prefetch_on: cfg.prefetch && !cfg.policy.is_belady(),
            ram_slots: (cfg.ram_slots as usize).max(1),
            vram: SlotCache::new(Tier::Vram, h.n_layer, h.n_expert, cfg.vram_slots),
            ram,
            counts: h
                .profile
                .clone()
                .unwrap_or_else(|| vec![vec![0; ne]; layers]),
            freq_pins: vec![Vec::new(); layers],
            pins_on: false,
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
            timing: None,
            depth2: true,
            decode_marks: Vec::new(),
        }
    }

    fn ram_has(&self, il: usize, e: u32) -> bool {
        match &self.ram {
            Ram::Slots(c) => c.layer(il).slot_of(e).is_some(),
            Ram::Belady(b) => b.layers[il].resident.contains(&e),
            Ram::Global(g) => g.contains(il as u32, e),
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
                    Look::Miss {
                        evicted: evicted.map(|old| (il, old)).into_iter().collect(),
                    }
                }
                Lookup::Bypass => Look::Bypass,
            },
            Ram::Belady(b) => b.lookup(il, e, chunk),
            Ram::Global(g) => match g.lookup_except(il as u32, e, chunk) {
                GlobalLookup::Hit => Look::Hit,
                GlobalLookup::Miss { evicted } => {
                    if self.check {
                        let bad = |&&(l, x): &&(u32, u32)| {
                            g.is_pinned(l, x) || (l == il as u32 && chunk.contains(&x))
                        };
                        self.violations += evicted.iter().filter(bad).count() as u64;
                    }
                    Look::Miss {
                        evicted: evicted.into_iter().map(|(l, x)| (l as usize, x)).collect(),
                    }
                }
                GlobalLookup::Bypass => Look::Bypass,
            },
        };
        let acc = &mut self.acc[self.phase];
        match look {
            Look::Hit => acc.layer_hits[il] += 1,
            _ => acc.layer_misses[il] += 1,
        }
        look
    }

    fn set_ram_pins(&mut self, il: usize, pins: &[u32]) {
        match &mut self.ram {
            Ram::Slots(c) => {
                c.layer_mut(il).set_hot_set(pins);
            }
            Ram::Global(g) => g.set_pins(il as u32, pins),
            Ram::Belady(_) => {}
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

    fn note_miss(&mut self, il: usize, e: u32) {
        let bytes = self.expert_bytes(il);
        if let Some(t) = self.timing.as_mut() {
            t.miss_submit(il, e, bytes);
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
                    for (l, old) in evicted {
                        self.note_wasted(l, old);
                    }
                    self.acc[self.phase].misses += 1;
                    self.acc[self.phase].sync_bytes += self.expert_bytes(il);
                    self.note_miss(il, e);
                }
                Look::Bypass => {
                    self.acc[self.phase].bypasses += 1;
                    self.acc[self.phase].misses += 1;
                    self.acc[self.phase].sync_bytes += self.expert_bytes(il);
                    self.note_miss(il, e);
                }
            }
        }
    }

    /// `Streamer::start_prefetch` on the shared selection.
    fn start_prefetch(&mut self, il: usize, predicted: &[u16]) {
        if il >= self.h.n_layer as usize {
            return;
        }
        if matches!(self.ram, Ram::Belady(_)) {
            return;
        }
        let global = matches!(self.ram, Ram::Global(_));
        let mut inflight: Vec<u32> = self
            .inflight
            .iter()
            .filter(|f| f.0 == il)
            .flat_map(|f| f.1.iter().copied())
            .collect();
        // Frequency pins hold slots like reads in flight do; in a shared pool they hold bytes of
        // the whole budget, not of this layer's read limit.
        if !global {
            inflight.extend(self.standing(il));
        }
        let predicted: Vec<i32> = predicted.iter().map(|&e| i32::from(e)).collect();
        let (mut pins, missing) = prefetch_plan_with(
            &predicted,
            self.h.n_expert,
            self.vram.layer(il),
            |e| self.ram_has(il, e),
            &inflight,
            self.ram_slots,
        );
        if missing.is_empty() {
            return;
        }
        if global {
            for &e in self.standing(il) {
                if !pins.contains(&e) {
                    pins.push(e);
                }
            }
        }
        pins.extend(&missing);
        // Pins beyond the slots would be dropped silently by `set_hot_set`.
        if self.check && !global && pins.len() > self.ram_slots {
            self.violations += 1;
        }
        self.set_ram_pins(il, &pins);
        let mut read = Vec::new();
        for &e in &missing {
            match self.ram_lookup(il, e, &[]) {
                Look::Miss { evicted } => {
                    for (l, old) in evicted {
                        self.note_wasted(l, old);
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
            let bytes = self.expert_bytes(il);
            if let Some(t) = self.timing.as_mut() {
                t.prefetch_submit(il, &read, bytes);
            }
            self.inflight.push_back((il, read));
        }
    }

    /// `Streamer::finish_prefetch`: the reads of layer `il` land; their pins are released (the
    /// frequency pins stay).
    fn finish_prefetch(&mut self, il: usize, settle: bool) {
        if let Some(t) = self.timing.as_mut() {
            t.prefetch_land(il, settle);
        }
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
        let pins = self.standing(il).to_vec();
        self.set_ram_pins(il, &pins);
    }

    /// The frequency pins in force for layer `il`.
    fn standing(&self, il: usize) -> &[u32] {
        if self.pins_on {
            &self.freq_pins[il]
        } else {
            &[]
        }
    }

    fn settle(&mut self) {
        while let Some(il) = self.inflight.front().map(|f| f.0) {
            self.finish_prefetch(il, true);
        }
    }

    /// Puts the pins in force on the slot tables (all of them, as a step starts).
    fn apply_freq_pins(&mut self) {
        if !matches!(
            self.cfg.policy,
            Policy::Freq { .. } | Policy::GlobalFreq { .. }
        ) {
            return;
        }
        for il in 0..self.h.n_layer as usize {
            let pins = self.standing(il).to_vec();
            self.set_ram_pins(il, &pins);
        }
    }

    fn refresh_freq_pins(&mut self) {
        if let Policy::GlobalFreq { share } = self.cfg.policy {
            self.refresh_global_freq_pins(share);
            return;
        }
        let Policy::Freq { share } = self.cfg.policy else {
            return;
        };
        let k = RamPolicy::Freq { share }.pin_count(self.cfg.ram_slots);
        for il in 0..self.h.n_layer as usize {
            self.freq_pins[il] = freq_pins(&self.counts[il], self.vram.layer(il), k);
            let pins = self.standing(il).to_vec();
            self.set_ram_pins(il, &pins);
        }
    }

    /// [`Policy::GlobalFreq`]: the most routed experts of all layers, outside the VRAM hot set,
    /// until `share` of the budget, less room for a decode step and its reads in flight: the
    /// chunk, two reads for the next layer (depth 2 from the layer before, depth 1 from this one)
    /// and one for the layer after, `n_used` experts each.
    fn refresh_global_freq_pins(&mut self, share: f64) {
        let Ram::Global(g) = &self.ram else {
            return;
        };
        let bytes = &self.h.expert_bytes;
        let largest = bytes.iter().copied().max().unwrap_or(0);
        let reserve = 4 * u64::from(self.h.n_used) * largest;
        let cap = ((share * g.budget() as f64) as u64).min(g.budget().saturating_sub(reserve));
        let mut ranked: Vec<(u64, usize, u32)> = Vec::new();
        for (il, counts) in self.counts.iter().enumerate() {
            for (e, &n) in counts.iter().enumerate() {
                if n > 0 && !self.vram.layer(il).is_pinned(e as u32) {
                    ranked.push((n, il, e as u32));
                }
            }
        }
        ranked.sort_by_key(|&(n, il, e)| (std::cmp::Reverse(n), il, e));
        for p in &mut self.freq_pins {
            p.clear();
        }
        let mut sum = 0u64;
        for (_, il, e) in ranked {
            if sum + bytes[il] > cap {
                break;
            }
            sum += bytes[il];
            self.freq_pins[il].push(e);
        }
        for il in 0..self.h.n_layer as usize {
            let pins = self.standing(il).to_vec();
            self.set_ram_pins(il, &pins);
        }
    }

    /// `Qwen3Moe::refresh_hot_set`.
    fn hot_refresh(&mut self) {
        if self.cfg.vram_policy != VramPolicy::Prefill {
            self.build_hot_set();
        }
    }

    fn build_hot_set(&mut self) {
        self.settle();
        self.ev += 1;
        self.set_event();
        for il in 0..self.h.n_layer as usize {
            let active = self.vram.layer(il).active();
            let hot = hot_from_counts(&self.counts[il], active);
            // `lru` pins only part of the hot set; the rest is a warm start.
            let pins = match self.cfg.vram_policy {
                VramPolicy::Lru { pin } => {
                    ((f64::from(active) * pin).floor() as usize).min(hot.len())
                }
                _ => hot.len(),
            };
            let change = self.vram.layer_mut(il).set_hot_set(&hot[..pins]);
            let wanted: Vec<u32> = if pins == hot.len() {
                change.missing
            } else {
                hot.iter()
                    .copied()
                    .filter(|&e| self.vram.layer(il).slot_of(e).is_none())
                    .collect()
            };
            if wanted.is_empty() {
                continue;
            }
            // make_resident: VRAM lookups (all misses), then the loads through the RAM tier in
            // hits-first groups of the RAM slot count.
            let mut loads = Vec::new();
            for &e in &wanted {
                if matches!(self.vram.layer_mut(il).lookup(e), Lookup::Miss { .. }) {
                    loads.push(e);
                }
            }
            let ordered = hits_first(&loads, |e| self.ram_has(il, e));
            for group in ordered.chunks(self.ram_slots) {
                self.fill_ram(il, group);
                if let Some(t) = self.timing.as_mut() {
                    let g: Vec<(u32, u32)> = group.iter().map(|&e| (e, 0)).collect();
                    t.cpu_group(il, &g, false);
                }
            }
        }
        self.refresh_freq_pins();
    }

    /// `VramPolicy::Freq`: re-pick the top experts of every layer by routing count; newcomers
    /// are copied into free or unpinned slots, and the bytes counted.
    fn refresh_vram_freq(&mut self) {
        for il in 0..self.h.n_layer as usize {
            let active = self.vram.layer(il).active();
            let hot = hot_from_counts(&self.counts[il], active);
            let change = self.vram.layer_mut(il).set_hot_set(&hot);
            for e in change.missing {
                if matches!(self.vram.layer_mut(il).lookup(e), Lookup::Miss { .. }) {
                    let bytes = self.expert_bytes(il);
                    self.acc[DECODE].admit_bytes += bytes;
                }
            }
        }
    }

    fn decode_bytes(&self) -> u64 {
        self.acc[DECODE].sync_bytes + self.acc[DECODE].prefetch_bytes
    }

    fn set_event(&mut self) {
        if let Ram::Belady(b) = &mut self.ram {
            b.ev = self.ev;
        }
    }

    /// One layer step of the run loop.
    fn layer_step(&mut self, s: &LayerStep) {
        if let Some(t) = self.timing.as_mut() {
            t.layer_begin(s.step, s.layer as usize, s.n_tokens == 1);
        }
        if self.cur_step != Some(s.step) {
            self.settle();
            self.cur_step = Some(s.step);
            let phase = if s.n_tokens == 1 { DECODE } else { PREFILL };
            // `prefill`: the hot set from the counts of the whole prefill, as decode starts (its
            // loads count in the prefill phase, like a marker's).
            if phase == DECODE
                && self.decode_steps == 0
                && self.cfg.vram_policy == VramPolicy::Prefill
            {
                self.build_hot_set();
            }
            self.phase = phase;
            if phase == DECODE {
                self.decode_marks.push(self.decode_bytes());
            }
            self.acc[self.phase].steps += 1;
            self.pins_on = self.phase == DECODE;
            if self.phase == DECODE {
                if self.decode_steps.is_multiple_of(FREQ_REFRESH_STEPS) {
                    self.refresh_freq_pins();
                    if self.cfg.vram_policy == VramPolicy::Freq {
                        self.refresh_vram_freq();
                    }
                }
                self.decode_steps += 1;
            }
            self.apply_freq_pins();
        }
        let il = s.layer as usize;
        self.ev += 1;
        self.set_event();
        self.finish_prefetch(il, false);
        if il == 0 {
            self.acc[self.phase].tokens += u64::from(s.n_tokens);
        }
        // plan_cpu: count the pairs, partition by VRAM residency, look up the resident ones.
        let mut seen = vec![false; self.h.n_expert as usize];
        let mut pairs_of = vec![0u32; self.h.n_expert as usize];
        let mut distinct = Vec::new();
        let mut vram_pairs = 0u64;
        for &e in &s.ids {
            self.counts[il][e as usize] += 1;
            pairs_of[e as usize] += 1;
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
        acc.layer_steps += 1;
        acc.full_steps += u64::from(vram_pairs == s.ids.len() as u64);
        let (compute, rest): (Vec<u32>, Vec<u32>) = distinct
            .into_iter()
            .partition(|&e| self.vram.layer(il).slot_of(e).is_some());
        for e in compute {
            self.vram.layer_mut(il).lookup(e);
        }
        let order = hits_first(&rest, |e| self.ram_has(il, e));
        // `lru`: the CPU experts of a decode step are admitted once the step has used them.
        let lru = matches!(self.cfg.vram_policy, VramPolicy::Lru { .. });
        let admit = if self.phase == DECODE && lru {
            rest.clone()
        } else {
            Vec::new()
        };
        if self.prefetch_on {
            // In a shared pool the reads for later layers could evict this step's RAM hits; they
            // stay pinned while the reads are placed (per layer the tables are apart).
            let global = matches!(self.ram, Ram::Global(_));
            let hits: Vec<u32> = rest
                .iter()
                .copied()
                .filter(|&e| self.ram_has(il, e))
                .collect();
            if global {
                let mut pins = self.standing(il).to_vec();
                pins.extend(&hits);
                self.set_ram_pins(il, &pins);
            }
            if let Some(p) = &s.predicted {
                self.start_prefetch(il + 1, p);
            }
            if let (Some(p), true) = (&s.predicted2, self.depth2) {
                self.start_prefetch(il + 2, p);
            }
            if global {
                let pins = self.standing(il).to_vec();
                self.set_ram_pins(il, &pins);
            }
            if self.check {
                self.violations += hits.iter().filter(|&&e| !self.ram_has(il, e)).count() as u64;
            }
        }
        for chunk in order.chunks(self.ram_slots) {
            self.fill_ram(il, chunk);
            if let Some(t) = self.timing.as_mut() {
                let g: Vec<(u32, u32)> = chunk.iter().map(|&e| (e, pairs_of[e as usize])).collect();
                t.cpu_group(il, &g, true);
            }
        }
        for e in admit {
            if matches!(self.vram.layer_mut(il).lookup(e), Lookup::Miss { .. }) {
                let bytes = self.expert_bytes(il);
                self.acc[DECODE].admit_bytes += bytes;
            }
        }
        if let Some(t) = self.timing.as_mut() {
            t.layer_end(il);
        }
        self.check_pins();
    }

    fn check_pins(&mut self) {
        if self.check {
            let bad = (0..self.vram.layers())
                .filter(|&il| self.vram.layer(il).pinned_count() > self.vram.layer(il).active())
                .count();
            self.violations += bad as u64;
        }
        if let (true, Ram::Slots(c)) = (self.check, &self.ram) {
            let bad = (0..c.layers())
                .filter(|&il| c.layer(il).pinned_count() > c.layer(il).active())
                .count();
            self.violations += bad as u64;
        }
        if let (true, Ram::Global(g)) = (self.check, &self.ram) {
            self.violations += u64::from(g.used() > g.budget());
        }
    }

    fn outcome(mut self) -> (Outcome, Option<AccessLog>) {
        self.settle();
        let end = self.decode_bytes();
        let decode_step_bytes = self
            .decode_marks
            .iter()
            .zip(self.decode_marks.iter().skip(1).chain([&end]))
            .map(|(a, b)| b - a)
            .collect();
        let layers = self.h.n_layer as usize;
        let ram_layer_residents = match &self.ram {
            _ if !self.cfg.policy.is_global() => None,
            Ram::Global(g) => Spread::of((0..layers).map(|il| g.residents(il as u32))),
            Ram::Belady(b) => Spread::of(b.layers.iter().map(|l| l.resident.len() as u32)),
            Ram::Slots(_) => None,
        };
        let adm = self.cfg.vram_policy != VramPolicy::Static;
        let phase = |a: &Acc| {
            let tok = a.tokens.max(1) as f64;
            let rate = |h: u64, m: u64| h as f64 / (h + m).max(1) as f64;
            let layers = a.layer_hits.len().max(1) as f64;
            PhaseOutcome {
                steps: a.steps,
                tokens: a.tokens,
                pairs: a.pairs,
                vram_share: a.vram_pairs as f64 / a.pairs.max(1) as f64,
                full_hit_share: a.full_steps as f64 / a.layer_steps.max(1) as f64,
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
                vram_admit_bytes_per_token: adm.then(|| a.admit_bytes as f64 / tok),
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
            vram_policy: self.cfg.vram_policy.name(),
            prefetch: self.prefetch_on,
            prefill: phase(&self.acc[PREFILL]),
            decode: phase(&self.acc[DECODE]),
            ram_layer_residents,
            ram_bypasses: Some(self.acc[PREFILL].bypasses + self.acc[DECODE].bypasses)
                .filter(|&n| n > 0 && self.cfg.policy.is_global()),
            layer_stats,
            decode_step_bytes,
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
    timing: Option<(&mut dyn Timing, bool)>,
) -> (Outcome, Option<AccessLog>, u64) {
    let mut sim = Sim::new(trace, cfg, times, log);
    sim.check = check;
    if let Some((t, depth2)) = timing {
        sim.timing = Some(t);
        sim.depth2 = depth2;
    }
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
    if cfg.policy.is_belady() {
        let probe = Config {
            policy: Policy::Lru,
            prefetch: false,
            ..*cfg
        };
        let (_, log, _) = run(trace, probe, None, true, false, None);
        return run(trace, *cfg, log, false, false, None).0;
    }
    run(trace, *cfg, None, false, false, None).0
}

/// [`replay`] that also reports its calls to `timing`, for the virtual clock. `depth2` false
/// ignores the recorded depth-2 predictions (prefetch depth 1). Bytes equal [`replay`]'s when
/// `depth2` is true.
pub fn replay_timing(
    trace: &Trace,
    cfg: &Config,
    timing: &mut dyn Timing,
    depth2: bool,
) -> Outcome {
    let mut times = None;
    if cfg.policy.is_belady() {
        let probe = Config {
            policy: Policy::Lru,
            prefetch: false,
            ..*cfg
        };
        times = run(trace, probe, None, true, false, None).1;
    }
    run(trace, *cfg, times, false, false, Some((timing, depth2))).0
}

/// [`replay`] for an LRU or frequency policy that also counts violations of the pin invariants:
/// more pins than active slots, pins the slot table had to drop, a pinned expert evicted.
pub fn replay_checked(trace: &Trace, cfg: &Config) -> (Outcome, u64) {
    let (out, _, violations) = run(trace, *cfg, None, false, true, None);
    (out, violations)
}

/// Every combination of the lists; prefetch on and off for the policies that have both.
pub fn replay_matrix(
    trace: &Trace,
    vram: &[u32],
    ram: &[u32],
    policies: &[Policy],
    vram_policies: &[VramPolicy],
    prefetch: &[bool],
) -> Vec<Outcome> {
    let mut out = Vec::new();
    for &vram_slots in vram {
        for &ram_slots in ram {
            for &policy in policies {
                for &vram_policy in vram_policies {
                    for &pf in prefetch {
                        if policy.is_belady() && pf {
                            continue;
                        }
                        let cfg = Config {
                            vram_slots,
                            ram_slots,
                            policy,
                            vram_policy,
                            prefetch: pf,
                        };
                        out.push(replay(trace, &cfg));
                    }
                }
            }
        }
    }
    out
}
