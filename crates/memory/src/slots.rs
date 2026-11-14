//! Expert slot cache, host side (ADR 0031): per MoE layer a table from expert to slot, a pinned
//! hot set from routing counts and LRU for the rest. No graph or device work: a lookup says
//! which slot holds the expert or which slot to load it into; the caller moves the bytes.
//!
//! One [`SlotCache`] per tier (VRAM, RAM). The number of slots in use per layer follows the
//! phase budget ([`SlotCache::set_budget`]): prefill shrinks it to the hot set so workspace can
//! borrow the bytes, decode grows it back. Slot indices below the active count stay stable.

use std::collections::HashMap;

/// Where a slot cache lives. Only used to label metrics and decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tier {
    Vram,
    Ram,
}

impl Tier {
    pub fn as_str(self) -> &'static str {
        match self {
            Tier::Vram => "vram",
            Tier::Ram => "ram",
        }
    }
}

/// Result of [`LayerSlots::lookup`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lookup {
    /// The expert is resident in this slot.
    Hit(u32),
    /// The expert was not resident. The slot is now assigned to it: load the weights there.
    /// `evicted` is the expert that held the slot.
    Miss { slot: u32, evicted: Option<u32> },
    /// Every active slot holds a pinned expert: compute from the cold tier without a slot.
    Bypass,
}

/// An expert that changed slots when the active slot count shrank: copy `from` to `to`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Move {
    pub expert: u32,
    pub from: u32,
    pub to: u32,
}

/// What a resize or a new hot set did to the residents.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Change {
    /// `(expert, slot)` no longer resident.
    pub evicted: Vec<(u32, u32)>,
    /// Pinned residents that had to leave the slots being removed.
    pub moved: Vec<Move>,
    /// Pinned experts that are not resident: prefetch them.
    pub missing: Vec<u32>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LayerStats {
    pub hits: u64,
    pub misses: u64,
    pub bypass: u64,
}

/// The slots of one MoE layer in one tier.
#[derive(Debug, Clone)]
pub struct LayerSlots {
    /// Experts in the layer.
    experts: u32,
    /// Slots in the arena; the active count never exceeds it.
    max: u32,
    active: u32,
    table: HashMap<u32, u32>,
    owner: Vec<Option<u32>>,
    /// Last use per slot, for LRU.
    stamp: Vec<u64>,
    clock: u64,
    pinned: Vec<bool>,
    pinned_count: u32,
    stats: LayerStats,
}

impl LayerSlots {
    pub fn new(experts: u32, max_slots: u32) -> Self {
        Self {
            experts,
            max: max_slots,
            active: max_slots,
            table: HashMap::new(),
            owner: vec![None; max_slots as usize],
            stamp: vec![0; max_slots as usize],
            clock: 0,
            pinned: vec![false; experts as usize],
            pinned_count: 0,
            stats: LayerStats::default(),
        }
    }

    pub fn active(&self) -> u32 {
        self.active
    }

    pub fn max_slots(&self) -> u32 {
        self.max
    }

    pub fn pinned_count(&self) -> u32 {
        self.pinned_count
    }

    pub fn is_pinned(&self, expert: u32) -> bool {
        self.pinned.get(expert as usize).copied().unwrap_or(false)
    }

    pub fn resident(&self) -> usize {
        self.table.len()
    }

    pub fn slot_of(&self, expert: u32) -> Option<u32> {
        self.table.get(&expert).copied()
    }

    pub fn owner_of(&self, slot: u32) -> Option<u32> {
        self.owner.get(slot as usize).copied().flatten()
    }

    pub fn stats(&self) -> LayerStats {
        self.stats
    }

    /// Hit: the slot. Miss: a slot assigned to the expert (a free one, else the least recently
    /// used unpinned one). An out-of-range expert is a bypass.
    pub fn lookup(&mut self, expert: u32) -> Lookup {
        self.lookup_except(expert, &[])
    }

    /// [`Self::lookup`] that never evicts an expert of `protect` (the ones the current layer
    /// step computes from their slots, ADR 0059): a bypass when only those are left.
    pub fn lookup_except(&mut self, expert: u32, protect: &[u32]) -> Lookup {
        if expert >= self.experts {
            self.stats.bypass += 1;
            return Lookup::Bypass;
        }
        self.clock += 1;
        if let Some(&slot) = self.table.get(&expert) {
            self.stamp[slot as usize] = self.clock;
            self.stats.hits += 1;
            return Lookup::Hit(slot);
        }
        let active = self.active as usize;
        let free = self.owner[..active].iter().position(Option::is_none);
        let victim = || {
            (0..active)
                .filter(|&s| {
                    self.owner[s].is_some_and(|e| !self.pinned[e as usize] && !protect.contains(&e))
                })
                .min_by_key(|&s| self.stamp[s])
        };
        let Some(slot) = free.or_else(victim) else {
            self.stats.bypass += 1;
            return Lookup::Bypass;
        };
        let evicted = self.owner[slot];
        if let Some(e) = evicted {
            self.table.remove(&e);
        }
        self.owner[slot] = Some(expert);
        self.table.insert(expert, slot as u32);
        self.stamp[slot] = self.clock;
        self.stats.misses += 1;
        Lookup::Miss {
            slot: slot as u32,
            evicted,
        }
    }

    /// Forget every resident (a load failed and the slots hold no valid weights). Pins and
    /// statistics stay.
    pub fn clear(&mut self) {
        self.table.clear();
        self.owner.fill(None);
        self.stamp.fill(0);
    }

    /// Pin these experts (at most the active slot count; extra entries and repeats are
    /// ignored). Experts that were pinned and are not any more become ordinary LRU entries.
    pub fn set_hot_set(&mut self, hot: &[u32]) -> Change {
        self.pinned.fill(false);
        self.pinned_count = 0;
        for &e in hot {
            if self.pinned_count < self.active
                && e < self.experts
                && !std::mem::replace(&mut self.pinned[e as usize], true)
            {
                self.pinned_count += 1;
            }
        }
        Change {
            missing: (0..self.experts)
                .filter(|&e| self.pinned[e as usize] && !self.table.contains_key(&e))
                .collect(),
            ..Change::default()
        }
    }

    /// Change the number of slots in use, clamped to `[pinned count, max]`. Unpinned residents
    /// of the removed slots are evicted; pinned ones move to a lower slot.
    pub fn set_active(&mut self, n: u32) -> Change {
        let n = n.clamp(self.pinned_count, self.max);
        let mut change = Change::default();
        for slot in (n..self.active).rev() {
            let Some(e) = self.owner[slot as usize].take() else {
                continue;
            };
            self.table.remove(&e);
            if !self.pinned[e as usize] {
                change.evicted.push((e, slot));
                continue;
            }
            // A free or unpinned slot below n exists: at most pinned_count - 1 others are pinned.
            let to = (0..n)
                .find(|&s| self.owner[s as usize].is_none())
                .or_else(|| {
                    (0..n)
                        .filter(|&s| {
                            self.owner[s as usize].is_some_and(|x| !self.pinned[x as usize])
                        })
                        .min_by_key(|&s| self.stamp[s as usize])
                })
                .expect("a slot below the new count is free or unpinned");
            if let Some(old) = self.owner[to as usize] {
                self.table.remove(&old);
                change.evicted.push((old, to));
            }
            self.owner[to as usize] = Some(e);
            self.table.insert(e, to);
            self.stamp[to as usize] = self.stamp[slot as usize];
            change.moved.push(Move {
                expert: e,
                from: slot,
                to,
            });
        }
        self.active = n;
        change
    }
}

/// Fixed-point unit of [`AdmitPolicy`] credit: one admission.
const ADMIT_UNIT: u64 = 1_000_000;

/// Decode admission into the compute tier under the CPU miss policy (ADR 0059): per layer a
/// credit that every compute-tier miss raises by `share`; a miss is due for admission while the
/// credit is at least one. With share `1/k`, every k-th miss of a layer. Integer credit, so the
/// sequence is exact and deterministic.
#[derive(Debug, Clone)]
pub struct AdmitPolicy {
    /// Share in millionths.
    step: u64,
    credit: Vec<u64>,
}

impl AdmitPolicy {
    /// `share` is clamped to `[0, 1]`.
    pub fn new(layers: usize, share: f64) -> Self {
        let share = if share.is_finite() {
            share.clamp(0.0, 1.0)
        } else {
            0.0
        };
        Self {
            step: (share * ADMIT_UNIT as f64).round() as u64,
            credit: vec![0; layers],
        }
    }

    pub fn share(&self) -> f64 {
        self.step as f64 / ADMIT_UNIT as f64
    }

    /// One compute-tier miss of `layer`: whether it is due for admission. Call
    /// [`Self::settle`] for every due miss.
    pub fn due(&mut self, layer: usize) -> bool {
        let c = &mut self.credit[layer];
        *c += self.step;
        *c >= ADMIT_UNIT
    }

    /// A due miss was admitted, or not (no slot): then one admission stays due and no more, so
    /// a blocked step is never followed by a burst.
    pub fn settle(&mut self, layer: usize, admitted: bool) {
        let c = &mut self.credit[layer];
        *c = if admitted {
            c.saturating_sub(ADMIT_UNIT)
        } else {
            ADMIT_UNIT
        };
    }
}

/// Top `k` experts by routing count (counts of zero never make the set), ties to the lower id.
pub fn hot_from_counts(counts: &[u64], k: u32) -> Vec<u32> {
    let mut ids: Vec<u32> = (0..counts.len() as u32)
        .filter(|&e| counts[e as usize] > 0)
        .collect();
    ids.sort_by_key(|&e| (std::cmp::Reverse(counts[e as usize]), e));
    ids.truncate(k as usize);
    ids
}

/// Frequency pins are refreshed every this many decode steps (and at every hot-set marker).
pub const FREQ_REFRESH_STEPS: u64 = 16;

/// The RAM-tier policy of the engine (ADR 0050 (c)): LRU, or LRU with the most routed experts
/// of each layer pinned. The replay's `Policy::Freq` has the same semantics.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum RamPolicy {
    #[default]
    Lru,
    /// Pin the top `ceil(share * slots)` experts of the layer by routing count.
    Freq { share: f64 },
}

impl RamPolicy {
    /// `lru` or `freq:<share in (0, 1]>`.
    pub fn parse(s: &str) -> Result<RamPolicy, String> {
        if s == "lru" {
            return Ok(RamPolicy::Lru);
        }
        s.strip_prefix("freq:")
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| *v > 0.0 && *v <= 1.0)
            .map(|share| RamPolicy::Freq { share })
            .ok_or_else(|| format!("ram policy {s:?}: expected lru or freq:<share in (0, 1]>"))
    }

    pub fn name(&self) -> String {
        match self {
            RamPolicy::Lru => "lru".into(),
            RamPolicy::Freq { share } => format!("freq:{share}"),
        }
    }

    /// Experts pinned per layer of a RAM tier of `slots`.
    pub fn pin_count(&self, slots: u32) -> u32 {
        match self {
            RamPolicy::Lru => 0,
            RamPolicy::Freq { share } => (share * f64::from(slots)).ceil() as u32,
        }
    }
}

/// The frequency pins of a layer: the top `k` experts by routing count, leaving out those
/// pinned in the compute tier (they never reach the RAM tier). Most routed first.
pub fn freq_pins(counts: &[u64], compute: &LayerSlots, k: u32) -> Vec<u32> {
    let mut c = counts.to_vec();
    for (e, n) in c.iter_mut().enumerate() {
        if compute.is_pinned(e as u32) {
            *n = 0;
        }
    }
    hot_from_counts(&c, k)
}

/// The pins for a layer step that stages `chunk` into `slots` RAM slots: `mandatory` (reads in
/// flight) first, then the frequency pins that `chunk` uses, then the other frequency pins as
/// far as the slots leave room, so the chunk's misses always find an unpinned slot. Nothing is
/// dropped when `|mandatory + chunk + freq| <= slots`.
pub fn pins_with_room(mandatory: &[u32], chunk: &[u32], freq: &[u32], slots: usize) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::with_capacity(slots);
    for &e in mandatory {
        if !out.contains(&e) {
            out.push(e);
        }
    }
    let mut used = out.clone();
    for &e in chunk {
        if !used.contains(&e) {
            used.push(e);
        }
    }
    let mut room = slots.saturating_sub(used.len());
    for &e in freq {
        if out.contains(&e) {
            continue;
        }
        if chunk.contains(&e) {
            out.push(e);
        } else if room > 0 {
            room -= 1;
            out.push(e);
        }
    }
    out
}

/// Slots per layer that `bytes` buy across `layers` MoE layers of `slot_bytes` each.
pub fn slots_per_layer(bytes: u64, layers: u32, slot_bytes: u64) -> u32 {
    let per_slot_all_layers = u64::from(layers).saturating_mul(slot_bytes);
    if per_slot_all_layers == 0 {
        return 0;
    }
    u32::try_from(bytes / per_slot_all_layers).unwrap_or(u32::MAX)
}

/// Result of [`GlobalSlots::lookup_except`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GlobalLookup {
    Hit,
    /// The expert is now resident; `evicted` are the `(layer, expert)` that made room, oldest
    /// first.
    Miss {
        evicted: Vec<(u32, u32)>,
    },
    /// Not enough evictable bytes (or out of range): compute from the cold tier without caching.
    Bypass,
}

/// A RAM tier as one byte budget across layers (ADR 0050 (c) addendum 2026-11-14), as the page
/// cache under an mmap is: LRU over every (layer, expert), experts sized per layer. Pins are per
/// layer, as in [`LayerSlots::set_hot_set`]. Host-side bookkeeping only, used by the replay.
#[derive(Debug, Clone)]
pub struct GlobalSlots {
    bytes: Vec<u64>,
    experts: u32,
    budget: u64,
    used: u64,
    clock: u64,
    /// Last use per (layer, expert), `None` when not resident.
    stamp: Vec<Vec<Option<u64>>>,
    /// Residents by last use: the eviction order.
    order: std::collections::BTreeSet<(u64, u32, u32)>,
    pinned: Vec<Vec<bool>>,
    residents: Vec<u32>,
}

impl GlobalSlots {
    /// `expert_bytes` per layer; `budget` in bytes.
    pub fn new(expert_bytes: &[u64], experts: u32, budget: u64) -> Self {
        let layers = expert_bytes.len();
        Self {
            bytes: expert_bytes.to_vec(),
            experts,
            budget,
            used: 0,
            clock: 0,
            stamp: vec![vec![None; experts as usize]; layers],
            order: std::collections::BTreeSet::new(),
            pinned: vec![vec![false; experts as usize]; layers],
            residents: vec![0; layers],
        }
    }

    pub fn budget(&self) -> u64 {
        self.budget
    }

    pub fn used(&self) -> u64 {
        self.used
    }

    pub fn layers(&self) -> usize {
        self.bytes.len()
    }

    /// Experts of `layer` resident.
    pub fn residents(&self, layer: u32) -> u32 {
        self.residents[layer as usize]
    }

    pub fn contains(&self, layer: u32, expert: u32) -> bool {
        self.stamp
            .get(layer as usize)
            .and_then(|l| l.get(expert as usize))
            .is_some_and(Option::is_some)
    }

    pub fn is_pinned(&self, layer: u32, expert: u32) -> bool {
        self.pinned
            .get(layer as usize)
            .and_then(|l| l.get(expert as usize))
            .copied()
            .unwrap_or(false)
    }

    /// Replace the pins of `layer` (out-of-range ids are ignored).
    pub fn set_pins(&mut self, layer: u32, pins: &[u32]) {
        let Some(p) = self.pinned.get_mut(layer as usize) else {
            return;
        };
        p.fill(false);
        for &e in pins {
            if let Some(x) = p.get_mut(e as usize) {
                *x = true;
            }
        }
    }

    /// Hit, or make room by evicting the least recently used experts of any layer that are
    /// neither pinned nor in `protect` (experts of `layer`). Nothing is evicted on a bypass.
    pub fn lookup_except(&mut self, layer: u32, expert: u32, protect: &[u32]) -> GlobalLookup {
        let (l, e) = (layer as usize, expert as usize);
        if l >= self.bytes.len() || expert >= self.experts {
            return GlobalLookup::Bypass;
        }
        self.clock += 1;
        if let Some(old) = self.stamp[l][e] {
            self.order.remove(&(old, layer, expert));
            self.order.insert((self.clock, layer, expert));
            self.stamp[l][e] = Some(self.clock);
            return GlobalLookup::Hit;
        }
        let need = self.bytes[l];
        if need > self.budget {
            return GlobalLookup::Bypass;
        }
        let mut free = self.budget - self.used;
        let mut victims = Vec::new();
        if free < need {
            for &(_, vl, ve) in &self.order {
                if self.pinned[vl as usize][ve as usize] || (vl == layer && protect.contains(&ve)) {
                    continue;
                }
                victims.push((vl, ve));
                free += self.bytes[vl as usize];
                if free >= need {
                    break;
                }
            }
            if free < need {
                return GlobalLookup::Bypass;
            }
        }
        for &(vl, ve) in &victims {
            let s = self.stamp[vl as usize][ve as usize]
                .take()
                .expect("a victim is resident");
            self.order.remove(&(s, vl, ve));
            self.used -= self.bytes[vl as usize];
            self.residents[vl as usize] -= 1;
        }
        self.stamp[l][e] = Some(self.clock);
        self.order.insert((self.clock, layer, expert));
        self.used += need;
        self.residents[l] += 1;
        GlobalLookup::Miss { evicted: victims }
    }
}

/// The slot cache of one tier: [`LayerSlots`] for every MoE layer of a stage.
#[derive(Debug, Clone)]
pub struct SlotCache {
    tier: Tier,
    layers: Vec<LayerSlots>,
}

impl SlotCache {
    pub fn new(tier: Tier, moe_layers: u32, experts: u32, max_slots: u32) -> Self {
        Self {
            tier,
            layers: (0..moe_layers)
                .map(|_| LayerSlots::new(experts, max_slots))
                .collect(),
        }
    }

    pub fn tier(&self) -> Tier {
        self.tier
    }

    pub fn layers(&self) -> usize {
        self.layers.len()
    }

    pub fn layer(&self, i: usize) -> &LayerSlots {
        &self.layers[i]
    }

    pub fn layer_mut(&mut self, i: usize) -> &mut LayerSlots {
        &mut self.layers[i]
    }

    /// Pin the top `hot` experts of every layer by its routing counts (`counts[layer][expert]`).
    pub fn set_hot_sets(&mut self, counts: &[Vec<u64>], hot: u32) -> Vec<Change> {
        self.layers
            .iter_mut()
            .zip(counts)
            .map(|(l, c)| l.set_hot_set(&hot_from_counts(c, hot)))
            .collect()
    }

    /// Slots per layer the budget buys, `expert_bytes` being the slot size. Always at least the
    /// pinned count and at most the arena.
    pub fn slots_for(&self, budget_bytes: u64, expert_bytes: u64) -> u32 {
        slots_per_layer(budget_bytes, self.layers.len() as u32, expert_bytes)
    }

    /// Switch to a phase's slot budget: every layer keeps its hot set and gets the slots the
    /// bytes buy (prefill: few, decode: many).
    pub fn set_budget(&mut self, budget_bytes: u64, expert_bytes: u64) -> Vec<Change> {
        let n = self.slots_for(budget_bytes, expert_bytes);
        self.layers.iter_mut().map(|l| l.set_active(n)).collect()
    }

    /// Bytes the active slots hold, for the `slots` memory metric.
    pub fn active_bytes(&self, expert_bytes: u64) -> u64 {
        self.layers
            .iter()
            .map(|l| u64::from(l.active) * expert_bytes)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lru_evicts_the_oldest_unpinned() {
        let mut l = LayerSlots::new(8, 3);
        l.set_hot_set(&[7]);
        assert_eq!(
            l.lookup(7),
            Lookup::Miss {
                slot: 0,
                evicted: None
            }
        );
        assert!(matches!(l.lookup(1), Lookup::Miss { slot: 1, .. }));
        assert!(matches!(l.lookup(2), Lookup::Miss { slot: 2, .. }));
        assert_eq!(l.lookup(1), Lookup::Hit(1));
        // 2 is now the oldest unpinned; 7 is older but pinned.
        assert_eq!(
            l.lookup(3),
            Lookup::Miss {
                slot: 2,
                evicted: Some(2)
            }
        );
        assert_eq!(l.lookup(7), Lookup::Hit(0));
        assert_eq!(l.stats().hits, 2);
        assert_eq!(l.stats().misses, 4);
    }

    #[test]
    fn full_of_pinned_bypasses() {
        let mut l = LayerSlots::new(8, 2);
        l.set_hot_set(&[0, 1]);
        l.lookup(0);
        l.lookup(1);
        assert_eq!(l.lookup(5), Lookup::Bypass);
        assert_eq!(l.stats().bypass, 1);
    }

    #[test]
    fn shrink_moves_pinned_and_evicts_the_rest() {
        let mut l = LayerSlots::new(8, 4);
        for e in [0, 1, 2, 3] {
            l.lookup(e);
        }
        l.set_hot_set(&[3]);
        let c = l.set_active(2);
        assert_eq!(l.active(), 2);
        assert_eq!(c.moved.len(), 1);
        assert_eq!(c.moved[0].expert, 3);
        assert!(c.moved[0].to < 2);
        assert!(l.slot_of(3).is_some_and(|s| s < 2));
        // 2 and one of 0 and 1 are evicted; the slot 3 moved into held the other.
        assert_eq!(l.resident(), 2);
        let c = l.set_active(4);
        assert!(c.evicted.is_empty() && c.moved.is_empty());
        assert_eq!(l.active(), 4);
    }

    #[test]
    fn active_never_drops_below_the_hot_set() {
        let mut l = LayerSlots::new(8, 4);
        l.set_hot_set(&[0, 1, 2]);
        l.set_active(0);
        assert_eq!(l.active(), 3);
    }

    #[test]
    fn clear_forgets_residents_and_keeps_stats_and_pins() {
        let mut l = LayerSlots::new(8, 2);
        l.set_hot_set(&[5]);
        l.lookup(5);
        l.lookup(1);
        l.clear();
        assert_eq!(l.resident(), 0);
        assert_eq!(l.slot_of(5), None);
        assert_eq!(l.owner_of(0), None);
        assert!(l.is_pinned(5));
        assert_eq!(l.stats().misses, 2);
        // The slots are free again.
        assert!(matches!(
            l.lookup(1),
            Lookup::Miss {
                slot: 0,
                evicted: None
            }
        ));
    }

    #[test]
    fn admit_every_kth_miss() {
        let mut p = AdmitPolicy::new(2, 0.25);
        let due: Vec<bool> = (0..8).map(|_| p.due(0)).collect();
        assert_eq!(
            due,
            [false, false, false, true, true, true, true, true],
            "credit is only spent by settle"
        );
        let mut p = AdmitPolicy::new(2, 0.25);
        let mut admitted = Vec::new();
        for i in 0..12 {
            if p.due(1) {
                p.settle(1, true);
                admitted.push(i);
            }
        }
        assert_eq!(admitted, [3, 7, 11]);
        // Layer 0 untouched; a blocked admission stays due once.
        assert!(!p.due(0));
        // Credit in halves: 0.5, 1.0 (blocked, stays 1.0), 1.5, admitted to 0.5, 1.0, 0, 0.5.
        let mut p = AdmitPolicy::new(1, 0.5);
        assert!(!p.due(0));
        assert!(p.due(0));
        p.settle(0, false);
        assert!(p.due(0));
        p.settle(0, true);
        assert!(p.due(0));
        p.settle(0, true);
        assert!(!p.due(0));
        assert!(!AdmitPolicy::new(1, 0.0).due(0));
        assert_eq!(AdmitPolicy::new(1, 7.0).share(), 1.0);
    }

    #[test]
    fn lookup_except_spares_the_protected() {
        let mut l = LayerSlots::new(8, 2);
        l.lookup(1);
        l.lookup(2);
        // 1 is the LRU but protected: 2 goes.
        assert_eq!(
            l.lookup_except(3, &[1]),
            Lookup::Miss {
                slot: 1,
                evicted: Some(2)
            }
        );
        assert_eq!(l.lookup_except(4, &[1, 3]), Lookup::Bypass);
    }

    #[test]
    fn hot_set_from_counts() {
        assert_eq!(hot_from_counts(&[5, 0, 9, 5, 1], 3), vec![2, 0, 3]);
        assert_eq!(hot_from_counts(&[0, 0], 4), Vec::<u32>::new());
        assert_eq!(slots_per_layer(100, 2, 10), 5);
        assert_eq!(slots_per_layer(100, 0, 10), 0);
    }

    #[test]
    fn phase_budget_switch() {
        let mut c = SlotCache::new(Tier::Vram, 2, 16, 8);
        c.set_hot_sets(&[vec![9, 8, 7, 0], vec![0, 0, 3, 0]], 2);
        assert_eq!(c.layer(0).pinned_count(), 2);
        assert_eq!(c.layer(1).pinned_count(), 1);
        // 2 layers x 10 bytes: 40 bytes buy 2 slots per layer, 160 bytes 8.
        c.set_budget(40, 10);
        assert_eq!((c.layer(0).active(), c.layer(1).active()), (2, 2));
        assert_eq!(c.active_bytes(10), 40);
        c.set_budget(160, 10);
        assert_eq!(c.layer(0).active(), 8);
    }
}
