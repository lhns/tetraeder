//! Memory and expert-cache metrics (ADR 0031, docs/plan.md "Metric catalog") through the
//! telemetry crate. Without telemetry every call is a no-op.

use crate::arena::{ArenaBuffer, Arenas, Purpose};
use crate::slots::{Lookup, Tier};
use std::sync::Arc;
use std::time::Duration;
use tetraeder_telemetry::{CacheResult, Telemetry};

#[derive(Clone, Default)]
pub struct MemoryMetrics {
    telemetry: Option<Arc<Telemetry>>,
}

impl MemoryMetrics {
    pub fn new(telemetry: Option<Arc<Telemetry>>) -> Self {
        Self { telemetry }
    }

    /// `tetraeder.memory.used` and `.capacity` for every arena: capacity is the buffer size,
    /// used the bytes the plan asked for. `tier` is the device's tier (`vram`, `ram`).
    pub fn arenas<B: ArenaBuffer>(&self, tier: &str, arenas: &Arenas<B>) {
        let Some(t) = &self.telemetry else { return };
        for a in arenas.iter() {
            let kind = a.purpose.as_str();
            t.metrics().memory_used(&a.device, tier, kind, a.requested);
            t.metrics()
                .memory_capacity(&a.device, tier, kind, a.buffer.size());
        }
    }

    /// The measured reserve of a device, as used and capacity of kind `reserve`.
    pub fn reserve(&self, device: &str, tier: &str, bytes: u64) {
        let Some(t) = &self.telemetry else { return };
        let kind = Purpose::Reserve.as_str();
        t.metrics().memory_used(device, tier, kind, bytes);
        t.metrics().memory_capacity(device, tier, kind, bytes);
    }

    /// Slot bytes in use under the current phase budget.
    pub fn slots_in_use(&self, device: &str, tier: &str, bytes: u64) {
        if let Some(t) = &self.telemetry {
            t.metrics()
                .memory_used(device, tier, Purpose::Slots.as_str(), bytes);
        }
    }

    /// `tetraeder.expert.cache.requests`: a bypass counts as a miss (the expert was not in a
    /// slot).
    pub fn lookup(&self, layer: u32, tier: Tier, result: &Lookup) {
        if let Some(t) = &self.telemetry {
            let r = match result {
                Lookup::Hit(_) => CacheResult::Hit,
                Lookup::Miss { .. } | Lookup::Bypass => CacheResult::Miss,
            };
            t.metrics()
                .expert_cache_requests(1, layer, tier.as_str(), r);
        }
    }

    /// `tetraeder.expert.layer_hit`: one MoE layer step, `full` when all top-k were resident.
    pub fn layer_step(&self, full: bool) {
        if let Some(t) = &self.telemetry {
            t.metrics().expert_layer_hit(full);
        }
    }

    /// `tetraeder.expert.load.duration`: `source` is the tier the bytes came from.
    pub fn expert_load(&self, tier: Tier, source: &str, took: Duration) {
        if let Some(t) = &self.telemetry {
            t.metrics()
                .expert_load_duration(took, tier.as_str(), source);
        }
    }

    /// `tetraeder.tier.read.bytes`.
    pub fn tier_read(&self, device: &str, tier: &str, bytes: u64) {
        if let Some(t) = &self.telemetry {
            t.metrics().tier_read_bytes(bytes, device, tier);
        }
    }
}
