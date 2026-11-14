//! Routing traces and the offline RAM-tier replay (ADR 0050 (c)).
//!
//! - [`trace`]: the `tetraeder-routing` v1 file the builder's recorder writes.
//! - [`sim`]: drives the real [`crate::slots::SlotCache`] per tier through a trace in the engine's
//!   call order and scores RAM-tier policies.
//! - [`timed`]: the same replay on a virtual clock with a disk model (ADR 0056).
//! - [`prefetch_plan`]: the prefetch selection, shared by the engine's `Streamer` and the replay.

pub mod sim;
pub mod timed;
pub mod trace;

/// Version of the replay's counters. Bump it when the outcome for an existing trace and
/// configuration changes; a committed `tier-replay.json` records it and a bench test replays the
/// committed file again (ADR 0056).
pub const REPLAY_VERSION: u32 = 1;

use crate::slots::LayerSlots;

/// Which experts a prefetch of one layer reads, and which it pins (ADR 0038).
///
/// `predicted` is `[n_used, n_tokens]`; the distinct predicted experts are taken most predicted
/// first, those in the compute tier (`in_compute`) are skipped, the rest split into resident in
/// the RAM tier (`ram`) and missing. `inflight` are the experts of earlier reads of the same
/// layer that are still pinned. Returns `(pins, missing)`: `pins` are the predicted resident
/// experts then the in-flight ones (everything to keep pinned so the new reads cannot evict it),
/// `missing` the experts to read, at most `ram_slots - pins.len()`. The caller pins
/// `pins + missing` and reads `missing`.
pub fn prefetch_plan(
    predicted: &[i32],
    n_expert: u32,
    in_compute: &LayerSlots,
    ram: &LayerSlots,
    inflight: &[u32],
    ram_slots: usize,
) -> (Vec<u32>, Vec<u32>) {
    prefetch_plan_with(
        predicted,
        n_expert,
        in_compute,
        |e| ram.slot_of(e).is_some(),
        inflight,
        ram_slots,
    )
}

/// [`prefetch_plan`] with RAM residency as a predicate, for a tier that is not a
/// [`LayerSlots`] (the replay's pool across layers).
pub fn prefetch_plan_with(
    predicted: &[i32],
    n_expert: u32,
    in_compute: &LayerSlots,
    in_ram: impl Fn(u32) -> bool,
    inflight: &[u32],
    ram_slots: usize,
) -> (Vec<u32>, Vec<u32>) {
    let mut votes = vec![0u32; n_expert as usize];
    for &e in predicted {
        if let Some(v) = usize::try_from(e).ok().and_then(|e| votes.get_mut(e)) {
            *v += 1;
        }
    }
    let mut wanted: Vec<u32> = (0..n_expert).filter(|&e| votes[e as usize] > 0).collect();
    wanted.sort_by_key(|&e| std::cmp::Reverse(votes[e as usize]));
    wanted.retain(|&e| in_compute.slot_of(e).is_none());
    let (resident, mut missing): (Vec<u32>, Vec<u32>) =
        wanted.into_iter().partition(|&e| in_ram(e));
    let mut pins = resident;
    for &e in inflight {
        if !pins.contains(&e) {
            pins.push(e);
        }
    }
    missing.truncate(ram_slots.saturating_sub(pins.len()));
    (pins, missing)
}
