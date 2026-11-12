//! Property tests on the frequency pins of the RAM tier (AGENTS.md rule 5, ADR 0050 (c)):
//! the pins never leave a chunk without slots, never evict a pinned expert, and pick the most
//! routed experts outside the compute tier.

use proptest::prelude::*;
use tetraeder_memory::slots::{LayerSlots, Lookup, RamPolicy, freq_pins, pins_with_room};

const EXPERTS: u32 = 24;

fn distinct(v: Vec<u32>) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::new();
    for e in v {
        if !out.contains(&e) {
            out.push(e);
        }
    }
    out
}

proptest! {
    /// The pins are a subset of the mandatory and frequency pins, mandatory first, without
    /// repeats; they never need more slots than the tier has beside the chunk.
    #[test]
    fn pins_leave_room_for_the_chunk(
        mandatory in proptest::collection::vec(0..EXPERTS, 0..4),
        chunk in proptest::collection::vec(0..EXPERTS, 0..9),
        freq in proptest::collection::vec(0..EXPERTS, 0..20),
        slots in 12usize..24,
    ) {
        let (mandatory, chunk, freq) = (distinct(mandatory), distinct(chunk), distinct(freq));
        let pins = pins_with_room(&mandatory, &chunk, &freq, slots);
        prop_assert_eq!(distinct(pins.clone()), pins.clone(), "repeats");
        prop_assert_eq!(&pins[..mandatory.len()], &mandatory[..], "mandatory first");
        for e in &pins[mandatory.len()..] {
            prop_assert!(freq.contains(e), "pin {} is not a frequency pin", e);
        }
        let mut need = pins.clone();
        need.extend(&chunk);
        let need = distinct(need);
        prop_assert!(need.len() <= slots, "{} pins and chunk in {} slots", need.len(), slots);
        // Nothing is dropped when everything fits.
        let mut all = mandatory.clone();
        all.extend(&chunk);
        all.extend(&freq);
        if distinct(all).len() <= slots {
            let mut want = mandatory.clone();
            want.extend(freq.iter().filter(|e| !mandatory.contains(e)));
            let mut got = pins.clone();
            got.sort_unstable();
            want.sort_unstable();
            prop_assert_eq!(got, want);
        }
    }

    /// Staging a chunk under the pins: every lookup hits or finds a slot, the experts of the
    /// chunk are resident afterwards, and no pinned expert is evicted.
    #[test]
    fn staging_under_pins_never_bypasses_or_evicts_a_pin(
        warm in proptest::collection::vec(0..EXPERTS, 0..40),
        rounds in proptest::collection::vec(
            (proptest::collection::vec(0..EXPERTS, 1..9), proptest::collection::vec(0..EXPERTS, 0..20)),
            1..30,
        ),
        slots in 8u32..20,
    ) {
        let mut l = LayerSlots::new(EXPERTS, slots);
        for e in warm {
            l.lookup(e);
        }
        for (chunk, freq) in rounds {
            let (chunk, freq) = (distinct(chunk), distinct(freq));
            let pins = pins_with_room(&[], &chunk, &freq, slots as usize);
            l.set_hot_set(&pins);
            // Pinned residents before the chunk.
            let held: Vec<u32> = pins.iter().copied().filter(|&e| l.slot_of(e).is_some()).collect();
            // Hits first, as the engine does.
            let mut order = chunk.clone();
            order.sort_by_key(|&e| l.slot_of(e).is_none());
            for &e in &order {
                match l.lookup(e) {
                    Lookup::Bypass => prop_assert!(false, "bypass for expert {}", e),
                    Lookup::Miss { evicted: Some(old), .. } => {
                        prop_assert!(!pins.contains(&old), "evicted pin {}", old);
                        prop_assert!(!chunk.contains(&old), "evicted chunk expert {}", old);
                    }
                    _ => {}
                }
            }
            for &e in &chunk {
                prop_assert!(l.slot_of(e).is_some(), "chunk expert {} not resident", e);
            }
            for &e in &held {
                prop_assert!(l.slot_of(e).is_some(), "pin {} lost", e);
            }
            prop_assert!(l.pinned_count() as usize <= slots as usize);
        }
    }

    /// The frequency pins are the most routed experts outside the compute tier.
    #[test]
    fn freq_pins_are_the_top_counts_outside_the_compute_tier(
        counts in proptest::collection::vec(0u64..6, EXPERTS as usize),
        compute_hot in proptest::collection::vec(0..EXPERTS, 0..6),
        k in 0u32..20,
    ) {
        let mut compute = LayerSlots::new(EXPERTS, 8);
        compute.set_hot_set(&compute_hot);
        let pins = freq_pins(&counts, &compute, k);
        prop_assert!(pins.len() <= k as usize);
        prop_assert_eq!(distinct(pins.clone()), pins.clone());
        for &e in &pins {
            prop_assert!(!compute.is_pinned(e), "compute-tier expert {} pinned", e);
            prop_assert!(counts[e as usize] > 0);
        }
        // Nothing left out routes more than something kept.
        if let Some(&last) = pins.last() {
            for e in 0..EXPERTS {
                if !pins.contains(&e) && !compute.is_pinned(e) && pins.len() == k as usize {
                    prop_assert!(counts[e as usize] <= counts[last as usize]);
                }
            }
        }
    }
}

#[test]
fn policy_parses_and_counts_pins() {
    assert_eq!(RamPolicy::parse("lru"), Ok(RamPolicy::Lru));
    assert_eq!(
        RamPolicy::parse("freq:0.5"),
        Ok(RamPolicy::Freq { share: 0.5 })
    );
    assert_eq!(
        RamPolicy::parse("freq:1"),
        Ok(RamPolicy::Freq { share: 1.0 })
    );
    for bad in ["freq", "freq:0", "freq:1.5", "freq:x", "belady", ""] {
        assert!(RamPolicy::parse(bad).is_err(), "{bad}");
    }
    assert_eq!(RamPolicy::Freq { share: 0.5 }.pin_count(77), 39);
    assert_eq!(RamPolicy::Freq { share: 1.0 }.pin_count(77), 77);
    assert_eq!(RamPolicy::Lru.pin_count(77), 0);
    assert_eq!(RamPolicy::Freq { share: 0.5 }.name(), "freq:0.5");
}
