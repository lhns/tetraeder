//! Property tests on the slot cache (AGENTS.md rule 5): a slot never holds two experts, pinned
//! experts are never evicted, counts stay within the active budget, and the table agrees with
//! the slots after any sequence of lookups, hot-set changes and budget switches.

use proptest::prelude::*;
use std::collections::{BTreeSet, HashSet};
use tetraeder_memory::slots::{LayerSlots, Lookup, SlotCache, Tier, hot_from_counts};

const EXPERTS: u32 = 12;
const MAX_SLOTS: u32 = 7;

#[derive(Debug, Clone)]
enum Op {
    Lookup(u32),
    /// Counts for every expert and the hot set size.
    HotSet(Vec<u64>, u32),
    Active(u32),
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        6 => (0..EXPERTS + 2).prop_map(Op::Lookup),
        1 => (proptest::collection::vec(0u64..5, EXPERTS as usize), 0..MAX_SLOTS + 2)
            .prop_map(|(c, k)| Op::HotSet(c, k)),
        1 => (0..MAX_SLOTS + 3).prop_map(Op::Active),
    ]
}

/// Table, owners, pins and the active budget agree.
fn check(l: &LayerSlots) {
    let mut seen = HashSet::new();
    let mut count = 0;
    for slot in 0..l.max_slots() {
        if let Some(e) = l.owner_of(slot) {
            // A slot never holds two experts, an expert never sits in two slots.
            assert!(seen.insert(e), "expert {e} in two slots");
            assert_eq!(l.slot_of(e), Some(slot));
            assert!(slot < l.active(), "resident in inactive slot {slot}");
            count += 1;
        }
    }
    assert_eq!(count, l.resident());
    assert!(l.resident() <= l.active() as usize);
    assert!(l.active() <= l.max_slots());
    assert!(l.pinned_count() <= l.active());
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn layer_invariants(ops in proptest::collection::vec(op(), 1..200)) {
        let mut l = LayerSlots::new(EXPERTS, MAX_SLOTS);
        for op in ops {
            let pinned_resident: BTreeSet<u32> = (0..EXPERTS)
                .filter(|&e| l.is_pinned(e) && l.slot_of(e).is_some())
                .collect();
            let slots_before: Vec<(u32, u32)> = (0..EXPERTS)
                .filter_map(|e| l.slot_of(e).map(|s| (e, s)))
                .collect();
            match op {
                Op::Lookup(e) => {
                    let before = l.slot_of(e);
                    match l.lookup(e) {
                        Lookup::Hit(s) => {
                            prop_assert_eq!(before, Some(s));
                        }
                        Lookup::Miss { slot, evicted } => {
                            prop_assert!(before.is_none());
                            prop_assert!(slot < l.active());
                            prop_assert_eq!(l.slot_of(e), Some(slot));
                            if let Some(v) = evicted {
                                prop_assert!(!l.is_pinned(v), "evicted pinned {}", v);
                                prop_assert!(l.slot_of(v).is_none());
                            }
                        }
                        Lookup::Bypass => {
                            prop_assert!(before.is_none());
                            prop_assert!(e >= EXPERTS || l.pinned_count() == l.active());
                            prop_assert!(l.slot_of(e).is_none());
                        }
                    }
                    // A pinned resident expert is still there after any lookup.
                    for e in &pinned_resident {
                        prop_assert!(l.slot_of(*e).is_some(), "pinned {} evicted", e);
                    }
                }
                Op::HotSet(counts, k) => {
                    let hot = hot_from_counts(&counts, k);
                    let change = l.set_hot_set(&hot);
                    prop_assert!(l.pinned_count() <= l.active());
                    prop_assert!(change.evicted.is_empty());
                    for e in &change.missing {
                        prop_assert!(l.is_pinned(*e) && l.slot_of(*e).is_none());
                    }
                    // Residents stay where they are.
                    for (e, slot) in &slots_before {
                        prop_assert_eq!(l.slot_of(*e), Some(*slot));
                    }
                }
                Op::Active(n) => {
                    let still_pinned: Vec<u32> = pinned_resident.iter().copied().collect();
                    let change = l.set_active(n);
                    prop_assert!(l.active() >= l.pinned_count());
                    for (e, _) in &change.evicted {
                        prop_assert!(!l.is_pinned(*e), "shrink evicted pinned {}", e);
                    }
                    for m in &change.moved {
                        prop_assert!(l.is_pinned(m.expert));
                        prop_assert_eq!(l.slot_of(m.expert), Some(m.to));
                        prop_assert!(m.to < l.active() && m.from >= l.active());
                    }
                    for e in still_pinned {
                        prop_assert!(l.slot_of(e).is_some(), "pinned {} lost on resize", e);
                    }
                }
            }
            check(&l);
        }
    }

    #[test]
    fn cache_counts_stay_within_budget(
        budget_slots in 0u32..10,
        hot in 0u32..5,
        looks in proptest::collection::vec((0usize..3, 0u32..EXPERTS), 1..200),
    ) {
        let mut c = SlotCache::new(Tier::Vram, 3, EXPERTS, 8);
        let counts: Vec<Vec<u64>> = (0..3)
            .map(|l| (0..EXPERTS as u64).map(|e| (e * 7 + l * 3) % 5).collect())
            .collect();
        c.set_hot_sets(&counts, hot);
        // 3 layers x 100 bytes per slot.
        c.set_budget(u64::from(budget_slots) * 300, 100);
        for (layer, expert) in looks {
            c.layer_mut(layer).lookup(expert);
        }
        for i in 0..3 {
            let l = c.layer(i);
            prop_assert_eq!(l.active(), budget_slots.min(8).max(l.pinned_count()));
            prop_assert!(l.resident() <= l.active() as usize);
        }
    }
}
