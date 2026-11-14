//! Property tests on the RAM tier as one pool across layers (AGENTS.md rule 5, ADR 0050 (c)
//! addendum 2026-11-14): the byte budget is never exceeded, no pinned or protected expert is
//! evicted, every lookup is served when enough bytes are evictable, and eviction is LRU across
//! layers.
//!
//! Hardware label: none (pure host code).

use proptest::prelude::*;
use tetraeder_memory::slots::{GlobalLookup, GlobalSlots};

#[test]
fn lru_across_layers_and_by_bytes() {
    // Layer 0 experts are 10 bytes, layer 1 experts 15; budget 30.
    let mut p = GlobalSlots::new(&[10, 15], 4, 30);
    assert_eq!(
        p.lookup_except(0, 0, &[]),
        GlobalLookup::Miss { evicted: vec![] }
    );
    assert_eq!(
        p.lookup_except(1, 0, &[]),
        GlobalLookup::Miss { evicted: vec![] }
    );
    assert_eq!(p.used(), 25);
    assert_eq!(p.lookup_except(0, 0, &[]), GlobalLookup::Hit);
    // 15 more bytes: the oldest is (1, 0), then nothing else is needed.
    assert_eq!(
        p.lookup_except(1, 1, &[]),
        GlobalLookup::Miss {
            evicted: vec![(1, 0)]
        }
    );
    assert_eq!(p.used(), 25);
    // Another 15 bytes: evicting the oldest, (0, 0), leaves room (25 - 10 + 15 = 30).
    assert_eq!(
        p.lookup_except(1, 2, &[]),
        GlobalLookup::Miss {
            evicted: vec![(0, 0)]
        }
    );
    assert_eq!(p.residents(0), 0);
    assert_eq!(p.residents(1), 2);
    // Pin (1, 1); protect (1, 2): nothing evictable for a new expert, so it bypasses and
    // evicts nothing.
    p.set_pins(1, &[1]);
    assert_eq!(p.lookup_except(1, 3, &[2]), GlobalLookup::Bypass);
    assert_eq!(p.used(), 30);
    assert!(p.contains(1, 1) && p.contains(1, 2));
    p.set_pins(1, &[]);
    assert_eq!(
        p.lookup_except(0, 3, &[]),
        GlobalLookup::Miss {
            evicted: vec![(1, 1)]
        }
    );
    // An out-of-range expert or layer bypasses.
    assert_eq!(p.lookup_except(0, 9, &[]), GlobalLookup::Bypass);
    assert_eq!(p.lookup_except(5, 0, &[]), GlobalLookup::Bypass);
}

#[test]
fn an_expert_larger_than_the_budget_bypasses() {
    let mut p = GlobalSlots::new(&[10, 40], 4, 30);
    p.lookup_except(0, 1, &[]);
    assert_eq!(p.lookup_except(1, 0, &[]), GlobalLookup::Bypass);
    assert!(p.contains(0, 1));
}

#[derive(Debug, Clone)]
enum Op {
    Look {
        layer: u32,
        expert: u32,
        protect: Vec<u32>,
    },
    Pin {
        layer: u32,
        pins: Vec<u32>,
    },
}

fn op(layers: u32, experts: u32) -> impl Strategy<Value = Op> {
    prop_oneof![
        4 => (0..layers, 0..experts, proptest::collection::vec(0..experts, 0..4))
            .prop_map(|(layer, expert, protect)| Op::Look { layer, expert, protect }),
        1 => (0..layers, proptest::collection::vec(0..experts, 0..5))
            .prop_map(|(layer, pins)| Op::Pin { layer, pins }),
    ]
}

proptest! {
    #[test]
    fn budget_pins_and_protection_hold(
        bytes in proptest::collection::vec(1u64..20, 1..5),
        budget in 0u64..120,
        ops in proptest::collection::vec(op(4, 12), 1..200),
    ) {
        let layers = bytes.len() as u32;
        let mut p = GlobalSlots::new(&bytes, 12, budget);
        // A model of the residents with their last use, to check LRU and the accounting.
        let mut last: Vec<Vec<Option<u64>>> = vec![vec![None; 12]; layers as usize];
        let mut pins: Vec<Vec<u32>> = vec![Vec::new(); layers as usize];
        let mut t = 0u64;
        for o in ops {
            match o {
                Op::Pin { layer, pins: ps } => {
                    if layer < layers {
                        p.set_pins(layer, &ps);
                        pins[layer as usize] = ps;
                    }
                }
                Op::Look { layer, expert, protect } => {
                    if layer >= layers {
                        continue;
                    }
                    t += 1;
                    let (l, e) = (layer as usize, expert as usize);
                    let was = last[l][e].is_some();
                    let evictable: u64 = (0..layers as usize)
                        .flat_map(|ll| (0..12usize).map(move |ee| (ll, ee)))
                        .filter(|&(ll, ee)| {
                            last[ll][ee].is_some()
                                && !pins[ll].contains(&(ee as u32))
                                && !(ll == l && protect.contains(&(ee as u32)))
                        })
                        .map(|(ll, _)| bytes[ll])
                        .sum();
                    let used: u64 = (0..layers as usize)
                        .map(|ll| last[ll].iter().filter(|s| s.is_some()).count() as u64 * bytes[ll])
                        .sum();
                    match p.lookup_except(layer, expert, &protect) {
                        GlobalLookup::Hit => {
                            prop_assert!(was);
                            last[l][e] = Some(t);
                        }
                        GlobalLookup::Miss { evicted } => {
                            prop_assert!(!was);
                            // The victims are the oldest evictable residents, oldest first.
                            let mut prev = 0u64;
                            for &(vl, ve) in &evicted {
                                let (vl, ve) = (vl as usize, ve as usize);
                                prop_assert!(!pins[vl].contains(&(ve as u32)), "pinned evicted");
                                prop_assert!(!(vl == l && protect.contains(&(ve as u32))), "protected evicted");
                                let s = last[vl][ve].expect("victim was resident");
                                prop_assert!(s > prev);
                                prev = s;
                                last[vl][ve] = None;
                            }
                            // Every evictable resident left is newer than the last victim.
                            for (ll, stamps) in last.iter().enumerate() {
                                for (ee, &stamp) in stamps.iter().enumerate() {
                                    let keep = pins[ll].contains(&(ee as u32))
                                        || (ll == l && protect.contains(&(ee as u32)));
                                    if let (Some(s), false) = (stamp, keep) {
                                        prop_assert!(s > prev, "a newer resident went first");
                                    }
                                }
                            }
                            last[l][e] = Some(t);
                        }
                        GlobalLookup::Bypass => {
                            // Only when the expert cannot fit even after evicting everything
                            // evictable.
                            prop_assert!(!was);
                            prop_assert!(used - evictable + bytes[l] > budget);
                        }
                    }
                    let used: u64 = (0..layers as usize)
                        .map(|ll| last[ll].iter().filter(|s| s.is_some()).count() as u64 * bytes[ll])
                        .sum();
                    prop_assert_eq!(p.used(), used);
                    prop_assert!(p.used() <= budget, "used {} over budget {}", p.used(), budget);
                    for ll in 0..layers {
                        for ee in 0..12u32 {
                            prop_assert_eq!(p.contains(ll, ee), last[ll as usize][ee as usize].is_some());
                        }
                    }
                }
            }
        }
    }
}
