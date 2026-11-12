//! Decode admission into the compute tier (ADR 0059) on the host-side slot tables: property tests
//! (AGENTS.md rule 5) that an admission never evicts a pinned expert or one the layer step
//! computes from its slot and never exceeds the active slots, and a synthetic routing on which
//! admissions raise the share of routed pairs served from the compute tier.

use proptest::prelude::*;
use std::collections::BTreeSet;
use tetraeder_memory::slots::{AdmitPolicy, LayerSlots, Lookup};

const EXPERTS: u32 = 16;
const MAX_SLOTS: u32 = 6;

/// One decode layer step as the streamer runs it under the CPU miss policy with admission: the
/// resident routed experts are looked up (compute hits), the others are misses computed on the
/// CPU, and the due ones are admitted without evicting this step's hits or earlier admissions.
/// Returns (hits, admitted).
fn step(l: &mut LayerSlots, p: &mut AdmitPolicy, routed: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let (hits, misses): (Vec<u32>, Vec<u32>) =
        routed.iter().partition(|&&e| l.slot_of(e).is_some());
    for &e in &hits {
        assert!(matches!(l.lookup(e), Lookup::Hit(_)));
    }
    let mut protect = hits.clone();
    let mut admitted = Vec::new();
    for &e in &misses {
        if !p.due(0) {
            continue;
        }
        match l.lookup_except(e, &protect) {
            Lookup::Miss { .. } => {
                p.settle(0, true);
                protect.push(e);
                admitted.push(e);
            }
            _ => {
                p.settle(0, false);
                break;
            }
        }
    }
    (hits, admitted)
}

fn distinct(v: Vec<u32>) -> Vec<u32> {
    let mut seen = BTreeSet::new();
    v.into_iter().filter(|e| seen.insert(*e)).collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn admission_keeps_pins_hits_and_budget(
        active in 1u32..=MAX_SLOTS,
        hot in proptest::collection::vec(0..EXPERTS, 0..4),
        share in 0.0f64..=1.0,
        steps in proptest::collection::vec(proptest::collection::vec(0..EXPERTS, 1..6), 1..60),
    ) {
        let mut l = LayerSlots::new(EXPERTS, MAX_SLOTS);
        l.set_active(active);
        l.set_hot_set(&hot);
        // The hot set resident first, as refresh_hot_set leaves it.
        for e in 0..EXPERTS {
            if l.is_pinned(e) {
                l.lookup(e);
            }
        }
        let mut p = AdmitPolicy::new(1, share);
        for routed in steps {
            let routed = distinct(routed);
            let pinned: Vec<u32> = (0..EXPERTS)
                .filter(|&e| l.is_pinned(e) && l.slot_of(e).is_some())
                .collect();
            let (hits, admitted) = step(&mut l, &mut p, &routed);
            for e in pinned.iter().chain(&hits).chain(&admitted) {
                prop_assert!(l.slot_of(*e).is_some(), "expert {} evicted in its own step", e);
            }
            prop_assert!(l.resident() <= l.active() as usize);
            prop_assert!(hits.len() + admitted.len() <= l.active() as usize);
            // One slot per expert, one expert per slot.
            let mut owners = BTreeSet::new();
            for s in 0..l.max_slots() {
                if let Some(e) = l.owner_of(s) {
                    prop_assert!(s < l.active());
                    prop_assert!(owners.insert(e));
                    prop_assert_eq!(l.slot_of(e), Some(s));
                }
            }
        }
    }

    /// The policy admits `floor(share x due misses)` give or take one when nothing blocks it.
    #[test]
    fn admission_rate_follows_the_share(share in 0.0f64..=1.0, misses in 1usize..500) {
        let mut p = AdmitPolicy::new(1, share);
        let mut n = 0usize;
        for _ in 0..misses {
            if p.due(0) {
                p.settle(0, true);
                n += 1;
            }
        }
        let want = p.share() * misses as f64;
        prop_assert!((n as f64 - want).abs() <= 1.0, "{} admitted, want {}", n, want);
    }
}

/// A routing that drifts away from the hot set: the static tier (share 0) serves almost no pair
/// after the drift, admission (share 0.25) serves most of them.
#[test]
fn admissions_raise_the_vram_pair_share_on_a_drifting_routing() {
    // Prefill favoured experts 0 to 3; decode routes top-2 out of 8..12, a working set of 4.
    let decode: Vec<Vec<u32>> = (0..200u32)
        .map(|t| vec![8 + t % 4, 8 + (t * 3 + 1) % 4])
        .map(distinct)
        .collect();
    let share_of = |share: f64| {
        let mut l = LayerSlots::new(EXPERTS, 4);
        l.set_hot_set(&[0, 1, 2, 3]);
        for e in 0..4 {
            l.lookup(e);
        }
        if share > 0.0 {
            // ADR 0059: with admission on the hot set is the initial contents, unpinned.
            l.set_hot_set(&[]);
        }
        let mut p = AdmitPolicy::new(1, share);
        let (mut pairs, mut vram, mut admitted) = (0usize, 0usize, 0usize);
        for routed in &decode {
            let (hits, adm) = if share > 0.0 {
                step(&mut l, &mut p, routed)
            } else {
                (
                    routed
                        .iter()
                        .copied()
                        .filter(|&e| l.slot_of(e).is_some())
                        .collect(),
                    Vec::new(),
                )
            };
            pairs += routed.len();
            vram += hits.len();
            admitted += adm.len();
        }
        (vram as f64 / pairs as f64, admitted)
    };
    let (static_share, none) = share_of(0.0);
    let (adapt_share, admitted) = share_of(0.25);
    eprintln!(
        "vram pair share: static {static_share:.3}, adaptive {adapt_share:.3} ({admitted} admissions)"
    );
    assert_eq!(none, 0);
    assert_eq!(static_share, 0.0);
    assert!(admitted > 0);
    assert!(
        admitted <= 4 + 1,
        "a working set of 4 needs about 4 admissions, got {admitted}"
    );
    assert!(adapt_share > 0.9, "adaptive share {adapt_share}");
}
