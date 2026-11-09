//! The RAM-tier replay (ADR 0050 (c)) on synthetic traces: Belady bounds LRU, frequency pins
//! help a skewed workload and are neutral on a uniform one, the pin invariants hold, and the
//! shared prefetch selection equals the engine's former inline code.
//!
//! Hardware label: none (pure host code).

use proptest::prelude::*;
use tetraeder_memory::replay::prefetch_plan;
use tetraeder_memory::replay::sim::{Config, Outcome, Policy, Trace, replay, replay_checked};
use tetraeder_memory::replay::trace::{LayerStep, Record, TraceHeader};
use tetraeder_memory::slots::{LayerSlots, Lookup};

const EXPERT_BYTES: u64 = 1000;

fn header(n_layer: u32, n_expert: u32, n_used: u32) -> TraceHeader {
    let mut h = TraceHeader::new(n_layer, n_expert, n_used);
    h.expert_bytes = vec![EXPERT_BYTES; n_layer as usize];
    h
}

/// Tiny deterministic generator (xorshift).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// One decode step per entry of `steps`, each `[layer][ids]`.
fn decode_trace(h: TraceHeader, steps: &[Vec<Vec<u16>>], marker_after: Option<usize>) -> Trace {
    let mut records = Vec::new();
    for (s, layers) in steps.iter().enumerate() {
        for (il, ids) in layers.iter().enumerate() {
            records.push(Record::Step(LayerStep {
                step: s as u32,
                layer: il as u16,
                n_tokens: 1,
                ids: ids.clone(),
                predicted: None,
                predicted2: None,
            }));
        }
        if marker_after == Some(s) {
            records.push(Record::HotRefresh { step: s as u32 });
        }
    }
    Trace { header: h, records }
}

fn cfg(vram: u32, ram: u32, policy: Policy, prefetch: bool) -> Config {
    Config {
        vram_slots: vram,
        ram_slots: ram,
        policy,
        prefetch,
    }
}

fn total_hits(o: &Outcome) -> u64 {
    o.prefill.ram_hits + o.decode.ram_hits
}

#[test]
fn lru_and_belady_on_a_hand_checked_sequence() {
    let steps: Vec<Vec<Vec<u16>>> = [0, 1, 0, 2, 1].iter().map(|&e| vec![vec![e]]).collect();
    let t = decode_trace(header(1, 4, 1), &steps, None);
    let lru = replay(&t, &cfg(0, 2, Policy::Lru, false));
    assert_eq!((lru.decode.ram_hits, lru.decode.ram_lookups), (1, 5));
    assert_eq!(lru.decode.tokens, 5);
    assert_eq!(
        lru.decode.sync_miss_bytes_per_token,
        4.0 * EXPERT_BYTES as f64 / 5.0
    );
    assert_eq!(lru.decode.prefetch_bytes_per_token, 0.0);
    let opt = replay(&t, &cfg(0, 2, Policy::Belady, false));
    assert_eq!((opt.decode.ram_hits, opt.decode.ram_lookups), (2, 5));
}

#[test]
fn vram_hot_set_takes_pairs_off_the_ram_tier() {
    // Expert 3 is routed most; after the marker it is VRAM-resident and leaves the RAM lookups.
    let steps: Vec<Vec<Vec<u16>>> = [3, 3, 3, 1, 3, 2, 3]
        .iter()
        .map(|&e| vec![vec![e]])
        .collect();
    let t = decode_trace(header(1, 8, 1), &steps, Some(2));
    let o = replay(&t, &cfg(1, 4, Policy::Lru, false));
    // Steps 0..=2 (all expert 3) hit nothing in VRAM; the marker makes 3 resident (one RAM hit);
    // later pairs of 3 (two) are VRAM pairs.
    assert_eq!(o.decode.pairs, 7);
    assert_eq!(o.decode.vram_share, 2.0 / 7.0);
}

/// Distinct ids per token, `n_used` per token, drawn with `draw`.
fn draw_ids(rng: &mut Rng, n_used: usize, draw: &mut impl FnMut(&mut Rng) -> u16) -> Vec<u16> {
    let mut ids: Vec<u16> = Vec::new();
    while ids.len() < n_used {
        let e = draw(rng);
        if !ids.contains(&e) {
            ids.push(e);
        }
    }
    ids
}

proptest! {
    /// Rule 5 style: without prefetch the offline optimum never has fewer RAM hits than LRU, at
    /// every slot count.
    #[test]
    fn belady_never_loses_to_lru(
        seed in 1u64..u64::MAX,
        n_expert in 6u32..24,
        n_used in 1usize..4,
        n_steps in 4usize..80,
        marker in proptest::option::of(0usize..20),
    ) {
        let mut rng = Rng(seed);
        let steps: Vec<Vec<Vec<u16>>> = (0..n_steps)
            .map(|_| {
                (0..2)
                    .map(|_| draw_ids(&mut rng, n_used, &mut |r| r.below(u64::from(n_expert)) as u16))
                    .collect()
            })
            .collect();
        let t = decode_trace(header(2, n_expert, n_used as u32), &steps, marker);
        for ram in (n_used as u32).max(2)..n_expert {
            let lru = replay(&t, &cfg(2, ram, Policy::Lru, false));
            let opt = replay(&t, &cfg(2, ram, Policy::Belady, false));
            prop_assert_eq!(lru.decode.ram_lookups + lru.prefill.ram_lookups,
                            opt.decode.ram_lookups + opt.prefill.ram_lookups);
            prop_assert!(total_hits(&opt) >= total_hits(&lru),
                "ram {}: belady {} < lru {}", ram, total_hits(&opt), total_hits(&lru));
        }
    }

    /// Frequency pins plus the pins of reads in flight never exceed the active slots, no pinned
    /// expert is evicted, and nothing is dropped from a pin set, with prefetch replayed from
    /// random predictions.
    #[test]
    fn pins_never_exceed_slots(
        seed in 1u64..u64::MAX,
        share in prop::sample::select(vec![0.25f64, 0.5, 0.75, 1.0]),
        ram in 2u32..14,
        n_steps in 4usize..60,
    ) {
        let (n_layer, n_expert, n_used) = (4u32, 16u32, 2usize);
        let mut rng = Rng(seed);
        let mut records = Vec::new();
        for s in 0..n_steps {
            let n_tokens = if s == 0 { 3 } else { 1 };
            for il in 0..n_layer {
                let mut ids = Vec::new();
                for _ in 0..n_tokens {
                    ids.extend(draw_ids(&mut rng, n_used, &mut |r| r.below(u64::from(n_expert)) as u16));
                }
                let pred = |r: &mut Rng| {
                    let mut p = Vec::new();
                    for _ in 0..n_tokens {
                        p.extend(draw_ids(r, n_used, &mut |r| r.below(u64::from(n_expert)) as u16));
                    }
                    p
                };
                let predicted = (il + 1 < n_layer).then(|| pred(&mut rng));
                let predicted2 = (il + 2 < n_layer).then(|| pred(&mut rng));
                records.push(Record::Step(LayerStep {
                    step: s as u32,
                    layer: il as u16,
                    n_tokens,
                    ids,
                    predicted,
                    predicted2,
                }));
            }
            if s == 0 {
                records.push(Record::HotRefresh { step: 0 });
            }
        }
        let t = Trace { header: header(n_layer, n_expert, n_used as u32), records };
        for policy in [Policy::Lru, Policy::Freq { share }] {
            let (_, violations) = replay_checked(&t, &cfg(2, ram, policy, true));
            prop_assert_eq!(violations, 0, "{:?}", policy);
        }
    }
}

/// A decode trace of one layer: `skew` draws from Zipf(1.1) over `n_expert` experts; every
/// `scan_every` steps the step is a sequential scan instead (`n_used` experts walking the id
/// space).
fn zipf_trace(
    n_expert: u32,
    n_used: usize,
    n_steps: usize,
    scan_every: usize,
    zipf: bool,
) -> Trace {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let weights: Vec<f64> = (0..n_expert)
        .map(|r| {
            if zipf {
                1.0 / f64::from(r + 1).powf(1.1)
            } else {
                1.0
            }
        })
        .collect();
    let total: f64 = weights.iter().sum();
    let mut scan = 0u32;
    let mut steps = Vec::new();
    for s in 0..n_steps {
        let ids = if scan_every > 0 && s % scan_every == scan_every - 1 {
            let ids: Vec<u16> = (0..n_used as u32)
                .map(|i| ((scan + i) % n_expert) as u16)
                .collect();
            scan = (scan + n_used as u32) % n_expert;
            ids
        } else {
            draw_ids(&mut rng, n_used, &mut |r| {
                let mut x = r.unit() * total;
                for (e, w) in weights.iter().enumerate() {
                    x -= w;
                    if x <= 0.0 {
                        return e as u16;
                    }
                }
                (n_expert - 1) as u16
            })
        };
        steps.push(vec![ids]);
    }
    decode_trace(header(1, n_expert, n_used as u32), &steps, None)
}

#[test]
fn frequency_beats_lru_on_skewed_and_ties_on_uniform() {
    let (n_expert, n_used, ram) = (128, 4, 24);
    let skewed = zipf_trace(n_expert, n_used, 20_000, 3, true);
    let lru = replay(&skewed, &cfg(0, ram, Policy::Lru, false));
    let freq = replay(&skewed, &cfg(0, ram, Policy::Freq { share: 0.5 }, false));
    eprintln!(
        "skewed: lru {:.4} freq:0.5 {:.4}",
        lru.decode.ram_hit_rate, freq.decode.ram_hit_rate
    );
    assert!(
        freq.decode.ram_hit_rate >= lru.decode.ram_hit_rate + 0.02,
        "freq {} lru {}",
        freq.decode.ram_hit_rate,
        lru.decode.ram_hit_rate
    );
    let uniform = zipf_trace(n_expert, n_used, 20_000, 0, false);
    let lru = replay(&uniform, &cfg(0, ram, Policy::Lru, false));
    let freq = replay(&uniform, &cfg(0, ram, Policy::Freq { share: 0.5 }, false));
    eprintln!(
        "uniform: lru {:.4} freq:0.5 {:.4}",
        lru.decode.ram_hit_rate, freq.decode.ram_hit_rate
    );
    assert!((freq.decode.ram_hit_rate - lru.decode.ram_hit_rate).abs() <= 0.01);
}

/// The inline selection of `Streamer::start_prefetch` before it moved into `prefetch_plan`.
fn former_inline(
    predicted: &[i32],
    n_expert: u32,
    in_compute: &LayerSlots,
    ram: &LayerSlots,
    inflight: &[Vec<u32>],
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
        wanted.into_iter().partition(|&e| ram.slot_of(e).is_some());
    let mut pins = resident;
    for f in inflight {
        for &e in f {
            if !pins.contains(&e) {
                pins.push(e);
            }
        }
    }
    missing.truncate(ram_slots.saturating_sub(pins.len()));
    (pins, missing)
}

#[test]
fn prefetch_plan_matches_streamer() {
    // The `tests/prefetch.rs` scenario: top-8 of 32 experts, a RAM tier of 12 slots (and 4,
    // below top-8), 8 compute slots; predictions of several tokens with repeated experts.
    let n_expert = 32;
    let mut rng = Rng(77);
    for ram_slots in [32usize, 12, 4] {
        for round in 0..200 {
            let mut compute = LayerSlots::new(n_expert, 8);
            let mut ram = LayerSlots::new(n_expert, ram_slots as u32);
            for _ in 0..rng.below(8) {
                compute.lookup(rng.below(32) as u32);
            }
            for _ in 0..rng.below(ram_slots as u64 + 4) {
                assert!(!matches!(ram.lookup(rng.below(32) as u32), Lookup::Bypass));
            }
            let n_tokens = 1 + rng.below(3) as usize;
            let predicted: Vec<i32> = (0..8 * n_tokens)
                .map(|_| {
                    rng.below(if round % 7 == 0 { 40 } else { 32 }) as i32
                        - (round % 11 == 0) as i32
                })
                .collect();
            let inflight: Vec<Vec<u32>> = (0..rng.below(3))
                .map(|_| (0..rng.below(4)).map(|_| rng.below(32) as u32).collect())
                .collect();
            let flat: Vec<u32> = inflight.iter().flatten().copied().collect();
            let want = former_inline(&predicted, n_expert, &compute, &ram, &inflight, ram_slots);
            let got = prefetch_plan(&predicted, n_expert, &compute, &ram, &flat, ram_slots);
            assert_eq!(got, want, "ram {ram_slots} round {round}");
        }
    }
}

#[test]
fn policy_names_parse_and_roundtrip() {
    for s in ["lru", "belady", "freq:0.25", "freq:1"] {
        let p = Policy::parse(s).unwrap();
        assert_eq!(Policy::parse(&p.name()).unwrap(), p);
    }
    assert!(Policy::parse("freq:0").is_err());
    assert!(Policy::parse("freq:2").is_err());
    assert!(Policy::parse("mru").is_err());
}

#[test]
fn full_hit_share_counts_layer_steps_without_a_vram_miss() {
    // Expert 3 is routed in both steps; the marker after step 0 makes it VRAM-resident, so step 0
    // had a miss and step 1 had none.
    let steps: Vec<Vec<Vec<u16>>> = [3, 3].iter().map(|&e| vec![vec![e]]).collect();
    let t = decode_trace(header(1, 8, 1), &steps, Some(0));
    let o = replay(&t, &cfg(1, 4, Policy::Lru, false));
    assert_eq!(o.decode.full_hit_share, 0.5);
    assert_eq!(o.decode.vram_share, 0.5);
}

#[test]
fn full_hit_share_is_one_with_every_expert_in_vram() {
    // A prefill step touches all four experts, then the marker makes all of them resident
    // (4 VRAM slots for 4 experts); every decode step afterwards is a full hit.
    let mut records = vec![Record::Step(LayerStep {
        step: 0,
        layer: 0,
        n_tokens: 2,
        ids: vec![0, 1, 2, 3],
        predicted: None,
        predicted2: None,
    })];
    records.push(Record::HotRefresh { step: 0 });
    for (s, e) in [2u16, 0, 3].into_iter().enumerate() {
        records.push(Record::Step(LayerStep {
            step: s as u32 + 1,
            layer: 0,
            n_tokens: 1,
            ids: vec![e],
            predicted: None,
            predicted2: None,
        }));
    }
    let t = Trace {
        header: header(1, 4, 1),
        records,
    };
    let o = replay(&t, &cfg(4, 4, Policy::Lru, false));
    assert_eq!(o.decode.full_hit_share, 1.0);
    assert_eq!(o.decode.vram_share, 1.0);
    // no VRAM slots: no step is a full hit
    let none = replay(&t, &cfg(0, 4, Policy::Lru, false));
    assert_eq!(none.decode.full_hit_share, 0.0);
}
