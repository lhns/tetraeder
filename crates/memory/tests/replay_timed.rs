//! The replay on a virtual clock (ADR 0056): same bytes as the untimed replay, every read lands
//! once, the main thread never waits longer than the reads span, and prefetching further ahead
//! never waits more. No sleeps: time is arithmetic on the disk curve.
//!
//! Hardware label: none (pure host code).

use proptest::prelude::*;
use tetraeder_memory::replay::sim::{Config, Policy, Trace, replay};
use tetraeder_memory::replay::timed::{ReadPath, TimeModel, TimedOutcome, replay_timed};
use tetraeder_memory::replay::trace::{LayerStep, Record, TraceHeader};
use tetraeder_profiler::QueuePoint;

const N_LAYER: u32 = 6;
const N_EXPERT: u32 = 16;
const N_USED: u32 = 3;
const EXPERT_BYTES: u64 = 1_000_000;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// A prefill step of 6 tokens, then `decode` single-token steps. Predictions are the next
/// layers' routing, wrong one time in four. `warm`: the prefill routes every expert and a hot-set
/// refresh follows it, so with enough VRAM slots the decode steps read nothing.
fn trace(decode: u32, seed: u64, warm: bool) -> Trace {
    let mut h = TraceHeader::new(N_LAYER, N_EXPERT, N_USED);
    h.expert_bytes = vec![EXPERT_BYTES; N_LAYER as usize];
    let mut rng = Rng(seed | 1);
    let mut records = Vec::new();
    for step in 0..=decode {
        let n_tokens = if step == 0 { 6 } else { 1 };
        let n = (N_USED * n_tokens) as usize;
        // Hot experts 0..6 carry most of the traffic.
        let route: Vec<Vec<u16>> = (0..N_LAYER)
            .map(|_| {
                if warm && step == 0 {
                    return (0..n).map(|j| (j % N_EXPERT as usize) as u16).collect();
                }
                (0..n)
                    .map(|_| {
                        let r = rng.next() >> 8;
                        (if r % 4 == 0 { r % 16 } else { r % 6 }) as u16
                    })
                    .collect()
            })
            .collect();
        let guess = |rng: &mut Rng, right: &[u16]| -> Vec<u16> {
            right
                .iter()
                .map(|&e| {
                    if rng.next() % 4 == 0 {
                        (rng.next() % u64::from(N_EXPERT)) as u16
                    } else {
                        e
                    }
                })
                .collect()
        };
        for il in 0..N_LAYER as usize {
            let predicted = route.get(il + 1).map(|r| guess(&mut rng, r));
            let predicted2 = route.get(il + 2).map(|r| guess(&mut rng, r));
            records.push(Record::Step(LayerStep {
                step,
                layer: il as u16,
                n_tokens,
                ids: route[il].clone(),
                predicted,
                predicted2,
            }));
        }
        if warm && step == 0 {
            records.push(Record::HotRefresh { step });
        }
    }
    Trace { header: h, records }
}

fn model(path: ReadPath, rate: f64) -> TimeModel {
    TimeModel {
        gpu_layer_s: 2e-3,
        cpu_expert_s: 3e-4,
        cpu_group_s: 1e-4,
        curve: vec![
            QueuePoint {
                depth: 1,
                bytes_per_s: rate,
            },
            QueuePoint {
                depth: 8,
                bytes_per_s: rate * 4.0,
            },
        ],
        chunk_bytes: 256 * 1024,
        workers: 8,
        path,
        sync_depth: 8,
        prefetch_depth: 2,
    }
}

fn cfg(vram: u32, ram: u32, prefetch: bool) -> Config {
    Config {
        vram_slots: vram,
        ram_slots: ram,
        policy: Policy::Lru,
        prefetch,
    }
}

const PATHS: [ReadPath; 3] = [
    ReadPath::Reader,
    ReadPath::Service,
    ReadPath::ServiceOverlap,
];

fn check_invariants(o: &TimedOutcome) {
    assert_eq!(o.units_submitted, o.units_landed, "{:?}", o.path);
    for s in &o.decode.steps {
        assert!(
            s.read_wait_s <= s.read_span_s + 1e-12,
            "{:?}: wait {} > span {}",
            o.path,
            s.read_wait_s,
            s.read_span_s
        );
        assert!(s.read_wait_s + s.cpu_ready_s + s.cpu_landed_s <= s.time_s + 1e-12);
    }
    assert!((0.0..=1.0).contains(&o.decode.hidden_share));
}

proptest! {
    /// Bytes equal the untimed replay's for any slot counts, path and prefetch setting, and
    /// every unit lands exactly once.
    #[test]
    fn bytes_equal_replay_and_every_unit_lands_once(
        vram in 0u32..=8,
        ram in 1u32..=16,
        prefetch in any::<bool>(),
        path in 0usize..3,
        seed in 1u64..1000,
    ) {
        let t = trace(10, seed, seed % 2 == 0);
        let c = cfg(vram, ram, prefetch);
        let timed = replay_timed(&t, &c, &model(PATHS[path], 4e8)).unwrap();
        prop_assert_eq!(&timed.bytes, &replay(&t, &c));
        check_invariants(&timed);
        // The disk served what the replay counted, as whole units of whole experts.
        let counted = (timed.bytes.prefill.disk_bytes_per_token * timed.bytes.prefill.tokens as f64
            + timed.bytes.decode.disk_bytes_per_token * timed.bytes.decode.tokens as f64)
            .round() as u64;
        prop_assert!(timed.disk_bytes >= counted, "{} < {}", timed.disk_bytes, counted);
    }
}

/// With every expert in VRAM nothing is read and a step costs exactly the GPU time.
#[test]
fn with_zero_disk_bytes_step_time_is_the_compute_sum() {
    let t = trace(8, 7, true);
    for path in PATHS {
        let m = model(path, 4e8);
        let o = replay_timed(&t, &cfg(N_EXPERT, 1, true), &m).unwrap();
        // The warm-up reads; the decode steps do not.
        assert!(o.units_submitted > 0);
        assert_eq!(o.bytes.decode.disk_bytes_per_token, 0.0);
        assert_eq!(o.decode.tokens, 8);
        for s in &o.decode.steps {
            assert!((s.time_s - f64::from(N_LAYER) * m.gpu_layer_s).abs() < 1e-12);
            assert_eq!(s.read_wait_s, 0.0);
        }
        assert_eq!(o.decode.hidden_share, 1.0);
    }
}

/// Fewer slots read more and take longer; a slower disk takes longer still.
#[test]
fn misses_and_a_slow_disk_cost_time() {
    let t = trace(12, 3, true);
    for path in PATHS {
        let none = replay_timed(&t, &cfg(N_EXPERT, 1, true), &model(path, 4e8)).unwrap();
        let some = replay_timed(&t, &cfg(2, 4, true), &model(path, 4e8)).unwrap();
        let slow = replay_timed(&t, &cfg(2, 4, true), &model(path, 1e8)).unwrap();
        check_invariants(&some);
        assert!(some.decode.time_s > none.decode.time_s, "{path:?}");
        assert!(slow.decode.time_s > some.decode.time_s, "{path:?}");
        assert!(
            slow.decode.read_wait_s_mean > some.decode.read_wait_s_mean,
            "{path:?}"
        );
    }
}

/// Reading two layers ahead leaves more time to hide the reads than one layer ahead, when the
/// disk is not saturated (an example, not a property: a mispredicted far prefetch can crowd
/// the disk).
#[test]
fn prefetch_depth_two_waits_no_more_than_depth_one() {
    let t = trace(16, 11, true);
    for path in [ReadPath::Service, ReadPath::ServiceOverlap] {
        let mut m = model(path, 2e9);
        m.prefetch_depth = 1;
        let one = replay_timed(&t, &cfg(2, 16, true), &m).unwrap();
        m.prefetch_depth = 2;
        let two = replay_timed(&t, &cfg(2, 16, true), &m).unwrap();
        assert!(
            two.decode.read_wait_s_mean <= one.decode.read_wait_s_mean + 1e-12,
            "{path:?}: depth 2 waits {} depth 1 waits {}",
            two.decode.read_wait_s_mean,
            one.decode.read_wait_s_mean
        );
        // Depth 1 ignores the depth-2 predictions, so it reads fewer bytes ahead.
        assert!(one.bytes.decode.prefetch_issued <= two.bytes.decode.prefetch_issued);
    }
}

#[test]
fn a_model_that_cannot_run_is_refused() {
    let t = trace(2, 1, false);
    let mut m = model(ReadPath::Service, 4e8);
    m.curve.clear();
    assert!(replay_timed(&t, &cfg(2, 4, true), &m).is_err());
    let mut m = model(ReadPath::Service, 4e8);
    m.workers = 0;
    assert!(replay_timed(&t, &cfg(2, 4, true), &m).is_err());
}
