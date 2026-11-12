//! Replay fidelity (ADR 0050 (c)): the routing the builder records, replayed with `lru` and the
//! run's own tier sizes, must give the engine's RAM-tier counters exactly. The synthetic run is
//! deterministic, so this is the local form of the 0.5-point gate on the box.
//!
//! Hardware label: `cpu_*`, no GPU needed.

mod common;

use common::argmax;
use common::synth::{MoeSpec, quantized_moe};
use common::trace::RoutingRecorder;
use std::path::PathBuf;
use tetraeder_ggml_sys::llama::Ftype;
use tetraeder_memory::RamPolicy;
use tetraeder_memory::replay::sim::{Config, Policy, Trace, VramPolicy, replay};
use tetraeder_model::{BackendChoice, ExpertStats, LoadOptions, MissPolicy, Qwen3Moe};

const CPU: BackendChoice = BackendChoice::Cpu { threads: Some(2) };
const N_PROMPT: usize = 24;
const N_DECODE: usize = 8;
const VRAM: usize = 6;

fn fixture() -> PathBuf {
    quantized_moe(&MoeSpec::REALISH, Ftype::Q4KM, "q4km-prefetch")
}

fn prompt() -> Vec<i32> {
    (0..N_PROMPT as i32).map(|i| (i * 37 + 5) % 256).collect()
}

/// `overlap`: the I/O service and arrival-order compute (ADR 0054), on `SimIo` (a virtual clock)
/// so reads are in flight when a layer is planned.
fn opts(ram: usize, depth: usize, prefetch: bool, overlap: bool) -> LoadOptions {
    LoadOptions {
        io: tetraeder_memory::IoOptions {
            service: overlap,
            workers: 4,
            ..Default::default()
        },
        overlap,
        io_factory: overlap.then(|| -> tetraeder_memory::io::IoFactory {
            std::sync::Arc::new(|paths| {
                let clock = tetraeder_memory::io::sim::SimClock::new();
                let io = tetraeder_memory::io::sim::SimIo::flat(paths, clock, 1e8, 4, 4096)?;
                Ok(Box::new(io))
            })
        }),
        expert_slots: Some(VRAM),
        ram_slots: Some(ram),
        miss: MissPolicy::Cpu,
        prefetch,
        prefetch_depth: depth,
        record_routing: true,
        ..LoadOptions::new(CPU)
    }
}

/// Mean over layers of the RAM-tier hit rate between two snapshots, as the exit harness reports it.
fn ram_hit_rate_mean(a: &ExpertStats, b: &ExpertStats) -> f64 {
    let rates: Vec<f64> = a
        .layers
        .iter()
        .zip(&b.layers)
        .map(|(la, lb)| {
            let (ra, rb) = (la.ram.unwrap(), lb.ram.unwrap());
            let (h, m) = (rb.hits - ra.hits, rb.misses - ra.misses);
            h as f64 / (h + m).max(1) as f64
        })
        .collect();
    rates.iter().sum::<f64>() / rates.len() as f64
}

/// hardware: cpu. Prefetch depth 1 and 2 and prefetch off, RAM tiers below and above top-8 plus
/// misses: per layer, the replay's RAM hits and misses equal `ExpertStats`, as do the prefetch
/// counters and the decode hit rate.
#[test]
fn cpu_replay_matches_the_engine_exactly() {
    let path = fixture();
    let dir = std::env::temp_dir();
    for (ram, depth, prefetch, overlap) in [
        (12, 1, true, false),
        (12, 2, true, false),
        (20, 1, true, false),
        (20, 2, true, false),
        (12, 1, false, false),
        (12, 1, true, true),
        (12, 2, true, true),
        (20, 1, true, true),
    ] {
        let what = format!("ram {ram} depth {depth} prefetch {prefetch} overlap {overlap}");
        let mut m = Qwen3Moe::load(&path, &opts(ram, depth, prefetch, overlap)).unwrap();
        let trace_path = dir.join(format!(
            "tetraeder-fidelity-{}-{ram}-{depth}-{prefetch}-{overlap}.trace",
            std::process::id()
        ));
        let mut rec =
            RoutingRecorder::create(&trace_path, &m, N_PROMPT, "test", "first_prefill", None);
        let p = prompt();
        let last = m.eval_last(&p).unwrap();
        rec.drain(&mut m);
        let s1 = m.expert_stats().unwrap();
        let mut next = argmax(&last) as i32;
        for _ in 0..N_DECODE {
            let l = m.eval(&[next]).unwrap();
            rec.drain(&mut m);
            next = argmax(&l) as i32;
        }
        rec.finish();
        let s2 = m.expert_stats().unwrap();

        let trace = Trace::load(&trace_path).unwrap();
        std::fs::remove_file(&trace_path).ok();
        assert_eq!(trace.header.vram_slots as usize, VRAM);
        assert_eq!(trace.header.ram_slots as usize, ram);
        assert_eq!(
            trace.header.prefetch_depth as usize,
            if prefetch { depth } else { 0 }
        );
        let out = replay(
            &trace,
            &Config {
                vram_slots: VRAM as u32,
                ram_slots: ram as u32,
                policy: Policy::Lru,
                vram_policy: VramPolicy::Static,
                prefetch,
            },
        );
        for (il, l) in s2.layers.iter().enumerate() {
            let r = l.ram.unwrap();
            assert_eq!(
                out.layer_stats[il],
                (r.hits, r.misses),
                "{what}: layer {il} RAM hits and misses"
            );
        }
        let (issued, used, wasted) = (
            out.prefill.prefetch_issued + out.decode.prefetch_issued,
            out.prefill.prefetch_used + out.decode.prefetch_used,
            out.prefill.prefetch_wasted + out.decode.prefetch_wasted,
        );
        assert_eq!(
            (issued, used, wasted),
            (
                s2.io.prefetch_issued,
                s2.io.prefetch_used,
                s2.io.prefetch_wasted
            ),
            "{what}: prefetch counters"
        );
        let want = ram_hit_rate_mean(&s1, &s2);
        assert!(
            (out.decode.ram_hit_rate_engine_mean - want).abs() < 1e-12,
            "{what}: decode hit rate {} against {want}",
            out.decode.ram_hit_rate_engine_mean
        );
        // Disk bytes: every expert read is one miss or one prefetch.
        let bytes = trace.header.expert_bytes[0];
        assert!(bytes > 0);
        eprintln!(
            "{what}: decode hit rate {want:.4}, replay {:.4}, issued {issued} used {used} wasted {wasted}",
            out.decode.ram_hit_rate_engine_mean
        );
    }
}

/// Decode steps of the frequency-pin tests: past `FREQ_REFRESH_STEPS`, so the pins are refreshed
/// from decode counts at least twice.
const N_DECODE_FREQ: usize = 40;
const SHARE: f64 = 0.5;

/// Prefill, then `N_DECODE_FREQ` greedy steps: the logits of every step, the engine stats after
/// the prefill and at the end, and the recorded trace.
fn run_freq(
    path: &std::path::Path,
    o: &LoadOptions,
    tag: &str,
) -> (Vec<Vec<f32>>, ExpertStats, ExpertStats, Trace) {
    let mut m = Qwen3Moe::load(path, o).unwrap();
    let trace_path = std::env::temp_dir().join(format!(
        "tetraeder-fidelity-freq-{}-{tag}.trace",
        std::process::id()
    ));
    let mut rec = RoutingRecorder::create(&trace_path, &m, N_PROMPT, "test", "first_prefill", None);
    let last = m.eval_last(&prompt()).unwrap();
    rec.drain(&mut m);
    let s1 = m.expert_stats().unwrap();
    let mut logits = vec![last.clone()];
    let mut next = argmax(&last) as i32;
    for _ in 0..N_DECODE_FREQ {
        let l = m.eval(&[next]).unwrap();
        rec.drain(&mut m);
        next = argmax(&l) as i32;
        logits.push(l);
    }
    rec.finish();
    let s2 = m.expert_stats().unwrap();
    let trace = Trace::load(&trace_path).unwrap();
    std::fs::remove_file(&trace_path).ok();
    (logits, s1, s2, trace)
}

fn same_bits(a: &[Vec<f32>], b: &[Vec<f32>]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits())
        })
}

/// hardware: cpu. `ram_policy: freq:0.5` in the engine: per layer RAM hits and misses and the
/// prefetch counters equal the replay's `Policy::Freq` (same share, same trace), and the logits
/// are bitwise those of the LRU run.
#[test]
fn cpu_freq_replay_matches_the_engine_exactly() {
    let path = fixture();
    // RAM tiers where the pins (half the slots) and a step's 8 experts always fit together.
    for (ram, depth, prefetch) in [(20, 1, true), (20, 2, true), (24, 1, false)] {
        let what = format!("ram {ram} depth {depth} prefetch {prefetch}");
        let lru_opts = opts(ram, depth, prefetch, false);
        let freq_opts = LoadOptions {
            ram_policy: RamPolicy::Freq { share: SHARE },
            ..opts(ram, depth, prefetch, false)
        };
        let (l_lru, _, s_lru, _) = run_freq(&path, &lru_opts, &format!("lru-{ram}-{depth}"));
        let (l_freq, s1, s2, trace) =
            run_freq(&path, &freq_opts, &format!("freq-{ram}-{depth}-{prefetch}"));
        assert!(same_bits(&l_lru, &l_freq), "{what}: logits differ from LRU");
        let out = replay(
            &trace,
            &Config {
                vram_slots: VRAM as u32,
                ram_slots: ram as u32,
                policy: Policy::Freq { share: SHARE },
                vram_policy: VramPolicy::Static,
                prefetch,
            },
        );
        for (il, l) in s2.layers.iter().enumerate() {
            let r = l.ram.unwrap();
            assert_eq!(
                out.layer_stats[il],
                (r.hits, r.misses),
                "{what}: layer {il} RAM hits and misses"
            );
        }
        let (issued, used, wasted) = (
            out.prefill.prefetch_issued + out.decode.prefetch_issued,
            out.prefill.prefetch_used + out.decode.prefetch_used,
            out.prefill.prefetch_wasted + out.decode.prefetch_wasted,
        );
        assert_eq!(
            (issued, used, wasted),
            (
                s2.io.prefetch_issued,
                s2.io.prefetch_used,
                s2.io.prefetch_wasted
            ),
            "{what}: prefetch counters"
        );
        let want = ram_hit_rate_mean(&s1, &s2);
        assert!(
            (out.decode.ram_hit_rate_engine_mean - want).abs() < 1e-12,
            "{what}: decode hit rate {} against {want}",
            out.decode.ram_hit_rate_engine_mean
        );
        let lru_misses: u64 = s_lru.layers.iter().map(|l| l.ram.unwrap().misses).sum();
        let freq_misses: u64 = s2.layers.iter().map(|l| l.ram.unwrap().misses).sum();
        eprintln!("{what}: RAM misses lru {lru_misses}, freq {freq_misses}");
    }
}

/// hardware: cpu. Under overlap (the I/O service on a virtual clock) and in tiers so small that
/// the pins and a step's experts cannot all fit, the pins give way and never fail a step; the
/// logits are bitwise those of the LRU run.
#[test]
fn cpu_freq_pins_give_way_and_keep_logits_bitwise() {
    let path = fixture();
    for (ram, share, overlap) in [
        (20, 0.5, true),
        (12, 1.0, true),
        (10, 1.0, false),
        (12, 0.75, false),
    ] {
        let lru_opts = opts(ram, 1, true, overlap);
        let freq_opts = LoadOptions {
            ram_policy: RamPolicy::Freq { share },
            ..opts(ram, 1, true, overlap)
        };
        let what = format!("ram {ram} share {share} overlap {overlap}");
        let (l_lru, ..) = run_freq(&path, &lru_opts, &format!("glru-{ram}-{overlap}"));
        let (l_freq, ..) = run_freq(&path, &freq_opts, &format!("gfreq-{ram}-{overlap}"));
        assert!(same_bits(&l_lru, &l_freq), "{what}: logits differ from LRU");
    }
}
