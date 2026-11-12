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
