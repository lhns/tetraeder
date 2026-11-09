//! The I/O service and arrival-order expert compute (ADR 0054) on the K-quant fixture of
//! `prefetch.rs`: top-8 of 32 experts, Q4_K_M, `MissPolicy::Cpu`, small VRAM and RAM tiers so most
//! decode layer steps have hits and misses. The service and overlap only move reads and group
//! boundaries, never an expert's rows, so the logits must equal the serial path bit for bit and
//! the cache counters must not change. `io_delay` makes each chunk take a while, so experts are
//! still in flight when a layer is planned.
//!
//! Hardware labels: `cpu_*` need no GPU; `gpu_*` skip without a GPU device.

mod common;

use common::argmax;
use common::synth::{MoeSpec, quantized_moe};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tetraeder_ggml_sys::ggml::Backend;
use tetraeder_ggml_sys::llama::Ftype;
use tetraeder_memory::IoOptions;
use tetraeder_model::{BackendChoice, ExpertStats, LoadOptions, MissPolicy, Qwen3Moe};

const CPU: BackendChoice = BackendChoice::Cpu { threads: Some(2) };
const N_PROMPT: usize = 64;
const N_DECODE: usize = 16;
const SLOTS: usize = 8;
const RAM: usize = 12;
const DELAY: Duration = Duration::from_millis(2);

fn fixture() -> PathBuf {
    quantized_moe(&MoeSpec::REALISH, Ftype::Q4KM, "q4km-prefetch")
}

fn prompt() -> Vec<i32> {
    (0..N_PROMPT as i32).map(|i| (i * 37 + 5) % 256).collect()
}

#[derive(Clone, Copy)]
struct Cfg {
    service: bool,
    overlap: bool,
    prefetch: bool,
    depth: usize,
    delay: bool,
}

impl Cfg {
    const SERIAL: Cfg = Cfg {
        service: false,
        overlap: false,
        prefetch: true,
        depth: 1,
        delay: false,
    };
    fn service(self) -> Cfg {
        Cfg {
            service: true,
            delay: true,
            ..self
        }
    }
    fn overlap(self) -> Cfg {
        Cfg {
            overlap: true,
            ..self.service()
        }
    }
}

fn opts(backend: BackendChoice, c: Cfg) -> LoadOptions {
    LoadOptions {
        expert_slots: Some(SLOTS),
        ram_slots: Some(RAM),
        miss: MissPolicy::Cpu,
        prefetch: c.prefetch,
        prefetch_depth: c.depth,
        io: IoOptions {
            service: c.service,
            workers: 4,
            ..IoOptions::default()
        },
        overlap: c.overlap,
        io_delay: c.delay.then_some(DELAY),
        ..LoadOptions::new(backend)
    }
}

fn load(path: &Path, backend: BackendChoice, c: Cfg) -> Qwen3Moe {
    Qwen3Moe::load(path, &opts(backend, c)).unwrap()
}

/// Prefill, then `N_DECODE` greedy decodes. Returns every eval's logits.
fn sequence(m: &mut Qwen3Moe) -> Vec<Vec<f32>> {
    let mut all = Vec::new();
    let l = m.eval(&prompt()).unwrap();
    let mut next = argmax(&l[(N_PROMPT - 1) * m.n_vocab()..]) as i32;
    all.push(l);
    for _ in 0..N_DECODE {
        let l = m.eval(&[next]).unwrap();
        next = argmax(&l) as i32;
        all.push(l);
    }
    all
}

fn assert_bitwise(want: &[Vec<f32>], got: &[Vec<f32>], what: &str) {
    assert_eq!(want.len(), got.len(), "{what}");
    for (i, (a, b)) in want.iter().zip(got).enumerate() {
        assert!(
            a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "{what}: eval {i} differs"
        );
    }
}

/// hardware: cpu. Reads through the service give the logits of the single reader thread, with
/// prefetch depth 1 and 2 and with prefetch off (then only the synchronous reads run).
#[test]
fn cpu_service_decode_is_bitwise_equal_to_reader() {
    let path = fixture();
    for (prefetch, depth) in [(true, 1), (true, 2), (false, 1)] {
        let base = Cfg {
            prefetch,
            depth,
            ..Cfg::SERIAL
        };
        let want = sequence(&mut load(&path, CPU, base));
        let mut m = load(&path, CPU, base.service());
        let what = format!("prefetch {prefetch} depth {depth}");
        assert_bitwise(&want, &sequence(&mut m), &what);
        let s = m.expert_stats().unwrap();
        if prefetch {
            assert!(s.io.prefetch_issued > 0, "{what}");
        }
    }
}

/// hardware: cpu. Computing the CPU experts as they land gives the logits of the serial path,
/// with prefetch depth 1 and 2 and with prefetch off (every miss is then an urgent read).
#[test]
fn cpu_overlap_decode_is_bitwise_equal_to_serial() {
    let path = fixture();
    for (prefetch, depth) in [(true, 1), (true, 2), (false, 1)] {
        let base = Cfg {
            prefetch,
            depth,
            ..Cfg::SERIAL
        };
        let want = sequence(&mut load(&path, CPU, base));
        let mut m = load(&path, CPU, base.overlap());
        let what = format!("prefetch {prefetch} depth {depth}");
        assert_bitwise(&want, &sequence(&mut m), &what);
        let p = m.take_profile();
        assert!(p.cpu_ready + p.cpu_landed > Duration::ZERO, "{what}");
        // The new parts are inside `streamer`, so the sum still adds up.
        assert!(
            p.cpu_ready + p.read_wait + p.cpu_landed + p.copy <= p.streamer,
            "{what}"
        );
        let parts = p.inputs + p.build + p.alloc + p.compute + p.readback + p.streamer;
        assert!(parts <= p.total + Duration::from_millis(1), "{what}");
    }
}

/// Per layer the cache counters, and the totals the replay of ADR 0050 (c) depends on.
fn cache_view(s: &ExpertStats) -> (Vec<String>, [u64; 6]) {
    let layers = s
        .layers
        .iter()
        .map(|l| {
            format!(
                "{:?} {:?} full {} partial {} cpu {} mixed {}",
                l.compute, l.ram, l.full_steps, l.partial_steps, l.cpu, l.mixed_steps
            )
        })
        .collect();
    let io = &s.io;
    (
        layers,
        [
            io.experts_read,
            io.bytes_read,
            io.prefetch_issued,
            io.prefetch_used,
            io.prefetch_wasted,
            io.prefetch_wasted_bytes,
        ],
    )
}

/// hardware: cpu. Overlap does not change which experts the RAM tier holds or reads: per layer
/// hits, misses and evictions, the prefetch counters and the bytes read equal the serial run's.
#[test]
fn cpu_overlap_keeps_cache_behaviour() {
    let path = fixture();
    for (prefetch, depth) in [(true, 1), (true, 2), (false, 1)] {
        let base = Cfg {
            prefetch,
            depth,
            ..Cfg::SERIAL
        };
        let mut serial = load(&path, CPU, base);
        sequence(&mut serial);
        let mut over = load(&path, CPU, base.overlap());
        sequence(&mut over);
        let what = format!("prefetch {prefetch} depth {depth}");
        let (sl, st) = cache_view(&serial.expert_stats().unwrap());
        let (ol, ot) = cache_view(&over.expert_stats().unwrap());
        assert_eq!(sl, ol, "{what}: per-layer counters");
        assert_eq!(st, ot, "{what}: io totals");
    }
}

/// hardware: cpu. With a delay experts are still in flight when their layer is computed: the
/// main thread waits for some (`read_wait`) and computes them as a later group (`cpu_landed`).
/// Without a delay the counts still equal the serial run's.
#[test]
fn cpu_overlap_runs_pending_groups() {
    let path = fixture();
    let mut slow = load(&path, CPU, Cfg::SERIAL.overlap());
    sequence(&mut slow);
    let p = slow.take_profile();
    let s = slow.expert_stats().unwrap();
    assert!(p.read_wait > Duration::ZERO, "{p:?}");
    assert!(p.cpu_landed > Duration::ZERO, "{p:?}");
    assert!(s.io.pending_steps > 0 && s.io.pending_experts >= s.io.pending_steps);
    assert!(s.io.urgent_wait_time > Duration::ZERO);
    assert!(s.io.overlap_span >= s.io.overlap_wait);

    let fast = Cfg {
        delay: false,
        ..Cfg::SERIAL.overlap()
    };
    let mut serial = load(&path, CPU, Cfg::SERIAL);
    let want = sequence(&mut serial);
    let mut quick = load(&path, CPU, fast);
    assert_bitwise(&want, &sequence(&mut quick), "no delay");
    assert_eq!(
        cache_view(&serial.expert_stats().unwrap()),
        cache_view(&quick.expert_stats().unwrap()),
        "no delay"
    );
}

/// hardware: cpu. Overlap needs the service.
#[test]
fn cpu_overlap_without_service_is_refused() {
    let path = fixture();
    let o = LoadOptions {
        overlap: true,
        ..opts(CPU, Cfg::SERIAL)
    };
    let err = Qwen3Moe::load(&path, &o).unwrap_err().to_string();
    assert!(err.contains("io.service"), "{err}");
}

/// hardware: cpu. A read that fails (the file is cut off) fails the step, also when the experts
/// are computed as they land, and after the file is restored the model gives the logits of a
/// fresh load.
#[test]
fn cpu_overlap_read_error_fails_the_step_and_recovers() {
    let src = fixture();
    let path = src.with_extension("overlap-cut.gguf");
    let original = std::fs::read(&src).unwrap();
    std::fs::write(&path, &original).unwrap();
    let want = sequence(&mut load(&path, CPU, Cfg::SERIAL));

    let mut m = load(&path, CPU, Cfg::SERIAL.overlap());
    m.eval(&prompt()).unwrap();
    let cut = std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .and_then(|f| f.set_len(0));
    if cut.is_err() {
        eprintln!("cannot truncate an open file here; skipped");
        let _ = std::fs::remove_file(path);
        return;
    }
    m.reset();
    // The RAM tier (12 of 32 experts) cannot hold the next prompt's experts.
    let other: Vec<i32> = (0..N_PROMPT as i32).map(|i| (i * 53 + 11) % 256).collect();
    let mut failed = m.eval(&other).is_err();
    for _ in 0..N_DECODE {
        if failed {
            break;
        }
        failed = m.eval(&[7]).is_err();
    }
    assert!(failed, "a read of the cut file fails");
    std::fs::write(&path, &original).unwrap();
    m.reset();
    assert_bitwise(&want, &sequence(&mut m), "after the error");
    let _ = std::fs::remove_file(path);
}

/// hardware: gpu (skips without a GPU device). The GPU builder with the service and overlap gives
/// the logits of the serial path: the CPU groups run before the segment, their rows are copied
/// to the GPU and scattered into the accumulator.
#[test]
fn gpu_overlap_logits_equal_serial() {
    if !Backend::gpu_available() {
        eprintln!("skip: no GPU device in this build or on this machine");
        return;
    }
    let path = fixture();
    let gpu = BackendChoice::Gpu { index: 0 };
    let want = sequence(&mut load(&path, gpu, Cfg::SERIAL));
    let mut m = load(&path, gpu, Cfg::SERIAL.overlap());
    assert_bitwise(&want, &sequence(&mut m), "gpu overlap");
}
