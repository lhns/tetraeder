//! Decode admission into the compute tier (ADR 0059) on the K-quant fixture of `cpu_experts.rs`:
//! top-8 of 32 experts, Q4_K_M, `MissPolicy::Cpu`, 8 compute slots. An admitted expert still runs
//! on the CPU in its step and from its compute slot afterwards; on the CPU backend both tiers are
//! CPU tensors, so the logits must equal the resident builder bit for bit, on the serial path
//! (one and two CPU chunks per layer step) and under overlap (ADR 0054). On a GPU the logits
//! stay within the tolerance of `cpu_experts.rs`. Every upload mode (addendum 2026-11-12b) gives
//! the same logits and cache counters; the test-only `Deferred` mode holds each copy until the
//! barrier and poisons the slot meanwhile, so a slot read before its copy landed would show.
//!
//! Hardware labels (AGENTS rule 9): `cpu_*` need no GPU; `gpu_*` skip without a GPU device.

mod common;

use common::synth::{MoeSpec, quantized_moe};
use common::{TOL_GPU, argmax};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tetraeder_ggml_sys::ggml::Backend;
use tetraeder_ggml_sys::llama::Ftype;
use tetraeder_memory::IoOptions;
use tetraeder_memory::io::IoFactory;
use tetraeder_memory::io::sim::{SimClock, SimIo};
use tetraeder_model::{AdmitUpload, BackendChoice, ExpertStats, LoadOptions, MissPolicy, Qwen3Moe};

const MODES: [AdmitUpload; 3] = [AdmitUpload::Sync, AdmitUpload::Async, AdmitUpload::Deferred];

const CPU: BackendChoice = BackendChoice::Cpu { threads: Some(2) };
const N_PROMPT: usize = 24;
const N_DECODE: usize = 8;
const SLOTS: usize = 8;

fn fixture() -> PathBuf {
    quantized_moe(&MoeSpec::REALISH, Ftype::Q4KM, "q4km-admit")
}

fn prompt() -> Vec<i32> {
    (0..N_PROMPT as i32).map(|i| (i * 37 + 5) % 256).collect()
}

/// `MissPolicy::Cpu` with `ram` RAM slots and admission at `share` (`None`: off).
fn cpu_opts(backend: BackendChoice, ram: usize, share: Option<f64>) -> LoadOptions {
    LoadOptions {
        expert_slots: Some(SLOTS),
        ram_slots: Some(ram),
        miss: MissPolicy::Cpu,
        admit_share: share,
        ..LoadOptions::new(backend)
    }
}

/// The same with the I/O service on a virtual clock and overlap (as in `overlap.rs`), so
/// experts are still in flight when their layer is planned.
fn overlap_opts(ram: usize, share: Option<f64>) -> LoadOptions {
    let sim: IoFactory = Arc::new(|paths| {
        let io = SimIo::flat(paths, SimClock::new(), 1e8, 4, 4096)?;
        Ok(Box::new(io))
    });
    LoadOptions {
        io: IoOptions {
            service: true,
            workers: 4,
            ..IoOptions::default()
        },
        overlap: true,
        io_factory: Some(sim),
        ..cpu_opts(CPU, ram, share)
    }
}

fn load(path: &Path, o: &LoadOptions) -> Qwen3Moe {
    Qwen3Moe::load(path, o).unwrap()
}

/// Prefill, then `N_DECODE` decode steps fed `forced` (else the greedy choice). Returns every
/// eval's logits and the decode tokens.
fn sequence(m: &mut Qwen3Moe, forced: Option<&[i32]>) -> (Vec<Vec<f32>>, Vec<i32>) {
    let p = prompt();
    let l = m.eval(&p).unwrap();
    let mut next = argmax(&l[(N_PROMPT - 1) * m.n_vocab()..]) as i32;
    let mut all = vec![l];
    let mut tokens = Vec::new();
    for i in 0..N_DECODE {
        if let Some(f) = forced {
            next = f[i];
        }
        tokens.push(next);
        let l = m.eval(&[next]).unwrap();
        next = argmax(&l) as i32;
        all.push(l);
    }
    (all, tokens)
}

fn sum(s: &ExpertStats, f: impl Fn(&tetraeder_model::LayerExpertStats) -> u64) -> u64 {
    s.layers.iter().map(f).sum()
}

fn assert_bitwise(want: &[Vec<f32>], got: &[Vec<f32>], what: &str) {
    assert_eq!(want.len(), got.len(), "{what}");
    for (i, (a, b)) in want.iter().zip(got).enumerate() {
        let delta = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(
            a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "{what}: eval {i} differs, max |delta| {delta:e}"
        );
    }
}

/// hardware: cpu. Admission at share 1 with 12 RAM slots (one CPU chunk per decode layer step)
/// and 4 (two chunks), serial and overlapped: logits equal the resident builder's bit for bit;
/// admissions happen, count as uploads after the hot set, and later decode steps hit them.
#[test]
fn cpu_admission_is_exact() {
    let path = fixture();
    let (want, tokens) = sequence(&mut load(&path, &LoadOptions::new(CPU)), None);
    let cases = [
        (
            "serial ram 12",
            cpu_opts(CPU, 12, Some(1.0)),
            cpu_opts(CPU, 12, None),
        ),
        (
            "serial ram 4",
            cpu_opts(CPU, 4, Some(1.0)),
            cpu_opts(CPU, 4, None),
        ),
        (
            "overlap ram 12",
            overlap_opts(12, Some(1.0)),
            overlap_opts(12, None),
        ),
    ];
    for (what, on, off) in cases {
        let mut m = load(&path, &on);
        let (got, _) = sequence(&mut m, Some(&tokens));
        assert_bitwise(&want, &got, what);
        let mut base = load(&path, &off);
        let (got_off, _) = sequence(&mut base, Some(&tokens));
        assert_bitwise(&want, &got_off, what);
        let s = m.expert_stats().unwrap();
        let b = base.expert_stats().unwrap();
        eprintln!(
            "{what}: admitted {} ({} bytes), uploads {} (off {}), compute hits {} (off {}), cpu {} (off {})",
            s.io.admitted,
            s.io.admitted_bytes,
            s.io.uploads,
            b.io.uploads,
            sum(&s, |l| l.compute.hits),
            sum(&b, |l| l.compute.hits),
            sum(&s, |l| l.cpu),
            sum(&b, |l| l.cpu),
        );
        assert!(s.io.admitted > 0, "{what}: decode admitted experts");
        assert!(s.io.admitted_bytes > 0, "{what}");
        assert_eq!(b.io.admitted, 0, "{what}: off by default");
        // The same first prefill gives the same hot set; every other upload is an admission.
        assert_eq!(s.io.uploads, b.io.uploads + s.io.admitted, "{what}");
        assert!(
            sum(&s, |l| l.compute.hits) > 0,
            "{what}: the tier still serves hits"
        );
    }
}

/// hardware: cpu. The same token fed at every decode step routes to nearly the same experts:
/// with admission they move into the compute tier after the first steps, so fewer routed
/// experts run on the CPU than with the static hot set, and more hit the compute tier.
#[test]
fn cpu_admission_raises_compute_hits_on_repeated_routing() {
    let path = fixture();
    let p = prompt();
    let run = |share: Option<f64>| {
        let mut m = load(&path, &cpu_opts(CPU, 12, share));
        m.eval(&p).unwrap();
        let before = m.expert_stats().unwrap();
        for _ in 0..12 {
            m.eval(&[p[3]]).unwrap();
        }
        let after = m.expert_stats().unwrap();
        let d = |f: fn(&tetraeder_model::LayerExpertStats) -> u64| sum(&after, f) - sum(&before, f);
        (d(|l| l.compute.hits), d(|l| l.cpu), after.io.admitted)
    };
    let (hits_off, cpu_off, _) = run(None);
    let (hits_on, cpu_on, admitted) = run(Some(0.5));
    eprintln!(
        "decode: compute hits {hits_off} -> {hits_on}, cpu {cpu_off} -> {cpu_on}, admitted {admitted}"
    );
    if cpu_off == 0 {
        eprintln!("the hot set already holds every routed expert; nothing to admit");
        return;
    }
    assert!(admitted > 0);
    assert!(cpu_on < cpu_off, "cpu {cpu_off} -> {cpu_on}");
    assert!(hits_on > hits_off, "hits {hits_off} -> {hits_on}");
}

/// hardware: cpu. Every upload mode, serial with one and two CPU chunks per layer step and
/// overlapped: logits equal the resident builder's bit for bit, also with every copy deferred to
/// the barrier and the slot poisoned until then. Cache and admission counters are the same in all
/// modes; only the deferred copies make the barrier wait, the synchronous ones never do.
#[test]
fn cpu_upload_modes_are_exact_and_count_alike() {
    let path = fixture();
    let (want, tokens) = sequence(&mut load(&path, &LoadOptions::new(CPU)), None);
    let cases: [(&str, fn() -> LoadOptions); 3] = [
        ("serial ram 12", || cpu_opts(CPU, 12, Some(1.0))),
        ("serial ram 4", || cpu_opts(CPU, 4, Some(1.0))),
        ("overlap ram 12", || overlap_opts(12, Some(1.0))),
    ];
    for (what, opts) in cases {
        let mut first: Option<ExpertStats> = None;
        for mode in MODES {
            let o = LoadOptions {
                admit_upload: mode,
                ..opts()
            };
            let mut m = load(&path, &o);
            let (got, _) = sequence(&mut m, Some(&tokens));
            let label = format!("{what} {mode:?}");
            assert_bitwise(&want, &got, &label);
            let s = m.expert_stats().unwrap();
            eprintln!(
                "{label}: admitted {} ({} bytes), admit {:?}, wait {:?}, pinned {:?}",
                s.io.admitted,
                s.io.admitted_bytes,
                s.io.admit_time,
                s.io.admit_wait,
                m.admit_staging_pinned()
            );
            assert!(s.io.admitted > 0, "{label}");
            match mode {
                AdmitUpload::Sync => {
                    assert_eq!(s.io.admit_wait, std::time::Duration::ZERO, "{label}");
                    assert_eq!(m.admit_staging_pinned(), None, "{label}");
                }
                // No pinned host memory on the CPU backend.
                AdmitUpload::Async => assert_eq!(m.admit_staging_pinned(), Some(false), "{label}"),
                AdmitUpload::Deferred => assert!(!s.io.admit_wait.is_zero(), "{label}"),
            }
            match &first {
                None => first = Some(s),
                Some(f) => {
                    let tier = |x: &ExpertStats| -> Vec<_> {
                        x.layers.iter().map(|l| (l.compute, l.cpu)).collect()
                    };
                    assert_eq!(tier(f), tier(&s), "{label}: compute-tier counters");
                    assert_eq!(f.io.admitted, s.io.admitted, "{label}");
                    assert_eq!(f.io.admitted_bytes, s.io.admitted_bytes, "{label}");
                    assert_eq!(f.io.uploads, s.io.uploads, "{label}");
                }
            }
        }
    }
}

/// hardware: cpu. The same token at every decode step, so admitted experts are routed again and
/// read from their compute slots in later steps: with every copy deferred to the barrier (the
/// slot poisoned until then) the logits equal the synchronous copies' bit for bit, and admission
/// raised the compute hits, so admitted slots were read.
#[test]
fn cpu_admitted_slots_are_read_only_after_their_copy() {
    let path = fixture();
    let p = prompt();
    let run = |share: Option<f64>, mode: AdmitUpload| {
        let o = LoadOptions {
            admit_upload: mode,
            ..cpu_opts(CPU, 12, share)
        };
        let mut m = load(&path, &o);
        let mut logits = vec![m.eval(&p).unwrap()];
        for _ in 0..12 {
            logits.push(m.eval(&[p[3]]).unwrap());
        }
        let s = m.expert_stats().unwrap();
        (logits, sum(&s, |l| l.compute.hits), s.io)
    };
    let (_, hits_off, _) = run(None, AdmitUpload::Sync);
    let (sync, hits_sync, _) = run(Some(0.5), AdmitUpload::Sync);
    let (deferred, hits_deferred, io) = run(Some(0.5), AdmitUpload::Deferred);
    eprintln!(
        "compute hits: off {hits_off}, sync {hits_sync}, deferred {hits_deferred}; admitted {}, wait {:?}",
        io.admitted, io.admit_wait
    );
    assert_bitwise(&sync, &deferred, "deferred against sync");
    assert_eq!(hits_sync, hits_deferred);
    if io.admitted == 0 {
        eprintln!("the hot set already holds every routed expert; nothing to admit");
        return;
    }
    assert!(hits_deferred > hits_off, "admitted slots were read");
    assert!(!io.admit_wait.is_zero(), "the barrier applied the copies");
}

/// hardware: gpu (skips without a GPU device). Admission on the GPU against admission off on the
/// same tokens: an admitted expert moves from the CPU to the GPU kernels, so the logits stay
/// within `TOL_GPU` plus twice the resident CPU-vs-GPU difference, as in `cpu_experts.rs`.
#[test]
fn gpu_admission_matches_static_tier() {
    if !Backend::gpu_available() {
        eprintln!("skip: no GPU device in this build or on this machine");
        return;
    }
    let gpu = BackendChoice::Gpu { index: 0 };
    let path = fixture();
    let (want, tokens) = sequence(&mut load(&path, &cpu_opts(gpu, 32, None)), None);
    let mut m = load(&path, &cpu_opts(gpu, 32, Some(1.0)));
    let (got, _) = sequence(&mut m, Some(&tokens));
    // The async copies (pinned ring, own stream) land the same bytes: equal to sync bit for bit.
    let o = LoadOptions {
        admit_upload: AdmitUpload::Async,
        ..cpu_opts(gpu, 32, Some(1.0))
    };
    let mut a = load(&path, &o);
    let (got_async, _) = sequence(&mut a, Some(&tokens));
    let sa = a.expert_stats().unwrap();
    eprintln!(
        "gpu async: pinned {:?}, admitted {}, admit {:?}, wait {:?}",
        a.admit_staging_pinned(),
        sa.io.admitted,
        sa.io.admit_time,
        sa.io.admit_wait
    );
    assert_bitwise(&got, &got_async, "gpu async against sync");
    let max_delta = |a: &[Vec<f32>], b: &[Vec<f32>]| {
        a.iter().zip(b).fold(0f32, |m, (x, y)| {
            x.iter().zip(y).fold(m, |m, (p, q)| m.max((p - q).abs()))
        })
    };
    let gpu_ref = sequence(&mut load(&path, &LoadOptions::new(gpu)), Some(&tokens)).0;
    let cpu_ref = sequence(&mut load(&path, &LoadOptions::new(CPU)), Some(&tokens)).0;
    let tol = TOL_GPU + 2.0 * max_delta(&gpu_ref, &cpu_ref);
    let worst = max_delta(&want, &got);
    let s = m.expert_stats().unwrap();
    eprintln!(
        "gpu: max |logit delta| {worst:e} (tolerance {tol:e}), admitted {}",
        s.io.admitted
    );
    assert!(s.io.admitted > 0);
    assert!(worst <= tol, "max |logit delta| {worst:e} above {tol:e}");
}
