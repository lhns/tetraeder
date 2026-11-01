//! CPU expert groups and the pinned hot set (ADR 0035, stages 2 and 3) on the K-quant fixture of
//! `tests/multipass.rs`: top-8 of 32 experts, Q4_K_M. Under `MissPolicy::Cpu` a routed expert
//! that is not in a compute slot is computed on the CPU from its RAM slot, in the same segment
//! as the compute-slot hits of its layer. On the CPU backend both tiers are CPU tensors, so the
//! groups must equal the resident builder bit for bit; on a GPU the scheduler splits the
//! segment between the backends and only top-1 and a tolerance are checked.
//!
//! The CPU's tiled K-quant kernel exists only in AVX builds; without AVX the exactness tests pass
//! either way (see `multipass.rs`).
//!
//! Hardware labels (AGENTS rule 9): `cpu_*` tests need no GPU; `gpu_*` tests skip cleanly when the
//! build has no GPU device.

mod common;

use common::synth::{MoeSpec, quantized_moe};
use common::{TOL_GPU, argmax};
use std::path::{Path, PathBuf};
use tetraeder_ggml_sys::ggml::Backend;
use tetraeder_ggml_sys::llama::Ftype;
use tetraeder_model::{BackendChoice, ExpertStats, LoadOptions, MissPolicy, Qwen3Moe};

const CPU: BackendChoice = BackendChoice::Cpu { threads: Some(2) };
const N_PROMPT: usize = 24;
const N_DECODE: usize = 3;
const SLOTS: usize = 8;

fn fixture() -> PathBuf {
    quantized_moe(&MoeSpec::REALISH, Ftype::Q4KM, "q4km")
}

fn prompt(a: i32, b: i32) -> Vec<i32> {
    (0..N_PROMPT as i32).map(|i| (i * a + b) % 256).collect()
}

fn opts(backend: BackendChoice, cpu_policy: Option<usize>) -> LoadOptions {
    match cpu_policy {
        None => LoadOptions::new(backend),
        Some(ram) => LoadOptions {
            expert_slots: Some(SLOTS),
            ram_slots: Some(ram),
            miss: MissPolicy::Cpu,
            ..LoadOptions::new(backend)
        },
    }
}

fn uploads(m: &Qwen3Moe) -> u64 {
    m.expert_stats().map_or(0, |s| s.io.uploads)
}

/// Prefill of `prompt`, then `N_DECODE` decode steps. The tokens fed to the decode steps are
/// `forced` when given, else the model's own greedy choice. Returns the logits of every eval and
/// the decode tokens.
fn run_prompt(
    m: &mut Qwen3Moe,
    prompt: &[i32],
    forced: Option<&[i32]>,
) -> (Vec<Vec<f32>>, Vec<i32>) {
    let n_vocab = m.n_vocab();
    let prefill = m.eval(prompt).unwrap();
    let mut next = argmax(&prefill[(prompt.len() - 1) * n_vocab..]) as i32;
    let mut logits = vec![prefill];
    let mut tokens = Vec::new();
    for i in 0..N_DECODE {
        if let Some(f) = forced {
            next = f[i];
        }
        tokens.push(next);
        let l = m.eval(&[next]).unwrap();
        next = argmax(&l) as i32;
        logits.push(l);
    }
    (logits, tokens)
}

/// Prompt A, decode, `reset`, prompt B (after the hot set), decode.
fn sequence(m: &mut Qwen3Moe, forced: Option<&[i32]>) -> (Vec<Vec<f32>>, Vec<i32>) {
    let (mut logits, mut tokens) = run_prompt(m, &prompt(37, 5), forced.map(|f| &f[..N_DECODE]));
    m.reset();
    let (l, t) = run_prompt(m, &prompt(53, 11), forced.map(|f| &f[N_DECODE..]));
    logits.extend(l);
    tokens.extend(t);
    (logits, tokens)
}

fn load(path: &Path, o: &LoadOptions) -> Qwen3Moe {
    Qwen3Moe::load(path, o).unwrap()
}

fn sum(s: &ExpertStats, f: impl Fn(&tetraeder_model::LayerExpertStats) -> u64) -> u64 {
    s.layers.iter().map(f).sum()
}

/// hardware: cpu. Resident builder against `Cpu` with 8 compute slots and 32, 12 and 4 RAM
/// slots: the first prefill runs every expert on CPU groups (the compute tier is empty), later
/// steps mix compute-slot hits and CPU groups. All logits are bitwise equal.
#[test]
fn cpu_cpu_groups_are_exact() {
    let path = fixture();
    let mut base = load(&path, &opts(CPU, None));
    let (want, tokens) = sequence(&mut base, None);
    for ram in [32, 12, 4] {
        let mut m = load(&path, &opts(CPU, Some(ram)));
        let (got, _) = sequence(&mut m, Some(&tokens));
        let s = m.expert_stats().unwrap();
        eprintln!(
            "ram {ram}: groups {}, group graphs {}, cpu experts {}, compute hits {}, mixed steps {}, uploads {}, experts read {}",
            m.expert_groups(),
            m.group_graphs(),
            sum(&s, |l| l.cpu),
            sum(&s, |l| l.compute.hits),
            sum(&s, |l| l.mixed_steps),
            s.io.uploads,
            s.io.experts_read,
        );
        assert!(m.expert_groups() > 0, "ram {ram}: CPU groups ran");
        assert!(
            sum(&s, |l| l.cpu) > 0,
            "ram {ram}: experts were counted as cpu"
        );
        assert!(
            sum(&s, |l| l.mixed_steps) > 0 && sum(&s, |l| l.compute.hits) > 0,
            "ram {ram}: after the hot set, compute hits and CPU experts share layer steps"
        );
        assert!(s.io.uploads > 0, "ram {ram}: the hot set was uploaded");
        for (i, (a, b)) in want.iter().zip(&got).enumerate() {
            let bitwise = a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
            let delta = a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            assert!(
                bitwise,
                "ram {ram}: eval {i} differs, max |delta| {delta:e}"
            );
        }
        if ram < 12 {
            assert!(
                m.group_graphs() > 0,
                "ram {ram}: several CPU rounds need group graphs"
            );
        }
    }
}

/// hardware: cpu. With the hot set in the compute slots, decode issues no uploads, and with every
/// expert in the RAM tier a second prefill of the same prompt reads nothing from the file.
#[test]
fn cpu_decode_with_hot_set_issues_no_uploads() {
    let path = fixture();
    let mut m = load(&path, &opts(CPU, Some(32)));
    let p = prompt(37, 5);
    let logits = m.eval(&p).unwrap();
    let after_prefill = m.expert_stats().unwrap().io;
    assert!(
        after_prefill.uploads > 0,
        "the first prefill built the hot set"
    );
    let mut next = argmax(&logits[(N_PROMPT - 1) * m.n_vocab()..]) as i32;
    for _ in 0..8 {
        next = argmax(&m.eval(&[next]).unwrap()) as i32;
    }
    let after_decode = m.expert_stats().unwrap().io;
    eprintln!(
        "uploads {} -> {}",
        after_prefill.uploads, after_decode.uploads
    );
    assert_eq!(
        after_decode.uploads, after_prefill.uploads,
        "decode uploads nothing"
    );
    m.reset();
    m.eval(&p).unwrap();
    let again = m.expert_stats().unwrap().io;
    assert_eq!(
        again.experts_read, after_decode.experts_read,
        "a second prefill reads nothing"
    );
    assert_eq!(again.uploads, after_prefill.uploads);
}

/// hardware: cpu. The routing counts of one model, as a profile, give another model its hot
/// set at load: uploads before the first eval, compute hits in the first prefill, no second
/// hot set afterwards, and the same logits as the first model's second run.
#[test]
fn cpu_routing_profile_sets_the_hot_set_at_load() {
    let path = fixture();
    let p = prompt(37, 5);
    let mut first = load(&path, &opts(CPU, Some(32)));
    first.eval(&p).unwrap();
    let profile = first.routing_counts();
    assert_eq!(profile.len(), first.hparams().n_layer as usize);
    first.reset();
    let want = first.eval(&p).unwrap();

    let o = LoadOptions {
        routing_profile: Some(profile),
        ..opts(CPU, Some(32))
    };
    let mut second = load(&path, &o);
    let at_load = uploads(&second);
    assert!(at_load > 0, "the profile uploaded the hot set at load");
    let got = second.eval(&p).unwrap();
    let s = second.expert_stats().unwrap();
    eprintln!(
        "at load {at_load} uploads, compute hits {}, cpu {}",
        sum(&s, |l| l.compute.hits),
        sum(&s, |l| l.cpu)
    );
    assert!(
        sum(&s, |l| l.compute.hits) > 0,
        "the first prefill already hits"
    );
    assert_eq!(
        s.io.uploads, at_load,
        "no hot set refresh after the first prefill"
    );
    assert!(
        got.iter()
            .zip(&want)
            .all(|(x, y)| x.to_bits() == y.to_bits())
    );
}

/// hardware: cpu. `Cpu` without RAM slots is a load error, and the hot set needs `Cpu`.
#[test]
fn cpu_miss_policy_cpu_needs_ram_slots() {
    let path = fixture();
    let o = LoadOptions {
        expert_slots: Some(SLOTS),
        miss: MissPolicy::Cpu,
        ..LoadOptions::new(CPU)
    };
    let err = Qwen3Moe::load(&path, &o).expect_err("a load error");
    assert!(err.to_string().contains("ram_slots"), "{err}");
    let mut upload = load(&path, &opts(CPU, None));
    assert!(upload.refresh_hot_set().is_err());
}

/// hardware: gpu (skips without a GPU device). The resident GPU builder against `Cpu` with 8
/// compute slots and 32 RAM slots on the same GPU: same top-1 everywhere, logits within a
/// tolerance (CPU and GPU kernels round differently), decode after the hot set uploads nothing,
/// and the scheduler splits a mixed segment between the backends.
#[test]
fn gpu_cpu_groups_match_resident() {
    if !Backend::gpu_available() {
        eprintln!("skip: no GPU device in this build or on this machine");
        return;
    }
    let gpu = BackendChoice::Gpu { index: 0 };
    let path = fixture();
    let (want, tokens) = {
        let mut base = load(&path, &opts(gpu, None));
        sequence(&mut base, None)
    };
    let mut m = load(&path, &opts(gpu, Some(32)));
    let (mut got, _) = run_prompt(&mut m, &prompt(37, 5), Some(&tokens[..N_DECODE]));
    let after_hot = uploads(&m);
    m.reset();
    got.extend(run_prompt(&mut m, &prompt(53, 11), Some(&tokens[N_DECODE..])).0);
    let n_vocab = m.n_vocab();
    let scale = want.iter().flatten().fold(0f32, |a, x| a.max(x.abs()));
    let tol = TOL_GPU + 1e-2 * scale;
    let mut worst = 0f32;
    let mut flips = Vec::new();
    for (i, (a, b)) in want.iter().zip(&got).enumerate() {
        for (pos, (x, y)) in a.chunks(n_vocab).zip(b.chunks(n_vocab)).enumerate() {
            if argmax(x) != argmax(y) {
                // a flip is a rounding effect only if the resident top-1 was a near tie
                let mut sorted = x.to_vec();
                sorted.sort_by(|p, q| q.total_cmp(p));
                flips.push((i, pos, argmax(x), argmax(y), sorted[0] - sorted[1]));
            }
        }
        worst = worst.max(
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max),
        );
    }
    let s = m.expert_stats().unwrap();
    eprintln!(
        "gpu: max |logit delta| {worst:e} (tolerance {tol:e}, logit scale {scale}), max splits {}, groups {}, uploads {} (after the first hot set {after_hot}), cpu experts {}, compute hits {}",
        m.max_splits(),
        m.expert_groups(),
        s.io.uploads,
        sum(&s, |l| l.cpu),
        sum(&s, |l| l.compute.hits),
    );
    eprintln!("top-1 flips (eval, position, resident, got, resident margin): {flips:?}");
    assert!(worst <= tol, "max |logit delta| {worst:e} above {tol:e}");
    assert!(flips.is_empty(), "top-1 differs: {flips:?}");
    assert_eq!(
        s.io.uploads, after_hot,
        "decode and the second prefill upload nothing"
    );
    assert!(m.max_splits() > 1, "a mixed segment runs on both backends");
}
