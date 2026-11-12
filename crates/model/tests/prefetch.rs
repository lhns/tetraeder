//! Predicted next-layer expert prefetch (ADR 0038) on the K-quant fixture of `cpu_experts.rs`:
//! top-8 of 32 experts, Q4_K_M, `MissPolicy::Cpu`. Prefetch only moves reads earlier, so the
//! logits must equal the run without it bit for bit; the counters must add up.
//!
//! Hardware label: `cpu_*`, no GPU needed.

mod common;

use common::argmax;
use common::synth::{MoeSpec, quantized_moe};
use std::path::{Path, PathBuf};
use tetraeder_ggml_sys::llama::Ftype;
use tetraeder_model::{BackendChoice, ExpertStats, LoadOptions, MissPolicy, Qwen3Moe};

const CPU: BackendChoice = BackendChoice::Cpu { threads: Some(2) };
const N_PROMPT: usize = 24;
const N_DECODE: usize = 4;
const SLOTS: usize = 8;

fn fixture() -> PathBuf {
    quantized_moe(&MoeSpec::REALISH, Ftype::Q4KM, "q4km-prefetch")
}

fn prompt(a: i32, b: i32) -> Vec<i32> {
    (0..N_PROMPT as i32).map(|i| (i * a + b) % 256).collect()
}

fn opts(ram: usize, prefetch: bool) -> LoadOptions {
    LoadOptions {
        expert_slots: Some(SLOTS),
        ram_slots: Some(ram),
        miss: MissPolicy::Cpu,
        prefetch,
        ..LoadOptions::new(CPU)
    }
}

/// Prefill, `N_DECODE` greedy decodes, `reset`, a second prompt and decodes. Returns every
/// eval's logits.
fn sequence(m: &mut Qwen3Moe) -> Vec<Vec<f32>> {
    let mut all = Vec::new();
    for (a, b) in [(37, 5), (53, 11)] {
        let p = prompt(a, b);
        let l = m.eval(&p).unwrap();
        let mut next = argmax(&l[(N_PROMPT - 1) * m.n_vocab()..]) as i32;
        all.push(l);
        for _ in 0..N_DECODE {
            let l = m.eval(&[next]).unwrap();
            next = argmax(&l) as i32;
            all.push(l);
        }
        m.reset();
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

fn load(path: &Path, o: &LoadOptions) -> Qwen3Moe {
    Qwen3Moe::load(path, o).unwrap()
}

fn check_counters(s: &ExpertStats, what: &str) {
    let io = &s.io;
    assert!(
        io.prefetch_used + io.prefetch_wasted <= io.prefetch_issued,
        "{what}: used {} + wasted {} > issued {}",
        io.prefetch_used,
        io.prefetch_wasted,
        io.prefetch_issued
    );
    assert!(io.prefetch_bytes >= io.prefetch_wasted_bytes, "{what}");
    assert!(io.experts_read >= io.prefetch_issued, "{what}");
    for (il, l) in s.layers.iter().enumerate() {
        assert!(l.predicted_hits <= l.predicted_total, "{what}: layer {il}");
        assert!(
            l.predicted2_hits <= l.predicted2_total,
            "{what}: layer {il} depth 2"
        );
        if let Some(a) = l.prediction_accuracy() {
            assert!((0.0..=1.0).contains(&a), "{what}: layer {il} accuracy {a}");
        }
    }
}

/// hardware: cpu. Prefetch on and off give the same logits with 32, 12 and 4 RAM slots (the last
/// is below top-8, so the prefetch budget is smaller than a layer step), and the counters add up.
#[test]
fn cpu_prefetch_is_exact_and_counted() {
    let path = fixture();
    let want = sequence(&mut load(&path, &opts(32, false)));
    for ram in [32, 12, 4] {
        let mut off = load(&path, &opts(ram, false));
        let off_logits = sequence(&mut off);
        assert_bitwise(&want, &off_logits, &format!("ram {ram} off"));
        let off_io = off.expert_stats().unwrap().io;
        assert_eq!(off_io.prefetch_issued, 0);
        assert!(
            off.expert_stats()
                .unwrap()
                .layers
                .iter()
                .all(|l| l.prediction_accuracy().is_none())
        );

        let mut on = load(&path, &opts(ram, true));
        let got = sequence(&mut on);
        assert_bitwise(&off_logits, &got, &format!("ram {ram} on"));
        let s = on.expert_stats().unwrap();
        check_counters(&s, &format!("ram {ram}"));
        eprintln!(
            "ram {ram}: issued {} used {} wasted {} bytes {} accuracy {:?}",
            s.io.prefetch_issued,
            s.io.prefetch_used,
            s.io.prefetch_wasted,
            s.io.prefetch_bytes,
            s.layers
                .iter()
                .map(|l| l.prediction_accuracy())
                .collect::<Vec<_>>()
        );
        assert!(
            s.io.prefetch_issued > 0,
            "ram {ram}: something was prefetched"
        );
        assert!(
            s.io.prefetch_used > 0,
            "ram {ram}: some prefetched experts were routed"
        );
        assert!(
            s.layers[0].prediction_accuracy().is_none(),
            "layer 0 has no prediction"
        );
        assert!(
            s.layers[1..].iter().all(|l| l.predicted_total > 0),
            "ram {ram}: every later layer was scored"
        );
    }
}

/// hardware: cpu. Prefetch with the upload policy (no RAM tier, nothing to prefetch) runs no
/// prediction at all: nothing is scored, nothing is issued, and the logits are unchanged.
#[test]
fn cpu_upload_policy_runs_no_prediction() {
    let path = fixture();
    let plain = LoadOptions {
        expert_slots: Some(SLOTS),
        ..LoadOptions::new(CPU)
    };
    let want = sequence(&mut load(&path, &plain));
    for depth in [1, 2] {
        let o = LoadOptions {
            prefetch: true,
            prefetch_depth: depth,
            ..plain.clone()
        };
        let mut m = load(&path, &o);
        assert_bitwise(&want, &sequence(&mut m), "upload policy");
        let s = m.expert_stats().unwrap();
        assert_eq!(s.io.prefetch_issued, 0);
        assert!(
            s.layers
                .iter()
                .all(|l| l.predicted_total == 0 && l.predicted2_total == 0),
            "depth {depth}: no prediction under the upload policy"
        );
        check_counters(&s, "upload policy");
    }
}

/// hardware: cpu. Depth 2 gives the logits of the run without prefetch with 32, 12 and 4 RAM
/// slots, the counters add up, and both predictions are scored where they exist.
#[test]
fn cpu_depth2_is_exact_and_counted() {
    let path = fixture();
    let want = sequence(&mut load(&path, &opts(32, false)));
    for ram in [32, 12, 4] {
        let o = LoadOptions {
            prefetch_depth: 2,
            ..opts(ram, true)
        };
        let mut m = load(&path, &o);
        assert_bitwise(&want, &sequence(&mut m), &format!("depth 2, ram {ram}"));
        let s = m.expert_stats().unwrap();
        check_counters(&s, &format!("depth 2, ram {ram}"));
        eprintln!(
            "depth 2, ram {ram}: issued {} used {} wasted {} accuracy2 {:?}",
            s.io.prefetch_issued,
            s.io.prefetch_used,
            s.io.prefetch_wasted,
            s.layers
                .iter()
                .map(|l| l.prediction2_accuracy())
                .collect::<Vec<_>>()
        );
        assert!(s.io.prefetch_issued > 0, "ram {ram}");
        assert!(s.io.prefetch_used > 0, "ram {ram}");
        assert!(s.layers[1..].iter().all(|l| l.predicted_total > 0));
        assert!(s.layers[..2].iter().all(|l| l.predicted2_total == 0));
        assert!(s.layers[2..].iter().all(|l| l.predicted2_total > 0));
    }
}

/// hardware: cpu. A depth-1 run has no depth-2 score.
#[test]
fn cpu_depth1_scores_no_depth2() {
    let path = fixture();
    let mut m = load(&path, &opts(12, true));
    sequence(&mut m);
    let s = m.expert_stats().unwrap();
    assert!(s.layers.iter().all(|l| l.predicted2_total == 0));
}

/// hardware: cpu. The polling level and the thread count while a read is in flight change speed,
/// not results.
#[test]
fn cpu_poll_and_prefetch_threads_are_exact() {
    let path = fixture();
    let want = sequence(&mut load(&path, &opts(12, false)));
    for (poll, threads, depth) in [
        (Some(0), None, 1),
        (Some(100), None, 2),
        (None, Some(1), 1),
        (Some(0), Some(1), 2),
        // More than the pool has is clamped.
        (None, Some(64), 1),
    ] {
        let o = LoadOptions {
            cpu_poll: poll,
            prefetch_threads: threads,
            prefetch_depth: depth,
            ..opts(12, true)
        };
        let mut m = load(&path, &o);
        let what = format!("poll {poll:?} threads {threads:?} depth {depth}");
        assert_bitwise(&want, &sequence(&mut m), &what);
        check_counters(&m.expert_stats().unwrap(), &what);
    }
}

/// hardware: cpu. The automatic polling level (ADR 0061) follows this machine's quota, an
/// explicit level wins, and no level changes the logits (bitwise on the CPU backend).
#[test]
fn cpu_poll_auto_and_explicit_levels_are_exact() {
    let path = fixture();
    let mut auto = load(&path, &opts(12, false));
    let p = auto.cpu_poll();
    assert!(p.auto);
    assert_eq!(p, tetraeder_model::cpu_poll::detect(None, 2));
    let want = sequence(&mut auto);
    for level in [0, 50, 100] {
        let o = LoadOptions {
            cpu_poll: Some(level),
            ..opts(12, false)
        };
        let mut m = load(&path, &o);
        let p = m.cpu_poll();
        assert_eq!((p.level, p.auto), (level, false));
        assert_bitwise(&want, &sequence(&mut m), &format!("poll {level}"));
    }
}

/// hardware: cpu. A prefetch depth other than 1 or 2 is refused at load.
#[test]
fn cpu_invalid_prefetch_depth_is_rejected() {
    let path = fixture();
    for depth in [0, 3] {
        let o = LoadOptions {
            prefetch_depth: depth,
            ..opts(12, true)
        };
        let err = Qwen3Moe::load(&path, &o).unwrap_err().to_string();
        assert!(err.contains("prefetch_depth"), "depth {depth}: {err}");
    }
}

/// hardware: cpu. The hot set refresh and `reset` after prefetching work, and nothing stays in
/// flight: a second pass equals a model that never prefetched.
#[test]
fn cpu_hot_set_refresh_and_reset_after_prefetch() {
    let path = fixture();
    let run = |prefetch: bool| {
        let mut m = load(&path, &opts(12, prefetch));
        let mut all = sequence(&mut m);
        m.refresh_hot_set().unwrap();
        m.reset();
        all.extend(sequence(&mut m));
        (all, m)
    };
    let (want, _) = run(false);
    let (got, m) = run(true);
    assert_bitwise(&want, &got, "refresh");
    check_counters(&m.expert_stats().unwrap(), "refresh");
}

/// hardware: cpu. A read that fails (the file is cut off) surfaces as an error from `eval` with
/// prefetch on, and after the file is restored the model gives the same logits again.
#[test]
fn cpu_read_error_surfaces_and_the_model_recovers() {
    let src = fixture();
    let path = src.with_extension("cut.gguf");
    let original = std::fs::read(&src).unwrap();
    std::fs::write(&path, &original).unwrap();
    let want = sequence(&mut load(&path, &opts(12, false)));

    let mut m = load(&path, &opts(12, true));
    let p = prompt(37, 5);
    m.eval(&p).unwrap();
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
    let failed = (0..4).any(|i| m.eval(&prompt(53 + i, 11)).is_err());
    assert!(failed, "a read of the cut file fails");
    std::fs::write(&path, &original).unwrap();
    m.reset();
    assert_bitwise(&want, &sequence(&mut m), "after the error");
    let _ = std::fs::remove_file(path);
}
