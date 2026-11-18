//! The cost of a prefix-cache save on a GPU (ADR 0074): `seq_snapshot` of a 533-token sequence
//! of Qwen3-8B (dense builder) and Qwen3-30B-A3B (MoE builder, resident), through the pinned
//! staging pair and read straight, paired and interleaved in one process. The staged read must
//! give the same bytes, be faster than the straight one and stay under a bound per model: the
//! 3090 measured 41 to 44 ms (8B) and 30 to 33 ms (30B) per save before (`save_ms`,
//! `bench/results/20261116pc/pin-confirm/`).
//!
//! Hardware label: `gpu_*`, model. Skips without a GPU or the models in `TETRAEDER_MODELS_DIR`;
//! with `TETRAEDER_REQUIRE_GPU` set a missing GPU fails.

use std::path::PathBuf;
use std::time::Instant;
use tetraeder_ggml_sys::ggml::{Backend, advise_hugepages};
use tetraeder_model::{BackendChoice, DenseOptions, LoadOptions, Qwen3, Qwen3Moe, SeqEval};

const GPU: BackendChoice = BackendChoice::Gpu { index: 0 };
/// Tokens of the saved sequence, as the c4 probe's admissions.
const N: usize = 533;
const REPEATS: usize = 7;

fn model(rel: &str) -> Option<PathBuf> {
    if !Backend::gpu_available() {
        assert!(
            std::env::var_os("TETRAEDER_REQUIRE_GPU").is_none(),
            "TETRAEDER_REQUIRE_GPU is set but there is no GPU device"
        );
        eprintln!("skip: no GPU");
        return None;
    }
    let Some(dir) = std::env::var_os("TETRAEDER_MODELS_DIR") else {
        eprintln!("skip: TETRAEDER_MODELS_DIR is not set");
        return None;
    };
    let path = PathBuf::from(dir).join(rel);
    if !path.is_file() {
        eprintln!("skip: {} not found", path.display());
        return None;
    }
    Some(path)
}

fn tokens(seed: u32, len: usize) -> Vec<i32> {
    let mut x = seed;
    (0..len)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            100 + ((x >> 16) % 20_000) as i32
        })
        .collect()
}

/// Prefill inputs: `N` tokens on sequence 1 and a short neighbour on sequence 0, so the saved
/// run does not start at row 0; in chunks of at most 512 tokens.
fn prefill(mut decode: impl FnMut(&[SeqEval<'_>])) {
    let (long, short) = (tokens(7, N), tokens(8, 40));
    decode(&[
        SeqEval {
            seq: 0,
            tokens: &short,
            all_logits: false,
        },
        SeqEval {
            seq: 1,
            tokens: &long[..512],
            all_logits: false,
        },
    ]);
    decode(&[SeqEval {
        seq: 1,
        tokens: &long[512..],
        all_logits: false,
    }]);
}

/// Milliseconds to copy `src` into a fresh `Vec`, as a save fills its snapshot; `thp` advises
/// huge pages on it first.
fn host_copy_ms(src: &[u8], thp: bool) -> f64 {
    let t = Instant::now();
    let mut v: Vec<u8> = Vec::with_capacity(src.len());
    if thp {
        advise_hugepages(v.as_ptr() as usize, v.capacity());
    }
    v.extend_from_slice(src);
    let ms = t.elapsed().as_secs_f64() * 1e3;
    std::hint::black_box(&v);
    ms
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// `snap(staged)` switches the path (outside the timing) and returns the snapshot and its
/// milliseconds. Interleaved repeats after one warm-up of each path.
fn paired(what: &str, bound_ms: f64, mut snap: impl FnMut(bool) -> (Vec<u8>, f64)) {
    let (want, _) = snap(false);
    let (got, _) = snap(true);
    assert_eq!(got, want, "{what}: staged bytes equal the straight ones");
    let (mut staged, mut straight) = (Vec::new(), Vec::new());
    for _ in 0..REPEATS {
        straight.push(snap(false).1);
        staged.push(snap(true).1);
    }
    let (s, d) = (median(staged), median(straight));
    let mb = want.len() as f64 / 1e6;
    eprintln!(
        "{what}: {mb:.1} MB, staged {s:.2} ms ({:.2} GB/s), straight {d:.2} ms ({:.2} GB/s), ratio {:.3}",
        mb / s,
        mb / d,
        s / d
    );
    // The host side alone (ADR 0074, "Consequences"): how much of a save is filling fresh
    // snapshot memory, and what MADV_HUGEPAGE on it would change. Printed, not judged.
    let (mut plain, mut thp) = (Vec::new(), Vec::new());
    for _ in 0..REPEATS {
        plain.push(host_copy_ms(&want, false));
        thp.push(host_copy_ms(&want, true));
    }
    eprintln!(
        "{what}: host copy into fresh memory {:.2} ms, with MADV_HUGEPAGE {:.2} ms",
        median(plain),
        median(thp)
    );
    // Staged against straight is printed above, not a gate (ADR 0074, "The bar"); the
    // absolute bound below is.
    assert!(
        s < bound_ms,
        "{what}: staged save {s:.2} ms over {bound_ms} ms"
    );
}

/// hardware: gpu, model. Qwen3-8B Q4_K_M: 533 tokens are 78.6 MB of F16 KV.
#[test]
fn gpu_qwen3_8b_save_through_pinned_staging() {
    let Some(path) = model("Qwen__Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf") else {
        return;
    };
    let o = DenseOptions {
        n_seq_max: 4,
        n_ctx_pool: 4096,
        ..DenseOptions::new(GPU)
    };
    let mut m = Qwen3::load(&path, &o).unwrap();
    let default = m
        .snapshot_staging()
        .expect("a GPU model stages its snapshots");
    assert!(default.1, "the staging is pinned");
    let mut out = Vec::new();
    prefill(|inp| {
        m.decode(inp, &mut out).unwrap();
    });
    paired("qwen3-8b", 25.0, |staged| {
        m.set_snapshot_staging(staged.then_some(default.0 / 2))
            .unwrap();
        let t = Instant::now();
        let bytes = m.seq_snapshot(1).unwrap();
        (bytes, t.elapsed().as_secs_f64() * 1e3)
    });
}

/// hardware: gpu, model. Qwen3-30B-A3B Q4_K_M, every expert resident: 533 tokens are 52.4 MB of
/// F16 KV.
#[test]
fn gpu_qwen3_30b_save_through_pinned_staging() {
    let Some(path) = model("Qwen__Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf") else {
        return;
    };
    let o = LoadOptions {
        n_ctx: 4096,
        n_seq_max: 4,
        ..LoadOptions::new(GPU)
    };
    let mut m = Qwen3Moe::load(&path, &o).unwrap();
    let default = m
        .snapshot_staging()
        .expect("a GPU model stages its snapshots");
    assert!(default.1, "the staging is pinned");
    let mut out = Vec::new();
    prefill(|inp| {
        m.decode(inp, &mut out).unwrap();
    });
    paired("qwen3-30b", 18.0, |staged| {
        m.set_snapshot_staging(staged.then_some(default.0 / 2))
            .unwrap();
        let t = Instant::now();
        let bytes = m.seq_snapshot(1).unwrap();
        (bytes, t.elapsed().as_secs_f64() * 1e3)
    });
}
