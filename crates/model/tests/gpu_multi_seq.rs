//! `multi_seq.rs` on a GPU (ADR 0065, 0067; phase 4, C4): several sequences in one step, resident
//! and streamed experts, `seq_cp`, snapshots and a chunked move mid-decode, within `GPU_TOL` of
//! each sequence alone (ADR 0071). The MoE fixture's head size of 16 has no CUDA flash-attention
//! kernel, so its attention runs on the CPU: CUDA flash attention with sequences mixed in one
//! query tile (ADR 0065, "To verify on the box") is checked by `gpu_dense.rs` (head 64).
//!
//! Hardware label: `gpu_*` (a GPU device in the build); skips cleanly without one, and fails
//! when `TETRAEDER_REQUIRE_GPU` is set.

mod common;

use common::multi_seq::*;
use common::synth::{MoeSpec, quantized_moe};
use common::{
    GPU_TOL, Logits, TOL_CPU, Tol, compare_scripts_tol, fixture, libllama_script_on, part,
    print_script_rows, run_all,
};
use tetraeder_ggml_sys::ggml::Backend;
use tetraeder_ggml_sys::llama::Ftype;
use tetraeder_model::{BackendChoice, LoadOptions, MissPolicy, RoutingStep};
use tetraeder_plan::KvFormat;

fn base() -> Option<LoadOptions> {
    if !Backend::gpu_available() {
        // The box's gpu-parity stage sets this: no GPU there is a failure, not a skip.
        assert!(
            std::env::var_os("TETRAEDER_REQUIRE_GPU").is_none(),
            "TETRAEDER_REQUIRE_GPU is set but there is no GPU device"
        );
        eprintln!("skip: no GPU device in this build or on this machine");
        return None;
    }
    Some(LoadOptions {
        n_ctx: 1024,
        ..LoadOptions::new(BackendChoice::Gpu { index: 0 })
    })
}

/// hardware: gpu.
#[test]
fn gpu_resident_batched_equals_alone() {
    let Some(o) = base() else { return };
    batched_equals_alone(&fixture(), &o, GPU_TOL, "gpu resident");
}

/// hardware: gpu. 2 of 8 experts per layer on the card, uploads on a miss.
#[test]
fn gpu_streamed_upload_batched_equals_alone() {
    let Some(o) = base() else { return };
    let o = LoadOptions {
        expert_slots: Some(2),
        ..o
    };
    let m = batched_equals_alone(&fixture(), &o, GPU_TOL, "gpu upload");
    assert!(m.expert_stats().expect("streaming").io.experts_read > 0);
}

/// hardware: gpu. A RAM tier; the misses run as CPU groups next to the GPU segments.
#[test]
fn gpu_streamed_cpu_miss_batched_equals_alone() {
    let Some(o) = base() else { return };
    let o = LoadOptions {
        expert_slots: Some(2),
        ram_slots: Some(3),
        miss: MissPolicy::Cpu,
        ..o
    };
    batched_equals_alone(&fixture(), &o, GPU_TOL, "gpu cpu miss");
}

/// hardware: gpu.
#[test]
fn gpu_seq_cp_snapshot_and_move() {
    let Some(o) = base() else { return };
    let big = LoadOptions {
        n_ctx: 2048,
        ..o.clone()
    };
    // The copy kernel on a quantized cache.
    let path = quantized_moe(&MoeSpec::REALISH, Ftype::Q8_0, "kv-q8");
    let q = LoadOptions {
        n_ctx: 2048,
        kv: KvFormat::Q8_0,
        kv_rotation: Some(true),
        ..o.clone()
    };
    run_all(vec![
        part("seq_cp", || {
            seq_cp_continues(&fixture(), &o, GPU_TOL, "gpu")
        }),
        part("snapshot", || {
            snapshot_roundtrip(&fixture(), &o, GPU_TOL, "gpu")
        }),
        part("move", || move_mid_decode(&fixture(), &big, GPU_TOL, "gpu")),
        part("move q8_0 kv", || {
            move_mid_decode(&path, &q, GPU_TOL, "gpu q8_0 kv")
        }),
    ]);
}

/// hardware: gpu. The same batched steps on the GPU builder and on the CPU builder, anchored on
/// libllama's own CUDA-against-CPU gap on the same steps and tokens (ADR 0071). CUDA and the CPU
/// run other kernels, and at b12426 the CPU's F16 matmuls keep src1 in F32 (#31854), so that gap
/// is llama.cpp's, not ours: libllama on each backend is the reference. Asserted: the builder
/// against libllama on the same backend and shapes (strict: `GPU_TOL` on CUDA, `TOL_CPU` on the
/// CPU), and the builder's CUDA-against-CPU gap within 1.5x libllama's own. Every table and the
/// routing difference of the two builders are printed first.
#[test]
fn gpu_batched_matches_the_cpu_builder() {
    let Some(o) = base() else { return };
    let cpu = LoadOptions {
        backend: BackendChoice::Cpu { threads: Some(2) },
        ..o.clone()
    };
    match tetraeder_ggml_sys::cpu::backend_info() {
        Ok(i) => eprintln!("cpu backend: tier {}, isa {:?}", i.tier, i.isa),
        Err(e) => eprintln!("cpu backend: {e}"),
    }
    let path = fixture();
    let prompts = [prompt(1, 7), prompt(2, 21), prompt(3, 24)];
    let alones: Vec<Alone> = prompts.iter().map(|p| alone(&path, &cpu, p, 6)).collect();
    let toks: Vec<Vec<i32>> = alones.iter().map(|a| a.toks.clone()).collect();
    let lens: Vec<usize> = prompts.iter().map(Vec::len).collect();
    let total: Vec<usize> = toks.iter().map(Vec::len).collect();
    let steps = schedule(&lens, &total, 16);
    let ours = |opts: &LoadOptions| {
        steps_logits(run_steps(&mut load(&path, opts, 3), &toks, &steps), &steps)
    };
    let (b_gpu, b_cpu) = (ours(&o), ours(&cpu));
    let items = script_items(&steps);
    let ll_gpu = libllama_script_on(&path, &toks, &items, o.kv, -1);
    let ll_cpu = libllama_script_on(&path, &toks, &items, o.kv, 0);
    // Each sequence alone on the CPU, keyed like the scripts; `only` keeps the rows of a script
    // that have an alone row (steps that end at or after the prompt).
    let mut a_cpu = Logits::new();
    for (k, a) in alones.iter().enumerate() {
        for (j, l) in a.logits.iter().enumerate() {
            a_cpu.insert((k as u32, a.prompt_len + j - 1), l.clone());
        }
    }
    let only = |l: &Logits| -> Logits {
        l.iter()
            .filter(|(key, _)| a_cpu.contains_key(*key))
            .map(|(key, v)| (*key, v.clone()))
            .collect()
    };
    let (ll_gpu_a, b_gpu_a, b_cpu_a) = (only(&ll_gpu), only(&b_gpu), only(&b_cpu));

    let pairs: [(&str, &Logits, &Logits); 7] = [
        ("libllama cuda / libllama cpu", &ll_cpu, &ll_gpu),
        ("libllama cuda / alone on cpu", &a_cpu, &ll_gpu_a),
        ("builder cuda / libllama cuda", &ll_gpu, &b_gpu),
        ("builder cpu / libllama cpu", &ll_cpu, &b_cpu),
        ("builder cpu / alone on cpu", &a_cpu, &b_cpu_a),
        ("builder cuda / builder cpu", &b_cpu, &b_gpu),
        ("builder cuda / alone on cpu", &a_cpu, &b_gpu_a),
    ];
    let worst: Vec<f32> = pairs
        .iter()
        .map(|(name, want, have)| print_script_rows(&format!("gpu {name}"), want, have))
        .collect();
    for ((name, _, _), w) in pairs.iter().zip(&worst) {
        eprintln!("gpu {name}: worst {w:.2e} of the largest logit");
    }
    // A diagnostic: its failure must not hide the assertions below.
    let diff = || print_routing_diff(&path, &o, &cpu, &toks, &steps);
    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(diff)).is_err() {
        eprintln!("gpu routing: the routing diagnostic panicked");
    }

    // libllama's own CUDA-against-CPU gap (batched, and against the CPU alone runs), widened
    // 1.5x, bounds the builder's; a builder bug shows in the strict same-backend pairs.
    let own = Tol {
        abs: GPU_TOL.abs,
        rel: GPU_TOL.rel.max(1.5 * worst[0].max(worst[1])),
    };
    eprintln!("gpu cuda / cpu bound: {own:?}");
    let check = |i: usize, tol: Tol| {
        let (name, want, have) = pairs[i];
        move || {
            compare_scripts_tol(want, have, tol, &format!("gpu {name}"));
        }
    };
    run_all(vec![
        part("builder cuda / libllama cuda", check(2, GPU_TOL)),
        part("builder cpu / libllama cpu", check(3, Tol::from(TOL_CPU))),
        part("builder cuda / builder cpu", check(5, own)),
        part("builder cuda / alone on cpu", check(6, own)),
    ]);
}

/// Prints where the GPU and the CPU builder route a token to other experts on `steps`, on models
/// of their own with `record_routing` (the asserted runs keep their graphs). Print only: a
/// near-tied router that flips on one backend moves that token's FFN output, and every later
/// row of its sequence sees it through attention.
fn print_routing_diff(
    path: &std::path::Path,
    gpu: &LoadOptions,
    cpu: &LoadOptions,
    toks: &[Vec<i32>],
    steps: &[Step],
) {
    // The log of each step on its own: entries are matched by layer, on the rows of the step.
    let log = |opts: &LoadOptions| -> Vec<Vec<RoutingStep>> {
        let o = LoadOptions {
            record_routing: true,
            ..opts.clone()
        };
        let mut m = load(path, &o, 3);
        (0..steps.len())
            .map(|si| {
                run_steps(&mut m, toks, &steps[si..=si]);
                m.take_routing_log()
                    .into_iter()
                    .filter(|r| !r.refresh)
                    .collect()
            })
            .collect()
    };
    let (g, c) = (log(gpu), log(cpu));
    let sorted = |v: &[i32]| {
        let mut v = v.to_vec();
        v.sort_unstable();
        v
    };
    let mut differ = vec![0usize; toks.len()];
    let mut first: Vec<Option<(usize, usize, usize)>> = vec![None; toks.len()];
    let mut printed = 0;
    for (si, (gs, cs)) in g.iter().zip(&c).enumerate() {
        let owners: Vec<(usize, usize)> = steps[si]
            .iter()
            .flat_map(|(k, r)| r.clone().map(move |p| (*k, p)))
            .collect();
        let full = |r: &&RoutingStep| r.n_tokens == owners.len();
        let layers: Vec<usize> = gs.iter().filter(full).map(|r| r.layer).collect();
        if layers.is_empty() || layers.len() != cs.iter().filter(full).count() {
            eprintln!(
                "gpu routing: step {si}: {} cuda and {} cpu entries ({} and {} on all {} rows), not compared",
                gs.len(),
                cs.len(),
                layers.len(),
                cs.iter().filter(full).count(),
                owners.len()
            );
            continue;
        }
        for x in gs.iter().filter(full) {
            let Some(y) = cs.iter().filter(full).find(|y| y.layer == x.layer) else {
                eprintln!("gpu routing: step {si}, layer {}: no cpu entry", x.layer);
                continue;
            };
            let nu = x.ids.len() / x.n_tokens.max(1);
            let same = y.ids.len() == x.ids.len()
                && x.weights.len() == x.ids.len()
                && y.weights.len() == y.ids.len();
            if nu == 0 || !same {
                eprintln!("gpu routing: step {si}, layer {}: other layouts", x.layer);
                continue;
            }
            for (t, &(k, p)) in owners.iter().enumerate() {
                let span = t * nu..(t + 1) * nu;
                if sorted(&x.ids[span.clone()]) == sorted(&y.ids[span.clone()]) {
                    continue;
                }
                differ[k] += 1;
                if first[k].is_none() {
                    first[k] = Some((si, x.layer, p));
                }
                if printed < 24 {
                    printed += 1;
                    eprintln!(
                        "gpu routing: step {si}, layer {}, sequence {k}, position {p}: cuda {:?} weights {:?}, cpu {:?} weights {:?}",
                        x.layer,
                        &x.ids[span.clone()],
                        &x.weights[span.clone()],
                        &y.ids[span.clone()],
                        &y.weights[span]
                    );
                }
            }
        }
    }
    for (k, n) in differ.iter().enumerate() {
        eprintln!(
            "gpu routing: sequence {k}: {n} (token, layer) choices differ, first (step, layer, position) {:?}",
            first[k]
        );
    }
}

/// hardware: gpu. libllama on CUDA, batched against split, as the anchor of `GPU_TOL`, and the
/// builder against libllama with the same step shapes (`libllama_anchor`). The fixture's head
/// size of 16 has no CUDA flash-attention kernel, so attention runs on the CPU here, for both;
/// `gpu_dense.rs` (head 64) covers CUDA flash attention with mixed sequences.
#[test]
fn gpu_libllama_batched_vs_split_anchor() {
    let Some(o) = base() else { return };
    let worst = libllama_anchor(&fixture(), &o, -1, GPU_TOL, "gpu");
    eprintln!("gpu anchor worst (of largest logit): {worst:?}");
}
