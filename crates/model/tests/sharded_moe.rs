//! The sharded MoE builder on CPU ranks (ADR 0069, 0070, 0071; Phase 5, E7): every expert
//! resident, one rank bit-equal to the resident builder, and tensor, expert, layer and
//! tensor-times-pipeline shapes against one rank within `tol_batch` (top-5 logits, greedy equal
//! or a near tie), batched sequences, skewed routing and KV snapshots across shapes.
//!
//! Hardware label: `cpu_*` need no GPU. The `gpu2_` and `gpu4_` suites of the same shapes on
//! Qwen3-30B-A3B run on the boxes (`gpu_sharded_moe.rs`).

mod common;

use common::multi_seq::{Alone, alone_on, close, prompt, same_token};
use common::synth::{MoeSpec, quantized_moe, write_moe_edit};
use common::{PROMPT, TOL_CPU, Tol, argmax, fixture, top_k};
use std::path::{Path, PathBuf};
use tetraeder_ggml_sys::llama::{Ftype, QuantizeParams, quantize};
use tetraeder_model::{BackendChoice, LoadOptions, Qwen3Moe, SeqEval};
use tetraeder_plan::{RankSpec, Span, StageSpec, Topology};

/// Four layers, eight heads with four KV heads, 32 experts: four-way tensor parallel divides it.
const SPEC4: MoeSpec = MoeSpec {
    layers: 4,
    embd: 256,
    heads: 8,
    heads_kv: 4,
    head_dim: 64,
    experts: 32,
    used: 8,
    ff: 256,
    seed: 20261116,
};

fn opts(n_seq_max: usize) -> LoadOptions {
    LoadOptions {
        n_ctx: 1024,
        n_seq_max,
        cpu_threads: Some(2),
        ..LoadOptions::new(BackendChoice::Cpu { threads: Some(2) })
    }
}

/// Q8_0 weights: the activations of every matmul are quantized to 8 bits, so an ulp of partial
/// sum noise sometimes flips a quantization step and moves the logits by a few thousandths of
/// the largest one. Exact for most rows, never as tight as F16.
fn synth4() -> PathBuf {
    quantized_moe(&SPEC4, Ftype::Q8_0, "sharded4")
}

fn synth4_f16() -> PathBuf {
    quantized_moe(&SPEC4, Ftype::F16, "sharded4f16")
}

/// The batched-against-alone rule (`tol_batch`) for F16 weights.
const REL_F16: f32 = 1e-3;
/// Q8_0 weights: a few activation quantization steps (see [`synth4`]); a wrong slice or a missing
/// partial sum is off by orders of magnitude more.
const REL_Q8: f32 = 3e-2;

// ---- topologies ---------------------------------------------------------------------------

fn ranks(n: usize) -> Vec<RankSpec> {
    (0..n)
        .map(|i| RankSpec {
            device: format!("cpu{i}"),
            gpu_index: None,
        })
        .collect()
}

/// Stages of `(first layer, end layer, ranks, ep)`; tp is the number of ranks.
fn topo(n_layer: u32, n_ranks: usize, stages: &[(u32, u32, Vec<usize>, bool)]) -> Topology {
    let n = stages.len();
    Topology {
        ranks: ranks(n_ranks),
        stages: stages
            .iter()
            .enumerate()
            .map(|(i, (a, b, rs, ep))| StageSpec {
                layers: Span(*a, *b),
                ranks: rs.clone(),
                tp: rs.len() as u32,
                ep: if *ep { rs.len() as u32 } else { 1 },
                embed: i == 0,
                head: i + 1 == n,
            })
            .collect(),
        n_layer,
    }
}

fn single(n_layer: u32) -> Topology {
    Topology::single("cpu0", None, n_layer)
}

fn tp(n_layer: u32, g: usize) -> Topology {
    topo(n_layer, g, &[(0, n_layer, (0..g).collect(), false)])
}

fn ep(n_layer: u32, g: usize) -> Topology {
    topo(n_layer, g, &[(0, n_layer, (0..g).collect(), true)])
}

/// Layer split at the given layers, one rank per stage.
fn layers(n_layer: u32, cuts: &[u32]) -> Topology {
    let mut bounds = vec![0];
    bounds.extend_from_slice(cuts);
    bounds.push(n_layer);
    let stages: Vec<_> = bounds
        .windows(2)
        .enumerate()
        .map(|(i, w)| (w[0], w[1], vec![i], false))
        .collect();
    topo(n_layer, stages.len(), &stages)
}

/// Two tensor-parallel stages of two ranks (four GPUs, ADR 0070).
fn tp2_pp2(n_layer: u32) -> Topology {
    let mid = n_layer / 2;
    topo(
        n_layer,
        4,
        &[
            (0, mid, vec![0, 1], false),
            (mid, n_layer, vec![2, 3], false),
        ],
    )
}

fn sharded(path: &Path, t: &Topology, o: &LoadOptions) -> Qwen3Moe {
    Qwen3Moe::load_sharded(path, t, o).unwrap_or_else(|e| panic!("load_sharded: {e}"))
}

// ---- comparisons --------------------------------------------------------------------------

/// The top-5 logits of `want` are within `max(TOL_CPU, rel * largest)` in `got`.
fn top5_close(got: &[f32], want: &[f32], rel: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    let tol = Tol { abs: TOL_CPU, rel }.of(want);
    let mut worst = 0f32;
    for i in top_k(want, 5) {
        worst = worst.max((got[i] - want[i]).abs());
    }
    let all = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("{what}: top-5 max |diff| {worst:e}, row max {all:e}, tolerance {tol:e}");
    assert!(worst <= tol, "{what}: top-5 diff {worst} > {tol}");
}

/// `m` follows the alone run teacher-forced: the logits after the prompt and after each greedy
/// token agree in the top 5, and the greedy token is equal or a near tie.
fn follows(m: &mut Qwen3Moe, want: &Alone, rel: f32, what: &str) {
    m.reset();
    let mut got = m.eval_last(&want.toks[..want.prompt_len]).unwrap();
    for (j, w) in want.logits.iter().enumerate() {
        let at = format!("{what}, step {j}");
        top5_close(&got, w, rel, &at);
        same_token(&got, w, Tol { abs: TOL_CPU, rel }, &at);
        if j + 1 < want.logits.len() {
            got = m.eval_last(&[want.toks[want.prompt_len + j]]).unwrap();
        }
    }
}

fn reference(path: &Path, prompt: &[i32], steps: usize) -> Alone {
    let mut m = Qwen3Moe::load(path, &opts(1)).unwrap();
    alone_on(&mut m, prompt, steps)
}

fn shapes_follow_one_rank(path: &Path, rel: f32, shapes: Vec<(&str, Topology)>) {
    let p = prompt(3, 12);
    let want = reference(path, &p, 12);
    for (name, t) in shapes {
        let mut m = sharded(path, &t, &opts(1));
        follows(&mut m, &want, rel, name);
    }
}

// ---- tests --------------------------------------------------------------------------------

/// hardware: cpu. A one-rank topology builds the ops of the resident whole-step graph, so the
/// logits are the same bits: every prompt position, then 12 greedy steps.
fn one_rank_is_bit_equal(path: &Path, n_layer: u32) {
    let mut old = Qwen3Moe::load(path, &opts(1)).unwrap();
    let mut new = sharded(path, &single(n_layer), &opts(1));
    assert_eq!(new.n_vocab(), old.n_vocab());
    let (a, b) = (old.eval(&PROMPT).unwrap(), new.eval(&PROMPT).unwrap());
    assert!(a == b, "prompt logits differ");
    let mut la = old.eval_last(&[7]).unwrap();
    let mut lb = new.eval_last(&[7]).unwrap();
    for step in 0..12 {
        assert!(la == lb, "greedy step {step} differs");
        let t = argmax(&la) as i32;
        la = old.eval_last(&[t]).unwrap();
        lb = new.eval_last(&[t]).unwrap();
    }
}

/// hardware: cpu.
#[test]
fn cpu_one_rank_is_bit_equal_to_the_resident_builder_tiny() {
    one_rank_is_bit_equal(&fixture(), 2);
}

/// hardware: cpu.
#[test]
fn cpu_one_rank_is_bit_equal_to_the_resident_builder_q8() {
    one_rank_is_bit_equal(&synth4(), 4);
}

/// hardware: cpu. `tp-2`, `ep-2` and `layer-2` on the tiny fixture (F16, 2 layers, 8 experts).
#[test]
fn cpu_tiny_fixture_shapes_follow_one_rank() {
    shapes_follow_one_rank(
        &fixture(),
        REL_F16,
        vec![
            ("tp-2", tp(2, 2)),
            ("ep-2", ep(2, 2)),
            ("layer-2", layers(2, &[1])),
        ],
    );
}

/// hardware: cpu. The synthetic MoE, 4 layers, F16 weights: two-way shapes and the layer split at
/// the first, middle and last boundary (Rule 2), within `tol_batch`.
#[test]
fn cpu_synthetic_two_rank_shapes_follow_one_rank() {
    shapes_follow_one_rank(
        &synth4_f16(),
        REL_F16,
        vec![
            ("tp-2", tp(4, 2)),
            ("ep-2", ep(4, 2)),
            ("layer-2 at 1", layers(4, &[1])),
            ("layer-2 at 2", layers(4, &[2])),
            ("layer-2 at 3", layers(4, &[3])),
        ],
    );
}

/// hardware: cpu. Four ranks, F16: `ep-4`, `tp-4`, `tp2-pp2` and a four-stage layer split.
#[test]
fn cpu_synthetic_four_rank_shapes_follow_one_rank() {
    shapes_follow_one_rank(
        &synth4_f16(),
        REL_F16,
        vec![
            ("ep-4", ep(4, 4)),
            ("tp-4", tp(4, 4)),
            ("tp2-pp2", tp2_pp2(4)),
            ("layer-4", layers(4, &[1, 2, 3])),
        ],
    );
}

/// hardware: cpu. The same shapes on Q8_0 weights, which cuts quantized rows and columns at block
/// boundaries (wo columns, expert down columns), within the Q8_0 bound.
#[test]
fn cpu_synthetic_q8_shapes_follow_one_rank() {
    shapes_follow_one_rank(
        &synth4(),
        REL_Q8,
        vec![
            ("q8 tp-2", tp(4, 2)),
            ("q8 ep-2", ep(4, 2)),
            ("q8 ep-4", ep(4, 4)),
            ("q8 tp-4", tp(4, 4)),
            ("q8 tp2-pp2", tp2_pp2(4)),
            ("q8 layer-2 at 2", layers(4, &[2])),
        ],
    );
}

/// The router chooses only experts 8 to 15 in every layer: the router row of expert `e` has one
/// entry, on element 0, positive for `e < 16` and negative above, with that element of the
/// normed input always positive (the embedding's element 0 is 1, the layers add nothing to it).
fn skew(name: &str, ne: &[usize], v: &mut [f32]) {
    if name == "token_embd.weight" {
        for row in v.chunks_mut(ne[0]) {
            row[0] = 1.0;
        }
    } else if name.ends_with("ffn_norm.weight") {
        v.fill(0.0);
        v[0] = 1.0;
    } else if name.ends_with("ffn_gate_inp.weight") {
        let experts = ne[1];
        for (e, row) in v.chunks_mut(ne[0]).enumerate() {
            row.fill(0.0);
            let c = 1.0 + e as f32 * 0.01;
            row[0] = if e < experts / 2 { c } else { -c };
        }
    } else if name.ends_with("attn_output.weight") {
        v[..ne[0]].fill(0.0);
    } else if name.ends_with("ffn_down_exps.weight") {
        for plane in v.chunks_mut(ne[0] * ne[1]) {
            plane[..ne[0]].fill(0.0);
        }
    }
}

fn skewed_model() -> PathBuf {
    let out =
        std::env::temp_dir().join(format!("tetraeder-synth-skew-{}.gguf", std::process::id()));
    if !out.is_file() {
        let src = out.with_extension("f32.tmp");
        let part = out.with_extension("part");
        write_moe_edit(&src, &MoeSpec::REALISH, 1024, 0, &mut skew);
        let mut qp = QuantizeParams::new(Ftype::Q8_0);
        qp.nthread = 2;
        quantize(&src, &part, &qp).unwrap();
        std::fs::remove_file(&src).ok();
        std::fs::rename(&part, &out).unwrap();
    }
    out
}

/// hardware: cpu. Every top-k choice lands on rank 0's experts: rank 1 maps all ids to -1 and
/// contributes zeros, and `ep-2` still equals one rank.
#[test]
fn cpu_expert_parallel_with_all_routing_on_one_rank() {
    let path = skewed_model();
    // The premise: the resident builder routes only to experts 8 to 15.
    let mut routed = Qwen3Moe::load(
        &path,
        &LoadOptions {
            record_routing: true,
            ..opts(1)
        },
    )
    .unwrap();
    routed.eval(&prompt(1, 10)).unwrap();
    let log = routed.take_routing_log();
    assert!(!log.is_empty());
    for step in &log {
        assert!(
            step.ids.iter().all(|&e| (8..16).contains(&e)),
            "layer {} routed to {:?}",
            step.layer,
            step.ids
        );
    }
    let p = prompt(2, 10);
    let want = reference(&path, &p, 8);
    let n_layer = MoeSpec::REALISH.layers as u32;
    for (name, t) in [("ep-2", ep(n_layer, 2)), ("tp-2", tp(n_layer, 2))] {
        let mut m = sharded(&path, &t, &opts(1));
        follows(&mut m, &want, REL_Q8, &format!("skewed {name}"));
    }
}

/// hardware: cpu. Three sequences of different lengths in one step, then decoding together, on
/// `ep-2` and `tp-2`: each row against the same sequence alone on the resident builder.
#[test]
fn cpu_three_sequences_batched_on_sharded_shapes() {
    let path = synth4_f16();
    let prompts = [prompt(1, 9), prompt(2, 17), prompt(3, 5)];
    let steps = 4;
    let wants: Vec<Alone> = prompts.iter().map(|p| reference(&path, p, steps)).collect();
    for (name, t) in [("ep-2", ep(4, 2)), ("tp-2", tp(4, 2))] {
        let mut m = sharded(&path, &t, &opts(3));
        let nv = m.n_vocab();
        let mut out = Vec::new();
        let inputs: Vec<SeqEval> = prompts
            .iter()
            .enumerate()
            .map(|(i, p)| SeqEval {
                seq: i as u32,
                tokens: p,
                all_logits: false,
            })
            .collect();
        let spans = m.decode(&inputs, &mut out).unwrap();
        for (i, r) in spans.iter().enumerate() {
            let row = &out[r.start * nv..r.end * nv];
            let at = format!("{name} prefill of sequence {i}");
            top5_close(row, &wants[i].logits[0], REL_F16, &at);
            same_token(row, &wants[i].logits[0], TOL_CPU, &at);
        }
        for j in 0..steps {
            let toks: Vec<[i32; 1]> = wants.iter().map(|w| [w.toks[w.prompt_len + j]]).collect();
            let inputs: Vec<SeqEval> = toks
                .iter()
                .enumerate()
                .map(|(i, t)| SeqEval {
                    seq: i as u32,
                    tokens: t,
                    all_logits: false,
                })
                .collect();
            let spans = m.decode(&inputs, &mut out).unwrap();
            for (i, r) in spans.iter().enumerate() {
                let row = &out[r.start * nv..r.end * nv];
                let at = format!("{name} decode {j} of sequence {i}");
                top5_close(row, &wants[i].logits[j + 1], REL_F16, &at);
                same_token(row, &wants[i].logits[j + 1], TOL_CPU, &at);
            }
        }
    }
}

/// F16 bits to f32 (normal numbers, subnormals and zero; the cache holds nothing else).
fn f16(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((bits >> 10) & 0x1f);
    let man = f32::from(bits & 0x3ff);
    if exp == 0 {
        sign * man * 2f32.powi(-24)
    } else {
        sign * (1.0 + man / 1024.0) * 2f32.powi(exp - 15)
    }
}

/// Two F16 byte strings agree to a few F16 steps: layer 0 sees the same inputs on every shape,
/// but a matmul over fewer output rows may take another kernel path and round differently.
fn f16_close(a: &[u8], b: &[u8], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}");
    let mut worst = 0f32;
    for (x, y) in a.chunks_exact(2).zip(b.chunks_exact(2)) {
        let (x, y) = (
            f16(u16::from_le_bytes([x[0], x[1]])),
            f16(u16::from_le_bytes([y[0], y[1]])),
        );
        let d = (x - y).abs();
        assert!(
            d <= 2e-3 * (x.abs() + y.abs()) + 1e-4,
            "{what}: {x} against {y}"
        );
        worst = worst.max(d);
    }
    eprintln!("{what}: max |diff| {worst:e}");
}

fn decode_one(m: &mut Qwen3Moe, seq: u32, token: i32) -> Vec<f32> {
    let mut out = Vec::new();
    m.decode(
        &[SeqEval {
            seq,
            tokens: &[token],
            all_logits: false,
        }],
        &mut out,
    )
    .unwrap();
    out
}

/// hardware: cpu. A snapshot has the one-rank layout whatever the shape: the same length, the
/// same bytes on layer 0 (whose inputs are identical), a round trip under one shape is
/// bit-equal, and one shape's snapshot restores under the others and decodes alike.
#[test]
fn cpu_snapshots_are_shape_independent() {
    let path = fixture();
    let p = prompt(5, 20);
    let n_layer = 2usize;
    let shapes: Vec<(&str, Topology)> = vec![
        ("single", single(2)),
        ("tp-2", tp(2, 2)),
        ("ep-2", ep(2, 2)),
        ("layer-2", layers(2, &[1])),
    ];
    let mut snaps: Vec<(String, Vec<u8>)> = Vec::new();
    let mut nexts: Vec<Vec<f32>> = Vec::new();
    for (name, t) in &shapes {
        let mut m = sharded(&path, t, &opts(2));
        let mut out = Vec::new();
        m.decode(
            &[SeqEval {
                seq: 1,
                tokens: &p,
                all_logits: false,
            }],
            &mut out,
        )
        .unwrap();
        let snap = m.seq_snapshot(1).unwrap();
        assert_eq!(snap.len(), p.len() * m.kv_row_bytes(), "{name}: length");
        // A round trip under the same shape.
        m.seq_restore(0, &snap, p.len()).unwrap();
        assert_eq!(m.seq_snapshot(0).unwrap(), snap, "{name}: round trip");
        nexts.push(decode_one(&mut m, 1, 9));
        snaps.push((name.to_string(), snap));
    }
    // Layer 0 of K and of V.
    let rb = snaps[0].1.len() / (2 * n_layer * p.len()) * p.len();
    let v0 = n_layer * rb;
    for (name, s) in &snaps[1..] {
        assert_eq!(s.len(), snaps[0].1.len(), "{name}: layout");
        f16_close(
            &s[..rb],
            &snaps[0].1[..rb],
            &format!("{name}: K of layer 0"),
        );
        f16_close(
            &s[v0..v0 + rb],
            &snaps[0].1[v0..v0 + rb],
            &format!("{name}: V of layer 0"),
        );
    }
    // Restored under every shape, each decodes like the shape that took it.
    for (from, snap) in &snaps {
        for (i, (to, t)) in shapes.iter().enumerate() {
            let mut m = sharded(&path, t, &opts(2));
            m.seq_restore(1, snap, p.len()).unwrap();
            let got = decode_one(&mut m, 1, 9);
            close(
                &got,
                &nexts[i],
                TOL_CPU,
                &format!("snapshot of {from} restored under {to}"),
            );
        }
    }
}

/// hardware: cpu. `seq_cp` copies the rows on every rank: a copied prefix decodes as a separate
/// prefill of it.
#[test]
fn cpu_seq_cp_under_tp2() {
    let path = fixture();
    let p = prompt(7, 18);
    let mut m = sharded(&path, &tp(2, 2), &opts(2));
    let mut out = Vec::new();
    m.decode(
        &[SeqEval {
            seq: 0,
            tokens: &p,
            all_logits: false,
        }],
        &mut out,
    )
    .unwrap();
    m.seq_cp(0, 1, 12).unwrap();
    let got = decode_one(&mut m, 1, 11);
    let mut fresh = sharded(&path, &tp(2, 2), &opts(2));
    fresh
        .decode(
            &[SeqEval {
                seq: 0,
                tokens: &p[..12],
                all_logits: false,
            }],
            &mut out,
        )
        .unwrap();
    let want = decode_one(&mut fresh, 0, 11);
    close(&got, &want, TOL_CPU, "copied prefix");
}

/// hardware: cpu. The options a sharded load cannot honour are refused, naming the option.
#[test]
fn cpu_load_sharded_refuses_streaming_options() {
    let path = fixture();
    let t = single(2);
    for (name, o) in [
        (
            "expert_slots",
            LoadOptions {
                expert_slots: Some(2),
                ..opts(1)
            },
        ),
        (
            "op_offload",
            LoadOptions {
                op_offload: true,
                ..opts(1)
            },
        ),
        (
            "ram_slots",
            LoadOptions {
                ram_slots: Some(4),
                ..opts(1)
            },
        ),
    ] {
        let err = Qwen3Moe::load_sharded(&path, &t, &o)
            .unwrap_err()
            .to_string();
        assert!(err.contains(name), "{name}: {err}");
    }
}

/// hardware: cpu. A shape the model cannot take is refused at load with the reason: four-way
/// tensor parallel on two KV heads.
#[test]
fn cpu_load_sharded_names_the_divisibility_reason() {
    let err = Qwen3Moe::load_sharded(fixture(), &tp(2, 4), &opts(1))
        .unwrap_err()
        .to_string();
    assert!(err.contains("KV heads"), "{err}");
}

/// hardware: cpu. `layer_limit` runs the first layers and the head: the logits have the right
/// size and are finite.
#[test]
fn cpu_layer_limit_runs_a_prefix() {
    let path = synth4();
    let o = LoadOptions {
        layer_limit: Some(2),
        ..opts(1)
    };
    for t in [single(4), tp(4, 2), layers(4, &[1, 3])] {
        let mut m = sharded(&path, &t, &o);
        let l = m.eval_last(&PROMPT).unwrap();
        assert_eq!(l.len(), m.n_vocab());
        assert!(l.iter().all(|x| x.is_finite()));
    }
}

/// hardware: cpu. Diagnostic, asserts nothing: the largest logit difference of each shape
/// against one rank when only the first L layers run (L = 1 to 4), to see where a gap starts.
#[test]
fn cpu_diag_gap_by_layer_limit() {
    let path = synth4();
    let p = prompt(3, 12);
    for l in 1..=4u32 {
        let o = LoadOptions {
            layer_limit: Some(l),
            ..opts(1)
        };
        let mut one = sharded(&path, &single(4), &o);
        let want = one.eval_last(&p).unwrap();
        for (name, t) in [
            ("tp-2", tp(4, 2)),
            ("ep-2", ep(4, 2)),
            ("layer-2", layers(4, &[1])),
        ] {
            let mut m = sharded(&path, &t, &o);
            let got = m.eval_last(&p).unwrap();
            let d = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!(
                "DIAG limit {l}, {name}: max |diff| {d:e}, largest {}",
                common::largest(&want)
            );
        }
    }
}
