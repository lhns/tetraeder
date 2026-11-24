//! Segments of the layers (ADR 0091, step 4): a builder that holds a share of the model takes
//! hidden rows in and returns hidden rows out. Parity per op and end to end (rule 2): every cut
//! of the layers, the head on the last segment or on a segment of its own, against the whole
//! model of the same builder, in process, no transport.
//!
//! Hardware labels (AGENTS rule 9): all tests are `cpu`.

mod common;

use common::synth::{MoeSpec, write_moe};
use common::{PROMPT, TOP_K, argmax, dense_fixture, dense_fixture_l4, fixture, tol_batch, top_k};
use std::path::PathBuf;
use tetraeder_model::{
    BackendChoice, DenseOptions, LoadOptions, Qwen3, Qwen3Moe, SegmentSpec, SeqEval,
};
use tetraeder_plan::KvFormat;

const CPU: BackendChoice = BackendChoice::Cpu { threads: Some(2) };
const N_GREEDY: usize = 8;

/// Either builder behind one call.
enum Net {
    Dense(Box<Qwen3>),
    Moe(Box<Qwen3Moe>),
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    Dense,
    /// Every expert resident: the whole-step graph.
    Moe,
    /// Experts streamed through slots: a graph per layer.
    MoeStreamed,
}

struct Model {
    kind: Kind,
    path: PathBuf,
    n_layer: u32,
    name: &'static str,
}

impl Net {
    fn load(m: &Model, segment: Option<SegmentSpec>) -> Net {
        match m.kind {
            Kind::Dense => Net::Dense(Box::new(
                Qwen3::load(
                    &m.path,
                    &DenseOptions {
                        kv: KvFormat::F16,
                        kv_rotation: Some(false),
                        extra_bufts: false,
                        n_seq_max: 2,
                        n_ctx_pool: 512,
                        segment,
                        ..DenseOptions::new(CPU)
                    },
                )
                .unwrap(),
            )),
            Kind::Moe | Kind::MoeStreamed => Net::Moe(Box::new(
                Qwen3Moe::load(
                    &m.path,
                    &LoadOptions {
                        kv: KvFormat::F16,
                        kv_rotation: Some(false),
                        n_seq_max: 2,
                        n_ctx: 512,
                        expert_slots: matches!(m.kind, Kind::MoeStreamed).then_some(3),
                        segment,
                        ..LoadOptions::new(CPU)
                    },
                )
                .unwrap(),
            )),
        }
    }

    fn decode(
        &mut self,
        inputs: &[SeqEval<'_>],
        hidden: Option<&[f32]>,
        out: &mut Vec<f32>,
    ) -> Vec<std::ops::Range<usize>> {
        match self {
            Net::Dense(m) => m.decode_hidden(inputs, hidden, out).unwrap(),
            Net::Moe(m) => m.decode_hidden(inputs, hidden, out).unwrap(),
        }
    }

    fn seq_rm(&mut self, seq: u32, keep: usize) {
        match self {
            Net::Dense(m) => m.seq_rm(seq, keep).unwrap(),
            Net::Moe(m) => m.seq_rm(seq, keep).unwrap(),
        }
    }

    fn out_width(&self) -> usize {
        match self {
            Net::Dense(m) => m.out_width(),
            Net::Moe(m) => m.out_width(),
        }
    }

    /// Bytes of weights and KV, from the memory report.
    fn bytes(&self) -> (u64, u64) {
        let mem = match self {
            Net::Dense(m) => m.memory(),
            Net::Moe(m) => m.memory(),
        };
        let sum = |kind: &str| mem.iter().filter(|u| u.kind == kind).map(|u| u.used).sum();
        (sum("weights"), sum("kv"))
    }
}

fn synth_moe(layers: usize) -> PathBuf {
    let spec = MoeSpec {
        layers,
        embd: 64,
        heads: 4,
        heads_kv: 2,
        head_dim: 16,
        experts: 8,
        used: 2,
        ff: 64,
        seed: 20261124,
    };
    let path = std::env::temp_dir().join(format!(
        "tetraeder-segments-moe{layers}-{}.gguf",
        std::process::id()
    ));
    write_moe(&path, &spec);
    path
}

fn models() -> Vec<Model> {
    vec![
        Model {
            kind: Kind::Dense,
            path: dense_fixture_l4(),
            n_layer: 4,
            name: "dense4",
        },
        Model {
            kind: Kind::Dense,
            path: dense_fixture(),
            n_layer: 2,
            name: "dense2",
        },
        Model {
            kind: Kind::Moe,
            path: fixture(),
            n_layer: 2,
            name: "moe2",
        },
        Model {
            kind: Kind::Moe,
            path: synth_moe(4),
            n_layer: 4,
            name: "moe4",
        },
        Model {
            kind: Kind::MoeStreamed,
            path: synth_moe(4),
            n_layer: 4,
            name: "moe4_streamed",
        },
    ]
}

/// Runs `inputs` through `segs` in order (each takes the previous one's hidden rows) and returns
/// the last one's rows, one per input row.
fn pipeline(segs: &mut [Net], inputs: &[SeqEval<'_>]) -> (Vec<f32>, Vec<std::ops::Range<usize>>) {
    let mut hidden: Option<Vec<f32>> = None;
    let mut out = Vec::new();
    let mut ranges = Vec::new();
    for s in segs.iter_mut() {
        out.clear();
        ranges = s.decode(inputs, hidden.as_deref(), &mut out);
        hidden = Some(out.clone());
    }
    (out, ranges)
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The cuts of one model and where the head goes: `(cut, head apart)`.
fn plans(n_layer: u32) -> Vec<(u32, bool)> {
    let mut v = Vec::new();
    for cut in 0..=n_layer {
        v.push((cut, false));
        v.push((cut, true));
    }
    v
}

fn specs(n_layer: u32, cut: u32, head_apart: bool) -> Vec<SegmentSpec> {
    let a = SegmentSpec {
        layers: 0..cut,
        embed: true,
        head: false,
    };
    if head_apart {
        vec![
            a,
            SegmentSpec {
                layers: cut..n_layer,
                embed: false,
                head: false,
            },
            SegmentSpec {
                layers: n_layer..n_layer,
                embed: false,
                head: true,
            },
        ]
    } else {
        vec![
            a,
            SegmentSpec {
                layers: cut..n_layer,
                embed: false,
                head: true,
            },
        ]
    }
}

/// Segments with no layers and no role are not specs; skip those cuts of a head-apart split
/// whose middle segment is empty (the pair of segments covers them).
fn valid(specs: &[SegmentSpec], n_layer: u32) -> bool {
    specs.iter().all(|s| s.validate(n_layer).is_ok())
}

/// hardware: cpu. Every cut of every builder path against the whole model: all the logits of a
/// prompt and of greedy decode steps, top-5 and the whole row within `tol_batch`, and the same
/// greedy tokens. One line per engine, cut and head placement with the largest difference.
#[test]
fn cpu_segments_match_the_whole_model() {
    for m in models() {
        let mut whole = Net::load(&m, None);
        // Reference rows: the prompt with all logits, then greedy steps.
        let mut want = Vec::new();
        let mut out = Vec::new();
        whole.decode(
            &[SeqEval {
                seq: 0,
                tokens: &PROMPT,
                all_logits: true,
            }],
            None,
            &mut out,
        );
        let nv = out.len() / PROMPT.len();
        want.push(out.clone());
        let mut tokens = vec![argmax(&out[out.len() - nv..]) as i32];
        for _ in 1..N_GREEDY {
            out.clear();
            whole.decode(
                &[SeqEval {
                    seq: 0,
                    tokens: &tokens[tokens.len() - 1..],
                    all_logits: false,
                }],
                None,
                &mut out,
            );
            want.push(out.clone());
            tokens.push(argmax(&out) as i32);
        }
        for (cut, apart) in plans(m.n_layer) {
            let sp = specs(m.n_layer, cut, apart);
            if !valid(&sp, m.n_layer) {
                continue;
            }
            let mut segs: Vec<Net> = sp.iter().map(|s| Net::load(&m, Some(s.clone()))).collect();
            assert_eq!(segs.last().unwrap().out_width(), nv);
            let mut worst = 0f32;
            let mut all_bits = true;
            let mut got_tokens = Vec::new();
            let mut check = |got: &[f32], want: &[f32], step: usize| {
                for (g, w) in got.chunks(nv).zip(want.chunks(nv)) {
                    let d = g
                        .iter()
                        .zip(w)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    worst = worst.max(d);
                    assert!(
                        d <= tol_batch(w),
                        "{} cut {cut} head apart {apart} step {step}: {d:e}",
                        m.name
                    );
                    assert_eq!(top_k(g, TOP_K).len(), TOP_K);
                    for id in top_k(w, TOP_K) {
                        assert!((g[id] - w[id]).abs() <= tol_batch(w));
                    }
                }
                all_bits &= bits(got) == bits(want);
            };
            let (got, ranges) = pipeline(
                &mut segs,
                &[SeqEval {
                    seq: 0,
                    tokens: &PROMPT,
                    all_logits: true,
                }],
            );
            assert_eq!(ranges, vec![0..PROMPT.len()]);
            check(&got, &want[0], 0);
            got_tokens.push(argmax(&got[got.len() - nv..]) as i32);
            for i in 1..N_GREEDY {
                let t = got_tokens[i - 1];
                let (got, _) = pipeline(
                    &mut segs,
                    &[SeqEval {
                        seq: 0,
                        tokens: &[t],
                        all_logits: false,
                    }],
                );
                check(&got, &want[i], i);
                got_tokens.push(argmax(&got) as i32);
            }
            assert_eq!(got_tokens, tokens, "{} cut {cut} apart {apart}", m.name);
            println!(
                "segment parity {} cut {cut} head {} max_logit_diff {worst:e} bit_equal {all_bits}",
                m.name,
                if apart { "on_leader" } else { "on_last" },
            );
        }
    }
}

/// hardware: cpu. Several sequences in one step, one with all logits and one with the last row
/// only, split at the middle: the rows of every input and the KV of both sequences match the
/// whole model, also after a `seq_rm` and a second step.
#[test]
fn cpu_segments_batch_sequences() {
    for m in models() {
        let cut = m.n_layer / 2;
        let tokens_a: Vec<i32> = vec![5, 9, 33, 100, 7];
        let tokens_b: Vec<i32> = vec![200, 41, 3];
        let ins = |all_b: bool| {
            vec![
                SeqEval {
                    seq: 0,
                    tokens: &tokens_a,
                    all_logits: true,
                },
                SeqEval {
                    seq: 1,
                    tokens: &tokens_b,
                    all_logits: all_b,
                },
            ]
        };
        let mut whole = Net::load(&m, None);
        let mut segs: Vec<Net> = specs(m.n_layer, cut, false)
            .into_iter()
            .map(|s| Net::load(&m, Some(s)))
            .collect();
        let mut want = Vec::new();
        let r_want = whole.decode(&ins(false), None, &mut want);
        let (got, r_got) = pipeline(&mut segs, &ins(false));
        assert_eq!(r_want, r_got, "{}", m.name);
        let nv = want.len() / r_want.last().unwrap().end;
        for (g, w) in got.chunks(nv).zip(want.chunks(nv)) {
            let d = g.iter().zip(w).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(d <= tol_batch(w), "{}: {d:e}", m.name);
        }
        // Cut sequence 0 back and run it again with one more sequence-1 token.
        whole.seq_rm(0, 2);
        for s in &mut segs {
            s.seq_rm(0, 2);
        }
        let next = [11, 12, 13];
        let again = |all: bool| {
            vec![SeqEval {
                seq: 0,
                tokens: &next,
                all_logits: all,
            }]
        };
        let mut want = Vec::new();
        whole.decode(&again(true), None, &mut want);
        let (got, _) = pipeline(&mut segs, &again(true));
        for (g, w) in got.chunks(nv).zip(want.chunks(nv)) {
            let d = g.iter().zip(w).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(d <= tol_batch(w), "{} after seq_rm: {d:e}", m.name);
        }
    }
}

/// hardware: cpu. A segment loads and holds only its share: the weights and the KV of two
/// segments are each below the whole model's, and together within it.
#[test]
fn cpu_segments_hold_only_their_share() {
    for m in models().into_iter().filter(|m| m.n_layer == 4) {
        let whole = Net::load(&m, None);
        let (w_all, kv_all) = whole.bytes();
        let sp = specs(4, 2, false);
        let a = Net::load(&m, Some(sp[0].clone()));
        let b = Net::load(&m, Some(sp[1].clone()));
        let (wa, kva) = a.bytes();
        let (wb, kvb) = b.bytes();
        assert!(wa < w_all && wb < w_all, "{} weights {wa} {wb} {w_all}", m.name);
        assert!(kva < kv_all && kvb < kv_all, "{} kv", m.name);
        assert!(kva + kvb <= kv_all, "{} kv sum", m.name);
        assert!(wa + wb <= w_all + w_all / 8, "{} weights sum", m.name);
        println!(
            "segment memory {} weights {wa} {wb} of {w_all} kv {kva} {kvb} of {kv_all}",
            m.name
        );
    }
}

/// hardware: cpu. Wrong inputs are refused with a clear error: hidden rows for a segment with
/// the embedding, none for one without, a wrong length, a spec outside the model.
#[test]
fn cpu_segments_refuse_bad_input() {
    let m = &models()[0];
    let sp = specs(4, 2, false);
    let mut first = Net::load(m, Some(sp[0].clone()));
    let Net::Dense(f) = &mut first else { panic!() };
    let mut out = Vec::new();
    let one = [SeqEval {
        seq: 0,
        tokens: &[1],
        all_logits: false,
    }];
    assert!(f.decode_hidden(&one, Some(&[0.0; 4]), &mut out).is_err());
    let mut second = Net::load(m, Some(sp[1].clone()));
    let Net::Dense(s) = &mut second else { panic!() };
    assert!(s.decode_hidden(&one, None, &mut out).is_err());
    assert!(s.decode_hidden(&one, Some(&[0.0; 3]), &mut out).is_err());
    let bad = DenseOptions {
        segment: Some(SegmentSpec {
            layers: 3..9,
            embed: false,
            head: true,
        }),
        ..DenseOptions::new(CPU)
    };
    assert!(Qwen3::load(&m.path, &bad).is_err());
}
