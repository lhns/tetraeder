//! Reusing the whole-step graph across steps (ADR 0036) changes no bit: the same logits and
//! routing as building a fresh graph every step, and a graph is built only when the token count,
//! the padded KV length or `last_only` changes.
//!
//! Hardware label: `cpu_*`.

mod common;

use common::fixture;
use tetraeder_model::{BackendChoice, LoadOptions, Qwen3Moe, RoutingStep, SeqEval};

const CPU: BackendChoice = BackendChoice::Cpu { threads: Some(2) };

/// Steps of the sequence: prefill, decodes, a last-only chunk, decodes.
const EVALS: u64 = 1 + 8 + 1 + 2;
/// Distinct (token count, last_only) shapes: prefill, decode, last-only chunk. The cache keeps
/// two graphs, so the decodes after the chunk reuse the decode graph.
const BUILDS_WITH_REUSE: u64 = 3;

struct Out {
    logits: Vec<Vec<f32>>,
    routing: Vec<RoutingStep>,
}

fn token(i: usize, n_vocab: usize) -> i32 {
    ((i * 37 + 5) % n_vocab) as i32
}

fn sequence(m: &mut Qwen3Moe) -> Out {
    let v = m.n_vocab();
    let mut i = 0;
    let mut next = |n: usize| -> Vec<i32> {
        (0..n)
            .map(|_| {
                i += 1;
                token(i, v)
            })
            .collect()
    };
    let mut logits = vec![m.eval(&next(5)).unwrap()];
    for _ in 0..8 {
        logits.push(m.eval(&next(1)).unwrap());
    }
    logits.push(m.eval_last(&next(3)).unwrap());
    for _ in 0..2 {
        logits.push(m.eval(&next(1)).unwrap());
    }
    Out {
        logits,
        routing: m.take_routing_log(),
    }
}

fn assert_same(a: &Out, b: &Out, what: &str) {
    assert_eq!(a.logits.len(), b.logits.len(), "{what}");
    for (step, (x, y)) in a.logits.iter().zip(&b.logits).enumerate() {
        assert_eq!(x.len(), y.len(), "{what}: step {step}");
        assert!(
            x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()),
            "{what}: logits of step {step} differ"
        );
    }
    assert_eq!(a.routing, b.routing, "{what}: routing");
}

fn load(reuse: bool, record_routing: bool) -> Qwen3Moe {
    let o = LoadOptions {
        reuse_step_graph: reuse,
        record_routing,
        ..LoadOptions::new(CPU)
    };
    Qwen3Moe::load(fixture(), &o).unwrap()
}

/// hardware: cpu
#[test]
fn cpu_reuse_equals_fresh_graph() {
    for record in [false, true] {
        let (mut off, mut on) = (load(false, record), load(true, record));
        let (a, b) = (sequence(&mut off), sequence(&mut on));
        let what = format!("record_routing={record}");
        assert_same(&a, &b, &what);
        if record {
            assert!(!a.routing.is_empty());
        }
        assert_eq!(off.step_graphs_built(), EVALS, "{what}");
        assert_eq!(on.step_graphs_built(), BUILDS_WITH_REUSE, "{what}");
    }
}

/// hardware: cpu. One model, both modes, and a `reset` in between.
#[test]
fn cpu_toggle_and_reset_reproduce_the_logits() {
    let mut m = load(false, true);
    let reference = sequence(&mut m);
    m.set_reuse_step_graph(true);
    m.reset();
    let built = m.step_graphs_built();
    let first = sequence(&mut m);
    assert_same(&reference, &first, "reuse on after off");
    assert_eq!(m.step_graphs_built() - built, BUILDS_WITH_REUSE);

    m.reset();
    let built = m.step_graphs_built();
    let again = sequence(&mut m);
    assert_same(&reference, &again, "reuse on after reset");
    // The graphs kept at the end of the first run are the decode and the chunk; the prefill and
    // the chunk are built again (the chunk was dropped for the prefill).
    assert_eq!(m.step_graphs_built() - built, 2);

    m.set_reuse_step_graph(false);
    m.reset();
    let built = m.step_graphs_built();
    let off = sequence(&mut m);
    assert_same(&reference, &off, "reuse off again");
    assert_eq!(m.step_graphs_built() - built, EVALS);
}

/// hardware: cpu. Decode steps crossing a padded KV length (256 tokens) build a new graph there
/// and still match.
#[test]
fn cpu_reuse_across_the_kv_padding_boundary() {
    let run = |reuse: bool| {
        let mut m = load(reuse, false);
        let v = m.n_vocab();
        m.eval(&(0..250).map(|i| token(i, v)).collect::<Vec<_>>())
            .unwrap();
        let logits: Vec<Vec<f32>> = (250..262)
            .map(|i| m.eval(&[token(i, v)]).unwrap())
            .collect();
        (logits, m.step_graphs_built())
    };
    let ((a, built_off), (b, built_on)) = (run(false), run(true));
    assert_eq!(a, b);
    // Prefill, decodes up to 256 tokens, decodes after.
    assert_eq!((built_off, built_on), (13, 3));
}

/// hardware: cpu. A CPU step alternates a group of decoders and a prompt chunk (two shapes): the
/// two graphs are kept, so repeated steps of the same shapes build nothing, and the logits are
/// those of building every step.
#[test]
fn cpu_two_alternating_shapes_are_built_once() {
    const ROUNDS: usize = 4;
    let run = |reuse: bool| {
        let o = LoadOptions {
            reuse_step_graph: reuse,
            n_ctx: 1024,
            n_seq_max: 3,
            ..LoadOptions::new(CPU)
        };
        let mut m = Qwen3Moe::load(fixture(), &o).unwrap();
        let v = m.n_vocab();
        let tok = |i: usize| token(i, v);
        let mut out = Vec::new();
        let mut rows = Vec::new();
        let mut built_after_round = Vec::new();
        for r in 0..ROUNDS {
            let (a, b, c): (Vec<i32>, Vec<i32>, Vec<i32>) = (
                vec![tok(r * 3)],
                vec![tok(r * 3 + 1)],
                (0..40).map(|i| tok(r * 100 + i)).collect(),
            );
            // Decoders of sequences 0 and 1, then a prompt chunk of sequence 2.
            for inputs in [
                vec![
                    SeqEval {
                        seq: 0,
                        tokens: &a,
                        all_logits: false,
                    },
                    SeqEval {
                        seq: 1,
                        tokens: &b,
                        all_logits: false,
                    },
                ],
                vec![SeqEval {
                    seq: 2,
                    tokens: &c,
                    all_logits: false,
                }],
            ] {
                m.decode(&inputs, &mut out).unwrap();
                rows.push(out.clone());
            }
            built_after_round.push(m.step_graphs_built());
        }
        (rows, built_after_round)
    };
    let (want, built_off) = run(false);
    let (got, built_on) = run(true);
    assert_eq!(want, got, "logits");
    assert_eq!(built_off.last(), Some(&(2 * ROUNDS as u64)));
    assert_eq!(
        built_on,
        vec![2; ROUNDS],
        "no graph is built after the first round"
    );
}
