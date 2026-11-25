//! Replay version 2 against the engine, per layer of a segment (ADR 0092 section 8, tree node
//! `planner-costmodel-v6-rest`): the model runs as two segment builders in a chain (layers
//! `0..cut` and `cut..n`), the routing is recorded, and the replay of the span `cut..n` with
//! `--layers`-scope gives, for every layer of the span, the counters the exit harness writes for
//! a segment run: the RAM-tier hit rate (`ram_hit_rate`), the share of routed pairs the compute
//! tier served (`vram_pair_share_layers`) and the full-hit share (`full_hit_rate`), plus the
//! pair counts that weight them (`pairs_layers`). The whole-model form is `replay_fidelity.rs`.
//!
//! Hardware label: `cpu_*`, no GPU needed.

mod common;

use common::synth::{MoeSpec, quantized_moe};
use common::trace::RoutingRecorder;
use common::{PROMPT, argmax};
use tetraeder_ggml_sys::llama::Ftype;
use tetraeder_memory::replay::sim::{Config, Policy, Scope, Trace, VramPolicy, replay_scoped};
use tetraeder_model::{
    BackendChoice, ExpertStats, LoadOptions, MissPolicy, Qwen3Moe, SegmentSpec, SeqEval,
};

const CPU: BackendChoice = BackendChoice::Cpu { threads: Some(2) };
const N_DECODE: usize = 10;
const VRAM: usize = 6;
const RAM: usize = 12;

fn opts(spec: SegmentSpec) -> LoadOptions {
    LoadOptions {
        expert_slots: Some(VRAM),
        ram_slots: Some(RAM),
        miss: MissPolicy::Cpu,
        prefetch: true,
        prefetch_depth: 1,
        record_routing: true,
        segment: Some(spec),
        ..LoadOptions::new(CPU)
    }
}

/// One step of the chain: `tokens` through every segment, each taking the previous one's rows;
/// the last one's output.
fn step(segs: &mut [Qwen3Moe], tokens: &[i32], all_logits: bool) -> Vec<f32> {
    let mut hidden: Option<Vec<f32>> = None;
    let mut out = Vec::new();
    for s in segs.iter_mut() {
        out.clear();
        s.decode_hidden(
            &[SeqEval {
                seq: 0,
                tokens,
                all_logits,
            }],
            hidden.as_deref(),
            &mut out,
        )
        .unwrap();
        hidden = Some(out.clone());
    }
    out
}

/// What `exit_regime`'s `layers_json` reads from two snapshots of a segment's counters, per layer
/// of `span`: `(pairs, compute-tier share, full-hit share, RAM hit rate)`.
fn engine_layers(
    a: &ExpertStats,
    b: &ExpertStats,
    span: std::ops::Range<usize>,
) -> Vec<(u64, Option<f64>, f64, f64)> {
    a.layers
        .iter()
        .zip(&b.layers)
        .skip(span.start)
        .take(span.len())
        .map(|(la, lb)| {
            let (f, p) = (lb.full_steps - la.full_steps, lb.partial_steps - la.partial_steps);
            let (ra, rb) = (la.ram.unwrap(), lb.ram.unwrap());
            let (rh, rm) = (rb.hits - ra.hits, rb.misses - ra.misses);
            let (h, c) = (lb.compute.hits - la.compute.hits, lb.cpu - la.cpu);
            (
                h + c,
                (h + c > 0).then(|| h as f64 / (h + c) as f64),
                f as f64 / (f + p).max(1) as f64,
                rh as f64 / (rh + rm).max(1) as f64,
            )
        })
        .collect()
}

/// hardware: cpu. For every cut, the replay of the segment's span equals the engine's per-layer
/// counters of the segment run over the decode steps.
#[test]
fn cpu_replay_of_a_span_matches_the_segment_run_per_layer() {
    let path = quantized_moe(&MoeSpec::REALISH, Ftype::Q4KM, "q4km-segments");
    let n_layer = MoeSpec::REALISH.layers as u32;
    for cut in 1..n_layer {
        let specs = [
            SegmentSpec {
                layers: 0..cut,
                embed: true,
                head: false,
            },
            SegmentSpec {
                layers: cut..n_layer,
                embed: false,
                head: true,
            },
        ];
        let mut segs: Vec<Qwen3Moe> = specs
            .iter()
            .map(|s| Qwen3Moe::load(&path, &opts(s.clone())).unwrap())
            .collect();
        let trace_path = std::env::temp_dir().join(format!(
            "tetraeder-segreplay-{}-{cut}.trace",
            std::process::id()
        ));
        let mut rec = RoutingRecorder::create(
            &trace_path,
            &segs[0],
            PROMPT.len(),
            "test",
            "first_prefill",
            None,
        );
        let prompt = PROMPT.to_vec();
        let nv = segs[1].n_vocab();
        let out = step(&mut segs, &prompt, true);
        {
            let [a, b] = &mut segs[..] else { unreachable!() };
            rec.drain_chain(&mut [a, b]);
        }
        let before = segs[1].expert_stats().unwrap();
        let mut next = argmax(&out[out.len() - nv..]) as i32;
        for _ in 0..N_DECODE {
            let out = step(&mut segs, &[next], false);
            let [a, b] = &mut segs[..] else { unreachable!() };
            rec.drain_chain(&mut [a, b]);
            next = argmax(&out) as i32;
        }
        rec.finish();
        let after = segs[1].expert_stats().unwrap();

        let trace = Trace::load(&trace_path).unwrap();
        std::fs::remove_file(&trace_path).ok();
        let span = (cut, n_layer);
        let out = replay_scoped(
            &trace,
            &Config {
                vram_slots: VRAM as u32,
                ram_slots: RAM as u32,
                policy: Policy::Lru,
                vram_policy: VramPolicy::Static,
                prefetch: true,
            },
            Scope {
                layers: Some(span),
                warm: 0,
            },
        );
        let want = engine_layers(&before, &after, cut as usize..n_layer as usize);
        let got = &out.decode.layers;
        assert_eq!(got.len(), want.len(), "cut {cut}");
        for (g, (pairs, share, full, ram)) in got.iter().zip(&want) {
            let tag = format!("cut {cut} layer {}", g.layer);
            assert_eq!(g.pairs, *pairs, "{tag}: pairs");
            assert!(
                (g.ram_hit_rate - ram).abs() < 1e-12,
                "{tag}: RAM hit {} against {ram}",
                g.ram_hit_rate
            );
            assert!(
                (g.full_hit_share - full).abs() < 1e-12,
                "{tag}: full hit {} against {full}",
                g.full_hit_share
            );
            let share = share.expect("every layer saw a pair");
            assert!(
                (g.vram_share - share).abs() < 1e-12,
                "{tag}: pair share {} against {share}",
                g.vram_share
            );
        }
        eprintln!(
            "cut {cut}: {} layers, pairs {:?}",
            got.len(),
            want.iter().map(|w| w.0).collect::<Vec<_>>()
        );
    }
}
