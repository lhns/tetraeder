//! Several sequences in one step on the MoE builder's KV pool (ADR 0065, 0067; phase 4, C4):
//! batched against each alone, against libllama's unified multi-sequence cache, `seq_cp`,
//! per-sequence snapshots and a chunked move in the middle of a decode. Resident and streamed
//! experts: slots with uploads, the RAM tier with CPU groups, prefetch, and the I/O service with
//! overlap. One sequence stays bit-equal to the old builder (`extract_bits.rs`).
//!
//! Hardware label: `cpu_*` need no GPU (`gpu_multi_seq.rs` runs the same on CUDA).

mod common;

use common::multi_seq::*;
use common::synth::{MoeSpec, quantized_moe};
use common::{TOL_CPU, fixture};
use std::path::PathBuf;
use std::sync::Arc;
use tetraeder_ggml_sys::llama::{
    Batch, Context, ContextParams, FlashAttn, Ftype, Model, ModelParams,
};
use tetraeder_memory::IoOptions;
use tetraeder_memory::io::IoFactory;
use tetraeder_memory::io::sim::{SimClock, SimIo};
use tetraeder_model::{BackendChoice, LoadOptions, MissPolicy, Qwen3Moe, SeqEval};
use tetraeder_plan::KvFormat;

const CPU: BackendChoice = BackendChoice::Cpu { threads: Some(2) };

fn base() -> LoadOptions {
    LoadOptions {
        n_ctx: 1024,
        ..LoadOptions::new(CPU)
    }
}

/// Every expert resident: one whole-step graph.
fn resident() -> (PathBuf, LoadOptions) {
    (fixture(), base())
}

/// The fixture's 8 experts in 2 compute slots per layer, uploaded on a miss.
fn upload() -> (PathBuf, LoadOptions) {
    (
        fixture(),
        LoadOptions {
            expert_slots: Some(2),
            ..base()
        },
    )
}

/// 2 compute slots and a RAM tier of 3: the misses run as CPU groups.
fn cpu_miss() -> (PathBuf, LoadOptions) {
    (
        fixture(),
        LoadOptions {
            expert_slots: Some(2),
            ram_slots: Some(3),
            miss: MissPolicy::Cpu,
            ..base()
        },
    )
}

/// The simulated disk of `overlap.rs`: reads land when a wait moves a virtual clock.
fn sim_io() -> IoFactory {
    Arc::new(|paths| {
        let io = SimIo::flat(paths, SimClock::new(), 1e8, 4, 4096)?;
        Ok(Box::new(io))
    })
}

/// Top-8 of 32 experts in Q4_K_M, 8 compute and 12 RAM slots, prefetch two layers ahead, the I/O
/// service with overlapped CPU groups.
fn overlap() -> (PathBuf, LoadOptions) {
    (
        quantized_moe(&MoeSpec::REALISH, Ftype::Q4KM, "q4km-prefetch"),
        LoadOptions {
            expert_slots: Some(8),
            ram_slots: Some(12),
            miss: MissPolicy::Cpu,
            prefetch: true,
            prefetch_depth: 2,
            io: IoOptions {
                service: true,
                workers: 4,
                ..IoOptions::default()
            },
            overlap: true,
            io_factory: Some(sim_io()),
            ..base()
        },
    )
}

/// hardware: cpu.
#[test]
fn cpu_resident_batched_equals_alone() {
    let (path, o) = resident();
    let m = batched_equals_alone(&path, &o, TOL_CPU, "resident");
    assert!(m.expert_stats().is_none());
}

/// hardware: cpu.
#[test]
fn cpu_streamed_upload_batched_equals_alone() {
    let (path, o) = upload();
    let m = batched_equals_alone(&path, &o, TOL_CPU, "upload");
    let s = m.expert_stats().expect("streaming");
    assert!(s.io.experts_read > 0);
    // 3 sequences give up to 24 distinct experts a layer step against 2 slots: pair groups.
    assert!(
        m.expert_groups() > 0,
        "more experts than slots run as groups"
    );
}

/// hardware: cpu.
#[test]
fn cpu_streamed_cpu_miss_batched_equals_alone() {
    let (path, o) = cpu_miss();
    let m = batched_equals_alone(&path, &o, TOL_CPU, "cpu miss");
    let s = m.expert_stats().expect("streaming");
    assert!(s.io.experts_read > 0);
    assert!(m.expert_groups() > 0);
}

/// hardware: cpu. Prefetch, the I/O service and overlap see 3 tokens per decode step and 40
/// more in a mixed one: the same logits as the serial streamed path on the same batches, and the
/// reads still happen.
#[test]
fn cpu_streamed_overlap_prefetch_matches_the_serial_path() {
    let (path, o) = overlap();
    let serial = LoadOptions {
        prefetch: false,
        io: IoOptions::default(),
        overlap: false,
        io_factory: None,
        ..o.clone()
    };
    let m = batched_paths_agree(&path, &o, &serial, TOL_CPU, "overlap");
    let s = m.expert_stats().expect("streaming");
    assert!(s.io.experts_read > 0);
    assert!(s.io.prefetch_issued > 0, "prefetch ran with several tokens");
}

/// hardware: cpu. The routing log lists the tokens of a step in step order.
#[test]
fn cpu_routing_log_records_every_token_of_a_step() {
    let (path, o) = resident();
    let o = LoadOptions {
        record_routing: true,
        ..o
    };
    let mut m = load(&path, &o, 3);
    let (a, b) = (prompt(1, 5), prompt(2, 3));
    let mut buf = Vec::new();
    m.decode(
        &[
            SeqEval {
                seq: 2,
                tokens: &a,
                all_logits: false,
            },
            SeqEval {
                seq: 0,
                tokens: &b,
                all_logits: false,
            },
        ],
        &mut buf,
    )
    .unwrap();
    let log = m.take_routing_log();
    let n_used = m.hparams().n_expert_used as usize;
    assert_eq!(log.len(), m.hparams().n_layer as usize);
    assert!(
        log.iter()
            .all(|r| r.n_tokens == 8 && r.ids.len() == 8 * n_used)
    );
    // The last token of the second input routes layer 0 as the same prompt alone does.
    let mut alone = load(&path, &o, 1);
    alone.eval_last(&b).unwrap();
    let lone = alone.take_routing_log();
    let sorted = |ids: &[i32]| {
        let mut v = ids.to_vec();
        v.sort_unstable();
        v
    };
    assert_eq!(
        sorted(&log[0].ids[7 * n_used..8 * n_used]),
        sorted(&lone[0].ids[2 * n_used..3 * n_used])
    );
}

/// hardware: cpu.
#[test]
fn cpu_seq_cp_continues_resident_and_streamed() {
    for (name, (path, o)) in [("resident", resident()), ("cpu miss", cpu_miss())] {
        seq_cp_continues(&path, &o, TOL_CPU, name);
    }
    // A quantized KV cache copies as rows of its own type.
    let path = quantized_moe(&MoeSpec::REALISH, Ftype::Q8_0, "kv-q8");
    let o = LoadOptions {
        kv: KvFormat::Q8_0,
        kv_rotation: Some(true),
        ..base()
    };
    seq_cp_continues(&path, &o, TOL_CPU, "q8_0 kv");
}

/// hardware: cpu.
#[test]
fn cpu_snapshot_roundtrip_per_sequence() {
    for (name, (path, o)) in [("resident", resident()), ("upload", upload())] {
        snapshot_roundtrip(&path, &o, TOL_CPU, name);
    }
}

/// hardware: cpu. Sequence 0 outgrows its 256 rows with a neighbour behind it, so the layout
/// moves it while it decodes.
#[test]
fn cpu_chunked_move_mid_decode_keeps_the_logits() {
    for (name, (path, o)) in [("resident", resident()), ("cpu miss", cpu_miss())] {
        let o = LoadOptions { n_ctx: 1024, ..o };
        move_mid_decode(&path, &o, TOL_CPU, name);
    }
    let path = quantized_moe(&MoeSpec::REALISH, Ftype::Q8_0, "kv-q8");
    let o = LoadOptions {
        n_ctx: 1024,
        kv: KvFormat::Q8_0,
        kv_rotation: Some(true),
        ..base()
    };
    move_mid_decode(&path, &o, TOL_CPU, "q8_0 kv");
}

/// hardware: cpu. The move with 8 threads, as on a box where the GPU builder's CPU attention runs
/// with the container's ~12: ggml's CPU flash attention splits a one-query step over `n_kv` in
/// `n_kv / threads` chunks once `n_kv` reaches 512 and accumulates V in F16, so the order of the
/// sums depends on where a sequence's rows sit and on `n_kv` (the CUDA kernels' split does too).
#[test]
fn cpu_chunked_move_mid_decode_with_8_threads() {
    let o = LoadOptions {
        n_ctx: 1024,
        ..LoadOptions::new(BackendChoice::Cpu { threads: Some(8) })
    };
    move_mid_decode(&fixture(), &o, TOL_CPU, "8 threads");
}

/// hardware: cpu. Pool limits: a full pool is an error that changes nothing, ids are checked.
#[test]
fn cpu_full_pool_and_bad_ids_change_nothing() {
    let (path, o) = resident();
    let o = LoadOptions { n_ctx: 512, ..o };
    let mut m = load(&path, &o, 2);
    let mut buf = Vec::new();
    let mut step = |m: &mut Qwen3Moe, seq: u32, n: usize| {
        let t = prompt(6, n);
        m.decode(
            &[SeqEval {
                seq,
                tokens: &t,
                all_logits: false,
            }],
            &mut buf,
        )
    };
    step(&mut m, 0, 300).unwrap();
    // 300 positions hold 512 rows; sequence 1 needs 256 more rows than are free.
    assert_eq!(m.pool().free(), 0);
    assert!(step(&mut m, 1, 5).is_err());
    assert_eq!((m.seq_len(0), m.seq_len(1), m.pool().free()), (300, 0, 0));
    assert!(!m.try_reserve(1, 1).unwrap());
    assert!(step(&mut m, 2, 1).is_err(), "sequence 2 does not exist");
    assert!(step(&mut m, 0, 300).is_err(), "beyond the pool");
    assert_eq!(m.seq_len(0), 300);
    // Freeing sequence 0 lets sequence 1 in; sequence 0 can come back.
    m.seq_rm(0, 0).unwrap();
    step(&mut m, 1, 5).unwrap();
    assert_eq!((m.seq_len(0), m.seq_len(1)), (0, 5));
    // The same tokens twice in one step, and an empty input, are refused.
    let t = [1, 2];
    let twice = [
        SeqEval {
            seq: 0,
            tokens: &t,
            all_logits: false,
        },
        SeqEval {
            seq: 0,
            tokens: &t,
            all_logits: false,
        },
    ];
    assert!(m.decode(&twice, &mut buf).is_err());
    let empty = [SeqEval {
        seq: 0,
        tokens: &[],
        all_logits: false,
    }];
    assert!(m.decode(&empty, &mut buf).is_err());
    assert_eq!((m.seq_len(0), m.seq_len(1)), (0, 5));
}

/// hardware: cpu. `all_logits` returns every row of that input, in order, next to a last-only
/// input.
#[test]
fn cpu_all_logits_rows_follow_the_input_order() {
    let (path, o) = resident();
    let p = prompt(7, 9);
    let mut m = load(&path, &o, 2);
    let (mut a, mut buf) = (Vec::new(), Vec::new());
    let spans = m
        .decode(
            &[
                SeqEval {
                    seq: 1,
                    tokens: &p,
                    all_logits: true,
                },
                SeqEval {
                    seq: 0,
                    tokens: &p[..3],
                    all_logits: false,
                },
            ],
            &mut buf,
        )
        .unwrap();
    assert_eq!(spans, [0..9, 9..10]);
    let nv = m.n_vocab();
    for k in [0usize, 4, 8] {
        let mut lone = load(&path, &o, 1);
        a.clear();
        a.extend(lone.eval_last(&p[..=k]).unwrap());
        close(&buf[k * nv..(k + 1) * nv], &a, TOL_CPU, &format!("row {k}"));
    }
}

/// libllama with `n_seq_max = 3` and the unified cache, the same steps.
fn libllama_steps(toks: &[Vec<i32>], steps: &[Step]) -> Vec<Vec<Vec<f32>>> {
    tetraeder_ggml_sys::llama::disable_attn_rot();
    let model = Arc::new(
        Model::load(
            fixture(),
            &ModelParams {
                n_gpu_layers: 0,
                extra_bufts: false,
                ..ModelParams::default()
            },
        )
        .unwrap(),
    );
    let mut ctx = Context::new(
        &model,
        &ContextParams {
            n_ctx: 1024,
            n_batch: 64,
            n_ubatch: 64,
            n_threads: 2,
            n_threads_batch: 2,
            flash_attn: FlashAttn::On,
            n_seq_max: 3,
            kv_unified: true,
            ..ContextParams::default()
        },
    )
    .unwrap();
    let mut batch = Batch::new(64);
    let mut out = Vec::new();
    for s in steps {
        batch.clear();
        let mut at = Vec::new();
        for (k, r) in s {
            for p in r.clone() {
                // Logits of the last token of each input only.
                batch
                    .push(toks[*k][p], p as i32, *k as i32, p + 1 == r.end)
                    .unwrap();
            }
        }
        ctx.decode(&batch).unwrap();
        let mut i = 0;
        for (_, r) in s {
            i += r.len();
            at.push(ctx.logits_ith(i as i32 - 1).unwrap().to_vec());
        }
        out.push(at);
    }
    out
}

/// hardware: cpu. The builder's batched steps against libllama's unified cache running the same
/// steps, and both against each sequence alone.
#[test]
fn cpu_batched_matches_libllama_unified_multi_sequence() {
    let (path, o) = resident();
    let prompts = [prompt(1, 7), prompt(2, 21), prompt(3, 24)];
    let n = 6;
    let alones: Vec<Alone> = prompts.iter().map(|p| alone(&path, &o, p, n)).collect();
    let toks: Vec<Vec<i32>> = alones.iter().map(|a| a.toks.clone()).collect();
    let lens: Vec<usize> = prompts.iter().map(Vec::len).collect();
    let total: Vec<usize> = toks.iter().map(Vec::len).collect();
    let steps = schedule(&lens, &total, 16);
    let mut m = load(&path, &o, 3);
    let ours = run_steps(&mut m, &toks, &steps);
    let theirs = libllama_steps(&toks, &steps);
    assert_eq!(ours.len(), theirs.len());
    for (si, (a, b)) in ours.iter().zip(&theirs).enumerate() {
        for (ii, (x, y)) in a.iter().zip(b).enumerate() {
            let what = format!("step {si}, input {ii}");
            close(x, y, TOL_CPU, &what);
            same_token(x, y, TOL_CPU, &what);
        }
    }
    check_steps(&theirs, &steps, &alones, TOL_CPU, "libllama");
}

/// hardware: cpu. `libllama_anchor` on the CPU, the body `gpu_multi_seq.rs` runs on CUDA.
#[test]
fn cpu_libllama_batched_vs_split_anchor() {
    let (path, o) = resident();
    let worst = libllama_anchor(&path, &o, 0, common::Tol::from(TOL_CPU), "cpu");
    eprintln!("cpu anchor worst (of largest logit): {worst:?}");
}
