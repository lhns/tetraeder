//! Temporary diagnostic for the pin move (track-pin-adopt): one multi-row step against one-row
//! steps on the spec_builder fixture and variants of it. Prints only; runs with
//! TETRAEDER_BATCH_DIAG=1.

mod common;

use common::synth::{MoeSpec, quantized_moe, write_moe};
use std::path::Path;
use tetraeder_ggml_sys::llama::Ftype;
use tetraeder_model::{BackendChoice, LoadOptions, Qwen3Moe};

const N_PROMPT: usize = 24;
const N_ROWS: usize = 7;

fn run(name: &str, path: &Path, threads: u32, rows: usize) {
    let mut o = LoadOptions::new(BackendChoice::Cpu {
        threads: Some(threads as usize),
    });
    o.n_ctx = 256;
    o.record_routing = true;
    let mut m = Qwen3Moe::load(path, &o).unwrap();
    let p: Vec<i32> = (0..N_PROMPT as i32).map(|i| (i * 37 + 5) % 256).collect();
    m.rewind(0).unwrap();
    m.eval_last(&p).unwrap();
    let n_past = N_PROMPT;
    m.take_routing_log();
    let toks: Vec<i32> = (0..rows as i32).map(|i| (i * 11 + 3) % 256).collect();
    let nv = m.n_vocab();
    let batched = m.eval(&toks).unwrap();
    let blog = m.take_routing_log();
    m.rewind(n_past).unwrap();
    for (i, &t) in toks.iter().enumerate() {
        let one = m.eval(&[t]).unwrap();
        let slog = m.take_routing_log();
        let row = &batched[i * nv..(i + 1) * nv];
        let scale = one.iter().fold(0f32, |a, x| a.max(x.abs()));
        let diff = one
            .iter()
            .zip(row)
            .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        let mut notes = Vec::new();
        for s in &slog {
            if s.n_tokens != 1 {
                continue;
            }
            let Some(b) = blog
                .iter()
                .find(|b| b.layer == s.layer && b.n_tokens == rows)
            else {
                continue;
            };
            let nu = s.ids.len();
            let bids = &b.ids[i * nu..(i + 1) * nu];
            let bw = &b.weights[i * nu..(i + 1) * nu];
            let mut a: Vec<i32> = s.ids.clone();
            let mut c: Vec<i32> = bids.to_vec();
            a.sort();
            c.sort();
            let wdiff = s
                .ids
                .iter()
                .zip(&s.weights)
                .filter_map(|(id, w)| bids.iter().position(|x| x == id).map(|j| (w - bw[j]).abs()))
                .fold(0f32, f32::max);
            let wmin = s.weights.iter().cloned().fold(f32::MAX, f32::min);
            notes.push(format!(
                "L{} {} wdiff {:.2e} wmin {:.4}",
                s.layer,
                if a == c { "same" } else { "FLIP" },
                wdiff,
                wmin
            ));
        }
        println!(
            "diag {name} t{threads} n{rows} row {i}: rel {:.3e} ({diff} of {scale}) [{}]",
            diff / scale,
            notes.join("; ")
        );
    }
}

#[test]
fn batch_diag() {
    if std::env::var("TETRAEDER_BATCH_DIAG").is_err() {
        return;
    }
    let info = tetraeder_ggml_sys::cpu::backend_info().expect("CPU backend");
    println!("diag isa {:?}", info.isa);
    let realish = MoeSpec::REALISH;
    let all = MoeSpec {
        used: 32,
        ..MoeSpec::REALISH
    };
    let one_layer = MoeSpec {
        layers: 1,
        ..MoeSpec::REALISH
    };
    let f32_path = std::env::temp_dir().join(format!("diag-f32-{}.gguf", std::process::id()));
    write_moe(&f32_path, &realish);
    let fixtures = [
        ("q4km", quantized_moe(&realish, Ftype::Q4KM, "diag-q4km")),
        ("q4km-all32", quantized_moe(&all, Ftype::Q4KM, "diag-all")),
        (
            "q4km-1layer",
            quantized_moe(&one_layer, Ftype::Q4KM, "diag-1l"),
        ),
        ("q8_0", quantized_moe(&realish, Ftype::Q8_0, "diag-q8")),
        ("f16", quantized_moe(&realish, Ftype::F16, "diag-f16")),
        ("f32", f32_path),
    ];
    for (name, path) in &fixtures {
        for threads in [1, 2] {
            run(name, path, threads, N_ROWS);
        }
    }
    for rows in 2..N_ROWS {
        run("q4km", &fixtures[0].1, 2, rows);
    }
}
