//! Predictions without `ranks` are bit-identical to those recorded before the multi-rank model
//! (ADR 0070, brief E3). The values below were recorded on the commit before `RankFeatures`.

use crate::predict::fixture::{features, hw};
use crate::predict::{RunFeatures, predict};
use tetraeder_profiler::HardwareGraph;

const TWO_GPU: &str = include_str!("../fixtures/two_gpu_cpu_nvme.toml");
const CALIB: &str = include_str!("../../../bench/results/20261103/planner-moe/calib.json");

/// `(case, [pp_tok_s, tg_tok_s, request_s])` as f64 bits.
type Row = (&'static str, [u64; 3]);

const RECORDED: &[Row] = &[];

fn graphs() -> Vec<(&'static str, HardwareGraph)> {
    let calib: serde_json::Value = serde_json::from_str(CALIB).unwrap();
    let fit = HardwareGraph::from_json(&calib["hw"].to_string()).unwrap();
    vec![
        ("fixture", hw()),
        (
            "two_gpu_cpu_nvme",
            HardwareGraph::from_toml(TWO_GPU).unwrap(),
        ),
        ("committed_fit", fit),
    ]
}

fn cases() -> Vec<(String, RunFeatures)> {
    let mut v = Vec::new();
    for (g, c) in [(10.0, 0.0), (6.0, 4.0), (0.0, 8.0), (20.0, 2.0)] {
        for (graphs, threads, ubatch) in [(true, 8, 512), (false, 4, 2048)] {
            let f = RunFeatures {
                graphs,
                threads,
                ubatch,
                ..features(g, c)
            };
            v.push((format!("g{g}-c{c}-graphs{graphs}-t{threads}-ub{ubatch}"), f));
        }
    }
    v
}

fn measure() -> Vec<(String, [u64; 3])> {
    let mut out = Vec::new();
    for (name, hw) in graphs() {
        for (case, f) in cases() {
            let p = match predict(&hw, &f, 2048, 128) {
                Ok(p) => [
                    p.pp_tok_s.to_bits(),
                    p.tg_tok_s.to_bits(),
                    p.request_s.to_bits(),
                ],
                Err(_) => [0; 3],
            };
            out.push((format!("{name}/{case}"), p));
        }
    }
    out
}

#[test]
fn predictions_without_ranks_match_the_recorded_bits() {
    let got = measure();
    if RECORDED.is_empty() {
        let mut s = String::from("RECORD\n");
        for (k, b) in &got {
            s += &format!("    (\"{k}\", [{:#x}, {:#x}, {:#x}]),\n", b[0], b[1], b[2]);
        }
        panic!("{s}");
    }
    assert_eq!(got.len(), RECORDED.len());
    for ((k, b), (rk, rb)) in got.iter().zip(RECORDED) {
        assert_eq!(k, rk);
        assert_eq!(b, rb, "{k}");
    }
}
