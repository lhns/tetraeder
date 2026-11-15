//! Predictions without `ranks` are bit-identical to those recorded before the multi-rank model
//! (ADR 0070, brief E3). The values below were recorded on the commit before `RankFeatures`.

use crate::predict::fixture::{features, hw};
use crate::predict::{RunFeatures, predict};
use tetraeder_profiler::HardwareGraph;

const TWO_GPU: &str = include_str!("../fixtures/two_gpu_cpu_nvme.toml");
const CALIB: &str = include_str!("../../../bench/results/20261103/planner-moe/calib.json");

/// `(case, [pp_tok_s, tg_tok_s, request_s])` as f64 bits.
type Row = (&'static str, [u64; 3]);

const RECORDED: &[Row] = &[
    (
        "fixture/g10-c0-graphstrue-t8-ub512",
        [0x409736ffcc268212, 0x4050c661360515a5, 0x400a49ce63891fb2],
    ),
    (
        "fixture/g10-c0-graphsfalse-t4-ub2048",
        [0x409731d47a5e58c9, 0x404574cecf910dd7, 0x40117322fefa3832],
    ),
    (
        "fixture/g6-c4-graphstrue-t8-ub512",
        [0x408bc20b7d4fbfee, 0x4026e5584aeaabac, 0x402af92ffe267f93],
    ),
    (
        "fixture/g6-c4-graphsfalse-t4-ub2048",
        [0x4091dbe82114bbf9, 0x401ed33d0e363d38, 0x403266d0f7ea276a],
    ),
    (
        "fixture/g0-c8-graphstrue-t8-ub512",
        [0x4035d103e0623fea, 0x4019840c408a6835, 0x405c7c299ff73d3c],
    ),
    (
        "fixture/g0-c8-graphsfalse-t4-ub2048",
        [0x4028ef8bc0de79ba, 0x40114722db7b065d, 0x40683ca89ddbffcc],
    ),
    (
        "fixture/g20-c2-graphstrue-t8-ub512",
        [0x408249b54c9bb527, 0x402cf99880fc2079, 0x4028ab68f7c50450],
    ),
    (
        "fixture/g20-c2-graphsfalse-t4-ub2048",
        [0x4083b98bb2e70fce, 0x402322ff8be4997e, 0x40309f40725954d1],
    ),
    (
        "two_gpu_cpu_nvme/g10-c0-graphstrue-t8-ub512",
        [0x409e754a3a73c53a, 0x404f3eb85f56f1b6, 0x4008caa5d612f19f],
    ),
    (
        "two_gpu_cpu_nvme/g10-c0-graphsfalse-t4-ub2048",
        [0x409e754a3a73c53a, 0x403d68d84a833ec7, 0x40159c9eb1b1686e],
    ),
    (
        "two_gpu_cpu_nvme/g6-c4-graphstrue-t8-ub512",
        [0x40912d99ed299e12, 0x40173f3c3c502eaa, 0x4037e31d7e0198f6],
    ),
    (
        "two_gpu_cpu_nvme/g6-c4-graphsfalse-t4-ub2048",
        [0x4094bc0d24d9da8e, 0x4007547b1ca4d0ed, 0x4046b7b7b5194213],
    ),
    (
        "two_gpu_cpu_nvme/g0-c8-graphstrue-t8-ub512",
        [0x402f2a6814cb3699, 0x40089fd42b7376c5, 0x4065a05f460ba800],
    ),
    (
        "two_gpu_cpu_nvme/g0-c8-graphsfalse-t4-ub2048",
        [0x401f2b6d330efc55, 0x3ff8cebe5eb12f4a, 0x4075960110cc2b11],
    ),
    (
        "two_gpu_cpu_nvme/g20-c2-graphstrue-t8-ub512",
        [0x4087870c23624656, 0x4021a90ad471dd3a, 0x4031374b1f1bfcc6],
    ),
    (
        "two_gpu_cpu_nvme/g20-c2-graphsfalse-t4-ub2048",
        [0x4088fef73e4c6803, 0x401172600107dee7, 0x403fe82489dbe428],
    ),
    (
        "committed_fit/g10-c0-graphstrue-t8-ub512",
        [0x40a8d2f2fd6d18ab, 0x4058b6901959cc72, 0x3fff07c2fa4049be],
    ),
    (
        "committed_fit/g10-c0-graphsfalse-t4-ub2048",
        [0x40a9287442584654, 0x40571f52a04e468a, 0x400028cceb3c7472],
    ),
    (
        "committed_fit/g6-c4-graphstrue-t8-ub512",
        [0x40882e538c514228, 0x401beb0c1b787e3c, 0x4034fc6c977d00ae],
    ),
    (
        "committed_fit/g6-c4-graphsfalse-t4-ub2048",
        [0x4092ebf246e54f30, 0x4011c4f9c867cd6a, 0x403e81342dec7740],
    ),
    (
        "committed_fit/g0-c8-graphstrue-t8-ub512",
        [0x401ec289b6cbbf04, 0x400d30aaaf9d4c89, 0x4072d66a9773bef2],
    ),
    (
        "committed_fit/g0-c8-graphsfalse-t4-ub2048",
        [0x4010999e23dad58d, 0x40024faeba7277cd, 0x40812b4f72442d01],
    ),
    (
        "committed_fit/g20-c2-graphstrue-t8-ub512",
        [0x408a602951110a24, 0x40267dd97211548e, 0x402b9df4fd8d2c99],
    ),
    (
        "committed_fit/g20-c2-graphsfalse-t4-ub2048",
        [0x40908c3a0677325a, 0x401e8798a5f1bbb8, 0x4032b4515203f360],
    ),
];

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
