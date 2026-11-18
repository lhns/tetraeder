//! MODEL_VERSION 2 (ADR 0070, History 2026-11-18) against its offline replica
//! `scripts/costmodel_replica.py` on the committed S2f and S2e session files: `predict` gives
//! the replica's numbers on the replica's fitted graphs, and `calibrate` lands on the replica's
//! fit. Hardware label: none (committed files only); skips cleanly without them.

use std::path::{Path, PathBuf};
use tetraeder_planner::multi_cli::{
    MultiFile, anchor_model, calibrate, merge_comm, parse_hw, parse_layout, predict_multi,
};
use tetraeder_profiler::HardwareGraph;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(p: &Path) -> String {
    std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// `(model, id) -> (pp, tg)` of every candidate of `dir` predicted on `hw`.
fn predictions(hw: &HardwareGraph, dir: &Path) -> Vec<(String, String, f64, f64)> {
    let mut out = Vec::new();
    let mut models: Vec<String> = std::fs::read_dir(dir.join("layouts"))
        .unwrap()
        .filter_map(|e| {
            let p = e.ok()?.path();
            Some(p.file_stem()?.to_str()?.to_string())
        })
        .collect();
    models.sort();
    for m in models {
        let layout = parse_layout(&read(&dir.join(format!("layouts/{m}.json")))).unwrap();
        let cands: MultiFile =
            serde_json::from_str(&read(&dir.join(format!("candidates-{m}.json")))).unwrap();
        let p = predict_multi(hw, &layout, &cands, (String::new(), None, None), "test").unwrap();
        for r in p.candidates {
            out.push((m.clone(), r.id, r.pp_tps, r.tg_tps));
        }
    }
    out
}

fn replica(name: &str) -> Option<serde_json::Value> {
    let p = repo()
        .join("bench/results/s2f-20261118/costmodel-v2")
        .join(name);
    if !p.is_file() {
        eprintln!("skip: {} not committed", p.display());
        return None;
    }
    Some(serde_json::from_str(&read(&p)).unwrap())
}

/// The largest relative difference between ours and the replica's rows.
fn worst(ours: &[(String, String, f64, f64)], rows: &serde_json::Value) -> f64 {
    let mut worst = 0.0f64;
    for r in rows.as_array().unwrap() {
        let (m, id) = (r["model"].as_str().unwrap(), r["id"].as_str().unwrap());
        let o = ours
            .iter()
            .find(|o| o.0 == m && o.1 == id)
            .unwrap_or_else(|| panic!("{m}/{id} not predicted"));
        for (a, b) in [
            (o.2, r["pp"].as_f64().unwrap()),
            (o.3, r["tg"].as_f64().unwrap()),
        ] {
            worst = worst.max((a / b - 1.0).abs());
            eprintln!("{m:<14} {id:<8} ours {a:12.4} replica {b:12.4}");
        }
    }
    worst
}

#[test]
fn predict_reproduces_the_replica_on_its_fitted_graphs() {
    for name in ["fit-s2f.json", "fit-s2e-eval-s2f.json"] {
        let Some(v) = replica(name) else { return };
        assert_eq!(v["version"], 2);
        let hw: HardwareGraph = serde_json::from_value(v["hw"].clone()).unwrap();
        let eval = repo().join(v["eval"].as_str().unwrap());
        let w = worst(&predictions(&hw, &eval), &v["rows"]);
        assert!(w < 1e-9, "{name}: largest difference {w:e}");
    }
}

/// The fit is the same algorithm in both, not the same bits: the predictions from Rust's fit
/// agree with the replica's within 1%.
#[test]
fn calibrate_lands_on_the_replica_fit() {
    for (name, fit_dir) in [
        ("fit-s2f.json", "bench/results/s2f-20261118/phase5"),
        ("fit-s2e-eval-s2f.json", "bench/results/s2e-20261118/phase5"),
    ] {
        let Some(v) = replica(name) else { return };
        let dir = repo().join(fit_dir);
        let mut hw = parse_hw(&read(&dir.join("hw.json"))).unwrap();
        merge_comm(&mut hw, &read(&dir.join("calib-comm.json"))).unwrap();
        let mut by_model: std::collections::BTreeMap<String, Vec<String>> = Default::default();
        for e in std::fs::read_dir(&dir).unwrap() {
            let p = e.unwrap().path();
            let file = p.file_name().unwrap().to_str().unwrap().to_string();
            if let Some(m) = anchor_model(&file) {
                by_model.entry(m).or_default().push(read(&p));
            }
        }
        let models: Vec<_> = by_model
            .into_iter()
            .map(|(m, files)| {
                let l = parse_layout(&read(&dir.join(format!("layouts/{m}.json")))).unwrap();
                (l, files)
            })
            .collect();
        let fit = calibrate(&hw, &models).unwrap();
        assert_eq!(fit.cost_model_version, 2);
        let rc = fit.hw.range_costs_or_prior();
        eprintln!(
            "{name}: band {:.3}, c_step {:e}, c_range {:e}, all-reduce latency share {:.3}",
            fit.band, rc.c_step_s, rc.c_range_graph_s, rc.allreduce_latency_share
        );
        let eval = repo().join(v["eval"].as_str().unwrap());
        let w = worst(&predictions(&fit.hw, &eval), &v["rows"]);
        assert!(w < 0.01, "{name}: largest difference {w}");
    }
}
