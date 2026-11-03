//! Scores the planner on one measured fixture (ADR 0040): calibrate the cost model from the
//! anchor runs only, predict every candidate, and compare with the measured request times.
//!
//! Inputs are the files the box campaign writes into `bench/results/<date>/planner-<fixture>/`:
//! `layout.json`, `candidates.json` and the harness results (`planner-<fixture>-<time>.json`).

use crate::{CampaignVariant, CandidatesFile, ModelLayout, Placement};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;
use tetraeder_costmodel::calib::{Anchor, fit};
use tetraeder_costmodel::predict::{DevicePart, RunFeatures, predict};
use tetraeder_profiler::HardwareGraph;

/// Spec-sheet prior of the RTX 3090 box (the fit replaces what the anchors measure).
pub const RTX3090_PRIOR: &str = include_str!("../fixtures/rtx3090_prior.toml");

/// Graph nodes per layer and device: a rough count; c0 is fitted, so only the ratio matters.
const OPS_DENSE_LAYER: f64 = 24.0;
const OPS_EXPERT_LAYER: f64 = 10.0;
const OPS_OUTPUT: f64 = 4.0;

/// The weights of one variant per device: `None` placement means all on the CPU.
pub fn features(
    layout: &ModelLayout,
    placement: Option<Placement>,
    threads: u32,
    args: &[String],
    env: &BTreeMap<String, String>,
) -> RunFeatures {
    let l = layout.n_layer as usize;
    let bpp = layout.main_format().bits_per_weight() / 8.0;
    let r = if layout.n_expert > 0 {
        f64::from(layout.n_expert_used) / f64::from(layout.n_expert)
    } else {
        0.0
    };
    let kv_layer = 2.0 * f64::from(layout.n_head_kv) * f64::from(layout.head_dim) * 2.0;
    let attn_layer = 4.0 * f64::from(layout.n_head) * f64::from(layout.head_dim);
    // Per layer: (dense on GPU, experts on GPU).
    let (on_gpu, out_gpu): (Vec<(bool, bool)>, bool) = match placement {
        None => (vec![(false, false); l], false),
        Some(Placement::Dense { gpu_layers }) => {
            let g = (gpu_layers as usize).min(l);
            ((0..l).map(|i| (i >= l - g, i >= l - g)).collect(), g > 0)
        }
        Some(Placement::Moe { gpu_expert_layers }) => {
            let k = gpu_expert_layers as usize;
            ((0..l).map(|i| (true, i < k)).collect(), true)
        }
    };
    let mut parts: BTreeMap<&str, DevicePart> = BTreeMap::new();
    let mut add = |dev: &'static str, bytes: f64, active: f64, ops: f64, kv: f64, attn: f64| {
        let p = parts.entry(dev).or_insert_with(|| DevicePart {
            device: dev.into(),
            weight_bytes_per_token: 0.0,
            params_per_token: 0.0,
            resident_matmul_bytes: 0.0,
            ops_per_token: 0.0,
            kv_bytes_per_ctx_token: 0.0,
            attn_flops_per_ctx_token: 0.0,
        });
        p.weight_bytes_per_token += active;
        p.params_per_token += active / bpp;
        p.resident_matmul_bytes += bytes;
        p.ops_per_token += ops;
        p.kv_bytes_per_ctx_token += kv;
        p.attn_flops_per_ctx_token += attn;
    };
    let dev = |gpu: bool| if gpu { "gpu0" } else { "cpu0" };
    let mut switches = 0.0;
    for (i, lb) in layout.layers.iter().enumerate() {
        let (dense, experts) = on_gpu[i];
        let d = (lb.attn_bytes + lb.ffn_bytes) as f64;
        add(dev(dense), d, d, OPS_DENSE_LAYER, kv_layer, attn_layer);
        if lb.expert_bytes > 0 {
            let e = lb.expert_bytes as f64;
            add(dev(experts), e, r * e, OPS_EXPERT_LAYER, 0.0, 0.0);
            if dense != experts {
                switches += 2.0;
            }
        }
    }
    let o = layout.output_bytes as f64;
    add(dev(out_gpu), o, o, OPS_OUTPUT, 0.0, 0.0);
    // Transitions along the dense path: embed (CPU), layers in order, output.
    let mut path = vec![false];
    path.extend(on_gpu.iter().map(|x| x.0));
    path.push(out_gpu);
    switches += path.windows(2).filter(|w| w[0] != w[1]).count() as f64;
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse::<u32>().ok())
    };
    RunFeatures {
        parts: parts.into_values().collect(),
        switches_per_token: switches,
        activation_bytes_per_token: 4.0 * f64::from(layout.n_embd),
        threads,
        ubatch: flag("-ub").unwrap_or(512),
        op_offload: placement.is_some() && flag("-nopo") != Some(1),
        graphs: env.get("GGML_CUDA_DISABLE_GRAPHS").is_none_or(|v| v != "1"),
    }
}

/// Median pp and tg tokens per second per variant over every sample in the harness results.
pub fn measured(results: &Value) -> BTreeMap<String, (Option<f64>, Option<f64>)> {
    let mut acc: BTreeMap<String, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    for pair in results["pairs"].as_array().into_iter().flatten() {
        for s in pair["samples"].as_array().into_iter().flatten() {
            let Some(v) = s["variant"].as_str() else {
                continue;
            };
            let e = acc.entry(v.to_string()).or_default();
            if let Some(x) = s["pp_ts"].as_f64() {
                e.0.push(x);
            }
            if let Some(x) = s["tg_ts"].as_f64() {
                e.1.push(x);
            }
        }
    }
    acc.into_iter()
        .map(|(k, (pp, tg))| (k, (median(pp), median(tg))))
        .collect()
}

fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    Some(if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    })
}

/// Spearman rank correlation with average ranks for ties.
pub fn spearman(a: &[f64], b: &[f64]) -> f64 {
    fn ranks(x: &[f64]) -> Vec<f64> {
        let mut idx: Vec<usize> = (0..x.len()).collect();
        idx.sort_by(|&i, &j| x[i].total_cmp(&x[j]));
        let mut r = vec![0.0; x.len()];
        let mut i = 0;
        while i < idx.len() {
            let mut j = i;
            while j + 1 < idx.len() && x[idx[j + 1]] == x[idx[i]] {
                j += 1;
            }
            for &k in &idx[i..=j] {
                r[k] = (i + j) as f64 / 2.0;
            }
            i = j + 1;
        }
        r
    }
    let (ra, rb) = (ranks(a), ranks(b));
    let n = a.len() as f64;
    let ma = ra.iter().sum::<f64>() / n;
    let mb = rb.iter().sum::<f64>() / n;
    let cov: f64 = ra.iter().zip(&rb).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let va: f64 = ra.iter().map(|x| (x - ma).powi(2)).sum();
    let vb: f64 = rb.iter().map(|y| (y - mb).powi(2)).sum();
    cov / (va * vb).sqrt()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateScore {
    pub id: String,
    pub description: String,
    pub predicted_s: f64,
    /// `None` when the run failed on the box.
    pub measured_s: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Score {
    pub fixture: String,
    pub candidates: Vec<CandidateScore>,
    /// Candidates the planner predicted feasible that failed on the box.
    pub failed: Vec<String>,
    pub analytic_pick: String,
    pub final_pick: String,
    pub best: String,
    /// Measured time of the pick over the best measured time, minus 1.
    pub regret_analytic: f64,
    pub regret_final: f64,
    pub spearman: f64,
    /// Relative error band from the anchor residuals.
    pub band: f64,
    pub anchor_residuals: Vec<(String, f64)>,
    /// Measured request seconds of each baseline next to the final pick's.
    pub baselines: BTreeMap<String, f64>,
    pub pick_s: f64,
    /// Fitted rates, for the record.
    pub calibrated: HardwareGraph,
}

fn request_s(v: &CampaignVariant, pp: Option<f64>, tg: Option<f64>) -> Option<f64> {
    Some(f64::from(v.prompt) / pp? + f64::from(v.r#gen) / tg?)
}

/// Scores one fixture directory.
pub fn score_dir(dir: &Path) -> Result<Score, String> {
    let read = |p: &Path| std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()));
    let layout: ModelLayout =
        serde_json::from_str(&read(&dir.join("layout.json"))?).map_err(|e| e.to_string())?;
    let cands: CandidatesFile =
        serde_json::from_str(&read(&dir.join("candidates.json"))?).map_err(|e| e.to_string())?;
    let mut results = BTreeMap::new();
    for e in std::fs::read_dir(dir).map_err(|e| e.to_string())? {
        let p = e.map_err(|e| e.to_string())?.path();
        let name = p
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        if name.starts_with("planner-") && name.ends_with(".json") {
            let v: Value = serde_json::from_str(&read(&p)?).map_err(|e| e.to_string())?;
            results.extend(measured(&v));
        }
    }
    let fixture = dir
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    score(&fixture, &layout, &cands, &results, cands.budget.threads)
}

/// Calibrates on the anchors, predicts the candidates, compares.
pub fn score(
    fixture: &str,
    layout: &ModelLayout,
    cands: &CandidatesFile,
    measured: &BTreeMap<String, (Option<f64>, Option<f64>)>,
    quota: u32,
) -> Result<Score, String> {
    let mut prior = HardwareGraph::from_toml(RTX3090_PRIOR).map_err(|e| e.to_string())?;
    if let Some(c) = prior.devices.iter_mut().find(|d| d.id == "cpu0") {
        c.threads = Some(quota);
    }
    let feat = |v: &CampaignVariant| features(layout, v.placement, v.threads, &v.args, &v.env);
    let anchors: Vec<Anchor> = cands
        .anchors
        .iter()
        .filter_map(|v| {
            let &(pp, tg) = measured.get(&v.id)?;
            Some(Anchor {
                id: v.id.clone(),
                features: feat(v),
                prompt: v.prompt,
                r#gen: v.r#gen,
                pp_tok_s: pp,
                tg_tok_s: tg,
            })
        })
        .collect();
    let f = fit(&prior, &anchors, "gpu0", "cpu0", "pcie-gpu0").map_err(|e| format!("{e:?}"))?;
    let mut scores = Vec::new();
    let mut failed = Vec::new();
    for e in &cands.candidates {
        let v = &e.variant;
        let p = predict(&f.hw, &feat(v), v.prompt, v.r#gen).map_err(|e| format!("{e:?}"))?;
        let m = measured
            .get(&v.id)
            .and_then(|&(pp, tg)| request_s(v, pp, tg));
        if m.is_none() {
            failed.push(e.candidate.id.clone());
        }
        scores.push(CandidateScore {
            id: e.candidate.id.clone(),
            description: v.args.join(" ") + &format!(" -t {}", v.threads),
            predicted_s: p.request_s,
            measured_s: m,
        });
    }
    let ran: Vec<&CandidateScore> = scores.iter().filter(|c| c.measured_s.is_some()).collect();
    if ran.len() < 2 {
        return Err(format!("{fixture}: only {} candidates ran", ran.len()));
    }
    let by_pred =
        |a: &&CandidateScore, b: &&CandidateScore| a.predicted_s.total_cmp(&b.predicted_s);
    let ms = |c: &CandidateScore| c.measured_s.unwrap_or(f64::INFINITY);
    // The pick is chosen among all candidates: one that failed on the box counts as infinitely slow.
    let mut order: Vec<&CandidateScore> = scores.iter().collect();
    order.sort_by(by_pred);
    let analytic = order[0];
    // Close call: the top two within the band are both benchmarked; the faster measured wins.
    let fin = if order.len() > 1 && order[1].predicted_s <= order[0].predicted_s * (1.0 + f.band) {
        if ms(order[1]) < ms(order[0]) {
            order[1]
        } else {
            order[0]
        }
    } else {
        order[0]
    };
    let best = ran
        .iter()
        .min_by(|a, b| ms(a).total_cmp(&ms(b)))
        .copied()
        .ok_or("no candidate")?;
    let pred: Vec<f64> = ran.iter().map(|c| c.predicted_s).collect();
    let meas: Vec<f64> = ran.iter().map(|c| ms(c)).collect();
    let mut baselines = BTreeMap::new();
    for v in &cands.baselines {
        if let Some(t) = measured
            .get(&v.id)
            .and_then(|&(pp, tg)| request_s(v, pp, tg))
        {
            baselines.insert(v.id.clone(), t);
        }
    }
    Ok(Score {
        fixture: fixture.to_string(),
        failed,
        analytic_pick: analytic.id.clone(),
        final_pick: fin.id.clone(),
        best: best.id.clone(),
        regret_analytic: ms(analytic) / ms(best) - 1.0,
        regret_final: ms(fin) / ms(best) - 1.0,
        spearman: spearman(&pred, &meas),
        band: f.band,
        anchor_residuals: f.residuals.clone(),
        baselines,
        pick_s: ms(fin),
        calibrated: f.hw,
        candidates: scores,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::fixtures;

    #[test]
    fn spearman_basics() {
        assert!((spearman(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0]) - 1.0).abs() < 1e-12);
        assert!((spearman(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]) + 1.0).abs() < 1e-12);
        let s = spearman(&[1.0, 1.0, 2.0, 3.0], &[1.0, 2.0, 3.0, 4.0]);
        assert!(s > 0.9 && s < 1.0);
    }

    #[test]
    fn features_split_weights_by_placement() {
        let m = fixtures::moe();
        let none = BTreeMap::new();
        let all_gpu = features(
            &m,
            Some(Placement::Moe {
                gpu_expert_layers: m.n_layer,
            }),
            8,
            &[],
            &none,
        );
        assert_eq!(all_gpu.parts.len(), 1);
        let half = features(
            &m,
            Some(Placement::Moe {
                gpu_expert_layers: m.n_layer / 2,
            }),
            8,
            &["-ub".into(), "2048".into(), "-nopo".into(), "1".into()],
            &none,
        );
        assert_eq!(half.ubatch, 2048);
        assert!(!half.op_offload);
        assert_eq!(
            half.switches_per_token,
            1.0 + 2.0 * f64::from(m.n_layer / 2)
        );
        let cpu = half.parts.iter().find(|p| p.device == "cpu0").unwrap();
        let gpu = half.parts.iter().find(|p| p.device == "gpu0").unwrap();
        assert!(
            cpu.resident_matmul_bytes > 0.0
                && gpu.weight_bytes_per_token > cpu.weight_bytes_per_token
        );
        let cpu_only = features(&m, None, 8, &[], &none);
        assert_eq!(cpu_only.parts.len(), 1);
        assert!(!cpu_only.op_offload);
        assert_eq!(cpu_only.switches_per_token, 0.0);
        let d = fixtures::dense();
        let g = features(&d, Some(Placement::Dense { gpu_layers: 40 }), 8, &[], &none);
        // embed on the CPU, CPU layers, then GPU layers and the output: one transition.
        assert_eq!(g.switches_per_token, 1.0);
    }

    /// The exit test (ADR 0040) on every committed measured fixture: regret at most 10% and
    /// Spearman at least 0.8 over at least 20 candidates. Skips cleanly when nothing is committed.
    #[test]
    fn committed_fixtures_meet_the_exit_test() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/results");
        let mut dirs = Vec::new();
        for day in std::fs::read_dir(&root).into_iter().flatten().flatten() {
            for d in std::fs::read_dir(day.path())
                .into_iter()
                .flatten()
                .flatten()
            {
                let name = d.file_name().to_string_lossy().to_string();
                if name.starts_with("planner-") && d.path().join("candidates.json").is_file() {
                    dirs.push(d.path());
                }
            }
        }
        if dirs.is_empty() {
            eprintln!("skip: no measured planner fixture committed");
            return;
        }
        for d in dirs {
            let s = score_dir(&d).unwrap();
            let ran = s
                .candidates
                .iter()
                .filter(|c| c.measured_s.is_some())
                .count();
            eprintln!(
                "{}: {ran} ran, regret analytic {:.3} final {:.3}, spearman {:.3}, band {:.3}",
                s.fixture, s.regret_analytic, s.regret_final, s.spearman, s.band
            );
            assert!(ran >= 20, "{}: {ran} candidates ran", s.fixture);
            assert!(
                s.regret_final <= 0.10,
                "{}: regret {}",
                s.fixture,
                s.regret_final
            );
            assert!(s.spearman >= 0.8, "{}: spearman {}", s.fixture, s.spearman);
        }
    }
}
