//! Scores the planner on one measured fixture (ADR 0040): calibrate the cost model from the
//! anchor runs only, predict every candidate, and compare with the measured request times.
//!
//! Inputs are the files the box campaign writes into `bench/results/<date>/planner-<fixture>/`:
//! `layout.json`, `candidates.json` and the harness results (`planner-<fixture>-<time>.json`).

use crate::{CampaignVariant, CandidatesFile, EngineKind, ModelLayout, Placement};
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

/// The weights of one variant per device: `None` placement means all on the CPU. A builder
/// placement is treated as experts on the CPU; builder runs use [`builder_features`].
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
        None | Some(Placement::Cpu) => (vec![(false, false); l], false),
        Some(Placement::Dense { gpu_layers }) => {
            let g = (gpu_layers as usize).min(l);
            ((0..l).map(|i| (i >= l - g, i >= l - g)).collect(), g > 0)
        }
        Some(Placement::Moe { gpu_expert_layers }) => {
            let k = gpu_expert_layers as usize;
            ((0..l).map(|i| (true, i < k)).collect(), true)
        }
        Some(Placement::Builder { .. }) => ((0..l).map(|_| (true, false)).collect(), true),
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
        op_offload: !matches!(placement, None | Some(Placement::Cpu)) && flag("-nopo") != Some(1),
        graphs: env.get("GGML_CUDA_DISABLE_GRAPHS").is_none_or(|v| v != "1"),
    }
}

/// Features of a builder run (ADR 0035): the dense weights are on the GPU; of the active expert
/// bytes of a token the share `hit` runs on the GPU from VRAM slots and the rest, the misses, on
/// the CPU. A prefill touches nearly all experts, so the resident bytes (what op offload moves
/// over PCIe) split by the slot share `vram_slots / n_expert`. The RAM tier is not a feature: it
/// decides disk reads, which the cost model does not have.
pub fn builder_features(
    layout: &ModelLayout,
    vram_slots: u32,
    hit: f64,
    (threads, ubatch, op_offload): (u32, u32, bool),
) -> RunFeatures {
    let bpp = layout.main_format().bits_per_weight() / 8.0;
    let n_expert = f64::from(layout.n_expert.max(1));
    let r = f64::from(layout.n_expert_used) / n_expert;
    let f = (f64::from(vram_slots) / n_expert).clamp(0.0, 1.0);
    let hit = hit.clamp(0.0, 1.0);
    let kv_layer = 2.0 * f64::from(layout.n_head_kv) * f64::from(layout.head_dim) * 2.0;
    let attn_layer = 4.0 * f64::from(layout.n_head) * f64::from(layout.head_dim);
    let mut gpu = DevicePart {
        device: "gpu0".into(),
        ..empty_part()
    };
    let mut cpu = DevicePart {
        device: "cpu0".into(),
        ..empty_part()
    };
    let add = |p: &mut DevicePart, resident: f64, active: f64, ops: f64| {
        p.resident_matmul_bytes += resident;
        p.weight_bytes_per_token += active;
        p.params_per_token += active / bpp;
        p.ops_per_token += ops;
    };
    for lb in &layout.layers {
        let d = (lb.attn_bytes + lb.ffn_bytes) as f64;
        add(&mut gpu, d, d, OPS_DENSE_LAYER);
        gpu.kv_bytes_per_ctx_token += kv_layer;
        gpu.attn_flops_per_ctx_token += attn_layer;
        let e = lb.expert_bytes as f64;
        add(&mut gpu, f * e, hit * r * e, hit * OPS_EXPERT_LAYER);
        add(
            &mut cpu,
            (1.0 - f) * e,
            (1.0 - hit) * r * e,
            (1.0 - hit) * OPS_EXPERT_LAYER,
        );
    }
    let o = layout.output_bytes as f64;
    add(&mut gpu, o, o, OPS_OUTPUT);
    let mut parts = vec![gpu];
    if hit < 1.0 {
        parts.push(cpu);
    }
    RunFeatures {
        parts,
        // embed (CPU) to the first layer, plus a GPU to CPU and back hop per layer step with a miss
        switches_per_token: 1.0 + 2.0 * f64::from(layout.n_layer) * (1.0 - hit),
        activation_bytes_per_token: 4.0 * f64::from(layout.n_embd),
        threads,
        ubatch,
        op_offload,
        graphs: true,
    }
}

fn empty_part() -> DevicePart {
    DevicePart {
        device: String::new(),
        weight_bytes_per_token: 0.0,
        params_per_token: 0.0,
        resident_matmul_bytes: 0.0,
        ops_per_token: 0.0,
        kv_bytes_per_ctx_token: 0.0,
        attn_flops_per_ctx_token: 0.0,
    }
}

/// Share of decode layer steps with every routed expert in VRAM, as a function of the slot
/// share `vram_slots / n_expert`: linear between the measured values at the smallest and the
/// largest slot count of the anchors (clamped outside, constant with one slot count).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HitModel {
    pub f_lo: f64,
    pub h_lo: f64,
    pub f_hi: f64,
    pub h_hi: f64,
}

impl HitModel {
    /// From (slot share, measured hit rate) points; points at the same share are averaged.
    /// `None` without points.
    pub fn from_points(points: &[(f64, f64)]) -> Option<Self> {
        let end = |pick: fn(f64, f64) -> f64, start: f64| -> Option<(f64, f64)> {
            let f = points.iter().map(|p| p.0).fold(start, pick);
            let hs: Vec<f64> = points
                .iter()
                .filter(|p| (p.0 - f).abs() < 1e-12)
                .map(|p| p.1)
                .collect();
            (!hs.is_empty()).then(|| (f, hs.iter().sum::<f64>() / hs.len() as f64))
        };
        let (f_lo, h_lo) = end(f64::min, f64::INFINITY)?;
        let (f_hi, h_hi) = end(f64::max, f64::NEG_INFINITY)?;
        Some(Self {
            f_lo,
            h_lo,
            f_hi,
            h_hi,
        })
    }

    pub fn at(&self, f: f64) -> f64 {
        let h = if self.f_hi - self.f_lo < 1e-12 {
            self.h_lo
        } else {
            let t = ((f - self.f_lo) / (self.f_hi - self.f_lo)).clamp(0.0, 1.0);
            self.h_lo + t * (self.h_hi - self.h_lo)
        };
        h.clamp(0.0, 1.0)
    }
}

/// What a builder variant measured: median prefill and decode tokens per second over its
/// repeats, and the median decode full-hit rate.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BuilderRun {
    pub pp_tok_s: Option<f64>,
    pub tg_tok_s: Option<f64>,
    pub hit_rate: Option<f64>,
}

/// Builder results from the `exit_regime` JSON files of one variant (one per repeat); the last
/// run of each file counts.
pub fn builder_run(files: &[Value]) -> BuilderRun {
    let col = |f: &dyn Fn(&Value) -> Option<f64>| {
        median(
            files
                .iter()
                .filter_map(|v| v["runs"].as_array()?.last().and_then(f))
                .collect(),
        )
    };
    BuilderRun {
        pp_tok_s: col(&|r| r["prefill_tok_s"].as_f64()),
        tg_tok_s: col(&|r| r["decode_tok_s"].as_f64()),
        hit_rate: col(&|r| r["decode_layers"]["full_hit_rate_mean"].as_f64()),
    }
}

/// Reads `<variant id>.r<k>.json` files (the campaign script's output names) of a directory.
pub fn read_builder_runs(dir: &Path) -> Result<BTreeMap<String, BuilderRun>, String> {
    let mut by_id: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for e in std::fs::read_dir(dir).map_err(|e| e.to_string())? {
        let p = e.map_err(|e| e.to_string())?.path();
        let name = p
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let Some((id, rep)) = name.strip_suffix(".json").and_then(|n| n.rsplit_once(".r")) else {
            continue;
        };
        if rep.is_empty() || !rep.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        let v: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", p.display()))?;
        by_id.entry(id.to_string()).or_default().push(v);
    }
    Ok(by_id
        .into_iter()
        .map(|(id, files)| (id, builder_run(&files)))
        .collect())
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

/// Engine flags of a libllama variant, or the environment of a builder variant.
fn describe(v: &CampaignVariant) -> String {
    if v.args.is_empty() {
        let env: Vec<String> = v.env.iter().map(|(k, x)| format!("{k}={x}")).collect();
        env.join(" ")
    } else {
        v.args.join(" ") + &format!(" -t {}", v.threads)
    }
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
    let fixture = dir
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    if cands
        .candidates
        .first()
        .is_some_and(|c| c.candidate.engine == EngineKind::Builder)
    {
        let runs = read_builder_runs(dir)?;
        return score_builder(&fixture, &layout, &cands, &runs, cands.budget.threads);
    }
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
    score(&fixture, &layout, &cands, &results, cands.budget.threads)
}

/// Scores a builder fixture: the hit-rate model comes from the anchors' measured hit rates
/// ([`HitModel`]), the features from [`builder_features`], then as [`score`].
pub fn score_builder(
    fixture: &str,
    layout: &ModelLayout,
    cands: &CandidatesFile,
    runs: &BTreeMap<String, BuilderRun>,
    quota: u32,
) -> Result<Score, String> {
    let n = f64::from(layout.n_expert.max(1));
    let slots = |v: &CampaignVariant| match v.placement {
        Some(Placement::Builder { vram_slots, .. }) => Some(vram_slots),
        _ => None,
    };
    let points: Vec<(f64, f64)> = cands
        .anchors
        .iter()
        .filter_map(|v| Some((f64::from(slots(v)?) / n, runs.get(&v.id)?.hit_rate?)))
        .collect();
    let model = HitModel::from_points(&points)
        .ok_or_else(|| format!("{fixture}: no anchor measured a hit rate"))?;
    let feat = |v: &CampaignVariant| {
        let s = slots(v).unwrap_or(0);
        let env = |k: &str| v.env.get(k).map(String::as_str);
        let ub = env("TETRAEDER_EXIT_UBATCH")
            .and_then(|x| x.parse().ok())
            .unwrap_or(256);
        let op = env("TETRAEDER_EXIT_OP_OFFLOAD").is_none_or(|x| x == "1");
        builder_features(layout, s, model.at(f64::from(s) / n), (v.threads, ub, op))
    };
    let measured: BTreeMap<String, (Option<f64>, Option<f64>)> = runs
        .iter()
        .map(|(k, r)| (k.clone(), (r.pp_tok_s, r.tg_tok_s)))
        .collect();
    score_with(fixture, cands, &measured, quota, &feat)
}

/// Calibrates on the anchors, predicts the candidates, compares.
pub fn score(
    fixture: &str,
    layout: &ModelLayout,
    cands: &CandidatesFile,
    measured: &BTreeMap<String, (Option<f64>, Option<f64>)>,
    quota: u32,
) -> Result<Score, String> {
    let feat = |v: &CampaignVariant| features(layout, v.placement, v.threads, &v.args, &v.env);
    score_with(fixture, cands, measured, quota, &feat)
}

fn score_with(
    fixture: &str,
    cands: &CandidatesFile,
    measured: &BTreeMap<String, (Option<f64>, Option<f64>)>,
    quota: u32,
    feat: &dyn Fn(&CampaignVariant) -> RunFeatures,
) -> Result<Score, String> {
    let mut prior = HardwareGraph::from_toml(RTX3090_PRIOR).map_err(|e| e.to_string())?;
    if let Some(c) = prior.devices.iter_mut().find(|d| d.id == "cpu0") {
        c.threads = Some(quota);
    }
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
            description: describe(v),
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

    // hardware: none
    #[test]
    fn hit_model_interpolates_between_the_ends() {
        let m = HitModel::from_points(&[(0.0625, 0.3), (0.0625, 0.5), (0.5, 0.9), (0.25, 0.1)])
            .unwrap();
        let near = |a: f64, b: f64| assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        near(m.f_lo, 0.0625);
        near(m.h_lo, 0.4);
        near(m.f_hi, 0.5);
        near(m.h_hi, 0.9);
        near(m.at(0.5), 0.9);
        near(m.at(0.28125), 0.65);
        // clamped outside the anchors
        near(m.at(0.0), 0.4);
        near(m.at(1.0), 0.9);
        let one = HitModel::from_points(&[(0.1, 0.6)]).unwrap();
        near(one.at(0.0), 0.6);
        near(one.at(0.9), 0.6);
        assert!(HitModel::from_points(&[]).is_none());
    }

    // hardware: none
    #[test]
    fn builder_features_move_work_from_cpu_to_gpu_with_the_hit_rate() {
        let l = fixtures::moe();
        let part = |f: &RunFeatures, d: &str| f.parts.iter().find(|p| p.device == d).cloned();
        let full = builder_features(&l, 128, 1.0, (15, 256, true));
        assert_eq!(full.parts.len(), 1);
        assert_eq!(full.switches_per_token, 1.0);
        let half = builder_features(&l, 64, 0.5, (15, 256, false));
        let low = builder_features(&l, 8, 0.1, (15, 256, false));
        assert!(!half.op_offload && half.threads == 15 && half.ubatch == 256);
        let (gh, gl) = (part(&half, "gpu0").unwrap(), part(&low, "gpu0").unwrap());
        let (ch, cl) = (part(&half, "cpu0").unwrap(), part(&low, "cpu0").unwrap());
        assert!(gh.weight_bytes_per_token > gl.weight_bytes_per_token);
        assert!(ch.weight_bytes_per_token < cl.weight_bytes_per_token);
        assert!(ch.resident_matmul_bytes < cl.resident_matmul_bytes);
        assert!(half.switches_per_token < low.switches_per_token);
        // active expert bytes over both devices are conserved
        let sum = |f: &RunFeatures| {
            f.parts
                .iter()
                .map(|p| p.weight_bytes_per_token)
                .sum::<f64>()
        };
        assert!((sum(&half) - sum(&low)).abs() < 1.0);
    }

    // hardware: none
    #[test]
    fn builder_run_takes_the_median_of_the_last_runs() {
        let f = |pp: f64, tg: f64, h: f64| {
            serde_json::json!({"runs": [
                {"prefill_tok_s": 1.0, "decode_tok_s": 1.0, "decode_layers": {"full_hit_rate_mean": 0.0}},
                {"prefill_tok_s": pp, "decode_tok_s": tg, "decode_layers": {"full_hit_rate_mean": h}}
            ]})
        };
        let r = builder_run(&[f(100.0, 10.0, 0.5), f(200.0, 20.0, 0.7)]);
        assert_eq!(r.pp_tok_s, Some(150.0));
        assert_eq!(r.tg_tok_s, Some(15.0));
        assert!((r.hit_rate.unwrap() - 0.6).abs() < 1e-12);
        assert_eq!(builder_run(&[]), BuilderRun::default());
    }

    fn truth_hit(f: f64) -> f64 {
        0.35 + 0.6 * f
    }

    /// Writes `layout.json`, `candidates.json` and `exit_regime`-shaped results (two repeats
    /// per variant, named like the campaign script's) of a synthetic builder box: the RTX 3090
    /// prior is the truth, the hit rate is linear in the slot share and the noise is 2%.
    fn write_synthetic_builder(dir: &Path) {
        use crate::candidate::{Budget, Workload};
        let l = fixtures::moe();
        let b = Budget {
            gpu_bytes: 14_000_000_000,
            ram_bytes: 60_000_000_000,
            threads: 15,
        };
        let w = Workload {
            prompt: 2048,
            r#gen: 128,
        };
        let en = crate::builder::enumerate_builder(&b, &l, &w);
        let c = crate::campaign::build_builder("moe", "org/name:file.gguf", &l, &b, &w, &en);
        let file = CandidatesFile::new(&c, &l, &b, &en);
        std::fs::create_dir_all(dir).unwrap();
        let put = |name: &str, text: String| std::fs::write(dir.join(name), text).unwrap();
        put("layout.json", serde_json::to_string_pretty(&l).unwrap());
        put(
            "candidates.json",
            serde_json::to_string_pretty(&file).unwrap(),
        );
        let mut truth = HardwareGraph::from_toml(RTX3090_PRIOR).unwrap();
        if let Some(cpu) = truth.devices.iter_mut().find(|d| d.id == "cpu0") {
            cpu.threads = Some(b.threads);
        }
        let n = f64::from(l.n_expert);
        let variants = file
            .anchors
            .iter()
            .chain(file.candidates.iter().map(|e| &e.variant));
        for (i, v) in variants.enumerate() {
            let Some(Placement::Builder { vram_slots, .. }) = v.placement else {
                panic!("{}: {:?}", v.id, v.placement)
            };
            let hit = truth_hit(f64::from(vram_slots) / n);
            let env = |k: &str| v.env[&format!("TETRAEDER_EXIT_{k}")].clone();
            let feats = builder_features(
                &l,
                vram_slots,
                hit,
                (
                    v.threads,
                    env("UBATCH").parse().unwrap(),
                    env("OP_OFFLOAD") == "1",
                ),
            );
            let p = predict(&truth, &feats, v.prompt, v.r#gen).unwrap();
            for rep in 1..=2 {
                let noise = 1.0 + 0.02 * ((i * 7 + rep * 3) as f64).sin();
                let run = serde_json::json!({"runs": [{
                    "prefill_tok_s": p.pp_tok_s * noise,
                    "decode_tok_s": p.tg_tok_s * noise,
                    "decode_layers": {"full_hit_rate_mean": hit},
                }]});
                put(&format!("{}.r{rep}.json", v.id), run.to_string());
            }
        }
    }

    fn check_builder_score(dir: &Path) {
        let s = score_dir(dir).unwrap();
        let ran = s
            .candidates
            .iter()
            .filter(|c| c.measured_s.is_some())
            .count();
        assert!(ran >= 20, "{ran} candidates ran");
        assert!(s.failed.is_empty(), "{:?}", s.failed);
        assert!(
            s.candidates[0]
                .description
                .contains("TETRAEDER_EXIT_VRAM_SLOT_BYTES")
        );
        assert!(s.regret_final <= 0.10, "regret {}", s.regret_final);
        assert!(s.spearman >= 0.8, "spearman {}", s.spearman);
        assert!(s.baselines.is_empty());
        assert!(!s.anchor_residuals.is_empty());
    }

    // hardware: none
    #[test]
    fn synthetic_builder_fixture_is_scored() {
        let dir = std::env::temp_dir().join(format!("tetraeder-builder-{}", std::process::id()));
        write_synthetic_builder(&dir);
        check_builder_score(&dir);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The committed synthetic directory scores like a measured one. Regenerate it with
    /// `cargo test -p tetraeder-planner regenerate_synthetic -- --ignored`.
    // hardware: none
    #[test]
    fn committed_synthetic_builder_fixture_is_scored() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/builder-synthetic");
        if !dir.join("candidates.json").is_file() {
            eprintln!("skip: no synthetic builder fixture committed");
            return;
        }
        check_builder_score(&dir);
    }

    #[test]
    #[ignore = "rewrites crates/planner/fixtures/builder-synthetic"]
    fn regenerate_synthetic_builder_fixture() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/builder-synthetic");
        std::fs::remove_dir_all(&dir).ok();
        write_synthetic_builder(&dir);
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
