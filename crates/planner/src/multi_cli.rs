//! `tetraeder-plan candidates --multi`, `calibrate` and `predict` (ADR 0068, 0070): the files of
//! a Phase 5 session. Pure over text so the tests need no files; the binary reads and writes them.

use crate::layout::ModelLayout;
use crate::multi::{
    MultiCandidate, Routing, Shape, candidate_for, enumerate_multi, features_multi,
    predict_candidate, topology_of,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use tetraeder_costmodel::MODEL_VERSION;
use tetraeder_costmodel::calib::{Anchor, fit};
use tetraeder_costmodel::predict::predict;
use tetraeder_plan::{Hints, Phase};
use tetraeder_profiler::{CollectiveCurve, DecodeClocks, DeviceKind, HardwareGraph};

/// The phase the multi-GPU commands plan for.
pub const PHASE: Phase = Phase(5);
/// The judged request (ADR 0068): 2048 prompt tokens, 128 generated.
pub const PROMPT: u32 = 2048;
pub const GEN: u32 = 128;

/// `candidates-<model>.json`: the enumeration with the pruned shapes and their reasons.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultiFile {
    pub model: String,
    pub gpus: usize,
    pub candidates: Vec<MultiCandidate>,
    pub pruned: Vec<Pruned>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pruned {
    pub id: String,
    pub reason: String,
}

/// One candidate's prediction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PredictRow {
    pub id: String,
    pub pp_tps: f64,
    pub tg_tps: f64,
    /// `prompt / pp_tps + gen / tg_tps`.
    pub t_s: f64,
    pub binding: String,
}

/// `predict-<model>-<box>.json`, written before the first candidate runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PredictFile {
    pub model: String,
    pub git_commit: String,
    pub hw_sha256: String,
    pub calib_sha256: Option<String>,
    pub comm_sha256: Option<String>,
    /// `routing-<model>.json`, when expert parallel was priced from recorded routing (ADR 0070).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing_sha256: Option<String>,
    pub prompt: u32,
    pub r#gen: u32,
    pub candidates: Vec<PredictRow>,
    pub pruned: Vec<Pruned>,
}

/// The fit and its residuals (`calib.json`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibFile {
    pub cost_model_version: u32,
    pub anchors: Vec<String>,
    pub band: f64,
    pub residuals: Vec<(String, f64)>,
    /// Anchors that did not enter the fit (the layer anchor of ADR 0082): `(id:tg, ln predicted -
    /// ln measured)` on the fitted graph, decode clocks included.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub held_out: Vec<(String, f64)>,
    pub hw: HardwareGraph,
    /// False when the anchors place no compute on the CPU and the CPU term was not fitted.
    #[serde(default = "default_true")]
    pub cpu_compute_used: bool,
}

fn default_true() -> bool {
    true
}

/// `routing-<model>.json` of `tetraeder bench --routing-out` (ADR 0070): per layer, how often the
/// router chose each expert over the judged token stream, on one GPU without a plan. `prefill`
/// counts the rows of the pp test, `decode` the tokens of the tg test.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutingFile {
    pub model: String,
    pub n_expert: u32,
    pub top_k: u32,
    pub prefill_tokens: u32,
    pub decode_tokens: u32,
    pub prefill: Vec<Vec<u64>>,
    pub decode: Vec<Vec<u64>>,
}

/// A routing file checked against the layout: one row per layer and one count per expert, in
/// both tables. A mismatch is an error, never a silent fall back to uniform routing.
pub fn parse_routing(text: &str, layout: &ModelLayout) -> Result<Routing, String> {
    let f: RoutingFile = serde_json::from_str(text).map_err(|e| format!("routing json: {e}"))?;
    if !layout.is_moe() {
        return Err(format!(
            "routing given for {}, which has no experts",
            layout.model
        ));
    }
    for (what, t) in [("prefill", &f.prefill), ("decode", &f.decode)] {
        if t.len() != layout.n_layer as usize {
            return Err(format!(
                "routing {what}: {} layers, the layout has {}",
                t.len(),
                layout.n_layer
            ));
        }
        if let Some((l, r)) = t
            .iter()
            .enumerate()
            .find(|(_, r)| r.len() != layout.n_expert as usize)
        {
            return Err(format!(
                "routing {what} layer {l}: {} experts, the layout has {}",
                r.len(),
                layout.n_expert
            ));
        }
    }
    Routing::from_counts(&f.prefill, &f.decode)
}

/// `calib-comm.json` of `tetraeder comm-probe` (only the curves are read).
#[derive(Debug, Clone, Deserialize)]
pub struct CommReport {
    #[serde(default)]
    pub comm_path: String,
    pub curves: Vec<CollectiveCurve>,
}

/// Puts the probe's curves into `hw`, replacing those of the same op and ranks. Measured
/// curves are never fitted (ADR 0070).
pub fn merge_comm(hw: &mut HardwareGraph, text: &str) -> Result<String, String> {
    let r: CommReport = serde_json::from_str(text).map_err(|e| format!("comm json: {e}"))?;
    for c in r.curves {
        hw.collectives
            .retain(|o| !(o.op == c.op && o.ranks == c.ranks));
        hw.collectives.push(c);
    }
    hw.validate()?;
    Ok(r.comm_path)
}

/// Duty threshold of ADR 0082.
pub const DUTY_THRESHOLD: f64 = 0.32;

/// `decode-clocks.json` of `scripts/decode_clocks.py`: the `DecodeClocks` shape plus the per-run
/// medians and the reasons for a withheld value, which are not read. A file without `low_mhz`
/// (the layer anchor was withheld) leaves the graph without clocks.
#[derive(Debug, Clone, Deserialize)]
struct DecodeClocksFile {
    low_mhz: Option<f64>,
    #[serde(default)]
    busy_mhz: BTreeMap<String, f64>,
    duty_threshold: Option<f64>,
}

/// Puts the clocks of `decode-clocks.json` into `hw` (ADR 0082). `Ok(false)`: the file withholds
/// `low_mhz`, nothing is set and every prediction stays as cost model 4.
pub fn merge_decode_clocks(hw: &mut HardwareGraph, text: &str) -> Result<bool, String> {
    let f: DecodeClocksFile =
        serde_json::from_str(text).map_err(|e| format!("decode clocks json: {e}"))?;
    let Some(low_mhz) = f.low_mhz else {
        return Ok(false);
    };
    let duty_threshold = f.duty_threshold.unwrap_or(DUTY_THRESHOLD);
    let ok = |x: f64| x > 0.0 && x.is_finite();
    if !ok(low_mhz) || !ok(duty_threshold) || !f.busy_mhz.values().all(|&x| ok(x)) {
        return Err("decode clocks: clocks and threshold must be positive".into());
    }
    hw.decode_clocks = Some(DecodeClocks {
        low_mhz,
        busy_mhz: f.busy_mhz,
        duty_threshold,
    });
    Ok(true)
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// A hardware graph from JSON (`hw.json`) or, when the text is not JSON, TOML.
pub fn parse_hw(text: &str) -> Result<HardwareGraph, String> {
    let t = text.trim_start();
    let g = if t.starts_with('{') {
        HardwareGraph::from_json(text).map_err(|e| format!("hardware json: {e}"))?
    } else {
        HardwareGraph::from_toml(text).map_err(|e| format!("hardware toml: {e}"))?
    };
    g.check_node_names()?;
    Ok(g)
}

pub fn parse_layout(text: &str) -> Result<ModelLayout, String> {
    serde_json::from_str(text).map_err(|e| format!("layout json: {e}"))
}

/// `candidates --multi`: the enumeration as a file.
pub fn candidates_multi(hw: &HardwareGraph, layout: &ModelLayout) -> MultiFile {
    let en = enumerate_multi(hw, layout, &Hints::default(), PHASE);
    MultiFile {
        model: layout.model.clone(),
        gpus: hw
            .devices
            .iter()
            .filter(|d| d.kind == DeviceKind::Gpu)
            .count(),
        candidates: en.candidates,
        pruned: en
            .pruned
            .into_iter()
            .map(|(id, reason)| Pruned { id, reason })
            .collect(),
    }
}

/// `candidates --multi --anchor <id>`: the plans of calibration anchors that are not in the box's
/// fixed set, such as `layer-4@L24` (ADR 0082), as a file of their own.
pub fn anchors_multi(
    hw: &HardwareGraph,
    layout: &ModelLayout,
    ids: &[String],
) -> Result<MultiFile, String> {
    let candidates = ids
        .iter()
        .map(|id| candidate_for(hw, layout, &Hints::default(), PHASE, id))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(MultiFile {
        model: layout.model.clone(),
        gpus: hw
            .devices
            .iter()
            .filter(|d| d.kind == DeviceKind::Gpu)
            .count(),
        candidates,
        pruned: Vec::new(),
    })
}

/// `predict`: every candidate of the file on `calibrated` (the fit's graph, or the measured
/// graph itself before a calibration), expert parallel priced from `routing` when given (its
/// table and the file's sha256). The hashes name the inputs.
pub fn predict_multi(
    calibrated: &HardwareGraph,
    layout: &ModelLayout,
    cands: &MultiFile,
    hashes: (String, Option<String>, Option<String>),
    routing: Option<(&Routing, String)>,
    git_commit: &str,
) -> Result<PredictFile, String> {
    let mut rows = Vec::new();
    let table = routing.as_ref().map(|r| r.0);
    for c in &cands.candidates {
        let p = predict_candidate(calibrated, layout, &c.plan, PHASE, (PROMPT, GEN), table)
            .map_err(|e| format!("{}: {e}", c.id))?;
        rows.push(PredictRow {
            id: c.id.clone(),
            pp_tps: p.pp_tok_s,
            tg_tps: p.tg_tok_s,
            t_s: p.request_s,
            binding: p.binding,
        });
    }
    Ok(PredictFile {
        model: cands.model.clone(),
        git_commit: git_commit.to_string(),
        hw_sha256: hashes.0,
        calib_sha256: hashes.1,
        comm_sha256: hashes.2,
        routing_sha256: routing.map(|r| r.1),
        prompt: PROMPT,
        r#gen: GEN,
        candidates: rows,
        pruned: cands.pruned.clone(),
    })
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

/// The `<model>` of `anchors-<model>-r<k>.json`.
pub fn anchor_model(file_name: &str) -> Option<String> {
    let stem = file_name.strip_prefix("anchors-")?.strip_suffix(".json")?;
    let (model, round) = stem.rsplit_once("-r")?;
    round.parse::<u32>().ok()?;
    Some(model.to_string())
}

/// Rates of one anchor run: `(model, plan_id, layer_limit)` to pp and tg samples.
#[derive(Default)]
struct Samples {
    pp: Vec<f64>,
    tg: Vec<f64>,
    prompt: Option<u32>,
    r#gen: Option<u32>,
}

/// Collects the llama-bench rows (`n_prompt`, `n_gen`, `avg_ts`) with `plan_id` and
/// `layer_limit` of the anchor files of one model.
fn collect(files: &[&str]) -> Result<BTreeMap<(String, Option<u32>), Samples>, String> {
    let mut out: BTreeMap<(String, Option<u32>), Samples> = BTreeMap::new();
    for text in files {
        let v: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("anchor json: {e}"))?;
        let rows = match &v {
            serde_json::Value::Array(a) => a.clone(),
            other => other
                .get("rows")
                .or_else(|| other.get("results"))
                .and_then(|r| r.as_array().cloned())
                .ok_or("anchor file is neither an array nor has rows")?,
        };
        for r in rows {
            let Some(id) = r.get("plan_id").and_then(|x| x.as_str()) else {
                continue;
            };
            let limit = r
                .get("layer_limit")
                .and_then(|x| x.as_u64())
                .map(|x| x as u32);
            let num = |k: &str| r.get(k).and_then(|x| x.as_u64()).unwrap_or(0) as u32;
            let (np, ng) = (num("n_prompt"), num("n_gen"));
            let Some(ts) = r.get("avg_ts").and_then(|x| x.as_f64()) else {
                continue;
            };
            let s = out.entry((id.to_string(), limit)).or_default();
            if np > 0 && ng == 0 {
                s.pp.push(ts);
                s.prompt = Some(np);
            } else if ng > 0 && np == 0 {
                s.tg.push(ts);
                s.r#gen = Some(ng);
            }
        }
    }
    Ok(out)
}

/// `calibrate`: the anchors of every model (`(layout, anchor file texts)`) become features with
/// [`features_multi`] and fit the parameters of `calib.rs`, `c_range` and `c_step` included.
/// No routing: the anchors (`single`, `tp-2`, `tp-4`) cut experts by columns, whose cost does
/// not depend on which experts a token chooses.
///
/// The fit runs on `hw` without its `decode_clocks` (ADR 0082) and the fitted graph carries them
/// again. A layer anchor `layer-<g>@L<n>` never enters the fit: its residual on the fitted graph
/// goes to `held_out`.
pub fn calibrate(
    hw: &HardwareGraph,
    models: &[(ModelLayout, Vec<String>)],
) -> Result<CalibFile, String> {
    let mut anchors: Vec<Anchor> = Vec::new();
    let mut held: Vec<Anchor> = Vec::new();
    for (layout, files) in models {
        let texts: Vec<&str> = files.iter().map(String::as_str).collect();
        for ((plan_id, limit), s) in collect(&texts)? {
            let cand = candidate_for(hw, layout, &Hints::default(), PHASE, &plan_id)
                .map_err(|e| format!("{}: {plan_id}: {e}", layout.model))?;
            let topo = topology_of(hw, layout, &cand.plan, PHASE)?;
            let f = features_multi(layout, &topo, &cand.plan.knobs, limit, None)?;
            let a = Anchor {
                id: format!(
                    "{}/{plan_id}/{}",
                    layout.model,
                    limit.map_or("all".to_string(), |l| format!("L{l}"))
                ),
                features: f,
                prompt: s.prompt.unwrap_or(PROMPT),
                r#gen: s.r#gen.unwrap_or(GEN),
                pp_tok_s: median(s.pp),
                tg_tok_s: median(s.tg),
            };
            if matches!(Shape::parse(&plan_id), Some(Shape::LayerAnchor { .. })) {
                held.push(a);
            } else {
                anchors.push(a);
            }
        }
    }
    let gpu = hw
        .devices
        .iter()
        .find(|d| d.kind == DeviceKind::Gpu)
        .ok_or("no GPU in the hardware graph")?;
    let cpu = hw
        .devices
        .iter()
        .find(|d| d.kind == DeviceKind::Cpu)
        .ok_or("no CPU in the hardware graph")?;
    let link = hw
        .link_between(&gpu.id, &cpu.id)
        .ok_or("no link between the first GPU and the CPU")?;
    let mut base = hw.clone();
    base.decode_clocks = None;
    let mut f = fit(&base, &anchors, &gpu.id, &cpu.id, &link.id).map_err(|e| e.to_string())?;
    f.hw.decode_clocks = hw.decode_clocks.clone();
    let mut held_out = Vec::new();
    for a in &held {
        let p =
            predict(&f.hw, &a.features, a.prompt, a.r#gen).map_err(|e| format!("{}: {e}", a.id))?;
        if let Some(m) = a.tg_tok_s {
            held_out.push((format!("{}:tg", a.id), (p.tg_tok_s / m).ln()));
        }
    }
    for (id, r) in &held_out {
        eprintln!(
            "held out {id}: {:+.3} (predicted / measured - 1)",
            r.exp() - 1.0
        );
    }
    if !f.cpu_compute_used {
        eprintln!("cpu compute: not used (no anchor places compute on the CPU)");
    }
    Ok(CalibFile {
        cost_model_version: MODEL_VERSION,
        anchors: anchors.iter().map(|a| a.id.clone()).collect(),
        band: f.band,
        residuals: f.residuals,
        held_out,
        hw: f.hw,
        cpu_compute_used: f.cpu_compute_used,
    })
}
