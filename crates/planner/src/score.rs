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
use tetraeder_costmodel::predict::{DevicePart, ReadPath, RunFeatures, TierFeatures, predict};
use tetraeder_profiler::{HardwareGraph, QueuePoint};

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
        tier: None,
    }
}

/// Features of a builder run (ADR 0035): the dense weights are on the GPU; of the active expert
/// bytes of a token the share `hit` runs on the GPU from VRAM slots and the rest, the misses, on
/// the CPU. A prefill touches nearly all experts, so the resident bytes (what op offload moves
/// over PCIe) split by the slot share `vram_slots / n_expert`. This form has no RAM tier: it
/// decides disk reads, which [`builder_features_tier`] adds from a routing replay.
pub fn builder_features(
    layout: &ModelLayout,
    vram_slots: u32,
    hit: f64,
    knobs: (u32, u32, bool),
) -> RunFeatures {
    builder_features_with(layout, vram_slots, hit, hit, knobs, None)
}

/// [`builder_features`] from a routing replay (ADR 0056): the GPU share of the active expert
/// bytes is the replay's decode `vram_share`, the switches follow its `full_hit_share`, and the
/// disk reads of the RAM tier become the run's [`TierFeatures`]. `path` and `(depth, sync_depth)`
/// describe how the run reads (see [`read_path`]); `disk` is the device id of the disk.
pub fn builder_features_tier(
    layout: &ModelLayout,
    vram_slots: u32,
    tier: &TierOutcome,
    knobs: (u32, u32, bool),
    path: ReadPath,
    (depth, sync_depth): (u32, u32),
    disk: &str,
) -> RunFeatures {
    let t = TierFeatures {
        disk: disk.to_string(),
        path,
        depth,
        sync_depth,
        decode_sync_bytes_per_token: tier.decode.sync_miss_bytes_per_token,
        decode_prefetch_bytes_per_token: tier.decode.prefetch_bytes_per_token,
        prefill_bytes_per_token: tier.prefill.disk_bytes_per_token,
    };
    builder_features_with(
        layout,
        vram_slots,
        tier.decode.vram_share,
        tier.decode.full_hit(),
        knobs,
        Some(t),
    )
}

fn builder_features_with(
    layout: &ModelLayout,
    vram_slots: u32,
    vram_share: f64,
    full_hit: f64,
    (threads, ubatch, op_offload): (u32, u32, bool),
    tier: Option<TierFeatures>,
) -> RunFeatures {
    let bpp = layout.main_format().bits_per_weight() / 8.0;
    let n_expert = f64::from(layout.n_expert.max(1));
    let r = f64::from(layout.n_expert_used) / n_expert;
    let f = (f64::from(vram_slots) / n_expert).clamp(0.0, 1.0);
    let hit = vram_share.clamp(0.0, 1.0);
    let full_hit = full_hit.clamp(0.0, 1.0);
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
        switches_per_token: 1.0 + 2.0 * f64::from(layout.n_layer) * (1.0 - full_hit),
        activation_bytes_per_token: 4.0 * f64::from(layout.n_embd),
        threads,
        ubatch,
        op_offload,
        graphs: true,
        tier,
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

/// One phase (prefill or decode) of a routing replay outcome (`tetraeder-bench replay`,
/// ADR 0050 c). Only the fields the planner reads; unknown fields are ignored, absent ones are 0.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TierPhase {
    /// Expert reads served from VRAM over all reads.
    pub vram_share: f64,
    /// Layer steps with every routed expert in VRAM; absent in files from before ADR 0056, where
    /// it stands at `vram_share`.
    pub full_hit_share: Option<f64>,
    /// Bytes per token the compute waited for (urgent misses).
    pub sync_miss_bytes_per_token: f64,
    pub prefetch_bytes_per_token: f64,
    pub disk_bytes_per_token: f64,
    /// The engine's mean over layers of the RAM hit rate, what `exit_regime` measures as
    /// `decode_layers.ram_hit_rate_mean`.
    pub ram_hit_rate_engine_mean: f64,
}

impl TierPhase {
    pub fn full_hit(&self) -> f64 {
        self.full_hit_share.unwrap_or(self.vram_share)
    }
}

/// One outcome of `tier-replay.json`: the byte counts of a VRAM and RAM slot count.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TierOutcome {
    pub vram_slots: u32,
    pub ram_slots: u32,
    pub policy: String,
    pub prefetch: bool,
    pub decode: TierPhase,
    pub prefill: TierPhase,
}

/// Reads the outcomes of a `tetraeder-bench replay` file.
pub fn read_tier_replay(path: &Path) -> Result<Vec<TierOutcome>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let v: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let outcomes = v
        .get("outcomes")
        .cloned()
        .ok_or_else(|| format!("{}: no outcomes", path.display()))?;
    serde_json::from_value(outcomes).map_err(|e| format!("{}: {e}", path.display()))
}

/// Disk id of the builder's expert files in the planner's hardware graphs.
pub const BUILDER_DISK: &str = "nvme0";
/// The reader's direct-I/O queue depth (`READER_QUEUE_DEPTH`), also the depth of the synchronous
/// prefill reads.
pub(crate) const READER_DEPTH: u32 = 16;
/// The service's default workers and chunk size (ADR 0054).
const SERVICE_WORKERS: u32 = 16;
const SERVICE_CHUNK: f64 = 1_048_576.0;

/// How a builder variant reads experts and at which queue depth, from its `exit_regime`
/// environment (ADR 0056): no service option means the reader; the service with
/// `TETRAEDER_EXIT_OVERLAP=1` is `ServiceOverlap`. The service's depth is
/// `min(workers, ceil(read bytes per layer step / chunk))`.
pub fn read_path(
    env: &BTreeMap<String, String>,
    layout: &ModelLayout,
    tier: &TierOutcome,
) -> (ReadPath, u32) {
    let flag = |k: &str| env.get(k).is_some_and(|v| v == "1");
    if !flag("TETRAEDER_EXIT_IO_SERVICE") {
        return (ReadPath::Reader, READER_DEPTH);
    }
    let path = if flag("TETRAEDER_EXIT_OVERLAP") {
        ReadPath::ServiceOverlap
    } else {
        ReadPath::Service
    };
    let workers = env
        .get("TETRAEDER_EXIT_IO_WORKERS")
        .and_then(|v| v.parse().ok())
        .unwrap_or(SERVICE_WORKERS);
    let per_step = (tier.decode.sync_miss_bytes_per_token + tier.decode.prefetch_bytes_per_token)
        / f64::from(layout.n_layer.max(1));
    let depth = (per_step / SERVICE_CHUNK)
        .ceil()
        .clamp(1.0, f64::from(workers.max(1)));
    (path, depth as u32)
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
    /// Median decode `ram_hit_rate_mean` (the fidelity gate of the routing replay).
    pub ram_hit_rate: Option<f64>,
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
        ram_hit_rate: col(&|r| r["decode_layers"]["ram_hit_rate_mean"].as_f64()),
    }
}

/// The `exit_regime` JSON files of a builder fixture directory, `<variant id>.r<k>.json` (the
/// campaign script's output names), by variant id.
pub(crate) fn read_builder_files(dir: &Path) -> Result<BTreeMap<String, Vec<Value>>, String> {
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
    Ok(by_id)
}

/// Builder results of a fixture directory, per variant id.
pub fn read_builder_runs(dir: &Path) -> Result<BTreeMap<String, BuilderRun>, String> {
    Ok(read_builder_files(dir)?
        .into_iter()
        .map(|(id, files)| (id, builder_run(&files)))
        .collect())
}

/// The harness results (`planner-*.json`) of a llama-bench fixture directory, sorted by name.
pub(crate) fn read_harness_files(dir: &Path) -> Result<Vec<Value>, String> {
    let mut paths = Vec::new();
    for e in std::fs::read_dir(dir).map_err(|e| e.to_string())? {
        let p = e.map_err(|e| e.to_string())?.path();
        let name = p.file_name().unwrap_or_default().to_string_lossy();
        if name.starts_with("planner-") && name.ends_with(".json") {
            paths.push(p);
        }
    }
    paths.sort();
    let mut files = Vec::new();
    for p in paths {
        let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        // a run killed before its first checkpoint leaves its claimed file empty
        if !text.trim().is_empty() {
            files.push(serde_json::from_str(&text).map_err(|e| e.to_string())?);
        }
    }
    Ok(files)
}

/// Median pp and tg tokens per second per variant over every sample in the harness results.
pub fn measured(results: &Value) -> BTreeMap<String, (Option<f64>, Option<f64>)> {
    measured_all(std::slice::from_ref(results))
}

/// [`measured`] over the samples of several results files pooled (a resumed run writes its
/// later repeats to a file of its own).
pub fn measured_all(files: &[Value]) -> BTreeMap<String, (Option<f64>, Option<f64>)> {
    let mut acc: BTreeMap<String, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    for pair in files
        .iter()
        .flat_map(|r| r["pairs"].as_array().into_iter().flatten())
    {
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
    /// Builder fixtures: where the RAM tier came from, `replay` (routing replay bytes),
    /// `hitmodel` (no replay file) or `hitmodel (fidelity gate failed: <anchor>, <delta>)`.
    /// Empty for fixtures without a builder.
    #[serde(default)]
    pub tier_source: String,
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

pub(crate) fn request_s(v: &CampaignVariant, pp: Option<f64>, tg: Option<f64>) -> Option<f64> {
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
        let replay_file = dir.join("tier-replay.json");
        let replay = if replay_file.is_file() {
            Some(read_tier_replay(&replay_file)?)
        } else {
            None
        };
        // the box stage `disk-curve` measures the disk under load: the session's own nvme0
        let curve_file = dir.join("disk-curve-loaded.json");
        let disk = if curve_file.is_file() {
            Some(read_disk_curve(&curve_file)?)
        } else {
            None
        };
        return score_builder_curve(
            &fixture,
            &layout,
            &cands,
            &runs,
            cands.budget.threads,
            replay.as_deref(),
            disk.as_ref(),
        );
    }
    let files = read_harness_files(dir)?;
    let results = measured_all(&files);
    score(&fixture, &layout, &cands, &results, cands.budget.threads)
}

/// Scores a builder fixture without a routing replay: the hit-rate model comes from the
/// anchors' measured hit rates ([`HitModel`]), the features from [`builder_features`], then as
/// [`score`].
pub fn score_builder(
    fixture: &str,
    layout: &ModelLayout,
    cands: &CandidatesFile,
    runs: &BTreeMap<String, BuilderRun>,
    quota: u32,
) -> Result<Score, String> {
    score_builder_tier(fixture, layout, cands, runs, quota, None)
}

fn variant_slots(v: &CampaignVariant) -> Option<(u32, u32)> {
    match v.placement {
        Some(Placement::Builder {
            vram_slots,
            ram_slots,
        }) => Some((vram_slots, ram_slots)),
        _ => None,
    }
}

/// Whether the variant runs with the expert prefetch on (the default).
fn prefetch_on(v: &CampaignVariant) -> bool {
    v.env
        .get("TETRAEDER_EXIT_PREFETCH")
        .is_none_or(|x| x != "0")
}

/// The LRU outcome of a variant's slot counts and prefetch setting.
pub fn outcome_of<'a>(outs: &'a [TierOutcome], v: &CampaignVariant) -> Option<&'a TierOutcome> {
    let (vram, ram) = variant_slots(v)?;
    let pf = prefetch_on(v);
    outs.iter().find(|o| {
        o.vram_slots == vram && o.ram_slots == ram && o.policy == "lru" && o.prefetch == pf
    })
}

/// ADR 0056 fidelity gate: every variant has an outcome and each anchor's replayed RAM hit rate
/// is within 0.5 points of the measured one. `Err` names the first miss.
fn fidelity(
    outs: &[TierOutcome],
    cands: &CandidatesFile,
    runs: &BTreeMap<String, BuilderRun>,
) -> Result<(), String> {
    let all = cands
        .anchors
        .iter()
        .chain(cands.candidates.iter().map(|e| &e.variant));
    for v in all {
        if outcome_of(outs, v).is_none() {
            return Err(format!("no outcome for {}, {:?}", v.id, variant_slots(v)));
        }
    }
    let mut checked = 0;
    for v in &cands.anchors {
        let (Some(o), Some(m)) = (
            outcome_of(outs, v),
            runs.get(&v.id).and_then(|r| r.ram_hit_rate),
        ) else {
            continue;
        };
        let delta = o.decode.ram_hit_rate_engine_mean - m;
        if delta.abs() > 0.005 {
            return Err(format!("{}, {delta:+.4}", v.id));
        }
        checked += 1;
    }
    if checked == 0 {
        return Err("no anchor measured a ram hit rate".into());
    }
    Ok(())
}

/// [`score_builder`] with a routing replay (`tier-replay.json`, ADR 0056): when the replay
/// passes the fidelity gate on the anchors, every variant's VRAM share and disk bytes come from
/// its outcome ([`builder_features_tier`]) and `Score::tier_source` is `replay`. Otherwise, and
/// without a replay, the [`HitModel`] path scores it and `tier_source` says why. Never an
/// error for a bad replay.
pub fn score_builder_tier(
    fixture: &str,
    layout: &ModelLayout,
    cands: &CandidatesFile,
    runs: &BTreeMap<String, BuilderRun>,
    quota: u32,
    replay: Option<&[TierOutcome]>,
) -> Result<Score, String> {
    score_builder_curve(fixture, layout, cands, runs, quota, replay, None)
}

/// A disk's measured read rate by queue depth (`tetraeder-bench disk-curve`): the session's own
/// `nvme0` in place of the prior's (ADR 0056).
#[derive(Debug, Clone, PartialEq)]
pub struct DiskCurve {
    pub curve: Vec<QueuePoint>,
    pub chunk_bytes: Option<u64>,
}

/// Reads the `queue_curve` and `queue_chunk_bytes` of a `disk-curve` output.
pub fn read_disk_curve(path: &Path) -> Result<DiskCurve, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let v: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let curve: Vec<QueuePoint> = serde_json::from_value(v["queue_curve"].clone())
        .map_err(|e| format!("{}: queue_curve: {e}", path.display()))?;
    if curve.is_empty() {
        return Err(format!("{}: empty queue_curve", path.display()));
    }
    Ok(DiskCurve {
        curve,
        chunk_bytes: v["queue_chunk_bytes"].as_u64(),
    })
}

/// [`score_builder_tier`] with the session's measured disk curve in place of the prior's.
pub fn score_builder_curve(
    fixture: &str,
    layout: &ModelLayout,
    cands: &CandidatesFile,
    runs: &BTreeMap<String, BuilderRun>,
    quota: u32,
    replay: Option<&[TierOutcome]>,
    disk: Option<&DiskCurve>,
) -> Result<Score, String> {
    let n = f64::from(layout.n_expert.max(1));
    let points: Vec<(f64, f64)> = cands
        .anchors
        .iter()
        .filter_map(|v| {
            Some((
                f64::from(variant_slots(v)?.0) / n,
                runs.get(&v.id)?.hit_rate?,
            ))
        })
        .collect();
    let model = HitModel::from_points(&points);
    let (outs, source) = match replay.map(|o| (o, fidelity(o, cands, runs))) {
        None => (None, "hitmodel".to_string()),
        Some((o, Ok(()))) => (Some(o), "replay".to_string()),
        Some((_, Err(why))) => (None, format!("hitmodel (fidelity gate failed: {why})")),
    };
    if outs.is_none() && model.is_none() {
        return Err(format!("{fixture}: no anchor measured a hit rate"));
    }
    let feat = |v: &CampaignVariant| {
        let s = variant_slots(v).map_or(0, |x| x.0);
        let env = |k: &str| v.env.get(k).map(String::as_str);
        let ub = env("TETRAEDER_EXIT_UBATCH")
            .and_then(|x| x.parse().ok())
            .unwrap_or(256);
        let op = env("TETRAEDER_EXIT_OP_OFFLOAD").is_none_or(|x| x == "1");
        let knobs = (v.threads, ub, op);
        match outs.and_then(|o| outcome_of(o, v)) {
            Some(o) => {
                let (path, depth) = read_path(&v.env, layout, o);
                builder_features_tier(
                    layout,
                    s,
                    o,
                    knobs,
                    path,
                    (depth, READER_DEPTH),
                    BUILDER_DISK,
                )
            }
            None => {
                let hit = model.map_or(0.0, |m| m.at(f64::from(s) / n));
                builder_features(layout, s, hit, knobs)
            }
        }
    };
    let measured: BTreeMap<String, (Option<f64>, Option<f64>)> = runs
        .iter()
        .map(|(k, r)| (k.clone(), (r.pp_tok_s, r.tg_tok_s)))
        .collect();
    score_with(fixture, cands, &measured, quota, &feat, &source, disk)
}

/// `(scored, dropped)` of [`predict_builder`]: request seconds per candidate id, and the
/// candidates that could not be predicted with the reason.
pub type BuilderPredictions = (Vec<(String, f64)>, Vec<(String, String)>);

/// What the planner predicts for every candidate of a builder enumeration on a hardware graph,
/// without measurements: the request time per candidate id, and the candidates that could not be
/// predicted (a disk or device the graph lacks) with the reason. `outs` are routing replay
/// outcomes, looked up by [`outcome_of`]; a candidate without one is dropped.
pub fn predict_builder(
    hw: &HardwareGraph,
    layout: &ModelLayout,
    cands: &CandidatesFile,
    outs: &[TierOutcome],
) -> BuilderPredictions {
    let (mut ok, mut dropped) = (Vec::new(), Vec::new());
    for e in &cands.candidates {
        let v = &e.variant;
        let Some(o) = outcome_of(outs, v) else {
            dropped.push((e.candidate.id.clone(), "no replay outcome".into()));
            continue;
        };
        let env = |k: &str| v.env.get(k).map(String::as_str);
        let ub = env("TETRAEDER_EXIT_UBATCH")
            .and_then(|x| x.parse().ok())
            .unwrap_or(256);
        let op = env("TETRAEDER_EXIT_OP_OFFLOAD").is_none_or(|x| x == "1");
        let (path, depth) = read_path(&v.env, layout, o);
        let slots = variant_slots(v).map_or(0, |x| x.0);
        let f = builder_features_tier(
            layout,
            slots,
            o,
            (v.threads, ub, op),
            path,
            (depth, READER_DEPTH),
            BUILDER_DISK,
        );
        match predict(hw, &f, v.prompt, v.r#gen) {
            Ok(p) => ok.push((e.candidate.id.clone(), p.request_s)),
            Err(err) => dropped.push((e.candidate.id.clone(), err.to_string())),
        }
    }
    (ok, dropped)
}

/// The virtual-clock time model of one builder candidate (ADR 0056): its features from the
/// routing replay, its prediction on `hw` (a calibrated graph), then
/// [`crate::timemodel::time_model`]. The service's workers and chunk come from the variant's
/// environment and the disk, else the service defaults; prefetch depth 2.
pub fn builder_time_model(
    hw: &HardwareGraph,
    layout: &ModelLayout,
    cands: &CandidatesFile,
    outs: &[TierOutcome],
    id: &str,
) -> Result<crate::timemodel::TimeModel, String> {
    let v = match cands.candidates.iter().find(|e| e.candidate.id == id) {
        Some(e) => &e.variant,
        None => cands
            .anchors
            .iter()
            .find(|a| a.id == id)
            .ok_or_else(|| format!("no candidate or anchor {id}"))?,
    };
    let o = outcome_of(outs, v).ok_or_else(|| format!("{id}: no replay outcome"))?;
    let env = |k: &str| v.env.get(k).map(String::as_str);
    let ub = env("TETRAEDER_EXIT_UBATCH")
        .and_then(|x| x.parse().ok())
        .unwrap_or(256);
    let op = env("TETRAEDER_EXIT_OP_OFFLOAD").is_none_or(|x| x == "1");
    let (path, depth) = read_path(&v.env, layout, o);
    let slots = variant_slots(v).map_or(0, |x| x.0);
    let f = builder_features_tier(
        layout,
        slots,
        o,
        (v.threads, ub, op),
        path,
        (depth, READER_DEPTH),
        BUILDER_DISK,
    );
    let workers = env("TETRAEDER_EXIT_IO_WORKERS")
        .and_then(|x| x.parse().ok())
        .unwrap_or(SERVICE_WORKERS);
    let chunk_bytes = hw
        .device(BUILDER_DISK)
        .and_then(|d| d.queue_chunk_bytes)
        .unwrap_or(SERVICE_CHUNK as u64);
    let cpu_pairs = f64::from(layout.n_expert_used)
        * f64::from(layout.n_layer)
        * (1.0 - o.decode.vram_share).max(0.0);
    crate::timemodel::time_model(
        hw,
        &f,
        layout.n_layer,
        cpu_pairs,
        v.prompt,
        v.r#gen,
        crate::timemodel::IoShape {
            workers,
            chunk_bytes,
            prefetch_depth: 2,
        },
    )
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
    score_with(fixture, cands, measured, quota, &feat, "", None)
}

fn score_with(
    fixture: &str,
    cands: &CandidatesFile,
    measured: &BTreeMap<String, (Option<f64>, Option<f64>)>,
    quota: u32,
    feat: &dyn Fn(&CampaignVariant) -> RunFeatures,
    tier_source: &str,
    disk: Option<&DiskCurve>,
) -> Result<Score, String> {
    let mut prior = HardwareGraph::from_toml(RTX3090_PRIOR).map_err(|e| e.to_string())?;
    if let Some(c) = prior.devices.iter_mut().find(|d| d.id == "cpu0") {
        c.threads = Some(quota);
    }
    if let (Some(d), Some(n)) = (
        disk,
        prior.devices.iter_mut().find(|d| d.id == BUILDER_DISK),
    ) {
        n.queue_curve = d.curve.clone();
        n.queue_chunk_bytes = d.chunk_bytes;
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
        tier_source: tier_source.to_string(),
        candidates: scores,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::fixtures;

    // hardware: none
    #[test]
    fn samples_pool_over_results_files() {
        let file = |samples: &[(&str, f64)]| {
            let s: Vec<Value> = samples
                .iter()
                .map(|(v, x)| serde_json::json!({"variant": v, "pp_ts": x, "tg_ts": x}))
                .collect();
            serde_json::json!({"pairs": [{"samples": s}]})
        };
        // a resumed run writes a second file with the later repeats of the same variants
        let files = [
            file(&[("a", 1.0), ("b", 10.0)]),
            file(&[("a", 3.0), ("a", 5.0)]),
        ];
        let m = measured_all(&files);
        assert_eq!(m["a"], (Some(3.0), Some(3.0)));
        assert_eq!(m["b"], (Some(10.0), Some(10.0)));
        assert_eq!(measured(&files[1])["a"], (Some(4.0), Some(4.0)));
    }

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

    // The scorer's numbers on the synthetic fixture before ADR 0056 (no tier file, no disk).
    const OLD_REGRET_ANALYTIC: f64 = 0.0;
    const OLD_REGRET_FINAL: f64 = 0.0;
    const OLD_SPEARMAN: f64 = 0.998_477_101_259_278;

    fn truth_hit(f: f64) -> f64 {
        0.35 + 0.6 * f
    }

    /// What the routing replay of the synthetic box says for a slot pair: the VRAM share is
    /// `truth_hit`, and the disk bytes per token fall linearly to 0 as the RAM tier reaches all
    /// experts outside VRAM. The RAM hit rate is the share of those experts the tier holds.
    fn truth_outcome(l: &ModelLayout, vram_slots: u32, ram_slots: u32) -> TierOutcome {
        let n = f64::from(l.n_expert);
        let share = truth_hit(f64::from(vram_slots) / n);
        let outside = f64::from(l.n_expert.saturating_sub(vram_slots).max(1));
        let held = (f64::from(ram_slots) / outside).clamp(0.0, 1.0);
        let experts: f64 = l.layers.iter().map(|b| b.expert_bytes as f64).sum();
        let active = experts * f64::from(l.n_expert_used) / n;
        let disk = (1.0 - share) * active * (1.0 - held);
        let phase = |sync: f64, prefetch: f64, per_token: f64| TierPhase {
            vram_share: share,
            full_hit_share: Some(share),
            sync_miss_bytes_per_token: sync,
            prefetch_bytes_per_token: prefetch,
            disk_bytes_per_token: per_token,
            ram_hit_rate_engine_mean: held,
        };
        // a prompt of 2048 tokens reads every missing expert about once
        let prefill = (1.0 - share) * experts * (1.0 - held) / 2048.0;
        TierOutcome {
            vram_slots,
            ram_slots,
            policy: "lru".into(),
            prefetch: true,
            decode: phase(0.6 * disk, 0.4 * disk, disk),
            prefill: phase(prefill, 0.0, prefill),
        }
    }

    /// Outcomes for every variant of a candidates file.
    fn truth_replay(l: &ModelLayout, file: &CandidatesFile) -> Vec<TierOutcome> {
        let mut seen = std::collections::BTreeSet::new();
        let mut out = Vec::new();
        for v in file
            .anchors
            .iter()
            .chain(file.candidates.iter().map(|e| &e.variant))
        {
            if let Some(Placement::Builder {
                vram_slots,
                ram_slots,
            }) = v.placement
                && seen.insert((vram_slots, ram_slots))
            {
                out.push(truth_outcome(l, vram_slots, ram_slots));
            }
        }
        out
    }

    fn synthetic_enumeration() -> (ModelLayout, crate::candidate::Budget, CandidatesFile) {
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
        (l, b, file)
    }

    /// Writes `layout.json`, `candidates.json` and `exit_regime`-shaped results (two repeats
    /// per variant, named like the campaign script's) of a synthetic builder box: the RTX 3090
    /// prior is the truth, the hit rate is linear in the slot share and the noise is 2%. With
    /// `tier` the box also reads from disk (the prior curve) as `truth_outcome` says and the
    /// directory gets the matching `tier-replay.json`.
    fn write_synthetic_builder(dir: &Path, tier: bool) {
        let (l, b, file) = synthetic_enumeration();
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
            let Some(Placement::Builder {
                vram_slots,
                ram_slots,
            }) = v.placement
            else {
                panic!("{}: {:?}", v.id, v.placement)
            };
            let hit = truth_hit(f64::from(vram_slots) / n);
            let env = |k: &str| v.env[&format!("TETRAEDER_EXIT_{k}")].clone();
            let knobs = (
                v.threads,
                env("UBATCH").parse().unwrap(),
                env("OP_OFFLOAD") == "1",
            );
            let o = truth_outcome(&l, vram_slots, ram_slots);
            let feats = if tier {
                let (path, depth) = read_path(&v.env, &l, &o);
                builder_features_tier(
                    &l,
                    vram_slots,
                    &o,
                    knobs,
                    path,
                    (depth, READER_DEPTH),
                    BUILDER_DISK,
                )
            } else {
                builder_features(&l, vram_slots, hit, knobs)
            };
            let p = predict(&truth, &feats, v.prompt, v.r#gen).unwrap();
            for rep in 1..=2 {
                let noise = 1.0 + 0.02 * ((i * 7 + rep * 3) as f64).sin();
                let run = serde_json::json!({"runs": [{
                    "prefill_tok_s": p.pp_tok_s * noise,
                    "decode_tok_s": p.tg_tok_s * noise,
                    "decode_layers": {
                        "full_hit_rate_mean": hit,
                        "ram_hit_rate_mean": o.decode.ram_hit_rate_engine_mean,
                    },
                }]});
                put(&format!("{}.r{rep}.json", v.id), run.to_string());
            }
        }
        if tier {
            let outcomes = truth_replay(&l, &file);
            put(
                "tier-replay.json",
                serde_json::to_string_pretty(&serde_json::json!({"outcomes": outcomes})).unwrap(),
            );
        }
    }

    fn check_builder_score(dir: &Path) {
        let s = score_dir(dir).unwrap();
        let has_file = dir.join("tier-replay.json").is_file();
        assert_eq!(s.tier_source, if has_file { "replay" } else { "hitmodel" });
        eprintln!(
            "{}: regret analytic {} final {}, spearman {}, tier {}",
            dir.display(),
            s.regret_analytic,
            s.regret_final,
            s.spearman,
            s.tier_source
        );
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
        write_synthetic_builder(&dir, true);
        check_builder_score(&dir);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Without the tier file (and a box that reads no disk) the scorer gives the numbers it gave
    /// before ADR 0056.
    // hardware: none
    #[test]
    fn synthetic_fixture_without_the_tier_file_reproduces_todays_numbers() {
        let dir = std::env::temp_dir().join(format!("tetraeder-builder-nt-{}", std::process::id()));
        write_synthetic_builder(&dir, false);
        check_builder_score(&dir);
        let s = score_dir(&dir).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        let near = |a: f64, b: f64| assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        near(s.regret_analytic, OLD_REGRET_ANALYTIC);
        near(s.regret_final, OLD_REGRET_FINAL);
        near(s.spearman, OLD_SPEARMAN);
    }

    /// Groups of candidates with equal predicted times (more than one member).
    fn tie_groups(s: &Score) -> usize {
        let mut p: Vec<f64> = s.candidates.iter().map(|c| c.predicted_s).collect();
        p.sort_by(f64::total_cmp);
        let (mut groups, mut run) = (0, 1);
        for w in p.windows(2) {
            if (w[1] - w[0]).abs() <= 1e-12 * w[0] {
                run += 1;
            } else {
                groups += usize::from(run > 1);
                run = 1;
            }
        }
        groups + usize::from(run > 1)
    }

    /// The replay prices the RAM tier: candidates that differed only in `ram_slots` no longer tie.
    // hardware: none
    #[test]
    fn ram_options_no_longer_tie_with_the_replay() {
        let dir =
            std::env::temp_dir().join(format!("tetraeder-builder-tie-{}", std::process::id()));
        write_synthetic_builder(&dir, true);
        let with = score_dir(&dir).unwrap();
        std::fs::remove_file(dir.join("tier-replay.json")).unwrap();
        let without = score_dir(&dir).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        eprintln!(
            "RAM tie groups on {} candidates: {} without the replay, {} with it",
            with.candidates.len(),
            tie_groups(&without),
            tie_groups(&with)
        );
        assert!(tie_groups(&without) > 0);
        assert_eq!(tie_groups(&with), 0);
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
        write_synthetic_builder(&dir, true);
    }

    // ---- ADR 0056: the tier term on simulated hardware (Rule 4) and properties (Rule 5)

    const TWO_GPU: &str = include_str!("../../costmodel/fixtures/two_gpu_cpu_nvme.toml");
    const WAN: &str = include_str!("../../costmodel/fixtures/wan_two_node.toml");

    fn graph(toml: &str, threads: u32) -> HardwareGraph {
        let mut g = HardwareGraph::from_toml(toml).unwrap();
        if let Some(c) = g.devices.iter_mut().find(|d| d.id == "cpu0") {
            c.threads = Some(threads);
        }
        g
    }

    /// (vram, ram, threads, decode disk bytes per token) of the candidate with that id.
    fn facts(file: &CandidatesFile, outs: &[TierOutcome], id: &str) -> (u32, u32, u32, f64) {
        let e = file
            .candidates
            .iter()
            .find(|e| e.candidate.id == id)
            .unwrap();
        let (vram, ram) = variant_slots(&e.variant).unwrap();
        let o = outcome_of(outs, &e.variant).unwrap();
        (vram, ram, e.variant.threads, o.decode.disk_bytes_per_token)
    }

    fn pick(ok: &[(String, f64)]) -> &str {
        &ok.iter().min_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0
    }

    // hardware: none
    #[test]
    fn rtx3090_and_two_gpu_priors_rank_ram_options_by_disk_bytes() {
        let (l, b, file) = synthetic_enumeration();
        let outs = truth_replay(&l, &file);
        for g in [graph(RTX3090_PRIOR, b.threads), graph(TWO_GPU, b.threads)] {
            let (ok, dropped) = predict_builder(&g, &l, &file, &outs);
            assert!(dropped.is_empty(), "{dropped:?}");
            assert_eq!(ok.len(), file.candidates.len());
            // candidates that differ only in ram_slots never predict equal
            for (i, x) in file.candidates.iter().enumerate() {
                for y in &file.candidates[i + 1..] {
                    let (a, b) = (&x.variant, &y.variant);
                    let same = a.threads == b.threads
                        && a.env
                            .iter()
                            .all(|(k, v)| k.contains("RAM_SLOT") || b.env.get(k) == Some(v));
                    let (va, ra) = variant_slots(a).unwrap();
                    let (vb, rb) = variant_slots(b).unwrap();
                    if same && va == vb && ra != rb {
                        let t = |id: &str| ok.iter().find(|e| e.0 == id).unwrap().1;
                        assert_ne!(
                            t(&x.candidate.id),
                            t(&y.candidate.id),
                            "{} vs {}",
                            x.candidate.id,
                            y.candidate.id
                        );
                    }
                }
            }
            // the pick holds the most RAM slots of its VRAM tier
            let (vram, ram, ..) = facts(&file, &outs, pick(&ok));
            let most = file
                .candidates
                .iter()
                .filter_map(|e| variant_slots(&e.variant))
                .filter(|s| s.0 == vram)
                .map(|s| s.1)
                .max()
                .unwrap();
            assert_eq!(ram, most);
        }
    }

    /// A session's measured disk curve replaces the prior's `nvme0`: a quarter of the rate makes
    /// the candidates that read from disk slower, and one file parses into the curve.
    // hardware: none
    #[test]
    fn a_measured_disk_curve_replaces_the_prior_in_the_score() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/builder-synthetic");
        let layout: ModelLayout =
            serde_json::from_str(&std::fs::read_to_string(dir.join("layout.json")).unwrap())
                .unwrap();
        let cands: CandidatesFile =
            serde_json::from_str(&std::fs::read_to_string(dir.join("candidates.json")).unwrap())
                .unwrap();
        let runs = read_builder_runs(&dir).unwrap();
        let outs = read_tier_replay(&dir.join("tier-replay.json")).unwrap();
        let prior = HardwareGraph::from_toml(RTX3090_PRIOR).unwrap();
        let slow = DiskCurve {
            curve: prior
                .device(BUILDER_DISK)
                .unwrap()
                .queue_curve
                .iter()
                .map(|p| QueuePoint {
                    depth: p.depth,
                    bytes_per_s: p.bytes_per_s * 0.25,
                })
                .collect(),
            chunk_bytes: Some(1 << 20),
        };
        let q = cands.budget.threads;
        let base = score_builder_curve("s", &layout, &cands, &runs, q, Some(&outs), None).unwrap();
        let same = score_builder_tier("s", &layout, &cands, &runs, q, Some(&outs)).unwrap();
        let times =
            |s: &Score| -> Vec<f64> { s.candidates.iter().map(|c| c.predicted_s).collect() };
        assert_eq!(times(&base), times(&same));
        let slowed =
            score_builder_curve("s", &layout, &cands, &runs, q, Some(&outs), Some(&slow)).unwrap();
        // the fit re-balances the other rates, so not every candidate slows; the numbers move
        assert_ne!(times(&slowed), times(&base));
        // the file form
        let tmp = std::env::temp_dir().join(format!("tetraeder-curve-{}.json", std::process::id()));
        std::fs::write(
            &tmp,
            serde_json::json!({
                "queue_chunk_bytes": 1 << 20,
                "queue_curve": [{"depth": 1, "bytes_per_s": 1.0e9}, {"depth": 16, "bytes_per_s": 4.0e9}],
            })
            .to_string(),
        )
        .unwrap();
        let d = read_disk_curve(&tmp).unwrap();
        assert_eq!((d.curve.len(), d.chunk_bytes), (2, Some(1 << 20)));
        std::fs::write(&tmp, r#"{"queue_curve": []}"#).unwrap();
        assert!(read_disk_curve(&tmp).is_err());
        let _ = std::fs::remove_file(tmp);
    }

    // hardware: none
    #[test]
    fn time_model_of_a_builder_candidate_reads_the_graph() {
        let (l, b, file) = synthetic_enumeration();
        let outs = truth_replay(&l, &file);
        let g = graph(RTX3090_PRIOR, b.threads);
        let id = &file.candidates[0].candidate.id;
        let m = builder_time_model(&g, &l, &file, &outs, id).unwrap();
        assert!(m.gpu_layer_s > 0.0, "{m:?}");
        assert!(m.cpu_group_s >= 0.0 && m.cpu_expert_s >= 0.0, "{m:?}");
        assert_eq!(m.curve, g.device(BUILDER_DISK).unwrap().queue_curve);
        assert_eq!((m.workers, m.prefetch_depth), (SERVICE_WORKERS, 2));
        assert!(builder_time_model(&g, &l, &file, &outs, "nope").is_err());
    }

    // hardware: none
    #[test]
    fn a_throttled_nvme_moves_the_pick_to_fewer_disk_bytes() {
        let (l, b, file) = synthetic_enumeration();
        let outs = truth_replay(&l, &file);
        for toml in [RTX3090_PRIOR, TWO_GPU] {
            let g = graph(toml, b.threads);
            let mut slow = g.clone();
            for d in &mut slow.devices {
                for q in &mut d.queue_curve {
                    q.bytes_per_s *= 0.25;
                }
            }
            let bytes = |g: &HardwareGraph| {
                let (ok, dropped) = predict_builder(g, &l, &file, &outs);
                assert!(dropped.is_empty(), "{dropped:?}");
                facts(&file, &outs, pick(&ok)).3
            };
            assert!(
                bytes(&slow) <= bytes(&g),
                "{} vs {}",
                bytes(&slow),
                bytes(&g)
            );
        }
    }

    // hardware: none
    #[test]
    fn a_cpu_cotenant_halves_the_quota_and_the_pick_fits() {
        use crate::candidate::Workload;
        let (l, b, _) = synthetic_enumeration();
        let quota = b.threads / 2;
        let half = crate::candidate::Budget {
            threads: quota,
            ..b
        };
        let w = Workload {
            prompt: 2048,
            r#gen: 128,
        };
        let en = crate::builder::enumerate_builder(&half, &l, &w);
        let c = crate::campaign::build_builder("moe", "org/name:file.gguf", &l, &half, &w, &en);
        let file = CandidatesFile::new(&c, &l, &half, &en);
        let outs = truth_replay(&l, &file);
        let g = graph(RTX3090_PRIOR, quota);
        let (ok, dropped) = predict_builder(&g, &l, &file, &outs);
        assert!(dropped.is_empty(), "{dropped:?}");
        let (.., threads, _) = facts(&file, &outs, pick(&ok));
        assert!(threads <= quota, "{threads} > {quota}");
    }

    // hardware: none
    #[test]
    fn a_graph_without_a_disk_drops_the_candidates_that_read_one() {
        let (l, b, file) = synthetic_enumeration();
        let outs = truth_replay(&l, &file);
        // the RTX 3090 prior without its NVMe: MissingSpec for every candidate with disk bytes
        let mut g = graph(RTX3090_PRIOR, b.threads);
        g.devices.retain(|d| d.id != "nvme0");
        g.links.retain(|k| k.a != "nvme0");
        let (ok, dropped) = predict_builder(&g, &l, &file, &outs);
        assert!(!dropped.is_empty());
        assert!(
            dropped.iter().all(|d| d.1.contains("disk rate")),
            "{dropped:?}"
        );
        for (id, _) in &dropped {
            assert!(facts(&file, &outs, id).3 > 0.0, "{id}");
        }
        for (id, _) in &ok {
            assert_eq!(facts(&file, &outs, id).3, 0.0, "{id}");
        }
        assert_eq!(ok.len() + dropped.len(), file.candidates.len());
        // the WAN fixture has no CPU and no disk: nothing of a builder plan scores on it
        let wan = graph(WAN, b.threads);
        let (ok, dropped) = predict_builder(&wan, &l, &file, &outs);
        assert!(ok.is_empty() && dropped.len() == file.candidates.len());
        // with a host CPU added (and still no disk), candidates that read the disk are dropped
        // with MissingSpec and the ones that do not score
        let rtx = graph(RTX3090_PRIOR, b.threads);
        let mut wan_cpu = wan.clone();
        wan_cpu
            .devices
            .extend(rtx.devices.iter().filter(|d| d.id == "cpu0").cloned());
        wan_cpu
            .links
            .extend(rtx.links.iter().filter(|k| k.id == "pcie-gpu0").cloned());
        let (ok, dropped) = predict_builder(&wan_cpu, &l, &file, &outs);
        assert!(!ok.is_empty() && !dropped.is_empty());
        assert!(
            dropped.iter().all(|d| d.1.contains("disk rate")),
            "{dropped:?}"
        );
        assert_eq!(ok.len() + dropped.len(), file.candidates.len());
        for (id, _) in &ok {
            assert_eq!(facts(&file, &outs, id).3, 0.0, "{id}");
        }
    }

    // hardware: none
    #[test]
    fn tier_none_and_a_zero_byte_tier_predict_bit_identically_on_every_fixture() {
        let zero = |mut f: RunFeatures| {
            f.tier = Some(TierFeatures {
                disk: BUILDER_DISK.into(),
                path: ReadPath::Service,
                depth: 8,
                sync_depth: 4,
                decode_sync_bytes_per_token: 0.0,
                decode_prefetch_bytes_per_token: 0.0,
                prefill_bytes_per_token: 0.0,
            });
            f
        };
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut dirs = vec![root.join("fixtures/builder-synthetic")];
        for day in std::fs::read_dir(root.join("../../bench/results"))
            .into_iter()
            .flatten()
            .flatten()
        {
            for d in std::fs::read_dir(day.path())
                .into_iter()
                .flatten()
                .flatten()
            {
                if d.path().join("candidates.json").is_file() {
                    dirs.push(d.path());
                }
            }
        }
        let mut n = 0;
        for dir in dirs.iter().filter(|d| d.join("candidates.json").is_file()) {
            let layout: ModelLayout =
                serde_json::from_str(&std::fs::read_to_string(dir.join("layout.json")).unwrap())
                    .unwrap();
            let cands: CandidatesFile = serde_json::from_str(
                &std::fs::read_to_string(dir.join("candidates.json")).unwrap(),
            )
            .unwrap();
            let Ok(s) = score_dir(dir) else { continue };
            let hw = s.calibrated.clone();
            for e in &cands.candidates {
                let v = &e.variant;
                let f = match v.placement {
                    Some(Placement::Builder { vram_slots, .. }) => {
                        builder_features(&layout, vram_slots, 0.7, (v.threads, 256, true))
                    }
                    p => features(&layout, p, v.threads, &v.args, &v.env),
                };
                let a = predict(&hw, &f, v.prompt, v.r#gen);
                let b = predict(&hw, &zero(f), v.prompt, v.r#gen);
                assert_eq!(a, b, "{}/{}", dir.display(), v.id);
                n += 1;
            }
        }
        assert!(n > 0);
    }

    /// Replay files from before ADR 0056 have no `full_hit_share`, and the reader ignores
    /// fields it does not know.
    // hardware: none
    #[test]
    fn replay_files_parse_with_missing_and_unknown_fields() {
        let dir = std::env::temp_dir().join(format!("tetraeder-replay-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("tier-replay.json");
        std::fs::write(
            &p,
            r#"{"tool": "x", "trace": {"fnv1a64": "00"}, "replay_version": 1, "outcomes": [
              {"vram_slots": 2, "ram_slots": 4, "policy": "lru", "prefetch": true, "extra": 1,
               "decode": {"vram_share": 0.2, "sync_miss_bytes_per_token": 3.0,
                          "prefetch_bytes_per_token": 4.0, "disk_bytes_per_token": 7.0,
                          "ram_hit_rate_engine_mean": 0.5, "steps": 48},
               "prefill": {"vram_share": 0.1, "disk_bytes_per_token": 1.0}}]}"#,
        )
        .unwrap();
        let o = read_tier_replay(&p).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(o.len(), 1);
        assert_eq!(o[0].decode.full_hit(), 0.2);
        assert_eq!(o[0].decode.sync_miss_bytes_per_token, 3.0);
        assert_eq!(o[0].prefill.disk_bytes_per_token, 1.0);
        assert!(o[0].prefetch && o[0].policy == "lru");
    }

    /// The committed 30B run (ADR 0056 context): at 32 VRAM slots, 32 against 128 RAM slots
    /// decoded 19.4 against 33.8 tok/s and the model without a RAM tier predicted them equal.
    /// The bytes here come from the run's expert counters (reads beyond the 2898 cold fills are
    /// decode reads), not from a replay: the 30B has no committed trace yet.
    // hardware: none
    #[test]
    fn the_committed_30b_ram_tiers_are_no_longer_predicted_equal() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../bench/results/20261102/builder-real-202732.json");
        let Ok(text) = std::fs::read_to_string(&path) else {
            eprintln!("skip: {} not committed", path.display());
            return;
        };
        let j: Value = serde_json::from_str(&text).unwrap();
        let l = fixtures::moe();
        let (prompt, gen_tokens) = (j["prompt_tokens"].as_u64().unwrap() as u32, 16);
        let num = |r: &Value, k: &str| r["experts"][k].as_f64().unwrap();
        let cold = num(&j["slots32_ram128_cpupolicy"], "experts_read");
        let outcome = |key: &str| {
            let r = &j[key];
            let bytes = num(r, "io_bytes") / num(r, "experts_read");
            let hits = num(r, "compute_hits");
            let steps = num(r, "full_steps") + num(r, "partial_steps") + num(r, "mixed_steps");
            let decode_reads = (num(r, "experts_read") - cold).max(0.0);
            let per_tok = decode_reads * bytes / f64::from(gen_tokens);
            let phase = |disk: f64| TierPhase {
                vram_share: hits / (hits + num(r, "compute_misses")),
                full_hit_share: Some(num(r, "full_steps") / steps),
                sync_miss_bytes_per_token: disk,
                disk_bytes_per_token: disk,
                ..TierPhase::default()
            };
            TierOutcome {
                vram_slots: 32,
                ram_slots: 0,
                policy: "lru".into(),
                prefetch: false,
                decode: phase(per_tok),
                prefill: phase(cold * bytes / f64::from(prompt)),
            }
        };
        let (small, big) = (
            outcome("slots32_ram32_cpupolicy"),
            outcome("slots32_ram128_cpupolicy"),
        );
        assert!(small.decode.disk_bytes_per_token > 0.0);
        assert_eq!(big.decode.disk_bytes_per_token, 0.0);
        let knobs = (5, 256, true);
        let g = graph(RTX3090_PRIOR, 5);
        // measured: the run read 3.3 GB/s from this disk
        let mut g_measured = g.clone();
        for d in &mut g_measured.devices {
            if d.id == BUILDER_DISK {
                d.queue_curve = vec![tetraeder_profiler::QueuePoint {
                    depth: 16,
                    bytes_per_s: 3.3e9,
                }];
            }
        }
        let tg = |g: &HardwareGraph, f: &RunFeatures| predict(g, f, prompt, 16).unwrap().tg_tok_s;
        // before: the RAM tier is not a feature, so the two predict the same
        let before = |o: &TierOutcome| builder_features(&l, 32, o.decode.vram_share, knobs);
        assert_eq!(tg(&g, &before(&small)), tg(&g, &before(&big)));
        let after = |o: &TierOutcome| {
            builder_features_tier(&l, 32, o, knobs, ReadPath::Reader, (16, 16), BUILDER_DISK)
        };
        let (a_small, a_big) = (tg(&g, &after(&small)), tg(&g, &after(&big)));
        let (m_small, m_big) = (
            tg(&g_measured, &after(&small)),
            tg(&g_measured, &after(&big)),
        );
        eprintln!(
            "30B decode tok/s, 32 vs 128 RAM slots: measured 19.40 vs 33.81 (1.74x); before {:.2} vs {:.2}; \
             after (prior curve) {a_small:.2} vs {a_big:.2} ({:.2}x); after (3.3 GB/s) {m_small:.2} vs {m_big:.2} ({:.2}x)",
            tg(&g, &before(&small)),
            tg(&g, &before(&big)),
            a_big / a_small,
            m_big / m_small
        );
        assert!(a_big > a_small * 1.1 && m_big > m_small * 1.1);
    }

    /// A replay that disagrees with the anchors' measured RAM hit rate by more than 0.5 points
    /// falls back to the hit model and says so; the fixture still scores.
    // hardware: none
    #[test]
    fn a_replay_that_fails_the_fidelity_gate_falls_back_to_the_hit_model() {
        let dir = std::env::temp_dir().join(format!("tetraeder-builder-fg-{}", std::process::id()));
        write_synthetic_builder(&dir, true);
        let mut outs = read_tier_replay(&dir.join("tier-replay.json")).unwrap();
        for o in &mut outs {
            o.decode.ram_hit_rate_engine_mean += 0.02;
        }
        std::fs::write(
            dir.join("tier-replay.json"),
            serde_json::to_string(&serde_json::json!({"outcomes": outs})).unwrap(),
        )
        .unwrap();
        let s = score_dir(&dir).unwrap();
        // a replay that lacks a variant's slot pair falls back too
        outs.truncate(1);
        std::fs::write(
            dir.join("tier-replay.json"),
            serde_json::to_string(&serde_json::json!({"outcomes": outs})).unwrap(),
        )
        .unwrap();
        let t = score_dir(&dir).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert!(
            s.tier_source
                .starts_with("hitmodel (fidelity gate failed: anchor"),
            "{}",
            s.tier_source
        );
        assert!(s.tier_source.contains("+0.02"), "{}", s.tier_source);
        assert!(
            t.tier_source
                .starts_with("hitmodel (fidelity gate failed: no outcome")
        );
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(48))]

        /// Every enumerated builder candidate, with the service's bounce buffers, fits the budget.
        #[test]
        fn builder_candidates_fit_their_budget_with_the_io_bounce(
            gpu in 4_000_000_000u64..40_000_000_000,
            ram in 8_000_000_000u64..120_000_000_000,
            threads in 2u32..32,
            shards in 1u64..=16,
        ) {
            use crate::candidate::{Budget, BUILDER_IO, Workload, footprint};
            let l = fixtures::moe();
            let w = Workload { prompt: 2048, r#gen: 128 };
            let b = Budget { gpu_bytes: gpu, ram_bytes: ram, threads };
            let en = crate::builder::enumerate_builder(&b, &l, &w);
            // the service's bounce: 16 workers * shards * 1 MiB, inside BUILDER_IO
            let bounce = 16 * shards * (1 << 20);
            proptest::prop_assert!(bounce <= BUILDER_IO);
            for c in &en.candidates {
                let f = footprint(&l, c.placement, c.ubatch(), &w, tetraeder_plan::KvFormat::default());
                proptest::prop_assert!(f.gpu <= b.gpu_bytes, "{} gpu", c.id);
                proptest::prop_assert!(f.host - BUILDER_IO + bounce <= b.ram_bytes, "{} host", c.id);
            }
        }

        /// A higher VRAM share (the disk bytes fixed) never predicts slower on the RTX 3090 prior,
        /// for the reader and the service. Not for `ServiceOverlap`: there the CPU experts hide
        /// behind the synchronous reads (`max(C_exp, t_s)`), so moving them to the GPU can only
        /// add GPU time while the disk bytes stay; with bytes that fall with the share (as in
        /// a replay) it holds again. Evidence: lo 0.51, up 0.90, sync 103 MB.
        #[test]
        fn a_higher_vram_share_is_never_slower(
            lo in 0.0f64..=1.0, up in 0.0f64..=1.0,
            sync in 0.0f64..5e8, pf in 0.0f64..5e8, p in 0u8..2,
        ) {
            let (share_lo, share_hi) = (lo.min(up), lo.max(up));
            let l = fixtures::moe();
            let g = graph(RTX3090_PRIOR, 12);
            let path = [ReadPath::Reader, ReadPath::Service][usize::from(p)];
            let time = |share: f64| {
                let o = TierOutcome {
                    decode: TierPhase {
                        vram_share: share,
                        full_hit_share: Some(share),
                        sync_miss_bytes_per_token: sync,
                        prefetch_bytes_per_token: pf,
                        ..TierPhase::default()
                    },
                    ..TierOutcome::default()
                };
                let f = builder_features_tier(&l, 32, &o, (12, 256, true), path, (16, 16), BUILDER_DISK);
                predict(&g, &f, 512, 64).unwrap().request_s
            };
            proptest::prop_assert!(time(share_hi) <= time(share_lo) * (1.0 + 1e-12));
        }
    }

    /// ADR 0056 section 3: every committed fixture re-predicted with its own session's fit matches
    /// the snapshot `bench/ledger.json`. After a model change: `tetraeder-plan ledger --write`.
    /// Skips cleanly when no snapshot is committed.
    #[test]
    fn committed_results_match_the_ledger() {
        use crate::ledger::{Ledger, check, evaluate};
        let bench = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench");
        // a box checkout (no .git) gets no bench/results (taxameter.toml [push])
        if !bench.join("../.git").exists() {
            eprintln!("skip: box checkout, bench/results not pushed");
            return;
        }
        let Ok(text) = std::fs::read_to_string(bench.join("ledger.json")) else {
            eprintln!("skip: no bench/ledger.json");
            return;
        };
        let snapshot: Ledger = serde_json::from_str(&text).unwrap();
        let problems = check(&evaluate(&bench.join("results")), &snapshot);
        assert!(problems.is_empty(), "{}", problems.join("\n"));
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
