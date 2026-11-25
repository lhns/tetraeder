//! One segment of a model priced on one node, calibrated on truncated anchors of its own span
//! (ADR 0092 section 6, tree node `bench-235b-segment-confirm`).
//!
//! The segment is a layer span with no embedding and no head, one builder placement (VRAM and
//! RAM slots per layer), the reader path. Its anchors are shorter spans of the same layers run
//! with the same slots; each is priced at the routing it measured (`measured_outcome`), the
//! cost model is fitted to them (`calib::fit`), and the segment is priced from the replay of its
//! own span (`tetraeder-bench replay --layers a..b --warm n`, per-layer rows).

use crate::candidate::{Placement, Workload, footprint};
use crate::cluster::segment_layout;
use crate::layout::ModelLayout;
use crate::score::{
    BUILDER_DISK, BuilderRun, READER_DEPTH, TierOutcome, builder_features_tier, builder_run,
};
use serde::Serialize;
use serde_json::Value;
use tetraeder_costmodel::calib::{Anchor, fit};
use tetraeder_costmodel::predict::{ReadPath, predict};
use tetraeder_plan::{KvFormat, SegmentShape, Span};
use tetraeder_profiler::{DeviceKind, HardwareGraph};

/// The largest fit band a prediction may be frozen on (ADR 0092 "Rerun rules", the value proposed; owner answer 7 of
/// ADR 0091): the band is the largest |ln(pred / meas)| of the anchors after the fit. Real-hidden
/// anchors measured 0.07; the owner set the limit at 0.10 (2026-11-25), under the 0.15 judging line, since an
/// anchor fit that misses its own anchors by more cannot be trusted to land inside it.
pub const FIT_BAND_LIMIT: f64 = 0.10;

/// One anchor: a span, its slot pair and the result files of its rounds (`exit_regime` JSON).
pub struct SegmentAnchor {
    pub id: String,
    pub span: Span,
    pub files: Vec<Value>,
}

/// What the fit and the prediction of a segment say.
#[derive(Debug, Serialize)]
pub struct SegmentPrediction {
    pub span: (u32, u32),
    pub vram_slots: u32,
    pub ram_slots: u32,
    pub ubatch: u32,
    pub threads: u32,
    /// Calibrated prediction (the frozen one).
    pub pp_tps: f64,
    pub tg_tps: f64,
    pub request_s: f64,
    /// The same features on the unfitted graph: the node-predict and cluster segment terms.
    pub uncalibrated_request_s: f64,
    pub binding: String,
    pub tg_breakdown: Vec<(String, f64)>,
    pub pp_breakdown: Vec<(String, f64)>,
    /// The same features on the unfitted graph (report only).
    pub uncalibrated_pp_tps: f64,
    pub uncalibrated_tg_tps: f64,
    pub band: f64,
    pub residuals: Vec<(String, f64)>,
    pub anchors: Vec<AnchorUsed>,
    /// Planner footprint of the segment against the graph's budgets, bytes.
    pub footprint_gpu: u64,
    pub footprint_host: u64,
    pub gpu_budget: u64,
    pub ram_budget: u64,
    /// Disk bytes per token the replay says the timed request reads (prefill, decode).
    pub replay_prefill_disk_bytes_per_token: f64,
    pub replay_decode_disk_bytes_per_token: f64,
    pub replay_vram_share: f64,
    pub replay_full_hit_share: f64,
    pub replay_ram_hit_rate: f64,
    pub replay_layer_rows: usize,
}

impl SegmentPrediction {
    /// Refuses a prediction whose anchors the fit does not reproduce within `limit`; a
    /// prediction is not frozen on such a fit.
    pub fn check_fit_band(&self, limit: f64) -> Result<(), String> {
        if self.band.is_finite() && self.band <= limit {
            Ok(())
        } else {
            Err(format!(
                "the fit band {:.3} is above the limit {limit}: the anchors do not agree, nothing is frozen",
                self.band
            ))
        }
    }
}

#[derive(Debug, Serialize)]
pub struct AnchorUsed {
    pub id: String,
    pub span: (u32, u32),
    pub rounds: usize,
    pub pp_tok_s: Option<f64>,
    pub tg_tok_s: Option<f64>,
    pub pair_share: Option<f64>,
    pub full_hit_rate: Option<f64>,
    pub ram_hit_rate: Option<f64>,
    pub prefill_bytes_per_token: Option<f64>,
}

fn used(a: &SegmentAnchor, r: &BuilderRun) -> AnchorUsed {
    AnchorUsed {
        id: a.id.clone(),
        span: (a.span.0, a.span.1),
        rounds: a.files.len(),
        pp_tok_s: r.pp_tok_s,
        tg_tok_s: r.tg_tok_s,
        pair_share: r.pair_share,
        full_hit_rate: r.hit_rate,
        ram_hit_rate: r.ram_hit_rate,
        prefill_bytes_per_token: r.prefill_bytes,
    }
}

/// Fits `hw` to `anchors` and prices the segment `seg` of `layout` at `(vram, ram)` slots from
/// `replay`. The anchors are always bare spans (no embedding, no head); `seg` says whether the
/// segment holds them, so the whole model `0..n` with both prices as `node-predict` does.
/// Returns the prediction and the fitted graph. `threads` and `ubatch` are the runs' (anchors
/// and judged share them). The replay must hold per-layer rows for exactly the span.
#[allow(clippy::too_many_arguments)]
pub fn segment_predict(
    hw: &HardwareGraph,
    layout: &ModelLayout,
    seg: SegmentShape,
    (vram, ram): (u32, u32),
    w: &Workload,
    threads: u32,
    ubatch: u32,
    anchors: &[SegmentAnchor],
    replay: &[TierOutcome],
) -> Result<(SegmentPrediction, HardwareGraph), String> {
    let span = seg.layers;
    if span.1 as usize > layout.layers.len() || span.0 >= span.1 {
        return Err(format!(
            "span {}..{} outside the layout's {} layers",
            span.0,
            span.1,
            layout.layers.len()
        ));
    }
    let knobs = (threads, ubatch, true);
    let depth = (READER_DEPTH, READER_DEPTH);
    let mut fitted_on = Vec::new();
    let mut list = Vec::new();
    for a in anchors {
        let sub = segment_layout(layout, a.span, false, false);
        let run = builder_run(&a.files);
        let o = run
            .measured_outcome(vram, ram)
            .ok_or_else(|| format!("anchor {}: no decode_layers.vram_pair_share", a.id))?;
        let features =
            builder_features_tier(&sub, vram, &o, knobs, ReadPath::Reader, depth, BUILDER_DISK)?;
        list.push(Anchor {
            id: a.id.clone(),
            features,
            prompt: w.prompt,
            r#gen: w.r#gen,
            pp_tok_s: run.pp_tok_s,
            tg_tok_s: run.tg_tok_s,
        });
        fitted_on.push(used(a, &run));
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
        .ok_or("no link between the GPU and the CPU")?;
    let f = fit(hw, &list, &gpu.id, &cpu.id, &link.id).map_err(|e| e.to_string())?;

    let sub = segment_layout(layout, span, seg.embed, seg.head);
    let o = replay
        .iter()
        .find(|t| t.vram_slots == vram && t.ram_slots == ram && t.policy == "lru" && t.prefetch)
        .ok_or_else(|| format!("no lru replay outcome with prefetch for v{vram} r{ram}"))?;
    let rows = o.decode.layers.as_ref().map_or(0, Vec::len);
    if rows != sub.layers.len() {
        return Err(format!(
            "the replay holds {rows} layer rows, the span {}..{} has {}",
            span.0,
            span.1,
            sub.layers.len()
        ));
    }
    let feat = builder_features_tier(&sub, vram, o, knobs, ReadPath::Reader, depth, BUILDER_DISK)?;
    let p = predict(&f.hw, &feat, w.prompt, w.r#gen).map_err(|e| e.to_string())?;
    let u = predict(hw, &feat, w.prompt, w.r#gen).map_err(|e| e.to_string())?;
    let fp = footprint(
        &sub,
        Placement::Builder {
            vram_slots: vram,
            ram_slots: ram,
        },
        ubatch,
        w,
        KvFormat::default(),
    );
    let out = SegmentPrediction {
        span: (span.0, span.1),
        vram_slots: vram,
        ram_slots: ram,
        ubatch,
        threads,
        pp_tps: p.pp_tok_s,
        tg_tps: p.tg_tok_s,
        request_s: p.request_s,
        uncalibrated_request_s: u.request_s,
        binding: p.binding,
        tg_breakdown: p.tg_breakdown,
        pp_breakdown: p.pp_breakdown,
        uncalibrated_pp_tps: u.pp_tok_s,
        uncalibrated_tg_tps: u.tg_tok_s,
        band: f.band,
        residuals: f.residuals.clone(),
        anchors: fitted_on,
        footprint_gpu: fp.gpu,
        footprint_host: fp.host,
        gpu_budget: gpu.capacity_bytes / 100 * 99,
        ram_budget: cpu.capacity_bytes / 100 * 99,
        replay_prefill_disk_bytes_per_token: o.prefill.disk_bytes_per_token,
        replay_decode_disk_bytes_per_token: o.decode.disk_bytes_per_token,
        replay_vram_share: o.decode.vram_share,
        replay_full_hit_share: o.decode.full_hit(),
        replay_ram_hit_rate: o.decode.ram_hit_rate_engine_mean,
        replay_layer_rows: rows,
    };
    Ok((out, f.hw))
}
