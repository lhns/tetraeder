//! The plain numbers the virtual-clock replay needs (ADR 0056), from a calibrated prediction.
//!
//! `tetraeder-bench replay --timed` reads this file. The planner keeps its own copy of the struct
//! (it does not depend on `memory`); `crates/memory/tests/data/time-model.json` is the sample
//! both sides' tests read, so a field that drifts fails a test on either side.

use serde::{Deserialize, Serialize};
use tetraeder_costmodel::predict::{ReadPath, RunFeatures, predict};
use tetraeder_profiler::{DeviceKind, HardwareGraph, QueuePoint};

fn two() -> u32 {
    2
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimeModel {
    /// GPU compute of one layer of one decode token, seconds (switches included).
    pub gpu_layer_s: f64,
    /// CPU time of one routed (token, expert) pair, seconds.
    pub cpu_expert_s: f64,
    /// CPU time per group of experts: `c0` times the expert ops of one layer.
    pub cpu_group_s: f64,
    /// The disk's bytes per second by queue depth.
    pub curve: Vec<QueuePoint>,
    pub chunk_bytes: u64,
    pub workers: u32,
    pub path: ReadPath,
    pub sync_depth: u32,
    #[serde(default = "two")]
    pub prefetch_depth: u32,
}

/// What the I/O side of a run is configured with.
#[derive(Debug, Clone, Copy)]
pub struct IoShape {
    pub workers: u32,
    pub chunk_bytes: u64,
    pub prefetch_depth: u32,
}

/// The time model of `f` on `hw`: GPU and CPU compute per layer from the prediction's decode
/// breakdown, the disk curve from the graph. `n_layer` and `cpu_pairs_per_token` (routed
/// (token, expert) pairs computed on the CPU per decode token) split the CPU part into its
/// per-group and per-pair costs. Errors when the run has no tier (nothing to read) or the
/// prediction fails.
pub fn time_model(
    hw: &HardwareGraph,
    f: &RunFeatures,
    n_layer: u32,
    cpu_pairs_per_token: f64,
    prompt: u32,
    gen_tokens: u32,
    io: IoShape,
) -> Result<TimeModel, String> {
    let tier = f
        .tier
        .as_ref()
        .ok_or("the run has no tier (no disk reads)")?;
    let disk = hw
        .device(&tier.disk)
        .ok_or_else(|| format!("no disk {} in the hardware graph", tier.disk))?;
    let curve = if disk.queue_curve.is_empty() {
        let bw = disk
            .disk_rate(1)
            .ok_or_else(|| format!("disk {} has no rate", tier.disk))?;
        vec![QueuePoint {
            depth: 1,
            bytes_per_s: bw,
        }]
    } else {
        disk.queue_curve.clone()
    };
    let p = predict(hw, f, prompt, gen_tokens).map_err(|e| e.to_string())?;
    let term = |label: &str| {
        p.tg_breakdown
            .iter()
            .filter(|(l, _)| l == label)
            .map(|(_, s)| s)
            .sum::<f64>()
    };
    let layers = f64::from(n_layer.max(1));
    let (mut gpu_s, mut cpu_s) = (term("switches"), 0.0);
    let mut cpu_c0_s = 0.0;
    for part in &f.parts {
        let Some(d) = hw.device(&part.device) else {
            continue;
        };
        let t = term(&format!("compute:{}", d.id));
        match d.kind {
            DeviceKind::Gpu => gpu_s += t,
            DeviceKind::Cpu => {
                cpu_s += t;
                cpu_c0_s += d.compute.map_or(0.0, |c| c.c0_eager_s) * part.ops_per_token;
            }
            DeviceKind::Disk => {}
        }
    }
    let cpu_expert_s = if cpu_pairs_per_token > 0.0 {
        ((cpu_s - cpu_c0_s) / cpu_pairs_per_token).max(0.0)
    } else {
        0.0
    };
    Ok(TimeModel {
        gpu_layer_s: gpu_s / layers,
        cpu_expert_s,
        cpu_group_s: cpu_c0_s / layers,
        curve,
        chunk_bytes: io.chunk_bytes,
        workers: io.workers,
        path: tier.path,
        sync_depth: tier.sync_depth,
        prefetch_depth: io.prefetch_depth,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The contract with `tetraeder_memory::replay::timed::TimeModel`.
    const SAMPLE: &str = include_str!("../../memory/tests/data/time-model.json");

    #[test]
    fn the_shared_sample_round_trips() {
        let m: TimeModel = serde_json::from_str(SAMPLE).unwrap();
        assert_eq!(m.path, ReadPath::ServiceOverlap);
        assert_eq!(m.curve.len(), 4);
        let again = serde_json::to_value(&m).unwrap();
        let want: serde_json::Value = serde_json::from_str(SAMPLE).unwrap();
        assert_eq!(again, want);
    }
}
