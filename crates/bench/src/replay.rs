//! `tetraeder-bench replay`: scores RAM-tier policies on a routing trace (ADR 0050 (c)).

use anyhow::{Context, Result};
use serde_json::json;
use std::path::Path;
use tetraeder_memory::replay::REPLAY_VERSION;
use tetraeder_memory::replay::sim::{Config, Policy, Trace, replay_matrix};
use tetraeder_memory::replay::timed::{TimeModel, replay_timed};
use tetraeder_memory::replay::trace::Record;

/// FNV-1a 64 of a byte string, the trace file's fingerprint in the output.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Replays `trace_path` for every combination of the lists (prefetch on and off when the trace
/// carries predictions) and writes the outcomes to `out`. Returns the number of configurations.
pub fn run(
    trace_path: &Path,
    vram: &[u32],
    ram: &[u32],
    policies: &[Policy],
    out: &Path,
) -> Result<usize> {
    run_timed(trace_path, vram, ram, policies, out, None)
}

/// [`run`] that also replays every configuration on the virtual clock of `time_model` (ADR 0056)
/// and writes the timings in `timed`, one entry per outcome in the same order.
pub fn run_timed(
    trace_path: &Path,
    vram: &[u32],
    ram: &[u32],
    policies: &[Policy],
    out: &Path,
    time_model: Option<&Path>,
) -> Result<usize> {
    let trace =
        Trace::load(trace_path).with_context(|| format!("read trace {}", trace_path.display()))?;
    let fingerprint = fnv1a64(
        &std::fs::read(trace_path)
            .with_context(|| format!("read trace {}", trace_path.display()))?,
    );
    let has_predictions = trace
        .records
        .iter()
        .any(|r| matches!(r, Record::Step(s) if s.predicted.is_some() || s.predicted2.is_some()));
    let modes: &[bool] = if has_predictions {
        &[true, false]
    } else {
        &[false]
    };
    let outcomes = replay_matrix(&trace, vram, ram, policies, modes);
    let timed = match time_model {
        None => None,
        Some(p) => {
            let raw: serde_json::Value = serde_json::from_slice(
                &std::fs::read(p).with_context(|| format!("read {}", p.display()))?,
            )
            .with_context(|| format!("parse {}", p.display()))?;
            let m: TimeModel = serde_json::from_value(raw.clone())
                .with_context(|| format!("time model {}", p.display()))?;
            let mut v = Vec::with_capacity(outcomes.len());
            for o in &outcomes {
                let cfg = Config {
                    vram_slots: o.vram_slots,
                    ram_slots: o.ram_slots,
                    policy: Policy::parse(&o.policy).map_err(anyhow::Error::msg)?,
                    prefetch: o.prefetch,
                };
                let t = replay_timed(&trace, &cfg, &m).map_err(anyhow::Error::msg)?;
                v.push(json!({
                    "vram_slots": o.vram_slots,
                    "ram_slots": o.ram_slots,
                    "policy": o.policy,
                    "prefetch": o.prefetch,
                    "timing": t,
                }));
            }
            Some((raw, v))
        }
    };
    let mut header = serde_json::to_value(&trace.header)?;
    if let Some(h) = header.as_object_mut() {
        // The profile is the size of the model's routing table: not part of the result.
        h.remove("profile");
    }
    let doc = json!({
        "tool": "tetraeder-bench replay",
        "adr": "0050 (c)",
        "replay_version": REPLAY_VERSION,
        "trace": {
            "fnv1a64": format!("{fingerprint:016x}"),
            "file": trace_path.file_name().map(|n| n.to_string_lossy().into_owned()),
            "header": header,
            "records": trace.records.len(),
            "has_predictions": has_predictions,
        },
        "grid": {
            "vram_slots": vram,
            "ram_slots": ram,
            "policies": policies.iter().map(Policy::name).collect::<Vec<_>>(),
        },
        "outcomes": outcomes,
    });
    let mut doc = doc;
    if let (Some((model, entries)), Some(d)) = (timed, doc.as_object_mut()) {
        d.insert("time_model".into(), model);
        d.insert("timed".into(), serde_json::Value::Array(entries));
    }
    if let Some(dir) = out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(out, serde_json::to_string_pretty(&doc)?)
        .with_context(|| format!("write {}", out.display()))?;
    Ok(outcomes.len())
}
