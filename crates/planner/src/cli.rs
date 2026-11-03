//! The `explain`, `diff` and `what-if` subcommands of `tetraeder-plan` as functions from file
//! contents to output text, so tests need no files. Inputs are TOML: hardware graph, model
//! description, hints and plans.

use crate::diff::plan_diff;
use crate::explain::explain;
use crate::profile_quality::{Band, mix_at_budget, tensor_formats};
use crate::quality::BitsQuality;
use crate::search::{SearchInput, search};
use crate::whatif::{Change, default_candidates, rank_purchases, what_if};
use anyhow::{Context, Result, anyhow};
use std::collections::BTreeMap;
use tetraeder_plan::{Hints, ModelDesc, Phase, Plan, QualityProfile};
use tetraeder_profiler::{DeviceKind, HardwareGraph};

/// Prompt tokens, generated tokens and concurrency of the workload the CLI plans for.
pub const WORKLOAD: (u32, u32, u32) = (512, 128, 1);

/// Memory kept free per device: a CUDA context and workspaces on a GPU, the OS on the CPU.
pub fn default_reserve(hw: &HardwareGraph) -> BTreeMap<String, u64> {
    hw.devices
        .iter()
        .map(|d| {
            let r = match d.kind {
                DeviceKind::Gpu => 600 << 20,
                DeviceKind::Cpu => 2 << 30,
                DeviceKind::Disk => 0,
            };
            (d.id.clone(), r)
        })
        .collect()
}

fn load(hw: &str, model: &str) -> Result<(HardwareGraph, ModelDesc)> {
    let hw = HardwareGraph::from_toml(hw).context("parse hardware")?;
    hw.validate().map_err(|e| anyhow!("hardware: {e}"))?;
    let model: ModelDesc = toml::from_str(model).context("parse model")?;
    Ok((hw, model))
}

fn hints_of(text: Option<&str>) -> Result<Hints> {
    let h = match text {
        Some(t) => Hints::from_toml(t).context("parse hints")?,
        None => Hints::default(),
    };
    h.check().map_err(|e| anyhow!("hints: {}", e.join("; ")))?;
    Ok(h)
}

/// Run `f` on a search input built from the files. The quality model is `BitsQuality` (loss from
/// bits per weight) until a measured quality profile can be loaded.
fn with_input<T>(
    hw: &HardwareGraph,
    model: &ModelDesc,
    hints: &Hints,
    phase: u8,
    linux: bool,
    f: impl FnOnce(&SearchInput) -> Result<T>,
) -> Result<T> {
    let q = BitsQuality::new(model, 1.0);
    f(&SearchInput {
        hw,
        model,
        hints,
        phase: Phase(phase),
        workload: WORKLOAD,
        quality: &q,
        reserve: default_reserve(hw),
        routing: None,
        linux,
    })
}

fn json<T: serde::Serialize>(v: &T) -> Result<String> {
    Ok(serde_json::to_string_pretty(v)?)
}

/// `explain`: search, then report why the chosen plan won.
pub fn explain_cmd(
    hw: &str,
    model: &str,
    hints: Option<&str>,
    phase: u8,
    linux: bool,
    as_json: bool,
) -> Result<String> {
    let (hw, model) = load(hw, model)?;
    let hints = hints_of(hints)?;
    with_input(&hw, &model, &hints, phase, linux, |inp| {
        let e = explain(inp, &search(inp));
        if as_json { json(&e) } else { Ok(e.to_text()) }
    })
}

/// `diff`: what moves between two plan files and how long the switch takes.
pub fn diff_cmd(hw: &str, model: &str, old: &str, new: &str, as_json: bool) -> Result<String> {
    let (hw, model) = load(hw, model)?;
    let old = Plan::from_toml(old).context("parse old plan")?;
    let new = Plan::from_toml(new).context("parse new plan")?;
    let s = plan_diff(&old, &new, &hw, &model).map_err(|e| anyhow!(e))?;
    if as_json { json(&s) } else { Ok(s.to_text()) }
}

/// `what-if`: the changes are `--add` strings (`gpu:4090`) and `--remove` device ids. With no
/// change at all the default purchase candidates are ranked.
#[allow(clippy::too_many_arguments)]
pub fn what_if_cmd(
    hw: &str,
    model: &str,
    hints: Option<&str>,
    phase: u8,
    linux: bool,
    add: &[String],
    remove: &[String],
    as_json: bool,
) -> Result<String> {
    let (hw, model) = load(hw, model)?;
    let hints = hints_of(hints)?;
    let mut changes = Vec::new();
    for a in add {
        changes.push(a.parse::<Change>().map_err(|e| anyhow!(e))?);
    }
    for r in remove {
        let id = r.strip_prefix("remove:").unwrap_or(r);
        changes.push(Change::Remove(id.to_string()));
    }
    with_input(&hw, &model, &hints, phase, linux, |inp| {
        if changes.is_empty() {
            let r = rank_purchases(inp, &default_candidates());
            if as_json {
                return json(&r);
            }
            let mut o = format!("before: {:.2} tok/s\n", r.before.tok_s);
            for p in &r.ranked {
                o += &format!(
                    "{:<18} {:+8.2} tok/s{}\n",
                    p.change,
                    p.gain_tok_s,
                    p.gain_per_1000_eur
                        .map_or(String::new(), |g| format!("  {g:+.2} per 1000 EUR"))
                );
            }
            for (c, why) in &r.skipped {
                o += &format!("{c}: skipped, {why}\n");
            }
            return Ok(o);
        }
        let w = what_if(inp, &changes).map_err(|e| anyhow!(e))?;
        if as_json { json(&w) } else { Ok(w.to_text()) }
    })
}

/// `mix`: the format mix within `budget` bytes. Returns the mix JSON and a summary. With a tensor
/// list (JSON array of names or of objects with a `name`, such as the tensor records of a
/// layout) the JSON is `{tensor: format}` as `compare_mix.py --mix` reads it; without, it is
/// `{group id: format}`.
pub fn mix_cmd(profile: &str, budget: u64, tensors: Option<&str>) -> Result<(String, String)> {
    let profile = QualityProfile::from_json(profile).map_err(|e| anyhow!(e))?;
    let mix = mix_at_budget(&profile, budget).map_err(|e| anyhow!(e))?;
    let band = Band::of(&profile);
    let formats = match tensors {
        None => mix.formats.clone(),
        Some(t) => {
            let v: Vec<serde_json::Value> = serde_json::from_str(t).context("parse tensor list")?;
            let names: Vec<&str> = v
                .iter()
                .filter_map(|x| x.as_str().or_else(|| x.get("name")?.as_str()))
                .collect();
            tensor_formats(&profile, &mix, names)
        }
    };
    let summary = format!(
        "{} bytes of {budget}, estimate {:.5}, corrected {:.5} (band {:.0}% corrected, {:.0}% raw)
",
        mix.bytes,
        mix.estimate,
        mix.corrected,
        band.corrected * 100.0,
        band.raw * 100.0
    );
    Ok((json(&formats)?, summary))
}
