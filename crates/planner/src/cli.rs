//! The `explain`, `diff` and `what-if` subcommands of `tetraeder-plan` as functions from file
//! contents to output text, so tests need no files. Inputs are TOML: hardware graph, model
//! description, hints and plans.

use crate::builder::{build_builder_candidate, slot_row_bytes};
use crate::campaign::{self, CandidatesFile};
use crate::candidate::{Budget, Enumeration, Workload};
use crate::diff::plan_diff;
use crate::explain::explain;
use crate::layout::ModelLayout;
use crate::profile_quality::{Band, mix_at_budget, tensor_formats};
use crate::quality::BitsQuality;
use crate::score::{builder_time_model, read_tier_replay};
use crate::search::{SearchInput, search};
use crate::whatif::{Change, default_candidates, rank_purchases, what_if};
use anyhow::{Context, Result, anyhow};
use std::collections::BTreeMap;
use std::path::Path;
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

/// `explain --time-model`: the time model (ADR 0056) of one candidate of a builder fixture
/// directory (`layout.json`, `candidates.json`, `tier-replay.json`) on the hardware graph `hw`,
/// as JSON for `tetraeder-bench replay --timed`.
pub fn time_model_cmd(hw: &str, fixture: &Path, candidate: &str) -> Result<String> {
    let hw = HardwareGraph::from_toml(hw).context("parse hardware")?;
    hw.validate().map_err(|e| anyhow!("hardware: {e}"))?;
    let read = |name: &str| {
        let p = fixture.join(name);
        std::fs::read_to_string(&p).with_context(|| format!("read {}", p.display()))
    };
    let layout = serde_json::from_str(&read("layout.json")?).context("parse layout.json")?;
    let cands = serde_json::from_str(&read("candidates.json")?).context("parse candidates.json")?;
    let outs = read_tier_replay(&fixture.join("tier-replay.json")).map_err(|e| anyhow!(e))?;
    let m = builder_time_model(&hw, &layout, &cands, &outs, candidate).map_err(|e| anyhow!(e))?;
    json(&m)
}

/// A builder run of an exit session as a variant of a ledger directory (ADR 0056): its tiers in
/// bytes (the `TETRAEDER_EXIT_*_SLOT_BYTES` it ran with), threads, ubatch, op offload and the
/// request it measured.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageSpec {
    pub id: String,
    pub vram_bytes: u64,
    pub ram_bytes: u64,
    pub threads: u32,
    pub ubatch: u32,
    pub op_offload: bool,
    pub prompt: u32,
    pub r#gen: u32,
}

impl std::str::FromStr for StageSpec {
    type Err = String;

    /// `id=VRAM_BYTES,RAM_BYTES,THREADS,UBATCH,OP_OFFLOAD(0|1),PROMPT,GEN`.
    fn from_str(s: &str) -> Result<Self, String> {
        let (id, rest) = s.split_once('=').ok_or_else(|| {
            format!("stage {s:?}: expected id=vram,ram,threads,ubatch,op,prompt,gen")
        })?;
        let f: Vec<&str> = rest.split(',').collect();
        let [vram, ram, threads, ubatch, op, prompt, gen_] = f[..] else {
            return Err(format!("stage {s:?}: expected 7 numbers after the id"));
        };
        let num = |v: &str| {
            v.parse::<u64>()
                .map_err(|e| format!("stage {s:?}: {v:?}: {e}"))
        };
        Ok(Self {
            id: id.to_string(),
            vram_bytes: num(vram)?,
            ram_bytes: num(ram)?,
            threads: num(threads)? as u32,
            ubatch: num(ubatch)? as u32,
            op_offload: num(op)? != 0,
            prompt: num(prompt)? as u32,
            r#gen: num(gen_)? as u32,
        })
    }
}

/// The `candidates.json` of a ledger directory (`ledger-<session>/`): the four builder anchors
/// at the `anchor` workload and the session's builder stages as candidates, each with its own
/// request, so the ledger reads the directory like a fixture (results are `<variant id>.r1.json`;
/// a stage `main` is variant `cand-main`).
pub fn ledger_candidates(
    layout: &ModelLayout,
    budget: &Budget,
    anchor: &Workload,
    (model_key, hf): (&str, &str),
    stages: &[StageSpec],
) -> CandidatesFile {
    let row = slot_row_bytes(layout).max(1);
    let candidates = stages
        .iter()
        .map(|s| {
            let slots = |b: u64| u32::try_from(b / row).unwrap_or(u32::MAX);
            build_builder_candidate(
                &s.id,
                layout,
                (slots(s.vram_bytes), slots(s.ram_bytes)),
                (s.threads, s.ubatch, s.op_offload),
                &Workload {
                    prompt: s.prompt,
                    r#gen: s.r#gen,
                },
            )
        })
        .collect();
    let en = Enumeration {
        candidates,
        dropped: vec![],
        placements: vec![],
    };
    let c = campaign::build_builder(model_key, hf, layout, budget, anchor, &en);
    let mut file = CandidatesFile::new(&c, layout, budget, &en);
    for (e, s) in file.candidates.iter_mut().zip(stages) {
        e.variant.prompt = s.prompt;
        e.variant.r#gen = s.r#gen;
    }
    file
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate::Placement;
    use crate::layout::fixtures;

    // hardware: none
    #[test]
    fn a_ledger_directory_has_the_anchors_and_the_stages_with_their_own_request() {
        let layout = fixtures::moe();
        let budget = Budget {
            gpu_bytes: 14_000_000_000,
            ram_bytes: 60_000_000_000,
            threads: 15,
        };
        let row = slot_row_bytes(&layout);
        let spec = format!("main={},{},15,2048,1,32768,64", 64 * row, 32 * row);
        let stage: StageSpec = spec.parse().unwrap();
        assert_eq!(
            (stage.threads, stage.ubatch, stage.op_offload),
            (15, 2048, true)
        );
        let f = ledger_candidates(
            &layout,
            &budget,
            &Workload {
                prompt: 512,
                r#gen: 32,
            },
            ("moe", "org/m:f.gguf"),
            &[stage],
        );
        assert!(!f.anchors.is_empty());
        assert!(f.anchors.iter().all(|a| (a.prompt, a.r#gen) == (512, 32)));
        let [c] = &f.candidates[..] else {
            panic!("{:?}", f.candidates.len());
        };
        assert_eq!(c.variant.id, "cand-main");
        assert_eq!((c.variant.prompt, c.variant.r#gen), (32768, 64));
        assert_eq!(
            c.candidate.placement,
            Placement::Builder {
                vram_slots: 64,
                ram_slots: 32
            }
        );
        assert_eq!(c.variant.placement, Some(c.candidate.placement));
    }

    // hardware: none
    #[test]
    fn malformed_stages_are_refused() {
        for bad in [
            "main",
            "main=1,2,3",
            "main=a,2,3,4,5,6,7",
            "=1,2,3,4,5,6,7,8",
        ] {
            assert!(bad.parse::<StageSpec>().is_err(), "{bad}");
        }
    }
}
