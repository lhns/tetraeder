//! Scores a detector campaign directory (ADR 0041, Phase 3a exit test).
//!
//! The directory holds `obs-NN.json` (an [`Observation`] each, no fault label) and
//! `manifest.jsonl` (`{"id":"run-NN","label":"cpu+disk","faults":["cpu","disk"]}`), written by
//! `scripts/detector_campaign.sh`. Calibration runs (label `calib`) are not scored. The bars are
//! the ones the simulator battery uses: top-1 on at least 90% of single-fault runs, at most 1
//! false positive in the no-fault runs, both faults in the top 3 on at least 90% of pair runs.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::detect::{FaultClass, detect};
use crate::observation::Observation;

/// Share of single-fault and pair runs that must be right.
pub const MIN_RATE: f64 = 0.9;
/// False positives allowed over the no-fault runs.
pub const MAX_FALSE_POSITIVES: usize = 1;
/// Pair runs count when both faults are within this many ranks.
pub const PAIR_TOP: usize = 3;

#[derive(Debug, Deserialize)]
struct ManifestRow {
    id: String,
    #[serde(default)]
    label: String,
    #[serde(default)]
    faults: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunScore {
    pub id: String,
    pub label: String,
    /// Detector faults in rank order.
    pub found: Vec<FaultClass>,
    pub ok: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Tally {
    pub runs: usize,
    pub ok: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Summary {
    /// Single-fault runs with the injected fault at rank 1.
    pub single_top1: Tally,
    /// No-fault runs; `ok` counts the clean ones.
    pub no_fault: Tally,
    pub false_positives: usize,
    /// Pair runs with both faults in the top 3.
    pub pair_top3: Tally,
    pub runs: Vec<RunScore>,
    pub single_pass: bool,
    pub no_fault_pass: bool,
    pub pair_pass: bool,
    pub pass: bool,
}

fn fault_class(name: &str) -> Result<FaultClass, String> {
    match name {
        "cpu" => Ok(FaultClass::CpuCotenant),
        "disk" => Ok(FaultClass::DiskSlow),
        "vram" => Ok(FaultClass::VramPressure),
        "throttle" => Ok(FaultClass::GpuThrottle),
        other => Err(format!("unknown fault name {other:?} in manifest")),
    }
}

fn rate_ok(t: &Tally) -> bool {
    t.runs > 0 && t.ok as f64 >= MIN_RATE * t.runs as f64
}

/// Detector classes in rank order for one observation file.
fn found(path: &Path) -> Result<Vec<FaultClass>, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let obs: Observation =
        serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))?;
    Ok(detect(&obs).faults.iter().map(|f| f.class).collect())
}

/// Score every manifest row but calibration runs. A later row with the same id replaces an
/// earlier one (a resumed campaign). Missing or unreadable files are errors, not misses.
pub fn score_dir(dir: &Path) -> Result<Summary, String> {
    let manifest = dir.join("manifest.jsonl");
    let text = std::fs::read_to_string(&manifest)
        .map_err(|e| format!("read {}: {e}", manifest.display()))?;
    let mut rows: BTreeMap<String, ManifestRow> = BTreeMap::new();
    for (i, line) in text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        let row: ManifestRow = serde_json::from_str(line)
            .map_err(|e| format!("{}:{}: {e}", manifest.display(), i + 1))?;
        rows.insert(row.id.clone(), row);
    }
    let (mut single, mut none, mut pair) = (Tally::default(), Tally::default(), Tally::default());
    let mut runs = Vec::new();
    for row in rows.values().filter(|r| r.label != "calib") {
        let n = row.id.strip_prefix("run-").unwrap_or(&row.id);
        let found = found(&dir.join(format!("obs-{n}.json")))?;
        let truth = row
            .faults
            .iter()
            .map(|f| fault_class(f))
            .collect::<Result<Vec<_>, _>>()?;
        let (tally, ok) = match truth.as_slice() {
            [] => (&mut none, found.is_empty()),
            [c] => (&mut single, found.first() == Some(c)),
            [a, b] => {
                let top: Vec<_> = found.iter().take(PAIR_TOP).collect();
                (&mut pair, top.contains(&a) && top.contains(&b))
            }
            more => {
                return Err(format!(
                    "{}: {} faults, expected at most 2",
                    row.id,
                    more.len()
                ));
            }
        };
        tally.runs += 1;
        tally.ok += usize::from(ok);
        runs.push(RunScore {
            id: row.id.clone(),
            label: row.label.clone(),
            found,
            ok,
        });
    }
    let false_positives = none.runs - none.ok;
    let single_pass = rate_ok(&single);
    let no_fault_pass = none.runs > 0 && false_positives <= MAX_FALSE_POSITIVES;
    let pair_pass = rate_ok(&pair);
    Ok(Summary {
        single_top1: single,
        no_fault: none,
        false_positives,
        pair_top3: pair,
        runs,
        single_pass,
        no_fault_pass,
        pair_pass,
        pass: single_pass && no_fault_pass && pair_pass,
    })
}

/// Result directories `<root>/*/detector/` that have a manifest, sorted.
pub fn campaign_dirs(results_root: &Path) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(results_root) else {
        return Vec::new();
    };
    let mut dirs: Vec<_> = rd
        .filter_map(Result::ok)
        .map(|e| e.path().join("detector"))
        .filter(|d| d.join("manifest.jsonl").is_file())
        .collect();
    dirs.sort();
    dirs
}
