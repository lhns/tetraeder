//! `tetraeder detector-score`: the Phase 3a detector exit test on a campaign directory
//! (`bench/results/<date>/detector/`). Writes `score.json` next to the results; the exit code
//! says whether the bars are met.

use anyhow::{Context, Result, bail};
use clap::Args;
use std::path::PathBuf;
use tetraeder_analysis::score::score_dir;

#[derive(Args)]
pub struct ScoreArgs {
    /// Directory with obs-NN.json and manifest.jsonl.
    dir: PathBuf,
}

pub fn run(a: &ScoreArgs) -> Result<()> {
    let s = score_dir(&a.dir).map_err(anyhow::Error::msg)?;
    let out = a.dir.join("score.json");
    std::fs::write(&out, serde_json::to_string_pretty(&s)? + "\n")
        .with_context(|| format!("write {}", out.display()))?;
    println!(
        "single top-1 {}/{}  no-fault false positives {}/{}  pair top-3 {}/{}  -> {}",
        s.single_top1.ok,
        s.single_top1.runs,
        s.false_positives,
        s.no_fault.runs,
        s.pair_top3.ok,
        s.pair_top3.runs,
        if s.pass { "PASS" } else { "FAIL" }
    );
    if !s.pass {
        bail!("detector exit bars not met (see {})", out.display());
    }
    Ok(())
}
