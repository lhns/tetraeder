//! `tetraeder-plan candidates`: enumerate the candidate plans of a fixture and write the
//! candidate set and the bench spec that measures it (ADR 0040).

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tetraeder_planner::{Budget, CandidatesFile, ModelLayout, Workload, campaign, enumerate};

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Enumerate candidates for one GPU plus the CPU.
    Candidates {
        /// Model layout JSON (`tetraeder plan layout`).
        #[arg(long)]
        layout: PathBuf,
        /// Free VRAM for the engine, bytes.
        #[arg(long)]
        gpu_bytes: u64,
        #[arg(long)]
        ram_bytes: u64,
        /// CPU thread quota.
        #[arg(long)]
        threads: u32,
        #[arg(long, default_value_t = 2048)]
        prompt: u32,
        #[arg(long, default_value_t = 128)]
        r#gen: u32,
        /// Model key in the spec.
        #[arg(long)]
        model_key: String,
        /// `org/name:file.gguf`.
        #[arg(long)]
        hf: String,
        #[arg(long)]
        out_candidates: PathBuf,
        #[arg(long)]
        out_spec: PathBuf,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Candidates {
            layout,
            gpu_bytes,
            ram_bytes,
            threads,
            prompt,
            r#gen,
            model_key,
            hf,
            out_candidates,
            out_spec,
        } => {
            let text = std::fs::read_to_string(&layout)
                .with_context(|| format!("read {}", layout.display()))?;
            let layout: ModelLayout = serde_json::from_str(&text).context("parse layout")?;
            let budget = Budget {
                gpu_bytes,
                ram_bytes,
                threads,
            };
            let workload = Workload { prompt, r#gen };
            let en = enumerate(&budget, &layout, &workload);
            anyhow::ensure!(
                !en.candidates.is_empty(),
                "no candidate fits: {:?}",
                en.dropped
            );
            let c = campaign::build(&model_key, &hf, &layout, &budget, &workload, &en);
            let file = CandidatesFile::new(&c, &layout, &budget, &en);
            std::fs::write(&out_candidates, serde_json::to_string_pretty(&file)?)
                .with_context(|| format!("write {}", out_candidates.display()))?;
            std::fs::write(&out_spec, c.to_toml())
                .with_context(|| format!("write {}", out_spec.display()))?;
            println!(
                "{} candidates, {} anchors, {} baselines, {} dropped",
                file.candidates.len(),
                file.anchors.len(),
                file.baselines.len(),
                file.dropped.len()
            );
            Ok(())
        }
    }
}
