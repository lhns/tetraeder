//! `tetraeder-plan`: `candidates` enumerates the candidate plans of a fixture and writes the
//! candidate set and the bench spec that measures it (ADR 0040); `explain`, `diff` and `what-if`
//! report on the search (docs/plan.md "Explain, diff, what-if").

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use tetraeder_planner::{
    Budget, CandidatesFile, ModelLayout, Workload, campaign, cli, enumerate, enumerate_builder,
};

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// What runs the candidates.
#[derive(Clone, Copy, ValueEnum)]
enum Engine {
    /// llama-bench through the bench harness (a spec is written).
    LlamaBench,
    /// The own MoE builder through the `exit_regime` test binary (expert slot counts per tier).
    Builder,
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
        /// Engine the candidates are for.
        #[arg(long, value_enum, default_value_t = Engine::LlamaBench)]
        engine: Engine,
        #[arg(long)]
        out_candidates: PathBuf,
        /// Bench spec; required for llama-bench, ignored for the builder.
        #[arg(long)]
        out_spec: Option<PathBuf>,
    },
    /// The `candidates.json` of a ledger directory (ADR 0056): the builder anchors plus the
    /// session's builder stages as candidates with their own request.
    LedgerDir {
        #[arg(long)]
        layout: PathBuf,
        /// Free VRAM for the engine, bytes.
        #[arg(long)]
        gpu_bytes: u64,
        #[arg(long)]
        ram_bytes: u64,
        #[arg(long)]
        threads: u32,
        /// The anchors' request.
        #[arg(long, default_value_t = 512)]
        prompt: u32,
        #[arg(long, default_value_t = 32)]
        r#gen: u32,
        #[arg(long)]
        model_key: String,
        #[arg(long)]
        hf: String,
        /// `id=VRAM_BYTES,RAM_BYTES,THREADS,UBATCH,OP_OFFLOAD,PROMPT,GEN`, repeatable.
        #[arg(long = "stage")]
        stages: Vec<cli::StageSpec>,
        #[arg(long)]
        out: PathBuf,
    },
    /// Why the search chose its plan: stages, binding resource, runner-up, formats, front.
    Explain {
        #[command(flatten)]
        common: Common,
        /// Hints TOML; none means no limits.
        #[arg(long)]
        hints: Option<PathBuf>,
        #[arg(long)]
        json: bool,
        /// Instead of the search: write the time model of a builder candidate for
        /// `tetraeder-bench replay --timed` (ADR 0056); needs `--fixture` and `--candidate`.
        #[arg(long, requires_all = ["fixture", "candidate"])]
        time_model: Option<PathBuf>,
        /// Builder fixture directory (layout.json, candidates.json, tier-replay.json).
        #[arg(long)]
        fixture: Option<PathBuf>,
        /// Candidate (or anchor) id in the fixture.
        #[arg(long)]
        candidate: Option<String>,
    },
    /// What moves between two plan files and how long the switch takes.
    Diff {
        #[arg(long)]
        hw: PathBuf,
        #[arg(long)]
        model: PathBuf,
        old: PathBuf,
        new: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// Predicted speed with hardware you do not have. With no --add and --remove, ranks a fixed
    /// list of single purchases.
    WhatIf {
        #[command(flatten)]
        common: Common,
        #[arg(long)]
        hints: Option<PathBuf>,
        /// `gpu:4090`, `ram:256G`, `nvme:gen4`, `net:25gbe`; repeatable.
        #[arg(long)]
        add: Vec<String>,
        /// A device id of the hardware file; repeatable.
        #[arg(long)]
        remove: Vec<String>,
        #[arg(long)]
        json: bool,
    },
    /// Formats per quality group (or per tensor) within a byte budget, from a quality profile
    /// (ADR 0048). Writes the `--mix` file of scripts/quality/compare_mix.py.
    Mix {
        /// `bench/quality/<model>.quality.json`.
        #[arg(long)]
        profile: PathBuf,
        #[arg(long)]
        budget_bytes: u64,
        /// Tensor names of the model (JSON array of names or of objects with `name`); the output
        /// is then `{tensor: format}`, else `{group id: format}`.
        #[arg(long)]
        tensors: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
    },
    /// Score a measured fixture directory (ADR 0040): calibrate on its anchors, predict the
    /// candidates, report regret and rank correlation as JSON.
    Score {
        /// `bench/results/<date>/planner-<fixture>`.
        dir: PathBuf,
        /// Write the JSON here instead of stdout.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Calibration ledger (ADR 0056): re-predict every committed fixture with its session's fit and
    /// print the error per session. `--write` runs the ratchet and rewrites `bench/ledger.json` and
    /// each session's `calib.json`; run it after a cost model version bump or a new fixture.
    Ledger {
        /// Results root; the ledger is `ledger.json` beside it.
        #[arg(long, default_value = "bench/results")]
        root: PathBuf,
        #[arg(long)]
        write: bool,
        /// `id=reason` (or `path/id=reason`): let this result's error rise past the ratchet.
        #[arg(long)]
        accept_worse: Vec<String>,
    },
}

/// Inputs every planning subcommand reads.
#[derive(clap::Args)]
struct Common {
    /// Hardware graph TOML.
    #[arg(long)]
    hw: PathBuf,
    /// Model description TOML.
    #[arg(long)]
    model: PathBuf,
    /// Roadmap phase of the executing engine.
    #[arg(long, default_value_t = 3)]
    phase: u8,
    /// Plan for a Linux box (tensor and expert parallel need NCCL).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    linux: bool,
}

fn read(p: &std::path::Path) -> Result<String> {
    std::fs::read_to_string(p).with_context(|| format!("read {}", p.display()))
}

fn read_opt(p: &Option<PathBuf>) -> Result<Option<String>> {
    p.as_deref().map(read).transpose()
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Ledger {
            root,
            write,
            accept_worse,
        } => {
            let t = tetraeder_planner::ledger::ledger_cmd(&root, write, &accept_worse)
                .map_err(anyhow::Error::msg)?;
            print!("{t}");
            Ok(())
        }
        Command::Score { dir, out } => {
            let s = tetraeder_planner::score::score_dir(&dir).map_err(anyhow::Error::msg)?;
            let json = serde_json::to_string_pretty(&s)?;
            match out {
                Some(p) => {
                    std::fs::write(&p, json).with_context(|| format!("write {}", p.display()))
                }
                None => {
                    println!("{json}");
                    Ok(())
                }
            }
        }
        Command::Mix {
            profile,
            budget_bytes,
            tensors,
            out,
        } => {
            let (json, summary) = cli::mix_cmd(
                &read(&profile)?,
                budget_bytes,
                read_opt(&tensors)?.as_deref(),
            )?;
            std::fs::write(&out, json).with_context(|| format!("write {}", out.display()))?;
            print!("{summary}");
            Ok(())
        }
        Command::LedgerDir {
            layout,
            gpu_bytes,
            ram_bytes,
            threads,
            prompt,
            r#gen,
            model_key,
            hf,
            stages,
            out,
        } => {
            let text = read(&layout)?;
            let layout: ModelLayout = serde_json::from_str(&text).context("parse layout")?;
            let budget = Budget {
                gpu_bytes,
                ram_bytes,
                threads,
            };
            let file = cli::ledger_candidates(
                &layout,
                &budget,
                &Workload { prompt, r#gen },
                (&model_key, &hf),
                &stages,
            );
            std::fs::write(&out, serde_json::to_string_pretty(&file)?)
                .with_context(|| format!("write {}", out.display()))?;
            println!(
                "{} anchors, {} stages -> {}",
                file.anchors.len(),
                file.candidates.len(),
                out.display()
            );
            Ok(())
        }
        Command::Explain {
            common,
            hints,
            json,
            time_model,
            fixture,
            candidate,
        } => {
            if let (Some(out), Some(fixture), Some(candidate)) = (time_model, fixture, candidate) {
                let text = cli::time_model_cmd(&read(&common.hw)?, &fixture, &candidate)?;
                std::fs::write(&out, text).with_context(|| format!("write {}", out.display()))?;
                println!("time model of {candidate} -> {}", out.display());
                return Ok(());
            }
            let out = cli::explain_cmd(
                &read(&common.hw)?,
                &read(&common.model)?,
                read_opt(&hints)?.as_deref(),
                common.phase,
                common.linux,
                json,
            )?;
            print!("{out}");
            Ok(())
        }
        Command::Diff {
            hw,
            model,
            old,
            new,
            json,
        } => {
            let out = cli::diff_cmd(
                &read(&hw)?,
                &read(&model)?,
                &read(&old)?,
                &read(&new)?,
                json,
            )?;
            print!("{out}");
            Ok(())
        }
        Command::WhatIf {
            common,
            hints,
            add,
            remove,
            json,
        } => {
            let out = cli::what_if_cmd(
                &read(&common.hw)?,
                &read(&common.model)?,
                read_opt(&hints)?.as_deref(),
                common.phase,
                common.linux,
                &add,
                &remove,
                json,
            )?;
            print!("{out}");
            Ok(())
        }
        Command::Candidates {
            layout,
            gpu_bytes,
            ram_bytes,
            threads,
            prompt,
            r#gen,
            model_key,
            hf,
            engine,
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
            let en = match engine {
                Engine::LlamaBench => enumerate(&budget, &layout, &workload),
                Engine::Builder => enumerate_builder(&budget, &layout, &workload),
            };
            anyhow::ensure!(
                !en.candidates.is_empty(),
                "no candidate fits: {:?}",
                en.dropped
            );
            let c = match engine {
                Engine::LlamaBench => {
                    campaign::build(&model_key, &hf, &layout, &budget, &workload, &en)
                }
                Engine::Builder => {
                    campaign::build_builder(&model_key, &hf, &layout, &budget, &workload, &en)
                }
            };
            let file = CandidatesFile::new(&c, &layout, &budget, &en);
            std::fs::write(&out_candidates, serde_json::to_string_pretty(&file)?)
                .with_context(|| format!("write {}", out_candidates.display()))?;
            match (engine, out_spec) {
                (Engine::LlamaBench, Some(p)) => std::fs::write(&p, c.to_toml())
                    .with_context(|| format!("write {}", p.display()))?,
                (Engine::LlamaBench, None) => {
                    anyhow::bail!("--out-spec is required for llama-bench")
                }
                (Engine::Builder, _) => {}
            }
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
