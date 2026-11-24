//! `tetraeder-plan`: `candidates` enumerates the candidate plans of a fixture and writes the
//! candidate set and the bench spec that measures it (ADR 0040); `explain`, `diff` and `what-if`
//! report on the search (docs/plan.md "Explain, diff, what-if").

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use std::collections::BTreeMap;
use std::path::PathBuf;
use tetraeder_planner::multi_cli;
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
    /// Enumerate candidates for one GPU plus the CPU, or with `--multi` the fixed Phase 5 set of
    /// the GPUs in `--hw` (ADR 0068, 0070).
    Candidates {
        /// Model layout JSON (`tetraeder plan layout`).
        #[arg(long)]
        layout: PathBuf,
        /// Free VRAM for the engine, bytes.
        #[arg(long, required_unless_present = "multi")]
        gpu_bytes: Option<u64>,
        #[arg(long, required_unless_present = "multi")]
        ram_bytes: Option<u64>,
        /// CPU thread quota.
        #[arg(long, required_unless_present = "multi")]
        threads: Option<u32>,
        #[arg(long, default_value_t = 2048)]
        prompt: u32,
        #[arg(long, default_value_t = 128)]
        r#gen: u32,
        /// Model key in the spec.
        #[arg(long, required_unless_present = "multi")]
        model_key: Option<String>,
        /// `org/name:file.gguf`.
        #[arg(long, required_unless_present = "multi")]
        hf: Option<String>,
        /// Engine the candidates are for.
        #[arg(long, value_enum, default_value_t = Engine::LlamaBench)]
        engine: Engine,
        #[arg(long, required_unless_present = "multi")]
        out_candidates: Option<PathBuf>,
        /// Bench spec; required for llama-bench, ignored for the builder.
        #[arg(long)]
        out_spec: Option<PathBuf>,
        /// The multi-GPU set: single, layer split, tensor, expert and tensor-times-pipeline
        /// parallel (needs `--hw` and `--out`).
        #[arg(long, requires_all = ["hw", "out"])]
        multi: bool,
        /// Hardware graph (`hw.json`, or TOML) of the box.
        #[arg(long)]
        hw: Option<PathBuf>,
        /// `candidates.json` of `--multi`.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Directory for `<model>-<id>.toml`, one plan per candidate.
        #[arg(long)]
        plans_dir: Option<PathBuf>,
        /// Write only the plan of this calibration anchor, such as `layer-4@L24` (ADR 0082),
        /// instead of the box's fixed set. Repeatable.
        #[arg(long, requires = "multi")]
        anchor: Vec<String>,
    },
    /// Fit the cost model to the anchor runs of a Phase 5 session (ADR 0068): `c_range` and
    /// `c_step` included. Reads `anchors-<model>-r<k>.json` and `<model>.json` layouts.
    Calibrate {
        #[arg(long)]
        hw: PathBuf,
        /// `calib-comm.json` of `tetraeder comm-probe`, merged into the graph.
        #[arg(long)]
        comm: Option<PathBuf>,
        /// `decode-clocks.json` of `scripts/decode_clocks.py` (ADR 0082): the clocks cost model 5
        /// prices decode at. Without it the graph has none and predicts as cost model 4.
        #[arg(long)]
        clocks: Option<PathBuf>,
        #[arg(long)]
        layout_dir: PathBuf,
        #[arg(long)]
        anchors: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Predicted `pp_tps`, `tg_tps` and `T` of the candidates of `candidates --multi`, written
    /// before the first candidate runs.
    Predict {
        #[arg(long)]
        hw: PathBuf,
        /// `calib-comm.json`; merged into the graph and hashed into the file.
        #[arg(long)]
        comm: Option<PathBuf>,
        #[arg(long)]
        layout: PathBuf,
        /// `calib.json` of `calibrate`; without it the graph itself is used.
        #[arg(long)]
        calib: Option<PathBuf>,
        #[arg(long)]
        candidates: PathBuf,
        /// `routing-<model>.json` of `tetraeder bench --routing-out`: expert parallel priced at
        /// the recorded routing (ADR 0070); hashed into the file. Without it, uniform routing.
        #[arg(long)]
        routing: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
    },
    /// The planner's pick and predicted `pp_tps` and `tg_tps` for the whole model on the one node
    /// of `--hw` (builder placement, uniform routing prior or decode from a routing replay), with
    /// the `exit_regime` environment that runs it. Written before the measured run (tree node
    /// `bench-235b-stream-predict`).
    NodePredict {
        #[arg(long)]
        hw: PathBuf,
        /// Model layout JSON (`tetraeder plan layout`).
        #[arg(long)]
        layout: PathBuf,
        #[arg(long, default_value_t = 2048)]
        prompt: u32,
        #[arg(long, default_value_t = 128)]
        r#gen: u32,
        /// Price only this `VRAM_SLOTS,RAM_SLOTS` pair instead of picking.
        #[arg(long)]
        pin: Option<String>,
        /// `tetraeder-bench replay` output: decode is priced from its `lru` outcome with prefetch.
        #[arg(long)]
        tier_replay: Option<PathBuf>,
        /// sha256 of the routing trace behind `--tier-replay`, recorded with the prediction.
        #[arg(long)]
        trace_sha256: Option<String>,
        /// Refuse anything but a prediction priced from a routing trace (ADR 0092): a judged
        /// prediction without `--tier-replay` exits non-zero.
        #[arg(long)]
        judged: bool,
        #[arg(long)]
        out: PathBuf,
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
        Command::NodePredict {
            hw,
            layout,
            prompt,
            r#gen,
            pin,
            tier_replay,
            trace_sha256,
            judged,
            out,
        } => {
            let routing = if tier_replay.is_some() {
                tetraeder_planner::RoutingSource::Trace
            } else {
                tetraeder_planner::RoutingSource::Uniform
            };
            if judged {
                routing.check_judged().map_err(anyhow::Error::msg)?;
            }
            let hw_text = read(&hw)?;
            let graph = multi_cli::parse_hw(&hw_text).map_err(anyhow::Error::msg)?;
            let layout_text = read(&layout)?;
            let layout = multi_cli::parse_layout(&layout_text).map_err(anyhow::Error::msg)?;
            let pin = pin
                .map(|p| {
                    let (v, r) = p.split_once(',').context("--pin is VRAM_SLOTS,RAM_SLOTS")?;
                    anyhow::Ok((v.trim().parse::<u32>()?, r.trim().parse::<u32>()?))
                })
                .transpose()?;
            let replay = match &tier_replay {
                Some(p) => {
                    Some(tetraeder_planner::read_tier_replay(p).map_err(anyhow::Error::msg)?)
                }
                None => None,
            };
            let w = Workload { prompt, r#gen };
            let (runs, dropped) =
                tetraeder_planner::single_node_runs(&graph, &layout, &w, pin, replay.as_deref())
                    .map_err(anyhow::Error::msg)?;
            let pick = runs.first().context("no placement fits the node")?;
            let row = tetraeder_planner::builder::slot_row_bytes(&layout);
            let env: BTreeMap<String, String> = BTreeMap::from([
                ("TETRAEDER_EXIT_GPU".into(), "1".into()),
                (
                    "TETRAEDER_EXIT_VRAM_SLOT_BYTES".into(),
                    (u64::from(pick.vram_slots) * row).to_string(),
                ),
                (
                    "TETRAEDER_EXIT_RAM_SLOT_BYTES".into(),
                    (u64::from(pick.ram_slots) * row).to_string(),
                ),
                ("TETRAEDER_EXIT_UBATCH".into(), pick.ubatch.to_string()),
                ("TETRAEDER_EXIT_OP_OFFLOAD".into(), "1".into()),
                ("TETRAEDER_EXIT_PREFILL".into(), prompt.to_string()),
                ("TETRAEDER_EXIT_DECODE".into(), r#gen.to_string()),
            ]);
            let doc = serde_json::json!({
                "hw_sha256": multi_cli::sha256_hex(hw_text.as_bytes()),
                "layout_sha256": multi_cli::sha256_hex(layout_text.as_bytes()),
                "routing": routing.name(),
                "routing_note": match &tier_replay {
                    Some(p) => format!("trace replay (decode), uniform prefill: {}", p.display()),
                    None => "uniform".to_string(),
                },
                "routing_trace": match &tier_replay {
                    Some(p) => {
                        let v: serde_json::Value = serde_json::from_str(&read(p)?)
                            .with_context(|| format!("parse {}", p.display()))?;
                        serde_json::json!({
                            "sha256": trace_sha256,
                            "fnv1a64": v["trace"]["fnv1a64"],
                            "replay_version": v["replay_version"],
                            "scope": v["scope"],
                        })
                    }
                    None => serde_json::Value::Null,
                },
                "judged": judged,
                "workload": {"prompt": prompt, "gen": r#gen},
                "pinned": pin.is_some(),
                "pick": pick,
                "exit_regime_env": env,
                "runners_up": runs.iter().skip(1).take(5).collect::<Vec<_>>(),
                "dropped": dropped,
            });
            std::fs::write(&out, serde_json::to_string_pretty(&doc)?)
                .with_context(|| format!("write {}", out.display()))?;
            println!(
                "v{} r{} ub{}: pp {:.2} tok/s, tg {:.3} tok/s, binding {} -> {}",
                pick.vram_slots,
                pick.ram_slots,
                pick.ubatch,
                pick.pp_tps,
                pick.tg_tps,
                pick.binding,
                out.display()
            );
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
        Command::Calibrate {
            hw,
            comm,
            clocks,
            layout_dir,
            anchors,
            out,
        } => {
            let mut hw = multi_cli::parse_hw(&read(&hw)?).map_err(anyhow::Error::msg)?;
            if let Some(c) = &comm {
                multi_cli::merge_comm(&mut hw, &read(c)?).map_err(anyhow::Error::msg)?;
            }
            if let Some(c) = &clocks
                && !multi_cli::merge_decode_clocks(&mut hw, &read(c)?)
                    .map_err(anyhow::Error::msg)?
            {
                eprintln!(
                    "decode clocks: {} withholds low_mhz, cost model 4 rates",
                    c.display()
                );
            }
            let mut by_model: BTreeMap<String, Vec<String>> = BTreeMap::new();
            for e in std::fs::read_dir(&anchors).with_context(|| anchors.display().to_string())? {
                let path = e?.path();
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if let Some(model) = multi_cli::anchor_model(name) {
                    by_model.entry(model).or_default().push(read(&path)?);
                }
            }
            anyhow::ensure!(
                !by_model.is_empty(),
                "no anchors-<model>-r<k>.json in {}",
                anchors.display()
            );
            let mut models = Vec::new();
            for (model, files) in by_model {
                let lp = layout_dir.join(format!("{model}.json"));
                let layout = multi_cli::parse_layout(&read(&lp)?).map_err(anyhow::Error::msg)?;
                models.push((layout, files));
            }
            let fit = multi_cli::calibrate(&hw, &models).map_err(anyhow::Error::msg)?;
            std::fs::write(&out, serde_json::to_string_pretty(&fit)?)
                .with_context(|| format!("write {}", out.display()))?;
            println!(
                "{} anchors, band {:.3} -> {}",
                fit.anchors.len(),
                fit.band,
                out.display()
            );
            Ok(())
        }
        Command::Predict {
            hw,
            comm,
            layout,
            calib,
            candidates,
            routing,
            out,
        } => {
            let hw_text = read(&hw)?;
            let mut graph = multi_cli::parse_hw(&hw_text).map_err(anyhow::Error::msg)?;
            let comm_sha = match &comm {
                Some(c) => {
                    let text = read(c)?;
                    multi_cli::merge_comm(&mut graph, &text).map_err(anyhow::Error::msg)?;
                    Some(multi_cli::sha256_hex(text.as_bytes()))
                }
                None => None,
            };
            let layout = multi_cli::parse_layout(&read(&layout)?).map_err(anyhow::Error::msg)?;
            let routing = match &routing {
                Some(p) => {
                    let text = read(p)?;
                    let r = multi_cli::parse_routing(&text, &layout)
                        .map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))?;
                    Some((r, multi_cli::sha256_hex(text.as_bytes())))
                }
                None => None,
            };
            let cands: multi_cli::MultiFile =
                serde_json::from_str(&read(&candidates)?).context("parse candidates")?;
            let (calibrated, calib_sha) = match calib {
                Some(p) => {
                    let text = read(&p)?;
                    let c: multi_cli::CalibFile =
                        serde_json::from_str(&text).context("parse calib")?;
                    (c.hw, Some(multi_cli::sha256_hex(text.as_bytes())))
                }
                None => (graph, None),
            };
            let commit = std::env::var("TETRAEDER_GIT_COMMIT")
                .ok()
                .unwrap_or_else(|| {
                    std::process::Command::new("git")
                        .args(["rev-parse", "HEAD"])
                        .output()
                        .ok()
                        .and_then(|o| String::from_utf8(o.stdout).ok())
                        .map_or_else(|| "unknown".to_string(), |s| s.trim().to_string())
                });
            let file = multi_cli::predict_multi(
                &calibrated,
                &layout,
                &cands,
                (
                    multi_cli::sha256_hex(hw_text.as_bytes()),
                    calib_sha,
                    comm_sha,
                ),
                routing.as_ref().map(|(r, sha)| (r, sha.clone())),
                &commit,
            )
            .map_err(anyhow::Error::msg)?;
            std::fs::write(&out, serde_json::to_string_pretty(&file)?)
                .with_context(|| format!("write {}", out.display()))?;
            for r in &file.candidates {
                println!(
                    "{:<10} pp {:8.1} tg {:7.2} T {:7.2} s",
                    r.id, r.pp_tps, r.tg_tps, r.t_s
                );
            }
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
            multi,
            hw,
            out,
            plans_dir,
            anchor,
        } => {
            if multi {
                let (Some(hw), Some(out)) = (hw, out) else {
                    anyhow::bail!("--multi needs --hw and --out");
                };
                let graph = multi_cli::parse_hw(&read(&hw)?).map_err(anyhow::Error::msg)?;
                let layout =
                    multi_cli::parse_layout(&read(&layout)?).map_err(anyhow::Error::msg)?;
                let file = if anchor.is_empty() {
                    multi_cli::candidates_multi(&graph, &layout)
                } else {
                    multi_cli::anchors_multi(&graph, &layout, &anchor)
                        .map_err(anyhow::Error::msg)?
                };
                std::fs::write(&out, serde_json::to_string_pretty(&file)?)
                    .with_context(|| format!("write {}", out.display()))?;
                if let Some(dir) = plans_dir {
                    std::fs::create_dir_all(&dir)?;
                    for c in &file.candidates {
                        let p = dir.join(format!("{}-{}.toml", file.model, c.id));
                        std::fs::write(&p, c.plan.to_toml()?)
                            .with_context(|| format!("write {}", p.display()))?;
                    }
                }
                for c in &file.candidates {
                    println!("{}: {}", c.id, c.notes.join("; "));
                }
                for p in &file.pruned {
                    println!("pruned {}: {}", p.id, p.reason);
                }
                return Ok(());
            }
            let (gpu_bytes, ram_bytes, threads) = (
                gpu_bytes.context("--gpu-bytes")?,
                ram_bytes.context("--ram-bytes")?,
                threads.context("--threads")?,
            );
            let (model_key, hf) = (model_key.context("--model-key")?, hf.context("--hf")?);
            let out_candidates = out_candidates.context("--out-candidates")?;
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

#[cfg(test)]
mod tests {
    use super::Cli;
    use clap::CommandFactory;

    #[test]
    fn the_command_line_is_consistent() {
        Cli::command().debug_assert();
    }
}
