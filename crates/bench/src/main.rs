use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::process::ExitCode;
use tetraeder_bench::golden::{Golden, Tolerance};
use tetraeder_bench::noise;
use tetraeder_bench::quality::{self, ParityFile, ParityOptions};
use tetraeder_bench::replay;
use tetraeder_bench::run::{self, Options};
use tetraeder_bench::{env, spec::Spec};

#[derive(Parser)]
#[command(
    name = "tetraeder-bench",
    about = "Run a pinned benchmark spec as paired, interleaved repeats"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum NoiseCmd {
    /// Scan results for A/A pairs and rewrite `<out>/<class>.json` (idempotent; sessions already in a file stay).
    Update {
        #[arg(long, default_value = "bench/results")]
        results: PathBuf,
        #[arg(long, default_value = "bench/noise")]
        out: PathBuf,
    },
    /// Print the band a run on this class would use.
    Show {
        /// Class slug, e.g. rtx-3090 or cpu.
        class: String,
        #[arg(long, default_value = "bench/noise")]
        dir: PathBuf,
    },
}

#[derive(Subcommand)]
enum Cmd {
    /// Run a spec and write bench/results/<date>/<spec>-<HHMMSS>.json. Exit 0 ok, 1 regression,
    /// 2 error, 3 inconclusive (noise pair outside its band).
    Run {
        spec: PathBuf,
        /// Results directory (default bench/results/<YYYYMMDD>).
        #[arg(long)]
        out_dir: Option<PathBuf>,
        /// Path of llama-bench (default: the spec's `bin`, or its `bin_env` variable; boxes use /opt/llama.cpp/bin).
        #[arg(long)]
        llama_bench: Option<PathBuf>,
        /// Model cache directory, `<org>__<name>/<file>` below it.
        #[arg(long, env = "TETRAEDER_MODELS_DIR", default_value = ".hf_cache")]
        models_dir: PathBuf,
        /// Override the repeats of the spec (smoke runs only; the committed spec is the gate).
        #[arg(long)]
        repeats: Option<usize>,
        /// Run only this pair id (repeatable).
        #[arg(long = "only")]
        only: Vec<String>,
        /// Count skipped pairs as errors.
        #[arg(long)]
        strict: bool,
        /// Parity results for the quality of Tetraeder gates (default: parity.json in the results directory;
        /// a gate without one is an error).
        #[arg(long)]
        parity: Option<PathBuf>,
        /// Run only the repeats that earlier results of the same spec text in the results directory
        /// lack (a relaunch after a kill, or `--repeats 1` first and the spec's repeats later).
        #[arg(long)]
        resume: bool,
        /// Measured noise bands (`noise update` writes them); a class with enough sessions replaces the spec's
        /// provisional band. A missing directory means provisional bands only.
        #[arg(long, default_value = "bench/noise")]
        noise_dir: PathBuf,
    },
    /// Measured noise bands per device and GPU class from the A/A pairs of committed results (see `noise.rs`).
    Noise {
        #[command(subcommand)]
        cmd: NoiseCmd,
    },
    /// Greedy golden check of the Tetraeder build against llama-simple for every Tetraeder gate of a spec;
    /// writes parity.json. Exit 0 all ok, 1 outputs differ, 2 a check could not run.
    Parity {
        spec: PathBuf,
        /// Results directory (default bench/results/<YYYYMMDD>).
        #[arg(long)]
        out_dir: Option<PathBuf>,
        /// Reference binary (default $LLAMA_SIMPLE, else /opt/llama.cpp/bin/llama-simple).
        #[arg(long)]
        llama_simple: Option<PathBuf>,
        /// Tetraeder binary (default: the spec's engine).
        #[arg(long)]
        tetraeder: Option<PathBuf>,
        #[arg(long, env = "TETRAEDER_MODELS_DIR", default_value = ".hf_cache")]
        models_dir: PathBuf,
    },
    /// Score RAM-tier policies offline on a routing trace written by the exit regime
    /// (`TETRAEDER_EXIT_ROUTING_OUT`, ADR 0050 (c)); writes one JSON file.
    Replay {
        #[arg(long)]
        trace: PathBuf,
        /// VRAM slots per layer, comma separated.
        #[arg(long, value_delimiter = ',', required = true)]
        vram: Vec<u32>,
        /// RAM slots per layer, comma separated.
        #[arg(long, value_delimiter = ',', required = true)]
        ram: Vec<u32>,
        /// `lru`, `freq:<share>` or `belady`, comma separated.
        #[arg(
            long,
            value_delimiter = ',',
            default_values_t = ["lru", "freq:0.25", "freq:0.5", "freq:0.75", "freq:1", "belady"].map(String::from)
        )]
        policies: Vec<String>,
        #[arg(long)]
        out: PathBuf,
        /// Also replay every configuration on a virtual clock with this time model
        /// (`tetraeder-plan explain --time-model`, ADR 0056); adds `timed` to the output.
        #[arg(long)]
        timed: Option<PathBuf>,
    },
    /// Parse and validate a spec.
    Check { spec: PathBuf },
    /// Print the captured environment as JSON.
    Env,
    /// Compare golden data: tokens exactly, top-k logits within a tolerance.
    GoldenCompare {
        expected: PathBuf,
        actual: PathBuf,
        #[arg(long, default_value_t = 1e-3)]
        logit_tol: f32,
    },
}

fn noise_class_of(path: &std::path::Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn main() -> ExitCode {
    match real_main() {
        Ok(code) => ExitCode::from(code),
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(2)
        }
    }
}

fn real_main() -> Result<u8> {
    match Cli::parse().cmd {
        Cmd::Run {
            spec,
            out_dir,
            llama_bench,
            models_dir,
            repeats,
            only,
            strict,
            parity,
            resume,
            noise_dir,
        } => {
            let text = std::fs::read_to_string(&spec)
                .with_context(|| format!("read {}", spec.display()))?;
            let parsed = Spec::parse(&text).with_context(|| format!("spec {}", spec.display()))?;
            let dir = out_dir
                .unwrap_or_else(|| PathBuf::from("bench/results").join(env::date_dir_name()));
            // A missing or unreadable file leaves the quality of Tetraeder gates missing (an error there).
            let parity_path = parity.unwrap_or_else(|| dir.join(quality::PARITY_FILE));
            let parity = match ParityFile::load(&parity_path) {
                Ok(f) => Some(f),
                Err(e) => {
                    if parity_path.exists() {
                        eprintln!("warning: {e:#}");
                    }
                    None
                }
            };
            let done_reps = if resume {
                let earlier = run::read_results_dir(&dir, &parsed.spec.name);
                run::recorded_reps(&earlier, &run::spec_hash(&text))
            } else {
                Default::default()
            };
            // claimed before the run and rewritten after every pair
            let path = run::reserve_results_path(&parsed.spec.name, &dir)?;
            let opts = Options {
                llama_bench,
                models_dir,
                repeats,
                only,
                strict,
                quiet: false,
                parity,
                checkpoint: Some(path.clone()),
                done_reps,
                noise_dir: Some(noise_dir),
            };
            let results = run::run_spec(&parsed, &text, &opts)?;
            run::write_results_at(&results, &path)?;
            print!("{}", run::summary_text(&results));
            println!("results: {}", path.display());
            Ok(results.exit_code as u8)
        }
        Cmd::Noise { cmd } => match cmd {
            NoiseCmd::Update { results, out } => {
                let (y, mo, d, ..) = env::utc_parts(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map_or(0, |t| t.as_secs()),
                );
                let today = format!("{y:04}-{mo:02}-{d:02}");
                for p in noise::update(&results, &out, std::path::Path::new("."), &today)? {
                    let f =
                        noise::load(&out, &noise_class_of(&p))?.context("written file vanished")?;
                    println!(
                        "{}: n={} hosts={} p95 d={:.4} band={:.4}",
                        p.display(),
                        f.n,
                        f.n_hosts,
                        f.d_p95,
                        f.band
                    );
                }
                Ok(0)
            }
            NoiseCmd::Show { class, dir } => {
                match noise::load(&dir, &class)? {
                    Some(f) => println!(
                        "{class}: n={} band={:.4} ({})",
                        f.n,
                        f.band,
                        if f.n >= noise::MIN_SESSIONS {
                            "used"
                        } else {
                            "too few sessions, provisional band stays"
                        }
                    ),
                    None => println!("{class}: no measured file, provisional band"),
                }
                Ok(0)
            }
        },
        Cmd::Parity {
            spec,
            out_dir,
            llama_simple,
            tetraeder,
            models_dir,
        } => {
            let parsed = Spec::load(&spec)?;
            let dir = out_dir
                .unwrap_or_else(|| PathBuf::from("bench/results").join(env::date_dir_name()));
            let file = quality::run_parity(
                &parsed,
                &ParityOptions {
                    llama_simple,
                    tetraeder,
                    models_dir,
                    quiet: false,
                },
            );
            for c in &file.checks {
                println!(
                    "{} {:?}: {:?} {}",
                    c.model, c.device, c.quality.status, c.quality.detail
                );
            }
            let path = quality::write_parity(&file, &dir)?;
            println!("parity: {}", path.display());
            let any_error = file
                .checks
                .iter()
                .any(|c| c.quality.status == quality::QualityStatus::Error);
            Ok(if any_error {
                2
            } else {
                u8::from(!file.all_ok())
            })
        }
        Cmd::Replay {
            trace,
            vram,
            ram,
            policies,
            out,
            timed,
        } => {
            let policies = policies
                .iter()
                .map(|p| tetraeder_memory::replay::sim::Policy::parse(p))
                .collect::<Result<Vec<_>, _>>()
                .map_err(anyhow::Error::msg)?;
            let n = replay::run_timed(&trace, &vram, &ram, &policies, &out, timed.as_deref())?;
            println!("replay: {n} configurations -> {}", out.display());
            Ok(0)
        }
        Cmd::Check { spec } => {
            let s = Spec::load(&spec)?;
            println!(
                "ok: {} ({} variants, {} pairs)",
                s.spec.name,
                s.variant.len(),
                s.pair.len()
            );
            Ok(0)
        }
        Cmd::Env => {
            println!("{}", serde_json::to_string_pretty(&env::capture())?);
            Ok(0)
        }
        Cmd::GoldenCompare {
            expected,
            actual,
            logit_tol,
        } => {
            let c = Golden::load(&expected)?.compare(
                &Golden::load(&actual)?,
                Tolerance {
                    logit_abs: logit_tol,
                },
            );
            println!(
                "{} positions, max logit diff {:e}, {} mismatches",
                c.positions,
                c.max_logit_diff,
                c.mismatches.len()
            );
            for m in c.mismatches.iter().take(20) {
                println!("  {m:?}");
            }
            Ok(u8::from(!c.ok()))
        }
    }
}
