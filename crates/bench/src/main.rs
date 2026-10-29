use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::process::ExitCode;
use tetraeder_bench::golden::{Golden, Tolerance};
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
        } => {
            let text = std::fs::read_to_string(&spec)
                .with_context(|| format!("read {}", spec.display()))?;
            let parsed = Spec::parse(&text).with_context(|| format!("spec {}", spec.display()))?;
            let opts = Options {
                llama_bench,
                models_dir,
                repeats,
                only,
                strict,
                quiet: false,
            };
            let results = run::run_spec(&parsed, &text, &opts)?;
            let dir = out_dir
                .unwrap_or_else(|| PathBuf::from("bench/results").join(env::date_dir_name()));
            let path = run::write_results(&results, &dir)?;
            print!("{}", run::summary_text(&results));
            println!("results: {}", path.display());
            Ok(results.exit_code as u8)
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
