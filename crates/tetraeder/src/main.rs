//! Tetraeder command line entry point.

mod advise;
mod bench;
mod detector_score;
mod doctor;
mod engine_args;
mod info;
mod plan;
mod quant;
mod run;
mod serve;
mod top;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Terminal view of host and GPU metrics and step statistics.
    Top(top::TopArgs),
    /// Generate a completion with libllama (Phase 1 engine), streamed to stdout.
    Run(run::RunArgs),
    /// llama-bench's pp and tg tests through the engine; prints llama-bench JSON rows.
    Bench(bench::BenchArgs),
    /// Quantization tools: convert tensor formats and export a GGUF.
    #[command(subcommand)]
    Quant(quant::QuantCommand),
    /// Planner inputs and tools.
    Plan(plan::PlanArgs),
    /// Detector report of an observation file (faults, limits, verdict).
    Doctor(doctor::DoctorArgs),
    /// Score a detector campaign directory against its manifest (Phase 3a exit test).
    DetectorScore(detector_score::ScoreArgs),
    /// Ranked improvement actions for an observation file; report only.
    Advise(advise::AdviseArgs),
    /// OpenAI and Anthropic APIs over an engine.
    Serve(serve::ServeArgs),
    /// The CPU backend this binary loaded: variant, ISA, module directory.
    Info(info::InfoArgs),
}

fn main() -> std::process::ExitCode {
    let res = match Cli::parse().command {
        Some(Command::Top(args)) => {
            top::run(&args);
            Ok(())
        }
        Some(Command::Run(args)) => run::run(&args),
        Some(Command::Bench(args)) => bench::run(&args),
        Some(Command::Quant(cmd)) => quant::run(&cmd),
        Some(Command::Plan(args)) => plan::run(&args),
        Some(Command::Doctor(args)) => doctor::run(&args),
        Some(Command::DetectorScore(args)) => detector_score::run(&args),
        Some(Command::Advise(args)) => advise::run(&args),
        Some(Command::Serve(args)) => serve::run(&args),
        Some(Command::Info(args)) => info::run(&args),
        None => {
            println!("tetraeder {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
    };
    match res {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            std::process::ExitCode::FAILURE
        }
    }
}
