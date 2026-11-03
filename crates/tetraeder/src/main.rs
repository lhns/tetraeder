//! Tetraeder command line entry point.

mod bench;
mod engine_args;
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
    /// OpenAI and Anthropic APIs over an engine.
    Serve(serve::ServeArgs),
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
        Some(Command::Serve(args)) => serve::run(&args),
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
