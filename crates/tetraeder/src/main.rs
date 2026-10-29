//! Tetraeder command line entry point.

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
}

fn main() {
    match Cli::parse().command {
        Some(Command::Top(args)) => top::run(&args),
        None => println!("tetraeder {}", env!("CARGO_PKG_VERSION")),
    }
}
