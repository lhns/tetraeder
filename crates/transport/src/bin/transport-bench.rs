//! Transport spike (ADR 0023): round-trip latency and throughput over TCP, TCP+TLS and QUIC.
//!
//!   transport-bench local                                  # all three on loopback, one process
//!   transport-bench server --addr 0.0.0.0:7000             # on box 1
//!   transport-bench client --addr <box1-ip>:7000           # on box 2
//!
//! With `--transport all`, one base port P serves tcp on TCP P, tls on TCP P+1 and quic on UDP P+2.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use std::net::SocketAddr;
use std::path::PathBuf;
use tetraeder_transport::bench::{self, Plan, Row, Transport};

#[derive(Parser)]
struct Cli {
    /// TLS/QUIC cipher preference. `chacha` helps on hosts without AES-NI.
    #[arg(long, value_enum, default_value = "aes", global = true)]
    cipher: Cipher,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Clone, Copy, ValueEnum)]
enum Cipher {
    Aes,
    Chacha,
}

#[derive(Clone, Copy, ValueEnum)]
enum Which {
    All,
    Tcp,
    Tls,
    Quic,
}

impl Which {
    fn list(self) -> Vec<Transport> {
        match self {
            Which::All => Transport::ALL.to_vec(),
            Which::Tcp => vec![Transport::Tcp],
            Which::Tls => vec![Transport::Tls],
            Which::Quic => vec![Transport::Quic],
        }
    }
}

#[derive(clap::Args)]
struct Measure {
    /// Message sizes in bytes: 8 KB is an activation hop, 1 MB a bulk transfer.
    #[arg(long, value_delimiter = ',', default_values_t = [8192usize, 1048576])]
    sizes: Vec<usize>,
    /// Round trips measured per size (a fifth of that for sizes of 256 KB and up).
    #[arg(long, default_value_t = 2000)]
    iters: usize,
    #[arg(long, default_value_t = 200)]
    warmup: usize,
    /// Bytes streamed one way per throughput measurement.
    #[arg(long, default_value_t = 256 << 20)]
    throughput_bytes: usize,
    /// Write the rows as JSON.
    #[arg(long)]
    json: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run servers until killed.
    Server {
        #[arg(long, default_value = "0.0.0.0:7000")]
        addr: SocketAddr,
        #[arg(long, value_enum, default_value = "all")]
        transport: Which,
    },
    /// Measure against servers started with `server`.
    Client {
        #[arg(long)]
        addr: SocketAddr,
        #[arg(long, value_enum, default_value = "all")]
        transport: Which,
        #[command(flatten)]
        m: Measure,
    },
    /// Servers and client in one process on 127.0.0.1.
    Local {
        #[arg(long, value_enum, default_value = "all")]
        transport: Which,
        #[command(flatten)]
        m: Measure,
    },
}

/// Address of `t` for base address `base` (see the module doc for the port layout).
fn addr_of(base: SocketAddr, which: Which, t: Transport) -> SocketAddr {
    let offset = match (which, t) {
        (Which::All, Transport::Tls) => 1,
        (Which::All, Transport::Quic) => 2,
        _ => 0,
    };
    SocketAddr::new(base.ip(), base.port() + offset)
}

fn report(rows: &[Row], m: &Measure) -> Result<()> {
    print!("{}", bench::table(rows));
    if let Some(p) = &m.json {
        std::fs::write(p, serde_json::to_string_pretty(rows)? + "\n")
            .with_context(|| p.display().to_string())?;
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    tetraeder_transport::tls::prefer_chacha(matches!(cli.cipher, Cipher::Chacha));
    match cli.cmd {
        Cmd::Server { addr, transport } => {
            let mut tasks = Vec::new();
            for t in transport.list() {
                let s = bench::bind(t, addr_of(addr, transport, t)).await?;
                eprintln!("{} listening on {}", t.name(), s.addr);
                tasks.push(tokio::spawn(s.run()));
            }
            for t in tasks {
                t.await??;
            }
        }
        Cmd::Client { addr, transport, m } => {
            let plan = Plan {
                sizes: m.sizes.clone(),
                iters: m.iters,
                warmup: m.warmup,
                throughput_bytes: m.throughput_bytes,
            };
            let mut rows = Vec::new();
            for t in transport.list() {
                rows.extend(bench::measure(t, addr_of(addr, transport, t), &plan).await?);
            }
            report(&rows, &m)?;
        }
        Cmd::Local { transport, m } => {
            let plan = Plan {
                sizes: m.sizes.clone(),
                iters: m.iters,
                warmup: m.warmup,
                throughput_bytes: m.throughput_bytes,
            };
            let mut rows = Vec::new();
            for t in transport.list() {
                let s = bench::bind(t, "127.0.0.1:0".parse()?).await?;
                let addr = s.addr;
                let task = tokio::spawn(s.run());
                rows.extend(bench::measure(t, addr, &plan).await?);
                task.abort();
            }
            report(&rows, &m)?;
        }
    }
    Ok(())
}
