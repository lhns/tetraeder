//! `tetraeder cluster ...`: tooling for a multi-node cluster (ADR 0091).

use crate::node::{ClusterArgs, parse_buffers};
use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use std::path::PathBuf;
use std::sync::Arc;
use tetraeder_control::client::{self, PROBE_SIZES, ProbeResult};
use tetraeder_control::membership::NodeState;
use tetraeder_control::monitor::Monitor;
use tetraeder_telemetry::{Telemetry, TelemetryConfig};
use tetraeder_transport::identity::{generate_self_signed, write_private};
use tetraeder_transport::{NodeId, SocketBuffers};

#[derive(Subcommand)]
pub enum ClusterCommand {
    /// Make this node's key pair and self-signed certificate. Run it on the box: the key is
    /// written there and never leaves it. Put the certificate (`cert.pem`) in `cluster.toml` as
    /// the node's anchor on every node.
    Cert(CertArgs),
    /// Merge the hardware graphs of every node into one cluster graph (ids `<node>/<id>`, `node`
    /// set), with the links and curves of a probe when given. Run it with the head's key.
    Profile(ProfileArgs),
    /// Time frames of 4 KiB to 64 MiB between every pair of nodes, both directions, through the
    /// nodes' own transports, and write the curves and links.
    Probe(ProbeArgs),
    /// Run the head's membership monitor and print every node transition (metrics included).
    Watch(WatchArgs),
}

#[derive(Args)]
pub struct ProfileArgs {
    #[command(flatten)]
    pub cluster: ClusterArgs,
    /// The output of `cluster probe`: adds its links and curves.
    #[arg(long)]
    pub probe: Option<PathBuf>,
    /// Output file (stdout when absent).
    #[arg(short, long)]
    pub out: Option<PathBuf>,
}

#[derive(Args)]
pub struct ProbeArgs {
    #[command(flatten)]
    pub cluster: ClusterArgs,
    /// Socket buffers of the probing connections: `auto` or `SEND,RECV` bytes. Run twice to
    /// compare kernel autotuning with explicit sizes.
    #[arg(long, default_value = "auto", value_parser = parse_buffers)]
    pub buffers: SocketBuffers,
    /// Round trips per size (the median is kept).
    #[arg(long, default_value_t = 20)]
    pub repeats: u32,
    /// Largest frame in bytes; sizes are powers of 4 from 4 KiB.
    #[arg(long, default_value_t = 64 << 20)]
    pub max_bytes: u64,
    #[arg(short, long)]
    pub out: PathBuf,
}

#[derive(Args)]
pub struct WatchArgs {
    #[command(flatten)]
    pub cluster: ClusterArgs,
    /// The output of `cluster probe`: its small-frame round trips feed the link-slow alert.
    #[arg(long)]
    pub probe: Option<PathBuf>,
}

#[derive(Args)]
pub struct CertArgs {
    /// The node id, the `tetraeder-node:<id>` name in the certificate.
    #[arg(long)]
    pub node: String,
    /// Directory for `key.pem` (owner-only) and `cert.pem`; created if missing.
    #[arg(long)]
    pub out: PathBuf,
    /// Replace existing files (a node that changes its key must be re-pinned everywhere).
    #[arg(long)]
    pub force: bool,
}

pub fn run(cmd: &ClusterCommand) -> Result<()> {
    match cmd {
        ClusterCommand::Cert(a) => cert(a),
        ClusterCommand::Profile(a) => profile(a),
        ClusterCommand::Probe(a) => probe(a),
        ClusterCommand::Watch(a) => watch(a),
    }
}

fn write_out(out: Option<&PathBuf>, text: &str) -> Result<()> {
    match out {
        Some(p) => std::fs::write(p, text).with_context(|| format!("write {}", p.display())),
        None => {
            println!("{text}");
            Ok(())
        }
    }
}

fn read_probe(p: &PathBuf) -> Result<ProbeResult> {
    serde_json::from_str(
        &std::fs::read_to_string(p).with_context(|| format!("read {}", p.display()))?,
    )
    .with_context(|| format!("{} is not a probe result", p.display()))
}

fn profile(a: &ProfileArgs) -> Result<()> {
    let (cfg, id) = a.cluster.load()?;
    let t = a.cluster.transport(&cfg, &id, None, SocketBuffers::Auto)?;
    let probe = a.probe.as_ref().map(read_probe).transpose()?;
    let hw = client::profile_cluster(&t, &cfg, probe.as_ref()).map_err(anyhow::Error::msg)?;
    if probe.is_none() {
        eprintln!("no probe given: the graph has no network links; run `cluster probe` first");
    }
    write_out(a.out.as_ref(), &hw.to_json()?)
}

fn probe(a: &ProbeArgs) -> Result<()> {
    let (cfg, id) = a.cluster.load()?;
    let t = a.cluster.transport(&cfg, &id, None, SocketBuffers::Auto)?;
    let sizes: Vec<u64> = PROBE_SIZES
        .iter()
        .copied()
        .filter(|b| *b <= a.max_bytes)
        .collect();
    anyhow::ensure!(!sizes.is_empty(), "--max-bytes is below 4 KiB");
    let buffers = match a.buffers {
        SocketBuffers::Auto => None,
        SocketBuffers::Explicit { send, recv } => Some((send, recv)),
    };
    let r =
        client::probe_cluster(&t, &cfg, &sizes, a.repeats, buffers).map_err(anyhow::Error::msg)?;
    for c in &r.curves {
        let (first, last) = (c.points.first(), c.points.last());
        println!(
            "{} to {}: {:.3} ms at {} B, {:.3} ms at {} B",
            c.ranks[0],
            c.ranks[1],
            first.map_or(0.0, |p| p.1 * 1e3),
            first.map_or(0.0, |p| p.0),
            last.map_or(0.0, |p| p.1 * 1e3),
            last.map_or(0.0, |p| p.0),
        );
    }
    std::fs::write(&a.out, serde_json::to_string_pretty(&r)?)
        .with_context(|| format!("write {}", a.out.display()))
}

fn watch(a: &WatchArgs) -> Result<()> {
    let (cfg, id) = a.cluster.load()?;
    anyhow::ensure!(
        id.node_id()? == cfg.head,
        "watch runs with the head's key and certificate"
    );
    let telemetry = Arc::new(Telemetry::new(&TelemetryConfig::from_env())?);
    telemetry.install_global()?;
    let t = Arc::new(a.cluster.transport(&cfg, &id, None, SocketBuffers::Auto)?);
    let monitor = Monitor::start(t, cfg.clone());
    if let Some(p) = &a.probe {
        // Small-frame round trip: twice the one-way time of the smallest probed frame, from the
        // head to each node.
        let head = format!("{}/", cfg.head);
        for c in read_probe(p)?.curves {
            if let (Some(to), Some(pt)) = (c.ranks.last(), c.points.first())
                && c.ranks[0].starts_with(&head)
            {
                let node = NodeId::new(to.split('/').next().unwrap_or(to));
                monitor.set_baseline(&node, pt.1 * 2.0);
            }
        }
    }
    let rx = monitor.subscribe();
    println!(
        "watching {} nodes of cluster {}",
        cfg.nodes.len(),
        cfg.cluster
    );
    while let Ok((n, st)) = rx.recv() {
        let note = if st == NodeState::Down {
            " (not reachable)"
        } else {
            ""
        };
        println!("node {n}: {}{note}", st.label());
    }
    Ok(())
}

fn cert(a: &CertArgs) -> Result<()> {
    let g = generate_self_signed(&a.node)?;
    std::fs::create_dir_all(&a.out)
        .with_context(|| format!("cannot create {}", a.out.display()))?;
    let (key, cert) = (a.out.join("key.pem"), a.out.join("cert.pem"));
    // The key first: if it cannot be written with restricted access, no certificate is left
    // behind that has no key.
    write_private(&key, g.key_pem.as_bytes(), a.force)?;
    if let Err(e) = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .create_new(!a.force)
        .truncate(true)
        .open(&cert)
        .and_then(|mut f| std::io::Write::write_all(&mut f, g.cert_pem.as_bytes()))
    {
        let _ = std::fs::remove_file(&key);
        return Err(anyhow::Error::new(e).context(format!("cannot write {}", cert.display())));
    }
    println!("node {}: certificate {}", a.node, cert.display());
    println!(
        "private key {} (owner only, keep it on this box)",
        key.display()
    );
    Ok(())
}
