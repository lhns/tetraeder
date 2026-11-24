//! `tetraeder replica`: the engine of `serve` without the HTTP front end, started by a node's
//! daemon for a replica this node leads (ADR 0091, step 5). It speaks the replica protocol
//! (`tetraeder_control::replica`) over stdin and stdout: the daemon relays the head's connection
//! into it, so stdout carries frames only and diagnostics go to stderr. It ends when stdin ends.
//!
//! A replica of one node runs the whole model as `serve` does. A replica that spans nodes
//! (`TETRAEDER_REPLICA_SEGMENTS`, the nodes of its segments in order, set by the daemon from
//! `StartReplica`) runs a `PipelineBackend` over a `RemoteBackend` per segment, each dialled with
//! this node's key; the head assigned the workers before it started the leader.

use crate::node::ClusterArgs;
use crate::serve::{ServeArgs, build_engine, scheduler_over};
use anyhow::Context;
use clap::Args;
use std::sync::Arc;
use tetraeder::pipeline::replica_pipeline;
use tetraeder::remote_backend::{RemoteBackend, RemoteConfig};
use tetraeder_control::replica::ReplicaServer;
use tetraeder_gateway::engine::Engine as GatewayEngine;
use tetraeder_scheduler::{BatchBackend, SegmentBackend};
use tetraeder_telemetry::{Telemetry, TelemetryConfig};
use tetraeder_transport::{NodeId, SocketBuffers, Transport};

pub const REPLICA_ENV: &str = "TETRAEDER_REPLICA";
pub const SEGMENTS_ENV: &str = "TETRAEDER_REPLICA_SEGMENTS";
pub const PREDICTED_ENV: &str = "TETRAEDER_REPLICA_PREDICTED";

#[derive(Args)]
pub struct ReplicaArgs {
    /// The same model, engine and placement arguments as `serve`; `--fake` serves the scripted
    /// engine (tests).
    #[command(flatten)]
    pub serve: ServeArgs,
}

fn env_list(name: &str) -> Vec<String> {
    std::env::var(name)
        .map(|v| {
            v.split(',')
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

pub fn run(a: &ReplicaArgs) -> anyhow::Result<()> {
    let telemetry = Arc::new(Telemetry::new(&TelemetryConfig::from_env())?);
    telemetry.install_global()?;
    let segments = env_list(SEGMENTS_ENV);
    let (engine, parallel): (Arc<dyn GatewayEngine>, usize) = if segments.len() > 1 {
        let replica: u32 = std::env::var(REPLICA_ENV)
            .context("TETRAEDER_REPLICA is not set")?
            .parse()
            .context("TETRAEDER_REPLICA is not a number")?;
        let predicted: Vec<f64> = env_list(PREDICTED_ENV)
            .iter()
            .filter_map(|s| s.parse().ok())
            .collect();
        // `--cluster`, `--key` and `--cert` of `serve`: this node's identity for the segments.
        let cluster = ClusterArgs {
            cluster: a
                .serve
                .cluster
                .clone()
                .unwrap_or_else(|| "cluster.toml".into()),
            key: a.serve.key.clone(),
            cert: a.serve.cert.clone(),
        };
        let (cfg, ident) = cluster.load()?;
        let t: Arc<dyn Transport> =
            Arc::new(cluster.transport(&cfg, &ident, None, SocketBuffers::Auto)?);
        let peers = segments
            .iter()
            .map(|node| {
                cfg.peer(&NodeId::new(node.clone()))
                    .with_context(|| format!("`{node}` is not a node of cluster.toml"))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let metrics = telemetry.metrics().clone();
        // The backend is built on the scheduler's thread: it is not `Send`.
        scheduler_over(&a.serve, &telemetry, move || {
            let mut backs: Vec<Box<dyn SegmentBackend>> = Vec::new();
            for (i, peer) in peers.into_iter().enumerate() {
                let rc = RemoteConfig::node(Arc::clone(&t), peer, replica, i as u32);
                backs.push(Box::new(
                    RemoteBackend::start(rc).map_err(|e| format!("segment {i}: {e}"))?,
                ));
            }
            let p = replica_pipeline(backs, Some(metrics), &predicted)?;
            Ok(Box::new(p) as Box<dyn BatchBackend>)
        })?
    } else {
        build_engine(&a.serve, &telemetry)?
    };
    let server = ReplicaServer::new(engine, u32::try_from(parallel).unwrap_or(u32::MAX), None);
    server.serve(std::io::stdin().lock(), std::io::stdout());
    telemetry.shutdown();
    Ok(())
}
