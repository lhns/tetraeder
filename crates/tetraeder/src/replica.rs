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
use crate::serve::{ServeArgs, build_scheduler, scheduler_over};
use anyhow::Context;
use clap::Args;
use std::sync::Arc;
use tetraeder::pipeline::replica_pipeline;
use tetraeder::remote_backend::{RemoteBackend, RemoteConfig};
use tetraeder_control::kv::{PrefixKey, PrefixStore, serve_prefix_io};
use tetraeder_control::replica::{KvHooks, PeerLookup, ReplicaServer};
use tetraeder_gateway::engine::Engine as GatewayEngine;
use tetraeder_scheduler::{BatchBackend, Scheduler, SegmentBackend};
use tetraeder_telemetry::{Telemetry, TelemetryConfig};
use tetraeder_transport::{NodeId, SocketBuffers, Transport};

pub const REPLICA_ENV: &str = "TETRAEDER_REPLICA";
pub const SEGMENTS_ENV: &str = "TETRAEDER_REPLICA_SEGMENTS";
pub const PREDICTED_ENV: &str = "TETRAEDER_REPLICA_PREDICTED";
/// Where the replica writes the loopback port of its prefix listener (set by the daemon, which
/// relays the `Prefix` connections of other nodes to it).
pub const PREFIX_PORT_FILE_ENV: &str = "TETRAEDER_PREFIX_PORT_FILE";

/// The replica's prefix cache as the cluster sees it: the scheduler answers on its own thread.
struct SchedulerStore(Arc<Scheduler>);

impl PrefixStore for SchedulerStore {
    fn kv_format(&self) -> String {
        self.0.kv_format().to_owned()
    }

    fn get(&self, key: &PrefixKey) -> Option<(Vec<u32>, Vec<u8>)> {
        self.0.prefix_get(&key.tokens)
    }

    fn put(&self, key: PrefixKey, bytes: Vec<u8>) {
        let _ = self.0.prefix_put(key.tokens, bytes);
    }

    fn row_bytes(&self) -> u64 {
        self.0.kv_row_bytes() as u64
    }
}

/// Serves the prefix cache to the daemon's relay on a loopback port, written to `file`
/// (atomically, so the daemon never reads half a number).
fn serve_prefix_on_loopback(
    store: Arc<dyn PrefixStore>,
    file: &std::path::Path,
) -> std::io::Result<()> {
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0))?;
    let port = listener.local_addr()?.port();
    let tmp = file.with_extension("tmp");
    std::fs::write(&tmp, port.to_string())?;
    std::fs::rename(&tmp, file)?;
    std::thread::Builder::new()
        .name("prefix-listen".into())
        .spawn(move || {
            for stream in listener.incoming().flatten() {
                let store = Arc::clone(&store);
                std::thread::spawn(move || {
                    let Ok(mut tx) = stream.try_clone() else {
                        return;
                    };
                    let mut rx = stream;
                    serve_prefix_io(&mut rx, &mut tx, store.as_ref());
                });
            }
        })?;
    Ok(())
}

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
    // `--cluster`, `--key` and `--cert` of `serve`: this node's identity, for the segments and
    // for prefixes fetched from other nodes. Without `--cluster` the replica stands alone.
    let net = match &a.serve.cluster {
        Some(c) => {
            let cluster = ClusterArgs {
                cluster: c.clone(),
                key: a.serve.key.clone(),
                cert: a.serve.cert.clone(),
            };
            let (cfg, ident) = cluster.load()?;
            let t: Arc<dyn Transport> =
                Arc::new(cluster.transport(&cfg, &ident, None, SocketBuffers::Auto)?);
            Some((cfg, t))
        }
        None => None,
    };
    let (scheduler, parallel): (Option<Arc<Scheduler>>, usize) = if segments.len() > 1 {
        let replica: u32 = std::env::var(REPLICA_ENV)
            .context("TETRAEDER_REPLICA is not set")?
            .parse()
            .context("TETRAEDER_REPLICA is not a number")?;
        let predicted: Vec<f64> = env_list(PREDICTED_ENV)
            .iter()
            .filter_map(|s| s.parse().ok())
            .collect();
        let (cfg, t) = net
            .clone()
            .context("a replica across nodes needs --cluster, --key and --cert")?;
        let peers = segments
            .iter()
            .map(|node| {
                cfg.peer(&NodeId::new(node.clone()))
                    .with_context(|| format!("`{node}` is not a node of cluster.toml"))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let metrics = telemetry.metrics().clone();
        // The plan's prefill micro-batches (ADR 0077): the replica's plan file, when it reads.
        let micro = std::env::var_os("TETRAEDER_REPLICA_PLAN")
            .and_then(|f| std::fs::read_to_string(f).ok())
            .and_then(|t| tetraeder_plan::Plan::from_toml(&t).ok())
            .map_or(1, |p| p.knobs.prefill_micro_batches);
        // The backend is built on the scheduler's thread: it is not `Send`.
        let (s, p) = scheduler_over(&a.serve, &telemetry, move || {
            let mut backs: Vec<Box<dyn SegmentBackend>> = Vec::new();
            for (i, peer) in peers.into_iter().enumerate() {
                let rc = RemoteConfig::node(Arc::clone(&t), peer, replica, i as u32);
                backs.push(Box::new(
                    RemoteBackend::start(rc).map_err(|e| format!("segment {i}: {e}"))?,
                ));
            }
            let p = replica_pipeline(backs, Some(metrics), &predicted)?
                .with_prefill_micro_batches(micro);
            Ok(Box::new(p) as Box<dyn BatchBackend>)
        })?;
        (Some(s), p)
    } else if a.serve.fake {
        (None, 1)
    } else {
        let (s, p) = build_scheduler(&a.serve, &telemetry)?;
        (Some(s), p)
    };
    let (engine, kv): (Arc<dyn GatewayEngine>, Option<KvHooks>) = match (&scheduler, &net) {
        (Some(s), Some((cfg, t))) => {
            let store: Arc<dyn PrefixStore> = Arc::new(SchedulerStore(Arc::clone(s)));
            if let Some(file) = std::env::var_os(PREFIX_PORT_FILE_ENV) {
                serve_prefix_on_loopback(Arc::clone(&store), std::path::Path::new(&file))
                    .context("cannot serve prefixes")?;
            }
            let cfg = cfg.clone();
            let peers: PeerLookup = Arc::new(move |n: &NodeId| cfg.peer(n));
            (
                Arc::clone(s) as Arc<dyn GatewayEngine>,
                Some(KvHooks {
                    store,
                    transport: Arc::clone(t),
                    peers,
                }),
            )
        }
        (Some(s), None) => (Arc::clone(s) as Arc<dyn GatewayEngine>, None),
        (None, _) => (crate::serve::build_engine(&a.serve, &telemetry)?.0, None),
    };
    let server = ReplicaServer::new(engine, u32::try_from(parallel).unwrap_or(u32::MAX), kv);
    server.serve(std::io::stdin().lock(), std::io::stdout());
    telemetry.shutdown();
    Ok(())
}
