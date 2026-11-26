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
use crate::serve::{ServeArgs, build_scheduler, scheduler_over_step};
use anyhow::Context;
use clap::Args;
use std::sync::Arc;
use tetraeder::pipeline::{plan_step_tokens, replica_pipeline};
use tetraeder::remote_backend::{RemoteBackend, RemoteConfig};
use tetraeder_control::kv::{
    PrefixGrants, PrefixKey, PrefixStore, read_relay_preface, serve_prefix_io,
};
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
/// The launch's token (set by the daemon): the first line of every connection the daemon relays
/// to the prefix listener carries it, with the authenticated peer. Without it, a connection is
/// dropped, so another process that finds the loopback port reads nothing.
pub const PREFIX_TOKEN_ENV: &str = "TETRAEDER_PREFIX_TOKEN";
/// A relayed connection must send its first line within this, and may idle this long.
const PREFIX_PREFACE_WAIT: std::time::Duration = std::time::Duration::from_secs(5);
const PREFIX_IDLE: std::time::Duration = std::time::Duration::from_secs(30);

/// The replica's prefix cache as the cluster sees it: the scheduler answers on its own thread.
pub(crate) struct SchedulerStore(pub(crate) Arc<Scheduler>);

impl PrefixStore for SchedulerStore {
    fn kv_format(&self) -> String {
        self.0.kv_format().to_owned()
    }

    fn get(&self, key: &PrefixKey) -> Option<(Vec<u32>, Vec<u8>)> {
        self.0.prefix_get(&key.tokens)
    }

    fn put(&self, key: PrefixKey, bytes: Vec<u8>) -> bool {
        self.0.prefix_put(key.tokens, bytes)
    }

    fn row_bytes(&self) -> u64 {
        self.0.kv_row_bytes() as u64
    }

    fn max_tokens(&self) -> u64 {
        self.0
            .models()
            .first()
            .map_or(u64::MAX, |m| u64::from(m.context_length))
    }
}

/// Writes `content` to `file` atomically, so the daemon never reads half of it. The file holds
/// a token: it is created readable by its owner only (Unix).
pub(crate) fn write_atomic(file: &std::path::Path, content: &str) -> std::io::Result<()> {
    use std::io::Write;
    let tmp = file.with_extension("tmp");
    let mut o = std::fs::OpenOptions::new();
    o.write(true).create(true).truncate(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut o, 0o600);
    o.open(&tmp)?.write_all(content.as_bytes())?;
    std::fs::rename(&tmp, file)
}

/// Creates `dir` and its parents; the directories this creates are for their owner only (Unix).
pub(crate) fn create_private_dir(dir: &std::path::Path) -> std::io::Result<()> {
    let mut b = std::fs::DirBuilder::new();
    b.recursive(true);
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut b, 0o700);
    b.create(dir)
}

/// Serves the prefix cache to the daemon's relay on a loopback port, which it returns. A
/// connection is served only when it opens with the launch's `token`; the line also names the
/// peer whose grants apply.
pub(crate) fn serve_prefix_on_loopback(
    store: Arc<dyn PrefixStore>,
    grants: Arc<PrefixGrants>,
    token: String,
) -> std::io::Result<u16> {
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0))?;
    let port = listener.local_addr()?.port();
    std::thread::Builder::new()
        .name("prefix-listen".into())
        .spawn(move || {
            for stream in listener.incoming().flatten() {
                let (store, grants, token) =
                    (Arc::clone(&store), Arc::clone(&grants), token.clone());
                std::thread::spawn(move || {
                    let Ok(mut tx) = stream.try_clone() else {
                        return;
                    };
                    let mut rx = stream;
                    let _ = rx.set_read_timeout(Some(PREFIX_PREFACE_WAIT));
                    let Some(peer) = read_relay_preface(&mut rx, &token) else {
                        tetraeder_transport::metrics::rejected("bad_relay");
                        return;
                    };
                    if std::io::Write::write_all(&mut tx, tetraeder_control::kv::RELAY_ACK).is_err()
                    {
                        return;
                    }
                    // A peer that opens a connection and goes quiet does not hold it.
                    let _ = rx.set_read_timeout(Some(PREFIX_IDLE));
                    serve_prefix_io(&mut rx, &mut tx, store.as_ref(), &grants, &peer);
                });
            }
        })?;
    Ok(port)
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
        let plan = std::env::var_os("TETRAEDER_REPLICA_PLAN").and_then(|f| {
            let read = std::fs::read_to_string(&f).map_err(|e| e.to_string());
            let plan =
                read.and_then(|t| tetraeder_plan::Plan::from_toml(&t).map_err(|e| e.to_string()));
            plan.map_err(|e| {
                eprintln!("warning: prefill is not pipelined, the plan is unreadable: {e}")
            })
            .ok()
        });
        let micro = plan.as_ref().map_or(1, |p| p.knobs.prefill_micro_batches);
        // One scheduler step is what the planner priced: the micro-batches of one step.
        let step = plan
            .as_ref()
            .and_then(|p| plan_step_tokens(p, a.serve.step_tokens));
        // The backend is built on the scheduler's thread: it is not `Send`.
        let (s, p) = scheduler_over_step(&a.serve, &telemetry, step, move || {
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
            let grants = Arc::new(PrefixGrants::default());
            if let Some(file) = std::env::var_os(PREFIX_PORT_FILE_ENV) {
                let token = std::env::var(PREFIX_TOKEN_ENV)
                    .context("TETRAEDER_PREFIX_TOKEN is not set (the daemon sets it)")?;
                let port = serve_prefix_on_loopback(Arc::clone(&store), Arc::clone(&grants), token)
                    .context("cannot serve prefixes")?;
                write_atomic(std::path::Path::new(&file), &port.to_string())
                    .context("cannot write the prefix port")?;
            }
            let cfg = cfg.clone();
            let peers: PeerLookup = Arc::new(move |n: &NodeId| cfg.peer(n));
            (
                Arc::clone(s) as Arc<dyn GatewayEngine>,
                Some(KvHooks {
                    store,
                    transport: Arc::clone(t),
                    peers,
                    grants,
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

#[cfg(test)]
mod tests {
    fn plan(micro: u32, ubatch: Option<u32>) -> tetraeder_plan::Plan {
        let text = "model = \"m\"\n[[replicas]]\n[[replicas.stages]]\nlayers = [0, 4]\n\
                    devices = [\"gpu0\"]\ntp = 1\nep = 1\n";
        let mut p = tetraeder_plan::Plan::from_toml(text).unwrap();
        p.knobs.prefill_micro_batches = micro;
        p.knobs.ubatch = ubatch;
        p
    }

    /// hardware: none. The Case S pick (4 micro-batches at ubatch 2048) steps 8192 tokens; the
    /// exit's pp2048 at ubatch 512 and four micro-batches steps 2048, today's default.
    #[test]
    fn the_step_is_the_plans_micro_batches_times_its_ubatch() {
        assert_eq!(
            super::plan_step_tokens(&plan(4, Some(2048)), None),
            Some(8192)
        );
        assert_eq!(
            super::plan_step_tokens(&plan(4, Some(512)), None),
            Some(2048)
        );
        assert_eq!(
            super::plan_step_tokens(&plan(1, Some(512)), None),
            Some(512)
        );
        assert_eq!(
            super::plan_step_tokens(&plan(0, Some(512)), None),
            Some(512)
        );
    }

    /// hardware: none. A plan without a ubatch, or an explicit `--step-tokens`, leaves the step
    /// to the flag and its default.
    #[test]
    fn the_flag_and_a_plan_without_ubatch_keep_the_default_step() {
        assert_eq!(super::plan_step_tokens(&plan(4, None), None), None);
        assert_eq!(
            super::plan_step_tokens(&plan(4, Some(2048)), Some(1024)),
            None
        );
    }

    /// hardware: none. The port and token files are for their owner only; the directories this
    /// creates are too.
    #[cfg(unix)]
    #[test]
    fn the_prefix_files_and_their_directory_are_private() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("tetraeder-private-{}", std::process::id()));
        let state = dir.join("state");
        super::create_private_dir(&state).unwrap();
        let file = state.join("prefix-head");
        super::write_atomic(&file, "40123 abc").unwrap();
        let mode = |p: &std::path::Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(&state), 0o700);
        assert_eq!(mode(&file), 0o600);
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "40123 abc");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// hardware: none. The content is replaced whole (a second write leaves no stale tail).
    #[test]
    fn write_atomic_replaces_the_file() {
        let dir = std::env::temp_dir().join(format!("tetraeder-atomic-{}", std::process::id()));
        super::create_private_dir(&dir).unwrap();
        let file = dir.join("f");
        super::write_atomic(&file, "a long first line").unwrap();
        super::write_atomic(&file, "short").unwrap();
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "short");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
