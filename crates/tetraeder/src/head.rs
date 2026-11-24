//! `tetraeder serve --cluster cluster.toml --replicas replicas.toml`: the head of a cluster
//! (ADR 0091, step 5). The HTTP gateway and the router run here; a replica this node leads runs
//! in process, every other replica is reached through a `RemoteEngine` to its leader's daemon,
//! which starts `tetraeder replica` for it. A replica joins the router when its node is `Up` and
//! it answers one probe step, and leaves when its node goes `Down` or its connection breaks.
//!
//! `replicas.toml`:
//!
//! ```toml
//! model_sha256 = "<64 hex digits>"   # the model's file name on every node, `<sha256>.gguf`
//!
//! [[replica]]
//! leader = "a"                       # the head's own node: in process
//! cap = 4                            # requests it takes at once (default: --parallel)
//!
//! [[replica]]
//! leader = "b"
//! # A replica that spans nodes lists its segments in order (node and single-node plan file,
//! # `plan::segment_plan`), and the planner's decode step time of each, seconds.
//! # segments = [{ node = "b", plan = "r1-s0.toml" }, { node = "c", plan = "r1-s1.toml" }]
//! # predicted_s = [0.012, 0.015]
//! ```

use crate::node::ClusterArgs;
use crate::serve::{ServeArgs, build_engine};
use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tetraeder_control::head::{
    ClusterEngine, ClusterHead, ReplicaTarget, SegmentAssign, StartSpec,
};
use tetraeder_control::monitor::Monitor;
use tetraeder_gateway::engine::Engine as GatewayEngine;
use tetraeder_gateway::fake::{FakeEngine, serve_script};
use tetraeder_telemetry::Telemetry;
use tetraeder_transport::{NodeId, SocketBuffers, Transport};

/// Replicas below this many shared prompt tokens do not count as holding a prefix (the prefix
/// cache's `RESTORE_MIN_GAIN`).
const MIN_PREFIX: u32 = tetraeder_scheduler::RESTORE_MIN_GAIN as u32;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplicasFile {
    pub model_sha256: String,
    #[serde(rename = "replica")]
    pub replicas: Vec<ReplicaEntry>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplicaEntry {
    pub leader: String,
    #[serde(default)]
    pub cap: Option<u32>,
    /// The replica's plan TOML, handed to the leader's engine.
    #[serde(default)]
    pub plan: Option<PathBuf>,
    #[serde(default)]
    pub segments: Vec<SegmentEntry>,
    #[serde(default)]
    pub predicted_s: Vec<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SegmentEntry {
    pub node: String,
    pub plan: PathBuf,
}

impl ReplicasFile {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("cannot read {}", path.display()))?;
        let f: Self = toml::from_str(&text).with_context(|| format!("{}", path.display()))?;
        anyhow::ensure!(
            f.model_sha256.len() == 64 && f.model_sha256.bytes().all(|b| b.is_ascii_hexdigit()),
            "model_sha256 is not 64 hex digits"
        );
        anyhow::ensure!(!f.replicas.is_empty(), "no [[replica]]");
        Ok(f)
    }
}

/// What keeps the head running: dropping it stops the join loops and the monitor.
pub struct HeadGuard {
    _engine: Arc<ClusterEngine>,
    _head: ClusterHead,
    _monitor: Arc<Monitor>,
}

/// The cluster engine and the number of requests the gateway lets run at once (the sum of the
/// replicas' caps, so requests wait at the router, not in the gateway).
pub fn start(
    a: &ServeArgs,
    telemetry: &Arc<Telemetry>,
) -> Result<(Arc<dyn GatewayEngine>, usize, HeadGuard)> {
    let cluster = ClusterArgs {
        cluster: a.cluster.clone().context("--cluster")?,
        key: a.key.clone(),
        cert: a.cert.clone(),
    };
    let (cfg, ident) = cluster.load()?;
    let me = ident.node_id()?;
    anyhow::ensure!(
        me == cfg.head,
        "serve --cluster runs with the head's key and certificate (`{}`), this is `{me}`",
        cfg.head
    );
    let file = ReplicasFile::load(
        a.replicas
            .as_deref()
            .context("--replicas FILE is required")?,
    )?;
    let transport: Arc<dyn Transport> =
        Arc::new(cluster.transport(&cfg, &ident, None, SocketBuffers::Auto)?);

    // The replica the head leads itself runs in process; it also answers tokenization.
    let local = file.replicas.iter().any(|r| r.leader == me.0);
    let (front, front_cap): (Arc<dyn GatewayEngine>, usize) = if local {
        build_engine(a, telemetry)?
    } else if a.fake {
        (Arc::new(FakeEngine::new(serve_script)), 1)
    } else {
        anyhow::bail!(
            "the head must lead one replica: this build tokenizes in the engine it runs \
             (a head without a replica needs a tokenizer-only engine, not built yet)"
        );
    };
    let mut targets = Vec::new();
    let mut remote_specs = Vec::new();
    let mut total = 0usize;
    for (i, r) in file.replicas.iter().enumerate() {
        let cap = r.cap.map_or(front_cap, |c| c as usize);
        total += cap;
        let node = NodeId::new(r.leader.clone());
        if node == me {
            targets.push(ReplicaTarget {
                engine: Arc::clone(&front),
                remote: None,
                cap: u32::try_from(cap)?,
            });
            continue;
        }
        let peer = cfg
            .peer(&node)
            .with_context(|| format!("`{node}` is not a node of cluster.toml"))?;
        let remote = tetraeder_control::replica::RemoteEngine::new(
            Arc::clone(&front),
            Arc::clone(&transport),
            peer,
            u32::try_from(i)?,
        );
        let mut segments = Vec::new();
        for s in &r.segments {
            segments.push(SegmentAssign {
                peer: cfg
                    .peer(&NodeId::new(s.node.clone()))
                    .with_context(|| format!("`{}` is not a node of cluster.toml", s.node))?,
                plan: std::fs::read_to_string(&s.plan)
                    .with_context(|| format!("cannot read {}", s.plan.display()))?,
            });
        }
        let plan = match &r.plan {
            Some(p) => std::fs::read_to_string(p)
                .with_context(|| format!("cannot read {}", p.display()))?,
            None => String::new(),
        };
        remote_specs.push((
            u32::try_from(i)?,
            node,
            StartSpec {
                plan,
                model_sha256: file.model_sha256.clone(),
                segments,
                predicted_s: r.predicted_s.clone(),
            },
        ));
        targets.push(ReplicaTarget {
            engine: Arc::clone(&remote) as Arc<dyn GatewayEngine>,
            remote: Some(remote),
            cap: u32::try_from(cap)?,
        });
    }
    let engine = Arc::new(ClusterEngine::new(front, targets, MIN_PREFIX, None));
    for (i, r) in file.replicas.iter().enumerate() {
        if r.leader == me.0 {
            engine.set_up(u32::try_from(i)?, true);
        }
    }
    let monitor = Arc::new(Monitor::start(Arc::clone(&transport), cfg));
    let head = ClusterHead::start(&engine, &monitor, &transport, remote_specs);
    eprintln!(
        "cluster {}: {} replicas, {total} requests at once",
        file.replicas.len(),
        file.replicas.len()
    );
    Ok((
        Arc::clone(&engine) as Arc<dyn GatewayEngine>,
        total,
        HeadGuard {
            _engine: engine,
            _head: head,
            _monitor: monitor,
        },
    ))
}
