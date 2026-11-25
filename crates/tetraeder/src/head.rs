//! `tetraeder serve --cluster cluster.toml`: the head of a cluster (ADR 0091, steps 5 and 6). The
//! HTTP gateway and the router run here; a replica this node leads runs in process, every other
//! replica is reached through a `RemoteEngine` to its leader's daemon, which starts
//! `tetraeder replica` for it. A replica joins the router when its node is `Up` and it answers
//! one probe step, and leaves when its node goes `Down` or its connection breaks.
//!
//! A head that leads no replica answers tokenization, templates and grammar checks from the
//! model's vocabulary (`-m`, no weights) and generates nowhere: every request goes to a replica.
//!
//! The replicas come from `--replicas replicas.toml` (written by hand) or from `--cluster-plan`
//! (derived from the cluster plan and its hardware graph; the file wins when both are given).
//! With a plan, a node that changes state replans the replicas that span it: the replica is
//! drained (its in-flight requests finish) and started again with the new plan.
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
//! prefill_tps = 800.0                # its prompt speed, for the prefix fetch rule (default: no fetch)
//! # A replica that spans nodes lists its segments in order (node and single-node plan file,
//! # `plan::segment_plan`), and the planner's decode step time of each, seconds.
//! # segments = [{ node = "b", plan = "r1-s0.toml" }, { node = "c", plan = "r1-s1.toml" }]
//! # predicted_s = [0.012, 0.015]
//! ```

use crate::node::ClusterArgs;
use crate::serve::{SchedParts, ServeArgs, build_engine, build_scheduler};
use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tetraeder_control::ClusterConfig;
use tetraeder_control::cluster_replan::{ClusterControl, Derive, ReplanWatcher};
use tetraeder_control::head::{
    ClusterEngine, ClusterHead, FetchPolicy, KvInfo, ReplicaTarget, SegmentAssign, StartSpec,
    specs_from_plan,
};
use tetraeder_control::kv::{
    PrefixGrants, PrefixKey, PrefixStore, fetch_beats_recompute, new_token,
};
use tetraeder_control::membership::NodeState;
use tetraeder_control::monitor::Monitor;
use tetraeder_control::replica::FetchFrom;
use tetraeder_control::{
    Config, Outcome, PlanSource, Proposal, Replanner, SystemClock, Trigger, client,
};
use tetraeder_gateway::engine::Engine as GatewayEngine;
use tetraeder_gateway::fake::{FakeEngine, serve_script};
use tetraeder_plan::{Forbid, Hints, Phase, Plan, segments};
use tetraeder_planner::cluster::{enumerate_cluster, predict_cluster_plan, single_node_references};
use tetraeder_planner::{ModelLayout, Workload};
use tetraeder_profiler::HardwareGraph;
use tetraeder_scheduler::FrontEngine;
use tetraeder_telemetry::Telemetry;
use tetraeder_transport::{NodeId, SocketBuffers, Transport};

/// Replicas below this many shared prompt tokens do not count as holding a prefix (the prefix
/// cache's `RESTORE_MIN_GAIN`).
const MIN_PREFIX: u32 = tetraeder_scheduler::RESTORE_MIN_GAIN as u32;

/// Copying a snapshot from the host into the engine, bytes per second (the host to device rate
/// of a PCIe 4 link, rounded down): the restore time of the fetch rule.
const RESTORE_BYTES_PER_S: f64 = 12e9;

/// How long the head waits for the daemons to answer a profile at start.
const PROFILE_WAIT: Duration = Duration::from_secs(120);

/// A replanned replica drains for at most this long before what is left in flight is cut.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(60);
/// A replanned replica has this long to join the router again.
const JOIN_TIMEOUT: Duration = Duration::from_secs(300);
/// What a replan costs in seconds of service (the drain, a model load), for the replanner's
/// gain rule; a node that is gone makes the gain large whatever it is.
const REPLAN_COST_S: f64 = 60.0;

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
    /// Prompt tokens per second of this replica: with the probed links it prices a prefix fetch
    /// against recomputing. Without it this replica never asks for a fetch.
    #[serde(default)]
    pub prefill_tps: Option<f64>,
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
            is_sha256(&f.model_sha256),
            "model_sha256 is not 64 hex digits"
        );
        anyhow::ensure!(!f.replicas.is_empty(), "no [[replica]]");
        Ok(f)
    }
}

fn is_sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// One replica as the head starts it.
struct Resolved {
    leader: NodeId,
    cap: Option<u32>,
    spec: StartSpec,
    /// Prompt tokens per second, 0 when not known.
    pp_tps: f64,
}

fn resolve_file(file: &ReplicasFile, cfg: &ClusterConfig) -> Result<Vec<Resolved>> {
    let mut out = Vec::new();
    for r in &file.replicas {
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
        out.push(Resolved {
            leader: NodeId::new(r.leader.clone()),
            cap: r.cap,
            spec: StartSpec {
                plan,
                model_sha256: file.model_sha256.clone(),
                segments,
                predicted_s: r.predicted_s.clone(),
            },
            pp_tps: r.prefill_tps.unwrap_or(0.0),
        });
    }
    Ok(out)
}

/// The sha256 that names the model file on every node.
fn model_sha(a: &ServeArgs) -> Result<String> {
    // The id of a single file is its stem, that of a split model the folder of its shards.
    let sha = match &a.model_sha256 {
        Some(s) => s.clone(),
        None => a
            .model
            .as_deref()
            .and_then(tetraeder_model::model_id::id_of_path)
            .unwrap_or_default(),
    };
    anyhow::ensure!(
        is_sha256(&sha),
        "--model-sha256 is not 64 hex digits (and -m is not named `<id>.gguf` or `<id>/<id>-00001-of-0000N.gguf`)"
    );
    Ok(sha.to_ascii_lowercase())
}

/// The cluster's hardware graph: `--cluster-hw`, else what the daemons report (they may still be
/// starting, so it is asked for until they answer).
fn hardware(
    a: &ServeArgs,
    cfg: &ClusterConfig,
    transport: &Arc<dyn Transport>,
) -> Result<HardwareGraph> {
    if let Some(p) = &a.cluster_hw {
        let text =
            std::fs::read_to_string(p).with_context(|| format!("cannot read {}", p.display()))?;
        return HardwareGraph::from_json(&text)
            .or_else(|_| HardwareGraph::from_toml(&text))
            .with_context(|| format!("{} is not a hardware graph", p.display()));
    }
    let t0 = std::time::Instant::now();
    loop {
        match client::profile_cluster(transport.as_ref(), cfg, None) {
            Ok(hw) => return Ok(hw),
            Err(e) if t0.elapsed() >= PROFILE_WAIT => {
                anyhow::bail!("cannot profile the cluster: {e}");
            }
            Err(_) => std::thread::sleep(Duration::from_millis(500)),
        }
    }
}

/// The cluster plan and what it is priced on, kept for replanning.
struct PlanState {
    plan: Plan,
    hw: HardwareGraph,
    layout: Option<ModelLayout>,
    sha: String,
}

const WORKLOAD: Workload = Workload {
    prompt: 2048,
    r#gen: 128,
};

/// The planner's decode step time of each segment and the prompt rate of replica 0 of a plan
/// priced on `layout`; empty and 0 when it cannot be priced.
fn price(st: &PlanState) -> (Vec<Vec<f64>>, Vec<f64>) {
    let Some(layout) = &st.layout else {
        return (Vec::new(), Vec::new());
    };
    if st.plan.replicas.len() != 1 {
        return (Vec::new(), Vec::new());
    }
    match predict_cluster_plan(&st.hw, layout, &st.plan, &WORKLOAD) {
        Ok(p) => (
            vec![
                p.segments
                    .iter()
                    .map(|s| 1.0 / s.tg_tok_s.max(f64::MIN_POSITIVE))
                    .collect(),
            ],
            vec![p.pp_tps],
        ),
        Err(e) => {
            tracing::warn!("the plan cannot be priced: {e}");
            (Vec::new(), Vec::new())
        }
    }
}

fn resolve_plan(
    a: &ServeArgs,
    path: &Path,
    cfg: &ClusterConfig,
    transport: &Arc<dyn Transport>,
) -> Result<(Vec<Resolved>, PlanState)> {
    let plan = Plan::from_toml(
        &std::fs::read_to_string(path)
            .with_context(|| format!("cannot read {}", path.display()))?,
    )
    .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    let hw = hardware(a, cfg, transport)?;
    let layout = a.model.as_deref().and_then(|m| crate::plan::layout(m).ok());
    let st = PlanState {
        plan,
        hw,
        layout,
        sha: model_sha(a)?,
    };
    let (predicted, pp) = price(&st);
    let specs = specs_from_plan(&st.plan, &st.hw, cfg, &st.sha, &predicted)
        .map_err(anyhow::Error::msg)
        .context("the cluster plan does not fit the cluster")?;
    let resolved = specs
        .into_iter()
        .map(|(r, leader, spec)| Resolved {
            leader,
            cap: None,
            spec,
            pp_tps: pp.get(r as usize).copied().unwrap_or(0.0),
        })
        .collect();
    Ok((resolved, st))
}

/// Prices a prefix fetch against recomputing it (`kv::fetch_beats_recompute`) on the probed
/// links of the hardware graph.
pub struct PricedFetch {
    hw: HardwareGraph,
    /// Leader node of each replica.
    nodes: Vec<String>,
    /// Prompt tokens per second of each replica; 0 never fetches.
    pp_tps: Vec<f64>,
}

impl PricedFetch {
    pub fn new(hw: HardwareGraph, nodes: Vec<String>, pp_tps: Vec<f64>) -> Self {
        Self { hw, nodes, pp_tps }
    }
}

impl FetchPolicy for PricedFetch {
    fn plan(&self, holder: u32, target: u32, tokens: &[u32], kv: &KvInfo) -> Option<FetchFrom> {
        let (from, to) = (
            self.nodes.get(holder as usize)?,
            self.nodes.get(target as usize)?,
        );
        if from == to || kv.row_bytes == 0 || kv.format.is_empty() {
            return None;
        }
        let n = u32::try_from(tokens.len()).ok()?;
        let bytes = u64::from(n) * kv.row_bytes;
        fetch_beats_recompute(
            &self.hw,
            from,
            to,
            bytes,
            n,
            self.pp_tps.get(target as usize).copied().unwrap_or(0.0),
            bytes as f64 / RESTORE_BYTES_PER_S,
            MIN_PREFIX,
        )
        .then(|| FetchFrom {
            node: NodeId::new(from.clone()),
            key: PrefixKey {
                format: kv.format.clone(),
                tokens: tokens.to_vec(),
            },
        })
    }
}

/// The plans for a cluster whose nodes change: when the current plan computes on a node that is
/// not `Up` any more, the best plan on the others (the planner with that node forbidden); the
/// plan on one node alone when only one is left.
pub struct ClusterPlanSource {
    hw: HardwareGraph,
    layout: Option<ModelLayout>,
    monitor: Arc<Monitor>,
}

impl ClusterPlanSource {
    fn nodes_of(&self, plan: &Plan) -> Vec<String> {
        let mut v = Vec::new();
        for i in 0..plan.replicas.len() {
            for s in segments(plan, i, &self.hw).unwrap_or_default() {
                if let Some(n) = s.node
                    && !v.contains(&n)
                {
                    v.push(n);
                }
            }
        }
        v
    }

    /// The best plan with every node in `gone` forbidden.
    pub fn plan_without(&self, gone: &[String]) -> Option<Plan> {
        let layout = self.layout.as_ref()?;
        let hints = Hints {
            forbid: gone
                .iter()
                .map(|n| Forbid {
                    units: None,
                    device: None,
                    node: Some(n.clone()),
                    tier: None,
                })
                .collect(),
            ..Hints::default()
        };
        let (cands, _) = enumerate_cluster(&self.hw, layout, &hints, Phase(6));
        match cands.into_iter().next() {
            Some(c) => Some(c.plan),
            // One node is left: the plan on it alone beats no service.
            None => single_node_references(&self.hw, layout, &hints)
                .into_iter()
                .min_by(|a, b| a.prediction.request_s.total_cmp(&b.prediction.request_s))
                .map(|c| c.plan),
        }
    }
}

impl PlanSource for ClusterPlanSource {
    fn propose(&mut self, current: &Plan, trigger: &Trigger) -> Option<Proposal> {
        if !matches!(trigger, Trigger::NodeChange { .. }) {
            return None;
        }
        let gone: Vec<String> = self
            .monitor
            .states()
            .into_iter()
            .filter(|(_, s)| !matches!(s, NodeState::Up))
            .map(|(n, _)| n.0)
            .collect();
        // Only a plan that computes on a node that left needs a new one.
        let used = self.nodes_of(current);
        if !used.iter().any(|n| gone.contains(n)) {
            return None;
        }
        let plan = self.plan_without(&gone)?;
        Some(Proposal {
            plan,
            predicted_gain_s_per_s: 1.0,
            switch_cost_s: REPLAN_COST_S,
            knobs_only: false,
        })
    }
}

/// What keeps the head running: dropping it stops the join loops, the replanner and the monitor.
pub struct HeadGuard {
    _watcher: Option<ReplanWatcher>,
    _engine: Arc<ClusterEngine>,
    _head: ClusterHead,
    _monitor: Arc<Monitor>,
    _prefix_file: Option<PrefixFile>,
}

/// The engine of a head that leads no replica: the vocabulary of `-m` and nothing else.
fn front_engine(a: &ServeArgs, telemetry: &Arc<Telemetry>) -> Result<Arc<dyn GatewayEngine>> {
    if a.fake {
        return Ok(Arc::new(FakeEngine::new(serve_script)));
    }
    let path = a.model.clone().context(
        "-m is required: the head answers tokenization from the model's vocabulary \
         (the file is read for its vocabulary only)",
    )?;
    let cfg = SchedParts::new(a)?.config(&path, telemetry.metrics().clone());
    Ok(Arc::new(
        FrontEngine::open(&path, cfg.alias, cfg.chat_template, a.ctx)
            .map_err(anyhow::Error::msg)?,
    ))
}

/// Makes the head's own replica `r` a holder other replicas fetch prefixes from: the router
/// learns its snapshot format and size (the fetch rule prices with them), fetches are announced to
/// its grants, and a loopback listener serves `store` to this node's daemon, which finds it
/// through `<node_state>/prefix-head` (`<port> <token>`) and relays the other nodes' `Prefix`
/// connections to it.
fn serve_local_prefixes(
    engine: &ClusterEngine,
    r: u32,
    store: Arc<dyn PrefixStore>,
    node_state: &Path,
) -> Result<PrefixFile> {
    let grants = Arc::new(PrefixGrants::default());
    engine.set_local_replica(r, Arc::clone(&store), Arc::clone(&grants));
    let token = new_token();
    let port = crate::replica::serve_prefix_on_loopback(store, grants, token.clone())
        .context("cannot serve prefixes")?;
    crate::replica::create_private_dir(node_state)
        .with_context(|| format!("cannot create {}", node_state.display()))?;
    let file = crate::node::head_prefix_file(node_state);
    crate::replica::write_atomic(&file, &format!("{port} {token}"))
        .with_context(|| format!("cannot write {}", file.display()))?;
    Ok(PrefixFile(file))
}

/// The head's `prefix-head` file, removed when the head stops: the daemon would otherwise relay
/// to a port that another process may hold by then (it also checks the listener's answer).
pub(crate) struct PrefixFile(PathBuf);

impl Drop for PrefixFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Whether the head runs replica `r` in process: it leads it and the replica is one whole model
/// on one node. A replica the head leads that spans nodes runs as a child of the head's own
/// daemon like any other leader's (the head process does not run the first segment itself: it
/// would load the whole model and never use the other nodes).
fn runs_in_process(r: &Resolved, me: &NodeId) -> bool {
    r.leader == *me && r.spec.segments.is_empty()
}

/// The head runs one engine in process, so it can lead one such replica; two would share that
/// engine and its slots under two router entries.
fn check_one_head_led<'a>(
    in_process: impl Iterator<Item = &'a Resolved>,
    me: &NodeId,
) -> Result<()> {
    let n = in_process.filter(|r| runs_in_process(r, me)).count();
    anyhow::ensure!(
        n <= 1,
        "{n} replicas are led by the head `{me}`: the head runs one engine in process and leads at most one replica;          lead the others from other nodes"
    );
    Ok(())
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
    let transport: Arc<dyn Transport> =
        Arc::new(cluster.transport(&cfg, &ident, None, SocketBuffers::Auto)?);
    let monitor = Arc::new(Monitor::start(Arc::clone(&transport), cfg.clone()));

    let (resolved, plan_state) = match (&a.replicas, &a.cluster_plan) {
        (Some(f), _) => (resolve_file(&ReplicasFile::load(f)?, &cfg)?, None),
        (None, Some(p)) => {
            let (r, st) = resolve_plan(a, p, &cfg, &transport)?;
            (r, Some(st))
        }
        (None, None) => anyhow::bail!("--replicas FILE or --cluster-plan FILE is required"),
    };
    anyhow::ensure!(!resolved.is_empty(), "the cluster has no replica");
    // Every node that runs a part of a replica holds the whole model (every shard of a split
    // one) before anything starts, so a missing or partial copy fails here with its node named.
    if !a.fake {
        let mut nodes: Vec<NodeId> = Vec::new();
        for r in &resolved {
            for n in r.spec.nodes(&r.leader) {
                if !nodes.contains(&n) {
                    nodes.push(n);
                }
            }
        }
        for r in &resolved {
            let unreached =
                client::check_model(transport.as_ref(), &cfg, &nodes, &r.spec.model_sha256)
                    .map_err(anyhow::Error::msg)?;
            for n in unreached {
                tracing::warn!(node = %n, "model not checked: the node is not reachable yet");
            }
        }
    }

    // The replica the head leads itself runs in process; without one, the head only tokenizes.
    check_one_head_led(resolved.iter(), &me)?;
    let local = resolved.iter().any(|r| runs_in_process(r, &me));
    let mut local_store: Option<Arc<dyn PrefixStore>> = None;
    let (front, front_cap): (Arc<dyn GatewayEngine>, usize) = if local && a.fake {
        build_engine(a, telemetry)?
    } else if local {
        // The scheduler itself, not only its engine face: its cache is what other replicas fetch.
        let (s, cap) = build_scheduler(a, telemetry)?;
        local_store = Some(Arc::new(crate::replica::SchedulerStore(Arc::clone(&s))));
        (s as Arc<dyn GatewayEngine>, cap)
    } else {
        (front_engine(a, telemetry)?, a.parallel as usize)
    };
    let mut targets = Vec::new();
    let mut remote_specs = Vec::new();
    let mut total = 0usize;
    for (i, r) in resolved.iter().enumerate() {
        let cap = r.cap.map_or(front_cap, |c| c as usize);
        total += cap;
        if runs_in_process(r, &me) {
            targets.push(ReplicaTarget {
                engine: Arc::clone(&front),
                remote: None,
                cap: u32::try_from(cap)?,
            });
            continue;
        }
        let peer = cfg
            .peer(&r.leader)
            .with_context(|| format!("`{}` is not a node of cluster.toml", r.leader))?;
        let remote = tetraeder_control::replica::RemoteEngine::new(
            Arc::clone(&front),
            Arc::clone(&transport),
            peer,
            u32::try_from(i)?,
        );
        remote_specs.push((u32::try_from(i)?, r.leader.clone(), r.spec.clone()));
        targets.push(ReplicaTarget {
            engine: Arc::clone(&remote) as Arc<dyn GatewayEngine>,
            remote: Some(remote),
            cap: u32::try_from(cap)?,
        });
    }
    // The hardware graph of a plan (or `--cluster-hw` next to a replicas file) prices fetches.
    let hw = match &plan_state {
        Some(st) => Some(st.hw.clone()),
        None if a.cluster_hw.is_some() => Some(hardware(a, &cfg, &transport)?),
        None => None,
    };
    let policy: Option<Arc<dyn FetchPolicy>> = hw.map(|hw| {
        Arc::new(PricedFetch::new(
            hw,
            resolved.iter().map(|r| r.leader.0.clone()).collect(),
            resolved.iter().map(|r| r.pp_tps).collect(),
        )) as Arc<dyn FetchPolicy>
    });
    let engine = Arc::new(ClusterEngine::new(front, targets, MIN_PREFIX, policy));
    let mut prefix_file = None;
    for (i, r) in resolved.iter().enumerate() {
        if runs_in_process(r, &me) {
            let i = u32::try_from(i)?;
            if let Some(store) = &local_store {
                prefix_file = Some(serve_local_prefixes(
                    &engine,
                    i,
                    Arc::clone(store),
                    &a.node_state,
                )?);
            }
            engine.set_up(i, true);
        }
    }
    let head = ClusterHead::start(&engine, &monitor, &transport, remote_specs);
    let watcher =
        plan_state.map(|st| start_replanner(&cfg, &monitor, &engine, &head, st, telemetry));
    eprintln!(
        "cluster {}: {} replicas, {total} requests at once",
        resolved.len(),
        resolved.len()
    );
    Ok((
        Arc::clone(&engine) as Arc<dyn GatewayEngine>,
        total,
        HeadGuard {
            _watcher: watcher,
            _engine: engine,
            _head: head,
            _monitor: monitor,
            _prefix_file: prefix_file,
        },
    ))
}

/// Replans on node changes: `Trigger::NodeChange` to the replanner of the cluster plan, with the
/// planner as the plan source and the head's replicas as the engine.
fn start_replanner(
    cfg: &ClusterConfig,
    monitor: &Arc<Monitor>,
    engine: &Arc<ClusterEngine>,
    head: &ClusterHead,
    st: PlanState,
    telemetry: &Arc<Telemetry>,
) -> ReplanWatcher {
    let (hw, layout, sha, plan) = (st.hw.clone(), st.layout.clone(), st.sha.clone(), st.plan);
    let derive: Derive = {
        let (hw, cfg, sha, layout) = (hw.clone(), cfg.clone(), sha, layout.clone());
        Arc::new(move |p: &Plan| {
            // The prediction of the plan being applied, not of the one it replaces.
            let priced = PlanState {
                plan: p.clone(),
                hw: hw.clone(),
                layout: layout.clone(),
                sha: sha.clone(),
            };
            specs_from_plan(p, &hw, &cfg, &sha, &price(&priced).0)
        })
    };
    let control = ClusterControl::new(
        Arc::clone(engine),
        head.specs(),
        derive,
        DRAIN_TIMEOUT,
        JOIN_TIMEOUT,
    );
    let source = ClusterPlanSource {
        hw,
        layout,
        monitor: Arc::clone(monitor),
    };
    let metrics = telemetry.metrics().clone();
    ReplanWatcher::start(monitor, move || {
        let mut r = Replanner::new(
            Config {
                // A node change is not a tuning: no dwell between two of them.
                min_dwell_s: 0.0,
                drain_timeout_s: DRAIN_TIMEOUT.as_secs_f64(),
                ..Config::default()
            },
            plan,
            source,
            control,
            SystemClock::new(),
        );
        r.on_outcome(Box::new(move |t: &Trigger, o: &Outcome, secs: f64| {
            metrics.replan(t.label(), o.label());
            metrics.replan_duration(Duration::from_secs_f64(secs));
        }));
        r
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolved(leader: &str, nodes: &[&str]) -> Resolved {
        Resolved {
            leader: NodeId::new(leader),
            cap: None,
            spec: StartSpec {
                plan: String::new(),
                model_sha256: "0".repeat(64),
                segments: nodes
                    .iter()
                    .map(|n| SegmentAssign {
                        peer: tetraeder_transport::Peer {
                            id: NodeId::new(*n),
                            addr: "127.0.0.1:1".into(),
                        },
                        plan: String::new(),
                    })
                    .collect(),
                predicted_s: Vec::new(),
            },
            pp_tps: 0.0,
        }
    }

    /// hardware: none. Two whole-model replicas led by the head are refused with a message that
    /// says why; a replica the head leads across nodes runs through its daemon, so it does not
    /// count (rehearsal 2: such a plan used to run the whole model in the head's process).
    #[test]
    fn a_plan_with_two_head_led_replicas_is_refused() {
        let a = NodeId::new("a");
        let whole = |l: &str| resolved(l, &[]);
        assert!(check_one_head_led([whole("a"), whole("b"), whole("b")].iter(), &a).is_ok());
        assert!(check_one_head_led([whole("b"), whole("b")].iter(), &a).is_ok());
        let e = check_one_head_led([whole("a"), whole("b"), whole("a")].iter(), &a).unwrap_err();
        assert!(
            e.to_string().contains("2 replicas are led by the head"),
            "{e}"
        );
        // Led by the head, two segments: not in process.
        let split = resolved("a", &["a", "b"]);
        assert!(!runs_in_process(&split, &a));
        assert!(runs_in_process(&whole("a"), &a));
        assert!(check_one_head_led([split, whole("a")].iter(), &a).is_ok());
    }

    /// hardware: none. The head's prefix file goes when the head's guard does.
    #[test]
    fn the_head_removes_its_prefix_file_on_exit() {
        let dir =
            std::env::temp_dir().join(format!("tetraeder-prefix-gone-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = crate::node::head_prefix_file(&dir);
        std::fs::write(&file, "1 t").unwrap();
        drop(PrefixFile(file.clone()));
        assert!(!file.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    const HW: &str = include_str!("../../costmodel/fixtures/cluster/cluster_three_mixed.toml");

    fn source() -> ClusterPlanSource {
        let hw = HardwareGraph::from_toml(HW).unwrap();
        // `plan_without` does not read the monitor; it only has to exist.
        let t: Arc<dyn Transport> =
            Arc::new(tetraeder_transport::MemTransport::switchboard().node(&NodeId::new("a")));
        let cfg = ClusterConfig::parse(
            "cluster = \"x\"\nhead = \"a\"\n[[nodes]]\nid = \"a\"\naddr = \"127.0.0.1:1\"\nanchor = \"a.pem\"\n",
        )
        .unwrap();
        ClusterPlanSource {
            hw,
            layout: Some(tetraeder_planner::layout::fixtures::moe()),
            monitor: Arc::new(Monitor::start(t, cfg)),
        }
    }

    /// hardware: none. With a node forbidden the planner's plan computes on the others only;
    /// with all but one forbidden it is the single-node plan.
    #[test]
    fn the_replan_plan_avoids_the_nodes_that_left() {
        let s = source();
        let full = s.plan_without(&[]).unwrap();
        assert!(s.nodes_of(&full).len() >= 2);
        let without_c = s.plan_without(&["c".to_owned()]).unwrap();
        let nodes = s.nodes_of(&without_c);
        assert!(
            !nodes.contains(&"c".to_owned()) && !nodes.is_empty(),
            "{nodes:?}"
        );
        let alone = s.plan_without(&["b".to_owned(), "c".to_owned()]).unwrap();
        assert_eq!(s.nodes_of(&alone), vec!["a".to_owned()]);
    }
}
