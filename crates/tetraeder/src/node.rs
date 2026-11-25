//! `tetraeder node`: the daemon of one box of a cluster (ADR 0091 step 3), and
//! `tetraeder node --drain` to tell a running daemon to stop taking assignments.
//!
//! The daemon listens on the address `cluster.toml` gives its id, answers the head's heartbeats,
//! reports its hardware graph and runs `tetraeder worker` children for the segments assigned
//! to it. The key stays on this box (`--key`, written by `tetraeder cluster cert`).

use anyhow::{Context, Result, anyhow, bail};
use clap::Args;
use std::collections::HashMap;
use std::ffi::OsString;
use std::io::{self};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tetraeder_control::ClusterConfig;
use tetraeder_control::client;
use tetraeder_control::kv::{new_token, relay_prefix};
use tetraeder_control::node::{
    Assignment, Daemon, Launcher, ModelStat, NodeOptions, RestartPolicy, WorkerIo,
};
use tetraeder_control::proto::NodeRequest;
use tetraeder_telemetry::{Telemetry, TelemetryConfig};
use tetraeder_transport::{Identity, NodeId, SocketBuffers, TlsTransport, Transport};

/// The cluster file and this process's identity: shared by the daemon and the cluster tools.
#[derive(Args, Clone)]
pub struct ClusterArgs {
    /// `cluster.toml` (nodes, addresses, pinned certificates; never keys).
    #[arg(long, default_value = "cluster.toml")]
    pub cluster: PathBuf,
    /// This node's private key (from `tetraeder cluster cert`).
    #[arg(long, default_value = "key.pem")]
    pub key: PathBuf,
    /// This node's certificate.
    #[arg(long, default_value = "cert.pem")]
    pub cert: PathBuf,
}

impl ClusterArgs {
    pub fn load(&self) -> Result<(ClusterConfig, Identity)> {
        let cfg = ClusterConfig::load(&self.cluster).map_err(anyhow::Error::msg)?;
        let id = Identity::from_pem(
            &std::fs::read(&self.cert).with_context(|| format!("read {}", self.cert.display()))?,
            &std::fs::read(&self.key).with_context(|| format!("read {}", self.key.display()))?,
        )?;
        let me = id.node_id()?;
        anyhow::ensure!(
            cfg.node(&me).is_some(),
            "the certificate names `{me}`, which cluster.toml does not list"
        );
        Ok((cfg, id))
    }

    /// A transport of this identity: listening on `listen`, or dial-only.
    pub fn transport(
        &self,
        cfg: &ClusterConfig,
        id: &Identity,
        listen: Option<&str>,
        buffers: SocketBuffers,
    ) -> Result<TlsTransport> {
        let anchors = cfg.anchors().map_err(anyhow::Error::msg)?;
        TlsTransport::new(id, &anchors, &cfg.ids(), listen, buffers)
    }
}

#[derive(Args)]
pub struct NodeArgs {
    #[command(flatten)]
    pub cluster: ClusterArgs,
    /// Tell node `--id` to drain (no new assignments; the head sees it as `Draining`). Run it
    /// with the head's key and certificate.
    #[arg(long)]
    pub drain: bool,
    /// With `--drain`, the node to drain.
    #[arg(long)]
    pub id: Option<String>,
    /// Socket buffers of the listener: `auto` (kernel autotuning) or `SEND,RECV` bytes.
    #[arg(long, default_value = "auto", value_parser = parse_buffers)]
    pub buffers: SocketBuffers,
    /// Where models live, as `<sha256>.gguf`; a segment's `model_sha256` names the file.
    #[arg(long, default_value = "models")]
    pub models_dir: PathBuf,
    /// Where the plans of assigned segments are written.
    #[arg(long, default_value = "node-state")]
    pub state_dir: PathBuf,
    /// An argument for every `tetraeder worker` child (the engine flags), repeatable.
    #[arg(long = "worker-arg")]
    pub worker_arg: Vec<OsString>,
    /// An argument for every `tetraeder replica` child (the engine flags), repeatable.
    #[arg(long = "replica-arg")]
    pub replica_arg: Vec<OsString>,
}

pub fn parse_buffers(s: &str) -> Result<SocketBuffers, String> {
    if s == "auto" {
        return Ok(SocketBuffers::Auto);
    }
    let (a, b) = s.split_once(',').ok_or("expected `auto` or `SEND,RECV`")?;
    let n = |x: &str| x.parse::<usize>().map_err(|e| format!("`{x}`: {e}"));
    Ok(SocketBuffers::Explicit {
        send: n(a)?,
        recv: n(b)?,
    })
}

/// Where the replica engine `replica` of this node writes its prefix listener's port.
fn prefix_port_file(state_dir: &std::path::Path, replica: u32) -> PathBuf {
    state_dir.join(format!("prefix-port-{replica}"))
}

/// Where `serve --cluster` writes the prefix listener of the replica it leads in process, as
/// `<port> <token>`: the head's replica is not a child of this daemon, so the daemon finds it
/// here (both default to the same `--state-dir`).
pub(crate) fn head_prefix_file(state_dir: &std::path::Path) -> PathBuf {
    state_dir.join("prefix-head")
}

/// The prefix listener of one launch of a replica engine: its port file and the token the
/// daemon must present (a fresh one per launch, passed to the child by environment).
struct PrefixTarget {
    file: PathBuf,
    token: String,
    launch: u64,
}

/// The prefix listeners of the replica engines this node runs, one entry per replica from its
/// launch until it ends. A fetch goes to the newest launch (a `Prefix` connection does not name
/// a replica; one engine per node serves prefixes in practice).
#[derive(Default)]
struct PrefixTargets {
    by_replica: Mutex<HashMap<u32, PrefixTarget>>,
    launches: AtomicU64,
    /// The head process's own listener (`head_prefix_file`), used when no child has one.
    head_file: Option<PathBuf>,
}

impl PrefixTargets {
    fn with_head_file(file: PathBuf) -> Self {
        Self {
            head_file: Some(file),
            ..Self::default()
        }
    }

    /// The head process's listener: `<port> <token>` in its file.
    fn head(&self) -> Option<(std::net::SocketAddr, String)> {
        let text = std::fs::read_to_string(self.head_file.as_ref()?).ok()?;
        let (port, token) = text.trim().split_once(' ')?;
        Some((
            std::net::SocketAddr::from(([127, 0, 0, 1], port.parse::<u16>().ok()?)),
            token.to_owned(),
        ))
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<u32, PrefixTarget>> {
        self.by_replica
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Registers a launch of `replica`; returns its token and launch number.
    fn add(&self, replica: u32, file: PathBuf) -> (String, u64) {
        let token = new_token();
        let launch = self.launches.fetch_add(1, Ordering::SeqCst);
        self.lock().insert(
            replica,
            PrefixTarget {
                file,
                token: token.clone(),
                launch,
            },
        );
        (token, launch)
    }

    /// The launch ended: its entry and port file go, unless a newer launch replaced it.
    fn remove(&self, replica: u32, launch: u64) {
        let mut m = self.lock();
        if m.get(&replica).is_some_and(|t| t.launch == launch)
            && let Some(t) = m.remove(&replica)
        {
            let _ = std::fs::remove_file(&t.file);
        }
    }

    /// The loopback address and token of the newest listener that has written its port.
    fn current(&self) -> Option<(std::net::SocketAddr, String)> {
        let child = {
            let m = self.lock();
            let mut all: Vec<&PrefixTarget> = m.values().collect();
            all.sort_by_key(|t| std::cmp::Reverse(t.launch));
            all.into_iter().find_map(|t| {
                let port = std::fs::read_to_string(&t.file)
                    .ok()?
                    .trim()
                    .parse::<u16>()
                    .ok()?;
                Some((
                    std::net::SocketAddr::from(([127, 0, 0, 1], port)),
                    t.token.clone(),
                ))
            })
        };
        child.or_else(|| self.head())
    }
}

/// Starts `tetraeder worker` for an assignment.
struct ChildLauncher {
    exe: PathBuf,
    models_dir: PathBuf,
    state_dir: PathBuf,
    args: Vec<OsString>,
    replica_args: Vec<OsString>,
    /// `--cluster`, `--key` and `--cert` of this node, handed to a replica that spans nodes.
    identity: Vec<OsString>,
    prefix: Arc<PrefixTargets>,
}

impl Launcher for ChildLauncher {
    fn check_model(&self, sha256: &str) -> io::Result<ModelStat> {
        match tetraeder_model::model_id::resolve(&self.models_dir, sha256) {
            Ok(m) => Ok(ModelStat {
                shards: u32::try_from(m.shards).unwrap_or(u32::MAX),
                bytes: m.bytes,
            }),
            // The scripted engine needs no model file.
            Err(_) if self.replica_args.iter().any(|x| x == "--fake") => Ok(ModelStat {
                shards: 0,
                bytes: 0,
            }),
            Err(e) => Err(e),
        }
    }

    fn launch(&self, a: &Assignment) -> io::Result<WorkerIo> {
        // A replica of the scripted engine (`--replica-arg --fake`) needs no model file.
        let fake = a.is_replica() && self.replica_args.iter().any(|x| x == "--fake");
        // A single file, or the first shard of a split model whose manifest and shards check.
        let model = match tetraeder_model::model_id::resolve(&self.models_dir, &a.model_sha256) {
            Ok(m) => m.path,
            Err(_) if fake => PathBuf::new(),
            Err(e) => return Err(e),
        };
        crate::replica::create_private_dir(&self.state_dir)?;
        let mut cmd = Command::new(&self.exe);
        // Set for a replica engine: its prefix listener's entry, removed when the child ends.
        let mut prefix_launch: Option<(u32, u64)> = None;
        if a.is_replica() {
            let plan = self
                .state_dir
                .join(format!("replica-{}.plan.toml", a.replica));
            std::fs::write(&plan, &a.plan)?;
            cmd.arg("replica");
            if model.is_file() {
                cmd.arg("--model").arg(&model);
            }
            cmd.args(&self.replica_args)
                .args(&self.identity)
                .env(crate::replica::REPLICA_ENV, a.replica.to_string())
                .env(crate::replica::SEGMENTS_ENV, a.segments.join(","))
                .env(
                    crate::replica::PREDICTED_ENV,
                    a.predicted_s
                        .iter()
                        .map(f64::to_string)
                        .collect::<Vec<_>>()
                        .join(","),
                )
                .env("TETRAEDER_REPLICA_PLAN", &plan);
            // A port left by an earlier engine must not be read for this one.
            let file = prefix_port_file(&self.state_dir, a.replica);
            let _ = std::fs::remove_file(&file);
            let (token, launch) = self.prefix.add(a.replica, file.clone());
            prefix_launch = Some((a.replica, launch));
            cmd.env(crate::replica::PREFIX_PORT_FILE_ENV, &file)
                .env(crate::replica::PREFIX_TOKEN_ENV, token);
        } else {
            let plan = self.state_dir.join(format!(
                "replica-{}-segment-{}.plan.toml",
                a.replica, a.segment
            ));
            std::fs::write(&plan, &a.plan)?;
            cmd.arg("worker")
                .arg("--model")
                .arg(&model)
                .args(&self.args)
                .env("TETRAEDER_SEGMENT_PLAN", &plan);
        }
        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| io::Error::other("no stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| io::Error::other("no stdout"))?;
        let child: Arc<Mutex<Child>> = Arc::new(Mutex::new(child));
        let prefix = Arc::clone(&self.prefix);
        Ok(WorkerIo {
            stdin: Box::new(stdin),
            stdout: Box::new(stdout),
            kill: Arc::new(move || {
                let mut c = child
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let _ = c.kill();
                let _ = c.wait();
                if let Some((replica, launch)) = prefix_launch {
                    prefix.remove(replica, launch);
                }
            }),
        })
    }
}

pub fn run(a: &NodeArgs) -> Result<()> {
    let (cfg, ident) = a.cluster.load()?;
    if a.drain {
        let target = NodeId::new(
            a.id.clone()
                .ok_or_else(|| anyhow!("--drain needs --id <node>"))?,
        );
        let me = ident.node_id()?;
        anyhow::ensure!(
            me == cfg.head,
            "--drain runs with the head's key and certificate"
        );
        let peer = cfg
            .peer(&target)
            .ok_or_else(|| anyhow!("`{target}` is not a node of cluster.toml"))?;
        let t = a
            .cluster
            .transport(&cfg, &ident, None, SocketBuffers::Auto)?;
        client::request(&t, &peer, &NodeRequest::Drain, &[]).map_err(anyhow::Error::msg)?;
        println!("node {target} drains");
        return Ok(());
    }
    let me = ident.node_id()?;
    if let Some(id) = &a.id
        && *id != me.0
    {
        bail!("--id {id} does not match the certificate (`{me}`)");
    }
    let telemetry = Arc::new(Telemetry::new(&TelemetryConfig::from_env())?);
    telemetry.install_global()?;
    let listen = cfg
        .listen_addr(&me)
        .ok_or_else(|| anyhow!("`{me}` is not listed"))?
        .to_owned();
    let transport = Arc::new(
        a.cluster
            .transport(&cfg, &ident, Some(&listen), a.buffers)?,
    );
    tracing::info!(node = %me, listen = %listen, "tetraeder node");
    println!("node {me} listening on {listen}");

    // Explicit buffers for a probe: a dial-only transport of the same identity.
    let (anchors, ids) = (cfg.anchors().map_err(anyhow::Error::msg)?, cfg.ids());
    let own: Arc<dyn Transport> = Arc::clone(&transport) as Arc<dyn Transport>;
    let ident = Arc::new(ident);
    let dial = Arc::new(
        move |b: Option<(usize, usize)>| -> io::Result<Arc<dyn Transport>> {
            match b {
                None => Ok(Arc::clone(&own)),
                Some((send, recv)) => {
                    let id = Identity {
                        cert: ident.cert.clone(),
                        key: ident.key.clone_key(),
                    };
                    TlsTransport::new(
                        &id,
                        &anchors,
                        &ids,
                        None,
                        SocketBuffers::Explicit { send, recv },
                    )
                    .map(|t| Arc::new(t) as Arc<dyn Transport>)
                    .map_err(io::Error::other)
                }
            }
        },
    );
    let cached: Mutex<Option<String>> = Mutex::new(None);
    let models_dir = a.models_dir.clone();
    let profile = Arc::new(move || {
        let mut c = cached
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if c.is_none() {
            let (hw, notes) =
                crate::replan::source::HwSource::Discover.read_in(true, Some(&models_dir))?;
            for n in notes {
                tracing::info!("hardware: {n}");
            }
            *c = Some(hw.to_json().map_err(|e| e.to_string())?);
        }
        Ok(c.clone().unwrap_or_default())
    });
    let prefix = Arc::new(PrefixTargets::with_head_file(head_prefix_file(
        &a.state_dir,
    )));
    let launcher = Arc::new(ChildLauncher {
        exe: std::env::current_exe()?,
        models_dir: a.models_dir.clone(),
        state_dir: a.state_dir.clone(),
        args: a.worker_arg.clone(),
        replica_args: a.replica_arg.clone(),
        identity: vec![
            "--cluster".into(),
            a.cluster.cluster.clone().into(),
            "--key".into(),
            a.cluster.key.clone().into(),
            "--cert".into(),
            a.cluster.cert.clone().into(),
        ],
        prefix: Arc::clone(&prefix),
    });
    let daemon = Daemon::start(
        transport,
        NodeOptions {
            cfg,
            launcher,
            profile,
            dial,
            restart: RestartPolicy::default(),
        },
    );
    // Other nodes fetch prefixes from this node's replica: its child listens on loopback and the
    // daemon relays the authenticated connection to it.
    daemon.set_prefix_handler(Arc::new(move |conn| match prefix.current() {
        Some((addr, token)) => relay_prefix(conn, addr, &token),
        None => {
            tracing::warn!(peer = %conn.peer, "a prefix connection arrived and no replica listens for it; closed");
            (conn.close)();
        }
    }));
    // The daemon runs until the process is stopped.
    loop {
        std::thread::park();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// hardware: none. The daemon relays `Prefix` connections to the head process's listener when
    /// no child replica has one: the head's replica runs in `serve`, not under the daemon.
    #[test]
    fn the_daemon_finds_the_head_processs_prefix_listener() {
        let dir =
            std::env::temp_dir().join(format!("tetraeder-prefix-head-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = head_prefix_file(&dir);
        let targets = PrefixTargets::with_head_file(file.clone());
        assert!(targets.current().is_none(), "no listener yet");
        std::fs::write(&file, "40123 abc123\n").unwrap();
        let (addr, token) = targets.current().expect("the head's listener");
        assert_eq!((addr.port(), token.as_str()), (40123, "abc123"));
        // A child replica's listener wins over the head's.
        let child = dir.join("prefix-port-1");
        std::fs::write(&child, "40999").unwrap();
        let (t, _) = targets.add(1, child);
        let (addr, token) = targets.current().unwrap();
        assert_eq!((addr.port(), token), (40999, t));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
