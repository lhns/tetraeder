//! `tetraeder node`: the daemon of one box of a cluster (ADR 0091 step 3), and
//! `tetraeder node --drain` to tell a running daemon to stop taking assignments.
//!
//! The daemon listens on the address `cluster.toml` gives its id, answers the head's heartbeats,
//! reports its hardware graph and runs `tetraeder worker` children for the segments assigned
//! to it. The key stays on this box (`--key`, written by `tetraeder cluster cert`).

use anyhow::{Context, Result, anyhow, bail};
use clap::Args;
use std::ffi::OsString;
use std::io::{self};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use tetraeder_control::ClusterConfig;
use tetraeder_control::client;
use tetraeder_control::kv::relay_prefix;
use tetraeder_control::node::{Assignment, Daemon, Launcher, NodeOptions, RestartPolicy, WorkerIo};
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

/// Where a replica engine of this node writes its prefix listener's port (one engine per node
/// serves prefixes: the last one started).
fn prefix_port_file(state_dir: &std::path::Path) -> PathBuf {
    state_dir.join("prefix-port")
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
}

impl Launcher for ChildLauncher {
    fn launch(&self, a: &Assignment) -> io::Result<WorkerIo> {
        // The sha256 names a file: only hex, so it cannot name another path.
        if a.model_sha256.len() != 64 || !a.model_sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "model_sha256 is not 64 hex digits",
            ));
        }
        let model = self
            .models_dir
            .join(format!("{}.gguf", a.model_sha256.to_ascii_lowercase()));
        // A replica of the scripted engine (`--replica-arg --fake`) needs no model file.
        let fake = a.is_replica() && self.replica_args.iter().any(|x| x == "--fake");
        if !model.is_file() && !fake {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("no model {}", model.display()),
            ));
        }
        std::fs::create_dir_all(&self.state_dir)?;
        let mut cmd = Command::new(&self.exe);
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
                .env("TETRAEDER_REPLICA_PLAN", &plan)
                .env(
                    crate::replica::PREFIX_PORT_FILE_ENV,
                    prefix_port_file(&self.state_dir),
                );
            // A port left by an earlier engine must not be read for this one.
            let _ = std::fs::remove_file(prefix_port_file(&self.state_dir));
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
        Ok(WorkerIo {
            stdin: Box::new(stdin),
            stdout: Box::new(stdout),
            kill: Arc::new(move || {
                let mut c = child
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let _ = c.kill();
                let _ = c.wait();
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
    let profile = Arc::new(move || {
        let mut c = cached
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if c.is_none() {
            let (hw, notes) = crate::replan::source::HwSource::Discover.read(true)?;
            for n in notes {
                tracing::info!("hardware: {n}");
            }
            *c = Some(hw.to_json().map_err(|e| e.to_string())?);
        }
        Ok(c.clone().unwrap_or_default())
    });
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
    });
    let state_dir = a.state_dir.clone();
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
    daemon.set_prefix_handler(Arc::new(move |conn| {
        let port = std::fs::read_to_string(prefix_port_file(&state_dir))
            .ok()
            .and_then(|t| t.trim().parse::<u16>().ok());
        match port {
            Some(p) => relay_prefix(conn, std::net::SocketAddr::from(([127, 0, 0, 1], p))),
            None => (conn.close)(),
        }
    }));
    // The daemon runs until the process is stopped.
    loop {
        std::thread::park();
    }
}
