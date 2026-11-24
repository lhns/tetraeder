//! The real binaries, once (ADR 0091, step 5): `tetraeder cluster cert` for two nodes, two
//! `tetraeder node` processes on loopback TLS, each starting `tetraeder replica --fake` (the
//! scripted engine, no model) when the head asks. The head, in the test, joins both replicas to
//! its router, streams requests, kills node b's process and keeps going: requests that arrive
//! once the node is found down all succeed on node a; node b is started again and its replica
//! rejoins without anyone asking.
//!
//! Hardware label: cpu (loopback).

use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tetraeder_control::ClusterConfig;
use tetraeder_control::head::{ClusterEngine, ClusterHead, ReplicaTarget, StartSpec};
use tetraeder_control::monitor::Monitor;
use tetraeder_control::replica::RemoteEngine;
use tetraeder_gateway::engine::{Engine, GenEvent, GenerateRequest, SamplingParams};
use tetraeder_gateway::fake::{FakeEngine, serve_script};
use tetraeder_transport::{Identity, NodeId, SocketBuffers, TlsTransport, Transport};
use tokio_util::sync::CancellationToken;

const SHA: &str = "5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed";
const DOWN_MS: u64 = 2000;

struct Procs(Vec<Child>);

impl Drop for Procs {
    fn drop(&mut self) {
        for c in &mut self.0 {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

fn tetraeder() -> Command {
    Command::new(env!("CARGO_BIN_EXE_tetraeder"))
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn wait_listening(port: u16, what: &str) {
    let t0 = Instant::now();
    while t0.elapsed() < Duration::from_secs(60) {
        if TcpStream::connect(("127.0.0.1", port)).is_ok() {
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    panic!("{what} did not listen on {port}");
}

fn spawn_node(dir: &std::path::Path, cfg: &std::path::Path, n: &str, port: u16) -> Child {
    let c = tetraeder()
        .arg("node")
        .arg("--cluster")
        .arg(cfg)
        .arg("--key")
        .arg(dir.join(n).join("key.pem"))
        .arg("--cert")
        .arg(dir.join(n).join("cert.pem"))
        .arg("--models-dir")
        .arg(dir.join("models"))
        .arg("--state-dir")
        .arg(dir.join(format!("state-{n}")))
        .arg("--replica-arg=--fake")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    wait_listening(port, &format!("node {n}"));
    c
}

/// One request; `Ok(text)` for a finished one, `Err(message)` for the one error it ended with.
fn ask(e: &Arc<ClusterEngine>, text: &str) -> Result<String, String> {
    let prompt = e.tokenize("fake", text).unwrap();
    let mut rx = e
        .generate(GenerateRequest {
            model: "fake".into(),
            prompt,
            params: SamplingParams {
                max_tokens: 64,
                ..SamplingParams::default()
            },
            constraint: None,
            cancel: CancellationToken::new(),
        })
        .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    let mut ended = 0;
    let mut err = None;
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        match rx.try_recv() {
            Ok(GenEvent::Token(t)) => out.push(t as u8),
            Ok(GenEvent::Finished { .. }) => ended += 1,
            Ok(GenEvent::Error(m) | GenEvent::ContextExceeded(m)) => {
                ended += 1;
                err = Some(m);
            }
            Ok(GenEvent::Probs(_)) => {}
            Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                std::thread::sleep(Duration::from_millis(2));
            }
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => break,
        }
    }
    assert_eq!(ended, 1, "a request ends exactly once");
    match err {
        Some(m) => Err(m),
        None => Ok(String::from_utf8_lossy(&out).into_owned()),
    }
}

fn wait_for(what: &str, within: Duration, mut f: impl FnMut() -> bool) -> Duration {
    let t0 = Instant::now();
    while t0.elapsed() < within {
        if f() {
            return t0.elapsed();
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    panic!("{what} did not happen within {within:?}");
}

/// hardware: cpu. Two node daemons and their replica engines as processes, the head in the test.
#[test]
fn cpu_two_replica_processes_keep_serving_when_one_node_is_killed() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("replica-smoke-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("models")).unwrap();
    for n in ["a", "b"] {
        let o = tetraeder()
            .args(["cluster", "cert", "--node", n, "--out"])
            .arg(dir.join(n))
            .output()
            .unwrap();
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    }
    let ports = [free_port(), free_port()];
    let mut toml = format!("cluster = \"smoke\"\nhead = \"a\"\nnode_down_after_ms = {DOWN_MS}\n");
    for (n, p) in ["a", "b"].iter().zip(ports) {
        toml += &format!(
            "[[nodes]]\nid = \"{n}\"\naddr = \"127.0.0.1:{p}\"\nanchor = \"{n}/cert.pem\"\n"
        );
    }
    let cfg_path = dir.join("cluster.toml");
    std::fs::write(&cfg_path, toml).unwrap();

    let mut procs = Procs(Vec::new());
    for (n, p) in ["a", "b"].iter().zip(ports) {
        procs.0.push(spawn_node(&dir, &cfg_path, n, p));
    }

    // The head: node a's key and certificate, dial only.
    let cfg = ClusterConfig::load(&cfg_path).unwrap();
    let ident = Identity::from_pem(
        &std::fs::read(dir.join("a/cert.pem")).unwrap(),
        &std::fs::read(dir.join("a/key.pem")).unwrap(),
    )
    .unwrap();
    let t: Arc<dyn Transport> = Arc::new(
        TlsTransport::new(
            &ident,
            &cfg.anchors().unwrap(),
            &cfg.ids(),
            None,
            SocketBuffers::Auto,
        )
        .unwrap(),
    );
    let monitor = Arc::new(Monitor::start(Arc::clone(&t), cfg.clone()));
    let front: Arc<dyn Engine> = Arc::new(FakeEngine::new(serve_script));
    let remotes: Vec<Arc<RemoteEngine>> = ["a", "b"]
        .iter()
        .enumerate()
        .map(|(i, n)| {
            RemoteEngine::new(
                Arc::clone(&front),
                Arc::clone(&t),
                cfg.peer(&NodeId::new(*n)).unwrap(),
                i as u32,
            )
        })
        .collect();
    let engine = Arc::new(ClusterEngine::new(
        Arc::clone(&front),
        remotes
            .iter()
            .map(|r| ReplicaTarget {
                engine: Arc::clone(r) as Arc<dyn Engine>,
                remote: Some(Arc::clone(r)),
                cap: 2,
            })
            .collect(),
        8,
        None,
    ));
    let spec = || StartSpec {
        plan: String::new(),
        model_sha256: SHA.into(),
        segments: Vec::new(),
        predicted_s: Vec::new(),
    };
    let _head = ClusterHead::start(
        &engine,
        &monitor,
        &t,
        vec![(0, NodeId::new("a"), spec()), (1, NodeId::new("b"), spec())],
    );
    let joined = wait_for("both replicas join", Duration::from_secs(120), || {
        engine.up_count() == 2
    });

    // Both replicas serve.
    for i in 0..20 {
        assert_eq!(
            ask(&engine, &format!("hello {i}")).unwrap(),
            format!("You said: hello {i}")
        );
    }
    assert!(engine.counts().ok >= 20);

    // Kill node b's process: its replica engine ends with it.
    let killed = Instant::now();
    procs.0[1].kill().unwrap();
    let _ = procs.0[1].wait();
    let (mut errors, mut ok_after) = (0, 0);
    let bound = Duration::from_millis(DOWN_MS + 1000);
    while killed.elapsed() < bound + Duration::from_secs(4) {
        let sent = killed.elapsed();
        match ask(&engine, "after the kill") {
            Ok(t) => {
                assert_eq!(t, "You said: after the kill");
                if sent >= bound {
                    ok_after += 1;
                }
            }
            Err(m) => {
                errors += 1;
                assert!(
                    sent < bound && m.contains("replica 1"),
                    "an error only for a request of the dead replica, within the bound: {m} at {sent:?}"
                );
            }
        }
    }
    assert!(ok_after > 0, "requests flow on node a");
    assert!(!engine.is_up(1) && engine.is_up(0));

    // Node b is started again: its replica joins the router on its own.
    procs.0[1] = spawn_node(&dir, &cfg_path, "b", ports[1]);
    let rejoin = wait_for("replica 1 rejoins", Duration::from_secs(60), || {
        engine.is_up(1)
    });
    for i in 0..10 {
        assert_eq!(
            ask(&engine, &format!("again {i}")).unwrap(),
            format!("You said: again {i}")
        );
    }
    println!(
        "replica smoke processes 2 transport tls joined_s {:.1} in_flight_errors {errors} ok_after_bound {ok_after} rejoin_s {:.1}",
        joined.as_secs_f64(),
        rejoin.as_secs_f64()
    );
    for c in &mut procs.0 {
        assert!(c.try_wait().unwrap().is_none(), "a daemon exited");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
