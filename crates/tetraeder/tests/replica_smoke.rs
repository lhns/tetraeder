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
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tetraeder_control::ClusterConfig;
use tetraeder_control::head::{ClusterEngine, ClusterHead, ReplicaTarget, StartSpec};
use tetraeder_control::monitor::Monitor;
use tetraeder_control::replica::RemoteEngine;
use tetraeder_gateway::engine::{Engine, GenEvent, GenerateRequest, SamplingParams};
use tetraeder_gateway::fake::{FakeEngine, serve_script};
use tetraeder_model::{BackendChoice, LoadOptions};
use tetraeder_scheduler::{Scheduler, SchedulerConfig, load_moe};
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

fn spawn_node(
    dir: &std::path::Path,
    cfg: &std::path::Path,
    n: &str,
    port: u16,
    replica_args: &[&str],
) -> Child {
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
        .args(replica_args.iter().map(|a| format!("--replica-arg={a}")))
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
        procs.0.push(spawn_node(&dir, &cfg_path, n, p, &["--fake"]));
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
    procs.0[1] = spawn_node(&dir, &cfg_path, "b", ports[1], &["--fake"]);
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

/// The tokens of one greedy request on `e`, or the one error it ended with.
fn greedy(e: &dyn Engine, prompt: &[u32], n: u32) -> Result<Vec<u32>, String> {
    let mut rx = e
        .generate(GenerateRequest {
            model: "tiny".into(),
            prompt: prompt.to_vec(),
            params: SamplingParams {
                max_tokens: n,
                temperature: Some(0.0),
                ..SamplingParams::default()
            },
            constraint: None,
            cancel: CancellationToken::new(),
        })
        .map_err(|e| e.to_string())?;
    let (mut out, mut ended, mut err) = (Vec::new(), 0, None);
    let deadline = Instant::now() + Duration::from_secs(60);
    while Instant::now() < deadline {
        match rx.try_recv() {
            Ok(GenEvent::Token(t)) => out.push(t),
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
    err.map_or(Ok(out), Err)
}

/// Four clients of greedy requests through the router until `until`; `(ok, failed)` in `done`.
/// A failure must name replica 1 and come from a request sent before `bound` after `from`.
fn drive(
    engine: &Arc<ClusterEngine>,
    prompt: &[u32],
    until: Instant,
    from: Instant,
    bound: Duration,
    done: &Arc<Mutex<(u32, u32)>>,
) {
    let hs: Vec<_> = (0..4u32)
        .map(|c| {
            let (e, p, d) = (Arc::clone(engine), prompt.to_vec(), Arc::clone(done));
            std::thread::spawn(move || {
                while Instant::now() < until {
                    let sent = from.elapsed();
                    let r = greedy(e.as_ref(), &p, 8 + c);
                    let mut g = d.lock().unwrap();
                    match r {
                        Ok(t) => {
                            assert_eq!(t.len(), (8 + c) as usize);
                            g.0 += 1;
                        }
                        Err(m) => {
                            g.1 += 1;
                            assert!(
                                sent < bound && m.contains("replica 1"),
                                "an error only for a request of the dead replica, within the bound: {m} at {sent:?}"
                            );
                        }
                    }
                }
            })
        })
        .collect();
    for h in hs {
        h.join().unwrap();
    }
}

/// hardware: cpu. Two real engines (the tiny MoE under the scheduler, in `tetraeder replica`
/// children of two daemons): a greedy request on either replica gives the tokens of the same
/// model run in the head's own process, and Case F's rules hold when one node is killed.
#[test]
fn cpu_two_moe_replica_processes_decode_like_in_process_and_survive_a_kill() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("replica-moe-smoke-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("models")).unwrap();
    let model = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-qwen3moe-f16.gguf");
    std::fs::copy(&model, dir.join("models").join(format!("{SHA}.gguf"))).unwrap();
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
    let args = [
        "--engine",
        "moe",
        "--ngl",
        "0",
        "--threads",
        "2",
        "--ctx",
        "256",
        "--parallel",
        "2",
    ];
    let mut procs = Procs(Vec::new());
    for (n, p) in ["a", "b"].iter().zip(ports) {
        procs.0.push(spawn_node(&dir, &cfg_path, n, p, &args));
    }

    // The head's own engine: the reference, and the tokenizer of the router.
    let front: Arc<Scheduler> = {
        let path = model.clone();
        let mut c = SchedulerConfig::new(&model);
        c.alias = Some("tiny".into());
        c.parallel = 2;
        Arc::new(
            Scheduler::start(c, move || {
                let mut o = LoadOptions::new(BackendChoice::Cpu { threads: Some(2) });
                o.n_ctx = 256;
                o.n_seq_max = 2;
                load_moe(&path, &o)
            })
            .unwrap(),
        )
    };
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
    let front_engine: Arc<dyn Engine> = Arc::clone(&front) as _;
    let remotes: Vec<Arc<RemoteEngine>> = ["a", "b"]
        .iter()
        .enumerate()
        .map(|(i, n)| {
            RemoteEngine::new(
                Arc::clone(&front_engine),
                Arc::clone(&t),
                cfg.peer(&NodeId::new(*n)).unwrap(),
                i as u32,
            )
        })
        .collect();
    let engine = Arc::new(ClusterEngine::new(
        Arc::clone(&front_engine),
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
    wait_for("both engines join", Duration::from_secs(180), || {
        engine.up_count() == 2
    });

    let prompt = front.tokenize("tiny", "Hello there").unwrap();
    let want = greedy(front.as_ref(), &prompt, 12).unwrap();
    assert_eq!(want.len(), 12);
    // Each replica, asked directly, gives the in-process tokens (one request at a time, so the
    // batches are the same).
    for (i, r) in remotes.iter().enumerate() {
        assert_eq!(
            greedy(r.as_ref(), &prompt, 12).unwrap(),
            want,
            "replica {i}"
        );
    }
    // Through the router, concurrently, every request succeeds.
    let done = Arc::new(Mutex::new((0u32, 0u32)));
    drive(
        &engine,
        &prompt,
        Instant::now() + Duration::from_secs(2),
        Instant::now(),
        Duration::ZERO,
        &done,
    );
    assert_eq!(
        done.lock().unwrap().1,
        0,
        "no request fails while all is well"
    );

    // Kill node b under load.
    let bound = Duration::from_millis(DOWN_MS + 1000);
    let killed = Instant::now();
    let killer = std::thread::spawn({
        let (engine, prompt) = (Arc::clone(&engine), prompt.clone());
        move || {
            drive(
                &engine,
                &prompt,
                killed + bound + Duration::from_secs(3),
                killed,
                bound,
                &Arc::new(Mutex::new((0, 0))),
            );
            engine
        }
    });
    std::thread::sleep(Duration::from_millis(300));
    procs.0[1].kill().unwrap();
    let _ = procs.0[1].wait();
    let engine = killer.join().unwrap();
    assert!(!engine.is_up(1) && engine.is_up(0));

    // Started again, node b's engine joins and decodes the same tokens.
    procs.0[1] = spawn_node(&dir, &cfg_path, "b", ports[1], &args);
    let rejoin = wait_for("replica 1 rejoins", Duration::from_secs(120), || {
        engine.is_up(1)
    });
    assert_eq!(greedy(remotes[1].as_ref(), &prompt, 12).unwrap(), want);
    println!(
        "replica moe smoke processes 2 transport tls tokens_equal_in_process true rejoin_s {:.1}",
        rejoin.as_secs_f64()
    );
    for c in &mut procs.0 {
        assert!(c.try_wait().unwrap().is_none(), "a daemon exited");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
