//! The Phase 6 path on real binaries and loopback TLS (ADR 0091, step 6): the pieces step 5
//! built, run together once on one machine. CPU only, tiny fixtures.
//!
//! - `tetraeder serve --cluster` as a process, a head that leads no replica (it tokenizes and
//!   checks grammars from the vocabulary; two `tetraeder node` daemons run the replicas), the
//!   gateway answering chat requests;
//! - the replicas derived from a cluster plan (`--cluster-plan`), one replica across two nodes,
//!   run through the daemons: its greedy answer equals the one-node run;
//! - the real scheduler's prefix cache served over the `Prefix` purpose and installed on the
//!   other replica, priced by the fetch rule: the fetched prefix decodes like a recomputed one;
//! - a node told to drain: the replica across it is drained and started again on the other
//!   node, and no request fails.
//!
//! Hardware label: cpu (loopback).

mod common;

use common::two_stage_plan;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tetraeder_control::cluster_replan::{ClusterControl, Derive, ReplanWatcher};
use tetraeder_control::head::{
    ClusterEngine, ClusterHead, FetchPolicy, KvInfo, ReplicaTarget, StartSpec, specs_from_plan,
};
use tetraeder_control::kv::{PrefixKey, fetch_beats_recompute};
use tetraeder_control::monitor::Monitor;
use tetraeder_control::proto::NodeRequest;
use tetraeder_control::replica::{FetchFrom, RemoteEngine};
use tetraeder_control::{ClusterConfig, Config, PlanSource, Proposal, Replanner, SystemClock};
use tetraeder_control::{Trigger, client};
use tetraeder_costmodel::cluster::hop_time;
use tetraeder_gateway::engine::{Engine, GenEvent, GenerateRequest, SamplingParams};
use tetraeder_model::{BackendChoice, Gguf, LoadOptions};
use tetraeder_plan::Plan;
use tetraeder_profiler::HardwareGraph;
use tetraeder_scheduler::{Scheduler, SchedulerConfig, load_moe};
use tetraeder_transport::{Identity, NodeId, SocketBuffers, TlsTransport, Transport};
use tokio_util::sync::CancellationToken;

/// A model is named by the 64 hex digits of its hash; any value names the copy of the fixture.
const SHA: &str = "5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed";
const DOWN_MS: u64 = 4000;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

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

/// A cluster of loopback daemons: certificates, `cluster.toml` (the first node is the head), a
/// models folder with the fixture as `<SHA>.gguf`.
struct Cluster {
    dir: PathBuf,
    names: Vec<&'static str>,
    ports: Vec<u16>,
    cfg_path: PathBuf,
    procs: Procs,
    model: PathBuf,
}

impl Cluster {
    fn new(tag: &str, names: &[&'static str], fixture_name: &str) -> Self {
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("models")).unwrap();
        let model = dir.join("models").join(format!("{SHA}.gguf"));
        std::fs::copy(fixture(fixture_name), &model).unwrap();
        for n in names {
            let o = tetraeder()
                .args(["cluster", "cert", "--node", n, "--out"])
                .arg(dir.join(n))
                .output()
                .unwrap();
            assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        }
        let ports: Vec<u16> = names.iter().map(|_| free_port()).collect();
        let mut toml = format!(
            "cluster = \"it\"\nhead = \"{}\"\nnode_down_after_ms = {DOWN_MS}\n",
            names[0]
        );
        for (n, p) in names.iter().zip(&ports) {
            toml += &format!(
                "[[nodes]]\nid = \"{n}\"\naddr = \"127.0.0.1:{p}\"\nanchor = \"{n}/cert.pem\"\n"
            );
        }
        let cfg_path = dir.join("cluster.toml");
        std::fs::write(&cfg_path, toml).unwrap();
        Self {
            dir,
            names: names.to_vec(),
            ports,
            cfg_path,
            procs: Procs(Vec::new()),
            model,
        }
    }

    fn index(&self, n: &str) -> usize {
        self.names.iter().position(|x| *x == n).unwrap()
    }

    /// A daemon for node `n` whose replica engines get `replica_args`.
    fn spawn_daemon(&mut self, n: &str, replica_args: &[&str], worker_args: &[&str]) {
        let i = self.index(n);
        let c = tetraeder()
            .arg("node")
            .arg("--cluster")
            .arg(&self.cfg_path)
            .arg("--key")
            .arg(self.dir.join(n).join("key.pem"))
            .arg("--cert")
            .arg(self.dir.join(n).join("cert.pem"))
            .arg("--models-dir")
            .arg(self.dir.join("models"))
            .arg("--state-dir")
            .arg(self.dir.join(format!("state-{n}")))
            .args(replica_args.iter().map(|a| format!("--replica-arg={a}")))
            .args(worker_args.iter().map(|a| format!("--worker-arg={a}")))
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        wait_listening(self.ports[i], &format!("node {n}"));
        self.procs.0.push(c);
    }

    fn cfg(&self) -> ClusterConfig {
        ClusterConfig::load(&self.cfg_path).unwrap()
    }

    /// The head's transport: node `names[0]`'s key and certificate, dial only.
    fn head_transport(&self) -> Arc<dyn Transport> {
        let cfg = self.cfg();
        let h = self.names[0];
        let ident = Identity::from_pem(
            &std::fs::read(self.dir.join(h).join("cert.pem")).unwrap(),
            &std::fs::read(self.dir.join(h).join("key.pem")).unwrap(),
        )
        .unwrap();
        Arc::new(
            TlsTransport::new(
                &ident,
                &cfg.anchors().unwrap(),
                &cfg.ids(),
                None,
                SocketBuffers::Auto,
            )
            .unwrap(),
        )
    }
}

/// One HTTP request, `Connection: close`: the status and the body.
fn http(port: u16, path: &str, body: Option<&str>) -> (u16, String) {
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(120))).unwrap();
    let req = match body {
        Some(b) => format!(
            "POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}",
            b.len()
        ),
        None => format!("GET {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n"),
    };
    s.write_all(req.as_bytes()).unwrap();
    let mut out = Vec::new();
    let _ = s.read_to_end(&mut out);
    let text = String::from_utf8_lossy(&out).into_owned();
    let status = text
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let body = text
        .split_once("\r\n\r\n")
        .map_or("", |(_, b)| b)
        .to_owned();
    (status, body)
}

/// The text of a chat completion (greedy), or the HTTP error.
fn chat(port: u16, user: &str, extra: &str) -> Result<String, (u16, String)> {
    let body = format!(
        "{{\"messages\":[{{\"role\":\"user\",\"content\":{}}}],\"temperature\":0,\"max_tokens\":12{extra}}}",
        serde_json::Value::String(user.to_owned())
    );
    let (status, text) = http(port, "/v1/chat/completions", Some(&body));
    if status != 200 {
        return Err((status, text));
    }
    let v: serde_json::Value = serde_json::from_str(&text).unwrap_or_else(|e| {
        panic!("not JSON ({e}): {text}");
    });
    Ok(v["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default()
        .to_owned())
}

fn serve_args<'a>(c: &'a Cluster, port: u16, model: &'a Path) -> Command {
    let mut cmd = tetraeder();
    cmd.arg("serve")
        .arg("-m")
        .arg(model)
        .arg("--cluster")
        .arg(&c.cfg_path)
        .arg("--key")
        .arg(c.dir.join(c.names[0]).join("key.pem"))
        .arg("--cert")
        .arg(c.dir.join(c.names[0]).join("cert.pem"))
        .args(["--host", "127.0.0.1", "--port"])
        .arg(port.to_string())
        .arg("--accept-any-model")
        .arg("--chat-template-file")
        .arg(fixture("chatml.jinja"))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit());
    cmd
}

fn wait_healthy(port: u16, head: &mut Child) {
    wait_for("the head answers /health", Duration::from_secs(240), || {
        assert!(head.try_wait().unwrap().is_none(), "the head exited");
        TcpStream::connect(("127.0.0.1", port)).is_ok() && http(port, "/health", None).0 == 200
    });
}

const MOE_ARGS: [&str; 10] = [
    "--engine",
    "moe",
    "--ngl",
    "0",
    "--threads",
    "2",
    "--ctx",
    "512",
    "--parallel",
    "2",
];

/// hardware: cpu. The head as a process, leading no replica: it answers from the vocabulary of
/// `-m`, two daemons run the replicas. Chat requests succeed, in parallel; a grammar goes to the
/// replicas and constrains their output; a grammar that does not parse is refused by the head.
#[test]
fn cpu_serve_cluster_head_without_a_replica_routes_chat_to_two_replicas() {
    let mut c = Cluster::new("it-head", &["a", "b", "c"], "tiny-qwen3moe-f16.gguf");
    for n in ["a", "b", "c"] {
        c.spawn_daemon(n, &MOE_ARGS, &[]);
    }
    std::fs::write(
        c.dir.join("replicas.toml"),
        format!(
            "model_sha256 = \"{SHA}\"\n[[replica]]\nleader = \"b\"\ncap = 2\n[[replica]]\nleader = \"c\"\ncap = 2\n"
        ),
    )
    .unwrap();
    let port = free_port();
    let mut cmd = serve_args(&c, port, &c.model.clone());
    cmd.arg("--replicas").arg(c.dir.join("replicas.toml"));
    let head = cmd.spawn().unwrap();
    let mut guard = Procs(vec![head]);
    wait_healthy(port, &mut guard.0[0]);

    let want = chat(port, "Hello there", "").unwrap();
    assert!(!want.is_empty(), "the model said something");
    let workers: Vec<_> = (0..8)
        .map(|i| {
            std::thread::spawn(move || {
                (0..3)
                    .map(|_| chat(port, "Hello there", "").map_err(|(s, b)| format!("{s} {b}")))
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(|r| (i, r))
                    .collect::<Vec<_>>()
            })
        })
        .collect();
    for w in workers {
        for (i, r) in w.join().unwrap() {
            // Greedy: whichever replica answers, the same text.
            assert_eq!(r.unwrap(), want, "client {i}");
        }
    }

    // The grammar is compiled where it runs (the replica); the output obeys it.
    let g = chat(port, "Say anything", ",\"grammar\":\"root ::= [ab]+\"").unwrap();
    assert!(
        !g.is_empty() && g.chars().all(|ch| ch == 'a' || ch == 'b'),
        "{g:?}"
    );
    // A grammar that does not parse is refused by the head, before any replica sees it.
    let bad = chat(port, "Say anything", ",\"grammar\":\"root ::= ((\"").unwrap_err();
    assert_eq!(bad.0, 400, "{}", bad.1);
    println!(
        "cluster serve head without replica processes 4 replicas 2 transport tls chat_ok 25 grammar_ok true bad_grammar_status 400"
    );
    for p in &mut c.procs.0 {
        assert!(p.try_wait().unwrap().is_none(), "a daemon exited");
    }
}

fn ref_serve(model: &Path, port: u16) -> Child {
    tetraeder()
        .arg("serve")
        .arg("-m")
        .arg(model)
        .args(["--engine", "dense", "--ngl", "0", "--ctx", "512"])
        .args(["--parallel", "2", "--threads", "2"])
        .args(["--host", "127.0.0.1", "--port"])
        .arg(port.to_string())
        .arg("--accept-any-model")
        .arg("--chat-template-file")
        .arg(fixture("chatml.jinja"))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap()
}

const DENSE_ARGS: [&str; 10] = [
    "--engine",
    "dense",
    "--ngl",
    "0",
    "--threads",
    "2",
    "--ctx",
    "512",
    "--parallel",
    "2",
];

/// hardware: cpu. `serve --cluster-plan`: the replicas are derived from a two-stage plan of the
/// 4-layer dense fixture (nodes b and c, found by profiling the daemons), so one replica spans
/// two nodes and runs through the daemons: the leader's engine over a segment worker on each
/// node. Its greedy chat answers equal those of the same model served on one node.
#[test]
fn cpu_serve_cluster_plan_runs_a_pipeline_replica_across_two_nodes_like_one_node() {
    let mut c = Cluster::new("it-plan", &["a", "b", "c"], "tiny-qwen3-l4-f16.gguf");
    for n in ["a", "b", "c"] {
        c.spawn_daemon(n, &DENSE_ARGS, &DENSE_ARGS);
    }
    let plan = two_stage_plan("tiny", 4, 2, "b/cpu0", "c/cpu0", None);
    std::fs::write(c.dir.join("plan.toml"), plan.to_toml().unwrap()).unwrap();
    let port = free_port();
    let mut cmd = serve_args(&c, port, &c.model.clone());
    cmd.arg("--cluster-plan").arg(c.dir.join("plan.toml"));
    let mut guard = Procs(vec![cmd.spawn().unwrap()]);
    wait_healthy(port, &mut guard.0[0]);

    let ref_port = free_port();
    guard.0.push(ref_serve(&c.model, ref_port));
    wait_healthy(ref_port, &mut guard.0[1]);

    let mut checked = 0;
    for prompt in [
        "Hello there",
        "What is two plus two?",
        "Tell me about tetrahedra",
    ] {
        let got = chat(port, prompt, ",\"max_tokens\":16").unwrap();
        let want = chat(ref_port, prompt, ",\"max_tokens\":16").unwrap();
        assert!(!want.is_empty());
        assert_eq!(got, want, "prompt {prompt:?}");
        checked += 1;
    }
    println!(
        "cluster plan pipeline replica dense4 segments 2 processes 3 transport tls greedy_equal_one_node true prompts {checked}"
    );
    for p in &mut c.procs.0 {
        assert!(p.try_wait().unwrap().is_none(), "a daemon exited");
    }
}

/// hardware: cpu. A plan whose replica the head's own node leads (rehearsal 2: it used to run the
/// whole model in the head's process and leave node b idle). With exactly two nodes the head
/// serves a two-segment plan through its own daemon: the greedy answers equal one node's, and
/// both daemons run a worker (a: the replica engine and its segment, b: its segment).
#[test]
fn cpu_a_plan_led_by_the_head_computes_on_both_nodes_like_one_node() {
    let mut c = Cluster::new("it-headled", &["a", "b"], "tiny-qwen3-l4-f16.gguf");
    for n in ["a", "b"] {
        c.spawn_daemon(n, &DENSE_ARGS, &DENSE_ARGS);
    }
    let plan = two_stage_plan("tiny", 4, 2, "a/cpu0", "b/cpu0", None);
    std::fs::write(c.dir.join("plan.toml"), plan.to_toml().unwrap()).unwrap();
    let port = free_port();
    let mut cmd = serve_args(&c, port, &c.model.clone());
    cmd.arg("--cluster-plan").arg(c.dir.join("plan.toml"));
    let mut guard = Procs(vec![cmd.spawn().unwrap()]);
    wait_healthy(port, &mut guard.0[0]);
    let ref_port = free_port();
    guard.0.push(ref_serve(&c.model, ref_port));
    wait_healthy(ref_port, &mut guard.0[1]);

    for prompt in ["Hello there", "What is two plus two?"] {
        let got = chat(port, prompt, ",\"max_tokens\":16").unwrap();
        let want = chat(ref_port, prompt, ",\"max_tokens\":16").unwrap();
        assert!(!want.is_empty());
        assert_eq!(got, want, "prompt {prompt:?}");
    }
    // Both nodes compute: each daemon runs the workers of its segment (node a also the engine).
    let (cfg, t) = (c.cfg(), c.head_transport());
    let workers = |n: &str| -> u32 {
        let (r, _) = client::request(
            t.as_ref(),
            &cfg.peer(&NodeId::new(n)).unwrap(),
            &NodeRequest::Heartbeat { n: 0 },
            &[],
        )
        .unwrap();
        match r {
            tetraeder_control::proto::NodeReply::Alive { workers, .. } => workers,
            other => panic!("{other:?}"),
        }
    };
    let (wa, wb) = (workers("a"), workers("b"));
    println!("head-led plan dense4 segments 2 nodes 2 workers_a {wa} workers_b {wb}");
    assert!(
        wa >= 2,
        "node a runs the replica engine and its segment: {wa}"
    );
    assert!(wb >= 1, "node b runs its segment: {wb}");
    for p in &mut c.procs.0 {
        assert!(p.try_wait().unwrap().is_none(), "a daemon exited");
    }
}

/// The model of a split fixture: the 4-layer dense model cut in three shards, installed on every
/// node as the folder `models/<id>/` (manifest and shards), as `tetraeder cluster model-id
/// --install` does. Returns the first shard's path.
fn install_split_model(c: &Cluster) -> PathBuf {
    use tetraeder_model::gguf_write::write_split;
    let src = Gguf::open(&c.model).unwrap();
    let names: Vec<String> = src.tensors.iter().map(|t| t.name.clone()).collect();
    let n = names.len();
    let groups = vec![
        names[n / 3..2 * n / 3].to_vec(),
        names[2 * n / 3..].to_vec(),
    ];
    let parts = c.dir.join("parts");
    std::fs::create_dir_all(&parts).unwrap();
    let shards = write_split(&src, &parts.join("tiny"), &groups).unwrap();
    assert_eq!(shards.len(), 3);
    let id = tetraeder_model::model_id::identify(&shards[0]).unwrap();
    let models = c.dir.join("models");
    std::fs::remove_file(&c.model).unwrap();
    tetraeder_model::model_id::install(&id, &models).unwrap()
}

/// hardware: cpu. A split GGUF (three shards, as the 235B ships) is one model on every node: it is
/// kept as `models/<id>/`, the head checks both nodes hold it whole, the daemons start the
/// workers from its first shard, and the greedy answers equal those of the unsplit file on one
/// node. A node that lacks a shard fails the head's start with its name.
#[test]
fn cpu_a_split_model_runs_through_the_daemons_like_the_whole_file() {
    let mut c = Cluster::new("it-shards", &["a", "b"], "tiny-qwen3-l4-f16.gguf");
    let whole = c.dir.join("whole.gguf");
    std::fs::copy(&c.model, &whole).unwrap();
    let first = install_split_model(&c);
    for n in ["a", "b"] {
        c.spawn_daemon(n, &DENSE_ARGS, &DENSE_ARGS);
    }
    let plan = two_stage_plan("tiny", 4, 2, "a/cpu0", "b/cpu0", None);
    std::fs::write(c.dir.join("plan.toml"), plan.to_toml().unwrap()).unwrap();

    // Node b loses the middle shard: the head refuses to start and names the node.
    let id = tetraeder_model::model_id::id_of_path(&first).unwrap();
    let shard2 = c
        .dir
        .join("models")
        .join(&id)
        .join(tetraeder_model::model_id::shard_file_name(&id, 2, 3));
    let kept = c.dir.join("shard2.keep");
    std::fs::rename(&shard2, &kept).unwrap();
    let mut cmd = serve_args(&c, free_port(), &first);
    cmd.arg("--cluster-plan").arg(c.dir.join("plan.toml"));
    cmd.stderr(Stdio::piped());
    let out = cmd.output().unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "the head started without a shard");
    assert!(err.contains("shard 2 of 3"), "{err}");
    std::fs::rename(&kept, &shard2).unwrap();

    let port = free_port();
    let mut cmd = serve_args(&c, port, &first);
    cmd.arg("--cluster-plan").arg(c.dir.join("plan.toml"));
    let mut guard = Procs(vec![cmd.spawn().unwrap()]);
    wait_healthy(port, &mut guard.0[0]);
    let ref_port = free_port();
    guard.0.push(ref_serve(&whole, ref_port));
    wait_healthy(ref_port, &mut guard.0[1]);
    for prompt in ["Hello there", "What is two plus two?"] {
        let got = chat(port, prompt, ",\"max_tokens\":16").unwrap();
        let want = chat(ref_port, prompt, ",\"max_tokens\":16").unwrap();
        assert!(!want.is_empty());
        assert_eq!(got, want, "prompt {prompt:?}");
    }
    println!("split model dense4 shards 3 nodes 2 greedy_equal_whole_file true");
    for p in &mut c.procs.0 {
        assert!(p.try_wait().unwrap().is_none(), "a daemon exited");
    }
}

/// The shape of the rehearsal's head: one 4096-token pool of four sequences.
const BIG_ARGS: [&str; 10] = [
    "--engine",
    "moe",
    "--ngl",
    "0",
    "--threads",
    "2",
    "--ctx",
    "4096",
    "--parallel",
    "4",
];

/// A split plan of the MoE fixture over nodes b and c (ubatch 512, 4 micro-batches, a 4096-row
/// pool of four sequences) behind a real head, and one node serving the same model: the
/// rehearsal's shape. Returns the cluster, the processes, and the two HTTP ports.
fn cold_cluster(tag: &str) -> (Cluster, Procs, u16, u16) {
    let mut c = Cluster::new(tag, &["a", "b", "c"], "tiny-qwen3moe-f16.gguf");
    for n in ["a", "b", "c"] {
        c.spawn_daemon(n, &BIG_ARGS, &BIG_ARGS);
    }
    let mut plan = two_stage_plan("tiny", 2, 1, "b/cpu0", "c/cpu0", None);
    plan.knobs.prefill_micro_batches = 4;
    plan.knobs.ubatch = Some(512);
    std::fs::write(c.dir.join("plan.toml"), plan.to_toml().unwrap()).unwrap();
    let port = free_port();
    let mut cmd = serve_args(&c, port, &c.model.clone());
    cmd.arg("--cluster-plan").arg(c.dir.join("plan.toml"));
    let mut guard = Procs(vec![cmd.spawn().unwrap()]);
    wait_healthy(port, &mut guard.0[0]);
    let ref_port = free_port();
    guard.0.push(
        tetraeder()
            .arg("serve")
            .arg("-m")
            .arg(&c.model)
            .args(["--engine", "moe", "--ngl", "0", "--ctx", "4096"])
            .args(["--parallel", "4", "--threads", "2"])
            .args(["--host", "127.0.0.1", "--port"])
            .arg(ref_port.to_string())
            .arg("--accept-any-model")
            .arg("--chat-template-file")
            .arg(fixture("chatml.jinja"))
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap(),
    );
    wait_healthy(ref_port, &mut guard.0[1]);
    (c, guard, port, ref_port)
}

/// hardware: cpu. Two different cold prompts, each longer than the plan's step (ubatch 512 times
/// 4 micro-batches), one after the other through the real head and a replica across two nodes:
/// both answer with tokens, and the same ones as one node (rehearsal 2: the second cold request
/// after a head start came back as an empty 200).
#[test]
fn cpu_two_cold_long_prompts_in_a_row_through_a_split_plan_both_answer_like_one_node() {
    let (mut c, _guard, port, ref_port) = cold_cluster("it-cold2");
    for tag in ["alpha", "bravo", "charlie"] {
        let prompt = format!("{tag} the quick brown fox jumps over the lazy dog. ").repeat(40);
        let got = chat(port, &prompt, ",\"max_tokens\":8").unwrap();
        let want = chat(ref_port, &prompt, ",\"max_tokens\":8").unwrap();
        assert!(!want.is_empty());
        assert_eq!(got, want, "cold prompt {tag}");
    }
    // The rehearsal's shape: streamed, ignore_eos, a fresh prompt each time.
    for tag in ["delta", "echo", "foxtrot"] {
        let prompt = format!("{tag} the quick brown fox jumps over the lazy dog. ").repeat(40);
        let body = format!(
            "{{\"messages\":[{{\"role\":\"user\",\"content\":{}}}],\"temperature\":0,\"max_tokens\":12,\"stream\":true,\"ignore_eos\":true}}",
            serde_json::Value::String(prompt)
        );
        let (status, text) = http(port, "/v1/chat/completions", Some(&body));
        assert_eq!(status, 200, "{text}");
        let chunks = text.matches("\"content\"").count();
        assert!(
            chunks >= 6,
            "streamed prompt {tag}: {chunks} chunks in {text}"
        );
    }
    for p in &mut c.procs.0 {
        assert!(p.try_wait().unwrap().is_none(), "a daemon exited");
    }
}

/// hardware: cpu. The rehearsal's eviction: each prompt (about 2100 tokens) fits the 4096-row
/// pool, two do not, so every cold prompt after the first must evict the idle sequence of the one
/// before it (a snapshot of its KV through the segment workers) and still answer, with the
/// tokens one node gives, streamed with ignore_eos as the round script asks.
#[test]
fn cpu_cold_prompts_that_do_not_fit_together_evict_the_idle_one_and_answer_like_one_node() {
    let (mut c, _guard, port, ref_port) = cold_cluster("it-evict");
    let long =
        |tag: &str| format!("{tag} the quick brown fox jumps over the lazy dog. ").repeat(45);
    for tag in ["alpha", "bravo", "charlie"] {
        let got = chat(port, &long(tag), ",\"max_tokens\":8").unwrap();
        let want = chat(ref_port, &long(tag), ",\"max_tokens\":8").unwrap();
        assert!(!want.is_empty());
        assert_eq!(got, want, "cold prompt {tag}");
    }
    for tag in ["delta", "echo"] {
        let body = format!(
            "{{\"messages\":[{{\"role\":\"user\",\"content\":{}}}],\"temperature\":0,\"max_tokens\":12,\"stream\":true,\"ignore_eos\":true}}",
            serde_json::Value::String(long(tag))
        );
        let (status, text) = http(port, "/v1/chat/completions", Some(&body));
        assert_eq!(status, 200, "{text}");
        let chunks = text.matches("\"content\"").count();
        assert!(
            chunks >= 6,
            "streamed prompt {tag}: {chunks} chunks in {text}"
        );
        assert!(!text.contains("\"error\""), "{text}");
    }
    for p in &mut c.procs.0 {
        assert!(p.try_wait().unwrap().is_none(), "a daemon exited");
    }
}

fn greedy_request(prompt: &[u32], n: u32) -> GenerateRequest {
    GenerateRequest {
        model: "tiny".into(),
        prompt: prompt.to_vec(),
        params: SamplingParams {
            max_tokens: n,
            temperature: Some(0.0),
            ..SamplingParams::default()
        },
        constraint: None,
        cancel: CancellationToken::new(),
    }
}

/// The tokens of one greedy request on `e`, the cached prompt tokens of the stats, or the error.
fn greedy(e: &dyn Engine, prompt: &[u32], n: u32) -> Result<(Vec<u32>, u32), String> {
    let rx = e
        .generate(greedy_request(prompt, n))
        .map_err(|e| e.to_string())?;
    collect(rx)
}

fn collect(mut rx: tokio::sync::mpsc::Receiver<GenEvent>) -> Result<(Vec<u32>, u32), String> {
    let (mut out, mut cached, mut err) = (Vec::new(), 0, None);
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        match rx.try_recv() {
            Ok(GenEvent::Token(t)) => out.push(t),
            Ok(GenEvent::Finished { stats, .. }) => {
                cached = stats.cached_tokens;
                break;
            }
            Ok(GenEvent::Error(m) | GenEvent::ContextExceeded(m)) => {
                err = Some(m);
                break;
            }
            Ok(GenEvent::Probs(_)) => {}
            Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                assert!(Instant::now() < deadline, "a request never ended");
                std::thread::sleep(Duration::from_millis(2));
            }
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                err = Some("the stream ended without a final event".into());
                break;
            }
        }
    }
    err.map_or(Ok((out, cached)), Err)
}

/// Fetches while the sum of its two rates says so: the real fetch rule on the probed graph, with
/// the prompt speed of the replicas switched by the test.
struct Priced {
    hw: HardwareGraph,
    nodes: [&'static str; 2],
    slow_prefill: AtomicBool,
    asked: Mutex<Vec<bool>>,
}

impl FetchPolicy for Priced {
    fn plan(&self, holder: u32, target: u32, tokens: &[u32], kv: &KvInfo) -> Option<FetchFrom> {
        assert!(kv.row_bytes > 0, "the replica reported its snapshot size");
        let n = tokens.len() as u32;
        let bytes = u64::from(n) * kv.row_bytes;
        // The prompt speed is set from the probed hop time of this very fetch, so the rule's
        // answer does not depend on how fast the machine's loopback is: a recompute ten times
        // slower than the hop, or ten times faster.
        let hop = hop_time(
            &self.hw,
            self.nodes[holder as usize],
            self.nodes[target as usize],
            bytes as f64,
        )
        .expect("the probe measured the link");
        assert!(hop > 0.0, "the probed hop of {bytes} B is {hop} s");
        let pp = if self.slow_prefill.load(Ordering::SeqCst) {
            f64::from(n) / (10.0 * hop)
        } else {
            f64::from(n) * 10.0 / hop
        };
        let fetch = fetch_beats_recompute(
            &self.hw,
            self.nodes[holder as usize],
            self.nodes[target as usize],
            bytes,
            n,
            pp,
            0.0,
            tetraeder_scheduler::RESTORE_MIN_GAIN as u32,
        );
        self.asked.lock().unwrap().push(fetch);
        fetch.then(|| FetchFrom {
            node: NodeId::new(self.nodes[holder as usize]),
            key: PrefixKey {
                format: kv.format.clone(),
                tokens: tokens.to_vec(),
            },
        })
    }
}

fn long_prompt(front: &Scheduler, seed: &str) -> Vec<u32> {
    let mut text = String::new();
    let mut toks = Vec::new();
    while toks.len() < 140 {
        text += &format!("{seed} the quick brown fox jumps over the lazy dog. ");
        toks = front.tokenize("tiny", &text).unwrap();
    }
    toks.truncate(140);
    toks
}

/// hardware: cpu. The prefix cache of a real scheduler served and installed over the `Prefix`
/// purpose. Two MoE replicas (nodes b and c) behind daemons; the probed links of the real
/// profile price the fetch. A follow-up of a conversation lands on the other replica because the
/// holder is busy: with a slow prefill the rule fetches the prefix (the replica restores it:
/// cached prompt tokens), with a fast one it recomputes; the tokens are the same either way and
/// equal the in-process engine's.
#[test]
fn cpu_a_fetched_prefix_decodes_like_a_recomputed_one() {
    let mut c = Cluster::new("it-kv", &["a", "b", "c"], "tiny-qwen3moe-f16.gguf");
    for n in ["a", "b", "c"] {
        c.spawn_daemon(n, &MOE_ARGS, &[]);
    }
    let cfg = c.cfg();
    let t = c.head_transport();
    // Real discovery and probe: the hardware graph with the links between the nodes.
    let probe = client::probe_cluster(t.as_ref(), &cfg, &[4096, 65536, 1 << 20], 5, None).unwrap();
    let hw = client::profile_cluster(t.as_ref(), &cfg, Some(&probe)).unwrap();

    let model = fixture("tiny-qwen3moe-f16.gguf");
    let front: Arc<Scheduler> = {
        let path = model.clone();
        let mut sc = SchedulerConfig::new(&model);
        sc.alias = Some("tiny".into());
        sc.parallel = 2;
        Arc::new(
            Scheduler::start(sc, move || {
                let mut o = LoadOptions::new(BackendChoice::Cpu { threads: Some(2) });
                o.n_ctx = 512;
                o.n_seq_max = 2;
                load_moe(&path, &o)
            })
            .unwrap(),
        )
    };
    let monitor = Arc::new(Monitor::start(Arc::clone(&t), cfg.clone()));
    let front_engine: Arc<dyn Engine> = Arc::clone(&front) as _;
    let remotes: Vec<Arc<RemoteEngine>> = ["b", "c"]
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
    let policy = Arc::new(Priced {
        hw,
        nodes: ["b", "c"],
        slow_prefill: AtomicBool::new(true),
        asked: Mutex::new(Vec::new()),
    });
    let engine = Arc::new(ClusterEngine::new(
        Arc::clone(&front_engine),
        remotes
            .iter()
            .map(|r| ReplicaTarget {
                engine: Arc::clone(r) as Arc<dyn Engine>,
                remote: Some(Arc::clone(r)),
                // One at a time: a busy replica sends the follow-up to the other one.
                cap: 1,
            })
            .collect(),
        tetraeder_scheduler::RESTORE_MIN_GAIN as u32,
        Some(Arc::clone(&policy) as Arc<dyn FetchPolicy>),
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
        vec![(0, NodeId::new("b"), spec()), (1, NodeId::new("c"), spec())],
    );
    wait_for("both replicas join", Duration::from_secs(240), || {
        engine.up_count() == 2
    });
    assert!(
        engine.kv_format(0).len() > 1,
        "the replica reports its snapshot format"
    );

    // Round one fetches, round two recomputes. The follow-up of a conversation that replica 0
    // holds is sent to replica 1 with the hint the rule gives (the router does the same when
    // replica 0 is full; the control crate's simulation covers that decision), so no timing
    // decides what is tested here.
    let mut cached_seen = Vec::new();
    for (round, slow) in [(0, true), (1, false)] {
        policy.slow_prefill.store(slow, Ordering::SeqCst);
        let base = long_prompt(&front, &format!("round {round}"));
        let (first, _) = greedy(remotes[0].as_ref(), &base, 6).unwrap();
        assert_eq!(first.len(), 6);
        let mut follow = base.clone();
        follow.extend(first.iter().copied());
        follow.extend(front.tokenize("tiny", " and then?").unwrap());
        let kv = KvInfo {
            format: engine.kv_format(0),
            row_bytes: engine.kv_row_bytes(0),
        };
        let hint = policy.plan(0, 1, &base, &kv);
        assert_eq!(
            hint.is_some(),
            slow,
            "the rule decides: fetch only when prefill is slow"
        );
        // The head announces the fetch to the holder first (the holder serves only that).
        if let Some(h) = &hint {
            assert!(remotes[0].offer(remotes[1].peer().id.clone(), h.key.clone()));
        }
        let rx = remotes[1]
            .generate_with(greedy_request(&follow, 12), hint)
            .unwrap();
        let (got, cached) = collect(rx).unwrap();
        let (want, _) = greedy(front.as_ref(), &follow, 12).unwrap();
        assert_eq!(
            got, want,
            "round {round}: same tokens as the in-process engine"
        );
        cached_seen.push(cached);
    }
    let asked = policy.asked.lock().unwrap().clone();
    assert!(
        asked.contains(&true) && asked.contains(&false),
        "the rule decided both ways: {asked:?}"
    );
    assert!(
        cached_seen[0] >= tetraeder_scheduler::RESTORE_MIN_GAIN as u32,
        "the fetched prefix was restored: {cached_seen:?}"
    );
    assert!(
        cached_seen[1] < tetraeder_scheduler::RESTORE_MIN_GAIN as u32,
        "recomputed: no prefix restored on the replica: {cached_seen:?}"
    );
    println!(
        "prefix fetch real scheduler processes 3 transport tls fetched_cached_tokens {} recomputed_cached_tokens {} tokens_equal true",
        cached_seen[0], cached_seen[1]
    );
    for p in &mut c.procs.0 {
        assert!(p.try_wait().unwrap().is_none(), "a daemon exited");
    }
}

/// Proposes the plan on node b alone when node c drains.
struct LeaveC;

impl PlanSource for LeaveC {
    fn propose(&mut self, _current: &Plan, t: &Trigger) -> Option<Proposal> {
        match t {
            Trigger::NodeChange { node, state } if node == "c" && state == "draining" => {
                Some(Proposal {
                    plan: Plan {
                        model: "on-b".into(),
                        knobs: Default::default(),
                        replicas: vec![],
                    },
                    predicted_gain_s_per_s: 1.0,
                    switch_cost_s: 0.0,
                    knobs_only: false,
                })
            }
            _ => None,
        }
    }
}

/// hardware: cpu. A replica across nodes b and c (the dense fixture cut in two), clients
/// streaming through the router; node c is told to drain. The head replans: replica 0 is drained
/// (its requests finish), its segment on c is released and it starts again on b alone. No
/// request fails, and its greedy tokens before and after are the same.
#[test]
fn cpu_draining_a_node_replans_a_pipeline_replica_without_failing_requests() {
    let mut c = Cluster::new("it-drain", &["a", "b", "c"], "tiny-qwen3-l4-f16.gguf");
    for n in ["a", "b", "c"] {
        c.spawn_daemon(n, &DENSE_ARGS, &DENSE_ARGS);
    }
    let cfg = c.cfg();
    let t = c.head_transport();
    let hw = client::profile_cluster(t.as_ref(), &cfg, None).unwrap();
    let across = two_stage_plan("tiny", 4, 2, "b/cpu0", "c/cpu0", None);
    let mut specs = specs_from_plan(&across, &hw, &cfg, SHA, &[]).unwrap();
    let (_, leader, spec) = specs.remove(0);
    assert_eq!(leader, NodeId::new("b"));
    assert_eq!(spec.segments.len(), 2, "a replica across two nodes");
    // The whole model on its leader.
    let alone = StartSpec {
        plan: String::new(),
        model_sha256: SHA.into(),
        segments: Vec::new(),
        predicted_s: Vec::new(),
    };

    let model = fixture("tiny-qwen3-l4-f16.gguf");
    let front: Arc<dyn Engine> = {
        let mut sc = SchedulerConfig::new(&model);
        sc.alias = Some("tiny".into());
        sc.parallel = 2;
        let path = model.clone();
        Arc::new(
            Scheduler::start(sc, move || {
                tetraeder_scheduler::load_dense(
                    &path,
                    &tetraeder_model::DenseOptions {
                        n_seq_max: 2,
                        n_ctx_pool: 512,
                        threads: Some(2),
                        ..tetraeder_model::DenseOptions::new(BackendChoice::Cpu {
                            threads: Some(2),
                        })
                    },
                )
            })
            .unwrap(),
        )
    };
    let monitor = Arc::new(Monitor::start(Arc::clone(&t), cfg.clone()));
    let remote = RemoteEngine::new(
        Arc::clone(&front),
        Arc::clone(&t),
        cfg.peer(&NodeId::new("b")).unwrap(),
        0,
    );
    let engine = Arc::new(ClusterEngine::new(
        Arc::clone(&front),
        vec![ReplicaTarget {
            engine: Arc::clone(&remote) as Arc<dyn Engine>,
            remote: Some(Arc::clone(&remote)),
            cap: 2,
        }],
        tetraeder_scheduler::RESTORE_MIN_GAIN as u32,
        None,
    ));
    let head = ClusterHead::start(&engine, &monitor, &t, vec![(0, leader.clone(), spec)]);
    let derive: Derive = {
        let alone = alone.clone();
        let leader = leader.clone();
        Arc::new(move |p: &Plan| {
            if p.model == "on-b" {
                Ok(vec![(0, leader.clone(), alone.clone())])
            } else {
                Err("unknown plan".into())
            }
        })
    };
    let control = ClusterControl::new(
        Arc::clone(&engine),
        head.specs(),
        derive,
        Duration::from_secs(60),
        Duration::from_secs(240),
    );
    let _watcher = ReplanWatcher::start(&monitor, move || {
        Replanner::new(
            Config {
                min_dwell_s: 0.0,
                ..Config::default()
            },
            across,
            LeaveC,
            control,
            SystemClock::new(),
        )
    });
    wait_for(
        "the replica across b and c joins",
        Duration::from_secs(240),
        || engine.up_count() == 1,
    );

    let prompt = front
        .tokenize("tiny", "Hello there, how are you today?")
        .unwrap();
    let (before, _) = greedy(engine.as_ref(), &prompt, 12).unwrap();
    let (reference, _) = greedy(front.as_ref(), &prompt, 12).unwrap();
    assert_eq!(before, reference, "across two nodes: the one-node tokens");

    // Four clients keep streaming greedy requests through the router while c drains.
    let stop = Arc::new(AtomicBool::new(false));
    let failed = Arc::new(Mutex::new(Vec::<String>::new()));
    let ok = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let clients: Vec<_> = (0..4)
        .map(|_| {
            let (e, p, stop, failed, ok) = (
                Arc::clone(&engine),
                prompt.clone(),
                Arc::clone(&stop),
                Arc::clone(&failed),
                Arc::clone(&ok),
            );
            std::thread::spawn(move || {
                while !stop.load(Ordering::SeqCst) {
                    match greedy(e.as_ref(), &p, 12) {
                        Ok(_) => {
                            ok.fetch_add(1, Ordering::SeqCst);
                        }
                        Err(m) => failed.lock().unwrap().push(m),
                    }
                }
            })
        })
        .collect();
    std::thread::sleep(Duration::from_secs(1));
    client::request(
        t.as_ref(),
        &cfg.peer(&NodeId::new("c")).unwrap(),
        &NodeRequest::Drain,
        &[],
    )
    .unwrap();
    let took = wait_for(
        "replica 0 runs on b alone",
        Duration::from_secs(300),
        || {
            head.specs().spec(0).is_some_and(|s| s.segments.is_empty())
                && engine.is_up(0)
                && !engine.is_draining(0)
        },
    );
    std::thread::sleep(Duration::from_secs(1));
    stop.store(true, Ordering::SeqCst);
    for h in clients {
        h.join().unwrap();
    }
    assert!(
        failed.lock().unwrap().is_empty(),
        "a planned drain fails nothing: {:?}",
        failed.lock().unwrap()
    );
    let (after, _) = greedy(engine.as_ref(), &prompt, 12).unwrap();
    assert_eq!(after, before, "the same greedy tokens on b alone");
    println!(
        "planned drain pipeline replica dense4 processes 3 transport tls replanned_s {:.1} failed_requests 0 ok_requests {} tokens_equal true",
        took.as_secs_f64(),
        ok.load(Ordering::SeqCst)
    );
    assert!(
        c.procs
            .0
            .iter_mut()
            .all(|p| p.try_wait().unwrap().is_none())
    );
}

// ---------------------------------------------------------------------------------------------
// The prefix fetch through the real head (rehearsal 2, `cluster-prefix-fetch-not-observed`).
// ---------------------------------------------------------------------------------------------

const PREFIX_ARGS: [&str; 10] = [
    "--engine",
    "moe",
    "--ngl",
    "0",
    "--threads",
    "2",
    "--ctx",
    "2048",
    "--parallel",
    "2",
];

/// The log without its colour codes.
fn plain(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(ch) = chars.next() {
        if ch == '\x1b' {
            for c in chars.by_ref() {
                if c == 'm' {
                    break;
                }
            }
        } else {
            out.push(ch);
        }
    }
    out
}

/// A `tetraeder serve --cluster` head whose stderr goes to a file.
struct LoggedHead {
    _procs: Procs,
    port: u16,
    log: PathBuf,
}

impl LoggedHead {
    fn log(&self) -> String {
        plain(&std::fs::read_to_string(&self.log).unwrap_or_default())
    }

    fn lines(&self, what: &str) -> Vec<String> {
        self.log()
            .lines()
            .filter(|l| l.contains(what))
            .map(str::to_owned)
            .collect()
    }

    /// The value of one sample of `/metrics`; 0 when it is not there.
    fn metric(&self, sample: &str) -> u64 {
        let (status, text) = http(self.port, "/metrics", None);
        assert_eq!(status, 200, "/metrics: {text}");
        text.lines()
            .find_map(|l| l.strip_prefix(sample)?.trim().parse().ok())
            .unwrap_or(0)
    }
}

/// The head for the two replicas on b and c, one request at a time each, both priced at
/// `prefill_tps` by the fetch rule on the graph `hw`. Waits until both joined.
fn prefix_head(c: &Cluster, tag: &str, prefill_tps: f64, hw: &Path) -> LoggedHead {
    let replicas = c.dir.join(format!("replicas-{tag}.toml"));
    std::fs::write(
        &replicas,
        format!(
            "model_sha256 = \"{SHA}\"\n[[replica]]\nleader = \"b\"\ncap = 1\nprefill_tps = {prefill_tps:e}\n[[replica]]\nleader = \"c\"\ncap = 1\nprefill_tps = {prefill_tps:e}\n"
        ),
    )
    .unwrap();
    let log = c.dir.join(format!("head-{tag}.log"));
    let port = free_port();
    let mut cmd = serve_args(c, port, &c.model);
    cmd.arg("--replicas")
        .arg(&replicas)
        .arg("--cluster-hw")
        .arg(hw)
        .arg("--metrics")
        .env("RUST_LOG", "info")
        .stderr(Stdio::from(std::fs::File::create(&log).unwrap()));
    let mut procs = Procs(vec![cmd.spawn().unwrap()]);
    wait_healthy(port, &mut procs.0[0]);
    let head = LoggedHead {
        _procs: procs,
        port,
        log,
    };
    wait_for("both replicas join", Duration::from_secs(240), || {
        head.lines("replica joined").len() >= 2
    });
    head
}

/// One greedy completion through the head: its text and the prompt tokens it restored.
fn complete(port: u16, prompt: &str, max_tokens: u32) -> (String, u64) {
    let body = serde_json::json!({
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": true,
    })
    .to_string();
    let (status, text) = http(port, "/v1/completions", Some(&body));
    assert_eq!(status, 200, "{text}");
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    (
        v["choices"][0]["text"]
            .as_str()
            .unwrap_or_default()
            .to_owned(),
        v["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap_or(0),
    )
}

/// A streamed completion that runs on: returns once its first token arrived, with the open
/// connection (the replica stays busy until it ends; read it to the end to let it finish).
fn busy(port: u16, prompt: &str, max_tokens: u32) -> TcpStream {
    let body = serde_json::json!({
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": true,
        "stream": true,
    })
    .to_string();
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(120))).unwrap();
    write!(
        s,
        "POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut seen = Vec::new();
    let mut buf = [0u8; 1024];
    while !String::from_utf8_lossy(&seen).contains("\"text\"") {
        let n = s.read(&mut buf).unwrap();
        assert!(
            n > 0,
            "the stream ended before its first token: {}",
            String::from_utf8_lossy(&seen)
        );
        seen.extend_from_slice(&buf[..n]);
    }
    s
}

/// A prompt of at least `n` tokens that starts with `tag`.
fn prompt_of(port: u16, tag: &str, n: usize) -> String {
    let mut text = format!("{tag}:");
    loop {
        text += " the quick brown fox jumps over the lazy dog.";
        let body = serde_json::json!({ "content": text }).to_string();
        let (status, t) = http(port, "/tokenize", Some(&body));
        assert_eq!(status, 200, "{t}");
        let v: serde_json::Value = serde_json::from_str(&t).unwrap();
        if v["tokens"].as_array().map_or(0, Vec::len) >= n {
            return text;
        }
    }
}

/// What one round saw: the head's lines for it, and the two answers.
struct Round {
    decision: String,
    done: Option<String>,
    first: String,
    again: (String, u64),
}

/// X1 (a prompt of about 120 tokens) runs on replica 0 (b) and finishes. Then replica 0 is made
/// busy: by an unrelated long request, or (`share`) by one that extends X1's prompt, as the
/// rehearsal's fillers did. X1 again then goes to replica 1 (c) with replica 0 as the holder,
/// and the head decides. `None`: the busy request ended before X1 again was routed (no decision
/// line), so the round is to be repeated.
fn prefix_round(head: &LoggedHead, tag: &str, share: bool) -> Option<Round> {
    let port = head.port;
    let p = prompt_of(port, tag, 120);
    let (first, _) = complete(port, &p, 8);
    let decisions_before = head.lines("prefix decision").len();
    let done_before = head.lines("prefix fetch done").len();
    let blocker = if share {
        format!("{p} And so the story goes on:")
    } else {
        format!("{tag} blocker: count from one to a thousand.")
    };
    let mut stream = busy(port, &blocker, 600);
    let again = complete(port, &p, 8);
    let mut rest = Vec::new();
    let _ = stream.read_to_end(&mut rest);
    // The outcome arrives before the request's first token, so it is logged by now; give the
    // log file a moment.
    std::thread::sleep(Duration::from_millis(200));
    let decision = head
        .lines("prefix decision")
        .get(decisions_before)
        .cloned()?;
    let done = head.lines("prefix fetch done").get(done_before).cloned();
    Some(Round {
        decision,
        done,
        first,
        again,
    })
}

/// [`prefix_round`] until a round saw a decision (at most four tries).
fn decided_round(head: &LoggedHead, tag: &str, share: bool) -> Round {
    for attempt in 0..4 {
        if let Some(r) = prefix_round(head, &format!("{tag}{attempt}"), share) {
            return r;
        }
        println!(
            "prefix round {tag}: try {attempt} saw no decision (the busy request ended first)"
        );
    }
    panic!(
        "no prefix decision in four rounds; head log:\n{}",
        head.log()
    );
}

/// A log field, quoted (`reason="gone"`) or not.
fn has_field(line: &str, key: &str, value: &str) -> bool {
    line.contains(&format!("{key}=\"{value}\"")) || line.contains(&format!("{key}={value} "))
}

/// hardware: cpu. The prefix fetch through the real head process (`serve --cluster --replicas
/// --cluster-hw --metrics`), two replica processes behind the daemons of b and c, one request at
/// a time each, the fetch rule priced on the probed graph with the declared prefill rate. A
/// conversation held by replica 0 that comes back while replica 0 is busy goes to replica 1:
/// - priced to fetch (prefill 0.001 tok/s): the head logs `decision=fetch`, the target reports
///   `fetched` with the snapshot's bytes, `/metrics` counts both, the prompt is restored (cached
///   tokens) and decodes like the first run;
/// - rehearsal 2's shape (the holder busy with a request that extends the same prompt): the fetch
///   is asked and the holder has nothing to serve, because its only copy of the prefix is the
///   sequence that request now extends in place (only idle sequences and cache entries are
///   served): `gone`, recomputed, the same tokens;
/// - priced to recompute (prefill 1e12 tok/s): `decision=recompute reason=priced_recompute`,
///   nothing fetched, the same tokens.
#[test]
fn cpu_the_head_fetches_a_prefix_as_priced_and_logs_and_counts_it() {
    let mut c = Cluster::new("it-prefix-head", &["a", "b", "c"], "tiny-qwen3moe-f16.gguf");
    for n in ["a", "b", "c"] {
        c.spawn_daemon(n, &PREFIX_ARGS, &[]);
    }
    // The graph a box gets from `tetraeder cluster profile`: discovery and the probed links.
    let hw_path = c.dir.join("hw.json");
    {
        let cfg = c.cfg();
        let t = c.head_transport();
        let probe =
            client::probe_cluster(t.as_ref(), &cfg, &[4096, 65536, 1 << 20], 5, None).unwrap();
        let hw = client::profile_cluster(t.as_ref(), &cfg, Some(&probe)).unwrap();
        std::fs::write(&hw_path, hw.to_json().unwrap()).unwrap();
    }
    let fetched = "tetraeder_kv_prefix_fetch_total{result=\"fetched\",reason=\"fetched\"}";
    let gone = "tetraeder_kv_prefix_fetch_total{result=\"recomputed\",reason=\"gone\"}";
    let priced =
        "tetraeder_kv_prefix_fetch_total{result=\"recomputed\",reason=\"priced_recompute\"}";
    let bytes = "tetraeder_kv_prefix_fetch_bytes_total";
    let gain = tetraeder_scheduler::RESTORE_MIN_GAIN as u64;

    let head = prefix_head(&c, "fetch", 0.001, &hw_path);
    let r = decided_round(&head, "fetch", false);
    let log = head.log();
    assert!(
        has_field(&r.decision, "holder", "0") && has_field(&r.decision, "to", "1"),
        "{}\n{log}",
        r.decision
    );
    assert!(
        has_field(&r.decision, "decision", "fetch"),
        "priced to fetch: {}\n{log}",
        r.decision
    );
    let done = r
        .done
        .clone()
        .unwrap_or_else(|| panic!("no fetch outcome:\n{log}"));
    assert!(has_field(&done, "reason", "fetched"), "{done}\n{log}");
    assert_eq!(
        r.again.0, r.first,
        "the fetched prefix decodes like the computed one"
    );
    assert!(
        r.again.1 >= gain,
        "the fetched prefix was restored: {} cached tokens\n{log}",
        r.again.1
    );
    assert_eq!(head.metric(fetched), 1, "{log}");
    let fetched_bytes = head.metric(bytes);
    assert!(fetched_bytes > 0, "{log}");
    println!(
        "prefix head fetch decision: {}\nprefix head fetch outcome: {done}\nprefix head fetch cached_tokens {} bytes {fetched_bytes} tokens_equal true",
        r.decision, r.again.1
    );

    // Rehearsal 2: the holder is busy with a request that extends the prefix.
    let r = decided_round(&head, "share", true);
    let log = head.log();
    assert!(
        has_field(&r.decision, "decision", "fetch"),
        "{}\n{log}",
        r.decision
    );
    let done = r
        .done
        .clone()
        .unwrap_or_else(|| panic!("no fetch outcome:\n{log}"));
    assert!(has_field(&done, "reason", "gone"), "{done}\n{log}");
    assert_eq!(r.again.0, r.first, "recomputed: the same tokens");
    assert!(r.again.1 < gain, "nothing restored: {} cached", r.again.1);
    assert_eq!(head.metric(gone), 1, "{log}");
    assert_eq!(head.metric(fetched), 1, "{log}");
    assert_eq!(head.metric(bytes), fetched_bytes, "{log}");
    println!(
        "prefix head shared holder decision: {}\nprefix head shared holder outcome: {done}",
        r.decision
    );
    drop(head);

    let head = prefix_head(&c, "recompute", 1e12, &hw_path);
    let r = decided_round(&head, "recompute", false);
    let log = head.log();
    assert!(
        has_field(&r.decision, "decision", "recompute")
            && has_field(&r.decision, "reason", "priced_recompute"),
        "{}\n{log}",
        r.decision
    );
    assert!(r.done.is_none(), "nothing was asked: {:?}", r.done);
    assert_eq!(r.again.0, r.first, "recomputed: the same tokens");
    assert!(r.again.1 < gain, "nothing restored: {} cached", r.again.1);
    assert_eq!(head.metric(priced), 1, "{log}");
    assert_eq!(head.metric(fetched), 0, "{log}");
    assert_eq!(head.metric(bytes), 0, "{log}");
    println!("prefix head recompute decision: {}", r.decision);
    for p in &mut c.procs.0 {
        assert!(p.try_wait().unwrap().is_none(), "a daemon exited");
    }
}
