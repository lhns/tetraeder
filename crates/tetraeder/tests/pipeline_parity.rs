//! A replica across two nodes against the one-node run of the same builder (ADR 0091, step 4;
//! rule 2): the head process assigns the segments of a plan to two `tetraeder node` daemons
//! (`Assign` with the sub-plan `plan::segment_plan` cut), connects a `RemoteBackend` to each
//! worker and drives them as a `PipelineBackend`. Top-5 logits and the whole row within
//! `tol_batch` at every layer boundary, the head on the last segment or on the leader, and the
//! greedy tokens equal. Over `MemTransport` for every case, over loopback `TlsTransport` once.
//!
//! The workers are in-process (a builder on a thread behind a pipe, as `tetraeder worker` serves
//! one over stdin and stdout); `pipeline_smoke.rs` runs the real binaries.
//!
//! Hardware label: cpu.

use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tetraeder::pipeline::{
    PipelineBackend, n_layer_of, segment_assignments, segment_spec_of, start_remote_segments,
};
use tetraeder::worker_loop::serve_loop;
use tetraeder_control::ClusterConfig;
use tetraeder_control::node::{Assignment, Daemon, Launcher, NodeOptions, RestartPolicy, WorkerIo};
use tetraeder_model::{BackendChoice, DenseOptions, LoadOptions, Qwen3, Qwen3Moe, SegmentSpec};
use tetraeder_plan::{KvFormat, Plan};
use tetraeder_profiler::HardwareGraph;
use tetraeder_scheduler::{BatchBackend, DenseBackend, MoeBackend, SeqInput};
use tetraeder_transport::identity::{Generated, certs_from_pem, generate_self_signed};
use tetraeder_transport::{
    CertificateDer, Identity, MemTransport, NodeId, SocketBuffers, TlsTransport, Transport,
};

mod common;

use common::{Run, argmax, check_row, two_stage_plan};

const CPU: BackendChoice = BackendChoice::Cpu { threads: Some(2) };
const PROMPT: [i32; 6] = [12, 77, 3, 200, 41, 9];
const N_GREEDY: usize = 8;
const SHA: &str = "0000000000000000000000000000000000000000000000000000000000000000";

#[derive(Clone, Copy, Debug)]
enum Kind {
    Dense,
    Moe,
}

#[derive(Clone)]
struct Case {
    name: &'static str,
    kind: Kind,
    path: PathBuf,
    n_layer: u32,
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "dense4",
            kind: Kind::Dense,
            path: fixture("tiny-qwen3-l4-f16.gguf"),
            n_layer: 4,
        },
        Case {
            name: "dense2",
            kind: Kind::Dense,
            path: fixture("tiny-qwen3-f16.gguf"),
            n_layer: 2,
        },
        Case {
            name: "moe2",
            kind: Kind::Moe,
            path: fixture("tiny-qwen3moe-f16.gguf"),
            n_layer: 2,
        },
    ]
}

fn dense_opts(segment: Option<SegmentSpec>) -> DenseOptions {
    DenseOptions {
        kv: KvFormat::F16,
        kv_rotation: Some(false),
        extra_bufts: false,
        n_seq_max: 2,
        n_ctx_pool: 512,
        segment,
        ..DenseOptions::new(CPU)
    }
}

fn moe_opts(segment: Option<SegmentSpec>) -> LoadOptions {
    LoadOptions {
        kv: KvFormat::F16,
        kv_rotation: Some(false),
        n_seq_max: 2,
        n_ctx: 512,
        segment,
        ..LoadOptions::new(CPU)
    }
}

/// The whole model of the case, the one-node reference.
fn whole(c: &Case) -> Box<dyn BatchBackend> {
    match c.kind {
        Kind::Dense => Box::new(DenseBackend::new(
            Qwen3::load(&c.path, &dense_opts(None)).unwrap(),
            false,
        )),
        Kind::Moe => Box::new(MoeBackend::new(
            Qwen3Moe::load(&c.path, &moe_opts(None)).unwrap(),
            false,
        )),
    }
}

// ---- in-process workers behind pipes ----

struct ChanWriter(Sender<Vec<u8>>);

impl Write for ChanWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0
            .send(buf.to_vec())
            .map_err(|_| io::Error::from(io::ErrorKind::BrokenPipe))?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

struct ChanReader {
    rx: Receiver<Vec<u8>>,
    buf: Vec<u8>,
    dead: Arc<AtomicBool>,
}

impl Read for ChanReader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        while self.buf.is_empty() {
            if self.dead.load(Ordering::SeqCst) {
                return Ok(0);
            }
            match self.rx.recv_timeout(Duration::from_millis(10)) {
                Ok(b) => self.buf = b,
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return Ok(0),
            }
        }
        let n = out.len().min(self.buf.len());
        out[..n].copy_from_slice(&self.buf[..n]);
        self.buf.drain(..n);
        Ok(n)
    }
}

fn pipe(dead: &Arc<AtomicBool>) -> (ChanWriter, ChanReader) {
    let (tx, rx) = mpsc::channel();
    (
        ChanWriter(tx),
        ChanReader {
            rx,
            buf: Vec::new(),
            dead: Arc::clone(dead),
        },
    )
}

/// Starts the worker of an assignment: the share of the model its sub-plan names, served by
/// `serve_loop` on a thread of its own (the builders are not `Send`, so it loads there).
struct ModelLauncher {
    case: Case,
    /// The specs the assignments gave, in order.
    seen: Mutex<Vec<SegmentSpec>>,
}

impl Launcher for ModelLauncher {
    fn launch(&self, a: &Assignment) -> io::Result<WorkerIo> {
        let n_layer = n_layer_of(&self.case.path).map_err(io::Error::other)?;
        let spec = segment_spec_of(&a.plan, n_layer).map_err(io::Error::other)?;
        self.seen.lock().unwrap().push(spec.clone());
        let dead = Arc::new(AtomicBool::new(false));
        let ((in_tx, mut in_rx), (out_tx, out_rx)) = (pipe(&dead), pipe(&dead));
        let (kind, path) = (self.case.kind, self.case.path.clone());
        std::thread::spawn(move || {
            let mut backend: Box<dyn BatchBackend> = match kind {
                Kind::Dense => Box::new(DenseBackend::new(
                    Qwen3::load(&path, &dense_opts(Some(spec))).unwrap(),
                    false,
                )),
                Kind::Moe => Box::new(MoeBackend::new(
                    Qwen3Moe::load(&path, &moe_opts(Some(spec))).unwrap(),
                    false,
                )),
            };
            let mut out = out_tx;
            let _ = serve_loop(backend.as_mut(), &mut in_rx, &mut out, &[]);
        });
        let d = Arc::clone(&dead);
        Ok(WorkerIo {
            stdin: Box::new(in_tx),
            stdout: Box::new(out_rx),
            kill: Arc::new(move || d.store(true, Ordering::SeqCst)),
        })
    }
}

// ---- the cluster ----

fn cluster_config(ports: &[u16], dir: &Path) -> ClusterConfig {
    let mut t = String::from("cluster = \"pipeline\"\nhead = \"a\"\nnode_down_after_ms = 2000\n");
    for (n, p) in ["a", "b"].iter().zip(ports) {
        t += &format!("[[nodes]]\nid = \"{n}\"\naddr = \"127.0.0.1:{p}\"\nanchor = \"{n}.pem\"\n");
    }
    let path = dir.join("cluster.toml");
    std::fs::write(&path, t).unwrap();
    ClusterConfig::load(&path).unwrap()
}

fn hardware() -> HardwareGraph {
    HardwareGraph::from_toml(include_str!(
        "../../costmodel/fixtures/cluster/cluster_two_3090_lan.toml"
    ))
    .unwrap()
}

/// The plan of two stages across nodes `first` and `second`, the head on `second` or, as a rule of
/// its own, on `head_on`.
fn plan(c: &Case, cut: u32, first: &str, second: &str, head_on: Option<&str>) -> Plan {
    let dev = |n: &str| format!("{n}/gpu0");
    two_stage_plan(
        c.name,
        c.n_layer,
        cut,
        &dev(first),
        &dev(second),
        head_on.map(dev).as_deref(),
    )
}

/// One decode of a sequence on a backend: the rows of the input.
fn step(b: &mut dyn BatchBackend, seq: u32, tokens: &[i32], all: bool) -> Vec<f32> {
    let out = b
        .decode(&[SeqInput {
            seq,
            tokens,
            all_logits: all,
        }])
        .unwrap();
    out.rows(0).to_vec()
}

/// The prompt with all logits and `N_GREEDY` greedy steps on both backends; the pipeline must
/// agree with the whole model at every row.
fn compare(whole: &mut dyn BatchBackend, pipe: &mut dyn BatchBackend, what: &str) -> Run {
    let mut run = Run {
        max_diff: 0.0,
        bit_equal: true,
    };
    let nv = whole.n_vocab();
    assert_eq!(pipe.n_vocab(), nv);
    let want = step(whole, 0, &PROMPT, true);
    let got = step(pipe, 0, &PROMPT, true);
    assert_eq!(got.len(), PROMPT.len() * nv, "{what}");
    for (g, w) in got.chunks(nv).zip(want.chunks(nv)) {
        check_row(&mut run, g, w, &format!("{what} prompt row"));
    }
    let mut t = argmax(&want[want.len() - nv..]) as i32;
    assert_eq!(t as usize, argmax(&got[got.len() - nv..]), "{what}");
    for i in 0..N_GREEDY {
        let want = step(whole, 0, &[t], false);
        let got = step(pipe, 0, &[t], false);
        check_row(&mut run, &got, &want, &format!("{what} greedy step {i}"));
        assert_eq!(argmax(&got), argmax(&want), "{what} step {i}");
        t = argmax(&want) as i32;
    }
    assert_eq!(whole.seq_len(0), pipe.seq_len(0), "{what}");
    run
}

/// Everything the daemons of one test hold.
struct Nodes {
    _daemons: Vec<Daemon>,
    head: Arc<dyn Transport>,
    cfg: ClusterConfig,
    launcher: Arc<ModelLauncher>,
    dir: PathBuf,
    name: &'static str,
}

impl Drop for Nodes {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn scratch(tag: &str) -> PathBuf {
    // Tests of one process run in parallel: each call gets a folder of its own.
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::SeqCst);
    let dir =
        std::env::temp_dir().join(format!("tetraeder-pipe-{tag}-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn daemon(cfg: &ClusterConfig, t: Arc<dyn Transport>, launcher: Arc<ModelLauncher>) -> Daemon {
    let dial = Arc::clone(&t);
    Daemon::start(
        t,
        NodeOptions {
            cfg: cfg.clone(),
            launcher,
            profile: Arc::new(|| Ok("{}".to_owned())),
            dial: Arc::new(move |_| Ok(Arc::clone(&dial))),
            restart: RestartPolicy {
                backoff_base: Duration::from_millis(20),
                max_restarts: 3,
                window: Duration::from_secs(60),
            },
        },
    )
}

/// Two daemons on a `MemTransport` switchboard, the head dialing as node `a`.
fn mem_nodes(c: &Case) -> Nodes {
    let dir = scratch(&format!("mem-{}", c.name));
    let board = MemTransport::switchboard();
    let cfg = cluster_config(&[1, 2], &dir);
    let launcher = Arc::new(ModelLauncher {
        case: c.clone(),
        seen: Mutex::new(Vec::new()),
    });
    let daemons = ["a", "b"]
        .iter()
        .map(|n| {
            let t: Arc<dyn Transport> = Arc::new(board.node(&NodeId::new(*n)));
            daemon(&cfg, t, Arc::clone(&launcher))
        })
        .collect();
    let head: Arc<dyn Transport> = Arc::new(board.node(&NodeId::new("a")));
    Nodes {
        _daemons: daemons,
        head,
        cfg,
        launcher,
        dir,
        name: "mem",
    }
}

/// Two daemons on loopback TLS with pinned self-signed certificates.
fn tls_nodes(c: &Case) -> Nodes {
    let dir = scratch(&format!("tls-{}", c.name));
    let names = [NodeId::new("a"), NodeId::new("b")];
    let gens: Vec<(&str, Generated)> = ["a", "b"]
        .iter()
        .map(|n| (*n, generate_self_signed(n).unwrap()))
        .collect();
    let mut trust: Vec<CertificateDer<'static>> = Vec::new();
    for (n, g) in &gens {
        std::fs::write(dir.join(format!("{n}.pem")), &g.cert_pem).unwrap();
        trust.extend(certs_from_pem(g.cert_pem.as_bytes()).unwrap());
    }
    let ident = |n: &str| {
        let g = &gens.iter().find(|(x, _)| *x == n).unwrap().1;
        Identity::from_pem(g.cert_pem.as_bytes(), g.key_pem.as_bytes()).unwrap()
    };
    let tls = |n: &str, listen: Option<&str>| {
        TlsTransport::new(&ident(n), &trust, &names, listen, SocketBuffers::Auto).unwrap()
    };
    let (ta, tb) = (tls("a", Some("127.0.0.1:0")), tls("b", Some("127.0.0.1:0")));
    let ports = [
        ta.local_addr().unwrap().port(),
        tb.local_addr().unwrap().port(),
    ];
    let cfg = cluster_config(&ports, &dir);
    let launcher = Arc::new(ModelLauncher {
        case: c.clone(),
        seen: Mutex::new(Vec::new()),
    });
    let daemons = vec![
        daemon(&cfg, Arc::new(ta), Arc::clone(&launcher)),
        daemon(&cfg, Arc::new(tb), Arc::clone(&launcher)),
    ];
    // The head dials from node `a`'s identity, with a transport that does not listen.
    let head: Arc<dyn Transport> = Arc::new(tls("a", None));
    Nodes {
        _daemons: daemons,
        head,
        cfg,
        launcher,
        dir,
        name: "tls",
    }
}

impl Nodes {
    /// Assigns the plan's segments and drives them as one backend.
    fn pipeline(&self, p: &Plan) -> PipelineBackend {
        let segs = segment_assignments(p, 0, &hardware(), &self.cfg).unwrap();
        let remote = start_remote_segments(&self.head, 0, &segs, SHA, &|c| {
            c.startup_timeout = Duration::from_secs(60);
            c.deadline_floor = Duration::from_secs(30);
        })
        .unwrap();
        PipelineBackend::new(
            remote
                .into_iter()
                .map(|r| Box::new(r) as Box<dyn tetraeder_scheduler::SegmentBackend>)
                .collect(),
        )
        .unwrap()
    }
}

fn run_case(nodes: &Nodes, c: &Case, cut: u32, head_apart: bool, inverted: bool) {
    let (first, second) = if inverted { ("b", "a") } else { ("a", "b") };
    let head_on = head_apart.then_some("a");
    let p = plan(c, cut, first, second, head_on);
    let mut pipe = nodes.pipeline(&p);
    let n = pipe.segments().len();
    assert_eq!(n, if head_apart && second != "a" { 3 } else { 2 }, "{p:#?}");
    let mut reference = whole(c);
    let what = format!(
        "{} cut {cut} head_apart {head_apart} inverted {inverted}",
        c.name
    );
    let r = compare(reference.as_mut(), &mut pipe, &what);
    println!(
        "pipeline parity {} cut {cut} head {} nodes {} transport {} max_logit_diff {:e} bit_equal {}",
        c.name,
        if head_apart { "on_leader" } else { "on_last" },
        if inverted { "b_then_a" } else { "a_then_b" },
        nodes.name,
        r.max_diff,
        r.bit_equal
    );
    // The workers loaded their share only.
    let seen = nodes.launcher.seen.lock().unwrap();
    assert!(seen.iter().any(|s| s.embed) && seen.iter().any(|s| s.head));
}

/// hardware: cpu. Every layer boundary of every fixture, the head on the last segment (node
/// `b`) or on the leader (node `a`), over `MemTransport`.
#[test]
fn cpu_two_nodes_match_one_node_over_mem() {
    for c in cases() {
        let nodes = mem_nodes(&c);
        for cut in 1..c.n_layer {
            run_case(&nodes, &c, cut, false, false);
            run_case(&nodes, &c, cut, true, false);
        }
    }
}

/// hardware: cpu. The first segment on node `b`, so the leader's own node holds the last
/// segment and the head, and the head on the leader after a middle segment on `b`.
#[test]
fn cpu_two_nodes_match_one_node_with_the_leader_last() {
    let c = cases().remove(0);
    let nodes = mem_nodes(&c);
    for cut in 1..c.n_layer {
        run_case(&nodes, &c, cut, false, true);
        run_case(&nodes, &c, cut, true, true);
    }
}

/// hardware: cpu. The same over loopback TLS with pinned certificates, once per builder: the
/// middle boundary, head on the last segment and on the leader.
#[test]
fn cpu_two_nodes_match_one_node_over_tls() {
    for c in cases().into_iter().filter(|c| c.n_layer > 1) {
        let nodes = tls_nodes(&c);
        let cut = c.n_layer / 2;
        run_case(&nodes, &c, cut, false, false);
        run_case(&nodes, &c, cut, true, false);
    }
}

/// hardware: cpu. Snapshots across segments: a snapshot of the pipeline restores into another
/// sequence of a pipeline cut the same way (equal rows after), and a snapshot of another size is
/// refused. The snapshot is the segments' snapshots in order, not the one-node layout.
#[test]
fn cpu_pipeline_snapshots_round_trip() {
    for c in cases().into_iter().filter(|c| c.n_layer > 1) {
        let nodes = mem_nodes(&c);
        let p = plan(&c, c.n_layer / 2, "a", "b", None);
        let mut pipe = nodes.pipeline(&p);
        step(&mut pipe, 0, &PROMPT, false);
        let snap = pipe.seq_snapshot(0).unwrap();
        let n = pipe.seq_len(0);
        assert_eq!(snap.len(), n * pipe.kv_row_bytes(), "{}", c.name);
        // One-node snapshot of the same tokens: same size, other layout.
        let mut one = whole(&c);
        step(one.as_mut(), 0, &PROMPT, false);
        let one_snap = one.seq_snapshot(0).unwrap();
        assert_eq!(snap.len(), one_snap.len(), "{}", c.name);
        println!(
            "pipeline snapshot {} bytes {} equal_to_one_node {}",
            c.name,
            snap.len(),
            snap == one_snap
        );
        pipe.seq_restore(1, &snap, n).unwrap();
        assert_eq!(pipe.seq_len(1), n);
        let a = step(&mut pipe, 0, &[5], false);
        let b = step(&mut pipe, 1, &[5], false);
        assert_eq!(
            a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "{}",
            c.name
        );
        let e = pipe.seq_restore(1, &snap[..snap.len() - 4], n).unwrap_err();
        assert!(e.to_string().contains("cut the same way"), "{e}");
        // seq_cp and seq_rm go to every segment.
        pipe.seq_rm(1, 0).unwrap();
        pipe.seq_cp(0, 1, n).unwrap();
        assert_eq!(pipe.seq_len(1), n, "{}", c.name);
    }
}

/// hardware: cpu. A worker that fails a step: the pipeline reports the error and the earlier
/// segments are back at the length from before, so the next step on the same sequence works.
#[test]
fn cpu_a_failed_step_is_rolled_back_on_every_segment() {
    let c = cases().remove(0);
    let nodes = mem_nodes(&c);
    let p = plan(&c, 2, "a", "b", None);
    let mut pipe = nodes.pipeline(&p);
    step(&mut pipe, 0, &PROMPT, false);
    let before = pipe.seq_len(0);
    // A token outside the vocabulary: the first segment refuses it.
    let bad = pipe.decode(&[SeqInput {
        seq: 0,
        tokens: &[1_000_000],
        all_logits: false,
    }]);
    assert!(bad.is_err());
    assert_eq!(pipe.seq_len(0), before);
    // The head segment refuses hidden rows of the wrong size: that is the second segment
    // failing after the first advanced. A decode of two sequences where the second is unknown.
    let bad = pipe.decode(&[
        SeqInput {
            seq: 0,
            tokens: &[3],
            all_logits: false,
        },
        SeqInput {
            seq: 7,
            tokens: &[3],
            all_logits: false,
        },
    ]);
    assert!(bad.is_err());
    assert_eq!(pipe.seq_len(0), before);
    let ok = step(&mut pipe, 0, &[3], false);
    assert_eq!(ok.len(), pipe.n_vocab());
}
