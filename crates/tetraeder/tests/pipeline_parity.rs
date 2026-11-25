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
use tetraeder_scheduler::{
    BackendError, BatchBackend, DenseBackend, MOE_CHUNK, MoeBackend, SeqInput,
};
use tetraeder_transport::identity::{Generated, certs_from_pem, generate_self_signed};
use tetraeder_transport::{
    CertificateDer, Identity, LinkFault, MemTransport, NodeId, SocketBuffers, Switchboard,
    TlsTransport, Transport,
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
    /// Tokens per model step of the MoE backends (the plan's ubatch); `None`: the default.
    chunk: Option<usize>,
    /// KV pool rows of the segment that holds the head and not the embedding.
    tail_pool: usize,
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
            chunk: None,
            tail_pool: 512,
        },
        Case {
            name: "dense2",
            kind: Kind::Dense,
            path: fixture("tiny-qwen3-f16.gguf"),
            n_layer: 2,
            chunk: None,
            tail_pool: 512,
        },
        Case {
            name: "moe2",
            kind: Kind::Moe,
            path: fixture("tiny-qwen3moe-f16.gguf"),
            n_layer: 2,
            chunk: None,
            tail_pool: 512,
        },
    ]
}

/// The cases with steps that mix decoders and prompt chunks, and a MoE that runs four tokens
/// per model step so a long input is several pieces.
fn mixed_cases() -> Vec<Case> {
    let mut v = cases();
    v.truncate(1);
    v.push(Case {
        name: "moe2_chunk4",
        chunk: Some(4),
        ..cases().remove(2)
    });
    v
}

fn dense_opts(segment: Option<SegmentSpec>, pool: usize) -> DenseOptions {
    DenseOptions {
        kv: KvFormat::F16,
        kv_rotation: Some(false),
        extra_bufts: false,
        n_seq_max: 2,
        n_ctx_pool: pool,
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
            Qwen3::load(&c.path, &dense_opts(None, 512)).unwrap(),
            false,
        )),
        Kind::Moe => Box::new(MoeBackend::with_chunk(
            Qwen3Moe::load(&c.path, &moe_opts(None)).unwrap(),
            false,
            c.chunk.unwrap_or(MOE_CHUNK),
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
        let chunk = self.case.chunk.unwrap_or(MOE_CHUNK);
        let pool = if spec.head && !spec.embed {
            self.case.tail_pool
        } else {
            512
        };
        std::thread::spawn(move || {
            let mut backend: Box<dyn BatchBackend> = match kind {
                Kind::Dense => Box::new(DenseBackend::new(
                    Qwen3::load(&path, &dense_opts(Some(spec), pool)).unwrap(),
                    false,
                )),
                Kind::Moe => Box::new(MoeBackend::with_chunk(
                    Qwen3Moe::load(&path, &moe_opts(Some(spec))).unwrap(),
                    false,
                    chunk,
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
    /// The simulated network of a `MemTransport` cluster, to put a fault on a link.
    board: Option<Arc<Switchboard>>,
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
    let dir = std::env::temp_dir().join(format!("tetraeder-pipe-{tag}-{}-{n}", std::process::id()));
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
        board: Some(board),
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
        board: None,
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
        // The key of the prefix cache carries the cut points, so a one-node snapshot, or one of
        // another cut, is not offered to this pipeline.
        assert_ne!(pipe.kv_format(), one.kv_format(), "{}", c.name);
        assert!(pipe.kv_format().starts_with(&one.kv_format()), "{}", c.name);
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

/// hardware: cpu. The tail segment has a pool half the size: a step that overflows it fails with
/// `KvFull` there after the first segment has taken it. The pipeline cuts the first segment back,
/// undoes an overflowing reservation, and the next step on the same sequence equals the whole
/// model's.
#[test]
fn cpu_a_failed_step_is_rolled_back_on_every_segment() {
    let c = Case {
        tail_pool: 256,
        ..cases().remove(0)
    };
    let nodes = mem_nodes(&c);
    let p = plan(&c, 2, "a", "b", None);
    let mut pipe = nodes.pipeline(&p);
    let mut reference = whole(&c);
    let want = step(reference.as_mut(), 0, &PROMPT, false);
    let got = step(&mut pipe, 0, &PROMPT, false);
    let mut run = Run {
        max_diff: 0.0,
        bit_equal: true,
    };
    check_row(&mut run, &got, &want, "prompt");
    let before = (pipe.seq_len(0), pipe.seq_reserved(0));
    // A reservation the tail segment cannot give: the first segment gives its share back.
    let e = pipe.seq_reserve(0, 300).unwrap_err();
    assert!(matches!(e, BackendError::KvFull { .. }), "{e}");
    assert_eq!((pipe.seq_len(0), pipe.seq_reserved(0)), before);
    // A step that outgrows the tail segment: the first one took it, then was cut back.
    let long: Vec<i32> = (0..300).map(|i| i % 200).collect();
    let e = pipe
        .decode(&[SeqInput {
            seq: 0,
            tokens: &long,
            all_logits: false,
        }])
        .err()
        .unwrap();
    assert!(matches!(e, BackendError::KvFull { .. }), "{e}");
    assert_eq!(pipe.seq_len(0), before.0);
    assert!(!pipe.seq_lost(0));
    // A token the first segment refuses.
    assert!(
        pipe.decode(&[SeqInput {
            seq: 0,
            tokens: &[1_000_000],
            all_logits: false,
        }])
        .is_err()
    );
    assert_eq!(pipe.seq_len(0), before.0);
    // Both segments are where the whole model is: the next step agrees with it.
    let want = step(reference.as_mut(), 0, &[3], false);
    let got = step(&mut pipe, 0, &[3], false);
    check_row(&mut run, &got, &want, "step after the failures");
    println!(
        "pipeline rollback dense4 max_logit_diff {:e} bit_equal {}",
        run.max_diff, run.bit_equal
    );
}

/// The rows of every input of one step.
fn rows(b: &mut dyn BatchBackend, ins: &[(u32, &[i32], bool)]) -> Vec<Vec<f32>> {
    let inputs: Vec<SeqInput<'_>> = ins
        .iter()
        .map(|&(seq, tokens, all_logits)| SeqInput {
            seq,
            tokens,
            all_logits,
        })
        .collect();
    let out = b.decode(&inputs).unwrap();
    (0..out.len()).map(|i| out.rows(i).to_vec()).collect()
}

/// hardware: cpu. Steps as the scheduler makes them on the CPU: a prompt chunk of 70 tokens with
/// a short input of five, decoders next to each other, then a decoder with a chunk. The worker
/// runs them as several groups, and a MoE that takes four tokens per model step as several
/// pieces; the hidden rows of each group and piece are cut from the payload in input order.
#[test]
fn cpu_mixed_steps_cut_the_hidden_rows_per_group_and_piece() {
    for c in mixed_cases() {
        let nodes = mem_nodes(&c);
        let p = plan(&c, c.n_layer / 2, "a", "b", None);
        let mut pipe = nodes.pipeline(&p);
        let mut reference = whole(&c);
        let long: Vec<i32> = (0..70).map(|i| (i * 7 + 3) % 200).collect();
        let short = [5, 9, 33, 100, 7];
        let steps: Vec<Vec<(u32, &[i32], bool)>> = vec![
            vec![(0, &long, false), (1, &short, true)],
            vec![(0, &[11], false), (1, &[12], false)],
            vec![(0, &long[..66], false), (1, &[13], true)],
        ];
        let mut run = Run {
            max_diff: 0.0,
            bit_equal: true,
        };
        for (n, ins) in steps.iter().enumerate() {
            let want = rows(reference.as_mut(), ins);
            let got = rows(&mut pipe, ins);
            assert_eq!(got.len(), want.len(), "{} step {n}", c.name);
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g.len(), w.len(), "{} step {n} input {i}", c.name);
                for (gr, wr) in g.chunks(pipe.n_vocab()).zip(w.chunks(pipe.n_vocab())) {
                    check_row(&mut run, gr, wr, &format!("{} step {n} input {i}", c.name));
                }
            }
        }
        println!(
            "pipeline mixed steps {} max_logit_diff {:e} bit_equal {}",
            c.name, run.max_diff, run.bit_equal
        );
    }
}

// ---- pipelined prefill (ADR 0077, ADR 0091 step 2 pricing) ----

/// A prompt of `n` tokens.
fn prompt(n: usize) -> Vec<i32> {
    (0..n).map(|i| ((i * 7 + 3) % 200) as i32).collect()
}

/// A prompt of `n` tokens prefilled walking the segments (sequence 0) and pipelined in `micro`
/// micro-batches (sequence 1), against the one-node run: every row within `tol_batch`, the
/// greedy tokens equal.
fn pipelined_case(nodes: &Nodes, c: &Case, cut: u32, head_apart: bool, micro: u32, n: usize) {
    let p = plan(c, cut, "a", "b", head_apart.then_some("a"));
    let mut pipe = nodes.pipeline(&p);
    let mut reference = whole(c);
    let what = format!(
        "{} cut {cut} head_apart {head_apart} micro {micro} tokens {n}",
        c.name
    );
    let toks = prompt(n);
    let nv = reference.n_vocab();
    let want = step(reference.as_mut(), 0, &toks, true);
    pipe.set_prefill_micro_batches(1);
    let walk = step(&mut pipe, 0, &toks, true);
    pipe.set_prefill_micro_batches(micro);
    let piped = step(&mut pipe, 1, &toks, true);
    assert_eq!(piped.len(), n * nv, "{what}");
    let mut run = Run {
        max_diff: 0.0,
        bit_equal: true,
    };
    for (i, ((g, w), k)) in piped
        .chunks(nv)
        .zip(want.chunks(nv))
        .zip(walk.chunks(nv))
        .enumerate()
    {
        check_row(&mut run, g, w, &format!("{what} row {i} against one node"));
        check_row(&mut run, g, k, &format!("{what} row {i} against the walk"));
    }
    let mut t = argmax(&want[want.len() - nv..]) as i32;
    assert_eq!(t as usize, argmax(&piped[piped.len() - nv..]), "{what}");
    for i in 0..N_GREEDY {
        let w = step(reference.as_mut(), 0, &[t], false);
        let a = step(&mut pipe, 0, &[t], false);
        let b = step(&mut pipe, 1, &[t], false);
        assert_eq!(argmax(&a), argmax(&w), "{what} walk step {i}");
        assert_eq!(argmax(&b), argmax(&w), "{what} pipelined step {i}");
        t = argmax(&w) as i32;
    }
    assert_eq!(pipe.seq_len(1), reference.seq_len(0), "{what}");
    println!(
        "pipelined prefill parity {what} max_logit_diff {:e} bit_equal {}",
        run.max_diff, run.bit_equal
    );
}

/// hardware: cpu. Pipelined prefill against the walk and the one-node run at the first, middle
/// and last boundary of the 4-layer model, with the head on a segment of its own, on the
/// 2-layer model, and on a MoE that runs four tokens per model step.
#[test]
fn cpu_pipelined_prefill_matches_the_walk_and_one_node() {
    let c = cases().remove(0);
    let nodes = mem_nodes(&c);
    for cut in 1..c.n_layer {
        pipelined_case(&nodes, &c, cut, false, 3, 64);
    }
    pipelined_case(&nodes, &c, 2, true, 4, 64);
    let c = cases().remove(1);
    pipelined_case(&mem_nodes(&c), &c, 1, false, 2, 33);
    let c = mixed_cases().remove(1);
    pipelined_case(&mem_nodes(&c), &c, 1, false, 3, 30);
}

/// hardware: cpu. A pipelined prefill that outgrows the tail segment: the segments hold
/// micro-batches by then, and every one is cut back; the next step equals the one-node run.
#[test]
fn cpu_a_failed_pipelined_prefill_is_rolled_back_on_every_segment() {
    let c = Case {
        tail_pool: 256,
        ..cases().remove(0)
    };
    let nodes = mem_nodes(&c);
    let p = plan(&c, 2, "a", "b", None);
    let mut pipe = nodes.pipeline(&p);
    pipe.set_prefill_micro_batches(4);
    let mut reference = whole(&c);
    let mut run = Run {
        max_diff: 0.0,
        bit_equal: true,
    };
    let want = step(reference.as_mut(), 0, &PROMPT, false);
    let got = step(&mut pipe, 0, &PROMPT, false);
    check_row(&mut run, &got, &want, "prompt");
    let before = pipe.seq_len(0);
    let long = prompt(300);
    let e = pipe
        .decode(&[SeqInput {
            seq: 0,
            tokens: &long,
            all_logits: false,
        }])
        .err()
        .unwrap();
    assert!(matches!(e, BackendError::KvFull { .. }), "{e}");
    assert_eq!(pipe.seq_len(0), before);
    assert!(!pipe.seq_lost(0));
    let want = step(reference.as_mut(), 0, &[3], false);
    let got = step(&mut pipe, 0, &[3], false);
    check_row(&mut run, &got, &want, "step after the failure");
}

/// hardware: cpu. With the links of the middle segment slowed (bytes per second, no latency), a
/// pipelined prefill keeps the link in and the link out busy at the same time and takes less
/// wall time than the walk, which pays one after the other. The model compute is a few
/// milliseconds, so the links decide.
#[test]
fn cpu_a_pipelined_prefill_overlaps_the_segments_on_a_slow_link() {
    let c = cases().remove(0);
    let nodes = mem_nodes(&c);
    // Embedding on `a`, a middle segment on `b`, the head on `a`: hidden rows cross the link to
    // `b` and back.
    let p = plan(&c, 2, "a", "b", Some("a"));
    let mut pipe = nodes.pipeline(&p);
    assert_eq!(pipe.segments().len(), 3);
    let n = 64;
    let toks = prompt(n);
    let hidden_bytes = (n * pipe.segments()[0].n_embd * 4) as f64;
    let link_s = 0.3;
    let slow = LinkFault::Slow {
        alpha: Duration::ZERO,
        bytes_per_s: hidden_bytes / link_s,
    };
    let board = nodes.board.as_ref().unwrap();
    // Warm up the workers before the clock runs, then put the fault on.
    step(&mut pipe, 1, &toks[..4], false);
    pipe.seq_rm(1, 0).unwrap();
    board.fault(&NodeId::new("a"), &NodeId::new("b"), slow);
    board.fault(&NodeId::new("b"), &NodeId::new("a"), slow);
    pipe.set_prefill_micro_batches(1);
    let t = std::time::Instant::now();
    let walk = step(&mut pipe, 0, &toks, false);
    let walk_s = t.elapsed().as_secs_f64();
    pipe.set_prefill_micro_batches(4);
    let t = std::time::Instant::now();
    let piped = step(&mut pipe, 1, &toks, false);
    let piped_s = t.elapsed().as_secs_f64();
    println!(
        "pipelined prefill slow link walk {walk_s:.3} s pipelined {piped_s:.3} s ratio {:.2}",
        piped_s / walk_s
    );
    let mut run = Run {
        max_diff: 0.0,
        bit_equal: true,
    };
    check_row(&mut run, &piped, &walk, "last row");
    // The walk pays the link in and the link out: about two link times.
    assert!(walk_s > 1.6 * link_s, "the fault is on: walk {walk_s:.3} s");
    // Four micro-batches: about 1.25 link times in the ideal case.
    assert!(
        piped_s < 0.85 * walk_s,
        "no overlap: pipelined {piped_s:.3} s, walk {walk_s:.3} s"
    );
}
