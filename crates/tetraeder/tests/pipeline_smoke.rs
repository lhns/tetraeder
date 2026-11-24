//! The real binaries, once (ADR 0091, step 4; step 3 left this open): `tetraeder cluster cert`
//! for two nodes, two `tetraeder node` processes on loopback TLS (the daemons start
//! `tetraeder worker` children through `ChildLauncher`, which writes the segment plan and sets
//! `TETRAEDER_SEGMENT_PLAN`), the head profiles them (real hardware discovery), cuts a two-stage
//! plan of the tiny dense fixture into segments, assigns them and decodes a prompt and a few
//! greedy tokens through a `PipelineBackend`, against the one-node run of the same builder.
//!
//! Hardware label: cpu (CPU nodes, loopback).

mod common;

use common::{Run, argmax, check_row, two_stage_plan};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tetraeder::pipeline::{PipelineBackend, segment_assignments, start_remote_segments};
use tetraeder_control::{ClusterConfig, client};
use tetraeder_model::{BackendChoice, DenseOptions, Qwen3};
use tetraeder_scheduler::{BatchBackend, DenseBackend, SegmentBackend, SeqInput};
use tetraeder_transport::{Identity, SocketBuffers, TlsTransport, Transport};

const PROMPT: [i32; 6] = [12, 77, 3, 200, 41, 9];
const N_GREEDY: usize = 4;
/// A model is named by the 64 hex digits of its hash; any value names the copy of the fixture.
const SHA: &str = "5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed";

/// Kills the daemons when the test ends or fails. A daemon's worker sees its stdin close and
/// leaves.
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

fn step(b: &mut dyn BatchBackend, tokens: &[i32], all: bool) -> Vec<f32> {
    b.decode(&[SeqInput {
        seq: 0,
        tokens,
        all_logits: all,
    }])
    .unwrap()
    .rows(0)
    .to_vec()
}

/// hardware: cpu. Two node daemons and their workers as processes, the head in the test.
#[test]
fn cpu_two_node_processes_decode_like_one() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("pipeline-smoke-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("models")).unwrap();
    let model =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3-l4-f16.gguf");
    std::fs::copy(&model, dir.join("models").join(format!("{SHA}.gguf"))).unwrap();

    // Keys stay in each node's own folder; cluster.toml names only the certificates.
    for n in ["a", "b"] {
        let o = tetraeder()
            .args(["cluster", "cert", "--node", n, "--out"])
            .arg(dir.join(n))
            .output()
            .unwrap();
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    }
    let ports = [free_port(), free_port()];
    let mut toml = String::from("cluster = \"smoke\"\nhead = \"a\"\nnode_down_after_ms = 4000\n");
    for (n, p) in ["a", "b"].iter().zip(ports) {
        toml += &format!(
            "[[nodes]]\nid = \"{n}\"\naddr = \"127.0.0.1:{p}\"\nanchor = \"{n}/cert.pem\"\n"
        );
    }
    let cfg_path = dir.join("cluster.toml");
    std::fs::write(&cfg_path, toml).unwrap();

    let mut procs = Procs(Vec::new());
    for (n, p) in ["a", "b"].iter().zip(ports) {
        let c = tetraeder()
            .arg("node")
            .arg("--cluster")
            .arg(&cfg_path)
            .arg("--key")
            .arg(dir.join(n).join("key.pem"))
            .arg("--cert")
            .arg(dir.join(n).join("cert.pem"))
            .arg("--models-dir")
            .arg(dir.join("models"))
            .arg("--state-dir")
            .arg(dir.join(format!("state-{n}")))
            .args([
                "--worker-arg=--engine",
                "--worker-arg=dense",
                "--worker-arg=--ngl",
                "--worker-arg=0",
                "--worker-arg=--ctx",
                "--worker-arg=512",
                "--worker-arg=--parallel",
                "--worker-arg=2",
                "--worker-arg=--threads",
                "--worker-arg=2",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        procs.0.push(c);
        wait_listening(p, &format!("node {n}"));
    }

    // The head: node a's key and certificate, dial only.
    let cfg = ClusterConfig::load(&cfg_path).unwrap();
    let ident = Identity::from_pem(
        &std::fs::read(dir.join("a/cert.pem")).unwrap(),
        &std::fs::read(dir.join("a/key.pem")).unwrap(),
    )
    .unwrap();
    let head: Arc<dyn Transport> = Arc::new(
        TlsTransport::new(
            &ident,
            &cfg.anchors().unwrap(),
            &cfg.ids(),
            None,
            SocketBuffers::Auto,
        )
        .unwrap(),
    );

    // Real discovery: each daemon reports this machine's hardware graph.
    let hw = client::profile_cluster(head.as_ref(), &cfg, None).unwrap();
    assert!(hw.device("a/cpu0").is_some() && hw.device("b/cpu0").is_some());

    for (cut, head_on) in [(2u32, None), (1, Some("a/cpu0")), (3, None)] {
        let plan = two_stage_plan("tiny", 4, cut, "a/cpu0", "b/cpu0", head_on);
        let segs = segment_assignments(&plan, 0, &hw, &cfg).unwrap();
        let n_segs = segs.len();
        let remote = start_remote_segments(&head, 0, &segs, SHA, &|c| {
            c.startup_timeout = Duration::from_secs(120);
            c.deadline_floor = Duration::from_secs(60);
        })
        .unwrap();
        let mut pipe = PipelineBackend::new(
            remote
                .into_iter()
                .map(|r| Box::new(r) as Box<dyn SegmentBackend>)
                .collect(),
        )
        .unwrap();
        assert_eq!(n_segs, if head_on.is_some() { 3 } else { 2 });

        let mut whole = DenseBackend::new(
            Qwen3::load(
                &model,
                &DenseOptions {
                    n_seq_max: 2,
                    n_ctx_pool: 512,
                    threads: Some(2),
                    ..DenseOptions::new(BackendChoice::Cpu { threads: Some(2) })
                },
            )
            .unwrap(),
            false,
        );
        let nv = whole.n_vocab();
        let mut run = Run {
            max_diff: 0.0,
            bit_equal: true,
        };
        let want = step(&mut whole, &PROMPT, true);
        let got = step(&mut pipe, &PROMPT, true);
        for (g, w) in got.chunks(nv).zip(want.chunks(nv)) {
            check_row(&mut run, g, w, "prompt row");
        }
        let mut t = argmax(&want[want.len() - nv..]) as i32;
        assert_eq!(t as usize, argmax(&got[got.len() - nv..]));
        for i in 0..N_GREEDY {
            let want = step(&mut whole, &[t], false);
            let got = step(&mut pipe, &[t], false);
            check_row(&mut run, &got, &want, &format!("greedy step {i}"));
            assert_eq!(argmax(&got), argmax(&want), "greedy step {i}");
            t = argmax(&want) as i32;
        }
        println!(
            "pipeline smoke dense4 cut {cut} head {} processes 2 transport tls max_logit_diff {:e} bit_equal {}",
            if head_on.is_some() {
                "on_leader"
            } else {
                "on_last"
            },
            run.max_diff,
            run.bit_equal
        );
        drop(pipe);
    }
    // Both daemons are still up.
    for c in &mut procs.0 {
        assert!(c.try_wait().unwrap().is_none(), "a daemon exited");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
