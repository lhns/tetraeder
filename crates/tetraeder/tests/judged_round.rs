//! `tetraeder cluster judged-round` on two loopback daemons (ADR 0091, 0092): the tiny MoE cut at
//! layer 1, one segment per node, the plan's own expert slots (no flag gives them to the
//! workers), the warm request and the timed one. The files it writes are what
//! `scripts/phase6_exit_s_summary.py round` reads; the test runs that check when Python is there.
//!
//! Hardware label: cpu (loopback). The python check skips without a `python3` or `python`.

mod common;

use common::two_stage_plan;
use serde_json::Value;
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use tetraeder_model::hidden_file::HiddenFile;
use tetraeder_plan::{SlotCache, Stage};

const SHA: &str = "5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed";
const N_PROMPT: usize = 40;
const GEN: usize = 6;

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

fn python() -> Option<&'static str> {
    ["python3", "python"].into_iter().find(|p| {
        Command::new(p)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    })
}

fn json(p: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

/// hardware: cpu. Two nodes, one segment each; the round's files are complete and clean.
#[test]
fn cpu_a_judged_round_across_two_loopback_nodes_writes_the_files_the_summary_reads() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("it-judged-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("models")).unwrap();
    std::fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3moe-f16.gguf"),
        dir.join("models").join(format!("{SHA}.gguf")),
    )
    .unwrap();
    for n in ["a", "b"] {
        let o = tetraeder()
            .args(["cluster", "cert", "--node", n, "--out"])
            .arg(dir.join(n))
            .output()
            .unwrap();
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    }
    let ports = [free_port(), free_port()];
    let mut toml = String::from("cluster = \"it\"\nhead = \"a\"\n");
    for (n, p) in ["a", "b"].iter().zip(ports) {
        toml += &format!(
            "[[nodes]]\nid = \"{n}\"\naddr = \"127.0.0.1:{p}\"\nanchor = \"{n}/cert.pem\"\n"
        );
    }
    let cfg = dir.join("cluster.toml");
    std::fs::write(&cfg, toml).unwrap();
    // No slot flag: the plan's expert cache gives the workers their slots.
    let worker_args = [
        "--engine",
        "moe",
        "--ngl",
        "0",
        "--miss",
        "cpu",
        "--threads",
        "2",
        "--ctx",
        "512",
        "--parallel",
        "1",
    ];
    let mut procs = Procs(Vec::new());
    for (n, p) in ["a", "b"].iter().zip(ports) {
        let c = tetraeder()
            .arg("node")
            .arg("--cluster")
            .arg(&cfg)
            .arg("--key")
            .arg(dir.join(n).join("key.pem"))
            .arg("--cert")
            .arg(dir.join(n).join("cert.pem"))
            .arg("--models-dir")
            .arg(dir.join("models"))
            .arg("--state-dir")
            .arg(dir.join(format!("state-{n}")))
            .args(worker_args.iter().map(|a| format!("--worker-arg={a}")))
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        procs.0.push(c);
        wait_listening(p, &format!("node {n}"));
    }

    // The judged plan: the tiny MoE (2 layers, 8 experts) cut at layer 1, 4 compute slots and 4
    // RAM slots per layer on each node (they cover the 8 experts), 2 micro-batches of 16 tokens.
    let mut plan = two_stage_plan("tiny", 2, 1, "a/cpu0", "b/cpu0", None);
    plan.knobs.prefill_micro_batches = 2;
    plan.knobs.ubatch = Some(16);
    // The ids of a cluster plan carry the node; the segment plan a worker reads has them stripped.
    for (st, node) in plan.replicas[0].stages.iter_mut().zip(["a", "b"]) {
        let cache = |d: &str| SlotCache {
            device: format!("{node}/{d}"),
            slots: 4,
        };
        let Stage { expert_cache, .. } = st;
        *expert_cache = vec![cache("gpu0"), cache("cpu0")];
    }
    let plan_file = dir.join("judged-plan.toml");
    std::fs::write(&plan_file, plan.to_toml().unwrap()).unwrap();

    // The frozen prompt, as the exit harness dumps it: ids and one row per token.
    let tokens: Vec<i32> = (0..(N_PROMPT + GEN) as i32)
        .map(|i| 3 + (i * 7) % 50)
        .collect();
    let dump = HiddenFile {
        layer: 1,
        n_embd: 64,
        n_prompt: N_PROMPT,
        rows: vec![0.0; tokens.len() * 64],
        tokens: tokens.clone(),
    };
    let hidden = dir.join("hidden.bin");
    dump.write(&hidden).unwrap();

    let out = dir.join("round");
    let run = tetraeder()
        .args(["cluster", "judged-round", "--cluster"])
        .arg(&cfg)
        .arg("--key")
        .arg(dir.join("a").join("key.pem"))
        .arg("--cert")
        .arg(dir.join("a").join("cert.pem"))
        .arg("--plan")
        .arg(&plan_file)
        .args([
            "--model-sha256",
            SHA,
            "--gen",
            &GEN.to_string(),
            "--startup-timeout-s",
            "120",
            "--deadline-s",
            "60",
        ])
        .arg("--hidden")
        .arg(&hidden)
        .arg("--out")
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );

    let cluster = json(&out.join("cluster.json"));
    let c = &cluster["runs"][0];
    assert!(
        c["prefill_tok_s"].as_f64().unwrap() > 0.0 && c["decode_tok_s"].as_f64().unwrap() > 0.0
    );
    assert_eq!(c["prompt"]["tokens"], N_PROMPT);
    assert_eq!(
        c["prompt"]["fnv1a64"],
        tetraeder::judged::tokens_hash(&tokens[..N_PROMPT])
    );
    assert_eq!(c["decode_fed"].as_array().unwrap().len(), GEN);
    assert_eq!(c["step_tokens"], 32);
    for (node, layers) in [("a", [0, 1]), ("b", [1, 2])] {
        let seg = json(&out.join(format!("seg-{node}.json")));
        let r = &seg["runs"][0];
        assert_eq!(r["segment"]["layers"], serde_json::json!(layers), "{node}");
        // the slots are the plan's, covering, and the timed prefill read nothing from disk after
        // the warm request
        assert_eq!(r["tiers"]["vram_slots"], 4, "{node}");
        assert_eq!(r["tiers"]["ram_slots"], 4, "{node}");
        assert_eq!(r["warm"]["covering"], true, "{node}");
        assert_eq!(r["warm"]["requests"], 1);
        assert_eq!(r["warm"]["unclean"], false, "{node}: {}", r["warm"]);
        assert_eq!(r["prefill_io"]["bytes"], 0, "{node}");
        // no prefix reuse: every prompt token was computed in the timed prefill
        assert_eq!(r["warm"]["computed_prompt_tokens"], N_PROMPT, "{node}");
        assert_eq!(r["warm"]["prefix_reuse"], false);
        // the timed prefill ran as the plan's step (32 then 8 tokens), each cut into micro-batches
        assert!(
            r["times"]["prefill"]["prefill_steps"].as_u64().unwrap() >= 3,
            "{node}"
        );
        assert_eq!(r["times"]["decode"]["decode_steps"], GEN, "{node}");
        assert!(r["decode_layers"]["vram_pair_share"].is_number(), "{node}");
        assert!(
            r["prefill_tok_s"].as_f64().unwrap() > 0.0 && r["decode_tok_s"].as_f64().unwrap() > 0.0
        );
        std::fs::write(
            out.join(format!("seg-{node}.gpu.txt")),
            "before: 0x0, 100.0 W, 350.0 W, 350.0 W\nafter: 0x0, 300.0 W, 350.0 W, 350.0 W\n",
        )
        .unwrap();
    }

    // the verdict script's own rule for a clean round
    if let Some(py) = python() {
        let script =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/phase6_exit_s_summary.py");
        let check = Command::new(py)
            .arg("-I")
            .arg(&script)
            .arg("round")
            .arg(&out)
            .args(["--prompt-hash", c["prompt"]["fnv1a64"].as_str().unwrap()])
            .output()
            .unwrap();
        assert!(
            check.status.success(),
            "the summary calls the round unclean: {}{}",
            String::from_utf8_lossy(&check.stdout),
            String::from_utf8_lossy(&check.stderr)
        );
    } else {
        eprintln!("no python: the summary script's round check is skipped");
    }
    for c in &mut procs.0 {
        assert!(c.try_wait().unwrap().is_none(), "a daemon exited");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
