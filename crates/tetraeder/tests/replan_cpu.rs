//! `serve --plan auto` end to end on the tiny MoE fixture, CPU only, through both engines
//! (ADR 0055): the plan is chosen from a hardware file, a trigger over the admin route changes the
//! hardware under it, the server drains and restarts, and greedy output stays the same.
//!
//! Hardware label: cpu.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

const CPU_ONLY: &str = include_str!("../../planner/fixtures/cpu_only.toml");

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3moe-f16.gguf")
}

fn hw_with_threads(n: u32) -> String {
    CPU_ONLY.replace("threads = 8", &format!("threads = {n}"))
}

/// The server process, killed when dropped (also on a panic).
struct Server {
    child: Child,
    api: String,
    admin: String,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn spawn(engine: &str, hw: &Path) -> Server {
    let mut child = Command::new(env!("CARGO_BIN_EXE_tetraeder"))
        .args(["serve", "-m"])
        .arg(fixture())
        .args(["--plan", "auto", "--hw"])
        .arg(hw)
        .args(["--port", "0", "--admin", "127.0.0.1:0", "--ctx", "256"])
        .args(["--engine", engine])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn tetraeder serve");
    let stderr = child.stderr.take().unwrap();
    let (tx, rx) = mpsc::channel::<String>();
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    let (mut api, mut admin) = (None, None);
    let mut log = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(120);
    let mut server = Server {
        child,
        api: String::new(),
        admin: String::new(),
    };
    while api.is_none() || admin.is_none() {
        let left = deadline.saturating_duration_since(Instant::now());
        let line = rx.recv_timeout(left).unwrap_or_else(|e| {
            panic!(
                "server did not come up ({e}); stderr so far:\n{}",
                log.join("\n")
            )
        });
        if let Some(a) = line.strip_prefix("listening on http://") {
            api = Some(a.trim().to_string());
        }
        if let Some(a) = line.strip_prefix("admin on http://") {
            admin = Some(a.trim().to_string());
        }
        log.push(line);
    }
    // Keep reading so the child never blocks on a full pipe.
    std::thread::spawn(move || for _ in rx {});
    server.api = api.unwrap();
    server.admin = admin.unwrap();
    server
}

fn dechunk(mut b: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let Some(nl) = b.windows(2).position(|w| w == b"\r\n") else {
            return out;
        };
        let n =
            usize::from_str_radix(std::str::from_utf8(&b[..nl]).unwrap().trim(), 16).unwrap_or(0);
        if n == 0 {
            return out;
        }
        out.extend_from_slice(&b[nl + 2..nl + 2 + n]);
        b = &b[nl + 2 + n + 2..];
    }
}

/// One HTTP/1.1 request with `Connection: close`; returns the status and the body.
fn http(addr: &str, method: &str, path: &str, body: Option<&str>) -> (u16, String) {
    let mut s = TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(300))).unwrap();
    let body = body.unwrap_or("");
    write!(
        s,
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).unwrap();
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .expect("response headers");
    let head = String::from_utf8_lossy(&raw[..split]).to_lowercase();
    let status = head.split_whitespace().nth(1).unwrap().parse().unwrap();
    let payload = &raw[split + 4..];
    let bytes = if head.contains("transfer-encoding: chunked") {
        dechunk(payload)
    } else {
        payload.to_vec()
    };
    (status, String::from_utf8(bytes).unwrap())
}

fn json(s: &str) -> serde_json::Value {
    serde_json::from_str(s).unwrap_or_else(|e| panic!("{e}: {s}"))
}

fn completion(srv: &Server) -> String {
    let req =
        r#"{"model":"tiny-qwen3moe-f16","prompt":"Hello there","max_tokens":16,"temperature":0}"#;
    let (st, body) = http(&srv.api, "POST", "/v1/completions", Some(req));
    assert_eq!(st, 200, "{body}");
    json(&body)["choices"][0]["text"]
        .as_str()
        .unwrap()
        .to_string()
}

fn plan_status(srv: &Server) -> serde_json::Value {
    let (st, body) = http(&srv.admin, "GET", "/admin/plan", None);
    assert_eq!(st, 200, "{body}");
    json(&body)
}

fn run(engine: &str) {
    let dir =
        std::env::temp_dir().join(format!("tetraeder-replan-{engine}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let hw = dir.join("hw.toml");
    std::fs::write(&hw, hw_with_threads(2)).unwrap();
    let srv = spawn(engine, &hw);

    let st = plan_status(&srv);
    eprintln!(
        "{engine}: auto picked {} threads={} ubatch={}",
        st["description"], st["threads"], st["ubatch"]
    );
    assert_eq!(st["threads"], 2, "{st}");
    assert_eq!(st["switches"], 0);
    let before = completion(&srv);
    assert!(!before.is_empty());

    // the box lost a thread: the plan with two no longer fits the quota
    std::fs::write(&hw, hw_with_threads(1)).unwrap();
    let t0 = Instant::now();
    let (code, body) = http(
        &srv.admin,
        "POST",
        "/admin/replan?wait=1",
        Some(r#"{"trigger":"device_change"}"#),
    );
    let took = t0.elapsed();
    eprintln!("{engine}: switch (drain, load, health) took {took:.2?}: {body}");
    assert_eq!(code, 200, "{body}");
    assert_eq!(json(&body)["outcome"], "switched", "{body}");

    let st = plan_status(&srv);
    assert_eq!(st["threads"], 1, "{st}");
    assert_eq!(st["switches"], 1, "{st}");
    assert_eq!(st["last_outcome"], "switched");
    assert_eq!(
        completion(&srv),
        before,
        "greedy output changed with the thread count (see the token ids in the report)"
    );

    // a second switch right away is held back by the dwell
    let (code, body) = http(
        &srv.admin,
        "POST",
        "/admin/replan?wait=1",
        Some(r#"{"trigger":"device_change"}"#),
    );
    assert_eq!(code, 200, "{body}");
    let v = json(&body);
    assert_eq!(
        (v["outcome"].as_str(), v["reason"].as_str()),
        (Some("rejected"), Some("dwell"))
    );
    std::fs::remove_dir_all(&dir).ok();
}

// hardware: cpu
#[test]
fn moe_builder_switches_on_a_device_change() {
    run("moe");
}

// hardware: cpu
#[test]
fn libllama_switches_on_a_device_change() {
    run("llama");
}
