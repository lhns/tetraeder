//! End-to-end tests of the CLI against a fake `llama-bench` (no GPU, no model needed).

use serde_json::Value;
use std::path::Path;
use std::process::{Command, Output};

const FAKE: &str = env!("CARGO_BIN_EXE_fake-llama-bench");
const BENCH: &str = env!("CARGO_BIN_EXE_tetraeder-bench");

struct Fixture {
    dir: tempfile::TempDir,
}

impl Fixture {
    /// `a_args` / `b_args` are the flags of the two variants; `role` is the pair role.
    fn new(role: &str, a_args: &[&str], b_args: &[&str], extra: &str) -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let model_dir = dir.path().join("models").join("org__name");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("f.gguf"), b"").unwrap();
        let q = |args: &[&str]| {
            args.iter()
                .map(|a| format!("'{a}'"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let b_id = if role == "noise" { "va" } else { "vb" };
        let spec = format!(
            r#"
[spec]
name = "t"
repeats = 3
warmup = 1
bootstrap_resamples = 500
[workload]
batch = 512
ubatch = 128
prompt_len = 16
gen_len = 8
[model.m]
hf = "org/name:f.gguf"
[[engine]]
id = "fake"
kind = "llama-bench"
bin = '{FAKE}'
[[engine]]
id = "tet"
kind = "tetraeder"
bin = 'nonexistent'
runnable = false
[[variant]]
id = "va"
engine = "fake"
model = "m"
device = "cpu"
args = [{a}]
[[variant]]
id = "vb"
engine = "fake"
model = "m"
device = "cpu"
args = [{b}]
[[variant]]
id = "vt"
engine = "tet"
model = "m"
device = "cpu"
[[pair]]
id = "p"
a = "va"
b = "{b_id}"
role = "{role}"
{extra}
[[noise_band]]
class = "cpu"
device = "cpu"
band = 0.05
"#,
            a = q(a_args),
            b = q(b_args),
        );
        std::fs::write(dir.path().join("spec.toml"), spec).unwrap();
        Fixture { dir }
    }

    fn run(&self, extra_args: &[&str], envs: &[(&str, &str)]) -> Output {
        let mut c = Command::new(BENCH);
        c.arg("run")
            .arg(self.dir.path().join("spec.toml"))
            .arg("--out-dir")
            .arg(self.dir.path().join("out"))
            .args(extra_args)
            .env_remove("CONTAINER_ID")
            .env("TETRAEDER_MODELS_DIR", self.dir.path().join("models"));
        for (k, v) in envs {
            c.env(k, v);
        }
        c.output().unwrap()
    }

    fn results(&self) -> Value {
        let mut files: Vec<_> = std::fs::read_dir(self.dir.path().join("out"))
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| {
                p.file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with("t-"))
            })
            .collect();
        // Names carry the run time (plus a counter within one second), so the last is the latest.
        files.sort_by_key(|p| p.metadata().unwrap().modified().unwrap());
        serde_json::from_slice(&std::fs::read(files.pop().unwrap()).unwrap()).unwrap()
    }
}

fn code(o: &Output) -> i32 {
    o.status.code().unwrap()
}

fn text_of(o: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    )
}

#[test]
fn runs_are_interleaved_a_b_a_b_after_warmup() {
    let f = Fixture::new("info", &["--fake-tag", "A"], &["--fake-tag", "B"], "");
    let log = f.dir.path().join("calls.log");
    let o = f.run(&[], &[("FAKE_LOG", log.to_str().unwrap())]);
    assert_eq!(code(&o), 0, "{}", text_of(&o));
    let calls: Vec<String> = std::fs::read_to_string(&log)
        .unwrap()
        .lines()
        .map(String::from)
        .collect();
    assert_eq!(calls.join(""), "ABABBAAB"); // warmup AB, then AB, BA, AB (alternating order)
    let r = f.results();
    let samples = r["pairs"][0]["samples"].as_array().unwrap();
    assert_eq!(samples.len(), 6);
    let sides: String = samples
        .iter()
        .map(|s| s["side"].as_str().unwrap())
        .collect();
    assert_eq!(sides, "abbaab");
    let orders: Vec<u64> = samples
        .iter()
        .map(|s| s["order"].as_u64().unwrap())
        .collect();
    assert_eq!(orders, [0, 0, 1, 1, 0, 0]);
    // each sample carries the settings the engine reported (null for the fake)
    assert!(samples[0]["settings"].is_object());
}

#[test]
fn cpu_variants_run_hidden_from_the_gpu_with_explicit_threads() {
    let f = Fixture::new("info", &[], &[], "");
    let log = f.dir.path().join("env.log");
    let o = f.run(
        &[],
        &[
            ("FAKE_ENV_LOG", log.to_str().unwrap()),
            ("CUDA_VISIBLE_DEVICES", "0"),
        ],
    );
    assert_eq!(code(&o), 0, "{}", text_of(&o));
    let lines = std::fs::read_to_string(&log).unwrap();
    assert!(lines.lines().count() >= 8);
    assert!(
        lines
            .lines()
            .all(|l| l.starts_with("CUDA_VISIBLE_DEVICES= threads=")),
        "{lines}"
    );
    assert!(lines.lines().all(|l| !l.ends_with("threads=")), "{lines}");
    let r = f.results();
    let n = r["pairs"][0]["samples"][0]["threads"].as_u64().unwrap();
    assert!(n >= 1);
    assert!(
        lines.lines().all(|l| l.ends_with(&format!("threads={n}"))),
        "{lines}"
    );
}

#[test]
fn slower_candidate_beyond_band_is_a_regression_with_nonzero_exit() {
    let f = Fixture::new("gate", &["--fake-tg", "100"], &["--fake-tg", "80"], "");
    let o = f.run(&[], &[]);
    assert_eq!(code(&o), 1, "{}", text_of(&o));
    let r = f.results();
    let p = &r["pairs"][0];
    assert_eq!(p["status"], "regression");
    let ratio = p["metrics"]["tg"]["ratio"]["median"].as_f64().unwrap();
    assert!((ratio - 0.8).abs() < 1e-9);
    assert_eq!(r["exit_code"], 1);
    assert!(r["env"]["host_id"].as_str().is_some_and(|h| !h.is_empty()));
}

#[test]
fn candidate_within_band_passes_even_with_jitter() {
    let f = Fixture::new(
        "gate",
        &["--fake-tg", "100", "--fake-jitter", "0.005"],
        &["--fake-tg", "98", "--fake-jitter", "0.005"],
        "",
    );
    let o = f.run(&[], &[]);
    assert_eq!(code(&o), 0, "{}", text_of(&o));
    assert_eq!(f.results()["pairs"][0]["status"], "ok");
}

#[test]
fn faster_candidate_is_not_a_regression() {
    let f = Fixture::new("gate", &["--fake-tg", "100"], &["--fake-tg", "150"], "");
    assert_eq!(code(&f.run(&[], &[])), 0);
}

#[test]
fn noise_pair_reports_the_box_noise() {
    let f = Fixture::new(
        "noise",
        &["--fake-tg", "100", "--fake-jitter", "0.2"],
        &[],
        "",
    );
    let o = f.run(&[], &[]);
    let p = &f.results()["pairs"][0];
    // The jitter is random, so either verdict can come out; each has its own exit code (3: inconclusive).
    let want = match p["status"].as_str().unwrap() {
        "ok" => 0,
        "noisy" => 3,
        s => panic!("unexpected status {s}"),
    };
    assert_eq!(code(&o), want, "{}", text_of(&o));
    assert!(p["metrics"]["tg"]["a"]["rel_iqr"].as_f64().unwrap() > 0.0);
}

#[test]
fn engine_failure_is_an_error() {
    let f = Fixture::new("gate", &["--fake-tg", "100"], &["--fake-fail"], "");
    let o = f.run(&[], &[]);
    assert_eq!(code(&o), 2, "{}", text_of(&o));
    let p = &f.results()["pairs"][0];
    assert_eq!(p["status"], "error");
    assert!(p["reason"].as_str().unwrap().contains("simulated failure"));
}

#[test]
fn hung_engine_is_killed_by_the_timeout() {
    let f = Fixture::new("gate", &["--fake-sleep-ms", "60000"], &[], "");
    let spec = f.dir.path().join("spec.toml");
    let text = std::fs::read_to_string(&spec).unwrap().replace(
        "warmup = 1",
        "warmup = 1
timeout_s = 1",
    );
    std::fs::write(&spec, text).unwrap();
    let t0 = std::time::Instant::now();
    let o = f.run(&[], &[]);
    assert!(t0.elapsed().as_secs() < 20, "timeout did not fire");
    assert_eq!(code(&o), 2, "{}", text_of(&o));
    assert!(
        f.results()["pairs"][0]["reason"]
            .as_str()
            .unwrap()
            .contains("timed out")
    );
}

#[test]
fn unrunnable_engine_is_skipped_unless_strict() {
    let extra = r#"
[[pair]]
id = "gate-tet"
a = "va"
b = "vt"
role = "gate"
"#;
    let f = Fixture::new("info", &[], &[], extra);
    let o = f.run(&["--only", "gate-tet"], &[]);
    assert_eq!(code(&o), 0, "{}", text_of(&o));
    let p = &f.results()["pairs"][0];
    assert_eq!(p["status"], "skipped");
    assert!(p["reason"].as_str().unwrap().contains("not runnable yet"));
    assert_eq!(code(&f.run(&["--only", "gate-tet", "--strict"], &[])), 2);
}

#[test]
fn missing_binary_or_model_is_skipped_with_reason() {
    let f = Fixture::new("info", &[], &[], "");
    let o = f.run(&["--llama-bench", "/does/not/exist"], &[]);
    assert_eq!(code(&o), 0);
    assert!(
        f.results()["pairs"][0]["reason"]
            .as_str()
            .unwrap()
            .contains("not found")
    );
    let o = f.run(&["--models-dir", "/no/models"], &[]);
    assert_eq!(code(&o), 0);
    assert!(
        f.results()["pairs"][0]["reason"]
            .as_str()
            .unwrap()
            .contains("model file")
    );
}

/// A second pair `q` whose b side hangs: the run is killed while it measures `q`.
const HANGING_PAIR: &str = r#"
[[variant]]
id = "vhang"
engine = "fake"
model = "m"
device = "cpu"
args = ['--fake-sleep-ms', '30000']
[[pair]]
id = "q"
a = "va"
b = "vhang"
role = "info"
"#;

#[test]
fn results_are_written_after_every_pair() {
    let f = Fixture::new("info", &[], &[], HANGING_PAIR);
    let mut child = Command::new(BENCH)
        .arg("run")
        .arg(f.dir.path().join("spec.toml"))
        .arg("--out-dir")
        .arg(f.dir.path().join("out"))
        .env_remove("CONTAINER_ID")
        .env("TETRAEDER_MODELS_DIR", f.dir.path().join("models"))
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    // pair p finishes in well under a second; q then hangs on its b side
    let t0 = std::time::Instant::now();
    let partial = || -> Option<Value> {
        std::fs::read_dir(f.dir.path().join("out"))
            .ok()?
            .filter_map(|e| std::fs::read(e.ok()?.path()).ok())
            .filter_map(|b| serde_json::from_slice::<Value>(&b).ok())
            .find(|r| r["pairs"].as_array().is_some_and(|p| !p.is_empty()))
    };
    let done = loop {
        if let Some(r) = partial() {
            break r;
        }
        assert!(t0.elapsed().as_secs() < 20, "no partial results written");
        std::thread::sleep(std::time::Duration::from_millis(100));
    };
    child.kill().unwrap();
    child.wait().unwrap();
    let pairs = done["pairs"].as_array().unwrap();
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0]["id"], "p");
    assert_eq!(pairs[0]["samples"].as_array().unwrap().len(), 6);
    // the file on disk after the kill still parses and keeps the finished pair
    assert_eq!(f.results()["pairs"][0]["id"], "p");
}

#[test]
fn resume_runs_only_the_missing_repeats() {
    let f = Fixture::new("info", &["--fake-tag", "A"], &["--fake-tag", "B"], "");
    let log = f.dir.path().join("calls.log");
    let envs = [("FAKE_LOG", log.to_str().unwrap())];
    let o = f.run(&["--repeats", "1", "--resume"], &envs);
    assert_eq!(code(&o), 0, "{}", text_of(&o));
    assert_eq!(
        std::fs::read_to_string(&log).unwrap().replace('\n', ""),
        "ABAB"
    ); // warmup, rep 0
    std::fs::remove_file(&log).unwrap();
    // the spec's 3 repeats: reps 1 and 2 are missing, in their ABBA order (BA, AB)
    let o = f.run(&["--resume"], &envs);
    assert_eq!(code(&o), 0, "{}", text_of(&o));
    assert_eq!(
        std::fs::read_to_string(&log).unwrap().replace('\n', ""),
        "ABBAAB"
    );
    let r = f.results();
    let reps: Vec<u64> = r["pairs"][0]["samples"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["rep"].as_u64().unwrap())
        .collect();
    assert_eq!(reps, [1, 1, 2, 2]);
    std::fs::remove_file(&log).unwrap();
    // nothing left to run: no engine call, no pair in the new file
    let o = f.run(&["--resume"], &envs);
    assert_eq!(code(&o), 0, "{}", text_of(&o));
    assert!(!log.exists());
    assert!(f.results()["pairs"].as_array().unwrap().is_empty());
    // without --resume everything runs again
    let o = f.run(&["--repeats", "1"], &envs);
    assert_eq!(code(&o), 0, "{}", text_of(&o));
    assert_eq!(
        std::fs::read_to_string(&log).unwrap().replace('\n', ""),
        "ABAB"
    );
    std::fs::remove_file(&log).unwrap();
    // results of another spec text do not count
    let spec = f.dir.path().join("spec.toml");
    let text = std::fs::read_to_string(&spec).unwrap() + "\n# changed\n";
    std::fs::write(&spec, text).unwrap();
    let o = f.run(&["--repeats", "1", "--resume"], &envs);
    assert_eq!(code(&o), 0, "{}", text_of(&o));
    assert_eq!(
        std::fs::read_to_string(&log).unwrap().replace('\n', ""),
        "ABAB"
    );
}

#[test]
fn host_id_prefers_container_id() {
    let f = Fixture::new("info", &[], &[], "");
    assert_eq!(code(&f.run(&[], &[("CONTAINER_ID", "C.12345")])), 0);
    assert_eq!(f.results()["env"]["host_id"], "C.12345");
}

#[test]
fn check_and_golden_compare_subcommands() {
    let f = Fixture::new("info", &[], &[], "");
    let o = Command::new(BENCH)
        .arg("check")
        .arg(f.dir.path().join("spec.toml"))
        .output()
        .unwrap();
    assert_eq!(code(&o), 0);
    let g = r#"{"schema":1,"model":"m","engine":"e","backend":"cpu","top_k":1,"cases":[{"id":"c","prompt_tokens":[1],"tokens":[2],"top":[[{"id":2,"logit":1.0}]]}]}"#;
    let (a, b) = (f.dir.path().join("a.json"), f.dir.path().join("b.json"));
    std::fs::write(&a, g).unwrap();
    std::fs::write(&b, g.replace("\"logit\":1.0", "\"logit\":1.5")).unwrap();
    let cmp = |x: &Path, y: &Path| {
        Command::new(BENCH)
            .arg("golden-compare")
            .arg(x)
            .arg(y)
            .output()
            .unwrap()
    };
    assert_eq!(code(&cmp(&a, &a)), 0);
    assert_eq!(code(&cmp(&a, &b)), 1);
}

const FAKE_PARITY: &str = env!("CARGO_BIN_EXE_fake-parity-engine");

/// A fixture with a runnable Tetraeder engine (the fake llama-bench) and a gate pair llama vs tetraeder.
fn tetraeder_gate() -> Fixture {
    let extra = r#"
[[pair]]
id = "gate-tet"
a = "va"
b = "vt"
role = "gate"
"#;
    let f = Fixture::new("info", &[], &[], extra);
    let spec = f.dir.path().join("spec.toml");
    let text = std::fs::read_to_string(&spec).unwrap().replace(
        "bin = 'nonexistent'
runnable = false",
        &format!("bin = '{FAKE}'"),
    );
    std::fs::write(&spec, text).unwrap();
    f
}

fn write_parity(f: &Fixture, status: &str) -> std::path::PathBuf {
    let path = f.dir.path().join("parity.json");
    let check = format!(
        r#"{{"schema":1,"spec":"t","checks":[{{"model":"m","device":"cpu","check":"greedy-golden","status":"{status}","detail":"d"}}]}}"#
    );
    std::fs::write(&path, check).unwrap();
    path
}

#[test]
fn tetraeder_gate_without_parity_is_an_error() {
    let f = tetraeder_gate();
    let o = f.run(&["--only", "gate-tet"], &[]);
    assert_eq!(code(&o), 2, "{}", text_of(&o));
    let p = &f.results()["pairs"][0];
    assert_eq!(p["status"], "error");
    assert_eq!(p["quality"]["status"], "missing");
}

#[test]
fn tetraeder_gate_with_ok_parity_passes_and_records_it() {
    let f = tetraeder_gate();
    let par = write_parity(&f, "ok");
    let o = f.run(
        &["--only", "gate-tet", "--parity", par.to_str().unwrap()],
        &[],
    );
    assert_eq!(code(&o), 0, "{}", text_of(&o));
    let p = &f.results()["pairs"][0];
    assert_eq!(p["status"], "ok");
    assert_eq!(p["quality"]["status"], "ok");
    assert_eq!(p["quality"]["check"], "greedy-golden");
}

#[test]
fn tetraeder_gate_with_failed_parity_is_an_error() {
    let f = tetraeder_gate();
    let par = write_parity(&f, "failed");
    let o = f.run(
        &["--only", "gate-tet", "--parity", par.to_str().unwrap()],
        &[],
    );
    assert_eq!(code(&o), 2, "{}", text_of(&o));
    assert_eq!(f.results()["pairs"][0]["status"], "error");
}

#[test]
fn llama_only_gate_needs_no_quality() {
    let f = Fixture::new("gate", &["--fake-tg", "100"], &["--fake-tg", "100"], "");
    assert_eq!(code(&f.run(&[], &[])), 0);
    assert!(f.results()["pairs"][0]["quality"].is_null());
}

#[test]
fn parity_subcommand_compares_against_the_reference() {
    let f = tetraeder_gate();
    let run = |envs: &[(&str, &str)]| {
        let mut c = Command::new(BENCH);
        c.arg("parity")
            .arg(f.dir.path().join("spec.toml"))
            .arg("--out-dir")
            .arg(f.dir.path().join("par"))
            .arg("--llama-simple")
            .arg(FAKE_PARITY)
            .arg("--tetraeder")
            .arg(FAKE_PARITY)
            .env("TETRAEDER_MODELS_DIR", f.dir.path().join("models"));
        for (k, v) in envs {
            c.env(k, v);
        }
        c.output().unwrap()
    };
    let o = run(&[]);
    assert_eq!(code(&o), 0, "{}", text_of(&o));
    let file: Value =
        serde_json::from_slice(&std::fs::read(f.dir.path().join("par/parity.json")).unwrap())
            .unwrap();
    assert_eq!(file["checks"][0]["model"], "m");
    assert_eq!(file["checks"][0]["status"], "ok");
    let o = run(&[("FAKE_PARITY_DIFF", "1")]);
    assert_eq!(code(&o), 1, "{}", text_of(&o));
    assert!(text_of(&o).contains("differ"));
    // and the results of a run next to it pick the file up by default
    let o = f.run(
        &[
            "--only",
            "gate-tet",
            "--parity",
            f.dir.path().join("par/parity.json").to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(code(&o), 2, "{}", text_of(&o));
}

fn noise_file(dir: &Path, n: usize, band: f64) -> std::path::PathBuf {
    let sessions: Vec<Value> = (0..n)
        .map(|i| {
            serde_json::json!({
                "file": format!("r{i}.json"), "date": "2026-11-09", "host_id": format!("h{i}"),
                "gpu": "", "driver": "", "pair": "noise-cpu-dense", "threads": 2, "repeats": 3,
                "ratios": {"tg": 1.0}, "d": 0.0
            })
        })
        .collect();
    let f = serde_json::json!({
        "schema": 1, "class": "cpu", "device": "cpu", "updated": "2026-11-09", "quantile": 0.95,
        "margin": 1.25, "floor": 0.02, "ceil": 0.25, "n": n, "n_hosts": n, "d_p95": 0.0,
        "band": band, "sessions": sessions
    });
    std::fs::create_dir_all(dir).unwrap();
    let path = dir.join("cpu.json");
    std::fs::write(&path, f.to_string()).unwrap();
    path
}

#[test]
fn measured_band_replaces_the_provisional_one_and_is_recorded() {
    // ratio 0.93 is a regression at the spec's 0.05 band and fine at a measured 0.10
    let f = Fixture::new("gate", &["--fake-tg", "100"], &["--fake-tg", "93"], "");
    let nd = f.dir.path().join("noise");
    noise_file(&nd, 3, 0.10);
    let o = f.run(&["--noise-dir", nd.to_str().unwrap()], &[]);
    assert_eq!(code(&o), 0, "{}", text_of(&o));
    let p = &f.results()["pairs"][0];
    assert_eq!(p["band"], 0.10);
    assert_eq!(p["band_source"], "measured cpu n=3");
    assert!(text_of(&o).contains("measured cpu n=3"));
}

#[test]
fn too_few_sessions_or_no_file_fall_back_to_the_provisional_band() {
    let f = Fixture::new("gate", &["--fake-tg", "100"], &["--fake-tg", "93"], "");
    let nd = f.dir.path().join("noise");
    noise_file(&nd, 2, 0.10);
    let o = f.run(&["--noise-dir", nd.to_str().unwrap()], &[]);
    assert_eq!(code(&o), 1, "{}", text_of(&o));
    let p = &f.results()["pairs"][0];
    assert_eq!(p["band"], 0.05);
    assert_eq!(p["band_source"], "provisional cpu (measured n=2 < 3)");

    let missing = f.dir.path().join("none");
    let o = f.run(&["--noise-dir", missing.to_str().unwrap()], &[]);
    assert_eq!(code(&o), 1, "{}", text_of(&o));
    assert_eq!(f.results()["pairs"][0]["band_source"], "provisional cpu");
}

#[test]
fn a_corrupt_measured_file_is_an_error_not_a_silent_fallback() {
    let f = Fixture::new("gate", &["--fake-tg", "100"], &["--fake-tg", "100"], "");
    let nd = f.dir.path().join("noise");
    std::fs::create_dir_all(&nd).unwrap();
    std::fs::write(nd.join("cpu.json"), "{ nope").unwrap();
    let o = f.run(&["--noise-dir", nd.to_str().unwrap()], &[]);
    assert_eq!(code(&o), 2, "{}", text_of(&o));
    assert_eq!(f.results()["pairs"][0]["status"], "error");
}

#[test]
fn noise_update_builds_a_class_file_from_aa_pairs_and_is_idempotent() {
    let f = Fixture::new(
        "noise",
        &["--fake-tg", "100", "--fake-jitter", "0.001"],
        &[],
        "",
    );
    for _ in 0..3 {
        assert_eq!(code(&f.run(&[], &[])), 0);
    }
    let nd = f.dir.path().join("noise");
    let update = || {
        let o = Command::new(BENCH)
            .args(["noise", "update", "--results"])
            .arg(f.dir.path().join("out"))
            .arg("--out")
            .arg(&nd)
            .output()
            .unwrap();
        assert_eq!(code(&o), 0, "{}", text_of(&o));
    };
    update();
    update();
    let v: Value = serde_json::from_slice(&std::fs::read(nd.join("cpu.json")).unwrap()).unwrap();
    assert_eq!(v["class"], "cpu");
    assert_eq!(v["n"], 3);
    assert!(v["band"].as_f64().unwrap() >= 0.02);
    // and a run now picks it up
    let o = f.run(&["--noise-dir", nd.to_str().unwrap()], &[]);
    assert_eq!(code(&o), 0, "{}", text_of(&o));
    assert_eq!(f.results()["pairs"][0]["band_source"], "measured cpu n=3");
}
