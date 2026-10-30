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
    assert_eq!(calls.join(""), "ABABABAB"); // 1 warmup + 3 repeats
    let r = f.results();
    let samples = r["pairs"][0]["samples"].as_array().unwrap();
    assert_eq!(samples.len(), 6);
    let sides: String = samples
        .iter()
        .map(|s| s["side"].as_str().unwrap())
        .collect();
    assert_eq!(sides, "ababab");
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
