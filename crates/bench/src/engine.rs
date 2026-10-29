//! Invoking engine binaries and parsing their output.

use crate::spec::{Variant, Workload};
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Result of one engine invocation, in tokens per second.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Measured {
    /// Prompt processing (`-p`).
    pub pp_ts: Option<f64>,
    /// Token generation (`-n`).
    pub tg_ts: Option<f64>,
    /// Engine build commit as reported by the engine, if any.
    pub build_commit: Option<String>,
    pub wall_s: f64,
}

#[derive(Deserialize)]
struct Row {
    #[serde(default)]
    n_prompt: u64,
    #[serde(default)]
    n_gen: u64,
    avg_ts: f64,
    #[serde(default)]
    build_commit: Option<String>,
}

/// Parse `llama-bench -o json` output: an array with one row per test (pp and tg rows).
/// Tolerates log lines before the array (even ones starting with `[`, like `[info] ...`): the
/// array is the first line starting with `[` that parses.
pub fn parse_llama_bench(stdout: &str) -> Result<Measured> {
    let mut starts = Vec::new();
    let mut off = 0;
    for line in stdout.split_inclusive('\n') {
        if line.trim_start().starts_with('[') {
            starts.push(off + line.len() - line.trim_start().len());
        }
        off += line.len();
    }
    let mut last_err = None;
    let mut parsed = None;
    for &start in &starts {
        let mut de = serde_json::Deserializer::from_str(&stdout[start..]);
        match Vec::<Row>::deserialize(&mut de) {
            Ok(rows) => {
                parsed = Some(rows);
                break;
            }
            Err(e) => last_err = Some(e),
        }
    }
    let rows = match (parsed, last_err) {
        (Some(rows), _) => rows,
        (None, Some(e)) => return Err(e).context("llama-bench JSON"),
        (None, None) => anyhow::bail!("no JSON array in llama-bench output"),
    };
    let mut m = Measured::default();
    for r in rows {
        if r.n_prompt > 0 && r.n_gen == 0 {
            m.pp_ts = Some(r.avg_ts);
        } else if r.n_gen > 0 && r.n_prompt == 0 {
            m.tg_ts = Some(r.avg_ts);
        }
        m.build_commit = m.build_commit.or(r.build_commit);
    }
    if m.pp_ts.is_none() && m.tg_ts.is_none() {
        bail!("llama-bench output has no pp or tg row");
    }
    Ok(m)
}

/// One `-r 1` invocation: the harness does the repeating so that variants can be interleaved.
pub fn llama_bench_args(w: &Workload, v: &Variant, model: &Path) -> Vec<String> {
    let mut a: Vec<String> = vec![
        "-m".into(),
        model.display().to_string(),
        "-p".into(),
        w.prompt_len.to_string(),
        "-n".into(),
        w.gen_len.to_string(),
        "-b".into(),
        w.batch.to_string(),
        "-ub".into(),
        w.ubatch.to_string(),
        "-r".into(),
        "1".into(),
        "-o".into(),
        "json".into(),
    ];
    // A second -t would add a test to the sweep instead of replacing the first.
    let own_threads = v.args.iter().any(|x| x == "-t" || x == "--threads");
    if let (Some(t), false) = (w.threads, own_threads) {
        a.extend(["-t".into(), t.to_string()]);
    }
    a.extend(v.args.iter().cloned());
    a
}

/// Run one engine invocation; kill it after `timeout` so a hung engine cannot block the night.
pub fn run_llama_bench(bin: &Path, args: &[String], timeout: Duration) -> Result<Measured> {
    let t0 = Instant::now();
    let mut child = Command::new(bin)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("spawn {}", bin.display()))?;
    // Drain both pipes on threads so a chatty engine cannot fill them and stall.
    let drain = |mut pipe: Box<dyn Read + Send>| {
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = pipe.read_to_end(&mut buf);
            String::from_utf8_lossy(&buf).into_owned()
        })
    };
    let out_h = drain(Box::new(child.stdout.take().context("stdout")?));
    let err_h = drain(Box::new(child.stderr.take().context("stderr")?));
    let status = loop {
        if let Some(st) = child.try_wait()? {
            break st;
        }
        if t0.elapsed() > timeout {
            let _ = child.kill();
            let _ = child.wait();
            bail!(
                "{} timed out after {} s and was killed",
                bin.display(),
                timeout.as_secs()
            );
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let wall_s = t0.elapsed().as_secs_f64();
    let (stdout, stderr) = (
        out_h.join().unwrap_or_default(),
        err_h.join().unwrap_or_default(),
    );
    if !status.success() {
        let tail: Vec<&str> = stderr.lines().rev().take(5).collect();
        bail!(
            "{} exited with {}: {}",
            bin.display(),
            status,
            tail.into_iter().rev().collect::<Vec<_>>().join(" | ")
        );
    }
    let mut m = parse_llama_bench(&stdout)?;
    m.wall_s = wall_s;
    Ok(m)
}

/// Find an executable: a path with separators is checked directly (also with `.exe`), a bare name is searched on PATH.
pub fn resolve_bin(bin: &str) -> Option<PathBuf> {
    let direct = |p: PathBuf| -> Option<PathBuf> {
        let exe = PathBuf::from(format!("{}.exe", p.display()));
        if p.is_file() {
            Some(p)
        } else {
            (cfg!(windows) && exe.is_file()).then_some(exe)
        }
    };
    if bin.contains('/') || bin.contains('\\') {
        return direct(PathBuf::from(bin));
    }
    std::env::split_paths(&std::env::var_os("PATH")?).find_map(|d| direct(d.join(bin)))
}

#[cfg(test)]
mod tests {
    use super::*;

    const REAL: &str = r#"[
  {"build_commit":"2e0d86be","build_number":1,"cpu_info":"x","gpu_info":"","backends":"CPU","model_filename":"m.gguf","n_batch":512,"n_ubatch":128,"n_threads":4,"n_prompt":16,"n_gen":0,"test_time":"t","avg_ns":1000,"stddev_ns":0,"avg_ts":160.5,"stddev_ts":0.0,"samples_ns":[1000],"samples_ts":[160.5]},
  {"build_commit":"2e0d86be","n_prompt":0,"n_gen":8,"avg_ts":42.25,"stddev_ts":0.0}
]"#;

    #[test]
    fn parses_pp_and_tg_rows() {
        let m = parse_llama_bench(&format!("load_backend: loaded CPU\n{REAL}\n")).unwrap();
        assert_eq!(m.pp_ts, Some(160.5));
        assert_eq!(m.tg_ts, Some(42.25));
        assert_eq!(m.build_commit.as_deref(), Some("2e0d86be"));
    }

    #[test]
    fn skips_bracketed_log_lines_before_the_array() {
        let log = format!("[info] loading model\n[warn] x: [1, 2\n{REAL}\n");
        let m = parse_llama_bench(&log).unwrap();
        assert_eq!(m.tg_ts, Some(42.25));
    }

    #[test]
    fn rejects_garbage_and_empty() {
        assert!(parse_llama_bench("no json").is_err());
        assert!(parse_llama_bench("[]").is_err());
        assert!(parse_llama_bench(r#"[{"n_prompt":4,"n_gen":4,"avg_ts":1.0}]"#).is_err());
    }

    #[test]
    fn args_carry_workload_and_variant_flags() {
        let w = Workload {
            batch: 2048,
            ubatch: 512,
            prompt_len: 512,
            gen_len: 128,
            threads: Some(4),
        };
        let v = Variant {
            id: "v".into(),
            engine: "e".into(),
            model: "m".into(),
            device: crate::spec::Device::Cuda,
            args: vec!["-ngl".into(), "99".into(), "-ot".into(), "exps=CPU".into()],
        };
        let a = llama_bench_args(&w, &v, Path::new("m.gguf"));
        assert_eq!(&a[..4], ["-m", "m.gguf", "-p", "512"]);
        assert!(a.windows(2).any(|p| p == ["-t", "4"]));
        let mut own = v.clone();
        own.args = vec!["-t".into(), "2".into()];
        let a2 = llama_bench_args(&w, &own, Path::new("m.gguf"));
        assert_eq!(a2.iter().filter(|x| *x == "-t").count(), 1);
        assert_eq!(&a[a.len() - 4..], ["-ngl", "99", "-ot", "exps=CPU"]);
    }
}
