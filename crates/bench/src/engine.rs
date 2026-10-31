//! Invoking engine binaries and parsing their output.

use crate::spec::{Device, Variant, Workload, has_flag};
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

/// Workload defaults, each replaced by the variant's own flag: `-p`, `-n` (a second one would add a test to the
/// sweep) and `-r` (1: the harness does the repeating so that variants can be interleaved).
fn common_args(
    w: &Workload,
    v: &Variant,
    model: &Path,
    threads: u32,
    ubatch_flag: &str,
) -> Vec<String> {
    let mut a: Vec<String> = vec!["-m".into(), model.display().to_string()];
    if !has_flag(&v.args, &["-p", "--n-prompt"]) {
        a.extend(["-p".into(), w.prompt_len.to_string()]);
    }
    if !has_flag(&v.args, &["-n", "--n-gen"]) {
        a.extend(["-n".into(), w.gen_len.to_string()]);
    }
    a.extend([
        "-b".into(),
        w.batch.to_string(),
        ubatch_flag.into(),
        w.ubatch.to_string(),
        "-t".into(),
        threads.to_string(),
    ]);
    if !has_flag(&v.args, &["-r", "--repetitions", "--reps"]) {
        a.extend(["-r".into(), "1".into()]);
    }
    a
}

/// One llama-bench invocation with an explicit thread count.
pub fn llama_bench_args(w: &Workload, v: &Variant, model: &Path, threads: u32) -> Vec<String> {
    let mut a = common_args(w, v, model, threads, "-ub");
    a.extend(["-o".into(), "json".into()]);
    a.extend(v.args.iter().cloned());
    a
}

/// `tetraeder bench` takes the same workload with its own flag names and prints llama-bench rows.
pub fn tetraeder_bench_args(w: &Workload, v: &Variant, model: &Path, threads: u32) -> Vec<String> {
    let mut a = vec!["bench".to_string()];
    a.extend(common_args(w, v, model, threads, "--ubatch"));
    a.extend(v.args.iter().cloned());
    a
}

/// Environment for a variant's child process. A CPU variant must not see the GPU: the CUDA build offloads big
/// batched matmuls to it even with `-ngl 0`. ggml reads `CUDA_VISIBLE_DEVICES` and `GGML_VK_VISIBLE_DEVICES`
/// (an empty value hides all devices).
pub fn device_env(device: Device) -> Vec<(&'static str, &'static str)> {
    match device {
        Device::Cpu => vec![
            ("CUDA_VISIBLE_DEVICES", ""),
            ("GGML_VK_VISIBLE_DEVICES", ""),
        ],
        Device::Cuda => Vec::new(),
    }
}

/// Run one engine invocation; kill it after `timeout` so a hung engine cannot block the night.
pub fn run_llama_bench(
    bin: &Path,
    args: &[String],
    env: &[(&str, &str)],
    timeout: Duration,
) -> Result<Measured> {
    let t0 = Instant::now();
    let mut child = Command::new(bin)
        .args(args)
        .envs(env.iter().copied())
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
    use crate::spec::threads_for;

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

    fn w(threads: Option<u32>) -> Workload {
        Workload {
            batch: 2048,
            ubatch: 512,
            prompt_len: 512,
            gen_len: 64,
            threads,
        }
    }

    fn v(device: Device, args: &[&str]) -> Variant {
        Variant {
            id: "v".into(),
            engine: "e".into(),
            model: "m".into(),
            device,
            threads: None,
            args: args.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn pair(a: &[String], k: &str, val: &str) -> bool {
        a.windows(2).any(|p| p == [k, val])
    }

    #[test]
    fn args_carry_workload_and_variant_flags() {
        let v = v(Device::Cuda, &["-ngl", "99", "-ot", "exps=CPU"]);
        let a = llama_bench_args(&w(None), &v, Path::new("m.gguf"), 4);
        assert_eq!(&a[..4], ["-m", "m.gguf", "-p", "512"]);
        assert!(pair(&a, "-t", "4"));
        assert!(pair(&a, "-r", "1"));
        assert_eq!(a.iter().filter(|x| *x == "-t").count(), 1);
        assert_eq!(&a[a.len() - 4..], ["-ngl", "99", "-ot", "exps=CPU"]);
    }

    #[test]
    fn variant_flags_replace_prompt_and_reps_defaults() {
        let v = v(Device::Cuda, &["-ngl", "99", "-p", "2048", "-r", "3"]);
        for a in [
            llama_bench_args(&w(None), &v, Path::new("m.gguf"), 4),
            tetraeder_bench_args(&w(None), &v, Path::new("m.gguf"), 4),
        ] {
            assert_eq!(a.iter().filter(|x| *x == "-p").count(), 1);
            assert_eq!(a.iter().filter(|x| *x == "-r").count(), 1);
            assert!(pair(&a, "-p", "2048"));
            assert!(pair(&a, "-r", "3"));
        }
    }

    #[test]
    fn tetraeder_args_use_its_subcommand_and_flags() {
        let v = v(Device::Cpu, &["--ngl", "0"]);
        let a = tetraeder_bench_args(&w(None), &v, Path::new("m.gguf"), 8);
        assert_eq!(&a[..3], ["bench", "-m", "m.gguf"]);
        assert!(pair(&a, "--ubatch", "512"));
        assert!(pair(&a, "-t", "8"));
        assert!(pair(&a, "-r", "1"));
        assert_eq!(&a[a.len() - 2..], ["--ngl", "0"]);
    }

    #[test]
    fn cpu_variants_hide_the_gpu_and_cuda_variants_do_not() {
        let cpu = device_env(Device::Cpu);
        assert!(cpu.contains(&("CUDA_VISIBLE_DEVICES", "")));
        assert!(cpu.contains(&("GGML_VK_VISIBLE_DEVICES", "")));
        assert!(device_env(Device::Cuda).is_empty());
    }

    #[test]
    fn thread_count_precedence() {
        let mut var = v(Device::Cpu, &[]);
        assert_eq!(threads_for(&w(None), &var, 11), 11);
        assert_eq!(threads_for(&w(Some(8)), &var, 11), 8);
        var.threads = Some(3);
        assert_eq!(threads_for(&w(Some(8)), &var, 11), 3);
    }
}
