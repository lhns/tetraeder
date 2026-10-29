//! Running a spec: paired, interleaved repeats, statistics, noise-band verdicts, results JSON.

use crate::engine::{self, Measured};
use crate::env::{self, EnvInfo, GpuSnap};
use crate::spec::{Device, EngineKind, PageCache, Pair, Role, Spec, Variant};
use crate::stats::{self, Ratio, Summary};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub const RESULTS_SCHEMA: u32 = 1;

#[derive(Debug, Clone, Default)]
pub struct Options {
    /// Overrides the path of every `llama-bench` engine.
    pub llama_bench: Option<PathBuf>,
    pub models_dir: PathBuf,
    pub repeats: Option<usize>,
    /// Run only these pair ids (empty = all).
    pub only: Vec<String>,
    /// Treat a skipped pair as an error.
    pub strict: bool,
    pub quiet: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Ok,
    /// Candidate slower than the noise band allows (gate pairs only).
    Regression,
    /// A/A pair outside its own band: the box is too noisy to trust gates from this session.
    Noisy,
    Skipped,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sample {
    pub rep: usize,
    /// "a" or "b".
    pub side: String,
    pub variant: String,
    pub pp_ts: Option<f64>,
    pub tg_ts: Option<f64>,
    pub wall_s: f64,
    pub build_commit: Option<String>,
    /// GPU state right after the sample (clocks, throttle reasons); empty without a GPU.
    pub gpu: Vec<GpuSnap>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricResult {
    pub a: Summary,
    pub b: Summary,
    /// b / a, paired by repeat.
    pub ratio: Ratio,
    /// The whole confidence interval lies below 1.
    pub significant_drop: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairResult {
    pub id: String,
    pub role: Role,
    pub a: String,
    pub b: String,
    pub device: Device,
    pub status: Status,
    pub reason: String,
    pub noise_class: Option<String>,
    pub band: Option<f64>,
    pub metrics: BTreeMap<String, MetricResult>,
    pub samples: Vec<Sample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Results {
    pub schema: u32,
    pub spec: String,
    pub spec_fnv1a: String,
    pub pins: BTreeMap<String, String>,
    pub repeats: usize,
    pub warmup: usize,
    pub page_cache: PageCache,
    /// Whether `cold` could actually be applied (root on Linux).
    pub page_cache_applied: bool,
    pub env: EnvInfo,
    pub pairs: Vec<PairResult>,
    pub exit_code: i32,
}

/// Exit code: 2 on any error (or skip under `--strict`), else 1 on any regression, else 0.
pub fn exit_code(pairs: &[PairResult], strict: bool) -> i32 {
    let any = |f: &dyn Fn(Status) -> bool| pairs.iter().any(|p| f(p.status));
    if any(&|s| s == Status::Error || (strict && s == Status::Skipped)) {
        2
    } else if any(&|s| s == Status::Regression) {
        1
    } else {
        0
    }
}

/// Verdict for a pair from its metrics. The gate compares the median paired ratio with `1 - band`.
pub fn verdict(
    role: Role,
    band: f64,
    metrics: &BTreeMap<String, MetricResult>,
) -> (Status, String) {
    let mut bad = Vec::new();
    for (name, m) in metrics {
        let r = m.ratio.median;
        match role {
            Role::Gate if r < 1.0 - band => {
                bad.push(format!("{name} ratio {r:.3} < {:.3}", 1.0 - band))
            }
            Role::Noise if (r - 1.0).abs() > band => {
                bad.push(format!("{name} A/A ratio {r:.3} outside +-{band}"))
            }
            _ => {}
        }
    }
    match (role, bad.is_empty()) {
        (_, true) => (Status::Ok, String::new()),
        (Role::Gate, false) => (Status::Regression, bad.join("; ")),
        (_, false) => (Status::Noisy, bad.join("; ")),
    }
}

fn fnv1a(text: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in text.bytes() {
        h = (h ^ u64::from(b)).wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

struct Resolved<'a> {
    variant: &'a Variant,
    bin: PathBuf,
    model: PathBuf,
    args: Vec<String>,
}

fn skip_reason<'a>(
    spec: &Spec,
    v: &'a Variant,
    opts: &Options,
    has_gpu: bool,
) -> Result<Result<Resolved<'a>, String>> {
    let engine = spec.engine(&v.engine).context("engine")?;
    if !engine.runnable {
        return Ok(Err(format!("engine `{}` is not runnable yet", engine.id)));
    }
    if engine.kind != EngineKind::LlamaBench {
        return Ok(Err(format!(
            "no harness adapter for engine kind of `{}`",
            engine.id
        )));
    }
    if v.device == Device::Cuda && !has_gpu {
        return Ok(Err("needs a CUDA GPU (nvidia-smi found none)".into()));
    }
    let configured = match &opts.llama_bench {
        Some(p) => p.display().to_string(),
        None => engine
            .bin_env
            .as_ref()
            .and_then(|k| std::env::var(k).ok())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| engine.bin.clone()),
    };
    let Some(bin) = engine::resolve_bin(&configured) else {
        return Ok(Err(format!("engine binary `{configured}` not found")));
    };
    let model = spec
        .model
        .get(&v.model)
        .context("model")?
        .path(&opts.models_dir)?;
    if !model.is_file() {
        return Ok(Err(format!("model file {} not found", model.display())));
    }
    let args = engine::llama_bench_args(&spec.workload, v, &model);
    Ok(Ok(Resolved {
        variant: v,
        bin,
        model,
        args,
    }))
}

fn drop_caches() -> bool {
    cfg!(target_os = "linux")
        && std::process::Command::new("sync").status().is_ok()
        && std::fs::write("/proc/sys/vm/drop_caches", "3").is_ok()
}

fn log(opts: &Options, msg: &str) {
    if !opts.quiet {
        let (_, _, _, h, m, s) = env::utc_parts(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        );
        eprintln!("[{h:02}:{m:02}:{s:02}Z] {msg}");
    }
}

fn run_one(r: &Resolved<'_>, timeout: Duration) -> Result<Measured> {
    engine::run_llama_bench(&r.bin, &r.args, timeout)
        .with_context(|| format!("variant `{}` ({})", r.variant.id, r.model.display()))
}

fn base_result(p: &Pair, device: Device) -> PairResult {
    PairResult {
        id: p.id.clone(),
        role: p.role,
        a: p.a.clone(),
        b: p.b.clone(),
        device,
        status: Status::Ok,
        reason: String::new(),
        noise_class: None,
        band: None,
        metrics: BTreeMap::new(),
        samples: Vec::new(),
    }
}

fn run_pair(
    spec: &Spec,
    p: &Pair,
    opts: &Options,
    env: &EnvInfo,
    cold_ok: bool,
) -> Result<PairResult> {
    let (va, vb) = (
        spec.variant(&p.a).context("variant a")?,
        spec.variant(&p.b).context("variant b")?,
    );
    let mut res = base_result(p, va.device);
    let gpu_name = env.gpus.first().map(|g| g.name.as_str());
    let band = if va.device == Device::Cuda {
        spec.band_for(va.device, gpu_name)
    } else {
        spec.band_for(va.device, None)
    };
    res.noise_class = band.map(|b| b.0.to_string());
    res.band = band.map(|b| b.1);

    let has_gpu = !env.gpus.is_empty();
    let (ra, rb) = (
        skip_reason(spec, va, opts, has_gpu)?,
        skip_reason(spec, vb, opts, has_gpu)?,
    );
    let (ra, rb) = match (ra, rb) {
        (Ok(a), Ok(b)) => (a, b),
        (a, b) => {
            res.status = Status::Skipped;
            res.reason = [a.err(), b.err()]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
                .join("; ");
            return Ok(res);
        }
    };
    let Some(band) = res.band else {
        res.status = Status::Error;
        res.reason = format!("no noise band for device {:?} in the spec", va.device);
        return Ok(res);
    };

    let repeats = opts.repeats.unwrap_or(spec.spec.repeats);
    let timeout = Duration::from_secs(spec.spec.timeout_s);
    let cold = spec.spec.page_cache == PageCache::Cold && cold_ok;
    let measured = (|| -> Result<()> {
        // Warmup loads the model into the page cache and settles clocks; its numbers are discarded.
        for _ in 0..spec.spec.warmup {
            for r in [&ra, &rb] {
                run_one(r, timeout)?;
            }
        }
        // Strictly interleaved: A, B, A, B, ... so drift hits both sides alike.
        for rep in 0..repeats {
            for (side, r) in [("a", &ra), ("b", &rb)] {
                if cold {
                    drop_caches();
                }
                let m = run_one(r, timeout)?;
                log(
                    opts,
                    &format!(
                        "pair {} rep {}/{repeats} {side} {}: pp {:?} tg {:?} t/s",
                        p.id,
                        rep + 1,
                        r.variant.id,
                        m.pp_ts,
                        m.tg_ts
                    ),
                );
                res.samples.push(Sample {
                    rep,
                    side: side.into(),
                    variant: r.variant.id.clone(),
                    pp_ts: m.pp_ts,
                    tg_ts: m.tg_ts,
                    wall_s: m.wall_s,
                    build_commit: m.build_commit,
                    gpu: env::gpu_snapshot(),
                });
            }
        }
        Ok(())
    })();
    if let Err(e) = measured {
        res.status = Status::Error;
        res.reason = format!("{e:#}");
        return Ok(res);
    }

    for (name, pick) in [
        ("pp", (|s: &Sample| s.pp_ts) as fn(&Sample) -> Option<f64>),
        ("tg", |s| s.tg_ts),
    ] {
        let series = |side: &str| -> Vec<f64> {
            res.samples
                .iter()
                .filter(|s| s.side == side)
                .filter_map(pick)
                .collect()
        };
        let (a, b) = (series("a"), series("b"));
        if a.is_empty() || a.len() != b.len() {
            continue;
        }
        let seed = spec.spec.seed ^ stats::Rng::new(fnv_seed(&p.id)).next_u64();
        if let Some(ratio) = stats::paired_ratio(
            &a,
            &b,
            spec.spec.bootstrap_resamples,
            spec.spec.confidence,
            seed,
        ) {
            res.metrics.insert(
                name.into(),
                MetricResult {
                    a: stats::summarize(&a),
                    b: stats::summarize(&b),
                    significant_drop: ratio.ci_hi < 1.0,
                    ratio,
                },
            );
        }
    }
    (res.status, res.reason) = verdict(p.role, band, &res.metrics);
    Ok(res)
}

fn fnv_seed(s: &str) -> u64 {
    u64::from_str_radix(&fnv1a(s), 16).unwrap_or(0)
}

/// Run all (selected) pairs of a spec and build the results.
pub fn run_spec(spec: &Spec, spec_text: &str, opts: &Options) -> Result<Results> {
    let env = env::capture();
    let cold_ok = spec.spec.page_cache == PageCache::Cold && drop_caches();
    let mut pairs = Vec::new();
    for p in spec
        .pair
        .iter()
        .filter(|p| opts.only.is_empty() || opts.only.contains(&p.id))
    {
        log(
            opts,
            &format!("pair {} ({:?}): {} vs {}", p.id, p.role, p.a, p.b),
        );
        let r = run_pair(spec, p, opts, &env, cold_ok)?;
        log(
            opts,
            &format!("pair {} -> {:?} {}", p.id, r.status, r.reason),
        );
        pairs.push(r);
    }
    let exit_code = exit_code(&pairs, opts.strict);
    Ok(Results {
        schema: RESULTS_SCHEMA,
        spec: spec.spec.name.clone(),
        spec_fnv1a: fnv1a(spec_text),
        pins: spec.pins.clone(),
        repeats: opts.repeats.unwrap_or(spec.spec.repeats),
        warmup: spec.spec.warmup,
        page_cache: spec.spec.page_cache,
        page_cache_applied: cold_ok,
        env,
        pairs,
        exit_code,
    })
}

pub fn write_results(results: &Results, out_dir: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(out_dir).with_context(|| format!("create {}", out_dir.display()))?;
    let path = out_dir.join(format!("{}.json", results.spec));
    std::fs::write(&path, serde_json::to_string_pretty(results)? + "\n")
        .with_context(|| format!("write {}", path.display()))?;
    Ok(path)
}

/// Human-readable summary, one line per pair and metric.
pub fn summary_text(r: &Results) -> String {
    let mut out = format!(
        "spec {} on {} ({} repeats)\n",
        r.spec, r.env.host_id, r.repeats
    );
    for p in &r.pairs {
        out += &format!("{:<28} {:?}", p.id, p.status);
        if !p.reason.is_empty() {
            out += &format!("  {}", p.reason);
        }
        out.push('\n');
        for (name, m) in &p.metrics {
            out += &format!(
                "    {name}: a {:.2} (rel IQR {:.1}%)  b {:.2} (rel IQR {:.1}%)  ratio {:.3} [{:.3}, {:.3}]  band -{:.0}%\n",
                m.a.median,
                m.a.rel_iqr * 100.0,
                m.b.median,
                m.b.rel_iqr * 100.0,
                m.ratio.median,
                m.ratio.ci_lo,
                m.ratio.ci_hi,
                p.band.unwrap_or(0.0) * 100.0
            );
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metric(ratio: f64) -> MetricResult {
        let s = stats::summarize(&[1.0, 2.0, 3.0]);
        MetricResult {
            a: s.clone(),
            b: s,
            ratio: Ratio {
                n: 3,
                median: ratio,
                ci_lo: ratio,
                ci_hi: ratio,
                confidence: 0.95,
            },
            significant_drop: false,
        }
    }

    fn m(ratio: f64) -> BTreeMap<String, MetricResult> {
        BTreeMap::from([("tg".to_string(), metric(ratio))])
    }

    #[test]
    fn gate_fails_only_beyond_the_band() {
        assert_eq!(verdict(Role::Gate, 0.05, &m(0.96)).0, Status::Ok);
        assert_eq!(verdict(Role::Gate, 0.05, &m(1.3)).0, Status::Ok);
        assert_eq!(verdict(Role::Gate, 0.05, &m(0.94)).0, Status::Regression);
    }

    #[test]
    fn noise_pair_flags_a_noisy_box_but_never_a_regression() {
        assert_eq!(verdict(Role::Noise, 0.05, &m(1.02)).0, Status::Ok);
        assert_eq!(verdict(Role::Noise, 0.05, &m(0.9)).0, Status::Noisy);
        assert_eq!(verdict(Role::Info, 0.05, &m(0.5)).0, Status::Ok);
    }

    #[test]
    fn exit_codes() {
        let mk = |s| PairResult {
            status: s,
            ..base_result(
                &Pair {
                    id: "x".into(),
                    a: "a".into(),
                    b: "b".into(),
                    role: Role::Gate,
                },
                Device::Cpu,
            )
        };
        assert_eq!(
            exit_code(
                &[mk(Status::Ok), mk(Status::Skipped), mk(Status::Noisy)],
                false
            ),
            0
        );
        assert_eq!(exit_code(&[mk(Status::Ok), mk(Status::Skipped)], true), 2);
        assert_eq!(
            exit_code(&[mk(Status::Regression), mk(Status::Ok)], false),
            1
        );
        assert_eq!(
            exit_code(&[mk(Status::Regression), mk(Status::Error)], false),
            2
        );
    }
}
