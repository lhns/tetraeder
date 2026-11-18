//! `tetraeder bench`: llama-bench's pp and tg tests through the Tetraeder engine path, printed
//! as llama-bench's JSON rows so `tetraeder-bench` parses both engines alike (ADR 0026).

use crate::engine_args::ModelArgs;
use clap::{Args, ValueEnum};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tetraeder_engine::{Engine, FlashAttn};
use tetraeder_ggml_sys::ggml::Backend;
use tetraeder_model::{
    BackendChoice, DenseOptions, LoadOptions, ModelError, Qwen3, Qwen3Moe, SeqEval,
};
use tetraeder_telemetry::{Telemetry, TelemetryConfig};

#[derive(Args)]
pub struct BenchArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// Prompt tokens of the pp test (0 = skip).
    #[arg(short = 'p', long, default_value_t = 512)]
    n_prompt: u32,
    /// Generated tokens of the tg test (0 = skip).
    #[arg(short = 'n', long, default_value_t = 128)]
    n_gen: u32,
    /// Repetitions per test.
    #[arg(short = 'r', long, alias = "repetitions", default_value_t = 5)]
    reps: u32,
    /// Output format; only json.
    #[arg(short = 'o', long, default_value = "json")]
    output: String,
    /// Step telemetry: on (production mode, no exporter unless OTEL_EXPORTER_OTLP_ENDPOINT) or off.
    #[arg(long, value_enum, default_value_t = Switch::On)]
    telemetry: Switch,
    /// `llama`: libllama through the engine. `dense`, `moe`: the own Qwen3 builders (ADR 0066,
    /// 0030), on the executor when `--plan` is given (the dense one also on a GPU).
    #[arg(long, value_enum, default_value_t = BenchEngine::Llama)]
    engine: BenchEngine,
    /// Plan TOML to run on the executor (ADR 0069, 0070); needs `--engine dense` or `moe`. The rows gain
    /// `plan_id`, `ranks`, `comm_path` and `ranges_per_token`.
    #[arg(long, value_name = "FILE")]
    plan: Option<PathBuf>,
    /// Hardware graph (TOML or JSON) the plan is validated against; default: discover this box.
    #[arg(long, value_name = "FILE", requires = "plan")]
    hw: Option<PathBuf>,
    /// Run only the first N layers and the head (calibration anchors, ADR 0068): the timing is
    /// real, the logits are not. Needs `--engine dense` or `moe`, and one stage.
    #[arg(long, value_name = "N")]
    layer_limit: Option<u32>,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum BenchEngine {
    Llama,
    Dense,
    /// The Qwen3 MoE builder (ADR 0033), on the executor when `--plan` is given.
    Moe,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Switch {
    On,
    Off,
}

pub fn run(a: &BenchArgs) -> anyhow::Result<()> {
    anyhow::ensure!(a.output == "json", "only `-o json` is supported");
    anyhow::ensure!(a.reps > 0, "reps must be positive");
    anyhow::ensure!(
        a.plan.is_none() || a.engine != BenchEngine::Llama,
        "--plan needs --engine dense or moe"
    );
    anyhow::ensure!(
        a.layer_limit.is_none() || a.engine != BenchEngine::Llama,
        "--layer-limit needs --engine dense or moe"
    );
    let telemetry = match a.telemetry {
        Switch::On => Some(Arc::new(Telemetry::new(&TelemetryConfig::from_env())?)),
        Switch::Off => None,
    };
    let mut rows = Vec::new();
    if a.engine != BenchEngine::Llama {
        rows = run_builder(a, &telemetry)?;
    }
    // Like llama-bench: one context per test, sized n_prompt + n_gen of that test.
    for (n_prompt, n_gen) in [(a.n_prompt, 0), (0, a.n_gen)] {
        if a.engine != BenchEngine::Llama {
            break;
        }
        if n_prompt + n_gen == 0 {
            continue;
        }
        let mut engine = Engine::load(
            &a.model.model,
            &a.model.placement(),
            &a.model.context(n_prompt + n_gen),
            telemetry.clone(),
        )?;
        engine.bench_reset();
        engine.bench_warmup(n_prompt as usize, n_gen as usize)?;
        // Counters are cumulative: the delta over the timed reps shows graph reuse without the warmup.
        let perf0 = engine.perf();
        let mut samples = Vec::with_capacity(a.reps as usize);
        for _ in 0..a.reps {
            engine.bench_reset();
            let d = if n_prompt > 0 {
                engine.bench_prompt(n_prompt as usize)?
            } else {
                engine.bench_gen(n_gen as usize)?
            };
            samples.push(d);
        }
        let perf1 = engine.perf();
        let settings = Settings {
            n_batch: a.model.batch,
            n_ubatch: a.model.ubatch,
            n_threads: a.model.threads.unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map_or(1, |n| i32::try_from(n.get()).unwrap_or(1))
            }),
            n_gpu_layers: a.model.ngl,
            flash_attn: match a.model.flash_attn {
                FlashAttn::Auto => -1,
                FlashAttn::Off => 0,
                FlashAttn::On => 1,
            },
            tensor_buft_overrides: overrides_str(
                a.model
                    .override_tensor
                    .iter()
                    .map(|o| (o.pattern.as_str(), o.buffer.as_str())),
            ),
            cuda_graphs: engine.graphs(),
            n_eval: perf1.n_eval - perf0.n_eval,
            n_reused: perf1.n_reused - perf0.n_reused,
        };
        rows.push(row(n_prompt, n_gen, &samples, &settings));
    }
    if let Some(t) = &telemetry {
        let st = t.ring().stats();
        eprintln!(
            "steps {} mean {:.2} ms p95 {:.2} ms",
            st.count, st.mean_ms, st.p95_ms
        );
        t.shutdown();
    }
    println!("{}", serde_json::to_string_pretty(&rows)?);
    Ok(())
}

fn model_err(e: ModelError) -> anyhow::Error {
    anyhow::anyhow!("{e}")
}

/// What the bench loop needs of an own builder.
trait BenchModel {
    fn reset(&mut self);
    /// One sequence-0 step over `tokens`; the last token's logits go into `out`.
    fn step(&mut self, tokens: &[i32], out: &mut Vec<f32>) -> anyhow::Result<()>;
    fn graphs_built(&self) -> u64;
    fn n_ranks(&self) -> usize;
    fn comm_path(&self) -> Option<&'static str>;
    fn reset_counters(&mut self);
    /// Ranges launched by rank 0 since the counters were reset (`None` off the executor).
    fn ranges0(&self) -> Option<u64>;
}

macro_rules! bench_model {
    ($t:ty) => {
        impl BenchModel for $t {
            fn reset(&mut self) {
                <$t>::reset(self);
            }
            fn step(&mut self, tokens: &[i32], out: &mut Vec<f32>) -> anyhow::Result<()> {
                let item = SeqEval {
                    seq: 0,
                    tokens,
                    all_logits: false,
                };
                self.decode(&[item], out).map_err(model_err)?;
                Ok(())
            }
            fn graphs_built(&self) -> u64 {
                self.step_graphs_built()
            }
            fn n_ranks(&self) -> usize {
                <$t>::n_ranks(self)
            }
            fn comm_path(&self) -> Option<&'static str> {
                <$t>::comm_path(self)
            }
            fn reset_counters(&mut self) {
                self.reset_exec_counters();
            }
            fn ranges0(&self) -> Option<u64> {
                self.exec_counters().and_then(|c| c.ranges.first().copied())
            }
        }
    };
}
bench_model!(Qwen3);
bench_model!(Qwen3Moe);

/// Loads the builder of `a.engine` for a test of `n_ctx` positions: across the ranks of a plan
/// when there is one, else on one backend.
fn load_builder(
    a: &BenchArgs,
    backend: BackendChoice,
    planned: &Option<crate::sharded::ExecPlan>,
    n_ctx: usize,
    telemetry: &Option<Arc<Telemetry>>,
) -> anyhow::Result<Box<dyn BenchModel>> {
    let path = &a.model.model;
    if a.engine == BenchEngine::Moe {
        let mut o = LoadOptions::new(backend);
        o.cpu_threads = a.model.threads.and_then(|t| usize::try_from(t).ok());
        o.n_ctx = n_ctx;
        o.telemetry = telemetry.clone();
        o.layer_limit = a.layer_limit;
        let m = match planned {
            Some(p) => Qwen3Moe::load_sharded(path, &p.topology, &o),
            None => Qwen3Moe::load(path, &o),
        };
        return Ok(Box::new(m.map_err(model_err)?));
    }
    let mut o = DenseOptions::new(backend);
    o.threads = a.model.threads;
    o.n_ctx_pool = n_ctx;
    o.telemetry = telemetry.clone();
    o.layer_limit = a.layer_limit;
    // A layer limit runs on the executor.
    o.use_executor |= a.layer_limit.is_some();
    let m = match planned {
        Some(p) => Qwen3::load_sharded(path, &p.topology, &o),
        None => Qwen3::load(path, &o),
    };
    Ok(Box::new(m.map_err(model_err)?))
}

/// The own builders' tests: llama-bench's pp and tg through [`Qwen3`] or [`Qwen3Moe`], on the
/// executor when a plan names the ranks, else on one backend. Rows carry
/// `plan_id`, `ranks`, `comm_path` (`null` without the executor) and `ranges_per_token`
/// (ranges launched by rank 0 per decode call during the timed repetitions).
fn run_builder(
    a: &BenchArgs,
    telemetry: &Option<Arc<Telemetry>>,
) -> anyhow::Result<Vec<serde_json::Value>> {
    let on_gpu = a.model.ngl != 0 && Backend::gpu_count() > 0;
    let backend = if on_gpu {
        BackendChoice::Gpu {
            index: usize::try_from(a.model.main_gpu).unwrap_or(0),
        }
    } else {
        BackendChoice::Cpu {
            threads: a.model.threads.and_then(|t| usize::try_from(t).ok()),
        }
    };
    let planned = match &a.plan {
        Some(p) => Some(crate::sharded::load_plan(
            &a.model.model,
            p,
            a.hw.as_deref(),
        )?),
        None => None,
    };
    let ubatch = a.model.ubatch.max(1) as usize;
    // A plan that pipelines its prefill gets `M x ubatch` tokens per step, which the builder cuts
    // into M micro-batches (ADR 0077); M = 1 keeps one ubatch per step.
    let micro = planned.as_ref().map_or(1, |p| {
        if p.topology.stages.len() > 1 {
            p.topology.prefill_micro_batches.max(1) as usize
        } else {
            1
        }
    });
    let prompt_step = ubatch * micro;
    let mut rows = Vec::new();
    for (n_prompt, n_gen) in [(a.n_prompt, 0), (0, a.n_gen)] {
        if n_prompt + n_gen == 0 {
            continue;
        }
        let mut m = load_builder(
            a,
            backend,
            &planned,
            (n_prompt + n_gen) as usize + 1,
            telemetry,
        )?;
        // Byte-level token ids, deterministic: valid in every vocabulary.
        let mut x = 7u32;
        let toks: Vec<i32> = (0..(n_prompt.max(n_gen) as usize))
            .map(|_| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((x >> 16) % 256) as i32
            })
            .collect();
        let mut out = Vec::new();
        let mut once = |m: &mut dyn BenchModel| -> anyhow::Result<Duration> {
            m.reset();
            let t0 = Instant::now();
            if n_prompt > 0 {
                for chunk in toks[..n_prompt as usize].chunks(prompt_step) {
                    m.step(chunk, &mut out)?;
                }
            } else {
                for i in 0..n_gen as usize {
                    m.step(&toks[i..=i], &mut out)?;
                }
            }
            Ok(t0.elapsed())
        };
        once(&mut *m)?;
        let built0 = m.graphs_built();
        m.reset_counters();
        let mut samples = Vec::with_capacity(a.reps as usize);
        for _ in 0..a.reps {
            samples.push(once(&mut *m)?);
        }
        let calls = u64::from(a.reps)
            * if n_prompt > 0 {
                (n_prompt as usize).div_ceil(prompt_step) as u64
            } else {
                u64::from(n_gen)
            };
        let built = m.graphs_built() - built0;
        let settings = Settings {
            n_batch: a.model.batch,
            n_ubatch: a.model.ubatch,
            n_threads: a.model.threads.unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map_or(1, |n| i32::try_from(n.get()).unwrap_or(1))
            }),
            n_gpu_layers: if on_gpu { 99 } else { 0 },
            flash_attn: 1,
            tensor_buft_overrides: overrides_str(std::iter::empty()),
            cuda_graphs: on_gpu,
            n_eval: i32::try_from(calls).unwrap_or(i32::MAX),
            n_reused: i32::try_from(calls.saturating_sub(built)).unwrap_or(i32::MAX),
        };
        let mut r = row(n_prompt, n_gen, &samples, &settings);
        if let Some(o) = r.as_object_mut() {
            o.insert(
                "builder".into(),
                if a.engine == BenchEngine::Moe {
                    "moe"
                } else {
                    "dense"
                }
                .into(),
            );
            o.insert(
                "plan_id".into(),
                planned
                    .as_ref()
                    .map_or(serde_json::Value::Null, |p| p.plan_id.clone().into()),
            );
            o.insert("ranks".into(), m.n_ranks().into());
            // Set only when limited: the harness merges anchors with `setdefault`.
            if let Some(n) = a.layer_limit {
                o.insert("layer_limit".into(), n.into());
            }
            o.insert("comm_path".into(), m.comm_path().into());
            let ranges0 = m.ranges0();
            o.insert(
                "ranges_per_token".into(),
                ranges0.map_or(serde_json::Value::Null, |n| {
                    (n as f64 / calls.max(1) as f64).into()
                }),
            );
        }
        rows.push(r);
    }
    Ok(rows)
}

/// Effective settings of one test, reported under llama-bench's field names (`flash_attn`: -1 auto, 0 off, 1 on).
/// `cuda_graphs`, `n_eval` and `n_reused` are Tetraeder additions.
struct Settings {
    n_batch: u32,
    n_ubatch: u32,
    n_threads: i32,
    n_gpu_layers: i32,
    flash_attn: i32,
    tensor_buft_overrides: String,
    cuda_graphs: bool,
    /// Decode calls and graph reuses during the timed repetitions.
    n_eval: i32,
    n_reused: i32,
}

/// llama-bench's spelling: `none`, else `pattern=buffer` joined by `;`.
fn overrides_str<'a>(ov: impl Iterator<Item = (&'a str, &'a str)>) -> String {
    let v: Vec<String> = ov.map(|(p, b)| format!("{p}={b}")).collect();
    if v.is_empty() {
        "none".into()
    } else {
        v.join(";")
    }
}

fn row(n_prompt: u32, n_gen: u32, samples: &[Duration], st: &Settings) -> serde_json::Value {
    let n = f64::from(n_prompt + n_gen);
    let ts: Vec<f64> = samples.iter().map(|d| n / d.as_secs_f64()).collect();
    let ns: Vec<u128> = samples.iter().map(Duration::as_nanos).collect();
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let avg_ts = mean(&ts);
    let stddev_ts = (ts.iter().map(|x| (x - avg_ts).powi(2)).sum::<f64>()
        / (ts.len().max(2) - 1) as f64)
        .sqrt();
    serde_json::json!({
        "build_commit": option_env!("TETRAEDER_GIT_SHA").unwrap_or("unknown"),
        "engine": "tetraeder",
        "n_batch": st.n_batch,
        "n_ubatch": st.n_ubatch,
        "n_threads": st.n_threads,
        "n_gpu_layers": st.n_gpu_layers,
        "flash_attn": st.flash_attn,
        "tensor_buft_overrides": st.tensor_buft_overrides,
        "cuda_graphs": st.cuda_graphs,
        "n_eval": st.n_eval,
        "n_reused": st.n_reused,
        "n_prompt": n_prompt,
        "n_gen": n_gen,
        "avg_ns": ns.iter().sum::<u128>() / ns.len() as u128,
        "avg_ts": avg_ts,
        "stddev_ts": stddev_ts,
        "samples_ns": ns,
        "samples_ts": ts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_has_llama_bench_shape() {
        let st = Settings {
            n_batch: 2048,
            n_ubatch: 512,
            n_threads: 4,
            n_gpu_layers: 99,
            flash_attn: 1,
            tensor_buft_overrides: overrides_str([("exps", "CPU")].into_iter()),
            cuda_graphs: true,
            n_eval: 16,
            n_reused: 15,
        };
        let r = row(0, 8, &[Duration::from_secs(1), Duration::from_secs(2)], &st);
        assert_eq!(r["flash_attn"], 1);
        assert_eq!(r["tensor_buft_overrides"], "exps=CPU");
        assert_eq!(r["n_reused"], 15);
        assert_eq!(r["cuda_graphs"], true);
        assert_eq!(overrides_str(std::iter::empty()), "none");
        assert_eq!(r["n_gen"], 8);
        assert_eq!(r["n_prompt"], 0);
        assert_eq!(r["avg_ts"], 6.0);
        assert_eq!(r["samples_ts"][1], 4.0);
    }

    #[test]
    fn the_harness_spelling_parses() {
        use clap::Parser;
        #[derive(Parser)]
        struct P {
            #[command(flatten)]
            bench: BenchArgs,
        }
        // The flags `scripts/phase5_box.sh` passes in `tet_cmd`.
        let p = P::parse_from([
            "x",
            "-m",
            "m.gguf",
            "--engine",
            "moe",
            "--plan",
            "p.toml",
            "-p",
            "2048",
            "-n",
            "128",
            "-r",
            "1",
            "-t",
            "8",
            "--ngl",
            "-1",
            "--ubatch",
            "512",
            "--flash-attn",
            "on",
            "-o",
            "json",
            "--layer-limit",
            "16",
        ]);
        let b = &p.bench;
        assert!(b.engine == BenchEngine::Moe);
        assert_eq!(b.layer_limit, Some(16));
        assert_eq!(b.plan.as_deref(), Some(std::path::Path::new("p.toml")));
        assert_eq!((b.n_prompt, b.n_gen, b.reps), (2048, 128, 1));
        assert!(
            P::parse_from(["x", "-m", "m.gguf", "--engine", "dense"])
                .bench
                .engine
                == BenchEngine::Dense
        );
        assert!(P::try_parse_from(["x", "-m", "m.gguf", "--engine", "gguf"]).is_err());
        let err = run(&P::parse_from(["x", "-m", "m.gguf", "--layer-limit", "2"]).bench)
            .unwrap_err()
            .to_string();
        assert!(err.contains("--engine dense"), "{err}");
    }

    #[test]
    fn a_plan_needs_the_dense_engine() {
        use clap::Parser;
        #[derive(Parser)]
        struct P {
            #[command(flatten)]
            bench: BenchArgs,
        }
        let p = P::parse_from(["x", "-m", "m.gguf", "--plan", "p.toml"]);
        let err = run(&p.bench).unwrap_err().to_string();
        assert!(err.contains("--engine dense"), "{err}");
        // --hw only goes with --plan.
        assert!(P::try_parse_from(["x", "-m", "m.gguf", "--hw", "h.toml"]).is_err());
    }
}
