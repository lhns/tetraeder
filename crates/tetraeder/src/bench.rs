//! `tetraeder bench`: llama-bench's pp and tg tests through the Tetraeder engine path, printed
//! as llama-bench's JSON rows so `tetraeder-bench` parses both engines alike (ADR 0026).

use crate::engine_args::ModelArgs;
use clap::{Args, ValueEnum};
use std::sync::Arc;
use std::time::Duration;
use tetraeder_engine::Engine;
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
    #[arg(short = 'r', long, default_value_t = 5)]
    reps: u32,
    /// Output format; only json.
    #[arg(short = 'o', long, default_value = "json")]
    output: String,
    /// Step telemetry: on (production mode, no exporter unless OTEL_EXPORTER_OTLP_ENDPOINT) or off.
    #[arg(long, value_enum, default_value_t = Switch::On)]
    telemetry: Switch,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Switch {
    On,
    Off,
}

pub fn run(a: &BenchArgs) -> anyhow::Result<()> {
    anyhow::ensure!(a.output == "json", "only `-o json` is supported");
    anyhow::ensure!(a.reps > 0, "reps must be positive");
    let telemetry = match a.telemetry {
        Switch::On => Some(Arc::new(Telemetry::new(&TelemetryConfig::from_env())?)),
        Switch::Off => None,
    };
    let mut rows = Vec::new();
    // Like llama-bench: one context per test, sized n_prompt + n_gen of that test.
    for (n_prompt, n_gen) in [(a.n_prompt, 0), (0, a.n_gen)] {
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
        rows.push(row(n_prompt, n_gen, &samples));
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

fn row(n_prompt: u32, n_gen: u32, samples: &[Duration]) -> serde_json::Value {
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
        let r = row(0, 8, &[Duration::from_secs(1), Duration::from_secs(2)]);
        assert_eq!(r["n_gen"], 8);
        assert_eq!(r["n_prompt"], 0);
        assert_eq!(r["avg_ts"], 6.0);
        assert_eq!(r["samples_ts"][1], 4.0);
    }
}
