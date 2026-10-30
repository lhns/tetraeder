//! `tetraeder run`: stream one completion to stdout, stats to stderr.

use crate::engine_args::ModelArgs;
use clap::Args;
use std::io::Write;
use std::ops::ControlFlow;
use std::sync::Arc;
use tetraeder_engine::{Engine, GenerateRequest, Prompt, Sampling};
use tetraeder_telemetry::{Telemetry, TelemetryConfig};

#[derive(Args)]
pub struct RunArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// Prompt text; special tokens such as `<|im_start|>` are parsed.
    #[arg(long)]
    prompt: String,
    /// Tokens to generate at most.
    #[arg(short = 'n', long, default_value_t = 128)]
    max_tokens: u32,
    /// Context size in tokens.
    #[arg(long, default_value_t = 4096)]
    ctx: u32,
    /// 0 = greedy.
    #[arg(long, default_value_t = 0.0)]
    temp: f32,
    #[arg(long, default_value_t = 0)]
    seed: u32,
    #[arg(long, default_value_t = 40)]
    top_k: i32,
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,
}

pub fn run(a: &RunArgs) -> anyhow::Result<()> {
    // OTLP export only when OTEL_EXPORTER_OTLP_ENDPOINT is set (ADR 0012).
    let telemetry = Arc::new(Telemetry::new(&TelemetryConfig::from_env())?);
    telemetry.install_global()?;
    let mut engine = Engine::load(
        &a.model.model,
        &a.model.placement(),
        &a.model.context(a.ctx),
        Some(Arc::clone(&telemetry)),
    )?;
    let mut req = GenerateRequest::new(Prompt::Text(a.prompt.clone()), a.max_tokens);
    if a.temp > 0.0 {
        req.sampling = Sampling::Random {
            seed: a.seed,
            temperature: a.temp,
            top_k: a.top_k,
            top_p: a.top_p,
        };
    }
    let mut out = std::io::stdout().lock();
    let report = engine.generate(&req, |ev| {
        // A closed stdout (e.g. `| head`) cancels the request.
        match out
            .write_all(ev.piece.as_bytes())
            .and_then(|()| out.flush())
        {
            Ok(()) => ControlFlow::Continue(()),
            Err(_) => ControlFlow::Break(()),
        }
    })?;
    let _ = writeln!(out);
    let decode_s = (report.total - report.ttft).as_secs_f64();
    let n = report.tokens.len();
    eprintln!(
        "prompt {} tokens, ttft {:.1} ms; generated {} tokens, {:.2} tok/s; stop {:?}",
        report.prompt_tokens.len(),
        report.ttft.as_secs_f64() * 1e3,
        n,
        if n > 1 && decode_s > 0.0 {
            (n - 1) as f64 / decode_s
        } else {
            0.0
        },
        report.stop_reason
    );
    telemetry.shutdown();
    Ok(())
}
