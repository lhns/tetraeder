//! `tetraeder serve`: the OpenAI and Anthropic APIs over an engine (ADR 0028, ADR 0046).

use crate::engine_args::PlacementArgs;
use anyhow::Context as _;
use clap::{Args, ValueEnum};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tetraeder_gateway::engine::Engine as GatewayEngine;
use tetraeder_gateway::fake::{FakeEngine, serve_script};
use tetraeder_gateway::gateway::{Gateway, GatewayConfig};
use tetraeder_gateway::reqlog::RequestLog;
use tetraeder_model::{BackendChoice, LoadOptions, MissPolicy};
use tetraeder_scheduler::{Scheduler, SchedulerConfig};
use tetraeder_telemetry::{Telemetry, TelemetryConfig};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum EngineKind {
    /// The Qwen3 MoE builder for GGUF architecture `qwen3moe`, else libllama.
    Auto,
    Llama,
    Moe,
}

#[derive(Args)]
pub struct ServeArgs {
    /// GGUF model file (not needed with --fake).
    #[arg(short = 'm', long, required_unless_present = "fake")]
    pub model: Option<PathBuf>,
    #[command(flatten)]
    pub place: PlacementArgs,
    #[arg(long, value_enum, default_value = "auto")]
    pub engine: EngineKind,
    /// Context size in tokens (the KV cache is allocated for all of it).
    #[arg(long, default_value_t = 32768)]
    pub ctx: u32,
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    #[arg(long, default_value_t = 8080)]
    pub port: u16,
    /// Served model id; default is the GGUF file name without extension.
    #[arg(long)]
    pub alias: Option<String>,
    /// Model that unknown ids resolve to (Claude Code sends its own `claude-*` ids).
    #[arg(long)]
    pub default_model: Option<String>,
    /// Resolve every unknown model id to the served model.
    #[arg(long, conflicts_with = "default_model")]
    pub accept_any_model: bool,
    /// Sampling defaults for requests that send none.
    #[arg(long, default_value_t = 0.8)]
    pub temp: f32,
    #[arg(long, default_value_t = 40)]
    pub top_k: u32,
    #[arg(long, default_value_t = 0.95)]
    pub top_p: f32,
    /// Requests that may wait for the engine; more get 429/529.
    #[arg(long, default_value_t = 16)]
    pub max_queued: usize,
    /// Host RAM for saved sequences (prefix cache), in MiB; 0 turns it off.
    #[arg(long, default_value_t = 8192)]
    pub cache_ram: usize,
    /// Append one JSON line per request (prompts included) to this file.
    #[arg(long)]
    pub request_log: Option<PathBuf>,
    /// Jinja chat template that replaces the one in the GGUF.
    #[arg(long)]
    pub chat_template_file: Option<PathBuf>,
    /// Serve a scripted engine without a model (conformance tests).
    #[arg(long)]
    pub fake: bool,
    /// Pause per token of the fake engine, in milliseconds.
    #[arg(long, default_value_t = 0)]
    pub fake_token_ms: u64,
    /// MoE builder: expert slots per layer in GPU memory (default: all experts resident).
    #[arg(long)]
    pub expert_slots: Option<usize>,
    /// MoE builder: expert slots per layer in host memory between the file and the GPU.
    #[arg(long)]
    pub ram_slots: Option<usize>,
    /// MoE builder: what a routed expert outside the GPU slots costs, `upload` or `cpu`.
    #[arg(long, default_value = "upload", value_parser = parse_miss)]
    pub miss: MissPolicy,
    /// MoE builder: read predicted experts ahead into the RAM tier.
    #[arg(long)]
    pub prefetch: bool,
}

fn parse_miss(s: &str) -> Result<MissPolicy, String> {
    match s {
        "upload" => Ok(MissPolicy::Upload),
        "cpu" => Ok(MissPolicy::Cpu),
        _ => Err(format!("`{s}`: expected upload or cpu")),
    }
}

pub fn run(a: &ServeArgs) -> anyhow::Result<()> {
    let telemetry = Arc::new(Telemetry::new(&TelemetryConfig::from_env())?);
    telemetry.install_global()?;
    crate::info::record_cpu_backend(telemetry.metrics());
    let engine = build_engine(a, &telemetry)?;
    let served = engine
        .models()
        .first()
        .map(|m| m.id.clone())
        .context("the engine serves no model")?;

    let cfg = GatewayConfig {
        max_queued: a.max_queued,
        default_model: if a.accept_any_model {
            Some(served.clone())
        } else {
            a.default_model.clone()
        },
        ..GatewayConfig::default()
    };
    let mut gw = Gateway::new(engine, cfg).with_meter(&opentelemetry::global::meter("tetraeder"));
    let log = match &a.request_log {
        Some(p) => {
            let log = RequestLog::open(p)
                .with_context(|| format!("cannot open request log {}", p.display()))?;
            gw = gw.with_cost_sink(log.clone());
            Some(log)
        }
        None => None,
    };

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let res = rt.block_on(async {
        let listener = tokio::net::TcpListener::bind((a.host.as_str(), a.port))
            .await
            .with_context(|| format!("cannot listen on {}:{}", a.host, a.port))?;
        let addr = listener.local_addr()?;
        eprintln!("listening on http://{addr}");
        tokio::select! {
            r = tetraeder_gateway::server::serve_with(listener, Arc::new(gw), log) => {
                r.context("server failed")
            }
            _ = tokio::signal::ctrl_c() => Ok(()),
        }
    });
    telemetry.shutdown();
    res
}

fn build_engine(
    a: &ServeArgs,
    telemetry: &Arc<Telemetry>,
) -> anyhow::Result<Arc<dyn GatewayEngine>> {
    if a.fake {
        let mut e = match &a.chat_template_file {
            Some(f) => {
                let t = std::fs::read_to_string(f)
                    .with_context(|| format!("cannot read chat template {}", f.display()))?;
                FakeEngine::with_template(Some(t), serve_script)
            }
            None => FakeEngine::new(serve_script),
        };
        e.set_model_id(a.alias.as_deref().unwrap_or("fake"));
        e.set_context_length(a.ctx);
        e.token_delay = Duration::from_millis(a.fake_token_ms);
        e.gpu_seconds = None;
        e.bytes_streamed = None;
        return Ok(Arc::new(e));
    }
    let path = a.model.clone().context("-m is required")?;
    let moe = match a.engine {
        EngineKind::Llama => false,
        EngineKind::Moe => true,
        EngineKind::Auto => {
            tetraeder_scheduler::architecture(&path)
                .map_err(anyhow::Error::msg)
                .context("cannot read the model architecture")?
                == "qwen3moe"
        }
    };
    let mut cfg = SchedulerConfig::new(&path);
    cfg.alias = a.alias.clone();
    cfg.temperature = a.temp;
    cfg.top_k = a.top_k;
    cfg.top_p = a.top_p;
    cfg.metrics = Some(telemetry.metrics().clone());
    cfg.cache_ram_bytes = a.cache_ram << 20;
    cfg.chat_template = match &a.chat_template_file {
        Some(f) => Some(
            std::fs::read_to_string(f)
                .with_context(|| format!("cannot read chat template {}", f.display()))?,
        ),
        None => tetraeder_scheduler::chat_template(&path),
    };
    let tel = Arc::clone(telemetry);
    let a = a.clone_load_args();
    let engine = Scheduler::start(cfg, move || {
        if moe {
            tetraeder_scheduler::load_moe(&a.path, &a.moe_options(tel))
        } else {
            tetraeder_scheduler::load_llama(
                &a.path,
                &a.place.placement(),
                &a.place.context(a.ctx),
                Some(tel),
            )
        }
    })
    .map_err(anyhow::Error::msg)?;
    Ok(Arc::new(engine))
}

/// What the worker thread needs to load a model.
struct LoadArgs {
    path: PathBuf,
    place: PlacementArgs,
    ctx: u32,
    expert_slots: Option<usize>,
    ram_slots: Option<usize>,
    miss: MissPolicy,
    prefetch: bool,
}

impl LoadArgs {
    /// `--ngl 0` keeps the MoE builder on the CPU; anything else uses GPU `main_gpu`.
    fn moe_options(&self, telemetry: Arc<Telemetry>) -> LoadOptions {
        let backend = if self.place.ngl == 0 {
            BackendChoice::Cpu {
                threads: self.place.threads.and_then(|t| usize::try_from(t).ok()),
            }
        } else {
            BackendChoice::Gpu {
                index: usize::try_from(self.place.main_gpu).unwrap_or(0),
            }
        };
        let mut o = LoadOptions::new(backend);
        o.cpu_threads = self.place.threads.and_then(|t| usize::try_from(t).ok());
        o.n_ctx = self.ctx as usize;
        o.expert_slots = self.expert_slots;
        o.ram_slots = self.ram_slots;
        o.miss = self.miss;
        o.prefetch = self.prefetch;
        o.telemetry = Some(telemetry);
        o
    }
}

impl ServeArgs {
    fn clone_load_args(&self) -> LoadArgs {
        LoadArgs {
            path: self.model.clone().unwrap_or_default(),
            place: self.place.clone(),
            ctx: self.ctx,
            expert_slots: self.expert_slots,
            ram_slots: self.ram_slots,
            miss: self.miss,
            prefetch: self.prefetch,
        }
    }
}
