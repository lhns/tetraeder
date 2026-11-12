//! `tetraeder serve`: the OpenAI and Anthropic APIs over an engine (ADR 0028, ADR 0046).

use crate::engine_args::PlacementArgs;
use crate::replan;
use anyhow::Context as _;
use clap::{Args, ValueEnum};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tetraeder_gateway::engine::Engine as GatewayEngine;
use tetraeder_gateway::fake::{FakeEngine, serve_script};
use tetraeder_gateway::gateway::{Gateway, GatewayConfig};
use tetraeder_gateway::reqlog::RequestLog;
use tetraeder_model::{BackendChoice, LoadOptions, MissPolicy};
use tetraeder_planner::Workload;
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
    /// MoE builder: read predicted experts ahead into the RAM tier, `on` or `off` (bare
    /// `--prefetch` is `on`). Default: on when there is a RAM tier.
    #[arg(
        long,
        value_name = "on|off",
        value_parser = parse_on_off,
        num_args = 0..=1,
        default_missing_value = "on"
    )]
    pub prefetch: Option<bool>,
    /// MoE builder: layers ahead the prefetch predicts, 1 or 2 (the measured 235B runs used 2).
    #[arg(long, default_value_t = 2, value_parser = clap::value_parser!(u8).range(1..=2))]
    pub prefetch_depth: u8,
    /// MoE builder: read experts through the persistent I/O service (ADR 0054), `on` or `off`.
    /// Default: on when there is a RAM tier.
    #[arg(long, value_name = "on|off", value_parser = parse_on_off)]
    pub io_service: Option<bool>,
    /// MoE builder: compute the CPU experts of a decode step as their reads land (ADR 0054),
    /// `on` or `off`. Needs the I/O service. Default: on when the service is on and `--miss` is
    /// `cpu`.
    #[arg(long, value_name = "on|off", value_parser = parse_on_off)]
    pub overlap: Option<bool>,
    /// Serve a plan and replan by drain and restart (ADR 0055): `auto` picks one of the
    /// executable candidates for this box and model, or a plan TOML file. Replaces the
    /// placement flags.
    #[arg(
        long,
        value_name = "auto|FILE",
        conflicts_with_all = [
            "ngl", "override_tensor", "split_mode", "tensor_split", "main_gpu", "threads",
            "batch", "ubatch", "expert_slots", "ram_slots", "fake",
        ]
    )]
    pub plan: Option<String>,
    /// Hardware graph (TOML or JSON, typically from `tetraeder-plan score`); default: discover
    /// the box and take GPU rates from the priors.
    #[arg(long, value_name = "FILE", requires = "plan")]
    pub hw: Option<PathBuf>,
    /// Request shape the plan is chosen for, `PROMPT:GENERATED` tokens.
    #[arg(long, default_value = "2048:128", value_parser = replan::start::parse_workload)]
    pub plan_workload: Workload,
    /// GPU memory the plan may use, in bytes (default: the GPU capacity).
    #[arg(long, requires = "plan")]
    pub plan_vram: Option<u64>,
    /// Host memory the plan may use, in bytes (default: the RAM capacity).
    #[arg(long, requires = "plan")]
    pub plan_ram: Option<u64>,
    /// Routing replay of the builder (`tetraeder-bench replay` on a recorded trace): the builder
    /// candidates of `--plan auto` are predicted from its disk bytes per token, so a larger RAM
    /// tier wins when the disk is slow and a drifting disk re-picks (ADR 0056). Without it the
    /// uniform routing prior is used and the disk costs nothing.
    #[arg(long, requires = "plan")]
    pub tier_replay: Option<PathBuf>,
    /// Admin listener of `--plan`: `POST /admin/replan`, `POST /api/v2/alerts` (the SLO burn
    /// alerts of vmalert) and `GET /admin/plan`. No authentication: keep it on loopback.
    #[arg(long, default_value = "127.0.0.1:8081")]
    pub admin: SocketAddr,
    /// Seconds between residual checks of the step time against the prediction.
    #[arg(long, default_value_t = 60)]
    pub replan_tick: u64,
    /// Seconds a switch waits for requests to finish before it cuts them.
    #[arg(long, default_value_t = 120)]
    pub drain_timeout: u64,
    /// Seconds after a switch before the next one.
    #[arg(long, default_value_t = 600)]
    pub replan_dwell: u64,
    /// Plan hints TOML; only `[stability]` (switch rate, freeze windows in UTC) is read.
    #[arg(long, value_name = "FILE")]
    pub hints: Option<PathBuf>,
}

fn parse_miss(s: &str) -> Result<MissPolicy, String> {
    match s {
        "upload" => Ok(MissPolicy::Upload),
        "cpu" => Ok(MissPolicy::Cpu),
        _ => Err(format!("`{s}`: expected upload or cpu")),
    }
}

fn parse_on_off(s: &str) -> Result<bool, String> {
    match s {
        "on" => Ok(true),
        "off" => Ok(false),
        _ => Err(format!("`{s}`: expected on or off")),
    }
}

/// Prefetch is on with a RAM tier unless `--prefetch off`.
pub fn resolve_prefetch(has_ram_tier: bool, prefetch: Option<bool>) -> bool {
    prefetch.unwrap_or(has_ram_tier)
}

/// The I/O service and overlap settings of the MoE builder (ADR 0054). Unset flags default to
/// the service on with a RAM tier and overlap on when the misses also run on the CPU; overlap
/// without the service is an error.
pub fn resolve_overlap(
    has_ram_tier: bool,
    miss: MissPolicy,
    io_service: Option<bool>,
    overlap: Option<bool>,
) -> Result<(bool, bool), String> {
    let service = io_service.unwrap_or(has_ram_tier);
    let overlap = overlap.unwrap_or(service && miss == MissPolicy::Cpu);
    if overlap && !service {
        return Err(if has_ram_tier {
            "--overlap on needs the I/O service (drop --io-service off)".into()
        } else {
            "--overlap on needs the I/O service, which needs a RAM tier (--ram-slots)".into()
        });
    }
    Ok((service, overlap))
}

pub fn run(a: &ServeArgs) -> anyhow::Result<()> {
    let telemetry = Arc::new(Telemetry::new(&TelemetryConfig::from_env())?);
    telemetry.install_global()?;
    crate::info::record_cpu_backend(telemetry.metrics(), a.place.threads);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let planned = match &a.plan {
        Some(spec) => Some(replan::start(a, spec, &telemetry, rt.handle().clone())?),
        None => None,
    };
    let engine: Arc<dyn GatewayEngine> = match &planned {
        Some(p) => p.engine.clone(),
        None => build_engine(a, &telemetry)?,
    };
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

    let res = rt.block_on(async {
        let listener = tokio::net::TcpListener::bind((a.host.as_str(), a.port))
            .await
            .with_context(|| format!("cannot listen on {}:{}", a.host, a.port))?;
        let addr = listener.local_addr()?;
        eprintln!("listening on http://{addr}");
        let armed = planned
            .as_ref()
            .and_then(|p| p.replan.clone().map(|h| (h, p.dwell_s)));
        let admin = match armed {
            Some((h, dwell)) => {
                let l = tokio::net::TcpListener::bind(a.admin)
                    .await
                    .with_context(|| format!("cannot listen on {}", a.admin))?;
                eprintln!("admin on http://{}", l.local_addr()?);
                Some((l, replan::admin::router(h, dwell)))
            }
            None => None,
        };
        let admin_server = async {
            match admin {
                Some((l, app)) => axum::serve(l, app).await.context("admin server failed"),
                None => std::future::pending().await,
            }
        };
        tokio::select! {
            r = tetraeder_gateway::server::serve_with(listener, Arc::new(gw), log) => {
                r.context("server failed")
            }
            r = admin_server => r,
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
    let cfg = SchedParts::new(a)?.config(&path, telemetry.metrics().clone());
    let tel = Arc::clone(telemetry);
    if moe {
        resolve_overlap(a.ram_slots.is_some(), a.miss, a.io_service, a.overlap)
            .map_err(anyhow::Error::msg)?;
    }
    let a = a.clone_load_args();
    let engine = Scheduler::start(cfg, move || {
        if moe {
            tetraeder_scheduler::load_moe(&a.path, &a.moe_options(tel)?)
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

/// The parts of a `SchedulerConfig` that do not change when the engine is reloaded.
pub struct SchedParts {
    alias: Option<String>,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    cache_ram_bytes: usize,
    chat_template: Option<String>,
}

impl SchedParts {
    pub fn new(a: &ServeArgs) -> anyhow::Result<Self> {
        let path = a.model.clone().context("-m is required")?;
        Ok(Self {
            alias: a.alias.clone(),
            temperature: a.temp,
            top_k: a.top_k,
            top_p: a.top_p,
            cache_ram_bytes: a.cache_ram << 20,
            chat_template: match &a.chat_template_file {
                Some(f) => Some(
                    std::fs::read_to_string(f)
                        .with_context(|| format!("cannot read chat template {}", f.display()))?,
                ),
                None => tetraeder_scheduler::chat_template(&path),
            },
        })
    }

    pub fn config(&self, path: &Path, metrics: tetraeder_telemetry::Metrics) -> SchedulerConfig {
        let mut cfg = SchedulerConfig::new(path);
        cfg.alias = self.alias.clone();
        cfg.temperature = self.temperature;
        cfg.top_k = self.top_k;
        cfg.top_p = self.top_p;
        cfg.metrics = Some(metrics);
        cfg.cache_ram_bytes = self.cache_ram_bytes;
        cfg.chat_template = self.chat_template.clone();
        cfg
    }
}

/// What the worker thread needs to load a model.
struct LoadArgs {
    path: PathBuf,
    place: PlacementArgs,
    ctx: u32,
    expert_slots: Option<usize>,
    ram_slots: Option<usize>,
    miss: MissPolicy,
    prefetch: Option<bool>,
    prefetch_depth: u8,
    io_service: Option<bool>,
    overlap: Option<bool>,
}

impl LoadArgs {
    /// `--ngl 0` keeps the MoE builder on the CPU; anything else uses GPU `main_gpu`.
    fn moe_options(&self, telemetry: Arc<Telemetry>) -> Result<LoadOptions, String> {
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
        o.prefetch = resolve_prefetch(self.ram_slots.is_some(), self.prefetch);
        o.prefetch_depth = usize::from(self.prefetch_depth);
        (o.io.service, o.overlap) = resolve_overlap(
            self.ram_slots.is_some(),
            self.miss,
            self.io_service,
            self.overlap,
        )?;
        o.telemetry = Some(telemetry);
        Ok(o)
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
            prefetch_depth: self.prefetch_depth,
            io_service: self.io_service,
            overlap: self.overlap,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct P {
        #[command(flatten)]
        serve: ServeArgs,
    }

    fn parse(extra: &[&str]) -> ServeArgs {
        let mut args = vec!["x", "-m", "m.gguf"];
        args.extend(extra);
        P::parse_from(args).serve
    }

    // hardware: none
    #[test]
    fn flags_parse_on_and_off() {
        let a = parse(&[]);
        assert_eq!((a.io_service, a.overlap), (None, None));
        let a = parse(&["--io-service", "on", "--overlap", "off"]);
        assert_eq!((a.io_service, a.overlap), (Some(true), Some(false)));
        assert!(P::try_parse_from(["x", "--overlap", "maybe"]).is_err());
    }

    // hardware: none
    #[test]
    fn defaults_follow_the_ram_tier_and_miss_policy() {
        use MissPolicy::{Cpu, Upload};
        assert_eq!(resolve_overlap(true, Cpu, None, None), Ok((true, true)));
        assert_eq!(resolve_overlap(true, Upload, None, None), Ok((true, false)));
        assert_eq!(resolve_overlap(false, Cpu, None, None), Ok((false, false)));
        assert_eq!(
            resolve_overlap(false, Upload, None, None),
            Ok((false, false))
        );
        assert_eq!(
            resolve_overlap(true, Cpu, Some(true), Some(false)),
            Ok((true, false))
        );
        assert_eq!(
            resolve_overlap(true, Upload, None, Some(true)),
            Ok((true, true))
        );
        assert_eq!(
            resolve_overlap(true, Cpu, Some(false), None),
            Ok((false, false))
        );
    }

    // hardware: none
    #[test]
    fn prefetch_defaults_to_the_ram_tier() {
        assert!(resolve_prefetch(true, None));
        assert!(!resolve_prefetch(false, None));
        assert!(!resolve_prefetch(true, Some(false)));
        assert!(resolve_prefetch(false, Some(true)));
        assert_eq!(parse(&[]).prefetch, None);
        assert_eq!(parse(&["--prefetch"]).prefetch, Some(true));
        assert_eq!(parse(&["--prefetch", "off"]).prefetch, Some(false));
        assert_eq!(parse(&[]).prefetch_depth, 2);
        assert!(P::try_parse_from(["x", "--prefetch-depth", "3"]).is_err());
    }

    // hardware: none
    #[test]
    fn overlap_without_the_service_is_an_error() {
        use MissPolicy::Cpu;
        assert!(resolve_overlap(true, Cpu, Some(false), Some(true)).is_err());
        assert!(resolve_overlap(false, Cpu, None, Some(true)).is_err());
    }
}
