//! `tetraeder serve`: the OpenAI and Anthropic APIs over an engine (ADR 0028, ADR 0046).

use crate::engine_args::PlacementArgs;
use crate::replan;
use anyhow::Context as _;
use clap::{Args, ValueEnum};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tetraeder::remote_backend::{RemoteBackend, RemoteConfig};
use tetraeder_gateway::engine::Engine as GatewayEngine;
use tetraeder_gateway::fake::{FakeEngine, serve_script};
use tetraeder_gateway::gateway::{Gateway, GatewayConfig};
use tetraeder_gateway::reqlog::RequestLog;
use tetraeder_ggml_sys::ggml::Backend;
use tetraeder_ggml_sys::llama::{Model, ModelParams};
use tetraeder_model::{BackendChoice, DenseOptions, LoadOptions, MissPolicy};
use tetraeder_planner::Workload;
use tetraeder_scheduler::spec::{DraftLen, NgramDrafter, SpecConfig, SpecMethod};
use tetraeder_scheduler::{Loaded, Scheduler, SchedulerConfig};
use tetraeder_telemetry::{Telemetry, TelemetryConfig};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum EngineKind {
    /// The Qwen3 MoE builder for GGUF architecture `qwen3moe`, else libllama.
    Auto,
    Llama,
    Moe,
    /// The Qwen3 dense builder (ADR 0066), for GGUF architecture `qwen3`; `auto` keeps libllama.
    Dense,
}

/// Sampling defaults for requests that send none: llama-server's flags and defaults. A request
/// field overrides each one.
#[derive(Args, Debug, Clone)]
pub struct SamplingArgs {
    #[arg(long, alias = "temperature", default_value_t = 0.8)]
    pub temp: f32,
    #[arg(long, default_value_t = 40)]
    pub top_k: u32,
    #[arg(long, default_value_t = 0.95)]
    pub top_p: f32,
    #[arg(long, default_value_t = 0.05)]
    pub min_p: f32,
    #[arg(long = "typical", alias = "typical-p", default_value_t = 1.0)]
    pub typical: f32,
    #[arg(long = "top-nsigma", default_value_t = -1.0, allow_hyphen_values = true)]
    pub top_nsigma: f32,
    #[arg(long, default_value_t = 0.0)]
    pub xtc_probability: f32,
    #[arg(long, default_value_t = 0.1)]
    pub xtc_threshold: f32,
    #[arg(long, default_value_t = 0.0)]
    pub dynatemp_range: f32,
    #[arg(long = "dynatemp-exp", default_value_t = 1.0)]
    pub dynatemp_exp: f32,
    #[arg(long, default_value_t = 64, value_parser = clap::value_parser!(i32).range(0..))]
    pub repeat_last_n: i32,
    #[arg(long, default_value_t = 1.0)]
    pub repeat_penalty: f32,
    #[arg(long, default_value_t = 0.0)]
    pub presence_penalty: f32,
    #[arg(long, default_value_t = 0.0)]
    pub frequency_penalty: f32,
    #[arg(long, default_value_t = 0.0)]
    pub dry_multiplier: f32,
    #[arg(long, default_value_t = 1.75)]
    pub dry_base: f32,
    #[arg(long, default_value_t = 2, value_parser = clap::value_parser!(i32).range(0..))]
    pub dry_allowed_length: i32,
    #[arg(long, default_value_t = 64, value_parser = clap::value_parser!(i32).range(0..))]
    pub dry_penalty_last_n: i32,
    /// A string that ends a DRY sequence; repeat the flag for several.
    #[arg(long)]
    pub dry_sequence_breaker: Vec<String>,
    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(i32).range(0..=2))]
    pub mirostat: i32,
    #[arg(long = "mirostat-lr", default_value_t = 0.1)]
    pub mirostat_lr: f32,
    #[arg(long = "mirostat-ent", default_value_t = 5.0)]
    pub mirostat_ent: f32,
    /// Sampler order, separated by `;` (default `penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature`).
    #[arg(long)]
    pub samplers: Option<String>,
    /// Seed of requests that send none; -1 draws a random one per request.
    #[arg(long, default_value_t = -1, allow_hyphen_values = true)]
    pub seed: i64,
    /// Ban the end-of-generation tokens.
    #[arg(long)]
    pub ignore_eos: bool,
}

impl SamplingArgs {
    pub fn params(&self) -> anyhow::Result<tetraeder_scheduler::sample::SampleParams> {
        use tetraeder_gateway::engine::SamplerKind;
        anyhow::ensure!(
            self.repeat_penalty.is_finite()
                && self.repeat_penalty > 0.0
                && (1.0 / self.repeat_penalty).is_finite(),
            "--repeat-penalty must be finite and greater than 0"
        );
        let mut p = tetraeder_scheduler::sample::SampleParams {
            temperature: self.temp.max(0.0),
            top_k: self.top_k,
            top_p: self.top_p.clamp(0.0, 1.0),
            min_p: self.min_p.clamp(0.0, 1.0),
            typical_p: self.typical,
            top_n_sigma: self.top_nsigma,
            xtc_probability: self.xtc_probability.clamp(0.0, 1.0),
            xtc_threshold: self.xtc_threshold.clamp(0.0, 1.0),
            dynatemp_range: self.dynatemp_range,
            dynatemp_exponent: self.dynatemp_exp,
            penalty_last_n: self.repeat_last_n,
            penalty_repeat: self.repeat_penalty,
            penalty_freq: self.frequency_penalty,
            penalty_present: self.presence_penalty,
            dry_multiplier: self.dry_multiplier,
            dry_allowed_length: self.dry_allowed_length,
            dry_penalty_last_n: self.dry_penalty_last_n,
            mirostat: self.mirostat,
            mirostat_tau: self.mirostat_ent,
            mirostat_eta: self.mirostat_lr,
            ignore_eos: self.ignore_eos,
            ..Default::default()
        };
        if self.dry_base >= 1.0 {
            p.dry_base = self.dry_base;
        }
        if !self.dry_sequence_breaker.is_empty() {
            p.dry_sequence_breakers = self.dry_sequence_breaker.clone().into();
        }
        if let Some(list) = &self.samplers {
            let kinds = list
                .split(';')
                .map(|n| {
                    SamplerKind::from_name(n.trim())
                        .with_context(|| format!("--samplers: unknown sampler '{n}'"))
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            p.samplers = kinds.into();
        }
        Ok(p)
    }

    /// `None`: a random seed per request.
    pub fn seed(&self) -> Option<u32> {
        u32::try_from(self.seed).ok()
    }
}

#[derive(Args)]
#[command(mut_arg(
    "batch",
    |a| a.help("Ignored by serve: the step token budget is --step-tokens")
))]
pub struct ServeArgs {
    /// GGUF model file (not needed with --fake).
    #[arg(short = 'm', long, required_unless_present = "fake")]
    pub model: Option<PathBuf>,
    #[command(flatten)]
    pub place: PlacementArgs,
    #[arg(long, value_enum, default_value = "auto")]
    pub engine: EngineKind,
    /// Context size in tokens (the KV cache is allocated for all of it). For the own dense and
    /// MoE builders it is one pool of rows shared by all sequences, not a context per slot as
    /// in llama.cpp's server: one request can use all of it.
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
    #[command(flatten)]
    pub sampling: SamplingArgs,
    /// Requests decoded together (sequences in the KV cache, all builders); `--plan` with the
    /// replanner serves one.
    #[arg(long, default_value_t = tetraeder_scheduler::PARALLEL as u32, value_parser = clap::value_parser!(u32).range(1..))]
    pub parallel: u32,
    /// Tokens per step over all sequences: decoding sequences take theirs, the rest goes to
    /// prompt chunks. Default 2048 (llama-server's n_batch), so prompts that arrive together
    /// share a step. libllama gets it as n_batch, and --ubatch (at most this) as n_ubatch; -b
    /// does not apply to serve.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    pub step_tokens: Option<u32>,
    /// Speculative decoding of a lone greedy request (ADR 0043, 0065): `ngram` (prompt lookup) or
    /// `draft:GGUF` (a small model of the same vocabulary, run by libllama with --ngl and -t).
    /// Off by default; sampled and constrained requests never speculate.
    #[arg(long, value_name = "ngram|draft:GGUF", value_parser = parse_spec)]
    pub spec: Option<SpecArg>,
    /// Sequences may speculate while at most this many are decoding. More than 1 gives up the
    /// bitwise equality with plain decoding that ADR 0043 states for one sequence.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
    pub spec_max_batch: u32,
    /// Most draft tokens per step; the adaptive policy picks up to this many.
    #[arg(long, default_value_t = 6, value_parser = clap::value_parser!(u32).range(0..=6))]
    pub spec_max_draft: u32,
    /// Requests that may wait for the engine; more get 429/529.
    #[arg(long, default_value_t = 16)]
    pub max_queued: usize,
    /// Host RAM for saved sequences (prefix cache), in MiB; 0 turns it off.
    #[arg(long, default_value_t = 8192)]
    pub cache_ram: usize,
    /// Directory for the prefix cache's disk tier: sequences the RAM budget evicts spill here
    /// and come back on a later turn (ADR 0087). The cache removes its files at startup and
    /// shutdown. Needs `--cache-disk-mib`.
    #[arg(long, value_name = "DIR")]
    pub cache_disk: Option<PathBuf>,
    /// Disk budget of the prefix cache's disk tier, in MiB; 0 turns it off.
    #[arg(long, default_value_t = 0)]
    pub cache_disk_mib: usize,
    /// Reuse no prefix: every request prefills from an empty sequence (no idle, busy or RAM
    /// reuse). For measuring what reuse is worth; `--cache-ram` is independent.
    #[arg(long)]
    pub no_prefix_reuse: bool,
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
    /// Run the backend in a child process (`tetraeder worker`) that is restarted after a crash,
    /// a sticky error or a missed deadline (ADR 0069). The queue survives a restart; the requests
    /// that were running fail.
    #[arg(long, conflicts_with_all = ["plan", "fake"])]
    pub worker: bool,
    /// Serve a plan and replan by drain and restart (ADR 0055): `auto` picks one of the
    /// executable candidates for this box and model, or a plan TOML file. Replaces the
    /// placement flags. With `--engine dense` only a plan file is allowed: it runs as written on
    /// the executor (ADR 0069, 0070), without replanning.
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

/// The drafter of `--spec`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpecArg {
    Ngram,
    Draft(PathBuf),
}

fn parse_spec(s: &str) -> Result<SpecArg, String> {
    match s.split_once(':') {
        None if s == "ngram" => Ok(SpecArg::Ngram),
        Some(("draft", p)) if !p.is_empty() => Ok(SpecArg::Draft(p.into())),
        _ => Err(format!("`{s}` is not `ngram` or `draft:<gguf>`")),
    }
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
    // A plan file the executor runs (see [`exec_plan`]) is served as written; every other
    // `--plan` is the replanner's.
    let planned = match &a.plan {
        Some(spec) if exec_plan(a)?.is_none() && a.engine != EngineKind::Dense => {
            Some(replan::start(a, spec, &telemetry, rt.handle().clone())?)
        }
        _ => None,
    };
    let (engine, parallel): (Arc<dyn GatewayEngine>, usize) = match &planned {
        Some(p) => (p.engine.clone(), 1),
        None => build_engine(a, &telemetry)?,
    };
    let served = engine
        .models()
        .first()
        .map(|m| m.id.clone())
        .context("the engine serves no model")?;

    let cfg = GatewayConfig {
        max_concurrent: parallel,
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

/// The engine and the number of requests the gateway lets generate at once.
fn build_engine(
    a: &ServeArgs,
    telemetry: &Arc<Telemetry>,
) -> anyhow::Result<(Arc<dyn GatewayEngine>, usize)> {
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
        return Ok((Arc::new(e), 1));
    }
    let path = a.model.clone().context("-m is required")?;
    let cfg = SchedParts::new(a)?.config(&path, telemetry.metrics().clone());
    let parallel = cfg.parallel;
    let engine = if a.worker {
        // The backend runs in a child process (ADR 0069); the tokenizer stays here.
        let argv = crate::worker::child_args(std::env::args())?;
        let exe = std::env::current_exe().context("cannot find this executable")?;
        let model = path.clone();
        Scheduler::start(cfg, move || {
            let tokenizer = Arc::new(
                Model::load(
                    &model,
                    &ModelParams {
                        vocab_only: true,
                        ..ModelParams::default()
                    },
                )
                .map_err(|e| e.to_string())?,
            );
            let backend = RemoteBackend::start(RemoteConfig::new(exe, argv))?;
            Ok(Loaded {
                backend: Box::new(backend),
                tokenizer,
            })
        })
    } else {
        Scheduler::start(cfg, loader(a, Arc::clone(telemetry))?)
    }
    .map_err(anyhow::Error::msg)?;
    Ok((Arc::new(engine), parallel))
}

/// Loads the backend of `a` in the calling thread (the MoE builder may not be `Send`); shared
/// by `serve` and `worker`.
pub type Loader = Box<dyn FnOnce() -> Result<Loaded, String> + Send>;

pub fn loader(a: &ServeArgs, tel: Arc<Telemetry>) -> anyhow::Result<Loader> {
    let path = a.model.clone().context("-m is required")?;
    let dense = a.engine == EngineKind::Dense;
    let moe = match a.engine {
        EngineKind::Llama | EngineKind::Dense => false,
        EngineKind::Moe => true,
        EngineKind::Auto => {
            tetraeder_scheduler::architecture(&path)
                .map_err(anyhow::Error::msg)
                .context("cannot read the model architecture")?
                == "qwen3moe"
        }
    };
    if moe {
        resolve_overlap(a.ram_slots.is_some(), a.miss, a.io_service, a.overlap)
            .map_err(anyhow::Error::msg)?;
    }
    let exec = exec_plan(a)?;
    let a = a.clone_load_args();
    Ok(Box::new(move || {
        if let (true, Some(topo)) = (moe, &exec) {
            let chunk = tetraeder_scheduler::MOE_CHUNK;
            tetraeder_scheduler::load_moe_sharded(&a.path, topo, &a.moe_options(tel)?, chunk)
        } else if moe {
            tetraeder_scheduler::load_moe(&a.path, &a.moe_options(tel)?)
        } else if let (true, Some(topo)) = (dense, &exec) {
            tetraeder_scheduler::load_dense_sharded(&a.path, topo, &a.dense_options(tel)?)
        } else if dense {
            tetraeder_scheduler::load_dense(&a.path, &a.dense_options(tel)?)
        } else {
            tetraeder_scheduler::load_llama(&a.path, &a.place.placement(), &a.llama, Some(tel))
        }
    }))
}

/// The topology of `--plan FILE` when the executor runs it (ADR 0069): always with
/// `--engine dense`; with `--engine moe` when the plan has more than one rank (a one-GPU MoE plan
/// stays with the replanner and the streaming builder). `None`: not an executor plan.
fn exec_plan(a: &ServeArgs) -> anyhow::Result<Option<tetraeder_plan::Topology>> {
    let Some(spec) = &a.plan else { return Ok(None) };
    let path = a.model.clone().context("-m is required")?;
    let load = || crate::sharded::load_plan(&path, std::path::Path::new(spec), a.hw.as_deref());
    let found = match a.engine {
        EngineKind::Dense => {
            anyhow::ensure!(
                spec != "auto",
                "--engine dense --plan runs a plan file; `auto` is the replanner's (libllama and MoE)"
            );
            load()?
        }
        EngineKind::Moe if spec != "auto" => match load() {
            Ok(p) if p.topology.ranks.len() > 1 => p,
            _ => return Ok(None),
        },
        _ => return Ok(None),
    };
    eprintln!(
        "plan {}: {} ranks, {} stages",
        found.plan_id,
        found.topology.ranks.len(),
        found.topology.stages.len()
    );
    Ok(Some(found.topology))
}

/// The parts of a `SchedulerConfig` that do not change when the engine is reloaded.
pub struct SchedParts {
    alias: Option<String>,
    sampling: tetraeder_scheduler::sample::SampleParams,
    seed: Option<u32>,
    cache_ram_bytes: usize,
    cache_disk_dir: Option<PathBuf>,
    cache_disk_bytes: usize,
    prefix_reuse: bool,
    chat_template: Option<String>,
    parallel: usize,
    step_tokens: usize,
    spec: Option<SpecConfig>,
}

impl SchedParts {
    pub fn new(a: &ServeArgs) -> anyhow::Result<Self> {
        let path = a.model.clone().context("-m is required")?;
        Ok(Self {
            alias: a.alias.clone(),
            sampling: a.sampling.params()?,
            seed: a.sampling.seed(),
            cache_ram_bytes: a.cache_ram << 20,
            cache_disk_dir: a.cache_disk.clone(),
            cache_disk_bytes: a.cache_disk_mib << 20,
            prefix_reuse: !a.no_prefix_reuse,
            parallel: a.parallel as usize,
            step_tokens: a.step_tokens() as usize,
            spec: a.spec_config(),
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
        cfg.sampling = self.sampling.clone();
        cfg.seed = self.seed;
        cfg.metrics = Some(metrics);
        cfg.cache_ram_bytes = self.cache_ram_bytes;
        cfg.cache_disk_dir.clone_from(&self.cache_disk_dir);
        cfg.cache_disk_bytes = self.cache_disk_bytes;
        cfg.prefix_reuse = self.prefix_reuse;
        cfg.chat_template = self.chat_template.clone();
        cfg.parallel = self.parallel;
        cfg.step_tokens = self.step_tokens;
        cfg.spec.clone_from(&self.spec);
        cfg
    }
}

/// What the worker thread needs to load a model.
struct LoadArgs {
    path: PathBuf,
    place: PlacementArgs,
    ctx: u32,
    parallel: usize,
    llama: tetraeder_engine::ContextConfig,
    expert_slots: Option<usize>,
    ram_slots: Option<usize>,
    miss: MissPolicy,
    prefetch: Option<bool>,
    prefetch_depth: u8,
    io_service: Option<bool>,
    overlap: Option<bool>,
}

/// Where the own builders run: `--ngl 0` or a machine without a GPU means the CPU, anything else
/// GPU `--main-gpu`. As libllama, the default `--ngl -1` (all layers) falls back to the CPU with a
/// warning when there is no GPU. An explicit `--ngl` above 0 does not: the user asked for the
/// GPU, and a broken CUDA init must not silently run a large model on the CPU.
fn builder_backend(place: &PlacementArgs, gpus: usize) -> Result<BackendChoice, String> {
    if place.ngl == 0 || gpus == 0 {
        if place.ngl > 0 {
            return Err(format!(
                "--ngl {} asks for the GPU but none was found (use --ngl 0 for the CPU)",
                place.ngl
            ));
        }
        if place.ngl != 0 {
            tracing::warn!(
                "no GPU found, --ngl {} is ignored: running on the CPU",
                place.ngl
            );
        }
        Ok(BackendChoice::Cpu {
            threads: place.threads.and_then(|t| usize::try_from(t).ok()),
        })
    } else {
        Ok(BackendChoice::Gpu {
            index: usize::try_from(place.main_gpu).unwrap_or(0),
        })
    }
}

impl LoadArgs {
    /// The dense builder's KV pool is `--ctx` rows shared by `--parallel` sequences.
    fn dense_options(&self, telemetry: Arc<Telemetry>) -> Result<DenseOptions, String> {
        let backend = builder_backend(&self.place, Backend::gpu_count())?;
        let mut o = DenseOptions::new(backend);
        o.threads = self.place.threads;
        o.n_ctx_pool = self.ctx as usize;
        o.n_seq_max = self.parallel.max(1);
        o.telemetry = Some(telemetry);
        Ok(o)
    }

    /// Placed like the dense builder ([`builder_backend`]).
    fn moe_options(&self, telemetry: Arc<Telemetry>) -> Result<LoadOptions, String> {
        let backend = builder_backend(&self.place, Backend::gpu_count())?;
        let mut o = LoadOptions::new(backend);
        o.cpu_threads = self.place.threads.and_then(|t| usize::try_from(t).ok());
        o.n_ctx = self.ctx as usize;
        o.n_seq_max = self.llama.n_seq_max as usize;
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
    /// The speculation settings of `--spec` and its companions; `None` when it is off.
    pub fn spec_config(&self) -> Option<SpecConfig> {
        let method = match self.spec.as_ref()? {
            SpecArg::Ngram => SpecMethod::Ngram(NgramDrafter::default()),
            SpecArg::Draft(path) => SpecMethod::Draft {
                path: path.clone(),
                n_ctx: self.ctx,
                n_gpu_layers: self.place.ngl,
                threads: self.place.threads.unwrap_or_else(|| {
                    std::thread::available_parallelism().map_or(1, |n| n.get() as i32)
                }),
                p_min: 0.0,
            },
        };
        let mut c = SpecConfig::new(method);
        c.len = DraftLen::Adaptive {
            max: self.spec_max_draft as usize,
        };
        c.max_batch = self.spec_max_batch as usize;
        Some(c)
    }

    /// Tokens per step: `--step-tokens`, else [`STEP_TOKENS`](tetraeder_scheduler::STEP_TOKENS).
    pub fn step_tokens(&self) -> u32 {
        self.step_tokens
            .unwrap_or(tetraeder_scheduler::STEP_TOKENS as u32)
            .max(1)
    }

    /// The libllama context of `serve`: one unified KV pool for `--parallel` sequences and a
    /// batch of exactly one step, run in `--ubatch` pieces.
    pub fn llama_context(&self) -> tetraeder_engine::ContextConfig {
        let step = self.step_tokens();
        tetraeder_engine::ContextConfig {
            n_batch: step,
            n_ubatch: self.place.ubatch.clamp(1, step),
            n_seq_max: self.parallel,
            kv_unified: true,
            ..self.place.context(self.ctx)
        }
    }

    fn clone_load_args(&self) -> LoadArgs {
        LoadArgs {
            path: self.model.clone().unwrap_or_default(),
            place: self.place.clone(),
            ctx: self.ctx,
            parallel: self.parallel as usize,
            llama: self.llama_context(),
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
    fn own_builders_default_to_the_gpu_and_fall_back_to_the_cpu() {
        let cpu = BackendChoice::Cpu { threads: Some(4) };
        let gpu = |index| BackendChoice::Gpu { index };
        let place = |extra: &[&str]| parse(&[&["-t", "4"], extra].concat()).place;
        // The default --ngl -1 on a machine without a GPU, as libllama does.
        let backend = |extra: &[&str], gpus| builder_backend(&place(extra), gpus).unwrap();
        assert_eq!(backend(&[], 0), cpu);
        assert_eq!(backend(&["--ngl", "-1"], 0), cpu);
        assert_eq!(backend(&[], 1), gpu(0));
        assert_eq!(backend(&["--main-gpu", "1"], 2), gpu(1));
        assert_eq!(backend(&["--ngl", "0"], 1), cpu);
        assert_eq!(backend(&["--ngl", "0"], 0), cpu);
        // An explicit --ngl above 0 without a GPU is an error, not a CPU run.
        let err = builder_backend(&place(&["--ngl", "99"]), 0).unwrap_err();
        assert!(err.contains("--ngl 99"), "{err}");
    }

    // hardware: none
    #[test]
    fn parallel_and_step_tokens_shape_the_llama_context() {
        let a = parse(&[]);
        assert_eq!((a.parallel, a.step_tokens()), (4, 2048));
        let c = a.llama_context();
        assert_eq!(
            (c.n_seq_max, c.n_batch, c.n_ubatch, c.kv_unified),
            (4, 2048, 512, true)
        );
        // The phase 4 box passes these.
        let a = parse(&[
            "--engine",
            "llama",
            "--ngl",
            "-1",
            "--parallel",
            "2",
            "--ctx",
            "32768",
            "-t",
            "8",
            "--flash-attn",
            "on",
            "--step-tokens",
            "256",
            "-b",
            "4096",
        ]);
        let c = a.llama_context();
        assert_eq!(
            (c.n_seq_max, c.n_batch, c.n_ubatch, c.n_ctx),
            (2, 256, 256, 32768)
        );
        assert_eq!(c.flash_attn, tetraeder_engine::FlashAttn::On);
        assert!(P::try_parse_from(["x", "--parallel", "0"]).is_err());
        // --ubatch sets the physical batch only.
        let a = parse(&["--ubatch", "128"]);
        assert_eq!((a.step_tokens(), a.llama_context().n_ubatch), (2048, 128));
    }

    // hardware: none
    #[test]
    fn spec_flags_parse_and_default_to_off() {
        assert!(parse(&[]).spec_config().is_none());
        let c = parse(&["--spec", "ngram"]).spec_config().unwrap();
        assert!(matches!(c.method, SpecMethod::Ngram(_)));
        assert_eq!((c.max_batch, c.len), (1, DraftLen::Adaptive { max: 6 }));
        let a = parse(&[
            "--spec",
            "draft:d.gguf",
            "--spec-max-batch",
            "2",
            "--spec-max-draft",
            "3",
            "--ngl",
            "0",
            "-t",
            "2",
        ]);
        assert_eq!(a.spec, Some(SpecArg::Draft("d.gguf".into())));
        let c = a.spec_config().unwrap();
        assert_eq!((c.max_batch, c.len), (2, DraftLen::Adaptive { max: 3 }));
        assert!(matches!(
            c.method,
            SpecMethod::Draft {
                n_gpu_layers: 0,
                threads: 2,
                ..
            }
        ));
        for bad in ["draft:", "ngrams", "draft"] {
            assert!(P::try_parse_from(["x", "--spec", bad]).is_err(), "{bad}");
        }
        assert!(P::try_parse_from(["x", "--spec-max-draft", "7"]).is_err());
    }

    // hardware: none
    #[test]
    fn no_prefix_reuse_with_cache_ram_zero_parses() {
        let a = parse(&[]);
        assert!(!a.no_prefix_reuse);
        let a = parse(&["--engine", "dense", "--no-prefix-reuse", "--cache-ram", "0"]);
        assert!(a.no_prefix_reuse);
        assert_eq!(a.cache_ram, 0);
    }

    // hardware: none
    #[test]
    fn cache_disk_flags_default_off() {
        let a = parse(&[]);
        assert_eq!((a.cache_disk.clone(), a.cache_disk_mib), (None, 0));
        let a = parse(&["--cache-disk", "/tmp/kv", "--cache-disk-mib", "20480"]);
        assert_eq!(a.cache_disk, Some(PathBuf::from("/tmp/kv")));
        assert_eq!(a.cache_disk_mib, 20480);
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
