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
use tetraeder_engine::{BufferOverride, CacheType, EmbedConfig, Pooling};
use tetraeder_gateway::engine::Engine as GatewayEngine;
use tetraeder_gateway::fake::{FakeEngine, serve_script};
use tetraeder_gateway::gateway::{Gateway, GatewayConfig};
use tetraeder_gateway::reqlog::RequestLog;
use tetraeder_ggml_sys::ggml::Backend;
use tetraeder_ggml_sys::llama::{Model, ModelParams};
use tetraeder_model::{BackendChoice, DenseOptions, LoadOptions, MissPolicy};
use tetraeder_plan::KvFormat;
use tetraeder_planner::{KvChoice, Placement, Workload, budget_of, pick_for_placement};
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
    /// The defaults under llama-server's names, for `/props` and `generation_settings`.
    pub fn generation_defaults(&self) -> serde_json::Map<String, serde_json::Value> {
        use serde_json::json;
        let mut m = serde_json::Map::new();
        let mut put = |k: &str, v: serde_json::Value| m.insert(k.to_owned(), v);
        put("temperature", json!(self.temp));
        put("top_k", json!(self.top_k));
        put("top_p", json!(self.top_p));
        put("min_p", json!(self.min_p));
        put("typical_p", json!(self.typical));
        put("top_n_sigma", json!(self.top_nsigma));
        put("xtc_probability", json!(self.xtc_probability));
        put("xtc_threshold", json!(self.xtc_threshold));
        put("dynatemp_range", json!(self.dynatemp_range));
        put("dynatemp_exponent", json!(self.dynatemp_exp));
        put("repeat_last_n", json!(self.repeat_last_n));
        put("repeat_penalty", json!(self.repeat_penalty));
        put("presence_penalty", json!(self.presence_penalty));
        put("frequency_penalty", json!(self.frequency_penalty));
        put("dry_multiplier", json!(self.dry_multiplier));
        put("dry_base", json!(self.dry_base));
        put("dry_allowed_length", json!(self.dry_allowed_length));
        put("dry_penalty_last_n", json!(self.dry_penalty_last_n));
        if !self.dry_sequence_breaker.is_empty() {
            put("dry_sequence_breakers", json!(self.dry_sequence_breaker));
        }
        put("mirostat", json!(self.mirostat));
        put("mirostat_tau", json!(self.mirostat_ent));
        put("mirostat_eta", json!(self.mirostat_lr));
        put("ignore_eos", json!(self.ignore_eos));
        if let Ok(seed) = u32::try_from(self.seed) {
            put("seed", json!(seed));
        }
        m
    }

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
    #[arg(short = 'm', long, required_unless_present_any = ["fake", "hf_repo"])]
    pub model: Option<PathBuf>,
    /// llama-server's `-hf USER/MODEL[:QUANT]`: refused, tetraeder does not download models
    /// (ADR 0090). Download the GGUF and pass `-m`.
    #[arg(
        long,
        value_name = "USER/MODEL[:QUANT]",
        env = "LLAMA_ARG_HF_REPO"
    )]
    pub hf_repo: Option<String>,
    /// llama-server's `-hff FILE`, refused with `--hf-repo`.
    #[arg(long, value_name = "FILE", env = "LLAMA_ARG_HF_FILE")]
    pub hf_file: Option<String>,
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
    /// API keys, comma separated: every route but `/health` then needs one in `Authorization`
    /// (with or without `Bearer `) or `X-Api-Key`, as in llama-server.
    #[arg(long, value_name = "KEY", env = "LLAMA_API_KEY")]
    pub api_key: Option<String>,
    /// File with API keys, one per line; lines starting with `#` are comments.
    #[arg(long, value_name = "FILE", env = "LLAMA_ARG_API_KEY_FILE")]
    pub api_key_file: Option<PathBuf>,
    /// Serve Prometheus metrics at `/metrics`, with llama-server's names.
    #[arg(long, env = "LLAMA_ARG_ENDPOINT_METRICS")]
    pub metrics: bool,
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
    /// KV cache format of the own builders (ADR 0050 b, 0086): `f16` (default), `bf16`, `q8_0`,
    /// `q8_0/q4_0`, `q4_0`, or `auto`: the first of f16, q8_0, q8_0/q4_0 that fits this box at
    /// `--ctx`, with the reason on stderr. libllama (`--engine llama`) keeps f16.
    #[arg(long, default_value = "f16", value_name = "FORMAT|auto")]
    pub kv: KvChoice,
    /// KV cache type of K, llama-server's `-ctk` (libllama): f32, f16, bf16, f8_e4m3, q8_0, q4_0,
    /// q4_1, iq4_nl, q5_0 or q5_1. The own builders take the pairs of `--kv` only: a pair of
    /// `-ctk` and `-ctv` that `--kv` allows is mapped onto it, any other is refused.
    #[arg(
        long,
        value_name = "TYPE",
        value_parser = tetraeder_engine::parse_kv_cache_type,
        env = "LLAMA_ARG_CACHE_TYPE_K",
        conflicts_with_all = ["kv", "plan"]
    )]
    pub cache_type_k: Option<CacheType>,
    /// KV cache type of V, llama-server's `-ctv`, as `--cache-type-k`.
    #[arg(
        long,
        value_name = "TYPE",
        value_parser = tetraeder_engine::parse_kv_cache_type,
        env = "LLAMA_ARG_CACHE_TYPE_V",
        conflicts_with_all = ["kv", "plan"]
    )]
    pub cache_type_v: Option<CacheType>,
    /// Keep the KV cache and the attention on the CPU, llama-server's `-nkvo` (libllama; ADR 0084
    /// step 0). The own builders refuse it.
    #[arg(long, overrides_with = "kv_offload", conflicts_with = "plan")]
    pub no_kv_offload: bool,
    /// The default of `--no-kv-offload`; the last of the two wins.
    #[arg(long, overrides_with = "no_kv_offload")]
    pub kv_offload: bool,
    /// When a sequence fills the context, drop tokens after `n_keep` (a request field) and go on,
    /// llama-server's `--context-shift` (ADR 0090). Off by default, as there: the request then
    /// ends with `length`. libllama only, on models whose cache can shift.
    #[arg(long, overrides_with = "no_context_shift")]
    pub context_shift: bool,
    /// The default of `--context-shift`; the last of the two wins.
    #[arg(long, overrides_with = "context_shift")]
    pub no_context_shift: bool,
    /// Directory for slot save files: turns on `POST /slots/{id}?action=save|restore|erase`
    /// (ADR 0090). Slot `i` is sequence `i` of `--parallel`. libllama only.
    #[arg(long, value_name = "PATH", env = "LLAMA_ARG_SLOT_SAVE_PATH", conflicts_with_all = ["plan", "fake"])]
    pub slot_save_path: Option<PathBuf>,
    /// Serve `/embedding`, `/embeddings` and `/v1/embeddings` from a second libllama context on
    /// the model (llama-server's `--embeddings`; ADR 0090). The own builders refuse it.
    #[arg(
        long,
        visible_alias = "embedding",
        env = "LLAMA_ARG_EMBEDDINGS",
        conflicts_with_all = ["plan", "worker", "fake"]
    )]
    pub embeddings: bool,
    /// Serve the rerank routes (llama-server's `--rerank`): the model needs a classification
    /// head, and the pooling is `rank`. libllama only.
    #[arg(
        long,
        visible_alias = "rerank",
        env = "LLAMA_ARG_RERANKING",
        conflicts_with_all = ["plan", "worker", "fake"]
    )]
    pub reranking: bool,
    /// Pooling of the embeddings: none, mean, cls, last or rank; the model's own by default.
    #[arg(long, value_name = "TYPE", env = "LLAMA_ARG_POOLING")]
    pub pooling: Option<Pooling>,
    /// Keep every expert tensor on the CPU, llama-server's `-cmoe` (libllama). The MoE builder
    /// places experts with `--expert-slots` and `--ram-slots` and refuses it.
    #[arg(long, env = "LLAMA_ARG_CPU_MOE", conflicts_with = "plan")]
    pub cpu_moe: bool,
    /// Keep the expert tensors of the first N layers on the CPU, llama-server's `-ncmoe` (libllama).
    #[arg(
        long,
        value_name = "N",
        env = "LLAMA_ARG_N_CPU_MOE",
        conflicts_with = "plan"
    )]
    pub n_cpu_moe: Option<usize>,
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
    anyhow::ensure!(
        a.hf_repo.is_none() && a.hf_file.is_none(),
        "--hf-repo and --hf-file: tetraeder does not download models (no download client, ADR 0090); \
         download the GGUF (for example `hf download <repo> <file>`) and pass it with -m"
    );
    if let Some(dir) = &a.slot_save_path {
        anyhow::ensure!(
            dir.is_dir(),
            "--slot-save-path: not a directory: {}",
            dir.display()
        );
    }
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
        api_keys: a.api_keys()?,
        metrics: a.metrics,
        defaults: a.sampling.generation_defaults(),
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
    a.check_libllama_flags(moe || dense, moe)?;
    let kv = resolve_kv(a, a.kv_choice(moe || dense)?, &path, moe, dense)?;
    let mut a = a.clone_load_args();
    a.kv = kv;
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
            let mut placement = a.place.placement();
            placement.overrides.extend(a.cpu_moe.iter().cloned());
            tetraeder_scheduler::load_llama(&a.path, &placement, &a.llama, Some(tel))
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
    embed: Option<EmbedConfig>,
    context_shift: bool,
    slot_dir: Option<PathBuf>,
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
            embed: a.embed_config()?,
            context_shift: a.context_shift,
            slot_dir: a.slot_save_path.clone(),
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
        cfg.embed.clone_from(&self.embed);
        cfg.context_shift = self.context_shift;
        cfg.slot_dir.clone_from(&self.slot_dir);
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
    kv: KvFormat,
    /// `--cpu-moe` and `--n-cpu-moe`, appended to the placement's overrides.
    cpu_moe: Vec<BufferOverride>,
}

/// The KV format of `--kv` for the engine that will run. `auto` asks the planner against the
/// discovered box (GPU capacity, not free memory, as `--plan` does) and the placement the
/// builder flags give; libllama and the replanner (`--plan`) size and run F16 only.
fn resolve_kv(
    a: &ServeArgs,
    kv_choice: KvChoice,
    path: &Path,
    moe: bool,
    dense: bool,
) -> anyhow::Result<KvFormat> {
    let own = moe || dense;
    if kv_choice.fixed() == Some(KvFormat::default()) {
        return Ok(KvFormat::default());
    }
    if !own || a.plan.is_some() {
        let why = if own {
            "--plan sizes and runs f16 only"
        } else {
            "libllama takes --cache-type-k and --cache-type-v, not --kv"
        };
        anyhow::ensure!(kv_choice == KvChoice::Auto, "--kv {kv_choice}: {why}");
        eprintln!("kv auto: f16 ({why})");
        return Ok(KvFormat::default());
    }
    let Some(kv) = kv_choice.fixed() else {
        let layout = crate::plan::layout(path)?;
        let (hw, _) = replan::source::HwSource::Discover
            .read(false)
            .map_err(anyhow::Error::msg)?;
        let quota = std::thread::available_parallelism().map_or(1, |n| n.get() as u32);
        let budget = budget_of(&hw, quota, None, None);
        let on_cpu = a.place.ngl == 0 || Backend::gpu_count() == 0;
        let placement = match (on_cpu, moe) {
            (true, _) => Placement::Cpu,
            (false, true) => Placement::Builder {
                vram_slots: a.expert_slots.map_or(layout.n_expert, |s| s as u32),
                ram_slots: a.ram_slots.map_or(0, |s| s as u32),
            },
            (false, false) => Placement::Dense {
                gpu_layers: layout.n_layer,
            },
        };
        let w = Workload {
            prompt: a.ctx.saturating_sub(128).max(1),
            r#gen: 128,
        };
        let at = (placement, tetraeder_planner::SERVE_BUILDER_UBATCH);
        let (kv, pick) = pick_for_placement(&budget, &layout, &w, at, kv_choice)
            .map_err(|e| anyhow::anyhow!("--kv auto at --ctx {}: {e}", a.ctx))?;
        if let Some(p) = pick {
            eprintln!("{}", p.explain());
        }
        return Ok(kv);
    };
    Ok(kv)
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
        o.kv = self.kv;
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
        o.kv = self.kv;
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

/// The keys of `--api-key` (comma separated) and `--api-key-file` (one per line, `#` comments),
/// as llama-server reads them.
pub fn read_api_keys(list: Option<&str>, file: Option<&Path>) -> anyhow::Result<Vec<String>> {
    let mut keys: Vec<String> = list
        .into_iter()
        .flat_map(|l| l.split(','))
        .filter(|k| !k.is_empty())
        .map(str::to_owned)
        .collect();
    if let Some(f) = file {
        let text = std::fs::read_to_string(f)
            .with_context(|| format!("cannot read API key file {}", f.display()))?;
        keys.extend(
            text.lines()
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .map(str::to_owned),
        );
    }
    Ok(keys)
}

impl ServeArgs {
    fn api_keys(&self) -> anyhow::Result<Vec<String>> {
        read_api_keys(self.api_key.as_deref(), self.api_key_file.as_deref())
    }

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
            offload_kqv: !self.no_kv_offload,
            type_k: self
                .cache_type_k
                .unwrap_or(tetraeder_engine::CacheType::F16),
            type_v: self
                .cache_type_v
                .unwrap_or(tetraeder_engine::CacheType::F16),
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
            kv: KvFormat::default(),
            cpu_moe: tetraeder_engine::cpu_moe_overrides(self.cpu_moe, self.n_cpu_moe),
        }
    }

    /// The embedding context of `--embeddings` and `--reranking`; `None` serves no embeddings.
    /// `--reranking` is the `rank` pooling, so a `--pooling` that says otherwise is an error.
    fn embed_config(&self) -> anyhow::Result<Option<EmbedConfig>> {
        if !self.embeddings && !self.reranking {
            if let Some(p) = self.pooling {
                eprintln!("--pooling {p:?} has no effect without --embeddings or --reranking");
            }
            return Ok(None);
        }
        let pooling = match (self.reranking, self.pooling) {
            (true, Some(p)) if p != Pooling::Rank => {
                anyhow::bail!("--reranking uses the rank pooling, not --pooling {p:?}")
            }
            (true, _) => Pooling::Rank,
            (false, Some(p)) => p,
            (false, None) => Pooling::Unspecified,
        };
        Ok(Some(EmbedConfig {
            pooling,
            n_batch: self.step_tokens(),
            threads: self.place.threads,
            flash_attn: self.place.flash_attn,
        }))
    }

    /// `--kv`, or the pair of `-ctk` and `-ctv` mapped onto it for the own builders (`own`): an
    /// unset side is f16. libllama gets the two types as they are (see [`Self::llama_context`]).
    fn kv_choice(&self, own: bool) -> anyhow::Result<KvChoice> {
        if !own || (self.cache_type_k.is_none() && self.cache_type_v.is_none()) {
            return Ok(self.kv);
        }
        let name = |t: Option<CacheType>| t.map_or_else(|| "f16".to_owned(), |t| t.name());
        let pair = format!("{}/{}", name(self.cache_type_k), name(self.cache_type_v));
        pair.parse::<KvChoice>().map_err(|_| {
            anyhow::anyhow!(
                "--cache-type-k/--cache-type-v {pair}: the own builders run these allowed pairs \
                 only: {}; use --engine llama for other types",
                KvFormat::ALLOWED.map(|f| f.to_string()).join(", ")
            )
        })
    }

    /// llama-server flags the own builders (`own`; `moe` for the MoE one) cannot honour, refused
    /// with a pointer to what works (ADR 0090). libllama honours all of them.
    fn check_libllama_flags(&self, own: bool, moe: bool) -> anyhow::Result<()> {
        if !own {
            return Ok(());
        }
        anyhow::ensure!(
            !self.no_kv_offload,
            "--no-kv-offload: the own builders keep the KV cache on the GPU that runs them \
             (ADR 0084 plans host KV); use --engine llama"
        );
        anyhow::ensure!(
            !self.context_shift,
            "--context-shift: the own builders' KV pool cannot shift positions; use --engine llama"
        );
        anyhow::ensure!(
            self.slot_save_path.is_none(),
            "--slot-save-path: slot save and restore needs libllama's sequence state; use --engine llama"
        );
        anyhow::ensure!(
            !self.embeddings && !self.reranking,
            "--embeddings and --reranking: only libllama serves embeddings and reranking; use --engine llama"
        );
        if self.cpu_moe || self.n_cpu_moe.is_some() {
            anyhow::bail!(
                "--cpu-moe and --n-cpu-moe: {}; use --engine llama for llama-server's tensor overrides",
                if moe {
                    "the MoE builder places experts with --expert-slots and --ram-slots (--miss cpu \
                     computes the ones outside the GPU slots on the CPU)"
                } else {
                    "the dense builder has no expert tensors"
                }
            );
        }
        Ok(())
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
    fn api_keys_come_from_the_list_and_the_file() {
        let dir = std::env::temp_dir().join(format!("tetraeder-keys-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("keys.txt");
        std::fs::write(&file, "# comment\nk3\n\nk4\n").unwrap();
        let keys = read_api_keys(Some("k1,,k2"), Some(&file)).unwrap();
        assert_eq!(keys, ["k1", "k2", "k3", "k4"]);
        assert!(read_api_keys(None, None).unwrap().is_empty());
        assert!(read_api_keys(None, Some(&dir.join("missing"))).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    // hardware: none
    #[test]
    fn generation_defaults_report_the_sampling_flags() {
        let d = parse(&["--temp", "0.5", "--top-k", "7", "--seed", "9"])
            .sampling
            .generation_defaults();
        assert_eq!(d["top_k"], 7);
        assert_eq!(d["seed"], 9);
        assert!((d["temperature"].as_f64().unwrap() - 0.5).abs() < 1e-6);
        // a random seed is not a default value to report
        assert!(
            !parse(&[])
                .sampling
                .generation_defaults()
                .contains_key("seed")
        );
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

    fn parse_llama(extra: &[&str]) -> ServeArgs {
        let argv = std::iter::once("x")
            .chain(["-m", "m.gguf"])
            .chain(extra.iter().copied());
        let argv = crate::llama_args::rewrite(argv.map(std::ffi::OsString::from));
        P::parse_from(argv).serve
    }

    // hardware: none
    #[test]
    fn kv_offload_and_cache_types_reach_the_libllama_context() {
        let c = parse(&[]).llama_context();
        assert!(c.offload_kqv);
        assert_eq!(
            (c.type_k.name(), c.type_v.name()),
            ("f16".into(), "f16".into())
        );
        let c = parse_llama(&["-nkvo", "-ctk", "q8_0", "-ctv", "q4_0"]).llama_context();
        assert!(!c.offload_kqv);
        assert_eq!(
            (c.type_k.name(), c.type_v.name()),
            ("q8_0".into(), "q4_0".into())
        );
        // the last of the pair wins
        assert!(parse_llama(&["-nkvo", "-kvo"]).llama_context().offload_kqv);
        assert!(
            !parse_llama(&["-kvo", "--no-kv-offload"])
                .llama_context()
                .offload_kqv
        );
        // names are llama-server's: exact, lower case
        assert!(P::try_parse_from(["x", "-m", "m", "--cache-type-k", "Q8_0"]).is_err());
        assert!(P::try_parse_from(["x", "-m", "m", "--cache-type-v", "q2_k"]).is_err());
    }

    // hardware: none
    #[test]
    fn own_builders_map_a_cache_type_pair_onto_kv_or_refuse_it() {
        let own = |extra: &[&str]| parse_llama(extra).kv_choice(true);
        assert_eq!(own(&[]).unwrap(), KvChoice::Fixed(KvFormat::F16));
        assert_eq!(
            own(&["-ctk", "q8_0", "-ctv", "q8_0"]).unwrap(),
            KvChoice::Fixed(KvFormat::Q8_0)
        );
        assert_eq!(
            own(&["-ctk", "q8_0", "-ctv", "q4_0"]).unwrap(),
            KvChoice::Fixed(KvFormat::Q8_0_Q4_0)
        );
        // one side alone leaves the other at f16: q8_0/f16 is not an allowed pair
        let e = own(&["-ctk", "q8_0"]).unwrap_err().to_string();
        assert!(
            e.contains("allowed pairs") && e.contains("q8_0/q4_0"),
            "{e}"
        );
        let e = own(&["-ctk", "q5_0", "-ctv", "q5_0"])
            .unwrap_err()
            .to_string();
        assert!(e.contains("--engine llama"), "{e}");
        // libllama takes the types as they are
        assert_eq!(
            parse_llama(&["-ctk", "q5_0"]).kv_choice(false).unwrap(),
            KvChoice::Fixed(KvFormat::F16)
        );
        // --kv and the types name the same thing twice; the default --kv is no conflict
        assert!(P::try_parse_from(["x", "-m", "m", "--kv", "q8_0", "-ctk", "q8_0"]).is_err());
        assert!(P::try_parse_from(["x", "-m", "m", "--cache-type-k", "q8_0"]).is_ok());
    }

    // hardware: none
    #[test]
    fn own_builders_refuse_what_only_libllama_honours() {
        let refused = |extra: &[&str], moe: bool| {
            parse_llama(extra)
                .check_libllama_flags(true, moe)
                .unwrap_err()
                .to_string()
        };
        assert!(refused(&["-nkvo"], true).contains("own builders keep the KV cache"));
        let e = refused(&["-cmoe"], true);
        assert!(
            e.contains("--expert-slots") && e.contains("--ram-slots"),
            "{e}"
        );
        assert!(refused(&["-ncmoe", "2"], false).contains("--engine llama"));
        for extra in [&[][..], &["-nkvo"], &["-cmoe"], &["-ncmoe", "1"]] {
            assert!(
                parse_llama(extra)
                    .check_libllama_flags(false, false)
                    .is_ok()
            );
        }
        assert!(parse_llama(&[]).check_libllama_flags(true, true).is_ok());
    }

    // hardware: none
    #[test]
    fn cpu_moe_flags_become_tensor_overrides() {
        let o = |extra: &[&str]| parse_llama(extra).clone_load_args().cpu_moe;
        assert!(o(&[]).is_empty());
        assert_eq!(o(&["-cmoe"]).len(), 1);
        assert_eq!(o(&["-ncmoe", "3"]).len(), 3);
        assert_eq!(o(&["-cmoe", "-ncmoe", "2"]).len(), 3);
        assert!(P::try_parse_from(["x", "-m", "m", "--n-cpu-moe", "-1"]).is_err());
    }

    // hardware: none
    #[test]
    fn hf_repo_needs_no_model_path_and_is_refused_by_run() {
        let a = P::try_parse_from(["x", "-hf", "a/b:Q4_K_M"]);
        assert!(
            a.is_err(),
            "-hf is a short flag cluster until it is rewritten"
        );
        let argv = crate::llama_args::rewrite(
            ["x", "-hf", "a/b:Q4_K_M", "-hff", "f.gguf"].map(std::ffi::OsString::from),
        );
        let a = P::parse_from(argv).serve;
        assert_eq!(
            (a.hf_repo.as_deref(), a.hf_file.as_deref()),
            (Some("a/b:Q4_K_M"), Some("f.gguf"))
        );
        let e = run(&a).unwrap_err().to_string();
        assert!(e.contains("download the GGUF") && e.contains("-m"), "{e}");
    }

    // hardware: none
    #[test]
    fn overlap_without_the_service_is_an_error() {
        use MissPolicy::Cpu;
        assert!(resolve_overlap(true, Cpu, Some(false), Some(true)).is_err());
        assert!(resolve_overlap(false, Cpu, None, Some(true)).is_err());
    }
}
