//! GGUF loading and architecture graph builders (no `unsafe`; raw ggml stays in `ggml-sys`).
//! What gets placed: the model as units with sizes (docs/plan.md, "The placement plan").
//!
//! Phase 2: the Qwen3 MoE builder of ADR 0030 ([`Qwen3Moe`]).

pub mod convert;
pub mod cpu_poll;
pub mod experts;
pub mod gguf;
pub mod gguf_write;
pub mod hparams;
pub mod qwen3moe;

pub use convert::{ConvertOptions, ConvertStats};
pub use cpu_poll::CpuPoll;
pub use experts::{ExpertStats, IoCounters, LayerExpertStats, MissPolicy, RoutingStep};
pub use gguf::{Gguf, GgufError, Slab, TensorInfo};
pub use hparams::{ExpertSlabs, Hparams, ModelFile};
pub use qwen3moe::{BackendChoice, LoadOptions, MemoryUse, ModelError, Qwen3Moe, StepProfile};
pub use tetraeder_memory::slots::LayerStats;
