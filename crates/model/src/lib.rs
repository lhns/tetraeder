//! GGUF loading and architecture graph builders (no `unsafe`; raw ggml stays in `ggml-sys`).
//! What gets placed: the model as units with sizes (docs/plan.md, "The placement plan").
//!
//! Phase 2: the Qwen3 MoE builder of ADR 0030 ([`Qwen3Moe`]).

pub mod gguf;
pub mod hparams;
pub mod qwen3moe;

pub use gguf::{Gguf, GgufError, TensorInfo};
pub use hparams::{ExpertSlabs, Hparams, ModelFile};
pub use qwen3moe::{BackendChoice, LoadOptions, ModelError, Qwen3Moe};
