//! Search, hints, explain, what-if and replanning (ADR 0040).
//!
//! Phase 3a, milestone A: the measurement contract.
//! - [`layout`]: per-tensor bytes of a model, built from a GGUF tensor list.
//! - [`candidate`]: [`enumerate`] the executable candidate plans for one GPU plus the CPU.
//! - [`builder`]: candidates for the own MoE builder (expert slot counts per tier).
//! - [`lower`]: a candidate as llama-bench arguments or `exit_regime` environment.
//! - [`campaign`]: the bench spec that measures the candidates, anchors and baselines.
//! - [`score`]: calibration from the anchors, regret and rank correlation over the candidates.
//!
//! This crate does not depend on ggml, the engine or the model crate.

pub mod builder;
pub mod campaign;
pub mod candidate;
pub mod layout;
pub mod lower;
pub mod score;

pub use builder::{build_builder_candidate, enumerate_builder};
pub use campaign::{Campaign, CampaignVariant, CandidateEntry, CandidatesFile, Group};
pub use candidate::{
    Budget, Candidate, EngineKind, Enumeration, Footprint, Placement, Workload, build_candidate,
    build_plan, enumerate, footprint,
};
pub use layout::{LayerBytes, ModelLayout, ModelMeta, TensorRec, TypeSummary, build_layout};
pub use lower::{range_regex, to_exit_regime_env, to_llama_bench_args, to_variant_args};
