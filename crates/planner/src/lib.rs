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
//! Phase 3a, milestone B: the search.
//! - [`quality`]: the [`QualityModel`] trait, null and table models.
//! - [`profile_quality`]: the measured profile as a quality model, its correction, and the mix at a byte budget.
//! - [`objective`]: [`Scored`], [`choose`] (limits, quality, cost, speed; Pareto front, tradeoff).
//! - [`search`]: skeletons, pruning, layer DP, expert tiers, format knapsack, scoring.
//!
//! Phase 3a, milestone C: explain, diff, what-if.
//! - [`explain`]: why the search chose its plan (binding resource, runner-up, formats, front).
//! - [`diff`]: [`plan_diff`], what moves between two plans and how long the switch takes.
//! - [`whatif`]: the priors database, hardware changes, [`what_if`] and [`rank_purchases`].
//! - [`cli`]: the `explain`, `diff` and `what-if` subcommands as functions.
//!
//! This crate does not depend on ggml, the engine or the model crate.

pub mod builder;
pub mod campaign;
pub mod candidate;
pub mod cli;
pub mod diff;
pub mod explain;
pub mod layout;
pub mod lower;
pub mod objective;
pub mod profile_quality;
pub mod quality;
pub mod score;
pub mod search;
pub mod serve;
pub mod whatif;

pub use builder::{build_builder_candidate, enumerate_builder};
pub use campaign::{Campaign, CampaignVariant, CandidateEntry, CandidatesFile, Group};
pub use candidate::{
    Budget, Candidate, EngineKind, Enumeration, Footprint, Placement, Workload, build_candidate,
    build_plan, enumerate, enumerate_cpu, footprint,
};
pub use diff::{PlanSwitch, plan_diff};
pub use explain::{Explanation, explain};
pub use layout::{LayerBytes, ModelLayout, ModelMeta, TensorRec, TypeSummary, build_layout};
pub use lower::{
    Applies, Lowered, SERVE_BUILDER_UBATCH, ServeEngine, llama_args, lower, range_regex, recognize,
    to_exit_regime_env, to_llama_bench_args, to_variant_args,
};
pub use objective::{Choice, SPEED_BAND, Scored, StageTime, choose, pareto_front, resource_cost};
pub use profile_quality::{
    Band, Correction, Mix, MixError, ProfileQuality, estimate_mix, estimate_mix_scaled,
    group_of_tensor, mix_at_budget, tensor_formats,
};
pub use quality::{BitsQuality, NullQuality, QualityModel, TableQuality};
pub use search::{SearchInput, SearchResult, Split, search, split_layers};
pub use serve::{
    Picked, Proposed, ServeInput, budget_of, candidates, correct_for_drift, fill_knobs, pick,
    predict_plan, propose, regret,
};
pub use whatif::{Change, Priors, WhatIf, apply, fill_from_priors, rank_purchases, what_if};
