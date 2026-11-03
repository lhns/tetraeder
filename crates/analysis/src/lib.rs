//! Bottleneck detector and improvement advisor (Phase 3a, ADR 0041, report only).
//!
//! - [`observation`]: the one input type, written by the simulator and by the box sampler;
//! - [`detect`]: observation to [`detect::Report`] (faults, limits, verdict);
//! - [`advise`]: report to ranked [`advise::Action`]s, nothing applied;
//! - [`simfault`]: synthetic observations with injected faults, for tests of the detector.

pub mod advise;
pub mod detect;
pub mod observation;
pub mod simfault;
