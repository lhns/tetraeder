//! The own executor (ADR 0069): per-rank graphs cut into ranges at collectives and copies,
//! issued in program order from one host thread. Concurrency comes from the device streams.
//!
//! A [`Program`] lists, per rank, a graph and the node indices ending its ranges, and the ops in
//! issue order: compute a range, all-reduce one tensor per rank, copy a tensor to another rank.
//! [`validate`] checks a program (property-tested); [`Executor::run`] issues it.

mod error;
mod exec;
mod fallback;
mod program;
mod ranks;
mod validate;

pub use error::ExecError;
pub use exec::{ExecCounters, Executor};
pub use program::{Op, Program, RankGraph, RankId};
pub use ranks::{RankDevice, Ranks};
pub use validate::validate;
