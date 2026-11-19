//! The own executor (ADR 0069): per-rank graphs cut into ranges at collectives and copies,
//! issued in program order from one host thread. Concurrency comes from the device streams.
//!
//! A [`Program`] lists, per rank, a graph and the node indices ending its ranges, and the ops in
//! issue order: compute a range, all-reduce one tensor per rank, copy a tensor to another rank.
//! [`validate`] checks a program (property-tested); [`Executor::run`] issues it. A pipelined
//! prefill (ADR 0077) puts several micro-batches in one graph per rank and issues them in
//! [`gpipe_sort`] order.

mod error;
mod exec;
mod fallback;
mod program;
mod ranks;
mod schedule;
mod trace;
mod validate;

pub use error::ExecError;
pub use exec::{ExecCounters, Executor};
pub use program::{Op, Program, RankGraph, RankId};
pub use ranks::{RankDevice, Ranks};
pub use schedule::gpipe_sort;
pub use trace::ExecOptions;
pub use validate::validate;
