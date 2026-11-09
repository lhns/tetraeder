//! `serve --plan`: a plan from a file or the executable enumerators, lowered to the engine's
//! load options, served behind a gate, and replanned by drain and restart (ADR 0055, ADR 0042).

pub mod admin;
#[cfg(test)]
mod closed_loop;
pub mod control;
pub mod engine;
pub mod glue;
pub mod monitor;
pub mod source;
pub mod start;

pub use start::start;
