//! The serving scheduler (ADR 0046): one worker thread over a [`Backend`], prefix reuse of the
//! last sequence and Rust sampling, behind the gateway's `Engine` trait. Phase 4 replaces the
//! loop with continuous batching, chunked prefill and priorities.

mod backend;
mod load;
pub mod sample;
mod scheduler;

pub use backend::{Backend, LlamaBackend, MOE_CHUNK, MoeBackend};
pub use load::{Loaded, architecture, chat_template, load_llama, load_moe, model_id};
pub use scheduler::{PREFILL_CHUNK, Scheduler, SchedulerConfig, build_vocab};
