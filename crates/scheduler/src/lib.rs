//! The serving scheduler (ADR 0046): one worker thread over a [`Backend`], prefix reuse of the
//! last sequence and Rust sampling, behind the gateway's `Engine` trait. Phase 4 replaces the
//! loop with continuous batching, chunked prefill and priorities; `spec` holds speculative decoding
//! (ADR 0043).

mod backend;
// The step planner of ADR 0065; the loop that uses it comes with C2b.
#[allow(dead_code)]
mod batch;
mod dense_backend;
mod load;
mod prefix_cache;
pub mod sample;
mod scheduler;

pub use backend::{
    Backend, BackendError, BatchBackend, LlamaBackend, MOE_CHUNK, MoeBackend, Outputs, SeqId,
    SeqInput, SingleSeq,
};
pub use dense_backend::{DenseBackend, LoadedDense, load_dense};
pub use load::{
    Loaded, architecture, chat_template, load_llama, load_moe, load_moe_chunked, model_id,
};
pub use prefix_cache::{PrefixCache, RESTORE_MIN_GAIN, SAVE_MIN_TOKENS};
pub use scheduler::{
    DEFAULT_CACHE_RAM_BYTES, PREFILL_CHUNK, Scheduler, SchedulerConfig, build_vocab,
};

pub mod spec;
