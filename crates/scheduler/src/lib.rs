//! The serving scheduler (ADR 0046, ADR 0065): one worker thread over a [`BatchBackend`],
//! continuous batching with chunked prefill, prefix reuse and Rust sampling, behind the
//! gateway's `Engine` trait; `spec` holds speculative decoding (ADR 0043).

mod backend;
mod batch;
mod dense_backend;
mod front;
mod load;
mod prefix_cache;
mod prefix_disk;
pub mod sample;
mod scheduler;
mod slot_file;

pub use backend::{
    BackendError, BatchBackend, Feed, LlamaBackend, MOE_CHUNK, MoeBackend, Outputs, SegmentBackend,
    SegmentInfo, SeqId, SeqInput, StepOut,
};
pub use dense_backend::DenseBackend;
pub use front::FrontEngine;
pub use load::{
    Loaded, architecture, chat_template, load_dense, load_dense_sharded, load_llama, load_moe,
    load_moe_chunked, load_moe_sharded, model_id,
};
pub use prefix_cache::{PrefixCache, RESTORE_MIN_GAIN, Restore, SAVE_MIN_TOKENS};
pub use prefix_disk::{ReadFn, Restored};
pub use scheduler::{
    DEFAULT_CACHE_RAM_BYTES, PARALLEL, STEP_TOKENS, Scheduler, SchedulerConfig, build_vocab,
};

pub mod spec;

#[cfg(test)]
extern crate self as tetraeder_scheduler;
