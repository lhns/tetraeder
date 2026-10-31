//! Per-device arenas, memory tiers, expert slot cache and direct I/O (ADR 0031).
//!
//! - [`arena`]: one backend buffer per (device, purpose), sized from the expanded plan.
//! - [`reserve`]: the per-device reserve, measured at startup.
//! - [`slots`]: host-side expert slot cache (pinned hot set, LRU, phase budgets).
//! - [`dio`]: aligned direct reads of file ranges.
//! - [`metrics`]: the catalog metrics for all of the above.
//!
//! `unsafe` lives in `dio` (aligned buffers, raw I/O) and `ggml` (raw backend calls), each
//! block with a safety comment.

pub mod arena;
pub mod dio;
pub mod fake;
#[cfg(feature = "ggml")]
pub mod ggml;
#[cfg(feature = "metrics")]
pub mod metrics;
pub mod reserve;
pub mod slots;

pub use arena::{
    Allocator, Arena, ArenaBuffer, ArenaError, Arenas, Phase, Purpose, arena_sizes,
    available_for_arenas, phase_bytes,
};
pub use dio::{AlignedBuf, DirectFile, Options as IoOptions, ReadReq, ReadStats};
pub use reserve::{ReserveReport, measure_reserve, reserve_map};
pub use slots::{Change, LayerSlots, Lookup, Move, SlotCache, Tier};
