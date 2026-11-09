//! Per-device arenas, memory tiers, expert slot cache and direct I/O (ADR 0031).
//!
//! - [`arena`]: one backend buffer per (device, purpose), sized from the expanded plan.
//! - [`reserve`]: the per-device reserve, measured at startup.
//! - [`slots`]: host-side expert slot cache (pinned hot set, LRU, phase budgets).
//! - [`dio`]: aligned direct reads of file ranges.
//! - [`reader`]: a worker thread running those reads in the background (ADR 0038).
//! - [`io`]: a persistent pool of read workers with per-expert completion (ADR 0054).
//! - [`replay`]: routing traces and the offline replay of the RAM tier (ADR 0050).
//! - [`metrics`]: the catalog metrics for all of the above.
//!
//! `unsafe` lives in `dio` (aligned buffers, raw I/O), `reader` (raw host destinations, also used by `io`) and `ggml`
//! (raw backend calls), each
//! block with a safety comment.

pub mod arena;
pub mod dio;
pub mod fake;
#[cfg(feature = "ggml")]
pub mod ggml;
pub mod io;
#[cfg(feature = "metrics")]
pub mod metrics;
pub mod process;
pub mod reader;
pub mod replay;
pub mod reserve;
pub mod slots;

pub use arena::{
    Allocator, Arena, ArenaBuffer, ArenaError, Arenas, Phase, Purpose, arena_sizes,
    available_for_arenas, phase_bytes,
};
pub use dio::{AlignedBuf, DirectFile, Options as IoOptions, Range, ReadReq, ReadStats};
pub use reserve::{ReserveReport, measure_reserve, reserve_map};
pub use slots::{Change, LayerSlots, Lookup, Move, SlotCache, Tier};
