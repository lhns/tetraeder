//! Expert reads that run beside compute (ADR 0054), and the virtual clock that tests them
//! without sleeping (ADR 0056).
//!
//! - [`queue`]: the earliest-deadline-first chunk queue and unit bookkeeping. Pure: no threads,
//!   no clock; the caller passes `now` in seconds.
//! - [`disk`]: a processor-sharing disk model over a measured queue-depth curve.
//! - [`service`]: [`IoService`], worker threads driving a [`Queue`] on the real clock.
//! - [`probe`]: the disk's read rate by queue depth through the service (`tetraeder-bench disk-curve`).
//! - `sim` (tests, feature `sim`): [`sim::SimIo`], the same queue driven by a [`disk::Disk`] and
//!   a virtual clock.
//!
//! The expert streamer holds a `Box<dyn Io>`, so production uses the threads and tests use the
//! virtual clock.

pub mod disk;
pub mod probe;
pub mod queue;
pub mod service;
#[cfg(any(test, feature = "sim"))]
pub mod sim;

pub use disk::Disk;
pub use queue::{Chunk, Deadline, Dst, Part, Queue, UnitDone, UnitId};
pub use service::{DEFAULT_CHUNK, DEFAULT_WORKERS, IoService, ServiceOptions};

use std::io;
use std::path::PathBuf;
use std::sync::Arc;

/// Builds the reads for the shard files of a model: the way tests put a simulator in place of
/// the threaded service.
pub type IoFactory = Arc<dyn Fn(&[PathBuf]) -> io::Result<Box<dyn Io>> + Send + Sync>;

/// What the expert streamer needs from the reads: queue units, collect the landed ones.
pub trait Io: Send {
    /// Queues the chunks of `parts` as one unit. A part whose destination does not match its
    /// range makes the unit fail at once (see the waits).
    fn submit(&self, deadline: Deadline, parts: Vec<Part>) -> UnitId;
    /// The completed units among `units`, without blocking; forgets them.
    fn poll(&self, units: &[UnitId]) -> Vec<(UnitId, io::Result<UnitDone>)>;
    /// Blocks until at least one of `units` has completed; returns every completed one among
    /// them and forgets them. Unknown (already collected) ids are ignored; with none known it
    /// returns at once.
    fn wait_any(&self, units: &[UnitId]) -> Vec<(UnitId, io::Result<UnitDone>)>;
    /// Blocks until every known unit of `units` has completed; returns and forgets them.
    fn wait_all(&self, units: &[UnitId]) -> Vec<(UnitId, io::Result<UnitDone>)>;
    /// Chunks queued or running.
    fn inflight(&self) -> usize;
    /// Bytes of the workers' bounce buffers; none for a simulator.
    fn bounce_bytes(&self) -> u64 {
        0
    }
}
