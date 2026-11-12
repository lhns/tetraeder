//! How a decode admission (ADR 0059) copies an expert from its RAM slot into its compute slot,
//! and the barrier before that slot is read (addendum 2026-11-12b).

use crate::qwen3moe::Result;
use std::rc::Rc;
use std::time::{Duration, Instant};
use tetraeder_ggml_sys::ggml::{Backend, Context, OwnedTensor, Tensor, Type};

/// `LoadOptions::admit_upload`: how an admitted expert reaches its compute slot.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AdmitUpload {
    /// Blocking copy on the main thread, straight from the RAM slot.
    #[default]
    Sync,
    /// The main thread copies the RAM slot into a pinned staging ring and enqueues the upload
    /// on a stream of its own; the slot is read only after that stream is synchronized.
    Async,
    /// Test only: every copy waits for the barrier, and the destination slot holds `0xff`
    /// bytes (a NaN scale in every K-quant block) until then, so a read before the barrier
    /// shows in the logits.
    #[doc(hidden)]
    Deferred,
}

/// Experts the staging ring holds. The ring is reused only after a stream sync; with this many
/// entries and DMA faster than the host copy into the ring, the sync finds the stream idle.
pub(crate) const RING_EXPERTS: usize = 8;

/// Which ring entry the next copy uses, and when the upload stream must be synchronized
/// first: an entry is handed out again only after a sync.
#[derive(Debug, Clone)]
pub(crate) struct RingCursor {
    entries: usize,
    next: usize,
    /// Entries handed out since the last sync.
    used: usize,
}

impl RingCursor {
    pub(crate) fn new(entries: usize) -> Self {
        Self {
            entries: entries.max(1),
            next: 0,
            used: 0,
        }
    }

    /// The next entry, and whether the stream must be synchronized before it is written.
    pub(crate) fn take(&mut self) -> (usize, bool) {
        let sync = self.used == self.entries;
        if sync {
            self.used = 0;
        }
        let e = self.next;
        self.next = (self.next + 1) % self.entries;
        self.used += 1;
        (e, sync)
    }

    /// The stream was synchronized: every entry is free.
    pub(crate) fn synced(&mut self) {
        self.used = 0;
    }
}

/// A copy held by [`AdmitUpload::Deferred`] until the barrier.
struct Job {
    dst: OwnedTensor,
    offset: usize,
    bytes: Vec<u8>,
}

enum Mode {
    Sync,
    Async {
        stream: Backend,
        ring: OwnedTensor,
        entry_bytes: usize,
        cursor: RingCursor,
    },
    Deferred(Vec<Job>),
}

pub(crate) struct Uploader {
    mode: Mode,
    /// Per layer: copies into its compute slots may still be in flight.
    pending: Vec<bool>,
    /// `Async`: the ring is in the device's pinned host memory.
    pinned: Option<bool>,
}

impl Uploader {
    /// `entry_bytes` is the largest expert (gate, up and down) of any layer; `primary` is the
    /// backend that owns the compute slots.
    pub(crate) fn new(
        mode: AdmitUpload,
        primary: &Backend,
        layers: usize,
        entry_bytes: usize,
    ) -> Result<Self> {
        let (mode, pinned) = match mode {
            AdmitUpload::Sync => (Mode::Sync, None),
            AdmitUpload::Deferred => (Mode::Deferred(Vec::new()), None),
            AdmitUpload::Async => {
                let ctx = Rc::new(Context::new(1, 0)?);
                let ring = ctx.new_persistent(Type::I8, &[(RING_EXPERTS * entry_bytes) as i64]);
                ring.get().set_name("admit_staging");
                let (bytes, pinned) = ctx.alloc_host_staging(primary)?;
                tracing::info!(bytes, pinned, "admission staging ring");
                let mode = Mode::Async {
                    stream: primary.sibling()?,
                    ring,
                    entry_bytes,
                    cursor: RingCursor::new(RING_EXPERTS),
                };
                (mode, Some(pinned))
            }
        };
        Ok(Self {
            mode,
            pending: vec![false; layers],
            pinned,
        })
    }

    /// `Async`: whether the staging ring is pinned; `None` in the other modes.
    pub(crate) fn pinned(&self) -> Option<bool> {
        self.pinned
    }

    /// The copy finishes inside [`Uploader::copy`], so its time is the transfer's.
    pub(crate) fn timed(&self) -> bool {
        matches!(self.mode, Mode::Sync)
    }

    pub(crate) fn pending(&self, il: usize) -> bool {
        self.pending[il]
    }

    pub(crate) fn any_pending(&self) -> bool {
        self.pending.iter().any(|&p| p)
    }

    /// Copies slot `rslot` of the RAM tensors `src` (gate, up, down, one expert's slabs of
    /// `sizes` bytes each) into slot `cslot` of the compute tensors `dst` of layer `il`. Returns
    /// the time it waited for a ring entry.
    pub(crate) fn copy(
        &mut self,
        il: usize,
        src: &[Tensor<'_>; 3],
        rslot: u32,
        dst: &[OwnedTensor; 3],
        cslot: u32,
        sizes: &[usize; 3],
    ) -> Result<Duration> {
        let mut waited = Duration::ZERO;
        match &mut self.mode {
            Mode::Sync => {
                for ((s, d), &size) in src.iter().zip(dst).zip(sizes) {
                    s.copy_to(rslot as usize * size, d.get(), cslot as usize * size, size)?;
                }
            }
            Mode::Async {
                stream,
                ring,
                entry_bytes,
                cursor,
            } => {
                let (entry, sync) = cursor.take();
                if sync {
                    let t = Instant::now();
                    stream.synchronize();
                    waited = t.elapsed();
                    // Every copy issued so far has finished.
                    self.pending.fill(false);
                }
                // Before the first enqueue: a failure halfway leaves copies in flight.
                self.pending[il] = true;
                let mut at = entry * *entry_bytes;
                for ((s, d), &size) in src.iter().zip(dst).zip(sizes) {
                    s.copy_to(rslot as usize * size, ring.get(), at, size)?;
                    ring.get()
                        .copy_to_async(at, stream, d.get(), cslot as usize * size, size)?;
                    at += size;
                }
            }
            Mode::Deferred(jobs) => {
                self.pending[il] = true;
                for ((s, d), &size) in src.iter().zip(dst).zip(sizes) {
                    let offset = cslot as usize * size;
                    let bytes = s.get_bytes(rslot as usize * size, size)?;
                    d.get().set_bytes(&vec![0xffu8; size], offset)?;
                    jobs.push(Job {
                        dst: d.clone(),
                        offset,
                        bytes,
                    });
                }
            }
        }
        Ok(waited)
    }

    /// The barrier: every copy issued so far has landed when it returns. Returns the time it
    /// waited (zero when nothing was pending).
    pub(crate) fn settle(&mut self) -> Result<Duration> {
        if !self.any_pending() {
            return Ok(Duration::ZERO);
        }
        let t = Instant::now();
        match &mut self.mode {
            Mode::Sync => {}
            Mode::Async { stream, cursor, .. } => {
                stream.synchronize();
                cursor.synced();
            }
            Mode::Deferred(jobs) => {
                for j in jobs.drain(..) {
                    j.dst.get().set_bytes(&j.bytes, j.offset)?;
                }
            }
        }
        self.pending.fill(false);
        Ok(t.elapsed())
    }
}

impl Drop for Uploader {
    fn drop(&mut self) {
        // The ring and the compute slots must not be freed under a copy in flight.
        if let Mode::Async { stream, .. } = &self.mode {
            stream.synchronize();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RingCursor;
    use proptest::prelude::*;

    proptest! {
        /// An entry is never handed out while a copy from it may be in flight: between two
        /// hand-outs of the same entry the stream was synchronized (by `take` asking for it or
        /// by a barrier in between).
        #[test]
        fn ring_never_reuses_an_entry_in_flight(
            entries in 1usize..10,
            ops in proptest::collection::vec(any::<bool>(), 1..200),
        ) {
            let mut c = RingCursor::new(entries);
            let mut in_flight = vec![false; entries];
            for barrier in ops {
                if barrier {
                    c.synced();
                    in_flight.fill(false);
                    continue;
                }
                let (e, sync) = c.take();
                if sync {
                    in_flight.fill(false);
                }
                prop_assert!(!in_flight[e], "entry {} reused in flight", e);
                in_flight[e] = true;
            }
        }
    }

    /// Without barriers the ring asks for a sync exactly every `entries` copies.
    #[test]
    fn ring_syncs_once_per_lap() {
        let mut c = RingCursor::new(3);
        let syncs: Vec<bool> = (0..7).map(|_| c.take().1).collect();
        assert_eq!(syncs, [false, false, false, true, false, false, true]);
    }
}
