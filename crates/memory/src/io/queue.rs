//! The chunk queue of the I/O service without threads or time (ADR 0056).
//!
//! A unit (one expert: its slabs) is split into chunks of at most `chunk` bytes. Chunks pop
//! earliest deadline first: the deadline is the `(step, layer)` order of the layer that needs
//! the unit, an urgent unit goes before prefetches of the same deadline, then submit order and
//! chunk index. A unit completes when its last chunk has been completed. The caller drives the
//! queue and passes `now` (seconds on its own clock) so [`UnitDone::took`] works for the
//! threaded service and for the virtual clock alike.

use crate::dio::Range;
use crate::reader::HostDst;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use std::io;
use std::time::Duration;

/// A destination that can be cut into the pieces of a chunked read.
pub trait Dst: Sized {
    /// The destination is exactly `len` bytes (always true when it is only bookkeeping).
    fn fits(&self, len: usize) -> bool;
    /// The bytes `offset..offset + len`.
    fn sub(&self, offset: usize, len: usize) -> io::Result<Self>;
}

impl Dst for HostDst {
    fn fits(&self, len: usize) -> bool {
        self.len() == len
    }
    fn sub(&self, offset: usize, len: usize) -> io::Result<Self> {
        HostDst::sub(self, offset, len)
    }
}

/// The timed replay moves no bytes, so its chunks carry no destination.
impl Dst for () {
    fn fits(&self, _len: usize) -> bool {
        true
    }
    fn sub(&self, _offset: usize, _len: usize) -> io::Result<Self> {
        Ok(())
    }
}

/// One slab of a unit: a file range and where it goes.
pub struct Part<D = HostDst> {
    pub shard: usize,
    pub range: Range,
    pub dst: D,
}

/// When a unit is needed. Smaller `order` first; at equal `order` urgent units go first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Deadline {
    pub order: u64,
    pub urgent: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct UnitId(u64);

/// What a landed unit moved and how long it took, from submit to its last chunk.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UnitDone {
    pub requested: u64,
    pub device: u64,
    pub took: Duration,
}

/// Smallest first: deadline, urgent before prefetch, submit order, chunk index.
type Key = (u64, bool, u64, u32);

/// A piece of a unit for a worker to read and complete.
pub struct Chunk<D> {
    key: Key,
    pub unit: UnitId,
    pub shard: usize,
    pub range: Range,
    pub dst: D,
    /// Its unit has failed (or is gone): read nothing, complete it with `Ok(0)`.
    pub skip: bool,
}

impl<D> PartialEq for Chunk<D> {
    fn eq(&self, o: &Self) -> bool {
        self.key == o.key
    }
}
impl<D> Eq for Chunk<D> {}
impl<D> PartialOrd for Chunk<D> {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl<D> Ord for Chunk<D> {
    fn cmp(&self, o: &Self) -> Ordering {
        self.key.cmp(&o.key)
    }
}

struct Unit {
    /// Chunks queued or running.
    remaining: usize,
    error: Option<io::Error>,
    requested: u64,
    device: u64,
    submitted: f64,
    took: Duration,
}

pub struct Queue<D = HostDst> {
    chunk: usize,
    heap: BinaryHeap<Reverse<Chunk<D>>>,
    units: HashMap<UnitId, Unit>,
    next_unit: u64,
    /// Chunks queued or running, over all units.
    chunks: usize,
}

impl<D: Dst> Queue<D> {
    /// `chunk`: the largest piece of a part (at least 1).
    pub fn new(chunk: usize) -> Self {
        Self {
            chunk: chunk.max(1),
            heap: BinaryHeap::new(),
            units: HashMap::new(),
            next_unit: 0,
            chunks: 0,
        }
    }

    /// Queues the chunks of `parts` as one unit at time `now`. A part whose destination does
    /// not match its range makes the unit fail at once: nothing of it is read and the unit is
    /// ready to collect.
    pub fn submit(&mut self, now: f64, deadline: Deadline, parts: Vec<Part<D>>) -> UnitId {
        let id = UnitId(self.next_unit);
        self.next_unit += 1;
        let mut unit = Unit {
            remaining: 0,
            error: None,
            requested: parts.iter().map(|p| p.range.len as u64).sum(),
            device: 0,
            submitted: now,
            took: Duration::ZERO,
        };
        let mut chunks = Vec::new();
        let mut idx = 0u32;
        'parts: for p in parts {
            if !p.dst.fits(p.range.len) {
                unit.error = Some(io::Error::other(format!(
                    "range of {} bytes for a destination of another size",
                    p.range.len
                )));
                break;
            }
            let mut at = 0;
            while at < p.range.len {
                let n = (p.range.len - at).min(self.chunk);
                let dst = match p.dst.sub(at, n) {
                    Ok(d) => d,
                    Err(e) => {
                        unit.error = Some(e);
                        break 'parts;
                    }
                };
                chunks.push(Chunk {
                    key: (deadline.order, !deadline.urgent, id.0, idx),
                    unit: id,
                    shard: p.shard,
                    range: Range {
                        offset: p.range.offset + at as u64,
                        len: n,
                    },
                    dst,
                    skip: false,
                });
                idx += 1;
                at += n;
            }
        }
        if unit.error.is_some() {
            // Nothing of a failed unit is read.
            chunks.clear();
        }
        unit.remaining = chunks.len();
        self.chunks += chunks.len();
        self.heap.extend(chunks.into_iter().map(Reverse));
        self.units.insert(id, unit);
        id
    }

    /// The next chunk to read: smallest key first. `skip` is set when its unit already failed.
    /// The caller must hand every popped chunk back to [`Queue::complete`].
    pub fn pop(&mut self) -> Option<Chunk<D>> {
        let Reverse(mut c) = self.heap.pop()?;
        c.skip = self.units.get(&c.unit).is_none_or(|u| u.error.is_some());
        Some(c)
    }

    /// A popped chunk is done at `now`: `res` is the device bytes it read, or its error (the
    /// first error of a unit is the unit's). Returns the unit when this was its last chunk.
    pub fn complete(&mut self, now: f64, chunk: Chunk<D>, res: io::Result<u64>) -> Option<UnitId> {
        self.chunks -= 1;
        let u = self.units.get_mut(&chunk.unit)?;
        match res {
            Ok(device) => u.device += device,
            Err(e) => {
                u.error.get_or_insert(e);
            }
        }
        u.remaining -= 1;
        if u.remaining == 0 {
            u.took = Duration::from_secs_f64((now - u.submitted).max(0.0));
            return Some(chunk.unit);
        }
        None
    }

    /// The completed units among `units`; forgets them. Unknown ids are ignored.
    pub fn collect(&mut self, units: &[UnitId]) -> Vec<(UnitId, io::Result<UnitDone>)> {
        let mut out = Vec::new();
        for &id in units {
            if self.units.get(&id).is_some_and(|u| u.remaining == 0) {
                let u = self.units.remove(&id).expect("checked above");
                let res = match u.error {
                    Some(e) => Err(e),
                    None => Ok(UnitDone {
                        requested: u.requested,
                        device: u.device,
                        took: u.took,
                    }),
                };
                out.push((id, res));
            }
        }
        out
    }

    /// Some of `units` is not collected yet.
    pub fn any_known(&self, units: &[UnitId]) -> bool {
        units.iter().any(|u| self.units.contains_key(u))
    }

    /// Some of `units` is known and has chunks queued or running.
    pub fn any_open(&self, units: &[UnitId]) -> bool {
        units
            .iter()
            .any(|u| self.units.get(u).is_some_and(|x| x.remaining > 0))
    }

    /// Chunks queued or running.
    pub fn inflight(&self) -> usize {
        self.chunks
    }

    /// Chunks waiting to be popped.
    pub fn queued(&self) -> usize {
        self.heap.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    const URGENT: Deadline = Deadline {
        order: 0,
        urgent: true,
    };

    fn part(len: usize) -> Part<()> {
        Part {
            shard: 0,
            range: Range { offset: 0, len },
            dst: (),
        }
    }

    fn dl(order: u64, urgent: bool) -> Deadline {
        Deadline { order, urgent }
    }

    /// Pops and completes everything in pop order; returns the order units finished in.
    fn drain(q: &mut Queue<()>) -> Vec<UnitId> {
        let mut done = Vec::new();
        while let Some(c) = q.pop() {
            let n = c.range.len as u64;
            done.extend(q.complete(0.0, c, Ok(n)));
        }
        done
    }

    #[test]
    fn earliest_deadline_first_urgent_before_prefetch() {
        let mut q = Queue::<()>::new(64);
        let a = q.submit(0.0, dl(5, false), vec![part(64)]);
        let b = q.submit(0.0, dl(3, false), vec![part(64)]);
        let c = q.submit(0.0, dl(5, true), vec![part(64)]);
        let blocker = q.submit(0.0, dl(0, false), vec![part(64)]);
        assert_eq!(drain(&mut q), [blocker, b, c, a]);
        assert_eq!(q.inflight(), 0);
    }

    #[test]
    fn a_unit_completes_with_its_last_chunk_and_reports_its_time() {
        let mut q = Queue::<()>::new(100);
        let id = q.submit(1.0, URGENT, vec![part(250)]);
        assert_eq!(q.inflight(), 3);
        let c1 = q.pop().unwrap();
        let c2 = q.pop().unwrap();
        let c3 = q.pop().unwrap();
        assert_eq!([c1.range.len, c2.range.len, c3.range.len], [100, 100, 50]);
        assert_eq!(q.complete(2.0, c2, Ok(100)), None);
        assert_eq!(q.complete(2.5, c1, Ok(100)), None);
        assert!(q.collect(&[id]).is_empty());
        assert_eq!(q.complete(3.0, c3, Ok(50)), Some(id));
        let done = q.collect(&[id]);
        let d = done[0].1.as_ref().unwrap();
        assert_eq!((d.requested, d.device), (250, 250));
        assert_eq!(d.took, Duration::from_secs(2));
        // Collected: unknown now.
        assert!(!q.any_known(&[id]));
        assert!(q.collect(&[id]).is_empty());
    }

    #[test]
    fn a_failed_chunk_fails_its_unit_alone_and_skips_its_other_chunks() {
        let mut q = Queue::<()>::new(10);
        let bad = q.submit(0.0, dl(0, false), vec![part(30)]);
        let good = q.submit(0.0, dl(1, false), vec![part(10)]);
        let c = q.pop().unwrap();
        assert!(!c.skip);
        q.complete(1.0, c, Err(io::Error::other("cut")));
        let c = q.pop().unwrap();
        assert_eq!(c.unit, bad);
        assert!(c.skip, "the rest of a failed unit is not read");
        q.complete(1.0, c, Ok(0));
        let c = q.pop().unwrap();
        assert!(c.skip);
        assert_eq!(q.complete(1.0, c, Ok(0)), Some(bad));
        let c = q.pop().unwrap();
        assert!(!c.skip);
        assert_eq!(q.complete(1.0, c, Ok(10)), Some(good));
        let res = q.collect(&[bad, good]);
        assert!(res[0].1.is_err());
        assert!(res[1].1.is_ok());
        assert_eq!(q.inflight(), 0);
    }

    /// A destination that does not fit its range fails the unit at submit, with nothing queued.
    #[test]
    fn a_wrong_sized_destination_fails_at_submit() {
        struct Fixed(usize);
        impl Dst for Fixed {
            fn fits(&self, len: usize) -> bool {
                self.0 == len
            }
            fn sub(&self, _: usize, len: usize) -> io::Result<Self> {
                Ok(Fixed(len))
            }
        }
        let mut q = Queue::<Fixed>::new(4096);
        let wrong = q.submit(
            0.0,
            URGENT,
            vec![Part {
                shard: 0,
                range: Range { offset: 0, len: 10 },
                dst: Fixed(3),
            }],
        );
        assert_eq!(q.inflight(), 0);
        assert!(q.pop().is_none());
        assert!(q.collect(&[wrong])[0].1.is_err());
    }

    #[test]
    fn host_dst_sub_rejects_out_of_range() {
        let mut v = vec![0u8; 10];
        // SAFETY: `v` outlives the destinations and is not touched meanwhile.
        let d = unsafe { HostDst::new(v.as_mut_ptr(), v.len()) };
        assert!(d.sub(0, 10).is_ok());
        assert!(d.sub(4, 6).is_ok());
        assert!(d.sub(4, 7).is_err());
        assert!(d.sub(11, 0).is_err());
        assert!(d.sub(usize::MAX, 2).is_err());
        assert_eq!(d.sub(4, 6).unwrap().len(), 6);
    }

    proptest! {
        /// Chunks pop sorted by (deadline, urgent first, submit order, chunk index), each exactly
        /// once, and every unit completes when its last chunk does.
        #[test]
        fn pop_order_is_sorted_and_every_chunk_pops_once(
            units in prop::collection::vec((0u64..6, any::<bool>(), 1usize..300), 1..12),
            chunk in 1usize..120,
        ) {
            let mut q = Queue::<()>::new(chunk);
            let ids: Vec<UnitId> = units
                .iter()
                .map(|&(o, u, len)| q.submit(0.0, dl(o, u), vec![part(len)]))
                .collect();
            let total: usize = units.iter().map(|u| u.2.div_ceil(chunk)).sum();
            prop_assert_eq!(q.inflight(), total);
            let mut popped = Vec::new();
            let mut finished = Vec::new();
            while let Some(c) = q.pop() {
                popped.push(c.key);
                let n = c.range.len as u64;
                finished.extend(q.complete(0.0, c, Ok(n)));
            }
            prop_assert_eq!(popped.len(), total);
            let mut sorted = popped.clone();
            sorted.sort();
            prop_assert_eq!(&popped, &sorted);
            sorted.dedup();
            prop_assert_eq!(sorted.len(), total);
            finished.sort();
            prop_assert_eq!(finished, ids);
            prop_assert_eq!(q.inflight(), 0);
        }
    }
}
