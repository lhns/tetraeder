//! [`SimIo`]: the I/O service on a virtual clock, for tests (ADR 0056).
//!
//! The same [`Queue`] as the threaded service, a [`Disk`] model instead of workers, and a
//! [`SimClock`] that only moves when a test advances it or a wait needs the next landing. No
//! thread, no sleep: a unit is in flight until the clock reaches its time on the disk curve, so
//! a test sees the layer steps with experts still in flight deterministically. A landing copies
//! the bytes from the shard file (a buffered read, instant in real time) into the destination,
//! so a model that reads through it computes the same logits as with the real service.

use super::Io;
use super::disk::Disk;
use super::queue::{Chunk, Deadline, Part, Queue, UnitDone, UnitId};
use crate::dio::{DirectFile, Options};
use crate::reader::HostDst;
use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use tetraeder_profiler::QueuePoint;

/// Virtual time in seconds, shared by clones. Only moves forward.
#[derive(Clone, Default)]
pub struct SimClock(Arc<AtomicU64>);

impl SimClock {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn now(&self) -> f64 {
        f64::from_bits(self.0.load(Ordering::SeqCst))
    }

    /// Moves to `t`; earlier times are ignored.
    pub fn set(&self, t: f64) {
        if t > self.now() {
            self.0.store(t.to_bits(), Ordering::SeqCst);
        }
    }

    pub fn advance(&self, dt: f64) {
        self.set(self.now() + dt);
    }
}

struct State {
    queue: Queue<HostDst>,
    disk: Disk<Chunk<HostDst>>,
    files: Vec<DirectFile>,
}

impl State {
    /// Starts queued chunks while a worker is free, at the disk's time.
    fn pump(&mut self) {
        while self.disk.free() > 0 {
            let Some(c) = self.queue.pop() else { break };
            let bytes = if c.skip { 0 } else { c.range.len as u64 };
            if self.disk.submit(bytes, c).is_err() {
                unreachable!("a worker was free");
            }
        }
    }

    /// Lands everything that finishes by `t` and leaves the disk at `t`.
    fn land_until(&mut self, t: f64) {
        loop {
            self.pump();
            match self.disk.next_completion() {
                Some(tc) if tc <= t => {
                    for (at, chunk) in self.disk.advance_to(tc) {
                        let res = self.read(&chunk);
                        self.queue.complete(at, chunk, res);
                    }
                }
                _ => {
                    self.disk.advance_to(t);
                    return;
                }
            }
        }
    }

    fn read(&self, chunk: &Chunk<HostDst>) -> io::Result<u64> {
        if chunk.skip {
            return Ok(0);
        }
        match self.files.get(chunk.shard) {
            Some(f) => f
                .read_batch_with(&[chunk.range], |_, bytes| chunk.dst.write(bytes))
                .map(|s| s.device),
            None => Err(io::Error::other(format!("no shard {}", chunk.shard))),
        }
    }
}

pub struct SimIo {
    clock: SimClock,
    state: Mutex<State>,
}

impl SimIo {
    /// `curve`: the disk's bytes per second by queue depth; `workers` chunks are in service at
    /// most; `chunk_bytes` is the size of a read.
    pub fn new(
        paths: &[PathBuf],
        clock: SimClock,
        curve: Vec<QueuePoint>,
        workers: usize,
        chunk_bytes: usize,
    ) -> io::Result<Self> {
        let opts = Options {
            buffered: true,
            ..Options::default()
        };
        let files = paths
            .iter()
            .map(|p| DirectFile::open_with(p, opts))
            .collect::<io::Result<Vec<_>>>()?;
        Ok(Self {
            state: Mutex::new(State {
                queue: Queue::new(chunk_bytes),
                disk: Disk::new(curve, workers),
                files,
            }),
            clock,
        })
    }

    /// A flat disk of `bytes_per_s`, whatever the depth.
    pub fn flat(
        paths: &[PathBuf],
        clock: SimClock,
        bytes_per_s: f64,
        workers: usize,
        chunk_bytes: usize,
    ) -> io::Result<Self> {
        let curve = vec![QueuePoint {
            depth: 1,
            bytes_per_s,
        }];
        Self::new(paths, clock, curve, workers, chunk_bytes)
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Io for SimIo {
    fn submit(&self, deadline: Deadline, parts: Vec<Part>) -> UnitId {
        let mut st = self.lock();
        let now = self.clock.now();
        st.land_until(now);
        let id = st.queue.submit(now, deadline, parts);
        st.pump();
        id
    }

    fn poll(&self, units: &[UnitId]) -> Vec<(UnitId, io::Result<UnitDone>)> {
        let mut st = self.lock();
        st.land_until(self.clock.now());
        st.queue.collect(units)
    }

    fn wait_any(&self, units: &[UnitId]) -> Vec<(UnitId, io::Result<UnitDone>)> {
        let mut st = self.lock();
        st.land_until(self.clock.now());
        loop {
            let done = st.queue.collect(units);
            if !done.is_empty() || !st.queue.any_known(units) {
                return done;
            }
            // Nothing in service while a unit is open cannot happen; do not hang if it does.
            let Some(tc) = st.disk.next_completion() else {
                return done;
            };
            self.clock.set(tc);
            st.land_until(self.clock.now());
        }
    }

    fn wait_all(&self, units: &[UnitId]) -> Vec<(UnitId, io::Result<UnitDone>)> {
        let mut st = self.lock();
        st.land_until(self.clock.now());
        while st.queue.any_open(units) {
            let Some(tc) = st.disk.next_completion() else {
                break;
            };
            self.clock.set(tc);
            st.land_until(self.clock.now());
        }
        st.queue.collect(units)
    }

    fn inflight(&self) -> usize {
        self.lock().queue.inflight()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dio::Range;
    use std::io::Write;

    fn file(len: usize) -> (PathBuf, Vec<u8>) {
        let data: Vec<u8> = (0..len).map(|i| (i * 7 + i / 251) as u8).collect();
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("tetraeder-sim-{}-{n}", std::process::id()));
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&data)
            .unwrap();
        (path, data)
    }

    fn dl(order: u64, urgent: bool) -> Deadline {
        Deadline { order, urgent }
    }

    fn unit(io: &SimIo, buf: &mut [u8], offset: u64, deadline: Deadline) -> UnitId {
        io.submit(
            deadline,
            vec![Part {
                shard: 0,
                range: Range {
                    offset,
                    len: buf.len(),
                },
                // SAFETY: `buf` is not touched until the unit has been waited for.
                dst: unsafe { HostDst::new(buf.as_mut_ptr(), buf.len()) },
            }],
        )
    }

    #[test]
    fn lands_byte_exact_and_only_when_the_clock_gets_there() {
        let (path, data) = file(50_000);
        let clock = SimClock::new();
        // 1 MB/s: a 10 000 byte unit takes 10 ms.
        let io = SimIo::flat(std::slice::from_ref(&path), clock.clone(), 1e6, 2, 4096).unwrap();
        let mut buf = vec![0u8; 10_000];
        let id = unit(&io, &mut buf, 5, dl(0, false));
        // Nothing has landed at time 0, however often asked.
        assert!(io.poll(&[id]).is_empty());
        assert!(io.inflight() > 0);
        clock.advance(0.005);
        assert!(io.poll(&[id]).is_empty());
        clock.advance(0.006);
        let done = io.poll(&[id]);
        let d = done[0].1.as_ref().unwrap();
        assert_eq!(d.requested, 10_000);
        assert_eq!(buf[..], data[5..10_005]);
        assert_eq!(io.inflight(), 0);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn wait_any_advances_the_clock_to_the_earliest_deadline_first() {
        let (path, _) = file(50_000);
        let clock = SimClock::new();
        // One worker: units run one after the other in queue order.
        let io = SimIo::flat(std::slice::from_ref(&path), clock.clone(), 1e6, 1, 4096).unwrap();
        let mut bufs: Vec<Vec<u8>> = (0..4).map(|_| vec![0; 1000]).collect();
        let [b0, b1, b2, b3] = &mut bufs[..] else {
            unreachable!()
        };
        // The first unit occupies the only worker while the others queue up.
        let blocker = unit(&io, b0, 0, dl(0, false));
        let a = unit(&io, b1, 0, dl(5, false));
        let b = unit(&io, b2, 0, dl(3, false));
        let c = unit(&io, b3, 0, dl(5, true));
        let mut left = vec![blocker, a, b, c];
        let mut order = Vec::new();
        let mut times = Vec::new();
        while !left.is_empty() {
            for (id, r) in io.wait_any(&left) {
                r.unwrap();
                order.push(id);
                left.retain(|x| *x != id);
            }
            times.push(clock.now());
        }
        // Earliest deadline first, urgent before prefetch at equal deadline.
        assert_eq!(order, [blocker, b, c, a]);
        for (i, t) in times.iter().enumerate() {
            assert!((t - 1e-3 * (i + 1) as f64).abs() < 1e-9, "{times:?}");
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn wait_all_lands_everything_and_a_collected_unit_never_blocks() {
        let (path, data) = file(50_000);
        let clock = SimClock::new();
        let io = SimIo::flat(std::slice::from_ref(&path), clock.clone(), 1e6, 2, 4096).unwrap();
        let mut b0 = vec![0u8; 6000];
        let mut b1 = vec![0u8; 100];
        let ids = [
            unit(&io, &mut b0, 100, dl(0, false)),
            unit(&io, &mut b1, 9000, dl(1, false)),
        ];
        let done = io.wait_all(&ids);
        assert_eq!(done.len(), 2);
        assert!(done.iter().all(|d| d.1.is_ok()));
        assert_eq!(b0[..], data[100..6100]);
        assert_eq!(b1[..], data[9000..9100]);
        assert!(clock.now() > 0.0);
        assert!(io.wait_any(&ids).is_empty());
        assert!(io.wait_all(&ids).is_empty());
        assert!(io.poll(&ids).is_empty());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_failing_unit_fails_alone() {
        let (path, _) = file(10_000);
        let clock = SimClock::new();
        let io = SimIo::flat(std::slice::from_ref(&path), clock, 1e6, 2, 4096).unwrap();
        let mut a = vec![0u8; 5_000];
        // Past the end of the file.
        let bad = unit(&io, &mut a, 9_000, dl(0, false));
        let mut small = [0u8; 3];
        // A destination of the wrong size fails at submit.
        let wrong = io.submit(
            dl(0, false),
            vec![Part {
                shard: 0,
                range: Range { offset: 0, len: 10 },
                // SAFETY: `small` is not touched until waited for.
                dst: unsafe { HostDst::new(small.as_mut_ptr(), small.len()) },
            }],
        );
        let mut ok = vec![0u8; 10];
        let good = unit(&io, &mut ok, 0, dl(0, false));
        let res = io.wait_all(&[bad, wrong, good]);
        assert!(res.iter().find(|r| r.0 == bad).unwrap().1.is_err());
        assert!(res.iter().find(|r| r.0 == wrong).unwrap().1.is_err());
        assert!(res.iter().find(|r| r.0 == good).unwrap().1.is_ok());
        assert_eq!(io.inflight(), 0);
        let _ = std::fs::remove_file(path);
    }
}
