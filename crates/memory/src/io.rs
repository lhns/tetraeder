//! A persistent I/O service for expert reads that run beside compute (ADR 0054).
//!
//! `workers` threads each own one [`DirectFile`] per shard (queue depth 1, so a read is inline)
//! and one bounce buffer. A worker takes the next chunk of at most `chunk_bytes`, reads it and
//! copies it into its destination from the same thread, so the copies run in parallel with the
//! device and never after it. A unit (one expert: its slabs) completes when its last chunk has
//! landed and its waiters wake. Chunks run earliest deadline first: the deadline is the
//! `(step, layer)` order of the layer that needs the unit, and an urgent unit goes before
//! prefetches of the same deadline.
//!
//! `unsafe` stays in [`HostDst`] (`reader.rs`); this module only moves destinations to workers.

use crate::dio::{ALIGN, DirectFile, Options, Range, ReadStats};
use crate::reader::HostDst;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use std::io;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Workers when [`ServiceOptions::workers`] is 0.
pub const DEFAULT_WORKERS: usize = 16;
/// Chunk size when [`ServiceOptions::chunk_bytes`] is 0 (ADR 0038 lever 2).
pub const DEFAULT_CHUNK: usize = 1 << 20;

#[derive(Debug, Clone, Copy, Default)]
pub struct ServiceOptions {
    /// Worker threads. 0: [`DEFAULT_WORKERS`].
    pub workers: usize,
    /// Largest piece of a slab read and copied by one worker, rounded up to 4 KiB.
    /// 0: [`DEFAULT_CHUNK`].
    pub chunk_bytes: usize,
    /// Open the files buffered (tests, filesystems that refuse direct I/O).
    pub buffered: bool,
    /// Test only: each worker sleeps this long before each chunk, so units stay in flight while
    /// the caller computes.
    #[doc(hidden)]
    pub delay: Option<Duration>,
}

/// One slab of a unit: a file range and where it goes.
pub struct Part {
    pub shard: usize,
    pub range: Range,
    pub dst: HostDst,
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

/// Smallest key first: deadline, urgent before prefetch, submit order, chunk index.
type Key = (u64, bool, u64, u32);

struct Chunk {
    key: Key,
    unit: UnitId,
    shard: usize,
    range: Range,
    dst: HostDst,
}

impl PartialEq for Chunk {
    fn eq(&self, o: &Self) -> bool {
        self.key == o.key
    }
}
impl Eq for Chunk {}
impl PartialOrd for Chunk {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Chunk {
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
    submitted: Instant,
    took: Duration,
}

#[derive(Default)]
struct State {
    heap: BinaryHeap<Reverse<Chunk>>,
    units: HashMap<UnitId, Unit>,
    next_unit: u64,
    /// Chunks queued or running, over all units.
    chunks: usize,
    closed: bool,
}

struct Inner {
    state: Mutex<State>,
    /// Workers wait here for chunks.
    work: Condvar,
    /// Waiters wait here for units.
    done: Condvar,
    delay: Option<Duration>,
}

impl Inner {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

pub struct IoService {
    inner: Arc<Inner>,
    workers: Vec<JoinHandle<()>>,
    chunk: usize,
    bounce: u64,
}

impl IoService {
    /// Opens every shard once per worker. Errors when a file cannot be opened.
    pub fn new(paths: &[PathBuf], opts: ServiceOptions) -> io::Result<Self> {
        let n = if opts.workers == 0 {
            DEFAULT_WORKERS
        } else {
            opts.workers
        };
        let chunk = if opts.chunk_bytes == 0 {
            DEFAULT_CHUNK
        } else {
            opts.chunk_bytes.next_multiple_of(ALIGN)
        };
        let dio = Options {
            queue_depth: 1,
            buffered: opts.buffered,
            bounce_bytes: chunk + 2 * ALIGN,
            chunk_bytes: chunk,
            ..Options::default()
        };
        let mut files: Vec<Vec<DirectFile>> = Vec::with_capacity(n);
        for _ in 0..n {
            files.push(
                paths
                    .iter()
                    .map(|p| DirectFile::open_with(p, dio))
                    .collect::<io::Result<_>>()?,
            );
        }
        let inner = Arc::new(Inner {
            state: Mutex::new(State::default()),
            work: Condvar::new(),
            done: Condvar::new(),
            delay: opts.delay,
        });
        let workers = files
            .into_iter()
            .enumerate()
            .map(|(i, f)| {
                let inner = inner.clone();
                std::thread::Builder::new()
                    .name(format!("tetraeder-io-{i}"))
                    .spawn(move || work(&inner, &f))
                    .expect("spawn an I/O worker")
            })
            .collect();
        Ok(Self {
            inner,
            workers,
            chunk,
            bounce: (n * paths.len() * (chunk + 2 * ALIGN)) as u64,
        })
    }

    /// Queues the chunks of `parts` as one unit. A part whose destination does not match its
    /// range makes the unit fail at once (see the waits).
    pub fn submit(&self, deadline: Deadline, parts: Vec<Part>) -> UnitId {
        let mut st = self.inner.lock();
        let id = UnitId(st.next_unit);
        st.next_unit += 1;
        let mut unit = Unit {
            remaining: 0,
            error: None,
            requested: parts.iter().map(|p| p.range.len as u64).sum(),
            device: 0,
            submitted: Instant::now(),
            took: Duration::ZERO,
        };
        let mut chunks = Vec::new();
        let mut idx = 0u32;
        'parts: for p in parts {
            if p.dst.len() != p.range.len {
                unit.error = Some(io::Error::other(format!(
                    "range of {} bytes for a destination of {}",
                    p.range.len,
                    p.dst.len()
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
        st.chunks += chunks.len();
        st.heap.extend(chunks.into_iter().map(Reverse));
        st.units.insert(id, unit);
        drop(st);
        self.inner.work.notify_all();
        self.inner.done.notify_all();
        id
    }

    /// Blocks until at least one of `units` has completed; returns every completed one among
    /// them and forgets them. Ids that are unknown (already collected) are ignored; with none
    /// known it returns at once.
    pub fn wait_any(&self, units: &[UnitId]) -> Vec<(UnitId, io::Result<UnitDone>)> {
        let mut st = self.inner.lock();
        loop {
            let done = collect(&mut st, units);
            if !done.is_empty() || !units.iter().any(|u| st.units.contains_key(u)) {
                return done;
            }
            st = self.inner.done.wait(st).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Blocks until every known unit of `units` has completed; returns and forgets them.
    pub fn wait_all(&self, units: &[UnitId]) -> Vec<(UnitId, io::Result<UnitDone>)> {
        let mut st = self.inner.lock();
        while units
            .iter()
            .any(|u| st.units.get(u).is_some_and(|x| x.remaining > 0))
        {
            st = self.inner.done.wait(st).unwrap_or_else(|e| e.into_inner());
        }
        collect(&mut st, units)
    }

    /// The completed units among `units`, without blocking; forgets them.
    pub fn poll(&self, units: &[UnitId]) -> Vec<(UnitId, io::Result<UnitDone>)> {
        collect(&mut self.inner.lock(), units)
    }

    /// Chunks queued or running.
    pub fn inflight(&self) -> usize {
        self.inner.lock().chunks
    }

    /// Bytes of the workers' bounce buffers (they are allocated on the first read).
    pub fn bounce_bytes(&self) -> u64 {
        self.bounce
    }
}

impl Drop for IoService {
    fn drop(&mut self) {
        // Workers finish every queued chunk before they exit, so no destination is written
        // after this returns.
        self.inner.lock().closed = true;
        self.inner.work.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

fn collect(st: &mut State, units: &[UnitId]) -> Vec<(UnitId, io::Result<UnitDone>)> {
    let mut out = Vec::new();
    for &id in units {
        if st.units.get(&id).is_some_and(|u| u.remaining == 0) {
            let u = st.units.remove(&id).expect("checked above");
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

fn work(inner: &Inner, files: &[DirectFile]) {
    loop {
        let (chunk, skip) = {
            let mut st = inner.lock();
            loop {
                if let Some(Reverse(c)) = st.heap.pop() {
                    let skip = st.units.get(&c.unit).is_none_or(|u| u.error.is_some());
                    break (c, skip);
                }
                if st.closed {
                    return;
                }
                st = inner.work.wait(st).unwrap_or_else(|e| e.into_inner());
            }
        };
        let res = if skip {
            Ok(ReadStats::default())
        } else {
            if let Some(d) = inner.delay {
                std::thread::sleep(d);
            }
            match files.get(chunk.shard) {
                Some(f) => f.read_batch_with(&[chunk.range], |_, bytes| chunk.dst.write(bytes)),
                None => Err(io::Error::other(format!("no shard {}", chunk.shard))),
            }
        };
        let mut st = inner.lock();
        st.chunks -= 1;
        if let Some(u) = st.units.get_mut(&chunk.unit) {
            match res {
                Ok(s) => u.device += s.device,
                Err(e) => {
                    u.error.get_or_insert(e);
                }
            }
            u.remaining -= 1;
            if u.remaining == 0 {
                u.took = u.submitted.elapsed();
                drop(st);
                inner.done.notify_all();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn file(len: usize) -> (PathBuf, Vec<u8>) {
        let data: Vec<u8> = (0..len).map(|i| (i * 7 + i / 251) as u8).collect();
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("tetraeder-io-{}-{n}", std::process::id()));
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&data)
            .unwrap();
        (path, data)
    }

    fn opts(workers: usize, buffered: bool) -> ServiceOptions {
        ServiceOptions {
            workers,
            chunk_bytes: 4096,
            buffered,
            delay: None,
        }
    }

    const RANGES: [Range; 3] = [
        Range {
            offset: 5,
            len: 10_000,
        },
        Range {
            offset: 70_001,
            len: 4096,
        },
        Range {
            offset: 20_000,
            len: 1,
        },
    ];

    fn parts(dst: &mut [Vec<u8>]) -> Vec<Part> {
        RANGES
            .iter()
            .zip(dst.iter_mut())
            .map(|(r, d)| Part {
                shard: 0,
                range: *r,
                // SAFETY: `dst` is not touched until the unit has been waited for.
                dst: unsafe { HostDst::new(d.as_mut_ptr(), d.len()) },
            })
            .collect()
    }

    fn lands_byte_exact(buffered: bool) {
        let (path, data) = file(100_000);
        let svc = IoService::new(std::slice::from_ref(&path), opts(3, buffered)).unwrap();
        let mut dst: Vec<Vec<u8>> = RANGES.iter().map(|r| vec![0; r.len]).collect();
        let id = svc.submit(
            Deadline {
                order: 0,
                urgent: false,
            },
            parts(&mut dst),
        );
        let done = svc.wait_all(&[id]);
        let d = done[0].1.as_ref().unwrap();
        assert_eq!(d.requested, 10_000 + 4096 + 1);
        assert!(d.device >= d.requested);
        for (r, d) in RANGES.iter().zip(&dst) {
            assert_eq!(d[..], data[r.offset as usize..][..r.len]);
        }
        assert_eq!(svc.inflight(), 0);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn unit_lands_byte_exact_across_chunks_buffered() {
        lands_byte_exact(true);
    }

    #[test]
    fn unit_lands_byte_exact_across_chunks_direct() {
        lands_byte_exact(false);
    }

    #[test]
    fn wait_any_returns_landed_units_and_never_blocks_on_collected_ones() {
        let (path, _) = file(50_000);
        let svc = IoService::new(std::slice::from_ref(&path), opts(2, true)).unwrap();
        let mut bufs: Vec<Vec<u8>> = (0..2).map(|_| vec![0; 100]).collect();
        let ids: Vec<_> = bufs
            .iter_mut()
            .enumerate()
            .map(|(i, b)| {
                svc.submit(
                    Deadline {
                        order: 0,
                        urgent: false,
                    },
                    vec![Part {
                        shard: 0,
                        range: Range {
                            offset: i as u64 * 100,
                            len: 100,
                        },
                        // SAFETY: `b` is not touched until waited for.
                        dst: unsafe { HostDst::new(b.as_mut_ptr(), b.len()) },
                    }],
                )
            })
            .collect();
        let mut got = Vec::new();
        while got.len() < 2 {
            for (id, r) in svc.wait_any(&ids) {
                r.unwrap();
                got.push(id);
            }
        }
        got.sort();
        assert_eq!(got, ids);
        // Both are collected: nothing is known any more, so no wait blocks.
        assert!(svc.wait_any(&ids).is_empty());
        assert!(svc.wait_all(&ids).is_empty());
        assert!(svc.poll(&ids).is_empty());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn earliest_deadline_first_urgent_before_prefetch() {
        let (path, _) = file(50_000);
        let o = ServiceOptions {
            delay: Some(Duration::from_millis(60)),
            ..opts(1, true)
        };
        let svc = IoService::new(std::slice::from_ref(&path), o).unwrap();
        let mut bufs: Vec<Vec<u8>> = (0..4).map(|_| vec![0; 64]).collect();
        let mut submit = |i: usize, order: u64, urgent: bool| {
            let b = &mut bufs[i];
            svc.submit(
                Deadline { order, urgent },
                vec![Part {
                    shard: 0,
                    range: Range { offset: 0, len: 64 },
                    // SAFETY: the buffers are not touched until the service is dropped.
                    dst: unsafe { HostDst::new(b.as_mut_ptr(), b.len()) },
                }],
            )
        };
        // The first unit occupies the only worker while the others queue up.
        let blocker = submit(0, 0, false);
        let a = submit(1, 5, false);
        let b = submit(2, 3, false);
        let c = submit(3, 5, true);
        let mut order = Vec::new();
        let mut left = vec![blocker, a, b, c];
        while !left.is_empty() {
            for (id, r) in svc.wait_any(&left) {
                r.unwrap();
                order.push(id);
                left.retain(|x| *x != id);
            }
        }
        assert_eq!(order, [blocker, b, c, a]);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_failing_unit_fails_alone_and_the_service_keeps_serving() {
        let (path, _) = file(10_000);
        let svc = IoService::new(std::slice::from_ref(&path), opts(2, true)).unwrap();
        let dl = Deadline {
            order: 0,
            urgent: false,
        };
        let mut a = vec![0u8; 5_000];
        // Past the end of the file.
        let bad = svc.submit(
            dl,
            vec![Part {
                shard: 0,
                range: Range {
                    offset: 9_000,
                    len: 5_000,
                },
                // SAFETY: `a` is not touched until waited for.
                dst: unsafe { HostDst::new(a.as_mut_ptr(), a.len()) },
            }],
        );
        // A destination of the wrong size fails at submit.
        let mut small = [0u8; 3];
        let wrong = svc.submit(
            dl,
            vec![Part {
                shard: 0,
                range: Range { offset: 0, len: 10 },
                // SAFETY: `small` is not touched until waited for.
                dst: unsafe { HostDst::new(small.as_mut_ptr(), small.len()) },
            }],
        );
        let mut ok = vec![0u8; 10];
        let good = svc.submit(
            dl,
            vec![Part {
                shard: 0,
                range: Range { offset: 0, len: 10 },
                // SAFETY: `ok` is not touched until waited for.
                dst: unsafe { HostDst::new(ok.as_mut_ptr(), ok.len()) },
            }],
        );
        let res = svc.wait_all(&[bad, wrong, good]);
        assert!(res.iter().find(|r| r.0 == bad).unwrap().1.is_err());
        assert!(res.iter().find(|r| r.0 == wrong).unwrap().1.is_err());
        assert!(res.iter().find(|r| r.0 == good).unwrap().1.is_ok());
        assert_eq!(svc.inflight(), 0);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn drop_waits_for_queued_units() {
        let (path, data) = file(20_000);
        let o = ServiceOptions {
            delay: Some(Duration::from_millis(20)),
            ..opts(1, true)
        };
        let svc = IoService::new(std::slice::from_ref(&path), o).unwrap();
        let mut bufs: Vec<Vec<u8>> = (0..3).map(|_| vec![0; 4096]).collect();
        for (i, b) in bufs.iter_mut().enumerate() {
            let _ = svc.submit(
                Deadline {
                    order: i as u64,
                    urgent: false,
                },
                vec![Part {
                    shard: 0,
                    range: Range {
                        offset: i as u64 * 4096,
                        len: 4096,
                    },
                    // SAFETY: `b` is not touched until the service is dropped.
                    dst: unsafe { HostDst::new(b.as_mut_ptr(), b.len()) },
                }],
            );
        }
        drop(svc);
        for (i, b) in bufs.iter().enumerate() {
            assert_eq!(b[..], data[i * 4096..][..4096]);
        }
        let _ = std::fs::remove_file(path);
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
}
