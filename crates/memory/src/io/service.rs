//! [`IoService`]: the threaded driver of the chunk queue (ADR 0054).
//!
//! `workers` threads each own one [`DirectFile`] per shard (queue depth 1, so a read is inline)
//! and one bounce buffer. A worker takes the next chunk of at most `chunk_bytes`, reads it and
//! copies it into its destination from the same thread, so the copies run in parallel with the
//! device and never after it. The order of chunks, the units and their completion are the
//! [`Queue`]'s; this module only runs them on threads and the real clock.
//!
//! `unsafe` stays in [`HostDst`] (`reader.rs`); this module only moves destinations to workers.

use super::Io;
use super::queue::{Chunk, Deadline, Part, Queue, UnitDone, UnitId};
use crate::dio::{ALIGN, DirectFile, Options};
use crate::reader::HostDst;
use std::io;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Instant;

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
}

struct State {
    queue: Queue<HostDst>,
    closed: bool,
}

struct Inner {
    state: Mutex<State>,
    /// Workers wait here for chunks.
    work: Condvar,
    /// Waiters wait here for units.
    done: Condvar,
    start: Instant,
}

impl Inner {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Seconds on the service's clock.
    fn now(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }
}

pub struct IoService {
    inner: Arc<Inner>,
    workers: Vec<JoinHandle<()>>,
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
            state: Mutex::new(State {
                queue: Queue::new(chunk),
                closed: false,
            }),
            work: Condvar::new(),
            done: Condvar::new(),
            start: Instant::now(),
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
            bounce: (n * paths.len() * (chunk + 2 * ALIGN)) as u64,
        })
    }

    /// Queues the chunks of `parts` as one unit. A part whose destination does not match its
    /// range makes the unit fail at once (see the waits).
    pub fn submit(&self, deadline: Deadline, parts: Vec<Part>) -> UnitId {
        let mut st = self.inner.lock();
        let id = st.queue.submit(self.inner.now(), deadline, parts);
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
            let done = st.queue.collect(units);
            if !done.is_empty() || !st.queue.any_known(units) {
                return done;
            }
            st = self.inner.done.wait(st).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Blocks until every known unit of `units` has completed; returns and forgets them.
    pub fn wait_all(&self, units: &[UnitId]) -> Vec<(UnitId, io::Result<UnitDone>)> {
        let mut st = self.inner.lock();
        while st.queue.any_open(units) {
            st = self.inner.done.wait(st).unwrap_or_else(|e| e.into_inner());
        }
        st.queue.collect(units)
    }

    /// The completed units among `units`, without blocking; forgets them.
    pub fn poll(&self, units: &[UnitId]) -> Vec<(UnitId, io::Result<UnitDone>)> {
        self.inner.lock().queue.collect(units)
    }

    /// Chunks queued or running.
    pub fn inflight(&self) -> usize {
        self.inner.lock().queue.inflight()
    }

    /// Bytes of the workers' bounce buffers (they are allocated on the first read).
    pub fn bounce_bytes(&self) -> u64 {
        self.bounce
    }
}

impl Io for IoService {
    fn submit(&self, deadline: Deadline, parts: Vec<Part>) -> UnitId {
        IoService::submit(self, deadline, parts)
    }
    fn poll(&self, units: &[UnitId]) -> Vec<(UnitId, io::Result<UnitDone>)> {
        IoService::poll(self, units)
    }
    fn wait_any(&self, units: &[UnitId]) -> Vec<(UnitId, io::Result<UnitDone>)> {
        IoService::wait_any(self, units)
    }
    fn wait_all(&self, units: &[UnitId]) -> Vec<(UnitId, io::Result<UnitDone>)> {
        IoService::wait_all(self, units)
    }
    fn inflight(&self) -> usize {
        IoService::inflight(self)
    }
    fn bounce_bytes(&self) -> u64 {
        IoService::bounce_bytes(self)
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

fn work(inner: &Inner, files: &[DirectFile]) {
    loop {
        let chunk: Chunk<HostDst> = {
            let mut st = inner.lock();
            loop {
                if let Some(c) = st.queue.pop() {
                    break c;
                }
                if st.closed {
                    return;
                }
                st = inner.work.wait(st).unwrap_or_else(|e| e.into_inner());
            }
        };
        let res = if chunk.skip {
            Ok(0)
        } else {
            match files.get(chunk.shard) {
                Some(f) => f
                    .read_batch_with(&[chunk.range], |_, bytes| chunk.dst.write(bytes))
                    .map(|s| s.device),
                None => Err(io::Error::other(format!("no shard {}", chunk.shard))),
            }
        };
        let mut st = inner.lock();
        let landed = st.queue.complete(inner.now(), chunk, res);
        drop(st);
        if landed.is_some() {
            inner.done.notify_all();
        }
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

    /// One worker, three queued units, no wait: the drop still lands them all.
    #[test]
    fn drop_waits_for_queued_units() {
        let (path, data) = file(20_000);
        let svc = IoService::new(std::slice::from_ref(&path), opts(1, true)).unwrap();
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
}
