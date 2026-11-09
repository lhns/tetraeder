//! Background reads of file ranges into caller-owned destinations (ADR 0038): one worker thread
//! runs [`DirectFile::read_batch_with`] for a job while the caller computes, and hands each
//! completed range to a sink built by the caller.
//!
//! The sink is a boxed closure, so the destination stays pluggable. [`copy_sink`] is the one for
//! raw host memory (a CPU buffer); a destination that needs a call per write, such as a repacked
//! buffer, builds its own sink. `unsafe` is confined to [`HostDst`].

use crate::dio::{DirectFile, Range, ReadStats};
use std::io;
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Receives range `i` of a job once it has landed. Runs on the worker thread.
pub type Sink = Box<dyn FnMut(usize, &[u8]) -> io::Result<()> + Send>;

/// Ranges of one file and where they go.
pub struct ReadJob {
    pub file: Arc<DirectFile>,
    pub ranges: Vec<Range>,
    pub sink: Sink,
}

/// A writable region of host memory that the worker thread may fill.
pub struct HostDst {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: a destination is plain data; `HostDst::new` makes the caller keep the memory valid and
// unaliased until the read has finished.
unsafe impl Send for HostDst {}

impl HostDst {
    /// # Safety
    /// `ptr` must be valid for writes of `len` bytes, and nothing else may access those bytes,
    /// from the submit of the job that uses this destination until its [`Reader::wait`] returns
    /// or the [`Reader`] is dropped, whichever comes first.
    pub unsafe fn new(ptr: *mut u8, len: usize) -> Self {
        Self { ptr, len }
    }

    /// The bytes `offset..offset + len` of this destination, for a worker that fills only that
    /// part. Bounds-checked; the sub-range keeps the contract of [`HostDst::new`], and ranges that
    /// do not overlap may be filled on different threads at once.
    pub fn sub(&self, offset: usize, len: usize) -> io::Result<HostDst> {
        if offset.checked_add(len).is_none_or(|end| end > self.len) {
            return Err(io::Error::other(format!(
                "sub-range {offset}+{len} outside a destination of {}",
                self.len
            )));
        }
        // SAFETY: `offset + len <= self.len`, so the pointer stays inside the allocation that
        // `new` vouched for; the caller of `new` keeps it valid and unaliased.
        Ok(HostDst {
            ptr: unsafe { self.ptr.add(offset) },
            len,
        })
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(crate) fn write(&self, bytes: &[u8]) -> io::Result<()> {
        if bytes.len() != self.len {
            return Err(io::Error::other(format!(
                "range of {} bytes for a destination of {}",
                bytes.len(),
                self.len
            )));
        }
        // SAFETY: `new` guarantees `ptr` is valid for `len` writes and not aliased while the job
        // runs; the length was checked above and the source is a separate allocation.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.ptr, self.len) };
        Ok(())
    }
}

/// The bytes `offset..offset + len` of a tensor in a host buffer (a CPU buffer), as a destination.
/// Errors when the tensor is not in host memory or the range is outside it.
///
/// Not checkable by the type system: the tensor and its buffer must stay alive, and the range
/// untouched by anyone else, until the job that gets the destination has finished. The expert
/// streamer keeps that: it pins the slots until it has waited for the job and drops the reader
/// before the RAM tier.
#[cfg(feature = "ggml")]
pub fn tensor_dst(
    t: tetraeder_ggml_sys::ggml::Tensor<'_>,
    offset: usize,
    len: usize,
) -> io::Result<HostDst> {
    let base = t
        .host_ptr()
        .ok_or_else(|| io::Error::other("tensor is not in host memory"))?;
    if offset.checked_add(len).is_none_or(|end| end > t.nbytes()) {
        return Err(io::Error::other("range outside the tensor"));
    }
    // SAFETY: the range is inside the tensor's data (checked above); the liveness and exclusivity
    // for the job's duration are the caller's obligation, documented above.
    Ok(unsafe { HostDst::new(base.add(offset), len) })
}

/// A sink that copies range `i` into `dsts[i]`; the lengths must match.
pub fn copy_sink(dsts: Vec<HostDst>) -> Sink {
    Box::new(move |i, bytes| match dsts.get(i) {
        Some(d) => d.write(bytes),
        None => Err(io::Error::other(format!("no destination for range {i}"))),
    })
}

/// What a finished job moved and how long the worker took (device reads plus the sink).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReadDone {
    pub stats: ReadStats,
    pub took: Duration,
}

/// Claim on a submitted job; give it to [`Reader::wait`].
#[must_use = "a job is only known to be finished after `wait`"]
pub struct Ticket(Receiver<io::Result<ReadDone>>);

type Submission = (Vec<ReadJob>, Sender<io::Result<ReadDone>>);

/// One worker thread running submitted jobs in order. Dropping the reader waits for the queue.
pub struct Reader {
    tx: Option<Sender<Submission>>,
    worker: Option<JoinHandle<()>>,
}

impl Default for Reader {
    fn default() -> Self {
        Self::new()
    }
}

impl Reader {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel::<Submission>();
        let worker = std::thread::Builder::new()
            .name("tetraeder-reader".into())
            .spawn(move || {
                for (jobs, done) in rx {
                    let _ = done.send(run(jobs));
                }
            })
            .expect("spawn the reader thread");
        Self {
            tx: Some(tx),
            worker: Some(worker),
        }
    }

    /// Queues `jobs`, run one after the other as one unit (for example one per shard).
    pub fn submit(&self, jobs: Vec<ReadJob>) -> Ticket {
        let (done, rx) = mpsc::channel();
        // A send fails only when the worker is gone (a sink panicked); `done` is dropped with
        // the message and `wait` reports the closed channel.
        if let Some(tx) = &self.tx {
            let _ = tx.send((jobs, done));
        }
        Ticket(rx)
    }

    /// Blocks until the job of `ticket` has finished. After it returns, the job's destinations
    /// are not touched any more.
    pub fn wait(&self, ticket: Ticket) -> io::Result<ReadDone> {
        ticket
            .0
            .recv()
            .unwrap_or_else(|_| Err(io::Error::other("the reader thread stopped")))
    }
}

impl Drop for Reader {
    fn drop(&mut self) {
        // Closing the channel ends the worker after it has run every queued job.
        self.tx = None;
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

fn run(jobs: Vec<ReadJob>) -> io::Result<ReadDone> {
    let t0 = Instant::now();
    let mut stats = ReadStats::default();
    for mut job in jobs {
        let s = job.file.read_batch_with(&job.ranges, &mut job.sink)?;
        stats.requested += s.requested;
        stats.device += s.device;
    }
    Ok(ReadDone {
        stats,
        took: t0.elapsed(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dio::Options;
    use std::io::Write;
    use std::sync::Mutex;

    fn file(len: usize) -> (std::path::PathBuf, Vec<u8>) {
        let data: Vec<u8> = (0..len).map(|i| (i * 7 + i / 251) as u8).collect();
        // One file per call: tests run in parallel and a shared name would be truncated under a reader.
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("tetraeder-reader-{}-{n}", std::process::id()));
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&data)
            .unwrap();
        (path, data)
    }

    fn open(path: &std::path::Path, buffered: bool) -> Arc<DirectFile> {
        let o = Options {
            buffered,
            ..Options::default()
        };
        Arc::new(DirectFile::open_with(path, o).unwrap())
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

    fn check(buffered: bool) {
        let (path, data) = file(100_000);
        let f = open(&path, buffered);
        let mut dst: Vec<Vec<u8>> = RANGES.iter().map(|r| vec![0; r.len]).collect();
        let dsts = dst
            .iter_mut()
            // SAFETY: `dst` is not touched until `wait` returns.
            .map(|d| unsafe { HostDst::new(d.as_mut_ptr(), d.len()) })
            .collect();
        let reader = Reader::new();
        let t = reader.submit(vec![ReadJob {
            file: f,
            ranges: RANGES.to_vec(),
            sink: copy_sink(dsts),
        }]);
        let done = reader.wait(t).unwrap();
        assert_eq!(done.stats.requested, 10_000 + 4096 + 1);
        for (r, d) in RANGES.iter().zip(&dst) {
            assert_eq!(d[..], data[r.offset as usize..][..r.len]);
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn reads_into_host_memory_buffered() {
        check(true);
    }

    #[test]
    fn reads_into_host_memory_direct() {
        // Falls back to buffered where the filesystem refuses direct I/O.
        check(false);
    }

    #[test]
    fn jobs_run_in_order_and_tickets_wait_out_of_order() {
        let (path, _) = file(50_000);
        let f = open(&path, true);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let reader = Reader::new();
        let tickets: Vec<_> = (0..4)
            .map(|k| {
                let seen = seen.clone();
                reader.submit(vec![ReadJob {
                    file: f.clone(),
                    ranges: vec![Range {
                        offset: k * 10,
                        len: 8,
                    }],
                    sink: Box::new(move |_, _| {
                        seen.lock().unwrap().push(k);
                        Ok(())
                    }),
                }])
            })
            .collect();
        for t in tickets.into_iter().rev() {
            reader.wait(t).unwrap();
        }
        assert_eq!(*seen.lock().unwrap(), [0, 1, 2, 3]);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn errors_reach_the_waiter_and_the_reader_survives() {
        let (path, _) = file(10_000);
        let f = open(&path, true);
        let reader = Reader::new();
        // Past the end of the file.
        let t = reader.submit(vec![ReadJob {
            file: f.clone(),
            ranges: vec![Range {
                offset: 9_000,
                len: 5_000,
            }],
            sink: Box::new(|_, _| Ok(())),
        }]);
        assert!(reader.wait(t).is_err());
        // A failing sink.
        let t = reader.submit(vec![ReadJob {
            file: f.clone(),
            ranges: vec![Range { offset: 0, len: 10 }],
            sink: Box::new(|_, _| Err(io::Error::other("sink failed"))),
        }]);
        assert_eq!(reader.wait(t).unwrap_err().to_string(), "sink failed");
        // A destination of the wrong size.
        let mut one = [0u8; 3];
        // SAFETY: `one` is not touched until `wait` returns.
        let d = unsafe { HostDst::new(one.as_mut_ptr(), one.len()) };
        let t = reader.submit(vec![ReadJob {
            file: f.clone(),
            ranges: vec![Range { offset: 0, len: 10 }],
            sink: copy_sink(vec![d]),
        }]);
        assert!(reader.wait(t).is_err());
        // Still serving.
        let t = reader.submit(vec![ReadJob {
            file: f,
            ranges: vec![Range { offset: 0, len: 10 }],
            sink: Box::new(|_, _| Ok(())),
        }]);
        reader.wait(t).unwrap();
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn drop_waits_for_jobs_in_flight() {
        let (path, _) = file(10_000);
        let f = open(&path, true);
        let finished = Arc::new(Mutex::new(0));
        let reader = Reader::new();
        for _ in 0..3 {
            let n = finished.clone();
            let _ticket = reader.submit(vec![ReadJob {
                file: f.clone(),
                ranges: vec![Range { offset: 0, len: 10 }],
                sink: Box::new(move |_, _| {
                    std::thread::sleep(Duration::from_millis(30));
                    *n.lock().unwrap() += 1;
                    Ok(())
                }),
            }]);
        }
        drop(reader);
        assert_eq!(*finished.lock().unwrap(), 3);
        let _ = std::fs::remove_file(path);
    }
}
