//! Aligned direct I/O for expert streaming (ADR 0031): read `(offset, len)` ranges of a file
//! without the page cache.
//!
//! Linux opens with `O_DIRECT` and reads through io_uring when the kernel and the container
//! allow it, else through a thread pool. Windows opens with `FILE_FLAG_NO_BUFFERING` and issues
//! overlapped reads. Both need aligned offsets, lengths and buffers, so every range is widened
//! to the enclosing 4 KiB-aligned range, read into an aligned bounce buffer and copied out.
//! 4 KiB covers 512 B and 4 KiB sectors. When the filesystem refuses direct I/O (tmpfs, some
//! overlay and fuse mounts) the file is read buffered instead and [`DirectFile::fallback`]
//! says why.

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;
use std::ptr::NonNull;

#[cfg(target_os = "linux")]
mod linux;
#[cfg(unix)]
mod pool;
#[cfg(windows)]
mod windows;

/// Alignment of offsets, lengths and buffers in direct mode.
pub const ALIGN: usize = 4096;
/// Largest single read submitted to the device; bigger ranges are split so the queue stays
/// busy and no length overflows a 32-bit API.
const MAX_CHUNK: usize = 64 << 20;

/// A byte range of a file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Range {
    pub offset: u64,
    pub len: usize,
}

/// One range and where to put it.
pub struct ReadReq<'a> {
    pub offset: u64,
    pub dst: &'a mut [u8],
}

/// Bytes moved by a batch: what the caller asked for and what the device delivered
/// (the aligned enclosing ranges), so read amplification is visible.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReadStats {
    pub requested: u64,
    pub device: u64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Options {
    /// Queue depth for io_uring and overlapped reads; thread count for the pool. 0: default.
    pub queue_depth: usize,
    /// Open buffered from the start (tests, filesystems known to refuse direct I/O).
    pub buffered: bool,
}

/// Page-aligned heap buffer.
pub struct AlignedBuf {
    ptr: NonNull<u8>,
    len: usize,
}

// SAFETY: the buffer owns its allocation; the raw pointer is not shared.
unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}

impl AlignedBuf {
    /// Zeroed buffer of `len` bytes (rounded up to a multiple of [`ALIGN`], at least one block).
    pub fn zeroed(len: usize) -> Self {
        let len = len.max(1).next_multiple_of(ALIGN);
        let layout = Layout::from_size_align(len, ALIGN).expect("buffer size");
        // SAFETY: the layout has non-zero size.
        let raw = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(raw).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        Self { ptr, len }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is valid for len initialized (zeroed) bytes while self lives.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, and &mut self is exclusive.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.len, ALIGN).expect("buffer size");
        // SAFETY: allocated in `zeroed` with this layout.
        unsafe { dealloc(self.ptr.as_ptr(), layout) }
    }
}

/// One device read: `len` bytes at `offset` into `ptr`. `filled` is set by the engine.
pub(crate) struct Job {
    pub offset: u64,
    pub ptr: *mut u8,
    pub len: usize,
    pub filled: usize,
}

// SAFETY: a job is plain data; the engine contract below keeps the buffer valid and unaliased.
unsafe impl Send for Job {}

impl Job {
    /// # Safety
    /// `ptr` must be valid for writes of `len` bytes, not accessed by anything else, until the
    /// engine call that gets this job returns.
    pub(crate) unsafe fn new(offset: u64, ptr: *mut u8, len: usize) -> Self {
        Self {
            offset,
            ptr,
            len,
            filled: 0,
        }
    }
}

/// Runs jobs to completion. All I/O has finished (or failed) when `read` returns: the buffers
/// of the jobs are not touched afterwards. A read that ends at end of file fills less than
/// `len`.
pub(crate) trait Engine: Send + Sync {
    fn name(&self) -> &'static str;
    fn read(&self, file: &File, jobs: &mut [Job]) -> io::Result<()>;
}

/// A file read by ranges, direct when the platform and filesystem allow it.
pub struct DirectFile {
    file: File,
    size: u64,
    direct: bool,
    fallback: Option<String>,
    engine: Box<dyn Engine>,
}

impl std::fmt::Debug for DirectFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DirectFile")
            .field("size", &self.size)
            .field("direct", &self.direct)
            .field("fallback", &self.fallback)
            .field("backend", &self.engine.name())
            .finish()
    }
}

impl DirectFile {
    pub fn open(path: &Path) -> io::Result<Self> {
        Self::open_with(path, Options::default())
    }

    pub fn open_with(path: &Path, opts: Options) -> io::Result<Self> {
        let mut fallback = None;
        let mut direct = false;
        let mut file = None;
        if opts.buffered {
            fallback = Some("buffered I/O requested".to_string());
        } else if !direct_supported() {
            fallback = Some("direct I/O is not implemented on this OS".to_string());
        } else {
            match open_file(path, true) {
                Ok(f) => {
                    file = Some(f);
                    direct = true;
                }
                Err(e) => fallback = Some(format!("direct open refused: {e}")),
            }
        }
        let file = match file {
            Some(f) => f,
            None => open_file(path, false)?,
        };
        let size = file.metadata()?.len();
        let mut this = Self {
            engine: new_engine(&opts),
            file,
            size,
            direct,
            fallback,
        };
        // Some mounts open with O_DIRECT and fail the first read.
        if direct && size > 0 {
            let mut probe = [0u8; 1];
            if let Err(e) = this.read_at(0, &mut probe) {
                this.file = open_file(path, false)?;
                this.engine = new_engine(&opts);
                this.direct = false;
                this.fallback = Some(format!("direct read refused: {e}"));
            }
        }
        Ok(this)
    }

    /// True when reads bypass the page cache.
    pub fn is_direct(&self) -> bool {
        self.direct
    }

    /// Why the file is read buffered, if it is.
    pub fn fallback(&self) -> Option<&str> {
        self.fallback.as_deref()
    }

    /// `io_uring`, `thread-pool` or `overlapped`.
    pub fn backend(&self) -> &'static str {
        self.engine.name()
    }

    pub fn len(&self) -> u64 {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Read exactly `dst.len()` bytes at `offset`.
    pub fn read_at(&self, offset: u64, dst: &mut [u8]) -> io::Result<ReadStats> {
        self.read_batch(&mut [ReadReq { offset, dst }])
    }

    /// Read every range; the device sees them all at once (up to the queue depth).
    pub fn read_batch(&self, reqs: &mut [ReadReq<'_>]) -> io::Result<ReadStats> {
        for r in reqs.iter() {
            let end = r.offset.checked_add(r.dst.len() as u64);
            if end.is_none_or(|e| e > self.size) {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!(
                        "range {}+{} is past the end of the file ({} bytes)",
                        r.offset,
                        r.dst.len(),
                        self.size
                    ),
                ));
            }
        }
        if self.direct {
            self.read_direct(reqs)
        } else {
            self.read_exact(reqs)
        }
    }

    /// Buffered: the destination is read into directly, no widening.
    fn read_exact(&self, reqs: &mut [ReadReq<'_>]) -> io::Result<ReadStats> {
        let mut jobs = Vec::new();
        let mut stats = ReadStats::default();
        for r in reqs.iter_mut() {
            stats.requested += r.dst.len() as u64;
            let mut at = 0;
            while at < r.dst.len() {
                let n = (r.dst.len() - at).min(MAX_CHUNK);
                // SAFETY: dst is exclusively borrowed for this call and the jobs finish in it.
                jobs.push(unsafe { Job::new(r.offset + at as u64, r.dst[at..].as_mut_ptr(), n) });
                at += n;
            }
        }
        self.engine.read(&self.file, &mut jobs)?;
        for j in &jobs {
            stats.device += j.filled as u64;
            if j.filled < j.len {
                return Err(short(j));
            }
        }
        Ok(stats)
    }

    /// Direct: each range is widened to aligned bounds, read into a bounce buffer, copied out.
    fn read_direct(&self, reqs: &mut [ReadReq<'_>]) -> io::Result<ReadStats> {
        struct Bounce {
            buf: AlignedBuf,
            /// Offset of the range inside the buffer.
            skip: usize,
            first_job: usize,
            jobs: usize,
        }
        let mut stats = ReadStats::default();
        let mut bounces: Vec<Option<Bounce>> = Vec::with_capacity(reqs.len());
        let mut jobs: Vec<Job> = Vec::new();
        for r in reqs.iter() {
            stats.requested += r.dst.len() as u64;
            if r.dst.is_empty() {
                bounces.push(None);
                continue;
            }
            let a = ALIGN as u64;
            let start = r.offset / a * a;
            let end = (r.offset + r.dst.len() as u64).next_multiple_of(a);
            let mut buf = AlignedBuf::zeroed((end - start) as usize);
            let (first_job, total) = (jobs.len(), (end - start) as usize);
            let mut at = 0;
            while at < total {
                let n = (total - at).min(MAX_CHUNK);
                // SAFETY: the buffer lives in `bounces` until the engine call returns, and its
                // heap block does not move when the Vec of bounces grows.
                jobs.push(unsafe { Job::new(start + at as u64, buf.as_mut_ptr().add(at), n) });
                at += n;
            }
            bounces.push(Some(Bounce {
                skip: (r.offset - start) as usize,
                first_job,
                jobs: jobs.len() - first_job,
                buf,
            }));
        }
        self.engine.read(&self.file, &mut jobs)?;
        for (r, b) in reqs.iter_mut().zip(&bounces) {
            let Some(b) = b else { continue };
            // Bytes available from the start of the enclosing range: chunks in order, up to the
            // first short one (end of file).
            let mut have = 0;
            for j in &jobs[b.first_job..b.first_job + b.jobs] {
                have += j.filled;
                stats.device += j.filled as u64;
                if j.filled < j.len {
                    break;
                }
            }
            let need = b.skip + r.dst.len();
            if have < need {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("direct read of {need} bytes at {} got {have}", r.offset),
                ));
            }
            r.dst.copy_from_slice(&b.buf.as_slice()[b.skip..need]);
        }
        Ok(stats)
    }
}

fn short(j: &Job) -> io::Error {
    io::Error::new(
        io::ErrorKind::UnexpectedEof,
        format!("read of {} bytes at {} got {}", j.len, j.offset, j.filled),
    )
}

const fn direct_supported() -> bool {
    cfg!(any(target_os = "linux", windows))
}

fn open_file(path: &Path, direct: bool) -> io::Result<File> {
    let mut o = OpenOptions::new();
    o.read(true);
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        if direct {
            o.custom_flags(libc::O_DIRECT);
        }
    }
    #[cfg(windows)]
    {
        use ::windows_sys::Win32::Storage::FileSystem::{
            FILE_FLAG_NO_BUFFERING, FILE_FLAG_OVERLAPPED,
        };
        use std::os::windows::fs::OpenOptionsExt;
        o.custom_flags(FILE_FLAG_OVERLAPPED | if direct { FILE_FLAG_NO_BUFFERING } else { 0 });
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    let _ = direct;
    o.open(path)
}

#[cfg(windows)]
fn new_engine(opts: &Options) -> Box<dyn Engine> {
    Box::new(windows::Overlapped::new(opts.queue_depth))
}

#[cfg(target_os = "linux")]
fn new_engine(opts: &Options) -> Box<dyn Engine> {
    match linux::Uring::new(opts.queue_depth) {
        Ok(u) => Box::new(u),
        Err(_) => Box::new(pool::Pool::new(opts.queue_depth)),
    }
}

#[cfg(all(unix, not(target_os = "linux")))]
fn new_engine(opts: &Options) -> Box<dyn Engine> {
    Box::new(pool::Pool::new(opts.queue_depth))
}
