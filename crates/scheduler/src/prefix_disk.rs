//! The disk tier of the prefix cache (ADR 0087): entries the RAM budget evicts are written to
//! files by a background thread, and read back by a second one (addendum 2026-11-23), so the
//! scheduler step never waits on disk IO. An entry is findable only once its write completed.
//! The files are valid for this process only: they are removed at startup and on drop, and a
//! write that fails drops the entry with a warning.

use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{
    Receiver, SendError, Sender, SyncSender, TryRecvError, TrySendError, channel, sync_channel,
};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tetraeder_model::snapshot_pool::SnapshotPool;

use crate::prefix_cache::common_prefix;

const MAGIC: &[u8; 8] = b"TTRPFX01";
const VERSION: u32 = 1;
const FILE_PREFIX: &str = "tetraeder-prefix-";
const FILE_EXT: &str = ".tkv";
/// Writes queued for the writer. Their buffers are held outside the RAM budget, so the queue is
/// short; a spill that finds it full is dropped.
const QUEUE: usize = 2;
/// Reads in flight at once. Each holds a pool buffer outside the RAM budget; a request that
/// finds the limit reached is placed without the disk entry.
pub const MAX_READS: usize = 2;

/// Reads a file back: path, format, payload size, pool for the buffer. Injectable for tests.
pub type ReadFn =
    Arc<dyn Fn(&Path, &str, usize, &SnapshotPool) -> io::Result<(Vec<i32>, Vec<u8>)> + Send + Sync>;

struct DiskEntry {
    id: u64,
    format: String,
    tokens: Vec<i32>,
    /// Payload bytes, counted against the budget from the moment the write is queued.
    size: usize,
    /// The write completed.
    ready: bool,
    /// A background read of it is running: it is not offered again and not evicted.
    reading: bool,
}

struct ReadJob {
    path: PathBuf,
    format: String,
    size: usize,
    reply: Sender<ReadDone>,
}

/// What the reader thread found.
pub struct ReadDone {
    pub res: io::Result<(Vec<i32>, Vec<u8>)>,
    /// The time of the file read alone.
    pub took: Duration,
}

/// What a finished background read gives: the entry's tokens and bytes (`None` when the file was
/// unreadable) and the time of the read.
pub type Restored = (Option<(Vec<i32>, Vec<u8>)>, Duration);

/// A background read of one disk entry; end it with `poll_read` (finished) or `abort_read`.
pub struct ReadHandle {
    id: u64,
    rx: Receiver<ReadDone>,
}

struct Job {
    id: u64,
    path: PathBuf,
    format: String,
    tokens: Vec<i32>,
    bytes: Vec<u8>,
}

struct Done {
    id: u64,
    path: PathBuf,
    ok: bool,
    bytes: Vec<u8>,
}

pub struct DiskTier {
    dir: PathBuf,
    budget: usize,
    held: usize,
    next_id: u64,
    /// Oldest first.
    entries: Vec<DiskEntry>,
    jobs: Option<SyncSender<Job>>,
    done: Receiver<Done>,
    writer: Option<JoinHandle<()>>,
    reads: Option<Sender<ReadJob>>,
    reader: Option<JoinHandle<()>>,
    pool: Arc<SnapshotPool>,
}

impl DiskTier {
    /// Creates `dir` if needed and removes the files an earlier process left there.
    pub fn open(
        dir: &Path,
        budget: usize,
        pool: Arc<SnapshotPool>,
        read: Option<ReadFn>,
    ) -> Result<Self, String> {
        fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        remove_stale(dir);
        let (jobs, rx) = sync_channel::<Job>(QUEUE);
        let (tx, done) = channel::<Done>();
        let writer = std::thread::Builder::new()
            .name("tetraeder-prefix-disk".into())
            .spawn(move || writer(&rx, &tx))
            .map_err(|e| format!("cannot start the disk writer: {e}"))?;
        let (reads, read_rx) = channel::<ReadJob>();
        let read: ReadFn = read.unwrap_or_else(|| Arc::new(read_file));
        let reader_pool = Arc::clone(&pool);
        let reader = std::thread::Builder::new()
            .name("tetraeder-prefix-read".into())
            .spawn(move || reader(&read_rx, &reader_pool, &read))
            .map_err(|e| format!("cannot start the disk reader: {e}"))?;
        Ok(Self {
            dir: dir.to_owned(),
            budget,
            held: 0,
            next_id: 0,
            entries: Vec::new(),
            jobs: Some(jobs),
            done,
            writer: Some(writer),
            reads: Some(reads),
            reader: Some(reader),
            pool,
        })
    }

    /// Payload bytes held or queued.
    pub fn bytes(&self) -> usize {
        self.held
    }

    /// Entries whose write completed.
    pub fn len(&self) -> usize {
        self.entries.iter().filter(|e| e.ready).count()
    }

    fn path(&self, id: u64) -> PathBuf {
        self.dir.join(format!(
            "{FILE_PREFIX}{}-{id}{FILE_EXT}",
            std::process::id()
        ))
    }

    /// Takes in the writes that finished.
    pub fn poll(&mut self) {
        while let Ok(d) = self.done.try_recv() {
            self.apply(d);
        }
    }

    fn apply(&mut self, d: Done) {
        self.pool.recycle(d.bytes);
        match self.entries.iter().position(|e| e.id == d.id) {
            Some(i) if d.ok => self.entries[i].ready = true,
            Some(i) => {
                // The writer warned.
                self.held -= self.entries.remove(i).size;
            }
            // Replaced or evicted while it was queued.
            None => remove_file(&d.path),
        }
    }

    /// Waits until every queued write is done (tests).
    pub fn flush(&mut self) {
        let t0 = std::time::Instant::now();
        self.poll();
        while self.entries.iter().any(|e| !e.ready) && t0.elapsed().as_secs() < 30 {
            if let Ok(d) = self.done.recv_timeout(std::time::Duration::from_millis(20)) {
                self.apply(d);
            }
        }
    }

    pub fn holds(&self, format: &str, tokens: &[i32]) -> bool {
        self.entries
            .iter()
            .any(|e| e.format == format && e.tokens.starts_with(tokens))
    }

    /// The completed entry of `format` sharing the longest prefix with `prompt`: its index and
    /// that length.
    pub fn best(&self, format: &str, prompt: &[i32]) -> Option<(usize, usize)> {
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.ready && !e.reading && e.format == format)
            .map(|(i, e)| (i, common_prefix(&e.tokens, prompt)))
            .filter(|&(_, n)| n > 0)
            .max_by_key(|&(_, n)| n)
    }

    /// Drops the entries of `format` that are a prefix of `tokens`.
    pub fn replace_prefixes(&mut self, format: &str, tokens: &[i32]) {
        let mut i = 0;
        while i < self.entries.len() {
            let e = &self.entries[i];
            if !e.reading && e.format == format && tokens.starts_with(&e.tokens) {
                self.drop_entry(i);
            } else {
                i += 1;
            }
        }
    }

    fn drop_entry(&mut self, i: usize) {
        let e = self.entries.remove(i);
        self.held -= e.size;
        // A queued write deletes its own file when it finds the entry gone.
        if e.ready {
            remove_file(&self.path(e.id));
        }
    }

    /// Queues a write of an entry the RAM tier evicted. The buffer goes back to the pool when
    /// the write is done, or at once when the entry is not kept.
    pub fn spill(&mut self, format: &str, tokens: Vec<i32>, bytes: Vec<u8>) {
        self.poll();
        if bytes.len() > self.budget || self.holds(format, &tokens) {
            self.pool.recycle(bytes);
            return;
        }
        let size = bytes.len();
        let id = self.next_id;
        self.next_id += 1;
        let job = Job {
            id,
            path: self.path(id),
            format: format.to_owned(),
            tokens: tokens.clone(),
            bytes,
        };
        let Some(jobs) = &self.jobs else {
            self.pool.recycle(job.bytes);
            return;
        };
        match jobs.try_send(job) {
            Ok(()) => {}
            Err(TrySendError::Full(job) | TrySendError::Disconnected(job)) => {
                tracing::debug!("prefix cache disk writer is busy: an evicted entry is dropped");
                self.pool.recycle(job.bytes);
                return;
            }
        }
        self.replace_prefixes(format, &tokens);
        while self.held + size > self.budget
            && let Some(k) = self.entries.iter().position(|e| !e.reading)
        {
            self.drop_entry(k);
        }
        self.held += size;
        self.entries.push(DiskEntry {
            id,
            format: format.to_owned(),
            tokens,
            size,
            ready: false,
            reading: false,
        });
    }

    /// Starts reading entry `i` into a pool buffer on the reader thread. `None` when the entry
    /// is not complete or is being read, or `MAX_READS` reads are running. The entry stays
    /// until `poll_read` consumes it, so a cancelled restore loses nothing.
    pub fn start_read(&mut self, i: usize) -> Option<ReadHandle> {
        if self.reading() >= MAX_READS
            || !self.entries.get(i).is_some_and(|e| e.ready && !e.reading)
        {
            return None;
        }
        let id = self.entries[i].id;
        let (reply, rx) = channel();
        let job = ReadJob {
            path: self.path(id),
            format: self.entries[i].format.clone(),
            size: self.entries[i].size,
            reply,
        };
        self.reads.as_ref()?.send(job).ok()?;
        self.entries[i].reading = true;
        Some(ReadHandle { id, rx })
    }

    /// Reads in flight.
    pub fn reading(&self) -> usize {
        self.entries.iter().filter(|e| e.reading).count()
    }

    /// The result of `read` if it is in: the entry's tokens and bytes and the read time. The
    /// entry and its file are gone then; an unreadable file is dropped with a warning and
    /// gives no bytes.
    pub fn poll_read(&mut self, read: &ReadHandle) -> Option<Restored> {
        let done = match read.rx.try_recv() {
            Ok(d) => d,
            Err(TryRecvError::Empty) => return None,
            Err(TryRecvError::Disconnected) => ReadDone {
                res: Err(bad("the disk reader stopped")),
                took: Duration::ZERO,
            },
        };
        if let Some(i) = self.entries.iter().position(|e| e.id == read.id) {
            let e = self.entries.remove(i);
            self.held -= e.size;
        }
        remove_file(&self.path(read.id));
        match done.res {
            Ok(r) => Some((Some(r), done.took)),
            Err(err) => {
                tracing::warn!("cannot read a saved sequence from disk: {err}");
                Some((None, done.took))
            }
        }
    }

    /// Gives up `read`: the entry and its file stay. The reader thread hands its buffer back
    /// to the pool when it finishes.
    pub fn abort_read(&mut self, read: ReadHandle) {
        if let Some(e) = self.entries.iter_mut().find(|e| e.id == read.id) {
            e.reading = false;
        }
    }
}

impl Drop for DiskTier {
    fn drop(&mut self) {
        self.jobs = None;
        self.reads = None;
        for t in [self.writer.take(), self.reader.take()]
            .into_iter()
            .flatten()
        {
            let _ = t.join();
        }
        remove_stale(&self.dir);
    }
}

fn remove_file(path: &Path) {
    if let Err(e) = fs::remove_file(path)
        && e.kind() != io::ErrorKind::NotFound
    {
        tracing::warn!("cannot remove {}: {e}", path.display());
    }
}

/// Removes the files of this cache (any process) in `dir`.
fn remove_stale(dir: &Path) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    for f in rd.flatten() {
        let name = f.file_name();
        let name = name.to_string_lossy();
        if name.starts_with(FILE_PREFIX) && (name.ends_with(FILE_EXT) || name.ends_with(".tmp")) {
            remove_file(&f.path());
        }
    }
}

fn writer(rx: &Receiver<Job>, tx: &Sender<Done>) {
    for job in rx {
        let ok = match write_file(&job.path, &job.format, &job.tokens, &job.bytes) {
            Ok(()) => true,
            Err(e) => {
                tracing::warn!(
                    "cannot write a saved sequence to {}: {e}; the entry is dropped",
                    job.path.display()
                );
                false
            }
        };
        let done = Done {
            id: job.id,
            path: job.path,
            ok,
            bytes: job.bytes,
        };
        if tx.send(done).is_err() {
            return;
        }
    }
}

fn reader(rx: &Receiver<ReadJob>, pool: &SnapshotPool, read: &ReadFn) {
    for job in rx {
        let t0 = Instant::now();
        let res = read(&job.path, &job.format, job.size, pool);
        let done = ReadDone {
            res,
            took: t0.elapsed(),
        };
        // Nobody waits any more (the request was cancelled): the buffer goes back to the pool.
        if let Err(SendError(ReadDone {
            res: Ok((_, bytes)),
            ..
        })) = job.reply.send(done)
        {
            pool.recycle(bytes);
        }
    }
}

fn tmp_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".tmp");
    PathBuf::from(s)
}

/// Magic, version, format, tokens, bytes; integers little endian. Written under a temporary name
/// and renamed, so a file under its final name is complete.
pub fn write_file(path: &Path, format: &str, tokens: &[i32], bytes: &[u8]) -> io::Result<()> {
    let tmp = tmp_path(path);
    let res = (|| {
        let mut w = BufWriter::new(File::create(&tmp)?);
        w.write_all(MAGIC)?;
        w.write_all(&VERSION.to_le_bytes())?;
        w.write_all(&(format.len() as u32).to_le_bytes())?;
        w.write_all(format.as_bytes())?;
        w.write_all(&(tokens.len() as u64).to_le_bytes())?;
        for t in tokens {
            w.write_all(&t.to_le_bytes())?;
        }
        w.write_all(&(bytes.len() as u64).to_le_bytes())?;
        w.write_all(bytes)?;
        w.flush()?;
        drop(w);
        fs::rename(&tmp, path)
    })();
    if res.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    res
}

fn bad(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_owned())
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut b = [0; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut b = [0; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

/// Reads a file `write_file` made. `format` and `size` are what the index says it holds.
pub fn read_file(
    path: &Path,
    format: &str,
    size: usize,
    pool: &SnapshotPool,
) -> io::Result<(Vec<i32>, Vec<u8>)> {
    let mut r = io::BufReader::new(File::open(path)?);
    let mut magic = [0; 8];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC || read_u32(&mut r)? != VERSION {
        return Err(bad("not a prefix cache file of this version"));
    }
    let flen = read_u32(&mut r)? as usize;
    if flen != format.len() {
        return Err(bad("format mismatch"));
    }
    let mut f = vec![0; flen];
    r.read_exact(&mut f)?;
    if f != format.as_bytes() {
        return Err(bad("format mismatch"));
    }
    let n = read_u64(&mut r)? as usize;
    let mut raw = vec![0u8; n.checked_mul(4).ok_or_else(|| bad("bad token count"))?];
    r.read_exact(&mut raw)?;
    let tokens = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let len = read_u64(&mut r)? as usize;
    if len != size {
        return Err(bad("size mismatch"));
    }
    let mut bytes = pool.take(len);
    let got = (&mut r).take(len as u64).read_to_end(&mut bytes)?;
    if got != len {
        pool.recycle(bytes);
        return Err(bad("truncated"));
    }
    Ok((tokens, bytes))
}
