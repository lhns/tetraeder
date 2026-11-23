//! The disk tier of the prefix cache (ADR 0087): entries the RAM budget evicts are written to
//! files by a background thread, so the scheduler step never waits on disk IO. An entry is
//! findable only once its write completed. The files are valid for this process only: they are
//! removed at startup and on drop, and a write that fails drops the entry with a warning.

use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, SyncSender, TrySendError, channel, sync_channel};
use std::thread::JoinHandle;
use tetraeder_model::snapshot_pool::SnapshotPool;

use crate::prefix_cache::common_prefix;

const MAGIC: &[u8; 8] = b"TTRPFX01";
const VERSION: u32 = 1;
const FILE_PREFIX: &str = "tetraeder-prefix-";
const FILE_EXT: &str = ".tkv";
/// Writes queued for the writer. Their buffers are held outside the RAM budget, so the queue is
/// short; a spill that finds it full is dropped.
const QUEUE: usize = 2;

struct DiskEntry {
    id: u64,
    format: String,
    tokens: Vec<i32>,
    /// Payload bytes, counted against the budget from the moment the write is queued.
    size: usize,
    /// The write completed.
    ready: bool,
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
    pool: Arc<SnapshotPool>,
}

impl DiskTier {
    /// Creates `dir` if needed and removes the files an earlier process left there.
    pub fn open(dir: &Path, budget: usize, pool: Arc<SnapshotPool>) -> Result<Self, String> {
        fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        remove_stale(dir);
        let (jobs, rx) = sync_channel::<Job>(QUEUE);
        let (tx, done) = channel::<Done>();
        let writer = std::thread::Builder::new()
            .name("tetraeder-prefix-disk".into())
            .spawn(move || writer(&rx, &tx))
            .map_err(|e| format!("cannot start the disk writer: {e}"))?;
        Ok(Self {
            dir: dir.to_owned(),
            budget,
            held: 0,
            next_id: 0,
            entries: Vec::new(),
            jobs: Some(jobs),
            done,
            writer: Some(writer),
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
            .filter(|(_, e)| e.ready && e.format == format)
            .map(|(i, e)| (i, common_prefix(&e.tokens, prompt)))
            .filter(|&(_, n)| n > 0)
            .max_by_key(|&(_, n)| n)
    }

    /// Drops the entries of `format` that are a prefix of `tokens`.
    pub fn replace_prefixes(&mut self, format: &str, tokens: &[i32]) {
        let mut i = 0;
        while i < self.entries.len() {
            let e = &self.entries[i];
            if e.format == format && tokens.starts_with(&e.tokens) {
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
        while self.held + size > self.budget && !self.entries.is_empty() {
            self.drop_entry(0);
        }
        self.held += size;
        self.entries.push(DiskEntry {
            id,
            format: format.to_owned(),
            tokens,
            size,
            ready: false,
        });
    }

    /// Reads entry `i` back into a pool buffer and deletes it. An unreadable file is dropped
    /// with a warning.
    pub fn take(&mut self, i: usize) -> Option<(Vec<i32>, Vec<u8>)> {
        if !self.entries.get(i).is_some_and(|e| e.ready) {
            return None;
        }
        let e = self.entries.remove(i);
        self.held -= e.size;
        let path = self.path(e.id);
        let read = read_file(&path, &e.format, e.size, &self.pool);
        remove_file(&path);
        match read {
            Ok(r) => Some(r),
            Err(err) => {
                tracing::warn!("cannot read a saved sequence from disk: {err}");
                None
            }
        }
    }
}

impl Drop for DiskTier {
    fn drop(&mut self) {
        self.jobs = None;
        if let Some(w) = self.writer.take() {
            let _ = w.join();
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
