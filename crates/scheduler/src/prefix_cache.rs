//! Saved sequences in host RAM, the oldest evicted first (ADR 0049). A restored entry leaves the
//! cache (the sequence is live again and is saved anew when it is dropped), so the age of an
//! entry is the time since it was saved or last used. An entry is keyed by its KV format too:
//! bytes of one format are never restored into a cache of another (ADR 0050 b). Buffers the cache
//! lets go of go back to the snapshot pool instead of being freed (ADR 0074). With a disk tier
//! (ADR 0087) an entry the RAM budget evicts spills to a file; indices run over the RAM entries,
//! then the completed disk entries. A disk entry is restored by a background read
//! (`start_restore`), never by `take`.

use crate::prefix_disk::{DiskTier, ReadFn, ReadHandle, Restored};
use std::path::Path;
use std::sync::Arc;
use tetraeder_model::snapshot_pool::{self, SnapshotPool};

/// A live sequence is saved before it loses at least this many tokens.
pub const SAVE_MIN_TOKENS: usize = 512;
/// A saved entry is restored when it beats the live sequence by at least this many tokens.
pub const RESTORE_MIN_GAIN: usize = 64;

struct Entry {
    /// `KvFormat` display form, for example `q8_0/q4_0`.
    format: String,
    tokens: Vec<i32>,
    bytes: Vec<u8>,
}

/// Entries hold the tokens of a sequence and its KV cache as backend bytes. The budget counts
/// the bytes only.
pub struct PrefixCache {
    entries: Vec<Entry>,
    budget: usize,
    held: usize,
    pool: Arc<SnapshotPool>,
    disk: Option<DiskTier>,
    /// Replaces the file reader of the disk tier (tests).
    reader: Option<ReadFn>,
}

/// A background read of a disk entry, started by `PrefixCache::start_restore`.
pub struct Restore(ReadHandle);

/// Length of the common prefix of `a` and `b`.
pub fn common_prefix(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

impl PrefixCache {
    /// A budget of 0 disables the cache.
    pub fn new(budget: usize) -> Self {
        Self::with_pool(budget, snapshot_pool::global())
    }

    /// A cache that returns released buffers to `pool`.
    pub fn with_pool(budget: usize, pool: Arc<SnapshotPool>) -> Self {
        Self {
            entries: Vec::new(),
            budget,
            held: 0,
            pool,
            disk: None,
            reader: None,
        }
    }

    /// Reads disk entries with `read` instead of the file reader; call before `with_disk` (tests
    /// make it slow or failing).
    #[must_use]
    pub fn with_disk_reader(mut self, read: ReadFn) -> Self {
        self.reader = Some(read);
        self
    }

    /// Adds the disk tier: evicted entries spill to files in `dir`, up to `budget` bytes. A
    /// directory that cannot be used is a warning and no disk tier, never an error.
    #[must_use]
    pub fn with_disk(mut self, dir: &Path, budget: usize) -> Self {
        if budget == 0 {
            return self;
        }
        match DiskTier::open(dir, budget, Arc::clone(&self.pool), self.reader.clone()) {
            Ok(d) => self.disk = Some(d),
            Err(e) => tracing::warn!("prefix cache disk tier is off: {e}"),
        }
        self
    }

    /// Bytes held in files or queued for them.
    pub fn disk_bytes(&self) -> usize {
        self.disk.as_ref().map_or(0, DiskTier::bytes)
    }

    /// Waits for the queued disk writes (tests).
    pub fn flush_disk(&mut self) {
        if let Some(d) = &mut self.disk {
            d.flush();
        }
    }

    /// Where entry `i` of `best` lives: `ram` or `disk`.
    pub fn source(&self, i: usize) -> &'static str {
        if i < self.entries.len() {
            "ram"
        } else {
            "disk"
        }
    }

    pub fn pool(&self) -> &Arc<SnapshotPool> {
        &self.pool
    }

    /// Gives a buffer taken out of the cache (a restored entry's bytes) back to the pool.
    pub fn release(&self, bytes: Vec<u8>) {
        self.pool.recycle(bytes);
    }

    pub fn enabled(&self) -> bool {
        self.budget > 0
    }

    /// Host bytes held.
    pub fn bytes(&self) -> usize {
        self.held
    }

    /// Entries in RAM and completed ones on disk.
    pub fn len(&self) -> usize {
        self.entries.len() + self.disk.as_ref().map_or(0, DiskTier::len)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The entry of `format` sharing the longest prefix with `prompt`: its index and that length.
    pub fn best(&mut self, format: &str, prompt: &[i32]) -> Option<(usize, usize)> {
        self.best_by(format, prompt, false)
    }

    /// Like `best`, among the entries `prompt` starts with: the only ones a model that cannot
    /// cut a suffix can use, and taking any other would destroy an entry for nothing.
    pub fn best_whole(&mut self, format: &str, prompt: &[i32]) -> Option<(usize, usize)> {
        self.best_by(format, prompt, true)
    }

    fn best_by(&mut self, format: &str, prompt: &[i32], whole: bool) -> Option<(usize, usize)> {
        if let Some(d) = &mut self.disk {
            d.poll();
        }
        let ram = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.format == format)
            .map(|(i, e)| (i, common_prefix(&e.tokens, prompt), e.tokens.len()))
            .filter(|&(_, n, len)| n > 0 && (!whole || n == len))
            .map(|(i, n, _)| (i, n))
            .max_by_key(|&(_, n)| n);
        let disk = self
            .disk
            .as_ref()
            .and_then(|d| d.best(format, prompt, whole))
            .map(|(i, n)| (self.entries.len() + i, n));
        // RAM wins a tie: it restores faster.
        match (ram, disk) {
            (Some(r), Some(d)) if d.1 > r.1 => Some(d),
            (Some(r), _) => Some(r),
            (None, d) => d,
        }
    }

    /// Whether an entry of `format` already holds all of `tokens`.
    pub fn holds(&self, format: &str, tokens: &[i32]) -> bool {
        self.entries
            .iter()
            .any(|e| e.format == format && e.tokens.starts_with(tokens))
            || self.disk.as_ref().is_some_and(|d| d.holds(format, tokens))
    }

    /// RAM entry `i` as it is (a disk entry is not read here), for a copy to another replica.
    pub fn peek(&self, i: usize) -> Option<(&[i32], &[u8])> {
        self.entries
            .get(i)
            .map(|e| (e.tokens.as_slice(), e.bytes.as_slice()))
    }

    /// Removes RAM entry `i` and returns its tokens and bytes. A disk entry is not read here: see
    /// `start_restore`.
    pub fn take(&mut self, i: usize) -> Option<(Vec<i32>, Vec<u8>)> {
        if i >= self.entries.len() {
            return None;
        }
        let e = self.entries.remove(i);
        self.held -= e.bytes.len();
        Some((e.tokens, e.bytes))
    }

    /// Starts reading disk entry `i` in the background. `None` when `i` is not a disk entry or
    /// the tier is busy (`MAX_READS` reads). The entry stays in the cache, marked as being
    /// read, until `poll_restore` consumes it.
    pub fn start_restore(&mut self, i: usize) -> Option<Restore> {
        let n = self.entries.len();
        i.checked_sub(n)
            .and_then(|k| self.disk.as_mut()?.start_read(k))
            .map(Restore)
    }

    /// The finished read, if it is in. The entry and its file are gone then.
    pub fn poll_restore(&mut self, r: &Restore) -> Option<Restored> {
        self.disk.as_mut()?.poll_read(&r.0)
    }

    /// Gives up a read: the entry and its file stay, and the buffer goes back to the pool when
    /// the reader is done with it.
    pub fn abort_restore(&mut self, r: Restore) {
        if let Some(d) = &mut self.disk {
            d.abort_read(r.0);
        }
    }

    /// Reads in flight.
    pub fn restoring(&self) -> usize {
        self.disk.as_ref().map_or(0, DiskTier::reading)
    }

    /// Adds a sequence, replacing entries that are a prefix of it and evicting the least recently
    /// used ones to fit the budget. Returns the number of entries dropped. A sequence larger than
    /// the whole budget is not stored.
    pub fn insert(&mut self, format: &str, tokens: Vec<i32>, bytes: Vec<u8>) -> usize {
        if bytes.len() > self.budget {
            self.pool.recycle(bytes);
            return 0;
        }
        if let Some(d) = &mut self.disk {
            d.replace_prefixes(format, &tokens);
        }
        let mut dropped = 0;
        let (replaced, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut self.entries)
            .into_iter()
            .partition(|e| e.format == format && tokens.starts_with(&e.tokens));
        self.entries = kept;
        for e in replaced {
            self.held -= e.bytes.len();
            self.pool.recycle(e.bytes);
        }
        // Replacing a prefix is not an eviction.
        while self.held + bytes.len() > self.budget {
            if self.entries.is_empty() {
                break;
            }
            let e = self.entries.remove(0);
            self.held -= e.bytes.len();
            match &mut self.disk {
                Some(d) => d.spill(&e.format, e.tokens, e.bytes),
                None => self.pool.recycle(e.bytes),
            }
            dropped += 1;
        }
        self.held += bytes.len();
        self.entries.push(Entry {
            format: format.to_owned(),
            tokens,
            bytes,
        });
        dropped
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tetraeder_model::snapshot_pool::PoolConfig;

    const F: &str = "f16/f16";

    fn seq(n: usize, base: i32) -> Vec<i32> {
        (0..n as i32).map(|i| base + i).collect()
    }

    #[test]
    fn best_finds_the_longest_prefix() {
        let mut c = PrefixCache::new(1000);
        c.insert(F, seq(10, 0), vec![0; 10]);
        let mut other = seq(10, 0);
        other[5] = 99;
        c.insert(F, other, vec![0; 10]);
        let mut p = seq(10, 0);
        p.push(7);
        assert_eq!(c.best(F, &p), Some((0, 10)));
        assert_eq!(c.best(F, &[42]), None);
    }

    #[test]
    fn best_whole_skips_entries_the_prompt_only_partly_shares() {
        let mut c = PrefixCache::new(1000);
        c.insert(F, seq(10, 0), vec![0; 10]);
        let mut shorter = seq(6, 0);
        shorter[5] = 99;
        c.insert(F, shorter, vec![0; 6]);
        let mut p = seq(8, 0);
        p.push(7);
        // The 10-token entry shares 8 of its tokens: a cut. Whole is the other one, if at all.
        assert_eq!(c.best(F, &p), Some((0, 8)));
        assert_eq!(c.best_whole(F, &p), None);
        let mut q = seq(10, 0);
        q.push(7);
        assert_eq!(c.best_whole(F, &q), Some((0, 10)));
    }

    #[test]
    fn prefix_entries_are_replaced_not_evicted() {
        let mut c = PrefixCache::new(1000);
        c.insert(F, seq(5, 0), vec![0; 5]);
        assert_eq!(c.insert(F, seq(8, 0), vec![0; 8]), 0);
        assert_eq!((c.len(), c.bytes()), (1, 8));
        assert!(c.holds(F, &seq(6, 0)));
    }

    #[test]
    fn the_oldest_entry_is_evicted_first() {
        let mut c = PrefixCache::new(20);
        c.insert(F, seq(3, 0), vec![0; 10]);
        c.insert(F, seq(3, 100), vec![0; 10]);
        assert_eq!(c.insert(F, seq(3, 200), vec![0; 10]), 1);
        assert!(!c.holds(F, &seq(3, 0)) && c.holds(F, &seq(3, 100)) && c.holds(F, &seq(3, 200)));
        assert_eq!(c.bytes(), 20);
        let (t, b) = c.take(0).unwrap();
        assert_eq!((t, b.len(), c.bytes()), (seq(3, 100), 10, 10));
    }

    #[test]
    fn oversized_and_disabled() {
        let mut c = PrefixCache::new(5);
        assert_eq!(c.insert(F, seq(3, 0), vec![0; 10]), 0);
        assert!(c.is_empty());
        assert!(!PrefixCache::new(0).enabled());
    }

    #[test]
    fn formats_never_mix() {
        let mut c = PrefixCache::new(1000);
        c.insert(F, seq(5, 0), vec![0; 5]);
        assert_eq!(c.best("q8_0/q8_0", &seq(5, 0)), None);
        assert!(!c.holds("q8_0/q8_0", &seq(5, 0)));
        // the same tokens in another format are another entry, not a replaced prefix
        c.insert("q8_0/q8_0", seq(8, 0), vec![0; 8]);
        assert_eq!(c.len(), 2);
        assert_eq!(c.best(F, &seq(8, 0)), Some((0, 5)));
    }

    fn pooled(budget: usize, spares: usize) -> (PrefixCache, Arc<SnapshotPool>) {
        let pool = Arc::new(SnapshotPool::new(PoolConfig {
            spares,
            thp: false,
            min_bytes: 1,
            round: 1,
        }));
        (PrefixCache::with_pool(budget, Arc::clone(&pool)), pool)
    }

    /// Evicted, replaced and oversized buffers are recycled, and a recycled buffer is the one
    /// the next save gets.
    #[test]
    fn released_buffers_go_back_to_the_pool() {
        let (mut c, pool) = pooled(20, 2);
        let buf = |n: usize| {
            let mut v = Vec::with_capacity(n);
            v.resize(n, 5u8);
            v
        };
        c.insert(F, seq(3, 0), buf(10));
        c.insert(F, seq(3, 100), buf(10));
        // Evicts the oldest.
        c.insert(F, seq(3, 200), buf(10));
        assert_eq!(pool.stats().recycled, 1);
        // Replaces a prefix.
        c.insert(F, seq(6, 200), buf(10));
        assert_eq!(pool.stats().recycled, 2);
        // Larger than the budget.
        c.insert(F, seq(3, 300), buf(30));
        assert_eq!(
            pool.stats().recycled,
            2,
            "the pool holds at most two spares"
        );
        assert_eq!(c.bytes(), 20);
        let v = pool.take(10);
        assert!(v.is_empty() && v.capacity() >= 10);
        assert_eq!(pool.stats().from_spare, 1);
        // A restored entry's bytes are released by the caller.
        let (_, bytes) = c.take(0).unwrap();
        let n = pool.stats().recycled;
        c.release(bytes);
        // Kept, unless the preparer was filling the second spare: then it is freed and the
        // fill makes two.
        let t0 = std::time::Instant::now();
        let settled = |s: tetraeder_model::snapshot_pool::PoolStats| s.recycled > n || s.ready == 2;
        while !settled(pool.stats()) && t0.elapsed() < std::time::Duration::from_secs(2) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(settled(pool.stats()), "{:?}", pool.stats());
    }

    /// With no spares the cache behaves as before and the pool stays idle.
    #[test]
    fn no_spares_recycles_nothing() {
        let (mut c, pool) = pooled(10, 0);
        c.insert(F, seq(3, 0), vec![0; 10]);
        assert_eq!(c.insert(F, seq(3, 100), vec![0; 10]), 1);
        c.insert(F, seq(3, 200), vec![0; 30]);
        assert_eq!(pool.stats(), Default::default());
    }

    // ---- disk tier (ADR 0087) ----

    /// A fresh directory under the system temp dir; removed when dropped.
    struct TmpDir(std::path::PathBuf);
    impl TmpDir {
        fn new(tag: &str) -> Self {
            let d = std::env::temp_dir().join(format!("tt-prefix-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&d);
            Self(d)
        }
        fn files(&self) -> usize {
            std::fs::read_dir(&self.0).map_or(0, |r| r.count())
        }
    }
    impl Drop for TmpDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn patterned(n: usize, seed: u8) -> Vec<u8> {
        (0..n)
            .map(|i| (i as u8).wrapping_mul(7).wrapping_add(seed))
            .collect()
    }

    fn with_disk(ram: usize, disk: usize, dir: &TmpDir) -> PrefixCache {
        let (c, _) = pooled(ram, 0);
        c.with_disk(&dir.0, disk)
    }

    /// Restores disk entry `i` and waits for the read.
    fn restore(c: &mut PrefixCache, i: usize) -> Restored {
        let r = c.start_restore(i).expect("a read starts");
        let t0 = std::time::Instant::now();
        loop {
            if let Some(done) = c.poll_restore(&r) {
                return done;
            }
            assert!(t0.elapsed().as_secs() < 30, "the read never finished");
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    /// A reader that waits until the test opens the gate.
    #[derive(Clone, Default)]
    struct Gate(Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>);
    impl Gate {
        fn open(&self) {
            *self.0.0.lock().unwrap() = true;
            self.0.1.notify_all();
        }
        fn reader(&self) -> ReadFn {
            let g = self.clone();
            Arc::new(move |path, format, size, pool| {
                let mut open = g.0.0.lock().unwrap();
                while !*open {
                    open = g.0.1.wait(open).unwrap();
                }
                drop(open);
                crate::prefix_disk::read_file(path, format, size, pool)
            })
        }
    }

    // hardware: none
    #[test]
    fn eviction_spills_and_restore_returns_the_same_bytes() {
        let dir = TmpDir::new("spill");
        let mut c = with_disk(20, 100, &dir);
        let a = patterned(10, 1);
        c.insert(F, seq(3, 0), a.clone());
        c.insert(F, seq(3, 100), patterned(10, 2));
        // Evicts the first one to disk.
        assert_eq!(c.insert(F, seq(3, 200), patterned(10, 3)), 1);
        c.flush_disk();
        assert_eq!((c.bytes(), c.disk_bytes(), c.len()), (20, 10, 3));
        let mut prompt = seq(3, 0);
        prompt.push(9);
        let (i, n) = c.best(F, &prompt).unwrap();
        assert_eq!((n, c.source(i)), (3, "disk"));
        assert_eq!(c.take(i), None, "a disk entry is read by start_restore");
        let (got, _) = restore(&mut c, i);
        assert_eq!(got, Some((seq(3, 0), a)));
        // A restored entry leaves the cache and its file is gone.
        assert_eq!((c.disk_bytes(), c.len(), dir.files()), (0, 2, 0));
        assert_eq!(c.best(F, &prompt), None);
    }

    // hardware: none
    #[test]
    fn an_entry_is_not_found_before_its_write_completed() {
        let dir = TmpDir::new("pending");
        let mut c = with_disk(10, 100, &dir);
        c.insert(F, seq(3, 0), vec![1; 10]);
        c.insert(F, seq(3, 100), vec![2; 10]);
        // Reserved at once, findable after the write.
        assert_eq!(c.disk_bytes(), 10);
        c.flush_disk();
        assert_eq!(
            c.best(F, &seq(3, 0)).map(|(i, _)| c.source(i)),
            Some("disk")
        );
    }

    // hardware: none
    #[test]
    fn the_disk_budget_evicts_the_oldest_file_first() {
        let dir = TmpDir::new("budget");
        let mut c = with_disk(10, 20, &dir);
        for k in 0..4 {
            c.insert(F, seq(3, k * 100), vec![k as u8; 10]);
            c.flush_disk();
        }
        // RAM holds #3, disk holds #1 and #2; #0 was spilled first and dropped.
        assert_eq!((c.bytes(), c.disk_bytes(), dir.files()), (10, 20, 2));
        assert!(!c.holds(F, &seq(3, 0)));
        assert!(c.holds(F, &seq(3, 100)) && c.holds(F, &seq(3, 200)) && c.holds(F, &seq(3, 300)));
    }

    // hardware: none
    #[test]
    fn a_longer_sequence_replaces_prefixes_on_both_tiers() {
        let dir = TmpDir::new("replace");
        let mut c = with_disk(10, 100, &dir);
        c.insert(F, seq(5, 0), vec![1; 10]);
        c.insert(F, seq(3, 100), vec![2; 10]);
        c.flush_disk();
        // seq(5, 0) is on disk; its extension replaces it.
        c.insert(F, seq(8, 0), vec![3; 10]);
        c.flush_disk();
        assert_eq!(c.disk_bytes(), 10);
        assert_eq!(dir.files(), 1);
        let (i, n) = c.best(F, &seq(8, 0)).unwrap();
        assert_eq!((n, c.source(i)), (8, "ram"));
        // A later eviction spills the extension next to the other disk entry.
        c.insert(F, seq(3, 200), vec![4; 10]);
        c.flush_disk();
        assert_eq!(c.len(), 3);
    }

    // hardware: none
    #[test]
    fn files_of_an_earlier_run_are_removed_at_startup_and_ours_on_drop() {
        let dir = TmpDir::new("cleanup");
        std::fs::create_dir_all(&dir.0).unwrap();
        // The pid of a process that is not running (Linux reads it from the name).
        let old = dir.0.join("tetraeder-prefix-4294967295-0-0.tkv");
        std::fs::write(&old, b"old").unwrap();
        // Written by a process that is gone: older than this one.
        std::fs::File::options()
            .write(true)
            .open(&old)
            .unwrap()
            .set_modified(std::time::UNIX_EPOCH)
            .unwrap();
        std::fs::write(dir.0.join("keep.txt"), b"mine").unwrap();
        let mut c = with_disk(10, 100, &dir);
        assert_eq!(dir.files(), 1);
        c.insert(F, seq(3, 0), vec![1; 10]);
        c.insert(F, seq(3, 100), vec![2; 10]);
        c.flush_disk();
        assert_eq!(dir.files(), 2);
        drop(c);
        assert_eq!(dir.files(), 1);
        assert!(dir.0.join("keep.txt").exists());
    }

    // hardware: none
    #[test]
    fn an_unusable_directory_leaves_the_ram_cache_as_before() {
        let dir = TmpDir::new("bad");
        // A file where the directory should be.
        std::fs::write(&dir.0, b"x").unwrap();
        let mut c = with_disk(10, 100, &dir);
        c.insert(F, seq(3, 0), vec![1; 10]);
        assert_eq!(c.insert(F, seq(3, 100), vec![2; 10]), 1);
        assert_eq!((c.len(), c.disk_bytes()), (1, 0));
        let _ = std::fs::remove_file(&dir.0);
    }

    // hardware: none
    #[test]
    fn a_failing_write_drops_the_entry_and_the_cache_goes_on() {
        let dir = TmpDir::new("fail");
        let mut c = with_disk(10, 100, &dir);
        // The directory disappears under the writer.
        std::fs::remove_dir_all(&dir.0).unwrap();
        c.insert(F, seq(3, 0), vec![1; 10]);
        c.insert(F, seq(3, 100), vec![2; 10]);
        c.flush_disk();
        assert_eq!((c.len(), c.disk_bytes()), (1, 0));
        assert_eq!(c.best(F, &seq(3, 0)), None);
        assert!(c.best(F, &seq(3, 100)).is_some());
    }

    // hardware: none
    #[test]
    fn a_damaged_file_is_dropped_not_returned() {
        let dir = TmpDir::new("damaged");
        let mut c = with_disk(10, 100, &dir);
        c.insert(F, seq(3, 0), vec![1; 10]);
        c.insert(F, seq(3, 100), vec![2; 10]);
        c.flush_disk();
        for f in std::fs::read_dir(&dir.0).unwrap().flatten() {
            std::fs::write(f.path(), b"junk").unwrap();
        }
        let (i, _) = c.best(F, &seq(3, 0)).unwrap();
        assert_eq!(restore(&mut c, i).0, None);
        assert_eq!((c.disk_bytes(), dir.files()), (0, 0));
    }

    /// A spilled entry on disk behind a gated reader: (cache, its pool, the gate).
    fn gated(dir: &TmpDir) -> (PrefixCache, Arc<SnapshotPool>, Gate) {
        let gate = Gate::default();
        let (c, pool) = pooled(10, 4);
        let mut c = c.with_disk_reader(gate.reader()).with_disk(&dir.0, 20);
        c.insert(F, seq(3, 0), vec![1; 10]);
        c.insert(F, seq(3, 100), vec![2; 10]);
        c.flush_disk();
        (c, pool, gate)
    }

    // hardware: none
    #[test]
    fn a_read_in_flight_keeps_its_entry_and_is_not_offered_again() {
        let dir = TmpDir::new("reading");
        let (mut c, _pool, gate) = gated(&dir);
        let (i, _) = c.best(F, &seq(3, 0)).unwrap();
        let r = c.start_restore(i).unwrap();
        assert_eq!(c.restoring(), 1);
        // Not offered to a second request, and not evicted by a spill that needs the room.
        assert_eq!(c.best(F, &seq(3, 0)), None);
        assert!(c.start_restore(i).is_none());
        c.insert(F, seq(3, 200), vec![3; 10]);
        c.insert(F, seq(3, 300), vec![4; 10]);
        c.flush_disk();
        assert!(
            c.holds(F, &seq(3, 0)) && !c.holds(F, &seq(3, 100)),
            "the oldest idle entry went instead"
        );
        assert!(c.poll_restore(&r).is_none());
        gate.open();
        let t0 = std::time::Instant::now();
        let done = loop {
            if let Some(d) = c.poll_restore(&r) {
                break d;
            }
            assert!(t0.elapsed().as_secs() < 30);
            std::thread::sleep(Duration::from_millis(2));
        };
        assert_eq!(done.0.map(|(t, _)| t), Some(seq(3, 0)));
        assert_eq!(c.restoring(), 0);
    }

    // hardware: none
    #[test]
    fn an_aborted_read_keeps_the_file_and_returns_the_buffer() {
        let dir = TmpDir::new("abort");
        let (mut c, pool, gate) = gated(&dir);
        let files = dir.files();
        let (i, _) = c.best(F, &seq(3, 0)).unwrap();
        let before = pool.stats().recycled;
        let r = c.start_restore(i).unwrap();
        c.abort_restore(r);
        gate.open();
        let t0 = std::time::Instant::now();
        // The reader gives the buffer back before it replies and the entry is offered again only
        // once the abort has been reaped, so a found entry means the buffer was handed over.
        let found = loop {
            c.flush_disk();
            if let Some(found) = c.best(F, &seq(3, 0)) {
                break found;
            }
            assert!(t0.elapsed().as_secs() < 30, "the abort was not reaped");
            std::thread::sleep(Duration::from_millis(2));
        };
        // The pool keeps the buffer unless its preparer has all four spares filled (or is
        // filling the last one) by then, which is a race with the writes above: then the buffer
        // is freed and the spares are full. Either way the buffer did not stay in the cache.
        let settled =
            |s: tetraeder_model::snapshot_pool::PoolStats| s.recycled > before || s.ready == 4;
        while !settled(pool.stats()) && t0.elapsed().as_secs() < 30 {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(settled(pool.stats()), "{:?}", pool.stats());
        assert_eq!(dir.files(), files, "the file was not consumed");
        // The entry can be restored later.
        let (i, n) = found;
        assert_eq!((n, c.source(i)), (3, "disk"));
        assert_eq!(restore(&mut c, i).0.map(|(t, _)| t), Some(seq(3, 0)));
    }

    // hardware: none
    #[test]
    fn reads_in_flight_are_bounded() {
        let dir = TmpDir::new("bounded");
        let gate = Gate::default();
        let (c, _) = pooled(10, 0);
        let mut c = c.with_disk_reader(gate.reader()).with_disk(&dir.0, 100);
        for k in 0..4 {
            c.insert(F, seq(3, k * 100), vec![k as u8; 10]);
            c.flush_disk();
        }
        // Disk holds #0..#2.
        let mut reads = Vec::new();
        for k in 0..3 {
            let (i, _) = c.best(F, &seq(3, k * 100)).unwrap();
            reads.push(c.start_restore(i));
        }
        assert_eq!(
            reads.iter().map(Option::is_some).collect::<Vec<_>>(),
            [true, true, false]
        );
        gate.open();
    }

    // hardware: none
    #[test]
    fn a_new_tier_on_the_same_directory_leaves_the_live_ones_files() {
        // A replan opens the next scheduler's tier while the old one still holds files.
        crate::prefix_disk::process_start();
        std::thread::sleep(Duration::from_millis(50));
        let dir = TmpDir::new("replan");
        let mut a = with_disk(10, 100, &dir);
        a.insert(F, seq(3, 0), vec![1; 10]);
        a.insert(F, seq(3, 100), vec![2; 10]);
        a.flush_disk();
        assert_eq!(dir.files(), 1);
        let b = with_disk(10, 100, &dir);
        assert_eq!(dir.files(), 1, "opening a tier removed a live file");
        drop(b);
        assert_eq!(dir.files(), 1, "dropping a tier removed a live file");
        let (i, _) = a.best(F, &seq(3, 0)).unwrap();
        assert_eq!(restore(&mut a, i).0.map(|(t, _)| t), Some(seq(3, 0)));
        a.insert(F, seq(3, 200), vec![3; 10]);
        a.flush_disk();
        assert_eq!(dir.files(), 1);
        drop(a);
        assert_eq!(dir.files(), 0);
    }

    // hardware: none
    #[test]
    fn an_aborted_read_still_counts_and_its_entry_is_not_evicted() {
        let dir = TmpDir::new("abort-busy");
        let gate = Gate::default();
        let (c, _) = pooled(10, 4);
        let mut c = c.with_disk_reader(gate.reader()).with_disk(&dir.0, 30);
        for k in 0..4 {
            c.insert(F, seq(3, k * 100), vec![k as u8; 10]);
            c.flush_disk();
        }
        // Disk holds #0..#2, RAM #3. Read #0 and #1, then give both up.
        for k in 0..2 {
            let (i, _) = c.best(F, &seq(3, k * 100)).unwrap();
            let r = c.start_restore(i).unwrap();
            c.abort_restore(r);
        }
        assert_eq!(c.restoring(), 2, "the readers are still running");
        let (i, _) = c.best(F, &seq(3, 200)).unwrap();
        assert!(c.start_restore(i).is_none(), "MAX_READS exceeded");
        // A spill needing room evicts the idle entry, never one a reader has open.
        c.insert(F, seq(3, 400), vec![9; 10]);
        c.flush_disk();
        assert!(c.holds(F, &seq(3, 0)) && c.holds(F, &seq(3, 100)));
        assert!(!c.holds(F, &seq(3, 200)));
        gate.open();
        let t0 = std::time::Instant::now();
        while c.restoring() > 0 {
            c.flush_disk();
            assert!(t0.elapsed().as_secs() < 30, "the readers never finished");
            std::thread::sleep(Duration::from_millis(2));
        }
        assert!(
            c.best(F, &seq(3, 0)).is_some(),
            "the entry is offered again"
        );
    }

    // hardware: none
    #[test]
    fn a_spill_that_cannot_make_room_is_dropped_not_over_budget() {
        let dir = TmpDir::new("overshoot");
        let gate = Gate::default();
        let (c, _) = pooled(10, 4);
        let mut c = c.with_disk_reader(gate.reader()).with_disk(&dir.0, 20);
        for k in 0..3 {
            c.insert(F, seq(3, k * 100), vec![k as u8; 10]);
            c.flush_disk();
        }
        // Both disk entries are being read: nothing can be evicted.
        let reads: Vec<_> = (0..2)
            .map(|k| {
                let (i, _) = c.best(F, &seq(3, k * 100)).unwrap();
                c.start_restore(i).unwrap()
            })
            .collect();
        c.insert(F, seq(3, 300), vec![7; 10]);
        c.flush_disk();
        assert_eq!(c.disk_bytes(), 20, "over the disk budget");
        assert!(!c.holds(F, &seq(3, 200)), "the spill was dropped");
        gate.open();
        drop(reads);
    }
}
