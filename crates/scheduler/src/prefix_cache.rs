//! Saved sequences in host RAM, the oldest evicted first (ADR 0049). A restored entry leaves the
//! cache (the sequence is live again and is saved anew when it is dropped), so the age of an
//! entry is the time since it was saved or last used. An entry is keyed by its KV format too:
//! bytes of one format are never restored into a cache of another (ADR 0050 b). Buffers the cache
//! lets go of go back to the snapshot pool instead of being freed (ADR 0074). With a disk tier
//! (ADR 0087) an entry the RAM budget evicts spills to a file; indices run over the RAM entries,
//! then the completed disk entries.

use crate::prefix_disk::DiskTier;
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
}

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
        }
    }

    /// Adds the disk tier: evicted entries spill to files in `dir`, up to `budget` bytes. A
    /// directory that cannot be used is a warning and no disk tier, never an error.
    #[must_use]
    pub fn with_disk(mut self, dir: &Path, budget: usize) -> Self {
        if budget == 0 {
            return self;
        }
        match DiskTier::open(dir, budget, Arc::clone(&self.pool)) {
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
        if let Some(d) = &mut self.disk {
            d.poll();
        }
        let ram = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.format == format)
            .map(|(i, e)| (i, common_prefix(&e.tokens, prompt)))
            .filter(|&(_, n)| n > 0)
            .max_by_key(|&(_, n)| n);
        let disk = self
            .disk
            .as_ref()
            .and_then(|d| d.best(format, prompt))
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

    /// Removes entry `i` and returns its tokens and bytes. A disk entry is read back first;
    /// `None` when that fails (the entry is gone either way).
    pub fn take(&mut self, i: usize) -> Option<(Vec<i32>, Vec<u8>)> {
        if i < self.entries.len() {
            let e = self.entries.remove(i);
            self.held -= e.bytes.len();
            return Some((e.tokens, e.bytes));
        }
        let n = self.entries.len();
        self.disk.as_mut()?.take(i - n)
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
        let (t, b) = c.take(i).unwrap();
        assert_eq!((t, b), (seq(3, 0), a));
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
        std::fs::write(dir.0.join("tetraeder-prefix-1-0.tkv"), b"old").unwrap();
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
        assert_eq!(c.take(i), None);
        assert_eq!((c.disk_bytes(), dir.files()), (0, 0));
    }
}
