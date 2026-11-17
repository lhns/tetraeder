//! Saved sequences in host RAM, the oldest evicted first (ADR 0049). A restored entry leaves the
//! cache (the sequence is live again and is saved anew when it is dropped), so the age of an
//! entry is the time since it was saved or last used. An entry is keyed by its KV format too:
//! bytes of one format are never restored into a cache of another (ADR 0050 b). Buffers the cache
//! lets go of go back to the snapshot pool instead of being freed (ADR 0074).

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

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The entry of `format` sharing the longest prefix with `prompt`: its index and that length.
    pub fn best(&self, format: &str, prompt: &[i32]) -> Option<(usize, usize)> {
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.format == format)
            .map(|(i, e)| (i, common_prefix(&e.tokens, prompt)))
            .filter(|&(_, n)| n > 0)
            .max_by_key(|&(_, n)| n)
    }

    /// Whether an entry of `format` already holds all of `tokens`.
    pub fn holds(&self, format: &str, tokens: &[i32]) -> bool {
        self.entries
            .iter()
            .any(|e| e.format == format && e.tokens.starts_with(tokens))
    }

    /// Removes entry `i` and returns its tokens and bytes.
    pub fn take(&mut self, i: usize) -> (Vec<i32>, Vec<u8>) {
        let e = self.entries.remove(i);
        self.held -= e.bytes.len();
        (e.tokens, e.bytes)
    }

    /// Adds a sequence, replacing entries that are a prefix of it and evicting the least recently
    /// used ones to fit the budget. Returns the number of entries dropped. A sequence larger than
    /// the whole budget is not stored.
    pub fn insert(&mut self, format: &str, tokens: Vec<i32>, bytes: Vec<u8>) -> usize {
        if bytes.len() > self.budget {
            self.pool.recycle(bytes);
            return 0;
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
            self.pool.recycle(e.bytes);
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
        let (t, b) = c.take(0);
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
        let (_, bytes) = c.take(0);
        let n = pool.stats().recycled;
        c.release(bytes);
        assert!(pool.stats().recycled > n || pool.stats().ready == 2);
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
}
