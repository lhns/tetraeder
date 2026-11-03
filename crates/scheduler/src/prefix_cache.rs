//! Saved sequences in host RAM, the oldest evicted first (ADR 0049). A restored entry leaves the
//! cache (the sequence is live again and is saved anew when it is dropped), so the age of an
//! entry is the time since it was saved or last used.

/// A live sequence is saved before it loses at least this many tokens.
pub const SAVE_MIN_TOKENS: usize = 512;
/// A saved entry is restored when it beats the live sequence by at least this many tokens.
pub const RESTORE_MIN_GAIN: usize = 64;

struct Entry {
    tokens: Vec<i32>,
    bytes: Vec<u8>,
}

/// Entries hold the tokens of a sequence and its KV cache as backend bytes. The budget counts
/// the bytes only.
pub struct PrefixCache {
    entries: Vec<Entry>,
    budget: usize,
    held: usize,
}

/// Length of the common prefix of `a` and `b`.
pub fn common_prefix(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

impl PrefixCache {
    /// A budget of 0 disables the cache.
    pub fn new(budget: usize) -> Self {
        Self {
            entries: Vec::new(),
            budget,
            held: 0,
        }
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

    /// The entry sharing the longest prefix with `prompt`: its index and that length.
    pub fn best(&self, prompt: &[i32]) -> Option<(usize, usize)> {
        self.entries
            .iter()
            .enumerate()
            .map(|(i, e)| (i, common_prefix(&e.tokens, prompt)))
            .filter(|&(_, n)| n > 0)
            .max_by_key(|&(_, n)| n)
    }

    /// Whether an entry already holds all of `tokens`.
    pub fn holds(&self, tokens: &[i32]) -> bool {
        self.entries.iter().any(|e| e.tokens.starts_with(tokens))
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
    pub fn insert(&mut self, tokens: Vec<i32>, bytes: Vec<u8>) -> usize {
        if bytes.len() > self.budget {
            return 0;
        }
        let mut dropped = 0;
        let held = &mut self.held;
        self.entries.retain(|e| {
            let prefix = tokens.starts_with(&e.tokens);
            if prefix {
                *held -= e.bytes.len();
            }
            !prefix
        });
        // Replacing a prefix is not an eviction.
        while self.held + bytes.len() > self.budget {
            if self.entries.is_empty() {
                break;
            }
            self.held -= self.entries.remove(0).bytes.len();
            dropped += 1;
        }
        self.held += bytes.len();
        self.entries.push(Entry { tokens, bytes });
        dropped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq(n: usize, base: i32) -> Vec<i32> {
        (0..n as i32).map(|i| base + i).collect()
    }

    #[test]
    fn best_finds_the_longest_prefix() {
        let mut c = PrefixCache::new(1000);
        c.insert(seq(10, 0), vec![0; 10]);
        let mut other = seq(10, 0);
        other[5] = 99;
        c.insert(other, vec![0; 10]);
        let mut p = seq(10, 0);
        p.push(7);
        assert_eq!(c.best(&p), Some((0, 10)));
        assert_eq!(c.best(&[42]), None);
    }

    #[test]
    fn prefix_entries_are_replaced_not_evicted() {
        let mut c = PrefixCache::new(1000);
        c.insert(seq(5, 0), vec![0; 5]);
        assert_eq!(c.insert(seq(8, 0), vec![0; 8]), 0);
        assert_eq!((c.len(), c.bytes()), (1, 8));
        assert!(c.holds(&seq(6, 0)));
    }

    #[test]
    fn the_oldest_entry_is_evicted_first() {
        let mut c = PrefixCache::new(20);
        c.insert(seq(3, 0), vec![0; 10]);
        c.insert(seq(3, 100), vec![0; 10]);
        assert_eq!(c.insert(seq(3, 200), vec![0; 10]), 1);
        assert!(!c.holds(&seq(3, 0)) && c.holds(&seq(3, 100)) && c.holds(&seq(3, 200)));
        assert_eq!(c.bytes(), 20);
        let (t, b) = c.take(0);
        assert_eq!((t, b.len(), c.bytes()), (seq(3, 100), 10, 10));
    }

    #[test]
    fn oversized_and_disabled() {
        let mut c = PrefixCache::new(5);
        assert_eq!(c.insert(seq(3, 0), vec![0; 10]), 0);
        assert!(c.is_empty());
        assert!(!PrefixCache::new(0).enabled());
    }
}
