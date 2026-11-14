//! Fixed blocks allocated on demand, lowest free block first (ADR 0067).

use crate::{KvError, Layout, RowCopy, SeqId};
use std::collections::{BTreeSet, HashMap};

pub struct Paged {
    block: usize,
    n_blocks: usize,
    free: BTreeSet<usize>,
    tables: HashMap<SeqId, Vec<usize>>,
}

impl Paged {
    /// `capacity` is cut down to whole blocks.
    pub fn new(capacity: usize, block: usize) -> Self {
        assert!(block > 0, "block size must be positive");
        let n_blocks = capacity / block;
        Self {
            block,
            n_blocks,
            free: (0..n_blocks).collect(),
            tables: HashMap::new(),
        }
    }
}

impl Layout for Paged {
    fn name(&self) -> &'static str {
        "paged"
    }

    fn capacity(&self) -> usize {
        self.n_blocks * self.block
    }

    fn free(&self) -> usize {
        self.free.len() * self.block
    }

    fn reserved(&self, seq: SeqId) -> usize {
        self.tables.get(&seq).map_or(0, |t| t.len() * self.block)
    }

    fn reserve(&mut self, seq: SeqId, n: usize) -> Result<Vec<RowCopy>, KvError> {
        let need = n.div_ceil(self.block);
        let have = self.tables.get(&seq).map_or(0, Vec::len);
        if need <= have {
            return Ok(Vec::new());
        }
        let extra = need - have;
        if extra > self.free.len() {
            return Err(KvError::Full {
                needed: extra * self.block,
                free: self.free(),
            });
        }
        let table = self.tables.entry(seq).or_default();
        for _ in 0..extra {
            table.push(self.free.pop_first().expect("checked above"));
        }
        Ok(Vec::new())
    }

    fn shrink(&mut self, seq: SeqId, keep: usize) {
        let Some(table) = self.tables.get_mut(&seq) else {
            return;
        };
        let blocks = keep.div_ceil(self.block);
        while table.len() > blocks {
            self.free.insert(table.pop().expect("non-empty"));
        }
        if table.is_empty() {
            self.tables.remove(&seq);
        }
    }

    fn row(&self, seq: SeqId, pos: usize) -> usize {
        let t = self
            .tables
            .get(&seq)
            .expect("row of a sequence without rows");
        t[pos / self.block] * self.block + pos % self.block
    }

    fn max_row(&self, seq: SeqId, n: usize) -> Option<usize> {
        if n == 0 {
            return None;
        }
        let t = self
            .tables
            .get(&seq)
            .expect("row of a sequence without rows");
        let blocks = n.div_ceil(self.block);
        t[..blocks]
            .iter()
            .enumerate()
            .map(|(i, b)| b * self.block + (n - i * self.block).min(self.block) - 1)
            .max()
    }
}
