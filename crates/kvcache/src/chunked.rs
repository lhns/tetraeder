//! One contiguous run per sequence, first fit from the lowest row (ADR 0067).

use crate::{KvError, Layout, ROW_ALIGN, RowCopy, SeqId, round_up};

/// Rows added when a sequence has to move (ADR 0065).
pub const GROW: usize = 1024;

#[derive(Debug, Clone, Copy)]
struct Run {
    seq: SeqId,
    start: usize,
    size: usize,
}

pub struct Chunked {
    capacity: usize,
    /// Sorted by `start`.
    runs: Vec<Run>,
}

impl Chunked {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            runs: Vec::new(),
        }
    }

    fn index(&self, seq: SeqId) -> Option<usize> {
        self.runs.iter().position(|r| r.seq == seq)
    }

    /// Lowest start of a free gap of `size` rows.
    fn first_fit(&self, size: usize) -> Option<usize> {
        let mut cursor = 0;
        for r in &self.runs {
            if r.start - cursor >= size {
                return Some(cursor);
            }
            cursor = r.start + r.size;
        }
        (self.capacity.saturating_sub(cursor) >= size).then_some(cursor)
    }

    fn insert(&mut self, run: Run) {
        let at = self.runs.partition_point(|r| r.start < run.start);
        self.runs.insert(at, run);
    }
}

impl Layout for Chunked {
    fn name(&self) -> &'static str {
        "chunked"
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn free(&self) -> usize {
        self.capacity - self.runs.iter().map(|r| r.size).sum::<usize>()
    }

    fn reserved(&self, seq: SeqId) -> usize {
        self.index(seq).map_or(0, |i| self.runs[i].size)
    }

    fn reserve(&mut self, seq: SeqId, n: usize) -> Result<Vec<RowCopy>, KvError> {
        let need = round_up(n, ROW_ALIGN);
        let Some(i) = self.index(seq) else {
            if need == 0 {
                return Ok(Vec::new());
            }
            let start = self.first_fit(need).ok_or(KvError::Full {
                needed: need,
                free: self.free(),
            })?;
            self.insert(Run {
                seq,
                start,
                size: need,
            });
            return Ok(Vec::new());
        };
        let run = self.runs[i];
        if run.size >= need {
            return Ok(Vec::new());
        }
        // Grow in place when the rows after the run are free.
        let next = self.runs.get(i + 1).map_or(self.capacity, |r| r.start);
        if run.start + need <= next {
            self.runs[i].size = need;
            return Ok(Vec::new());
        }
        // Else move to a free run of the needed size plus GROW, or of the needed size alone when
        // no gap holds the extra. The old run stays occupied while the destination is chosen, so
        // source and destination never overlap.
        let (start, size) = match self.first_fit(need + GROW) {
            Some(start) => (start, need + GROW),
            None => (
                self.first_fit(need).ok_or(KvError::Full {
                    needed: need,
                    free: self.free(),
                })?,
                need,
            ),
        };
        self.runs.remove(i);
        self.insert(Run { seq, start, size });
        Ok(vec![RowCopy {
            src: run.start,
            dst: start,
            n: run.size,
        }])
    }

    fn shrink(&mut self, seq: SeqId, keep: usize) {
        let Some(i) = self.index(seq) else { return };
        if keep == 0 {
            self.runs.remove(i);
        } else {
            self.runs[i].size = self.runs[i].size.min(round_up(keep, ROW_ALIGN));
        }
    }

    fn row(&self, seq: SeqId, pos: usize) -> usize {
        let i = self.index(seq).expect("row of a sequence without rows");
        debug_assert!(pos < self.runs[i].size, "position outside the reservation");
        self.runs[i].start + pos
    }

    fn run_start(&self, seq: SeqId) -> Option<usize> {
        self.index(seq).map(|i| self.runs[i].start)
    }

    fn max_row(&self, seq: SeqId, n: usize) -> Option<usize> {
        (n > 0).then(|| self.row(seq, n - 1))
    }
}
