//! A processor-sharing disk over a measured queue-depth curve (ADR 0056).
//!
//! Up to `workers` chunks are in service. With `n` in service the disk moves `rate(n)` bytes per
//! second in total, shared equally, so each chunk progresses at `rate(n) / n`. `rate` is the
//! curve of `tetraeder-bench disk-curve` (`profiler::curve_rate`). Deterministic: chunks that
//! finish together leave in submit order. The caller owns the queue and the clock: it submits
//! while [`Disk::free`] is positive and moves time with [`Disk::advance_to`].

use tetraeder_profiler::{QueuePoint, curve_rate};

struct Active<T> {
    remaining: f64,
    tag: T,
}

pub struct Disk<T> {
    curve: Vec<QueuePoint>,
    workers: usize,
    now: f64,
    /// In submit order.
    active: Vec<Active<T>>,
}

impl<T> Disk<T> {
    /// `curve` must have a point; a single point at depth 1 is a flat disk.
    pub fn new(curve: Vec<QueuePoint>, workers: usize) -> Self {
        assert!(!curve.is_empty(), "a disk needs a rate curve");
        Self {
            curve,
            workers: workers.max(1),
            now: 0.0,
            active: Vec::new(),
        }
    }

    /// Time of the last advance, seconds.
    pub fn now(&self) -> f64 {
        self.now
    }

    /// Chunks in service.
    pub fn busy(&self) -> usize {
        self.active.len()
    }

    /// Chunks that can start now.
    pub fn free(&self) -> usize {
        self.workers - self.active.len()
    }

    /// Starts a chunk of `bytes` at [`Disk::now`]. Gives the tag back when all workers are busy.
    pub fn submit(&mut self, bytes: u64, tag: T) -> Result<(), T> {
        if self.free() == 0 {
            return Err(tag);
        }
        self.active.push(Active {
            remaining: bytes as f64,
            tag,
        });
        Ok(())
    }

    /// Bytes per second each chunk in service gets.
    fn share(&self) -> f64 {
        let n = self.active.len().max(1);
        let depth = u32::try_from(n).unwrap_or(u32::MAX);
        curve_rate(&self.curve, depth).unwrap_or(0.0) / n as f64
    }

    /// When the first chunk in service finishes, if nothing joins before.
    pub fn next_completion(&self) -> Option<f64> {
        let min = self
            .active
            .iter()
            .map(|a| a.remaining)
            .fold(f64::INFINITY, f64::min);
        if !min.is_finite() {
            return None;
        }
        let share = self.share();
        if share <= 0.0 {
            return None;
        }
        Some(self.now + min / share)
    }

    /// Moves time to `t` (never back) and returns the chunks that finished on the way with their
    /// finish times, in order. Workers freed on the way stay empty: to refill them at the finish
    /// time, step with [`Disk::next_completion`].
    pub fn advance_to(&mut self, t: f64) -> Vec<(f64, T)> {
        let mut out = Vec::new();
        let t = t.max(self.now);
        loop {
            let Some(tc) = self.next_completion() else {
                self.now = t;
                break;
            };
            let share = self.share();
            if tc > t {
                let dt = t - self.now;
                for a in &mut self.active {
                    a.remaining -= share * dt;
                }
                self.now = t;
                break;
            }
            let dt = tc - self.now;
            let first = self
                .active
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.remaining.total_cmp(&b.1.remaining))
                .map(|(i, _)| i)
                .expect("a chunk is in service");
            for a in &mut self.active {
                a.remaining -= share * dt;
            }
            self.active[first].remaining = 0.0;
            self.now = tc;
            let mut i = 0;
            while i < self.active.len() {
                if self.active[i].remaining <= 1e-6 {
                    out.push((tc, self.active.remove(i).tag));
                } else {
                    i += 1;
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn flat(rate: f64) -> Vec<QueuePoint> {
        vec![QueuePoint {
            depth: 1,
            bytes_per_s: rate,
        }]
    }

    fn curve(points: &[(u32, f64)]) -> Vec<QueuePoint> {
        points
            .iter()
            .map(|&(depth, bytes_per_s)| QueuePoint { depth, bytes_per_s })
            .collect()
    }

    #[test]
    fn one_chunk_at_depth_one_takes_chunk_over_rate() {
        let mut d = Disk::new(curve(&[(1, 1e9), (4, 3e9)]), 4);
        d.submit(1_000_000, 'a').unwrap();
        let t = d.next_completion().unwrap();
        assert!((t - 1e-3).abs() < 1e-12, "{t}");
        assert!(d.advance_to(t - 1e-6).is_empty());
        let done = d.advance_to(t);
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].1, 'a');
        assert_eq!(d.busy(), 0);
        assert!(d.next_completion().is_none());
    }

    #[test]
    fn two_equal_chunks_at_depth_two_finish_together_in_submit_order() {
        let mut d = Disk::new(curve(&[(1, 1e9), (2, 1.8e9)]), 4);
        d.submit(1_000_000, 1).unwrap();
        d.submit(1_000_000, 2).unwrap();
        // Each gets rate(2) / 2; both finish at 2 * chunk / rate(2).
        let want = 2.0 * 1e6 / 1.8e9;
        let t = d.next_completion().unwrap();
        assert!((t - want).abs() < 1e-12, "{t} {want}");
        let done = d.advance_to(t);
        assert_eq!(done.iter().map(|x| x.1).collect::<Vec<_>>(), [1, 2]);
        assert!(done.iter().all(|x| (x.0 - want).abs() < 1e-12));
    }

    #[test]
    fn a_chunk_that_joins_slows_the_one_in_service() {
        // Flat total rate: one chunk alone gets all of it, two get half each.
        let mut d = Disk::new(flat(1e9), 2);
        d.submit(1_000_000, 'a').unwrap();
        // a is half done at 0.5 ms; b joins and both get 0.5 GB/s: a needs 1 ms more.
        assert!(d.advance_to(0.5e-3).is_empty());
        d.submit(1_000_000, 'b').unwrap();
        let t = d.next_completion().unwrap();
        assert!((t - 1.5e-3).abs() < 1e-9, "{t}");
        let done = d.advance_to(t);
        assert_eq!(done.iter().map(|x| x.1).collect::<Vec<_>>(), ['a']);
        // b has 0.5 MB left and the disk to itself: 0.5 ms.
        let t = d.next_completion().unwrap();
        assert!((t - 2.0e-3).abs() < 1e-9, "{t}");
    }

    #[test]
    fn workers_bound_the_chunks_in_service() {
        let mut d = Disk::new(flat(1e9), 2);
        d.submit(10, 1).unwrap();
        d.submit(10, 2).unwrap();
        assert_eq!(d.free(), 0);
        assert_eq!(d.submit(10, 3), Err(3));
    }

    proptest! {
        /// On a curve that scales at most linearly with depth (so a chunk never goes faster than
        /// alone at depth 1), no chunk finishes before `bytes / rate(1)` after its start, and
        /// the bytes served equal the bytes submitted.
        #[test]
        fn no_early_completion_and_all_bytes_served(
            r1 in 1e8f64..5e9,
            f2 in 1.0f64..2.0,
            f4 in 1.0f64..4.0,
            sizes in prop::collection::vec(1u64..4_000_000, 1..14),
            workers in 1usize..6,
        ) {
            let c = curve(&[(1, r1), (2, r1 * f2), (4, r1 * f2.max(f4))]);
            let mut d: Disk<(u64, f64)> = Disk::new(c, workers);
            let submitted: u64 = sizes.iter().sum();
            let mut waiting: std::collections::VecDeque<u64> = sizes.into();
            let mut served = 0u64;
            loop {
                while d.free() > 0 {
                    let Some(b) = waiting.pop_front() else { break };
                    d.submit(b, (b, d.now())).unwrap();
                }
                let Some(tc) = d.next_completion() else { break };
                for (at, (bytes, start)) in d.advance_to(tc) {
                    served += bytes;
                    prop_assert!(at - start + 1e-12 >= bytes as f64 / r1, "{at} {start} {bytes}");
                }
            }
            prop_assert_eq!(served, submitted);
        }
    }
}
