//! Step planning for continuous batching (ADR 0065): pure functions, no backend.

use crate::backend::SeqId;

/// A live sequence as the planner sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Live {
    pub seq: SeqId,
    /// Tokens it wants in the next step: 1 (or 1 + drafts) when decoding, the rest of the prompt
    /// when prefilling.
    pub pending: usize,
    pub decoding: bool,
    /// Its channel is full; it stays out of steps and keeps its rows.
    pub paused: bool,
    /// Admission order, smaller is older.
    pub admitted: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Take {
    pub seq: SeqId,
    pub n: usize,
}

/// Decoding, unpaused sequences take their pending tokens (1, or 1 + drafts) even beyond the
/// budget; prefilling ones take chunks in admission order from what is left. Decoders come
/// first in the result, in input order.
pub(crate) fn plan_step(budget: usize, live: &[Live]) -> Vec<Take> {
    let mut out: Vec<Take> = live
        .iter()
        .filter(|l| l.decoding && !l.paused && l.pending > 0)
        .map(|l| Take {
            seq: l.seq,
            n: l.pending,
        })
        .collect();
    let mut left = budget.saturating_sub(out.iter().map(|t| t.n).sum());
    let mut prefill: Vec<&Live> = live
        .iter()
        .filter(|l| !l.decoding && !l.paused && l.pending > 0)
        .collect();
    prefill.sort_by_key(|l| l.admitted);
    for l in prefill {
        if left == 0 {
            break;
        }
        let n = l.pending.min(left);
        out.push(Take { seq: l.seq, n });
        left -= n;
    }
    out
}

/// The sequence to preempt on KvFull, after every idle sequence has been evicted (oldest first,
/// saved to RAM per ADR 0049): the most recently admitted other live one, paused or not.
pub(crate) fn victim(live: &[Live], needing: SeqId) -> Option<SeqId> {
    live.iter()
        .filter(|l| l.seq != needing)
        .max_by_key(|l| l.admitted)
        .map(|l| l.seq)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn live(seq: SeqId, pending: usize, decoding: bool, admitted: u64) -> Live {
        Live {
            seq,
            pending,
            decoding,
            paused: false,
            admitted,
        }
    }

    #[test]
    fn budget_split_decoders_first_then_prefill_in_admission_order() {
        let l = [
            live(0, 1000, false, 5), // newer prefill
            live(1, 1, true, 1),
            live(2, 300, false, 2), // older prefill
            live(3, 1, true, 3),
        ];
        let t = plan_step(512, &l);
        assert_eq!(
            t,
            vec![
                Take { seq: 1, n: 1 },
                Take { seq: 3, n: 1 },
                Take { seq: 2, n: 300 },
                Take { seq: 0, n: 210 },
            ]
        );
        assert_eq!(t.iter().map(|t| t.n).sum::<usize>(), 512);
    }

    #[test]
    fn decoders_beyond_the_budget_still_run_and_starve_prefill() {
        let l = [
            live(0, 1, true, 0),
            live(1, 3, true, 1), // 1 + 2 drafts
            live(2, 1, true, 2),
            live(3, 50, false, 3),
        ];
        let t = plan_step(4, &l);
        assert_eq!(
            t,
            vec![
                Take { seq: 0, n: 1 },
                Take { seq: 1, n: 3 },
                Take { seq: 2, n: 1 }
            ]
        );
    }

    #[test]
    fn paused_sequences_are_skipped() {
        let mut a = live(0, 1, true, 0);
        a.paused = true;
        let mut b = live(1, 100, false, 1);
        b.paused = true;
        let c = live(2, 100, false, 2);
        assert_eq!(plan_step(64, &[a, b, c]), vec![Take { seq: 2, n: 64 }]);
        assert!(plan_step(64, &[a, b]).is_empty());
    }

    #[test]
    fn empty_and_zero_pending() {
        assert!(plan_step(512, &[]).is_empty());
        assert!(plan_step(512, &[live(0, 0, false, 0)]).is_empty());
        assert!(plan_step(0, &[live(0, 5, false, 0)]).is_empty());
    }

    #[test]
    fn victim_is_the_most_recently_admitted_other_sequence() {
        let l = [
            live(0, 1, true, 1),
            live(1, 1, true, 9),
            live(2, 1, true, 4),
        ];
        assert_eq!(victim(&l, 0), Some(1));
        assert_eq!(victim(&l, 1), Some(2));
        let mut p = l;
        p[1].paused = true;
        assert_eq!(victim(&p, 0), Some(1), "paused sequences can be preempted");
    }

    #[test]
    fn victim_none_when_alone() {
        assert_eq!(victim(&[live(3, 1, true, 0)], 3), None);
        assert_eq!(victim(&[], 0), None);
    }
}
