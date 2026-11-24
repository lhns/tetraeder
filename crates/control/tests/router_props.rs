//! Property tests of the router (ADR 0091, rule 5) over random sequences of requests, finishes,
//! cancels and replicas leaving and joining. The router is a pure state machine, so the sequence
//! runs without threads or clocks:
//!
//! - every request ends exactly once (finished, failed by a leaving replica, or cancelled);
//! - none is sent to a replica that is not up, and no replica gets more than its cap;
//! - a request that goes out at once goes to the replica holding the longest prefix among those
//!   with room, else to the one with the fewest outstanding requests;
//! - a leaving replica fails exactly the requests in flight on it, and keeps none.
//!
//! Hardware label: none.

use proptest::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
use tetraeder_control::router::{Action, ReplicaId, ReqId, Router};

const CAPS: [u32; 3] = [2, 1, 3];
const MIN_PREFIX: u32 = 4;

#[derive(Debug, Clone)]
enum Op {
    Submit { kind: u8, tail: Vec<u8> },
    Done(usize),
    Cancel(usize),
    Up(u8),
    Down(u8),
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        4 => (0u8..4, proptest::collection::vec(0u8..3, 0..3))
            .prop_map(|(kind, tail)| Op::Submit { kind, tail }),
        3 => (0usize..64).prop_map(Op::Done),
        1 => (0usize..64).prop_map(Op::Cancel),
        2 => (0u8..3).prop_map(Op::Up),
        2 => (0u8..3).prop_map(Op::Down),
    ]
}

fn prompt(kind: u8, tail: &[u8]) -> Vec<u32> {
    let mut p = vec![u32::from(kind) + 100; 10];
    p.extend(tail.iter().map(|&t| u32::from(t)));
    p
}

#[derive(Default)]
struct Model {
    /// Submitted, not sent yet.
    queued: BTreeSet<ReqId>,
    inflight: BTreeMap<ReqId, ReplicaId>,
    ends: BTreeMap<ReqId, usize>,
}

impl Model {
    fn end(&mut self, req: ReqId) {
        *self.ends.entry(req).or_default() += 1;
    }

    fn apply(&mut self, r: &Router, actions: &[Action]) {
        for a in actions {
            match a {
                Action::Dispatch { req, to, .. } => {
                    assert!(self.queued.remove(req), "dispatched a request not waiting");
                    assert!(r.is_up(*to), "dispatched to a replica that is not up");
                    self.inflight.insert(*req, *to);
                }
                Action::Fail { req, replica } => {
                    assert_eq!(
                        self.inflight.remove(req),
                        Some(*replica),
                        "failed a request that was not in flight on that replica"
                    );
                    self.end(*req);
                }
            }
        }
        for (i, cap) in CAPS.iter().enumerate() {
            let on = self.inflight.values().filter(|x| x.0 as usize == i).count() as u32;
            assert_eq!(
                on,
                r.outstanding(ReplicaId(i as u32)),
                "outstanding matches"
            );
            assert!(on <= *cap, "replica {i} holds {on} of {cap}");
            if !r.is_up(ReplicaId(i as u32)) {
                assert_eq!(on, 0, "a replica that is not up holds nothing");
            }
        }
        assert_eq!(self.queued.len(), r.queued());
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    // hardware: none
    #[test]
    fn requests_end_exactly_once_and_never_go_to_a_replica_that_is_down(
        ops in proptest::collection::vec(op(), 1..120)
    ) {
        let mut r = Router::new(&CAPS, MIN_PREFIX);
        let mut m = Model::default();
        let mut next: ReqId = 1;
        let mut all: Vec<ReqId> = Vec::new();
        for o in ops {
            match o {
                Op::Submit { kind, tail } => {
                    let p = prompt(kind, &tail);
                    // What the rule says, before the request goes in.
                    let cand: Vec<usize> = (0..CAPS.len())
                        .filter(|&i| r.is_up(ReplicaId(i as u32)) && r.outstanding(ReplicaId(i as u32)) < CAPS[i])
                        .collect();
                    let sc: Vec<u32> = (0..CAPS.len())
                        .map(|i| {
                            let h = r.holds(ReplicaId(i as u32), &p);
                            if h >= MIN_PREFIX { h } else { 0 }
                        })
                        .collect();
                    let ou: Vec<u32> = (0..CAPS.len()).map(|i| r.outstanding(ReplicaId(i as u32))).collect();
                    let score = |i: usize| sc[i];
                    let best = cand.iter().map(|&i| score(i)).max();
                    let fewest = |among: &[usize]| among.iter().map(|&i| ou[i]).min();
                    let expect_now = r.queued() == 0 && !cand.is_empty();
                    let id = next;
                    next += 1;
                    all.push(id);
                    m.queued.insert(id);
                    let a = r.submit(id, p);
                    if expect_now {
                        let to = match a.first() {
                            Some(Action::Dispatch { req, to, .. }) if *req == id => to.0 as usize,
                            other => panic!("a request with room waiting: {other:?}"),
                        };
                        let best = best.unwrap();
                        prop_assert_eq!(score(to), best, "the longest prefix wins among replicas with room");
                        if best == 0 {
                            let zero: Vec<usize> = cand.iter().copied().filter(|&i| score(i) == 0).collect();
                            prop_assert_eq!(
                                Some(ou[to]),
                                fewest(&zero),
                                "else the fewest outstanding"
                            );
                        }
                    }
                    m.apply(&r, &a);
                }
                Op::Done(i) => {
                    if let Some((&req, _)) = m.inflight.iter().nth(i % m.inflight.len().max(1)) {
                        m.inflight.remove(&req);
                        m.end(req);
                        let a = r.done(req);
                        m.apply(&r, &a);
                    } else if let Some(&old) = all.get(i % all.len().max(1)) {
                        // A request that already ended: ignored.
                        if !m.queued.contains(&old) && !m.inflight.contains_key(&old) {
                            prop_assert!(r.done(old).is_empty());
                        }
                    }
                }
                Op::Cancel(i) => {
                    let live: Vec<ReqId> = m.queued.iter().copied().chain(m.inflight.keys().copied()).collect();
                    if let Some(&req) = live.get(i % live.len().max(1)) {
                        m.queued.remove(&req);
                        m.inflight.remove(&req);
                        m.end(req);
                        let a = r.cancel(req);
                        m.apply(&r, &a);
                    }
                }
                Op::Up(x) => {
                    let a = r.set_up(ReplicaId(u32::from(x)), true);
                    m.apply(&r, &a);
                }
                Op::Down(x) => {
                    let a = r.set_up(ReplicaId(u32::from(x)), false);
                    m.apply(&r, &a);
                }
            }
        }
        // Drain: every replica up, finish whatever is in flight until nothing is left.
        for x in 0..CAPS.len() {
            let a = r.set_up(ReplicaId(x as u32), true);
            m.apply(&r, &a);
        }
        while let Some((&req, _)) = m.inflight.iter().next() {
            m.inflight.remove(&req);
            m.end(req);
            let a = r.done(req);
            m.apply(&r, &a);
        }
        prop_assert!(m.queued.is_empty(), "everything waiting was sent");
        for id in all {
            prop_assert_eq!(m.ends.get(&id).copied(), Some(1), "request {} ends exactly once", id);
        }
    }

    // hardware: none
    #[test]
    fn a_conversation_goes_back_to_the_replica_that_holds_it(
        turns in 2usize..6,
        others in proptest::collection::vec(0u8..4, 0..6),
    ) {
        let mut r = Router::new(&[4, 4], MIN_PREFIX);
        r.set_up(ReplicaId(0), true);
        r.set_up(ReplicaId(1), true);
        let mut id = 1;
        let base: Vec<u32> = (0..16).collect();
        let mut first = None;
        for t in 0..turns {
            // Other traffic in between, finished at once.
            for k in &others {
                id += 1;
                let a = r.submit(id, vec![200 + u32::from(*k); 8]);
                if let Some(Action::Dispatch { req, .. }) = a.first() {
                    r.done(*req);
                }
            }
            id += 1;
            let mut p = base.clone();
            p.extend((0..t as u32).map(|x| 50 + x));
            let a = r.submit(id, p);
            let to = match a.first() {
                Some(Action::Dispatch { to, .. }) => *to,
                other => panic!("room on both replicas: {other:?}"),
            };
            match first {
                None => first = Some(to),
                Some(f) => prop_assert_eq!(to, f, "turn {} left its replica", t),
            }
            r.done(id);
        }
    }
}
