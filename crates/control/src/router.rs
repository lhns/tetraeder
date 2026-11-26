//! The router of the head (ADR 0091, "Replicas and failover"): a pure state machine, the
//! caller supplies events and carries out the actions it returns, as `membership.rs` does.
//!
//! It owns the wait queue and the table of in-flight requests, so the properties that matter
//! (a request ends exactly once, none goes to a replica that is not up, a conversation keeps its
//! replica) are properties of this one type and are tested without threads or clocks.
//!
//! A request goes to an `Up` replica with room (fewer outstanding than its cap): first the one
//! holding the longest prefix of the prompt (live or cached), else the one with the fewest
//! outstanding requests. Without room anywhere it waits here and can go to any replica later.

use std::collections::{HashMap, VecDeque};

pub type TokenId = u32;
/// Identifies a request in the router (the caller's counter).
pub type ReqId = u64;

/// Prefix entries kept per replica; the oldest cached one goes first.
const MAX_ENTRIES: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ReplicaId(pub u32);

/// Why a request went to a replica.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    /// The replica holds a prefix of the prompt.
    Affinity,
    /// Fewest outstanding requests.
    Least,
}

/// A prefix another replica holds than the one chosen: the caller prices a fetch against
/// recomputing it (`kv::fetch_beats_recompute`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixHit {
    pub holder: ReplicaId,
    pub tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Send `req` to replica `to`.
    Dispatch {
        req: ReqId,
        to: ReplicaId,
        route: Route,
        /// The request waited here while a replica left, and goes to another one now.
        moved: bool,
        /// A longer prefix held by a replica other than `to`.
        elsewhere: Option<PrefixHit>,
    },
    /// `req` was in flight on `replica`, which left: end it with one error event.
    Fail { req: ReqId, replica: ReplicaId },
}

struct Entry {
    tokens: Vec<TokenId>,
    /// The request that is computing it; the entry is cached once that ends.
    live: Option<ReqId>,
}

struct Replica {
    cap: u32,
    up: bool,
    /// Takes no new request; its in-flight ones go on (a planned drain).
    draining: bool,
    outstanding: u32,
    entries: Vec<Entry>,
}

struct Waiting {
    req: ReqId,
    prompt: Vec<TokenId>,
    /// `Router::leaves` when it was queued.
    leaves: u64,
}

pub struct Router {
    replicas: Vec<Replica>,
    queue: VecDeque<Waiting>,
    inflight: HashMap<ReqId, (ReplicaId, Vec<TokenId>)>,
    leaves: u64,
    /// Shortest shared prefix that counts as a hit.
    min_prefix: u32,
}

fn lcp(a: &[TokenId], b: &[TokenId]) -> u32 {
    a.iter().zip(b).take_while(|(x, y)| x == y).count() as u32
}

impl Router {
    /// `caps[r]` is replica r's sequence limit: no more requests are sent to it at once. All
    /// replicas start down; `set_up` brings them in.
    pub fn new(caps: &[u32], min_prefix: u32) -> Self {
        Self {
            replicas: caps
                .iter()
                .map(|&cap| Replica {
                    cap,
                    up: false,
                    draining: false,
                    outstanding: 0,
                    entries: Vec::new(),
                })
                .collect(),
            queue: VecDeque::new(),
            inflight: HashMap::new(),
            leaves: 0,
            min_prefix,
        }
    }

    pub fn replicas(&self) -> usize {
        self.replicas.len()
    }

    pub fn is_up(&self, r: ReplicaId) -> bool {
        self.replicas.get(r.0 as usize).is_some_and(|x| x.up)
    }

    pub fn is_draining(&self, r: ReplicaId) -> bool {
        self.replicas.get(r.0 as usize).is_some_and(|x| x.draining)
    }

    pub fn up_count(&self) -> usize {
        self.replicas.iter().filter(|r| r.up).count()
    }

    pub fn outstanding(&self, r: ReplicaId) -> u32 {
        self.replicas.get(r.0 as usize).map_or(0, |x| x.outstanding)
    }

    pub fn queued(&self) -> usize {
        self.queue.len()
    }

    pub fn in_flight(&self, req: ReqId) -> Option<ReplicaId> {
        self.inflight.get(&req).map(|(r, _)| *r)
    }

    /// The requests in flight on `r`, in id order.
    pub fn in_flight_on(&self, r: ReplicaId) -> Vec<ReqId> {
        let mut v: Vec<ReqId> = self
            .inflight
            .iter()
            .filter(|(_, (x, _))| *x == r)
            .map(|(id, _)| *id)
            .collect();
        v.sort_unstable();
        v
    }

    /// Longest prefix of `prompt` that replica `r` holds, live or cached.
    pub fn holds(&self, r: ReplicaId, prompt: &[TokenId]) -> u32 {
        self.replicas.get(r.0 as usize).map_or(0, |x| {
            x.entries
                .iter()
                .map(|e| lcp(&e.tokens, prompt))
                .max()
                .unwrap_or(0)
        })
    }

    /// A replica saved a prefix to its cache (the replicas' save events feed this).
    pub fn prefix_saved(&mut self, r: ReplicaId, tokens: &[TokenId]) {
        if let Some(x) = self.replicas.get_mut(r.0 as usize) {
            Self::remember(x, tokens.to_vec(), None);
        }
    }

    /// A replica dropped a prefix from its cache.
    pub fn prefix_evicted(&mut self, r: ReplicaId, tokens: &[TokenId]) {
        if let Some(x) = self.replicas.get_mut(r.0 as usize) {
            x.entries.retain(|e| e.tokens != tokens || e.live.is_some());
        }
    }

    /// Replica `r` was asked for the prefix `tokens` and held no idle sequence or cache entry
    /// with it: every entry there that starts with it is gone, except those a request still
    /// computes.
    pub fn prefix_gone(&mut self, r: ReplicaId, tokens: &[TokenId]) {
        if let Some(x) = self.replicas.get_mut(r.0 as usize) {
            x.entries
                .retain(|e| e.live.is_some() || !e.tokens.starts_with(tokens));
        }
    }

    fn remember(x: &mut Replica, tokens: Vec<TokenId>, live: Option<ReqId>) {
        if let Some(e) = x.entries.iter_mut().find(|e| e.tokens == tokens) {
            e.live = e.live.or(live);
            return;
        }
        x.entries.push(Entry { tokens, live });
        while x.entries.len() > MAX_ENTRIES {
            // The oldest entry nobody computes now.
            match x.entries.iter().position(|e| e.live.is_none()) {
                Some(i) => {
                    x.entries.remove(i);
                }
                None => break,
            }
        }
    }

    /// The replica `prompt` would go to now: an up replica with room.
    fn pick(&self, prompt: &[TokenId]) -> Option<(ReplicaId, Route, Option<PrefixHit>)> {
        let mut best: Option<(u32, u32, usize)> = None; // (prefix, outstanding, index)
        for (i, x) in self.replicas.iter().enumerate() {
            if !x.up || x.draining || x.outstanding >= x.cap {
                continue;
            }
            let p = self.holds(ReplicaId(i as u32), prompt);
            let p = if p >= self.min_prefix { p } else { 0 };
            let better = match best {
                None => true,
                Some((bp, bo, _)) => p > bp || (p == bp && x.outstanding < bo),
            };
            if better {
                best = Some((p, x.outstanding, i));
            }
        }
        let (p, _, i) = best?;
        let to = ReplicaId(i as u32);
        // A longer prefix on an up replica that is full: a fetch candidate.
        let elsewhere = self
            .replicas
            .iter()
            .enumerate()
            .filter(|(j, x)| *j != i && x.up && !x.draining)
            .map(|(j, _)| (j, self.holds(ReplicaId(j as u32), prompt)))
            .filter(|&(_, h)| h >= self.min_prefix && h > p)
            .max_by_key(|&(j, h)| (h, std::cmp::Reverse(j)))
            .map(|(j, h)| PrefixHit {
                holder: ReplicaId(j as u32),
                tokens: h,
            });
        Some((
            to,
            if p > 0 { Route::Affinity } else { Route::Least },
            elsewhere,
        ))
    }

    fn pump(&mut self) -> Vec<Action> {
        let mut out = Vec::new();
        while let Some(front) = self.queue.front() {
            let Some((to, route, elsewhere)) = self.pick(&front.prompt) else {
                break;
            };
            let Some(w) = self.queue.pop_front() else {
                break;
            };
            let moved = w.leaves < self.leaves;
            let x = &mut self.replicas[to.0 as usize];
            x.outstanding += 1;
            Self::remember(x, w.prompt.clone(), Some(w.req));
            self.inflight.insert(w.req, (to, w.prompt));
            out.push(Action::Dispatch {
                req: w.req,
                to,
                route,
                moved,
                elsewhere,
            });
        }
        out
    }

    /// A new request: it goes out at once or waits.
    pub fn submit(&mut self, req: ReqId, prompt: Vec<TokenId>) -> Vec<Action> {
        self.queue.push_back(Waiting {
            req,
            prompt,
            leaves: self.leaves,
        });
        self.pump()
    }

    /// The engine ended `req` (finished or errored): its slot is free. Unknown requests (failed
    /// or cancelled earlier) are ignored, so every request ends once here.
    pub fn done(&mut self, req: ReqId) -> Vec<Action> {
        let Some((r, _)) = self.inflight.remove(&req) else {
            return Vec::new();
        };
        let x = &mut self.replicas[r.0 as usize];
        x.outstanding = x.outstanding.saturating_sub(1);
        for e in &mut x.entries {
            if e.live == Some(req) {
                e.live = None;
            }
        }
        self.pump()
    }

    /// The client gave up: drops a waiting request, or frees the slot of a running one.
    pub fn cancel(&mut self, req: ReqId) -> Vec<Action> {
        self.queue.retain(|w| w.req != req);
        self.done(req)
    }

    /// A dispatch that could not be delivered (the link was down): the request goes back to the
    /// head of the queue, as if it had not left. `take_down` also takes the replica out of the
    /// router, or the same replica would be chosen again.
    pub fn requeue(&mut self, req: ReqId, take_down: bool) -> Vec<Action> {
        let Some((r, prompt)) = self.inflight.remove(&req) else {
            return Vec::new();
        };
        let x = &mut self.replicas[r.0 as usize];
        x.outstanding = x.outstanding.saturating_sub(1);
        x.entries.retain(|e| e.live != Some(req));
        self.queue.push_front(Waiting {
            req,
            prompt,
            leaves: self.leaves,
        });
        if take_down {
            self.set_up(r, false)
        } else {
            self.pump()
        }
    }

    /// Sets replica `r`'s limit (what its engine reports, at most what the plan gives it).
    pub fn set_cap(&mut self, r: ReplicaId, cap: u32) -> Vec<Action> {
        match self.replicas.get_mut(r.0 as usize) {
            Some(x) if x.cap != cap => x.cap = cap,
            _ => return Vec::new(),
        }
        self.pump()
    }

    /// A planned drain: a draining replica gets no new request and keeps its in-flight ones,
    /// which end as usual (`done`). Clearing it lets the waiting requests in.
    pub fn set_draining(&mut self, r: ReplicaId, draining: bool) -> Vec<Action> {
        match self.replicas.get_mut(r.0 as usize) {
            Some(x) if x.draining != draining => x.draining = draining,
            _ => return Vec::new(),
        }
        self.pump()
    }

    /// A replica joins (loaded, probed) or leaves (its node went `Down`, or its link broke). A
    /// leaving replica's in-flight requests fail; its prefixes are forgotten; the requests
    /// waiting here go to the others.
    pub fn set_up(&mut self, r: ReplicaId, up: bool) -> Vec<Action> {
        let Some(x) = self.replicas.get_mut(r.0 as usize) else {
            return Vec::new();
        };
        if x.up == up {
            return Vec::new();
        }
        x.up = up;
        x.draining = false;
        let mut out = Vec::new();
        if !up {
            self.leaves += 1;
            x.entries.clear();
            x.outstanding = 0;
            for req in self.in_flight_on(r) {
                self.inflight.remove(&req);
                out.push(Action::Fail { req, replica: r });
            }
        }
        out.extend(self.pump());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dispatches(a: &[Action]) -> Vec<(ReqId, u32)> {
        a.iter()
            .filter_map(|x| match x {
                Action::Dispatch { req, to, .. } => Some((*req, to.0)),
                Action::Fail { .. } => None,
            })
            .collect()
    }

    fn up2() -> Router {
        let mut r = Router::new(&[2, 2], 4);
        r.set_up(ReplicaId(0), true);
        r.set_up(ReplicaId(1), true);
        r
    }

    #[test]
    fn least_outstanding_then_waits_at_the_cap() {
        let mut r = up2();
        let a = (0..5)
            .flat_map(|i| r.submit(i, vec![i as u32]))
            .collect::<Vec<_>>();
        assert_eq!(dispatches(&a), vec![(0, 0), (1, 1), (2, 0), (3, 1)]);
        assert_eq!(r.queued(), 1);
        let a = r.done(1);
        assert_eq!(dispatches(&a), vec![(4, 1)]);
    }

    #[test]
    fn a_conversation_goes_back_to_its_replica() {
        let mut r = up2();
        let p1: Vec<u32> = (0..10).collect();
        r.submit(0, p1.clone());
        r.submit(1, vec![99; 10]);
        r.done(0);
        r.done(1);
        // Replica 0 holds p1; replica 1 has fewer outstanding or equal, affinity wins.
        let mut p2 = p1;
        p2.extend([50, 51]);
        let a = r.submit(2, p2);
        assert!(matches!(
            a[0],
            Action::Dispatch {
                to: ReplicaId(0),
                route: Route::Affinity,
                ..
            }
        ));
    }

    /// cluster-short-prefix-dropped: a holder that answered `gone` no longer holds the prefix,
    /// so it stops being an affinity target or a fetch candidate for it; a request still
    /// computing that prefix there keeps its entry.
    #[test]
    fn a_prefix_the_holder_lost_is_forgotten_but_a_live_one_kept() {
        let mut r = up2();
        let p: Vec<u32> = (0..10).collect();
        r.submit(0, p.clone());
        r.done(0);
        let mut longer = p.clone();
        longer.extend([60, 61]);
        r.submit(1, longer.clone()); // affinity: on 0, live
        assert_eq!(r.holds(ReplicaId(0), &p), 10);
        r.prefix_gone(ReplicaId(0), &p[..8]);
        assert_eq!(r.holds(ReplicaId(0), &p), 10, "the live request keeps it");
        r.done(1);
        r.prefix_gone(ReplicaId(0), &p[..8]);
        assert_eq!(r.holds(ReplicaId(0), &longer), 0);
        assert_eq!(r.holds(ReplicaId(1), &longer), 0);
    }

    #[test]
    fn a_full_holder_sends_the_request_elsewhere_with_a_fetch_candidate() {
        let mut r = Router::new(&[1, 1], 4);
        r.set_up(ReplicaId(0), true);
        r.set_up(ReplicaId(1), true);
        let p: Vec<u32> = (0..10).collect();
        r.submit(0, p.clone());
        r.done(0);
        r.submit(1, vec![7; 8]); // least outstanding tie goes to 0 (lowest id)
        let mut q = p;
        q.push(1);
        let a = r.submit(2, q);
        match &a[0] {
            Action::Dispatch { to, elsewhere, .. } => {
                assert_eq!(to.0, 1);
                assert_eq!(
                    *elsewhere,
                    Some(PrefixHit {
                        holder: ReplicaId(0),
                        tokens: 10
                    })
                );
            }
            Action::Fail { .. } => panic!("expected a dispatch"),
        }
    }

    #[test]
    fn a_leaving_replica_fails_its_requests_and_the_waiting_ones_move() {
        let mut r = Router::new(&[1, 1], 4);
        r.set_up(ReplicaId(0), true);
        r.set_up(ReplicaId(1), true);
        r.submit(0, vec![1]);
        r.submit(1, vec![2]);
        r.submit(2, vec![3]); // waits
        let a = r.set_up(ReplicaId(1), false);
        assert_eq!(a.len(), 1);
        assert!(matches!(a[0], Action::Fail { req: 1, .. }));
        // The waiting one still waits: replica 0 is full.
        assert_eq!(r.queued(), 1);
        let a = r.done(0);
        assert!(matches!(
            a[0],
            Action::Dispatch {
                req: 2,
                to: ReplicaId(0),
                moved: true,
                ..
            }
        ));
    }

    #[test]
    fn done_after_fail_is_ignored() {
        let mut r = up2();
        r.submit(0, vec![1]);
        let to = r.in_flight(0).unwrap();
        r.set_up(to, false);
        assert!(r.done(0).is_empty());
        assert_eq!(r.outstanding(to), 0);
    }

    #[test]
    fn a_requeued_request_waits_for_another_replica() {
        let mut r = up2();
        r.submit(0, vec![1]);
        let to = r.in_flight(0).unwrap();
        r.set_up(to, false);
        // `set_up` already failed it; requeue of an unknown request does nothing.
        assert!(r.requeue(0, false).is_empty());
        let a = r.submit(1, vec![2]);
        assert_ne!(dispatches(&a)[0].1, to.0);
    }

    #[test]
    fn nothing_goes_to_a_replica_that_is_not_up() {
        let mut r = Router::new(&[4], 4);
        assert!(r.submit(0, vec![1]).is_empty());
        assert_eq!(r.queued(), 1);
        let a = r.set_up(ReplicaId(0), true);
        assert_eq!(dispatches(&a), vec![(0, 0)]);
    }
}
