//! The head's side of the replicas (ADR 0091, step 5): [`ClusterEngine`] is the gateway's
//! `Engine` over every replica, with the router in front; [`ClusterHead`] keeps the replicas
//! joined: a node that is `Up` and whose replica answers one probe step joins, a node that goes
//! `Down` or a connection that breaks leaves, and a restored node joins again without anyone
//! asking.
//!
//! A request is *in flight* once the router sent it to a replica. When that replica leaves, the
//! request ends with exactly one `GenEvent::Error` and no event follows it; requests still
//! waiting at the router go to the other replicas. The terminal event of a request has one owner:
//! whoever takes the sender out of its [`Slot`] first (the replica's final event, the loss of the
//! replica, or a cancel).

use crate::kv::{PrefixGrants, PrefixKey, PrefixStore};
use crate::membership::NodeState;
use crate::metrics;
use crate::monitor::Monitor;
use crate::proto::{NodeReply, NodeRequest};
use crate::replica::{FetchFrom, RemoteEngine};
use crate::router::{Action, PrefixHit, ReplicaId, ReqId, Route, Router};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Duration;
use tetraeder_gateway::engine::{
    Engine, EngineError, FimTokens, GenEvent, GenerateRequest, ModelInfo, Vocab,
};
use tetraeder_transport::{NodeId, Transport};
use tokio::sync::mpsc;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Events a client's channel holds before the replica's stream waits for it.
const CLIENT_BUFFER: usize = 1024;

/// Decides whether a prefix another replica holds is fetched or recomputed (`kv::fetch_beats_recompute`
/// priced on the hardware graph; the head builds it from the probe curves).
pub trait FetchPolicy: Send + Sync {
    /// `tokens`: the prefix the holder has (`holder` is a replica index); `holder_kv`: what the
    /// holder reported about its snapshots. `Some`: ask the target to fetch it from the holder's
    /// node first.
    fn plan(
        &self,
        holder: u32,
        target: u32,
        tokens: &[u32],
        holder_kv: &KvInfo,
    ) -> Option<FetchFrom>;

    /// [`FetchPolicy::plan`] with why and what it priced, for the head's decision log line. A
    /// policy that prices overrides it; the default only knows its answer.
    fn decide(
        &self,
        holder: u32,
        target: u32,
        tokens: &[u32],
        holder_kv: &KvInfo,
    ) -> FetchDecision {
        let fetch = self.plan(holder, target, tokens, holder_kv);
        FetchDecision {
            reason: if fetch.is_some() {
                "priced_fetch"
            } else {
                "policy_recompute"
            },
            fetch,
            ..FetchDecision::default()
        }
    }
}

/// What the head decided for a prefix another replica holds, with the inputs it was priced on.
#[derive(Debug, Clone, Default)]
pub struct FetchDecision {
    /// `Some`: ask the target to fetch it.
    pub fetch: Option<FetchFrom>,
    /// `priced_fetch`, or why not (one of `metrics::FETCH_REASONS`).
    pub reason: &'static str,
    /// Snapshot bytes (tokens x the holder's row bytes); 0 when not known.
    pub bytes: u64,
    /// Predicted link time of the snapshot; `None` without a probed curve.
    pub hop_s: Option<f64>,
    /// Predicted copy of the snapshot into the target's KV.
    pub restore_s: f64,
    /// Predicted prefill of the prefix on the target; `None` without a rate.
    pub recompute_s: Option<f64>,
}

impl FetchDecision {
    /// Recompute, for `reason`.
    pub fn recompute(reason: &'static str) -> Self {
        Self {
            reason,
            ..Self::default()
        }
    }
}

/// A predicted time for the log: seconds, or `-` when not known.
fn secs(t: Option<f64>) -> String {
    t.map_or_else(|| "-".into(), |t| format!("{t:.6}"))
}

/// What a replica reported about its snapshots when it joined.
#[derive(Debug, Clone, Default)]
pub struct KvInfo {
    pub format: String,
    /// Bytes of one cached token; 0 when not known.
    pub row_bytes: u64,
}

/// One replica as the router sees it.
pub struct ReplicaTarget {
    /// What requests go to (the head's own engine, or a `RemoteEngine`).
    pub engine: Arc<dyn Engine>,
    /// Set for a replica reached over the transport (it takes a prefix fetch hint).
    pub remote: Option<Arc<RemoteEngine>>,
    /// Requests it takes at once (`n_seq_max`).
    pub cap: u32,
}

/// The terminal-event owner of a request.
struct Slot {
    tx: Mutex<Option<mpsc::Sender<GenEvent>>>,
}

impl Slot {
    /// Delivers `ev` (not a terminal one); false when the request has ended or the client is gone.
    fn send(&self, mut ev: GenEvent) -> bool {
        loop {
            {
                let mut g = lock(&self.tx);
                let Some(tx) = g.as_ref() else { return false };
                match tx.try_send(ev) {
                    Ok(()) => return true,
                    Err(mpsc::error::TrySendError::Closed(_)) => {
                        *g = None;
                        return false;
                    }
                    Err(mpsc::error::TrySendError::Full(e)) => ev = e,
                }
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// Delivers the terminal event; true when this call ended the request.
    fn finish(&self, mut ev: GenEvent) -> bool {
        loop {
            {
                let mut g = lock(&self.tx);
                let Some(tx) = g.as_ref() else { return false };
                match tx.try_send(ev) {
                    Ok(()) | Err(mpsc::error::TrySendError::Closed(_)) => {
                        *g = None;
                        return true;
                    }
                    Err(mpsc::error::TrySendError::Full(e)) => ev = e,
                }
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn gone(&self) -> bool {
        lock(&self.tx).as_ref().is_none_or(mpsc::Sender::is_closed)
    }
}

struct Pending {
    /// Taken when the request is sent to a replica.
    req: Option<GenerateRequest>,
    slot: Arc<Slot>,
    prompt: Vec<u32>,
    /// What the client asked for: a leader that streams more is cut off at this many tokens.
    max_tokens: u32,
    /// When it entered the router's queue.
    queued_at: std::time::Instant,
}

struct State {
    router: Router,
    reqs: HashMap<ReqId, Pending>,
    next: ReqId,
    kv_formats: Vec<String>,
    kv_row_bytes: Vec<u64>,
    /// Replicas a planned drain replaces: between the old one leaving and the new one joining
    /// they have no replica up, yet requests wait for them (see [`ClusterEngine::drain`]).
    replacing: Vec<bool>,
}

struct Inner {
    front: Arc<dyn Engine>,
    targets: Vec<ReplicaTarget>,
    state: Mutex<State>,
    policy: Option<Arc<dyn FetchPolicy>>,
    /// The grants of a replica that runs on the head's own node (see `offer_prefix`).
    local_grants: Mutex<Option<Arc<PrefixGrants>>>,
    stop: AtomicBool,
    counts: [AtomicU64; 4],
    /// How long a request may wait at the router for a replica with room.
    queue_deadline: Mutex<Duration>,
}

/// The default for [`ClusterEngine::set_queue_deadline`]: a request that no replica took in this
/// long ends with an error instead of waiting for a cluster that does not come back.
const QUEUE_DEADLINE: Duration = Duration::from_secs(120);

/// What the router did, as the `tetraeder.router.requests` results count it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counts {
    pub ok: u64,
    pub failed_in_flight: u64,
    pub moved: u64,
    pub affinity: u64,
}

/// The gateway's engine over the replicas. Waiting happens here: a replica gets at most its
/// `cap` requests, the rest queue and go to whichever replica has room.
pub struct ClusterEngine {
    inner: Arc<Inner>,
    tick: Option<JoinHandle<()>>,
}

fn terminal(ev: &GenEvent) -> bool {
    matches!(
        ev,
        GenEvent::Finished { .. } | GenEvent::Error(_) | GenEvent::ContextExceeded(_)
    )
}

impl ClusterEngine {
    /// `front` answers models, tokenization and grammar checks (the head holds the tokenizer).
    /// `min_prefix`: the shortest shared prefix that counts as held (the cache's
    /// `RESTORE_MIN_GAIN`). Every replica starts out of the router; see [`ClusterEngine::set_up`].
    pub fn new(
        front: Arc<dyn Engine>,
        targets: Vec<ReplicaTarget>,
        min_prefix: u32,
        policy: Option<Arc<dyn FetchPolicy>>,
    ) -> Self {
        let caps: Vec<u32> = targets.iter().map(|t| t.cap).collect();
        metrics::replicas_planned(caps.len());
        metrics::replicas_up(0);
        let inner = Arc::new(Inner {
            state: Mutex::new(State {
                router: Router::new(&caps, min_prefix),
                reqs: HashMap::new(),
                next: 1,
                kv_formats: vec![String::new(); targets.len()],
                kv_row_bytes: vec![0; targets.len()],
                replacing: vec![false; targets.len()],
            }),
            front,
            targets,
            policy,
            local_grants: Mutex::new(None),
            stop: AtomicBool::new(false),
            counts: Default::default(),
            queue_deadline: Mutex::new(QUEUE_DEADLINE),
        });
        let t = Arc::clone(&inner);
        let tick = std::thread::Builder::new()
            .name("router-tick".into())
            .spawn(move || {
                while !t.stop.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(20));
                    t.sweep();
                }
            })
            .ok();
        Self { inner, tick }
    }

    /// Sets how long a request may wait at the router before it ends with an error.
    pub fn set_queue_deadline(&self, d: Duration) {
        *lock(&self.inner.queue_deadline) = d;
    }

    /// The grants served by a replica that runs on the head's node (a target without a
    /// connection): a fetch from it is announced here instead of over a replica connection.
    pub fn set_local_grants(&self, g: Arc<PrefixGrants>) {
        *lock(&self.inner.local_grants) = Some(g);
    }

    /// Replica `r` runs on the head's own node, in this process: its snapshot format and size
    /// reach the fetch policy as a remote replica's do at join (without them a policy that
    /// prices the fetch has nothing to price and recomputes), and `grants` take the fetches
    /// announced for it (`set_local_grants`). The caller serves `store` on the node's `Prefix`
    /// purpose.
    pub fn set_local_replica(
        &self,
        r: u32,
        store: Arc<dyn PrefixStore>,
        grants: Arc<PrefixGrants>,
    ) {
        self.set_kv_format(r, &store.kv_format());
        self.set_kv_row_bytes(r, store.row_bytes());
        self.set_local_grants(grants);
    }

    /// A replica joins (`true`) or leaves the router. Leaving ends its in-flight requests with an
    /// error each; the requests waiting at the router go to the others.
    pub fn set_up(&self, r: u32, up: bool) {
        self.inner.set_up(r, up);
    }

    /// What replica `r` reported it takes at once: the router sends it at most the smaller of
    /// that and the cap it was configured with (requests over its limit would queue inside the
    /// replica, where they cannot move).
    pub fn set_reported_cap(&self, r: u32, reported: u32) {
        let configured = self.inner.targets[r as usize].cap;
        let cap = if reported == 0 {
            configured
        } else {
            configured.min(reported)
        };
        let actions = lock(&self.inner.state).router.set_cap(ReplicaId(r), cap);
        self.inner.execute(actions);
    }

    /// A planned drain starts (`true`) or ends: a draining replica takes no new request and keeps
    /// what is in flight. Unlike `set_up(r, false)` nothing fails.
    pub fn set_draining(&self, r: u32, draining: bool) {
        let actions = lock(&self.inner.state)
            .router
            .set_draining(ReplicaId(r), draining);
        self.inner.execute(actions);
    }

    /// The replacement of replica `r` is given up: with no replica up, requests are refused again.
    pub fn end_replace(&self, r: u32) {
        lock(&self.inner.state).replacing[r as usize] = false;
    }

    pub fn is_draining(&self, r: u32) -> bool {
        lock(&self.inner.state).router.is_draining(ReplicaId(r))
    }

    /// Drains replica `r` to replace it: no new requests, then waits until its in-flight ones
    /// have ended, at most `timeout`. `true`: nothing is left in flight. The replica stays
    /// draining; leaving the router (`set_up(r, false)`) or `set_draining(r, false)` ends it.
    /// Until the replacement joins (`set_up(r, true)`) or is given up ([`Self::end_replace`])
    /// requests wait at the router even while no replica is up, instead of being refused.
    pub fn drain(&self, r: u32, timeout: Duration) -> bool {
        let t0 = std::time::Instant::now();
        lock(&self.inner.state).replacing[r as usize] = true;
        self.set_draining(r, true);
        let drained = loop {
            if self.outstanding(r) == 0 {
                break true;
            }
            if t0.elapsed() >= timeout {
                break false;
            }
            std::thread::sleep(Duration::from_millis(5));
        };
        metrics::drain(
            r,
            if drained { "drained" } else { "timed_out" },
            t0.elapsed(),
        );
        drained
    }

    pub fn set_kv_format(&self, r: u32, format: &str) {
        if let Some(f) = lock(&self.inner.state).kv_formats.get_mut(r as usize) {
            *f = format.to_owned();
        }
    }

    pub fn set_kv_row_bytes(&self, r: u32, bytes: u64) {
        if let Some(b) = lock(&self.inner.state).kv_row_bytes.get_mut(r as usize) {
            *b = bytes;
        }
    }

    pub fn kv_row_bytes(&self, r: u32) -> u64 {
        lock(&self.inner.state)
            .kv_row_bytes
            .get(r as usize)
            .copied()
            .unwrap_or(0)
    }

    pub fn kv_format(&self, r: u32) -> String {
        lock(&self.inner.state)
            .kv_formats
            .get(r as usize)
            .cloned()
            .unwrap_or_default()
    }

    pub fn is_up(&self, r: u32) -> bool {
        lock(&self.inner.state).router.is_up(ReplicaId(r))
    }

    pub fn up_count(&self) -> usize {
        lock(&self.inner.state).router.up_count()
    }

    pub fn queued(&self) -> usize {
        lock(&self.inner.state).router.queued()
    }

    pub fn outstanding(&self, r: u32) -> u32 {
        lock(&self.inner.state).router.outstanding(ReplicaId(r))
    }

    pub fn counts(&self) -> Counts {
        let c = |i: usize| self.inner.counts[i].load(Ordering::SeqCst);
        Counts {
            ok: c(0),
            failed_in_flight: c(1),
            moved: c(2),
            affinity: c(3),
        }
    }

    /// A replica reported a prefix saved to or evicted from its cache.
    pub fn prefix_saved(&self, r: u32, tokens: &[u32]) {
        lock(&self.inner.state)
            .router
            .prefix_saved(ReplicaId(r), tokens);
    }

    pub fn prefix_evicted(&self, r: u32, tokens: &[u32]) {
        lock(&self.inner.state)
            .router
            .prefix_evicted(ReplicaId(r), tokens);
    }
}

impl Drop for ClusterEngine {
    fn drop(&mut self) {
        self.inner.stop.store(true, Ordering::SeqCst);
        if let Some(t) = self.tick.take() {
            let _ = t.join();
        }
    }
}

impl Inner {
    fn count(&self, replica: u32, result: &'static str) {
        let i = match result {
            "ok" => 0,
            "failed_in_flight" => 1,
            "moved" => 2,
            _ => 3,
        };
        self.counts[i].fetch_add(1, Ordering::SeqCst);
        metrics::request(replica, result);
    }

    fn set_up(self: &Arc<Self>, r: u32, up: bool) {
        let (actions, n) = {
            let mut st = lock(&self.state);
            let a = st.router.set_up(ReplicaId(r), up);
            if up {
                st.replacing[r as usize] = false;
            }
            (a, st.router.up_count())
        };
        metrics::replicas_up(n);
        self.execute(actions);
    }

    /// Drops requests whose client left while they waited, and ends those that waited longer than
    /// the queue deadline.
    fn sweep(self: &Arc<Self>) {
        let deadline = *lock(&self.queue_deadline);
        let actions = {
            let mut st = lock(&self.state);
            let expired: Vec<ReqId> = st
                .reqs
                .iter()
                .filter(|(_, p)| p.req.is_some() && p.queued_at.elapsed() >= deadline)
                .map(|(id, _)| *id)
                .collect();
            let mut out = Vec::new();
            for id in expired {
                out.extend(st.router.cancel(id));
                if let Some(p) = st.reqs.remove(&id) {
                    p.slot.finish(GenEvent::Error(format!(
                        "no replica took the request within {deadline:?}"
                    )));
                }
            }
            let gone: Vec<ReqId> = st
                .reqs
                .iter()
                .filter(|(_, p)| {
                    p.req
                        .as_ref()
                        .is_some_and(|r| r.cancel.is_cancelled() || p.slot.gone())
                })
                .map(|(id, _)| *id)
                .collect();
            for id in gone {
                st.reqs.remove(&id);
                out.extend(st.router.cancel(id));
            }
            out
        };
        self.execute(actions);
    }

    fn execute(self: &Arc<Self>, actions: Vec<Action>) {
        for a in actions {
            match a {
                Action::Dispatch {
                    req,
                    to,
                    route,
                    moved,
                    elsewhere,
                } => self.dispatch(req, to, route, moved, elsewhere),
                Action::Fail { req, replica } => {
                    let p = lock(&self.state).reqs.remove(&req);
                    if let Some(p) = p
                        && p.slot.finish(GenEvent::Error(format!(
                            "replica {} failed while the request was in flight",
                            replica.0
                        )))
                    {
                        self.count(replica.0, "failed_in_flight");
                    }
                }
            }
        }
    }

    fn dispatch(
        self: &Arc<Self>,
        id: ReqId,
        to: ReplicaId,
        route: Route,
        moved: bool,
        elsewhere: Option<PrefixHit>,
    ) {
        // A replica whose link is down cannot take it: nothing was started, so the request goes
        // back to the queue and the replica leaves until it is joined again.
        let target = &self.targets[to.0 as usize];
        if target.remote.as_ref().is_some_and(|r| !r.is_connected()) {
            let actions = lock(&self.state).router.requeue(id, true);
            self.execute(actions);
            return;
        }
        let (req, slot, prompt, max_tokens) = {
            let mut st = lock(&self.state);
            let Some(p) = st.reqs.get_mut(&id) else {
                // The client left between the router's decision and now.
                let a = st.router.cancel(id);
                drop(st);
                self.execute(a);
                return;
            };
            (
                p.req.take(),
                Arc::clone(&p.slot),
                p.prompt.clone(),
                p.max_tokens,
            )
        };
        let Some(req) = req else { return };
        if route == Route::Affinity {
            self.count(to.0, "affinity");
        }
        if moved {
            self.count(to.0, "moved");
        }
        let fetch = elsewhere.and_then(|hit| self.decide_fetch(id, hit, to, &prompt));
        let result = match &target.remote {
            Some(r) => r.generate_with(req, fetch),
            None => target.engine.generate(req),
        };
        match result {
            Ok(rx) => {
                let me = Arc::clone(self);
                let spawned = std::thread::Builder::new()
                    .name(format!("router-pump-{id}"))
                    .spawn(move || me.pump(id, to, &slot, rx, max_tokens));
                if spawned.is_err() {
                    self.settle(id);
                }
            }
            Err(e) => {
                // Rejected before it ran (the link broke under the call, or the request is
                // invalid): one error, and the slot is free.
                let _ = slot.finish(GenEvent::Error(e.to_string()));
                self.settle(id);
            }
        }
    }

    /// Fetch or recompute the prefix `hit` of `prompt` that another replica holds, for request
    /// `id` going to `to`. Every decision is logged with what it was priced on; a recompute is
    /// counted here, a fetch when the target reports how it went.
    fn decide_fetch(
        &self,
        id: ReqId,
        hit: PrefixHit,
        to: ReplicaId,
        prompt: &[u32],
    ) -> Option<FetchFrom> {
        let (holder, n) = (hit.holder.0, hit.tokens);
        let tokens = &prompt[..(n as usize).min(prompt.len())];
        let kv = {
            let st = lock(&self.state);
            KvInfo {
                format: st
                    .kv_formats
                    .get(holder as usize)
                    .cloned()
                    .unwrap_or_default(),
                row_bytes: st.kv_row_bytes.get(holder as usize).copied().unwrap_or(0),
            }
        };
        let mut d = match (&self.policy, &self.targets[to.0 as usize].remote) {
            (None, _) => FetchDecision::recompute("no_policy"),
            // The fetch hint rides the replica connection, which the head's own replica lacks.
            (Some(_), None) => FetchDecision::recompute("target_in_process"),
            (Some(policy), Some(_)) => policy.decide(holder, to.0, tokens, &kv),
        };
        // The holder serves only what it was told to: no announcement, no fetch.
        if let Some(f) = &d.fetch
            && !self.offer_prefix(holder, to.0, f)
        {
            d.fetch = None;
            d.reason = "not_offered";
        }
        tracing::info!(
            request = id,
            holder,
            to = to.0,
            tokens = n,
            bytes = d.bytes,
            hop_s = %secs(d.hop_s),
            restore_s = %secs(Some(d.restore_s)),
            recompute_s = %secs(d.recompute_s),
            decision = if d.fetch.is_some() { "fetch" } else { "recompute" },
            reason = d.reason,
            "prefix decision"
        );
        if d.fetch.is_none() {
            metrics::prefix_fetch("recomputed", metrics::fetch_reason(d.reason));
        }
        d.fetch
    }

    /// Announces to the holder's leader that the node of replica `to` may fetch `f` once; false
    /// when it could not be told (the fetch is then not asked for).
    fn offer_prefix(&self, holder: u32, to: u32, f: &FetchFrom) -> bool {
        let Some(requester) = self.targets[to as usize]
            .remote
            .as_ref()
            .map(|r| r.peer().id.clone())
        else {
            return false;
        };
        match &self.targets[holder as usize].remote {
            Some(r) => r.offer(requester, f.key.clone()),
            None => lock(&self.local_grants).as_ref().is_some_and(|g| {
                g.offer(requester, f.key.clone());
                true
            }),
        }
    }

    /// Forwards a replica's events to the client until the final one. A replica that streams
    /// more tokens than the client asked for is cut off at `max_tokens` (a hostile or confused
    /// leader cannot make a request run on).
    fn pump(
        self: &Arc<Self>,
        id: ReqId,
        to: ReplicaId,
        slot: &Slot,
        mut rx: mpsc::Receiver<GenEvent>,
        max_tokens: u32,
    ) {
        let mut tokens = 0u32;
        // Whoever delivers the final event counts the request, so each is counted once, also
        // when a leaving replica raced with its last event.
        loop {
            match rx.blocking_recv() {
                Some(GenEvent::Token(_)) if max_tokens > 0 && tokens >= max_tokens => {
                    let done = GenEvent::Finished {
                        reason: tetraeder_gateway::engine::FinishReason::Length,
                        stats: tetraeder_gateway::engine::EngineStats {
                            generated_tokens: tokens,
                            ..Default::default()
                        },
                    };
                    if slot.finish(done) {
                        self.count(to.0, "ok");
                    }
                    // Dropping `rx` makes the replica's link cancel the request.
                    break;
                }
                Some(ev @ GenEvent::Token(_)) => {
                    tokens += 1;
                    slot.send(ev);
                }
                Some(ev) if terminal(&ev) => {
                    let ok = !matches!(ev, GenEvent::Error(_));
                    if slot.finish(ev) {
                        self.count(to.0, if ok { "ok" } else { "failed_in_flight" });
                    }
                    break;
                }
                Some(ev) => {
                    slot.send(ev);
                }
                None => {
                    let msg = format!("replica {} ended the stream without a final event", to.0);
                    if slot.finish(GenEvent::Error(msg)) {
                        self.count(to.0, "failed_in_flight");
                    }
                    break;
                }
            }
        }
        self.settle(id);
    }

    /// The request ended: free its slot and send waiting requests out. A request the router
    /// already failed (its replica left) is not in flight any more and frees nothing.
    fn settle(self: &Arc<Self>, id: ReqId) {
        let actions = {
            let mut st = lock(&self.state);
            st.reqs.remove(&id);
            st.router.done(id)
        };
        self.execute(actions);
    }
}

impl Engine for ClusterEngine {
    fn models(&self) -> Vec<ModelInfo> {
        self.inner.front.models()
    }

    fn vocab(&self, model: &str) -> Result<Arc<Vocab>, EngineError> {
        self.inner.front.vocab(model)
    }

    fn tokenize(&self, model: &str, text: &str) -> Result<Vec<u32>, EngineError> {
        self.inner.front.tokenize(model, text)
    }

    fn tokenize_with(
        &self,
        model: &str,
        text: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<u32>, EngineError> {
        self.inner
            .front
            .tokenize_with(model, text, add_special, parse_special)
    }

    fn fim_tokens(&self, model: &str) -> FimTokens {
        self.inner.front.fim_tokens(model)
    }

    fn ready(&self) -> bool {
        self.up_count() > 0
    }

    fn metrics_text(&self) -> String {
        metrics::prefix_text()
    }

    fn generate(&self, req: GenerateRequest) -> Result<mpsc::Receiver<GenEvent>, EngineError> {
        let (tx, rx) = mpsc::channel(CLIENT_BUFFER);
        let slot = Arc::new(Slot {
            tx: Mutex::new(Some(tx)),
        });
        let actions = {
            let mut st = lock(&self.inner.state);
            // No replica is up (a draining one still is: a planned drain replaces it and the
            // requests wait for that). Waiting for a cluster that is gone is no answer: the
            // gateway turns this into its overloaded response with a retry hint.
            if st.router.up_count() == 0 && !st.replacing.iter().any(|&r| r) {
                return Err(EngineError::Overloaded);
            }
            let id = st.next;
            st.next += 1;
            let prompt = req.prompt.clone();
            let max_tokens = req.params.max_tokens;
            st.reqs.insert(
                id,
                Pending {
                    req: Some(req),
                    slot,
                    prompt: prompt.clone(),
                    max_tokens,
                    queued_at: std::time::Instant::now(),
                },
            );
            st.router.submit(id, prompt)
        };
        // Sending to a replica writes to a transport connection, which blocks on the transport's
        // own runtime: not from the gateway's async handler, so it runs on a thread of its own.
        let inner = Arc::clone(&self.inner);
        let spawned = std::thread::Builder::new()
            .name("router-submit".into())
            .spawn(move || inner.execute(actions));
        if let Err(e) = spawned {
            tracing::error!("cannot start the dispatch thread: {e}");
        }
        Ok(rx)
    }

    fn json_schema_to_grammar(&self, model: &str, schema: &str) -> Result<String, EngineError> {
        self.inner.front.json_schema_to_grammar(model, schema)
    }

    fn check_grammar(&self, model: &str, grammar: &str) -> Result<(), EngineError> {
        self.inner.front.check_grammar(model, grammar)
    }
}

// ---------------------------------------------------------------------------------------------
// Join and leave.
// ---------------------------------------------------------------------------------------------

/// What the daemon of a replica's leader node is told to start.
#[derive(Debug, Clone)]
pub struct StartSpec {
    pub plan: String,
    pub model_sha256: String,
    /// The segments of a replica that spans nodes, in order: the head assigns each to its
    /// node's daemon before the leader starts. Empty: the whole model on the leader.
    pub segments: Vec<SegmentAssign>,
    /// The planner's decode step time of each segment, seconds.
    pub predicted_s: Vec<f64>,
}

impl PartialEq for StartSpec {
    fn eq(&self, o: &Self) -> bool {
        self.plan == o.plan
            && self.model_sha256 == o.model_sha256
            && self.predicted_s == o.predicted_s
            && self.segments.len() == o.segments.len()
            && self
                .segments
                .iter()
                .zip(&o.segments)
                .all(|(a, b)| a.peer.id == b.peer.id && a.plan == b.plan)
    }
}

impl StartSpec {
    /// The nodes this replica needs: its leader and the node of every segment.
    pub fn nodes(&self, leader: &NodeId) -> Vec<NodeId> {
        let mut v = vec![leader.clone()];
        for s in &self.segments {
            if !v.contains(&s.peer.id) {
                v.push(s.peer.id.clone());
            }
        }
        v
    }
}

/// One segment of a replica: the daemon that runs its worker and its single-node plan.
#[derive(Debug, Clone)]
pub struct SegmentAssign {
    pub peer: tetraeder_transport::Peer,
    pub plan: String,
}

/// A replica's current spec, replaceable while the head runs (a replan). The generation changes
/// with every replacement, so the join loop notices.
struct SpecCell {
    spec: Mutex<StartSpec>,
    generation: AtomicU64,
}

impl SpecCell {
    fn snapshot(&self) -> (u64, StartSpec) {
        let g = lock(&self.spec);
        (self.generation.load(Ordering::SeqCst), g.clone())
    }
}

/// Keeps one remote replica joined to the router while its node is `Up`.
pub struct ClusterHead {
    stop: Arc<AtomicBool>,
    threads: Vec<JoinHandle<()>>,
    specs: Specs,
}

/// The specs of the remote replicas of a [`ClusterHead`]: readable and replaceable while it runs.
#[derive(Clone)]
pub struct Specs(Arc<HashMap<u32, (NodeId, Arc<SpecCell>)>>);

impl Specs {
    /// The replicas that have a spec (the remote ones), in order.
    pub fn replicas(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self.0.keys().copied().collect();
        v.sort_unstable();
        v
    }

    /// The spec replica `r` runs now (or will start with).
    pub fn spec(&self, r: u32) -> Option<StartSpec> {
        self.0.get(&r).map(|(_, c)| c.snapshot().1)
    }

    /// The leader of remote replica `r`; a replica keeps it (its connection is to that node).
    pub fn leader(&self, r: u32) -> Option<NodeId> {
        self.0.get(&r).map(|(n, _)| n.clone())
    }

    /// Replaces replica `r`'s spec. A joined replica leaves the router and starts again with it,
    /// so drain it first (`ClusterEngine::drain`) or its in-flight requests fail. `false`: not a
    /// remote replica.
    pub fn set(&self, r: u32, spec: StartSpec) -> bool {
        let Some((_, c)) = self.0.get(&r) else {
            return false;
        };
        let mut g = lock(&c.spec);
        if *g != spec {
            *g = spec;
            c.generation.fetch_add(1, Ordering::SeqCst);
        }
        true
    }
}

impl ClusterHead {
    /// For each remote replica `(index, node, spec)` of `engine`: while the node is `Up`, start
    /// its engine (`StartReplica`, repeated on every attempt; the daemon keeps a running one),
    /// connect, run one probe step, and then join the router. A connection that ends, or a node
    /// that leaves `Up`, takes the replica out; the loop then starts over, so a restored node (or
    /// a restarted engine) joins again without anyone asking.
    pub fn start(
        engine: &Arc<ClusterEngine>,
        monitor: &Arc<Monitor>,
        transport: &Arc<dyn Transport>,
        replicas: Vec<(u32, NodeId, StartSpec)>,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let mut threads = Vec::new();
        let mut specs = HashMap::new();
        for (r, node, spec) in replicas {
            let Some(remote) = engine.inner.targets[r as usize].remote.clone() else {
                continue;
            };
            let cell = Arc::new(SpecCell {
                spec: Mutex::new(spec),
                generation: AtomicU64::new(0),
            });
            specs.insert(r, (node.clone(), Arc::clone(&cell)));
            let e = Arc::clone(engine);
            remote.on_lost({
                let e = Arc::clone(engine);
                move || e.set_up(r, false)
            });
            let (m, t, stop) = (
                Arc::clone(monitor),
                Arc::clone(transport),
                Arc::clone(&stop),
            );
            if let Ok(h) = std::thread::Builder::new()
                .name(format!("join-{r}"))
                .spawn(move || join_loop(&e, &m, &t, r, &node, &cell, &remote, &stop))
            {
                threads.push(h);
            }
        }
        Self {
            stop,
            threads,
            specs: Specs(Arc::new(specs)),
        }
    }

    /// The specs of the remote replicas, to read and replace while the head runs.
    pub fn specs(&self) -> Specs {
        self.specs.clone()
    }

    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

impl Drop for ClusterHead {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Pause between two tries to start and join a replica, first and longest.
const RETRY_MIN: Duration = Duration::from_millis(100);
const RETRY_MAX: Duration = Duration::from_secs(2);
/// A joined replica that stays up this long has joined for good: its backoff starts over. One
/// that is lost sooner is a failed join (a crash loop), counted so and paced by the backoff.
const STABLE_AFTER: Duration = Duration::from_secs(30);

/// What a lost replica does to the join loop: `Some(pause)` before the next try when it was up
/// for less than `STABLE_AFTER` (and the backoff doubles), `None` when it was stable (the
/// backoff starts over and the next try is at once).
fn after_loss(stayed: Duration, backoff: &mut Duration) -> Option<Duration> {
    if stayed >= STABLE_AFTER {
        *backoff = RETRY_MIN;
        return None;
    }
    let pause = *backoff;
    *backoff = (*backoff * 2).min(RETRY_MAX);
    Some(pause)
}

fn sleep_unless(stop: &AtomicBool, d: Duration) {
    let t0 = std::time::Instant::now();
    while t0.elapsed() < d && !stop.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(10).min(d));
    }
}

/// Asks `peer` to stop the worker of `(replica, segment)`; best effort.
fn release(
    transport: &dyn Transport,
    peer: &tetraeder_transport::Peer,
    replica: u32,
    segment: u32,
) {
    let _ = crate::client::request(
        transport,
        peer,
        &NodeRequest::Release { replica, segment },
        &[],
    );
}

#[allow(clippy::too_many_arguments)]
fn join_loop(
    e: &Arc<ClusterEngine>,
    monitor: &Arc<Monitor>,
    transport: &Arc<dyn Transport>,
    r: u32,
    node: &NodeId,
    cell: &SpecCell,
    remote: &Arc<RemoteEngine>,
    stop: &AtomicBool,
) {
    // The spec the last attempt started: what a new spec must clean up.
    let mut previous: Option<StartSpec> = None;
    // Pause between failed tries: doubles up to `RETRY_MAX`, starts over after a join that
    // stayed up for `STABLE_AFTER`.
    let mut backoff = RETRY_MIN;
    let mut attempts = 0u64;
    while !stop.load(Ordering::SeqCst) {
        let (generation, spec) = cell.snapshot();
        let needed = spec.nodes(node);
        // Starting needs every node `Up` (a draining one takes no assignments); a joined replica
        // stays while its nodes are `Up` or `Draining` until a replan replaces it.
        if needed
            .iter()
            .any(|n| monitor.state(n) != Some(NodeState::Up))
        {
            sleep_unless(stop, Duration::from_millis(20));
            continue;
        }
        let alive = || {
            needed
                .iter()
                .all(|n| matches!(monitor.state(n), Some(NodeState::Up | NodeState::Draining)))
        };
        let mut attempt = || -> Result<crate::replica::ReplicaInfo, String> {
            if let Some(old) = previous.replace(spec.clone())
                && old != spec
            {
                // Workers of the old plan that the new one does not reuse at the same place.
                for (i, s) in old.segments.iter().enumerate() {
                    let same = spec.segments.get(i).is_some_and(|n| n.peer.id == s.peer.id);
                    if !same {
                        release(
                            transport.as_ref(),
                            &s.peer,
                            r,
                            u32::try_from(i).unwrap_or(0),
                        );
                    }
                }
            }
            if !spec.segments.is_empty() {
                // A replica across nodes starts as a whole: the leader goes first, so that it
                // meets segment workers with empty KV (an `Assign` restarts its worker).
                release(
                    transport.as_ref(),
                    remote.peer(),
                    r,
                    crate::node::REPLICA_SEGMENT,
                );
            }
            for (i, s) in spec.segments.iter().enumerate() {
                crate::client::request(
                    transport.as_ref(),
                    &s.peer,
                    &NodeRequest::AssignLed {
                        replica: r,
                        segment: u32::try_from(i).map_err(|e| e.to_string())?,
                        plan: s.plan.clone(),
                        model_sha256: spec.model_sha256.clone(),
                        leader: remote.peer().id.clone(),
                    },
                    &[],
                )?;
            }
            let (reply, _) = crate::client::request(
                transport.as_ref(),
                remote.peer(),
                &NodeRequest::StartReplica {
                    replica: r,
                    plan: spec.plan.clone(),
                    model_sha256: spec.model_sha256.clone(),
                    segments: spec.segments.iter().map(|s| s.peer.id.0.clone()).collect(),
                    predicted_s: spec.predicted_s.clone(),
                },
                &[],
            )?;
            if !matches!(reply, NodeReply::Started) {
                return Err(format!("unexpected reply {reply:?}"));
            }
            let info = remote.connect(&|close| monitor.watch_conn(node, close))?;
            remote.probe()?;
            Ok(info)
        };
        attempts += 1;
        tracing::info!(replica = r, %node, attempts, "starting the replica and joining it");
        match attempt() {
            Ok(info) => {
                metrics::join_attempt(r, "joined");
                let joined_at = std::time::Instant::now();
                tracing::info!(replica = r, %node, "replica joined");
                e.set_kv_format(r, &info.kv_format);
                e.set_kv_row_bytes(r, info.kv_row_bytes);
                e.set_reported_cap(r, info.cap);
                let current = || cell.generation.load(Ordering::SeqCst) == generation;
                // A node may have gone down, or the spec changed, while the probe ran.
                if alive() && remote.is_connected() && current() {
                    e.set_up(r, true);
                }
                // A node told to drain (`tetraeder node --drain`) takes no new request on this
                // replica; what is in flight finishes. With a cluster plan the replanner drains
                // and replaces the replica as well, and sets the same flag.
                let mut draining = false;
                while !stop.load(Ordering::SeqCst) && remote.is_connected() && alive() && current()
                {
                    let d = needed
                        .iter()
                        .any(|n| monitor.state(n) == Some(NodeState::Draining));
                    if d != draining {
                        draining = d;
                        tracing::info!(replica = r, %node, draining, "node drain: replica {}", if d { "takes no new request" } else { "takes requests again" });
                        e.set_draining(r, d);
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
                // Lost, as opposed to stopped or replaced by a new spec.
                let lost = !stop.load(Ordering::SeqCst)
                    && current()
                    && (!remote.is_connected() || !alive());
                e.set_up(r, false);
                remote.disconnect();
                if lost && let Some(pause) = after_loss(joined_at.elapsed(), &mut backoff) {
                    metrics::join_attempt(r, "failed");
                    tracing::warn!(replica = r, %node, attempts, "replica lost {:?} after it joined; next try in {pause:?}", joined_at.elapsed());
                    sleep_unless(stop, pause);
                }
            }
            Err(err) => {
                metrics::join_attempt(r, "failed");
                tracing::warn!(replica = r, %node, attempts, "join attempt failed: {err}; next in {backoff:?}");
                remote.disconnect();
                sleep_unless(stop, backoff);
                backoff = (backoff * 2).min(RETRY_MAX);
            }
        }
    }
    remote.disconnect();
}

/// The fetch key of a prefix: the holder's cache format and the tokens.
pub fn prefix_key(format: &str, tokens: &[u32]) -> PrefixKey {
    PrefixKey {
        format: format.to_owned(),
        tokens: tokens.to_vec(),
    }
}

/// The replicas of a cluster plan as the head starts them: for replica `i` of `plan`, its leader
/// (the node of its first segment, ADR 0091) and its [`StartSpec`]. A replica of one segment runs
/// whole on its leader (`segments` empty, `plan` the segment's single-node plan); one of several
/// lists every segment with its single-node plan (`plan::segments`, `plan::segment_plan`).
/// `predicted_s[i]` is the planner's decode step time of each segment of replica `i` (empty when
/// not priced).
pub fn specs_from_plan(
    plan: &tetraeder_plan::Plan,
    hw: &tetraeder_profiler::HardwareGraph,
    cfg: &crate::ClusterConfig,
    model_sha256: &str,
    predicted_s: &[Vec<f64>],
) -> Result<Vec<(u32, NodeId, StartSpec)>, String> {
    let mut out = Vec::new();
    for i in 0..plan.replicas.len() {
        let segs = tetraeder_plan::segments(plan, i, hw).map_err(|e| e.join("; "))?;
        let mut assigns = Vec::new();
        for s in &segs {
            let node = s.node.clone().ok_or_else(|| {
                format!("replica {i}: a segment of a cluster plan sits on a node")
            })?;
            let peer = cfg
                .peer(&NodeId::new(node.clone()))
                .ok_or_else(|| format!("`{node}` is not a node of cluster.toml"))?;
            let sub = tetraeder_plan::segment_plan(plan, i, s);
            let plan_toml = sub.to_toml().map_err(|e| e.to_string())?;
            assigns.push(SegmentAssign {
                peer,
                plan: plan_toml,
            });
        }
        let leader = assigns
            .first()
            .map(|a| a.peer.id.clone())
            .ok_or_else(|| format!("replica {i} has no segment"))?;
        let predicted = predicted_s.get(i).cloned().unwrap_or_default();
        let (plan_toml, segments, predicted_s) = if assigns.len() == 1 {
            (assigns.remove(0).plan, Vec::new(), Vec::new())
        } else {
            // The leader's engine reads the knobs (prefill micro-batches, ubatch) from this.
            (
                plan.to_toml().map_err(|e| e.to_string())?,
                assigns,
                predicted,
            )
        };
        out.push((
            u32::try_from(i).map_err(|e| e.to_string())?,
            leader,
            StartSpec {
                plan: plan_toml,
                model_sha256: model_sha256.to_owned(),
                segments,
                predicted_s,
            },
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod join_tests {
    use super::*;

    /// A replica lost soon after it joined is a failed join and backs off; one that stayed up
    /// starts over (a crash loop must not retry at full speed or look healthy).
    #[test]
    fn a_join_lost_within_the_stable_time_backs_off() {
        let mut b = RETRY_MIN;
        let mut pauses = Vec::new();
        for _ in 0..6 {
            pauses.push(after_loss(Duration::from_secs(1), &mut b).unwrap());
        }
        assert_eq!(pauses[0], RETRY_MIN);
        assert!(pauses.windows(2).all(|w| w[1] >= w[0]));
        assert_eq!(*pauses.last().unwrap(), RETRY_MAX);
        assert_eq!(after_loss(STABLE_AFTER, &mut b), None);
        assert_eq!(b, RETRY_MIN, "a stable replica resets the backoff");
        assert_eq!(
            after_loss(STABLE_AFTER - Duration::from_millis(1), &mut b),
            Some(RETRY_MIN)
        );
    }
}
