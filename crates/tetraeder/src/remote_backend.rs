//! `RemoteBackend`: a [`BatchBackend`] served by a worker process over its stdin and stdout
//! (ADR 0069 "Worker process, one per box" and "Failure handling"), and the wire format both
//! sides share.
//!
//! Frame: `u32` LE length of the rest, `u32` LE header length, a `serde_json` header, then a raw
//! payload (the logits of a `decode` reply as little-endian f32, snapshot bytes).
//!
//! Supervision: a call that meets EOF, a broken pipe, a malformed frame, a sticky error of the
//! worker or no reply within the deadline kills the worker and fails with
//! [`BackendError::Failed`]. The next call starts it again (after a backoff) with the same
//! arguments. Sequences do not survive a restart: the backend then reports every sequence as
//! length 0 and refuses `decode` and `seq_cp` on those that were live, until they are freed
//! (`seq_rm(seq, 0)`) or filled again (`seq_cp` into them, `seq_restore`). The scheduler's
//! `Failed` path then fails exactly their requests and the queue survives.

use opentelemetry::KeyValue;
use opentelemetry::metrics::{Counter, Histogram};
use std::collections::{HashSet, VecDeque};
use std::ffi::OsString;
use std::io::{BufReader, Read, Write};
use std::ops::Range;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::time::{Duration, Instant};
use tetraeder_scheduler::{
    BackendError, BatchBackend, Feed, Outputs, SegmentBackend, SegmentInfo, SeqId, SeqInput,
    StepOut,
};
use tetraeder_transport::{LinkMetrics, Peer, Purpose, Transport};

/// What the two processes say to each other.
pub mod wire {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct WireInput {
        pub seq: u32,
        pub tokens: Vec<i32>,
        pub all_logits: bool,
    }

    /// One call. Tokens travel in the header; bytes (a restored snapshot) in the frame payload.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(tag = "call", rename_all = "snake_case")]
    pub enum Request {
        Info,
        SeqReserve {
            seq: u32,
            n: usize,
        },
        SeqRm {
            seq: u32,
            keep: usize,
        },
        SeqCp {
            src: u32,
            dst: u32,
            n: usize,
        },
        Decode {
            inputs: Vec<WireInput>,
            /// W3C trace context of the step (ADR 0091). Left out of the frame when `None`, so
            /// a header without a trace is byte-identical to the one before the field existed.
            #[serde(
                default,
                rename = "traceparent",
                skip_serializing_if = "Option::is_none"
            )]
            trace: Option<String>,
        },
        SeqSnapshot {
            seq: u32,
        },
        /// The snapshot is the frame payload.
        SeqRestore {
            seq: u32,
            n: usize,
        },
        /// The worker's expert-cache counters (`BatchBackend::expert_stats_json`), in the reply.
        Stats,
    }

    impl Request {
        pub fn name(&self) -> &'static str {
            match self {
                Self::Info => "info",
                Self::SeqReserve { .. } => "seq_reserve",
                Self::SeqRm { .. } => "seq_rm",
                Self::SeqCp { .. } => "seq_cp",
                Self::Decode { .. } => "decode",
                Self::SeqSnapshot { .. } => "seq_snapshot",
                Self::SeqRestore { .. } => "seq_restore",
                Self::Stats => "stats",
            }
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum WireError {
        KvFull {
            seq: u32,
            needed: usize,
            free: usize,
        },
        /// `sticky`: the worker cannot be trusted with another call (a CUDA error); the parent
        /// kills and restarts it.
        Failed { msg: String, sticky: bool },
    }

    #[derive(Debug, Clone, Default, Serialize, Deserialize)]
    pub struct SeqState {
        pub len: usize,
        pub reserved: usize,
    }

    /// The mutable facts the trait reads through `&self`; every reply carries them.
    #[derive(Debug, Clone, Default, Serialize, Deserialize)]
    pub struct State {
        pub kv_free: usize,
        pub seqs: Vec<SeqState>,
        pub bytes_streamed: Option<u64>,
    }

    /// What a worker that runs a share of the layers holds (ADR 0091, step 4).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub struct SegmentInfo {
        pub embed: bool,
        pub head: bool,
        pub n_embd: usize,
        /// Layers computed, `[start, end)`.
        pub layers: (u32, u32),
    }

    /// The constant facts, sent in the reply to `Info`.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Info {
        pub n_vocab: usize,
        pub n_seq_max: usize,
        pub n_ctx_seq: usize,
        pub kv_capacity: usize,
        pub kv_format: String,
        pub on_gpu: bool,
        pub kv_row_bytes: usize,
        pub max_step_tokens: u64,
        /// `None` for a whole model. A `decode` to a segment without the embedding carries the
        /// hidden rows of every input token as its payload (F32 little endian, `n_embd` per
        /// token, tokens still in the header); a reply from a segment without the head carries
        /// the hidden rows of every input token, and `spans` count rows.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub segment: Option<SegmentInfo>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Reply {
        pub err: Option<WireError>,
        /// Row ranges (start, end) of each input of a `decode` in the payload.
        #[serde(default)]
        pub spans: Vec<(usize, usize)>,
        #[serde(default)]
        pub info: Option<Info>,
        pub state: State,
        /// Seconds the worker spent in the backend for a `decode`: the compute time of a segment,
        /// apart from the network (`tetraeder.segment.duration`).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub step_s: Option<f64>,
        /// The answer to `Stats`: the backend's expert-cache counters, when it has any.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub stats: Option<serde_json::Value>,
    }
}

use wire::{Reply, Request, WireError};

// The frame format lives in `tetraeder-transport` (ADR 0091), byte for byte as it was here.
pub use tetraeder_transport::frame::{
    Frame, MAX_FRAME, encode, encode_raw, parse_header, read_frame, write_frame,
};

/// Where the worker runs.
pub enum Endpoint {
    /// A child process on this box, framed over its stdin and stdout (ADR 0069).
    Child {
        program: PathBuf,
        args: Vec<OsString>,
    },
    /// A worker on a node: connect with `Purpose::Segment`. The node's daemon starts and
    /// restarts the worker; this side reconnects with the same backoff and restart cap.
    Node {
        transport: Arc<dyn Transport>,
        peer: Peer,
        replica: u32,
        segment: u32,
    },
}

/// How to start and supervise the worker.
pub struct RemoteConfig {
    pub endpoint: Endpoint,
    /// Predicted step time of the served plan, when there is one.
    pub predicted_step: Option<Duration>,
    /// Shortest reply deadline (5 s).
    pub deadline_floor: Duration,
    /// How long a (re)started worker may take to load and answer `Info`.
    pub startup_timeout: Duration,
    /// Restart delays are `backoff_base` times 1, 2, 4, 4, ... (1 s base).
    pub backoff_base: Duration,
    /// Restarts allowed within `window` (5 in 10 minutes); one more gives up.
    pub max_restarts: usize,
    pub window: Duration,
    /// Called once when the cap is reached. The default prints and exits non-zero.
    pub on_give_up: Box<dyn Fn(&str) + Send>,
}

impl RemoteConfig {
    pub fn new(program: impl Into<PathBuf>, args: Vec<OsString>) -> Self {
        Self::with_endpoint(Endpoint::Child {
            program: program.into(),
            args,
        })
    }

    /// A worker on a node of the cluster, reached over `transport`.
    pub fn node(transport: Arc<dyn Transport>, peer: Peer, replica: u32, segment: u32) -> Self {
        Self::with_endpoint(Endpoint::Node {
            transport,
            peer,
            replica,
            segment,
        })
    }

    fn with_endpoint(endpoint: Endpoint) -> Self {
        Self {
            endpoint,
            predicted_step: None,
            deadline_floor: Duration::from_secs(5),
            startup_timeout: Duration::from_secs(30 * 60),
            backoff_base: Duration::from_secs(1),
            max_restarts: 5,
            window: Duration::from_secs(600),
            on_give_up: Box::new(|why| {
                eprintln!("error: {why}");
                std::process::exit(1);
            }),
        }
    }

    /// `max(5 s, 10 x predicted step time)`.
    pub fn deadline(&self) -> Duration {
        self.predicted_step
            .map_or(self.deadline_floor, |p| (p * 10).max(self.deadline_floor))
    }
}

/// How the worker is reached, and what ending it means.
enum Link {
    Proc(std::process::Child),
    /// The `close` of the connection: a blocked read or write on it fails at once.
    Conn(Arc<dyn Fn() + Send + Sync>),
}

struct Child {
    link: Link,
    tx: Sender<Vec<u8>>,
    rx: Receiver<Result<Frame, String>>,
}

impl Child {
    fn kill(self) {
        match self.link {
            Link::Proc(mut p) => {
                let _ = p.kill();
                let _ = p.wait();
            }
            Link::Conn(close) => close(),
        }
    }
}

/// A thread for each direction, so a worker that stops reading or writing never blocks the
/// caller past its deadline.
fn pump(
    mut out: impl Write + Send + 'static,
    input: impl Read + Send + 'static,
) -> (Sender<Vec<u8>>, Receiver<Result<Frame, String>>) {
    let (tx, wrx) = mpsc::channel::<Vec<u8>>();
    std::thread::spawn(move || {
        while let Ok(bytes) = wrx.recv() {
            if out.write_all(&bytes).and_then(|()| out.flush()).is_err() {
                break;
            }
        }
    });
    let (rtx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut input = BufReader::new(input);
        loop {
            match read_frame(&mut input) {
                Ok(Some(f)) => {
                    if rtx.send(Ok(f)).is_err() {
                        break;
                    }
                }
                Ok(None) => {
                    let _ = rtx.send(Err("worker closed its output".into()));
                    break;
                }
                Err(e) => {
                    let _ = rtx.send(Err(e));
                    break;
                }
            }
        }
    });
    (tx, rx)
}

/// Starts the worker, or connects to it.
fn open(cfg: &RemoteConfig) -> Result<Child, String> {
    match &cfg.endpoint {
        Endpoint::Child { program, args } => {
            let mut proc = Command::new(program)
                .args(args)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()
                .map_err(|e| format!("cannot start {}: {e}", program.display()))?;
            let stdin = proc.stdin.take().ok_or("worker has no stdin")?;
            let stdout = proc.stdout.take().ok_or("worker has no stdout")?;
            let (tx, rx) = pump(stdin, stdout);
            Ok(Child {
                link: Link::Proc(proc),
                tx,
                rx,
            })
        }
        Endpoint::Node {
            transport,
            peer,
            replica,
            segment,
        } => {
            let conn = transport
                .connect(
                    peer,
                    Purpose::Segment {
                        replica: *replica,
                        segment: *segment,
                    },
                )
                .map_err(|e| format!("cannot connect to node {}: {e}", peer.id))?;
            let (tx, rx) = pump(conn.tx, conn.rx);
            Ok(Child {
                link: Link::Conn(conn.close),
                tx,
                rx,
            })
        }
    }
}

fn send_request(child: &Child, req: &Request, payload: &[u8]) -> Result<(), String> {
    let bytes = encode(req, payload)?;
    child
        .tx
        .send(bytes)
        .map_err(|_| "broken pipe to the worker".to_owned())
}

fn exchange(
    child: &Child,
    req: &Request,
    payload: &[u8],
    deadline: Duration,
) -> Result<(Reply, Vec<u8>), String> {
    send_request(child, req, payload)?;
    receive_reply(child, deadline)
}

fn reply_of(msg: Result<Frame, String>) -> Result<(Reply, Vec<u8>), String> {
    let frame = msg?;
    Ok((parse_header(&frame)?, frame.payload))
}

fn receive_reply(child: &Child, deadline: Duration) -> Result<(Reply, Vec<u8>), String> {
    match child.rx.recv_timeout(deadline) {
        Ok(msg) => reply_of(msg),
        Err(RecvTimeoutError::Timeout) => Err(format!("no reply within {deadline:?}")),
        Err(RecvTimeoutError::Disconnected) => Err("worker exited".into()),
    }
}

/// Starts a worker and reads its `Info`.
fn handshake(cfg: &RemoteConfig) -> Result<(Child, wire::Info, wire::State), String> {
    let child = open(cfg)?;
    match exchange(&child, &Request::Info, &[], cfg.startup_timeout) {
        Ok((
            Reply {
                info: Some(info),
                state,
                err: None,
                ..
            },
            _,
        )) => Ok((child, info, state)),
        Ok((reply, _)) => {
            child.kill();
            Err(format!("the worker's answer to info was {:?}", reply.err))
        }
        Err(e) => {
            child.kill();
            Err(format!("worker did not start: {e}"))
        }
    }
}

pub struct RemoteBackend {
    cfg: RemoteConfig,
    child: Option<Child>,
    info: wire::Info,
    state: wire::State,
    /// Sequences that held rows when the worker died and have not been freed or refilled since.
    lost: HashSet<u32>,
    restarts: VecDeque<Instant>,
    n_restarts: usize,
    gave_up: bool,
    logits: Vec<f32>,
    spans: Vec<Range<usize>>,
    ipc: Histogram<f64>,
    restart_counter: Counter<u64>,
    /// Round trips as `tetraeder.link.latency`, for a worker on a node.
    link: Option<LinkLatency>,
    /// The worker's own time for the last `decode` (`Reply::step_s`).
    last_step: Option<Duration>,
    /// Steps sent by `start_step` and not yet finished, oldest first: (sent at, inputs).
    pending: VecDeque<(Instant, usize)>,
    /// A reply `reply_ready` took off the channel, with its arrival time, for the next finish.
    stash: Option<(Instant, Result<Frame, String>)>,
}

/// `tetraeder.link.latency` of one link, by what the call carries.
struct LinkLatency {
    hidden: LinkMetrics,
    logits: LinkMetrics,
    kv: LinkMetrics,
    request: LinkMetrics,
}

impl LinkLatency {
    fn new(transport: &dyn Transport, peer: &Peer) -> Self {
        let m = |p| LinkMetrics::new(transport.local(), &peer.id, transport.name(), p);
        Self {
            hidden: m("hidden"),
            logits: m("logits"),
            kv: m("kv"),
            request: m("request"),
        }
    }

    /// `logits`: the reply of a `decode` is logits (the worker holds the head), else hidden rows.
    fn record(&self, call: &str, secs: f64, logits: bool) {
        match call {
            "decode" if logits => &self.logits,
            "decode" => &self.hidden,
            "seq_snapshot" | "seq_restore" => &self.kv,
            _ => &self.request,
        }
        .latency(secs);
    }
}

/// The W3C `traceparent` of the current span, if there is a valid one.
fn current_traceparent() -> Option<String> {
    use opentelemetry::trace::TraceContextExt;
    use tracing_opentelemetry::OpenTelemetrySpanExt;
    let from_tracing = tracing::Span::current().context();
    let sc = from_tracing.span().span_context().clone();
    let sc = if sc.is_valid() {
        sc
    } else {
        opentelemetry::Context::current()
            .span()
            .span_context()
            .clone()
    };
    sc.is_valid().then(|| {
        format!(
            "00-{}-{}-{:02x}",
            sc.trace_id(),
            sc.span_id(),
            sc.trace_flags().to_u8()
        )
    })
}

impl RemoteBackend {
    /// Starts the worker and waits until it has loaded. An error is a startup error.
    pub fn start(cfg: RemoteConfig) -> Result<Self, String> {
        let (child, info, state) = handshake(&cfg)?;
        let link = match &cfg.endpoint {
            Endpoint::Node {
                transport, peer, ..
            } => Some(LinkLatency::new(transport.as_ref(), peer)),
            Endpoint::Child { .. } => None,
        };
        let meter = opentelemetry::global::meter("tetraeder");
        Ok(Self {
            ipc: meter
                .f64_histogram("tetraeder.executor.ipc.duration")
                .with_unit("s")
                .with_description("Round trip of one call to the worker process")
                .build(),
            restart_counter: meter
                .u64_counter("tetraeder.executor.worker.restarts")
                .with_unit("{restart}")
                .with_description("Worker processes started again after a failure")
                .build(),
            cfg,
            child: Some(child),
            info,
            state,
            lost: HashSet::new(),
            restarts: VecDeque::new(),
            n_restarts: 0,
            gave_up: false,
            logits: Vec::new(),
            spans: Vec::new(),
            link,
            last_step: None,
            pending: VecDeque::new(),
            stash: None,
        })
    }

    /// Worker restarts so far.
    pub fn restarts(&self) -> usize {
        self.n_restarts
    }

    /// Kills the worker; its sequences are gone.
    fn kill(&mut self) {
        self.pending.clear();
        self.stash = None;
        if let Some(c) = self.child.take() {
            c.kill();
        }
        for (i, s) in self.state.seqs.iter().enumerate() {
            if s.len > 0 || s.reserved > 0 {
                self.lost.insert(i as u32);
            }
        }
        self.state = wire::State {
            kv_free: self.info.kv_capacity,
            seqs: vec![wire::SeqState::default(); self.info.n_seq_max],
            bytes_streamed: self.state.bytes_streamed,
        };
    }

    fn give_up(&mut self, why: String) -> String {
        self.gave_up = true;
        (self.cfg.on_give_up)(&why);
        why
    }

    /// Starts the worker again if it is dead: backoff 1 s, 2 s, 4 s, ...; the cap gives up.
    fn ensure_running(&mut self) -> Result<(), String> {
        if self.gave_up {
            return Err("the worker restart cap was reached".into());
        }
        while self.child.is_none() {
            let now = Instant::now();
            while self
                .restarts
                .front()
                .is_some_and(|t| now.duration_since(*t) > self.cfg.window)
            {
                self.restarts.pop_front();
            }
            if self.restarts.len() >= self.cfg.max_restarts {
                let why = format!(
                    "the worker died {} times within {:?}; giving up",
                    self.cfg.max_restarts + 1,
                    self.cfg.window
                );
                return Err(self.give_up(why));
            }
            std::thread::sleep(self.cfg.backoff_base * (1u32 << self.restarts.len().min(2)));
            self.restarts.push_back(Instant::now());
            self.n_restarts += 1;
            self.restart_counter.add(1, &[]);
            match handshake(&self.cfg) {
                Ok((child, info, state))
                    if (info.n_vocab, info.n_seq_max, info.kv_capacity)
                        == (
                            self.info.n_vocab,
                            self.info.n_seq_max,
                            self.info.kv_capacity,
                        ) =>
                {
                    self.child = Some(child);
                    self.state = state;
                }
                Ok((child, ..)) => {
                    child.kill();
                    eprintln!("warning: the restarted worker has a different shape");
                }
                Err(e) => eprintln!("warning: worker restart failed: {e}"),
            }
        }
        Ok(())
    }

    /// One call. `Ok` only for a reply without an error; a sticky error kills the worker.
    fn call(&mut self, req: &Request, payload: &[u8]) -> Result<(Reply, Vec<u8>), BackendError> {
        let t0 = self.send_call(req, payload)?;
        self.recv_call(req.name(), t0)
    }

    /// Sends one request without waiting; `recv_call` takes its reply. Replies come in the order
    /// of the requests, the worker serves one at a time.
    fn send_call(&mut self, req: &Request, payload: &[u8]) -> Result<Instant, BackendError> {
        self.ensure_running().map_err(BackendError::Failed)?;
        let child = self.child.as_ref().expect("running");
        let t0 = Instant::now();
        match send_request(child, req, payload) {
            Ok(()) => Ok(t0),
            Err(e) => {
                self.record_call(req.name(), t0, Instant::now());
                self.kill();
                Err(BackendError::Failed(format!("worker {}: {e}", req.name())))
            }
        }
    }

    /// The call as `tetraeder.ipc` and, for a worker on a node, `tetraeder.link.latency`: one
    /// sample per call, so one per micro-batch of a pipelined prefill.
    fn record_call(&self, name: &'static str, t0: Instant, t1: Instant) {
        let took = t1.duration_since(t0).as_secs_f64();
        self.ipc
            .record(took, &[KeyValue::new("tetraeder.call", name)]);
        if let Some(l) = &self.link {
            let logits = self.info.segment.is_none_or(|s| s.head);
            l.record(name, took, logits);
        }
    }

    /// The reply to the oldest request that `send_call` made at `t0`. `Ok` only for a reply
    /// without an error; a sticky error kills the worker.
    fn recv_call(
        &mut self,
        name: &'static str,
        t0: Instant,
    ) -> Result<(Reply, Vec<u8>), BackendError> {
        // A reply that `reply_ready` already took off the channel keeps its arrival time.
        let (res, t1) = match self.stash.take() {
            Some((at, msg)) => (reply_of(msg), at),
            None => {
                let Some(child) = self.child.as_ref() else {
                    return Err(BackendError::Failed(format!(
                        "worker {name}: worker exited"
                    )));
                };
                (receive_reply(child, self.cfg.deadline()), Instant::now())
            }
        };
        self.record_call(name, t0, t1);
        match res {
            Err(e) => {
                self.kill();
                Err(BackendError::Failed(format!("worker {name}: {e}")))
            }
            Ok((reply, payload)) => {
                self.state = reply.state.clone();
                match reply.err.clone() {
                    None => Ok((reply, payload)),
                    Some(WireError::KvFull { seq, needed, free }) => {
                        Err(BackendError::KvFull { seq, needed, free })
                    }
                    Some(WireError::Failed { msg, sticky }) => {
                        if sticky {
                            self.kill();
                        }
                        Err(BackendError::Failed(msg))
                    }
                }
            }
        }
    }

    fn protocol_error(&mut self, what: &str) -> BackendError {
        self.kill();
        BackendError::Failed(format!("worker sent a bad reply: {what}"))
    }

    fn check_lost(&self, seq: SeqId) -> Result<(), BackendError> {
        if self.lost.contains(&seq) {
            Err(BackendError::Failed(format!(
                "sequence {seq} was lost when the worker restarted"
            )))
        } else {
            Ok(())
        }
    }
}

impl Drop for RemoteBackend {
    fn drop(&mut self) {
        if let Some(c) = self.child.take() {
            c.kill();
        }
    }
}

impl BatchBackend for RemoteBackend {
    fn n_vocab(&self) -> usize {
        self.info.n_vocab
    }

    fn n_seq_max(&self) -> usize {
        self.info.n_seq_max
    }

    fn n_ctx_seq(&self) -> usize {
        self.info.n_ctx_seq
    }

    fn kv_capacity(&self) -> usize {
        self.info.kv_capacity
    }

    fn kv_free(&self) -> usize {
        self.state.kv_free
    }

    fn seq_len(&self, seq: SeqId) -> usize {
        self.state.seqs.get(seq as usize).map_or(0, |s| s.len)
    }

    fn seq_reserved(&self, seq: SeqId) -> usize {
        self.state.seqs.get(seq as usize).map_or(0, |s| s.reserved)
    }

    fn seq_lost(&self, seq: SeqId) -> bool {
        self.lost.contains(&seq)
    }

    fn seq_reserve(&mut self, seq: SeqId, n: usize) -> Result<(), BackendError> {
        self.call(&Request::SeqReserve { seq, n }, &[]).map(|_| ())
    }

    fn seq_rm(&mut self, seq: SeqId, keep: usize) -> Result<(), BackendError> {
        if self.child.is_none() {
            // Dead worker: nothing is held, and freeing must not start it again.
            if keep == 0 {
                self.lost.remove(&seq);
            }
            return Ok(());
        }
        self.call(&Request::SeqRm { seq, keep }, &[])?;
        if keep == 0 {
            self.lost.remove(&seq);
        }
        Ok(())
    }

    fn seq_cp(&mut self, src: SeqId, dst: SeqId, n: usize) -> Result<(), BackendError> {
        self.check_lost(src)?;
        self.call(&Request::SeqCp { src, dst, n }, &[])?;
        self.lost.remove(&dst);
        Ok(())
    }

    fn decode(&mut self, inputs: &[SeqInput<'_>]) -> Result<Outputs<'_>, BackendError> {
        self.decode_feed(inputs, None)
    }

    fn kv_format(&self) -> String {
        self.info.kv_format.clone()
    }

    fn on_gpu(&self) -> bool {
        self.info.on_gpu
    }

    fn bytes_streamed_total(&self) -> Option<u64> {
        self.state.bytes_streamed
    }

    fn kv_row_bytes(&self) -> usize {
        self.info.kv_row_bytes
    }

    fn max_step_tokens(&self) -> usize {
        usize::try_from(self.info.max_step_tokens).unwrap_or(usize::MAX)
    }

    fn seq_snapshot(&mut self, seq: SeqId) -> Result<Vec<u8>, BackendError> {
        self.check_lost(seq)?;
        self.call(&Request::SeqSnapshot { seq }, &[])
            .map(|(_, payload)| payload)
    }

    fn seq_restore(&mut self, seq: SeqId, bytes: &[u8], n: usize) -> Result<(), BackendError> {
        self.call(&Request::SeqRestore { seq, n }, bytes)?;
        self.lost.remove(&seq);
        Ok(())
    }
}

impl RemoteBackend {
    /// Floats per reply row: `n_vocab`, or `n_embd` from a segment without the head.
    fn out_width(&self) -> usize {
        match self.info.segment {
            Some(s) if !s.head => s.n_embd,
            _ => self.info.n_vocab,
        }
    }

    /// One `decode`; `hidden` is the input of a segment without the embedding.
    fn decode_feed(
        &mut self,
        inputs: &[SeqInput<'_>],
        hidden: Option<&[f32]>,
    ) -> Result<Outputs<'_>, BackendError> {
        self.start_feed(inputs, hidden)?;
        self.finish_feed()
    }

    /// Sends a `decode` and returns; `finish_feed` takes the reply.
    fn start_feed(
        &mut self,
        inputs: &[SeqInput<'_>],
        hidden: Option<&[f32]>,
    ) -> Result<(), BackendError> {
        for i in inputs {
            self.check_lost(i.seq)?;
        }
        let takes_hidden = self.info.segment.is_some_and(|s| !s.embed);
        match (takes_hidden, hidden) {
            (true, None) => {
                return Err(BackendError::Failed(
                    "this worker's segment takes hidden rows, not tokens".into(),
                ));
            }
            (false, Some(_)) => {
                return Err(BackendError::Failed(
                    "this worker's segment takes tokens, not hidden rows".into(),
                ));
            }
            _ => {}
        }
        let payload: Vec<u8> = hidden
            .into_iter()
            .flatten()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let req = Request::Decode {
            inputs: inputs
                .iter()
                .map(|i| wire::WireInput {
                    seq: i.seq,
                    tokens: i.tokens.to_vec(),
                    all_logits: i.all_logits,
                })
                .collect(),
            trace: current_traceparent(),
        };
        let t0 = self.send_call(&req, &payload)?;
        self.pending.push_back((t0, inputs.len()));
        Ok(())
    }

    /// The reply to the oldest `start_feed`.
    fn finish_feed(&mut self) -> Result<Outputs<'_>, BackendError> {
        let Some((t0, n_inputs)) = self.pending.pop_front() else {
            return Err(BackendError::Failed("no step was started".into()));
        };
        let (reply, payload) = self.recv_call("decode", t0)?;
        self.last_step = reply.step_s.map(Duration::from_secs_f64);
        let nv = self.out_width();
        if reply.spans.len() != n_inputs || nv == 0 {
            return Err(self.protocol_error("span count"));
        }
        let rows = payload.len() / 4 / nv;
        let ok =
            payload.len() == rows * nv * 4 && reply.spans.iter().all(|&(a, b)| a <= b && b <= rows);
        if !ok {
            return Err(self.protocol_error("logits do not match their spans"));
        }
        self.logits.clear();
        self.logits.extend(
            payload
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        );
        self.spans.clear();
        self.spans.extend(reply.spans.iter().map(|&(a, b)| a..b));
        Ok(Outputs::new(nv, &self.logits, &self.spans))
    }

    /// The worker's own time for the last `decode`, apart from the network.
    pub fn last_step(&self) -> Option<Duration> {
        self.last_step
    }
}

impl SegmentBackend for RemoteBackend {
    fn segment(&self) -> SegmentInfo {
        match self.info.segment {
            Some(s) => SegmentInfo {
                embed: s.embed,
                head: s.head,
                n_embd: s.n_embd,
                layers: s.layers,
            },
            None => SegmentInfo {
                embed: true,
                head: true,
                n_embd: 0,
                layers: (0, 0),
            },
        }
    }

    fn step(
        &mut self,
        inputs: &[SeqInput<'_>],
        feed: Feed<'_>,
    ) -> Result<StepOut<'_>, BackendError> {
        let hidden = match feed {
            Feed::Tokens => None,
            Feed::Hidden(h) => Some(h),
        };
        let head = self.info.segment.is_none_or(|s| s.head);
        let out = self.decode_feed(inputs, hidden)?;
        Ok(if head {
            StepOut::Logits(out)
        } else {
            StepOut::Hidden(out)
        })
    }

    fn last_step_time(&self) -> Option<Duration> {
        self.last_step
    }

    fn expert_stats(&mut self) -> Option<serde_json::Value> {
        self.call(&Request::Stats, &[]).ok()?.0.stats
    }

    fn can_overlap(&self) -> bool {
        true
    }

    fn start_step(&mut self, inputs: &[SeqInput<'_>], feed: Feed<'_>) -> Result<(), BackendError> {
        let hidden = match feed {
            Feed::Tokens => None,
            Feed::Hidden(h) => Some(h),
        };
        self.start_feed(inputs, hidden)
    }

    fn reply_ready(&mut self, wait: Duration) -> bool {
        if self.stash.is_some() {
            return true;
        }
        let Some(&(t0, _)) = self.pending.front() else {
            return true;
        };
        let Some(child) = self.child.as_ref() else {
            return true;
        };
        let deadline = self.cfg.deadline();
        // A reply that does not come is the finish's error, after the same deadline as a call.
        if t0.elapsed() >= deadline {
            self.stash = Some((Instant::now(), Err(format!("no reply within {deadline:?}"))));
            return true;
        }
        match child.rx.recv_timeout(wait) {
            Ok(msg) => {
                self.stash = Some((Instant::now(), msg));
                true
            }
            Err(RecvTimeoutError::Timeout) => false,
            Err(RecvTimeoutError::Disconnected) => {
                self.stash = Some((Instant::now(), Err("worker exited".into())));
                true
            }
        }
    }

    fn finish_step(&mut self) -> Result<StepOut<'_>, BackendError> {
        let head = self.info.segment.is_none_or(|s| s.head);
        let out = self.finish_feed()?;
        Ok(if head {
            StepOut::Logits(out)
        } else {
            StepOut::Hidden(out)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // hardware: none
    #[test]
    fn deadline_is_ten_steps_but_at_least_the_floor() {
        let mut c = RemoteConfig::new("x", Vec::new());
        assert_eq!(c.deadline(), Duration::from_secs(5));
        c.predicted_step = Some(Duration::from_millis(100));
        assert_eq!(c.deadline(), Duration::from_secs(5));
        c.predicted_step = Some(Duration::from_secs(2));
        assert_eq!(c.deadline(), Duration::from_secs(20));
    }
}
