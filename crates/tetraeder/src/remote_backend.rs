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
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::collections::{HashSet, VecDeque};
use std::ffi::OsString;
use std::io::{BufReader, Read, Write};
use std::ops::Range;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::time::{Duration, Instant};
use tetraeder_scheduler::{BackendError, BatchBackend, Outputs, SeqId, SeqInput};

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
        },
        SeqSnapshot {
            seq: u32,
        },
        /// The snapshot is the frame payload.
        SeqRestore {
            seq: u32,
            n: usize,
        },
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
    }
}

use wire::{Reply, Request, WireError};

/// Largest frame accepted; a length beyond it is a protocol error, not an allocation.
pub const MAX_FRAME: u64 = 1 << 30;

#[derive(Debug)]
pub struct Frame {
    pub header: Vec<u8>,
    pub payload: Vec<u8>,
}

/// A frame with an already serialized header.
pub fn encode_raw(header: &[u8], payload: &[u8]) -> Result<Vec<u8>, String> {
    let total = 4 + header.len() + payload.len();
    let total32 = u32::try_from(total).map_err(|_| "frame too large".to_owned())?;
    let mut out = Vec::with_capacity(4 + total);
    out.extend_from_slice(&total32.to_le_bytes());
    out.extend_from_slice(&(header.len() as u32).to_le_bytes());
    out.extend_from_slice(header);
    out.extend_from_slice(payload);
    Ok(out)
}

pub fn encode<T: Serialize>(header: &T, payload: &[u8]) -> Result<Vec<u8>, String> {
    let h = serde_json::to_vec(header).map_err(|e| format!("cannot encode a header: {e}"))?;
    encode_raw(&h, payload)
}

pub fn write_frame<W: Write, T: Serialize>(
    w: &mut W,
    header: &T,
    payload: &[u8],
) -> Result<(), String> {
    let bytes = encode(header, payload)?;
    w.write_all(&bytes)
        .and_then(|()| w.flush())
        .map_err(|e| format!("write failed: {e}"))
}

/// The next frame; `Ok(None)` at a clean end of input (EOF before the first byte).
pub fn read_frame<R: Read>(r: &mut R) -> Result<Option<Frame>, String> {
    let mut len = [0u8; 4];
    let mut got = 0;
    while got < len.len() {
        match r.read(&mut len[got..]) {
            Ok(0) if got == 0 => return Ok(None),
            Ok(0) => return Err("truncated frame".into()),
            Ok(n) => got += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(format!("read failed: {e}")),
        }
    }
    let total = u64::from(u32::from_le_bytes(len));
    if !(4..=MAX_FRAME).contains(&total) {
        return Err(format!("bad frame length {total}"));
    }
    // `take` grows the buffer as bytes arrive, so a corrupt length cannot allocate ahead.
    let mut body = Vec::new();
    r.by_ref()
        .take(total)
        .read_to_end(&mut body)
        .map_err(|e| format!("read failed: {e}"))?;
    if body.len() as u64 != total {
        return Err("truncated frame".into());
    }
    let hlen = u32::from_le_bytes([body[0], body[1], body[2], body[3]]) as usize;
    if 4 + hlen > body.len() {
        return Err("bad header length".into());
    }
    let payload = body.split_off(4 + hlen);
    let header = body.split_off(4);
    Ok(Some(Frame { header, payload }))
}

pub fn parse_header<T: DeserializeOwned>(f: &Frame) -> Result<T, String> {
    serde_json::from_slice(&f.header).map_err(|e| format!("bad header: {e}"))
}

/// How to start and supervise the worker.
pub struct RemoteConfig {
    pub program: PathBuf,
    pub args: Vec<OsString>,
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
        Self {
            program: program.into(),
            args,
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

struct Child {
    proc: std::process::Child,
    tx: Sender<Vec<u8>>,
    rx: Receiver<Result<Frame, String>>,
}

impl Child {
    fn kill(mut self) {
        let _ = self.proc.kill();
        let _ = self.proc.wait();
    }
}

/// Starts the worker with a thread for each pipe, so a worker that stops reading or writing
/// never blocks the caller past its deadline.
fn spawn_child(cfg: &RemoteConfig) -> Result<Child, String> {
    let mut proc = Command::new(&cfg.program)
        .args(&cfg.args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("cannot start {}: {e}", cfg.program.display()))?;
    let mut stdin = proc.stdin.take().ok_or("worker has no stdin")?;
    let stdout = proc.stdout.take().ok_or("worker has no stdout")?;
    let (tx, wrx) = mpsc::channel::<Vec<u8>>();
    std::thread::spawn(move || {
        while let Ok(bytes) = wrx.recv() {
            if stdin
                .write_all(&bytes)
                .and_then(|()| stdin.flush())
                .is_err()
            {
                break;
            }
        }
    });
    let (rtx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut out = BufReader::new(stdout);
        loop {
            match read_frame(&mut out) {
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
    Ok(Child { proc, tx, rx })
}

fn exchange(
    child: &Child,
    req: &Request,
    payload: &[u8],
    deadline: Duration,
) -> Result<(Reply, Vec<u8>), String> {
    let bytes = encode(req, payload)?;
    child
        .tx
        .send(bytes)
        .map_err(|_| "broken pipe to the worker".to_owned())?;
    match child.rx.recv_timeout(deadline) {
        Ok(Ok(frame)) => Ok((parse_header(&frame)?, frame.payload)),
        Ok(Err(e)) => Err(e),
        Err(RecvTimeoutError::Timeout) => Err(format!("no reply within {deadline:?}")),
        Err(RecvTimeoutError::Disconnected) => Err("worker exited".into()),
    }
}

/// Starts a worker and reads its `Info`.
fn handshake(cfg: &RemoteConfig) -> Result<(Child, wire::Info, wire::State), String> {
    let child = spawn_child(cfg)?;
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
}

impl RemoteBackend {
    /// Starts the worker and waits until it has loaded. An error is a startup error.
    pub fn start(cfg: RemoteConfig) -> Result<Self, String> {
        let (child, info, state) = handshake(&cfg)?;
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
        })
    }

    /// Worker restarts so far.
    pub fn restarts(&self) -> usize {
        self.n_restarts
    }

    /// Kills the worker; its sequences are gone.
    fn kill(&mut self) {
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
        self.ensure_running().map_err(BackendError::Failed)?;
        let child = self.child.as_ref().expect("running");
        let t0 = Instant::now();
        let res = exchange(child, req, payload, self.cfg.deadline());
        self.ipc.record(
            t0.elapsed().as_secs_f64(),
            &[KeyValue::new("tetraeder.call", req.name())],
        );
        match res {
            Err(e) => {
                self.kill();
                Err(BackendError::Failed(format!("worker {}: {e}", req.name())))
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
        for i in inputs {
            self.check_lost(i.seq)?;
        }
        let req = Request::Decode {
            inputs: inputs
                .iter()
                .map(|i| wire::WireInput {
                    seq: i.seq,
                    tokens: i.tokens.to_vec(),
                    all_logits: i.all_logits,
                })
                .collect(),
        };
        let (reply, payload) = self.call(&req, &[])?;
        let nv = self.info.n_vocab;
        if reply.spans.len() != inputs.len() || nv == 0 {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    // hardware: none
    #[test]
    fn frames_round_trip() {
        let mut buf = Vec::new();
        write_frame(&mut buf, &Request::SeqRm { seq: 3, keep: 7 }, &[1, 2, 3]).unwrap();
        write_frame(&mut buf, &Request::Info, &[]).unwrap();
        let mut r = buf.as_slice();
        let f = read_frame(&mut r).unwrap().unwrap();
        assert_eq!(f.payload, [1, 2, 3]);
        assert!(matches!(
            parse_header::<Request>(&f).unwrap(),
            Request::SeqRm { seq: 3, keep: 7 }
        ));
        let f = read_frame(&mut r).unwrap().unwrap();
        assert!(matches!(
            parse_header::<Request>(&f).unwrap(),
            Request::Info
        ));
        assert!(read_frame(&mut r).unwrap().is_none());
    }

    // hardware: none
    #[test]
    fn bad_frames_are_errors_not_allocations() {
        // Longer than the cap.
        let mut r: &[u8] = &u32::MAX.to_le_bytes();
        assert!(read_frame(&mut r).unwrap_err().contains("length"));
        // Shorter than its own header length field.
        let mut r: &[u8] = &[2, 0, 0, 0, 1, 0];
        assert!(read_frame(&mut r).unwrap_err().contains("length"));
        // Truncated body and truncated length.
        let mut r: &[u8] = &[100, 0, 0, 0, 1, 2];
        assert!(read_frame(&mut r).unwrap_err().contains("truncated"));
        let mut r: &[u8] = &[100, 0];
        assert!(read_frame(&mut r).unwrap_err().contains("truncated"));
        // Header length beyond the frame.
        let mut r: &[u8] = &[4, 0, 0, 0, 9, 0, 0, 0];
        assert!(read_frame(&mut r).unwrap_err().contains("header"));
    }

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
