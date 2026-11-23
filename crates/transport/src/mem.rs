//! In-process transport for simulated nodes (ADR 0091). Nodes share a [`Switchboard`]; each link
//! (an ordered pair of nodes) follows a [`LinkFault`] that can be changed at any time.
//!
//! Semantics, chosen to match TCP where a test can see the difference:
//! - Dropping or closing a half is a FIN for the peer: it reads the bytes already sent, then EOF,
//!   and its writes fail with `BrokenPipe`. The closing side's own calls fail with
//!   `ConnectionAborted`.
//! - [`LinkFault::Close`] is a crash: every open connection between the two nodes fails on both
//!   sides with `ConnectionReset` at once, and new connections from `from` to `to` are refused
//!   until the link is set to another fault.
//! - [`LinkFault::Blackhole`] drops what the node on `from` writes toward `to` and closes nothing:
//!   the reader waits (a silent node, or a partition when set in both directions). Bytes dropped
//!   are gone when the fault is lifted.
//! - [`LinkFault::Slow`] delays each write by `alpha + len / bytes_per_s` after the previous one
//!   (a link that serializes its bytes).
//! - Like the TLS transport, a second connection of the same `(from, to, purpose)` while the
//!   first is open is refused (`AlreadyExists`, counted as `duplicate`).

use crate::metrics::{self, CountingWriter, LinkMetrics};
use crate::{Conn, NodeId, Peer, Purpose, Transport};
use std::collections::{HashMap, VecDeque};
use std::io::{self, Read, Write};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LinkFault {
    None,
    /// Close every connection (a crashed process: the peer sees a reset).
    Close,
    /// Pass no bytes, close nothing (a silent node or a partition, one direction).
    Blackhole,
    /// Delay and bandwidth limit per frame.
    Slow {
        alpha: Duration,
        bytes_per_s: f64,
    },
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[derive(Default)]
struct PipeState {
    /// Bytes with the time they may be read.
    q: VecDeque<(Instant, Vec<u8>)>,
    /// Bytes of the front chunk already read.
    off: usize,
    /// The writer is done (FIN).
    eof: bool,
    /// The reader is gone.
    rclosed: bool,
    reset: bool,
    /// Delivery time of the last chunk, so a slow link keeps its order.
    last_at: Option<Instant>,
}

#[derive(Default)]
struct Pipe {
    st: Mutex<PipeState>,
    cv: Condvar,
}

impl Pipe {
    fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        let mut st = lock(&self.st);
        loop {
            if st.reset {
                return Err(io::ErrorKind::ConnectionReset.into());
            }
            if st.rclosed {
                return Err(io::ErrorKind::ConnectionAborted.into());
            }
            let now = Instant::now();
            match st.q.front().map(|(t, _)| *t) {
                Some(t) if t <= now => {
                    let off = st.off;
                    let (n, done) = {
                        let chunk = &st.q.front().expect("front").1;
                        let n = buf.len().min(chunk.len() - off);
                        buf[..n].copy_from_slice(&chunk[off..off + n]);
                        (n, off + n == chunk.len())
                    };
                    if done {
                        st.q.pop_front();
                        st.off = 0;
                    } else {
                        st.off = off + n;
                    }
                    return Ok(n);
                }
                Some(t) => {
                    st = self
                        .cv
                        .wait_timeout(st, t - now)
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .0;
                }
                None if st.eof => return Ok(0),
                None => {
                    st = self
                        .cv
                        .wait(st)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                }
            }
        }
    }

    fn write(&self, buf: &[u8], fault: LinkFault) -> io::Result<usize> {
        let mut st = lock(&self.st);
        if st.reset {
            return Err(io::ErrorKind::ConnectionReset.into());
        }
        if st.eof {
            return Err(io::ErrorKind::ConnectionAborted.into());
        }
        if st.rclosed {
            return Err(io::ErrorKind::BrokenPipe.into());
        }
        if buf.is_empty() {
            return Ok(0);
        }
        let now = Instant::now();
        let at = match fault {
            LinkFault::None => now,
            LinkFault::Blackhole => return Ok(buf.len()),
            LinkFault::Close => {
                st.reset = true;
                self.cv.notify_all();
                return Err(io::ErrorKind::ConnectionReset.into());
            }
            LinkFault::Slow { alpha, bytes_per_s } => {
                let serial = if bytes_per_s > 0.0 {
                    Duration::from_secs_f64(buf.len() as f64 / bytes_per_s)
                } else {
                    Duration::ZERO
                };
                st.last_at.map_or(now, |l| l.max(now)) + alpha + serial
            }
        };
        st.last_at = Some(at);
        st.q.push_back((at, buf.to_vec()));
        self.cv.notify_all();
        Ok(buf.len())
    }

    fn set(&self, f: impl FnOnce(&mut PipeState)) {
        f(&mut lock(&self.st));
        self.cv.notify_all();
    }

    /// Both ends of the connection are unusable or the writer is done.
    fn is_dead(&self) -> bool {
        let st = lock(&self.st);
        st.reset || st.eof || st.rclosed
    }
}

struct Rx(Arc<Pipe>);
impl Read for Rx {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read(buf)
    }
}
impl Drop for Rx {
    fn drop(&mut self) {
        self.0.set(|s| s.rclosed = true);
    }
}

struct Tx {
    pipe: Arc<Pipe>,
    board: Arc<Switchboard>,
    from: NodeId,
    to: NodeId,
}
impl Write for Tx {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let fault = self.board.fault_of(&self.from, &self.to);
        self.pipe.write(buf, fault)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
impl Drop for Tx {
    fn drop(&mut self) {
        self.pipe.set(|s| s.eof = true);
    }
}

struct Inbox {
    q: Mutex<(VecDeque<Conn>, bool)>,
    cv: Condvar,
}

struct OpenConn {
    a: NodeId,
    b: NodeId,
    pipes: [Arc<Pipe>; 2],
}

#[derive(Default)]
struct Board {
    inboxes: HashMap<NodeId, Arc<Inbox>>,
    faults: HashMap<(NodeId, NodeId), LinkFault>,
    open: Vec<OpenConn>,
    /// The pipe of each `(from, to, purpose)` connection, to refuse a second one while it lives.
    live: HashMap<(NodeId, NodeId, Purpose), Arc<Pipe>>,
}

/// The shared network of simulated nodes: who exists and how each link behaves.
pub struct Switchboard {
    st: Mutex<Board>,
}

impl Switchboard {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            st: Mutex::new(Board::default()),
        })
    }

    /// The transport of node `id`; calling it again for the same id shares the node's inbox.
    pub fn node(self: &Arc<Self>, id: &NodeId) -> MemTransport {
        let inbox = Arc::clone(lock(&self.st).inboxes.entry(id.clone()).or_insert_with(|| {
            Arc::new(Inbox {
                q: Mutex::new((VecDeque::new(), false)),
                cv: Condvar::new(),
            })
        }));
        MemTransport {
            id: id.clone(),
            board: Arc::clone(self),
            inbox,
        }
    }

    /// Sets how the link `from` to `to` behaves from now on.
    pub fn fault(&self, from: &NodeId, to: &NodeId, f: LinkFault) {
        let mut b = lock(&self.st);
        if f == LinkFault::None {
            b.faults.remove(&(from.clone(), to.clone()));
            return;
        }
        b.faults.insert((from.clone(), to.clone()), f);
        if f == LinkFault::Close {
            b.open.retain(|c| {
                let between = (&c.a == from && &c.b == to) || (&c.a == to && &c.b == from);
                if between {
                    for p in &c.pipes {
                        p.set(|s| s.reset = true);
                    }
                }
                !between
            });
        }
    }

    fn fault_of(&self, from: &NodeId, to: &NodeId) -> LinkFault {
        lock(&self.st)
            .faults
            .get(&(from.clone(), to.clone()))
            .copied()
            .unwrap_or(LinkFault::None)
    }
}

/// One simulated node's side of a [`Switchboard`].
pub struct MemTransport {
    id: NodeId,
    board: Arc<Switchboard>,
    inbox: Arc<Inbox>,
}

impl MemTransport {
    /// A new network with no nodes.
    pub fn switchboard() -> Arc<Switchboard> {
        Switchboard::new()
    }

    fn half(
        &self,
        purpose: &Purpose,
        rx: Arc<Pipe>,
        tx: Arc<Pipe>,
        local: &NodeId,
        remote: &NodeId,
        close: Arc<dyn Fn() + Send + Sync>,
    ) -> Conn {
        let m = LinkMetrics::new(local, remote, "mem", purpose.label());
        Conn {
            peer: remote.clone(),
            purpose: purpose.clone(),
            rx: Box::new(Rx(rx)),
            tx: Box::new(CountingWriter::new(
                Tx {
                    pipe: tx,
                    board: Arc::clone(&self.board),
                    from: local.clone(),
                    to: remote.clone(),
                },
                m,
            )),
            close,
        }
    }
}

impl Transport for MemTransport {
    fn connect(&self, peer: &Peer, purpose: Purpose) -> io::Result<Conn> {
        let inbox = {
            let b = lock(&self.board.st);
            if matches!(
                b.faults.get(&(self.id.clone(), peer.id.clone())),
                Some(LinkFault::Close)
            ) {
                return Err(io::ErrorKind::ConnectionRefused.into());
            }
            Arc::clone(
                b.inboxes
                    .get(&peer.id)
                    .ok_or_else(|| io::Error::from(io::ErrorKind::ConnectionRefused))?,
            )
        };
        let (ab, ba) = (Arc::new(Pipe::default()), Arc::new(Pipe::default()));
        {
            let mut b = lock(&self.board.st);
            let key = (self.id.clone(), peer.id.clone(), purpose.clone());
            if b.live.get(&key).is_some_and(|p| !p.is_dead()) {
                metrics::rejected("duplicate");
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    format!(
                        "{} already has a {} connection to {}",
                        self.id,
                        purpose.label(),
                        peer.id
                    ),
                ));
            }
            b.live.insert(key, Arc::clone(&ab));
            b.open.retain(|c| !c.pipes.iter().all(|p| p.is_dead()));
            b.open.push(OpenConn {
                a: self.id.clone(),
                b: peer.id.clone(),
                pipes: [Arc::clone(&ab), Arc::clone(&ba)],
            });
        }
        // `close`: this side's reads fail, the peer reads EOF and its writes fail.
        let closer = |rx: &Arc<Pipe>, tx: &Arc<Pipe>| -> Arc<dyn Fn() + Send + Sync> {
            let (rx, tx) = (Arc::clone(rx), Arc::clone(tx));
            Arc::new(move || {
                tx.set(|s| s.eof = true);
                rx.set(|s| s.rclosed = true);
            })
        };
        let theirs = self.half(
            &purpose,
            Arc::clone(&ab),
            Arc::clone(&ba),
            &peer.id,
            &self.id,
            closer(&ab, &ba),
        );
        let mine = self.half(
            &purpose,
            Arc::clone(&ba),
            Arc::clone(&ab),
            &self.id,
            &peer.id,
            closer(&ba, &ab),
        );
        let mut q = lock(&inbox.q);
        if q.1 {
            return Err(io::ErrorKind::ConnectionRefused.into());
        }
        q.0.push_back(theirs);
        inbox.cv.notify_all();
        Ok(mine)
    }

    fn accept(&self) -> io::Result<Conn> {
        let mut q = lock(&self.inbox.q);
        loop {
            if q.1 {
                return Err(io::Error::new(
                    io::ErrorKind::NotConnected,
                    "transport shut down",
                ));
            }
            if let Some(c) = q.0.pop_front() {
                return Ok(c);
            }
            q = self
                .inbox
                .cv
                .wait(q)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    fn name(&self) -> &'static str {
        "mem"
    }

    fn local(&self) -> &NodeId {
        &self.id
    }

    fn shutdown(&self) {
        lock(&self.inbox.q).1 = true;
        self.inbox.cv.notify_all();
    }
}
