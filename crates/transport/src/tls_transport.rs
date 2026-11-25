//! `TlsTransport`: TCP + TLS 1.3 (rustls, ring) with mutual authentication (ADR 0091 step 1).
//!
//! Each connection runs one handshake: TCP (`TCP_NODELAY`, socket buffers), TLS 1.3 with both
//! certificates checked against the trust anchors and the configured node ids
//! ([`crate::identity`]), then a one-frame preface from the dialer naming the [`Purpose`] and a
//! one-frame answer from the acceptor (`ok`, or the reason it refused). TLS 1.3 reports a
//! refused client certificate to the dialer only with its first read, so the answer is what
//! makes `connect` fail for a refused peer.
//!
//! The transport owns a small tokio runtime; the halves of a [`Conn`] are blocking `Read` and
//! `Write` objects that drive it, so they work from plain threads (as `RemoteBackend` drives a
//! child's pipes). Do not call them from inside another tokio runtime's async code.
//!
//! Socket buffers: `SocketBuffers::Auto` leaves the kernel's autotuning alone;
//! `SocketBuffers::Explicit` sets `SO_SNDBUF` and `SO_RCVBUF` before connect and listen (Linux
//! caps them at `net.core.{w,r}mem_max` and turns autotuning off for that socket). Either way
//! the effective sizes are read back with `getsockopt`, logged and kept (`link_buffers`).

use crate::identity::{Anchors, Identity, PeerVerifier, node_id_of, reject_reason};
use crate::metrics::{self, LinkMetrics};
use crate::{Conn, NodeId, Peer, Purpose, Transport, frame};
use anyhow::{Context, Result, anyhow};
use rustls::pki_types::{CertificateDer, ServerName};
use rustls::{ClientConfig, ServerConfig};
use serde::{Deserialize, Serialize};
use socket2::SockRef;
use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::net::{IpAddr, Shutdown, SocketAddr, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, ReadHalf, WriteHalf};
use tokio::net::{TcpListener, TcpSocket, TcpStream};
use tokio::runtime::{Builder, Handle, Runtime};
use tokio::sync::watch;
use tokio_rustls::client::TlsStream as ClientTls;
use tokio_rustls::server::TlsStream as ServerTls;
use tokio_rustls::{TlsAcceptor, TlsConnector};

/// ALPN of the node protocol; a different value refuses the handshake.
const ALPN: &[u8] = b"tetraeder/1";
/// Dial, handshake and preface together.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
/// An inbound connection that has not finished its preface by then is dropped.
const ACCEPT_TIMEOUT: Duration = Duration::from_secs(10);
/// Until TLS completes (the peer is authenticated then); the longer timeout above only applies
/// after that, so a socket that stays silent holds a handshake slot for a few seconds.
const TLS_TIMEOUT: Duration = Duration::from_secs(3);
/// Name sent as SNI; identity is the certificate's node id, not a host name.
const SNI: &str = "tetraeder-node";
const MAX_PREFACE: usize = 64 * 1024;

/// Kernel autotuning, or explicit sizes (Linux caps them at `net.core.{r,w}mem_max` and
/// turns autotuning off). The probe picks one per link; the effective sizes are read back
/// with `getsockopt` and logged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SocketBuffers {
    Auto,
    Explicit { send: usize, recv: usize },
}

/// `(SO_SNDBUF, SO_RCVBUF)` as the kernel reports them (Linux reports twice the requested size).
pub fn effective_buffers(s: &std::net::TcpStream) -> io::Result<(usize, usize)> {
    let r = SockRef::from(s);
    Ok((r.send_buffer_size()?, r.recv_buffer_size()?))
}

/// What a connection ended up with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkBuffers {
    pub peer: NodeId,
    pub purpose: Purpose,
    pub send: usize,
    pub recv: usize,
    /// The setting asked for.
    pub requested: SocketBuffers,
}

#[derive(Serialize, Deserialize)]
struct Hello {
    purpose: Purpose,
}

#[derive(Serialize, Deserialize)]
struct Answer {
    ok: bool,
    #[serde(default)]
    reason: String,
}

struct Rt(Option<Runtime>);

impl Drop for Rt {
    fn drop(&mut self) {
        if let Some(rt) = self.0.take() {
            rt.shutdown_background();
        }
    }
}

/// One live inbound connection: its `dead` flag and a handle to shut its socket down.
struct LiveEntry {
    dead: Arc<AtomicBool>,
    sock: std::net::TcpStream,
}

type Live = Mutex<HashMap<(NodeId, Purpose), LiveEntry>>;

/// In-flight handshakes per source address.
#[derive(Default)]
struct IpLimiter {
    counts: Mutex<HashMap<IpAddr, usize>>,
}

/// One in-flight handshake of a source address; gives its slot back when dropped.
struct IpPermit {
    limiter: Arc<IpLimiter>,
    ip: IpAddr,
}

impl IpLimiter {
    fn try_acquire(self: &Arc<Self>, ip: IpAddr, max: usize) -> Option<IpPermit> {
        let mut c = lock(&self.counts);
        let n = c.entry(ip).or_insert(0);
        if *n >= max {
            return None;
        }
        *n += 1;
        Some(IpPermit {
            limiter: Arc::clone(self),
            ip,
        })
    }
}

impl Drop for IpPermit {
    fn drop(&mut self) {
        let mut c = lock(&self.limiter.counts);
        if let Some(n) = c.get_mut(&self.ip) {
            *n -= 1;
            if *n == 0 {
                c.remove(&self.ip);
            }
        }
    }
}

/// Registration of an inbound connection in the live map; clears it when dropped, so a handshake
/// cancelled by the timeout or a connection that ended never leaves a stale entry.
struct LiveGuard {
    live: Arc<Live>,
    key: (NodeId, Purpose),
    dead: Arc<AtomicBool>,
}

impl Drop for LiveGuard {
    fn drop(&mut self) {
        self.dead.store(true, Ordering::SeqCst);
        let mut live = lock(&self.live);
        if live
            .get(&self.key)
            .is_some_and(|e| Arc::ptr_eq(&e.dead, &self.dead))
        {
            live.remove(&self.key);
        }
    }
}

/// Handshakes in flight before authentication; further connections are dropped at once.
const MAX_HANDSHAKES: usize = 64;
/// Of those, per source address (Vast boxes cannot firewall ports to peer addresses, so any host
/// can open sockets): one source cannot take the slots of the others. The slot is given back
/// when TLS completes.
const MAX_HANDSHAKES_PER_IP: usize = 8;
/// Registered connections of one peer (replica, segment and probe numbers are the peer's choice):
/// a peer cannot fill the map for the others.
const MAX_LIVE_PER_PEER: usize = 64;
/// Registered inbound connections (peers x purposes); a replica or segment number is chosen by
/// the peer, so the map is bounded.
const MAX_LIVE: usize = 4096;
/// Accepted connections waiting for `accept`.
const INBOX: usize = 256;
/// TCP keepalive: a peer that crashed or whose NAT entry vanished is noticed in about 25 s,
/// so its entry does not linger. Membership notices faster (heartbeats).
const KEEPALIVE_IDLE: Duration = Duration::from_secs(10);
const KEEPALIVE_INTERVAL: Duration = Duration::from_secs(5);
/// Longest `reason` of a refusal that is echoed into an error.
const MAX_REASON: usize = 120;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

struct Inner {
    me: NodeId,
    client: TlsConnector,
    acceptor: TlsAcceptor,
    nodes_anchors: Arc<Anchors>,
    buffers: SocketBuffers,
    live: Arc<Live>,
    log: Mutex<Vec<LinkBuffers>>,
    stop: watch::Sender<bool>,
}

/// An accepted connection before the caller's thread wraps it.
struct Raw {
    tls: ServerTls<TcpStream>,
    sock: std::net::TcpStream,
    peer: NodeId,
    purpose: Purpose,
    guard: LiveGuard,
}

/// TCP + TLS 1.3 (rustls, ring), mutual authentication.
pub struct TlsTransport {
    inner: Arc<Inner>,
    rt: Arc<Rt>,
    handle: Handle,
    local_addr: Option<SocketAddr>,
    inbox: Mutex<mpsc::Receiver<Raw>>,
    listening: bool,
}

impl TlsTransport {
    /// `identity` names this node; `anchors` are the trust anchors (pinned self-signed
    /// certificates or a CA certificate) and `nodes` the ids a peer may have. `listen` is the
    /// address to bind (`None`: a dial-only node).
    pub fn new(
        identity: &Identity,
        anchors: &[CertificateDer<'static>],
        nodes: &[NodeId],
        listen: Option<&str>,
        buffers: SocketBuffers,
    ) -> Result<Self> {
        let me = identity.node_id()?;
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let trust = Arc::new(Anchors::new(
            anchors,
            nodes,
            provider.signature_verification_algorithms,
        )?);
        let verifier = PeerVerifier::new(Arc::clone(&trust), &provider);
        let chain = vec![identity.cert.clone()];
        let mut server = ServerConfig::builder_with_provider(Arc::clone(&provider))
            .with_protocol_versions(&[&rustls::version::TLS13])?
            .with_client_cert_verifier(verifier.clone())
            .with_single_cert(chain.clone(), identity.key.clone_key())?;
        server.alpn_protocols = vec![ALPN.to_vec()];
        // No session resumption: every connection checks the certificate chain and the node id.
        server.session_storage = Arc::new(rustls::server::NoServerSessionStorage {});
        server.send_tls13_tickets = 0;
        let mut client = ClientConfig::builder_with_provider(provider)
            .with_protocol_versions(&[&rustls::version::TLS13])?
            .dangerous()
            .with_custom_certificate_verifier(verifier)
            .with_client_auth_cert(chain, identity.key.clone_key())?;
        client.alpn_protocols = vec![ALPN.to_vec()];
        client.resumption = rustls::client::Resumption::disabled();

        let rt = Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("tetraeder-transport")
            .enable_all()
            .build()
            .context("cannot start the transport runtime")?;
        let handle = rt.handle().clone();
        let rt = Arc::new(Rt(Some(rt)));
        let (stop, stop_rx) = watch::channel(false);
        let inner = Arc::new(Inner {
            me,
            client: TlsConnector::from(Arc::new(client)),
            acceptor: TlsAcceptor::from(Arc::new(server)),
            nodes_anchors: trust,
            buffers,
            live: Arc::new(Mutex::new(HashMap::new())),
            log: Mutex::new(Vec::new()),
            stop,
        });
        let (tx, rx) = mpsc::sync_channel(INBOX);
        let mut local_addr = None;
        let listening = if let Some(addr) = listen {
            let std_listener = bind(addr, buffers)?;
            local_addr = Some(std_listener.local_addr()?);
            let _guard = handle.enter();
            let listener = TcpListener::from_std(std_listener)?;
            handle.spawn(accept_loop(Arc::clone(&inner), listener, tx, stop_rx));
            true
        } else {
            false
        };
        Ok(Self {
            inner,
            rt,
            handle,
            local_addr,
            inbox: Mutex::new(rx),
            listening,
        })
    }

    /// The address bound (the real port when `listen` asked for port 0).
    pub fn local_addr(&self) -> Option<SocketAddr> {
        self.local_addr
    }

    /// The effective socket buffers of every connection made or accepted so far (also logged
    /// at `info` as they open).
    pub fn link_buffers(&self) -> Vec<LinkBuffers> {
        lock(&self.inner.log).clone()
    }

    fn wrap(&self, raw: Raw) -> Conn {
        let m = LinkMetrics::new(&self.inner.me, &raw.peer, "tls", raw.purpose.label());
        let (rd, wr) = tokio::io::split(raw.tls);
        build(
            &self.rt,
            &self.handle,
            rd,
            wr,
            raw.sock,
            Arc::clone(&raw.guard.dead),
            Some(raw.guard),
            raw.peer,
            raw.purpose,
            m,
        )
    }
}

impl Drop for TlsTransport {
    fn drop(&mut self) {
        let _ = self.inner.stop.send(true);
    }
}

fn bind(addr: &str, buffers: SocketBuffers) -> Result<std::net::TcpListener> {
    use socket2::{Domain, Socket, Type};
    let addr = addr
        .to_socket_addrs()
        .with_context(|| format!("cannot resolve {addr}"))?
        .next()
        .ok_or_else(|| anyhow!("{addr} has no address"))?;
    let s = Socket::new(Domain::for_address(addr), Type::STREAM, None)?;
    #[cfg(unix)]
    s.set_reuse_address(true)?;
    if let SocketBuffers::Explicit { send, recv } = buffers {
        s.set_send_buffer_size(send)?;
        s.set_recv_buffer_size(recv)?;
    }
    s.bind(&addr.into())
        .with_context(|| format!("cannot bind {addr}"))?;
    s.listen(1024)?;
    s.set_nonblocking(true)?;
    Ok(s.into())
}

async fn write_small<W: AsyncWrite + Unpin, T: Serialize>(w: &mut W, v: &T) -> io::Result<()> {
    let b = frame::encode(v, &[]).map_err(io::Error::other)?;
    w.write_all(&b).await?;
    w.flush().await
}

async fn read_small<R: AsyncRead + Unpin, T: serde::de::DeserializeOwned>(
    r: &mut R,
) -> io::Result<T> {
    let mut len = [0u8; 4];
    r.read_exact(&mut len).await?;
    let total = u32::from_le_bytes(len) as usize;
    if !(4..=MAX_PREFACE).contains(&total) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "bad preface length",
        ));
    }
    let mut buf = vec![0u8; 4 + total];
    buf[..4].copy_from_slice(&len);
    r.read_exact(&mut buf[4..]).await?;
    let f = frame::read_frame(&mut buf.as_slice())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
        .ok_or_else(|| io::Error::from(io::ErrorKind::UnexpectedEof))?;
    frame::parse_header(&f).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn peer_id(certs: Option<&[CertificateDer<'static>]>) -> io::Result<NodeId> {
    certs
        .and_then(<[_]>::first)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::PermissionDenied,
                "the peer sent no certificate",
            )
        })
        .and_then(|c| node_id_of(c).map_err(|e| io::Error::new(io::ErrorKind::PermissionDenied, e)))
}

fn keepalive(tcp: &TcpStream) -> io::Result<()> {
    let ka = socket2::TcpKeepalive::new()
        .with_time(KEEPALIVE_IDLE)
        .with_interval(KEEPALIVE_INTERVAL);
    SockRef::from(tcp).set_tcp_keepalive(&ka)
}

/// A refusal reason from the peer, cut so it cannot flood an error message or a log.
fn short(s: &str) -> String {
    s.chars().take(MAX_REASON).collect()
}

impl Inner {
    /// Registers an inbound connection, closing the one it replaces: the peer is authenticated,
    /// so a second connection for the same purpose means the first is stale (a crashed process,
    /// a dropped NAT entry). `None` when the map is full.
    fn register(
        &self,
        peer: &NodeId,
        purpose: &Purpose,
        sock: &std::net::TcpStream,
    ) -> Option<LiveGuard> {
        let key = (peer.clone(), purpose.clone());
        let dead = Arc::new(AtomicBool::new(false));
        let entry = LiveEntry {
            dead: Arc::clone(&dead),
            sock: sock.try_clone().ok()?,
        };
        let mut live = lock(&self.live);
        if !live.contains_key(&key) {
            if live.len() >= MAX_LIVE {
                metrics::rejected("limit");
                return None;
            }
            if live.keys().filter(|(p, _)| *p == key.0).count() >= MAX_LIVE_PER_PEER {
                metrics::rejected("peer_limit");
                return None;
            }
        }
        if let Some(old) = live.insert(key.clone(), entry) {
            old.dead.store(true, Ordering::SeqCst);
            let _ = old.sock.shutdown(Shutdown::Both);
        }
        Some(LiveGuard {
            live: Arc::clone(&self.live),
            key,
            dead,
        })
    }

    /// Keeps a handle that can shut the socket down, and records the effective buffers.
    fn tune(
        &self,
        tcp: TcpStream,
        peer: &NodeId,
        purpose: &Purpose,
    ) -> io::Result<(TcpStream, std::net::TcpStream)> {
        tcp.set_nodelay(true)?;
        keepalive(&tcp)?;
        let std = tcp.into_std()?;
        let keep = std.try_clone()?;
        match effective_buffers(&std) {
            Ok((send, recv)) => {
                tracing::info!(%peer, purpose = purpose.label(), send, recv, requested = ?self.buffers, "socket buffers");
                let mut log = lock(&self.log);
                if log.len() >= 1024 {
                    log.remove(0);
                }
                log.push(LinkBuffers {
                    peer: peer.clone(),
                    purpose: purpose.clone(),
                    send,
                    recv,
                    requested: self.buffers,
                });
            }
            Err(e) => tracing::warn!(%peer, "cannot read the socket buffers back: {e}"),
        }
        Ok((TcpStream::from_std(std)?, keep))
    }

    async fn dial(
        &self,
        peer: &Peer,
        purpose: &Purpose,
    ) -> io::Result<(ClientTls<TcpStream>, std::net::TcpStream, Arc<AtomicBool>)> {
        let addr = tokio::net::lookup_host(&peer.addr)
            .await?
            .next()
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("{} has no address", peer.addr),
                )
            })?;
        let sock = if addr.is_ipv4() {
            TcpSocket::new_v4()?
        } else {
            TcpSocket::new_v6()?
        };
        if let SocketBuffers::Explicit { send, recv } = self.buffers {
            sock.set_send_buffer_size(u32::try_from(send).unwrap_or(u32::MAX))?;
            sock.set_recv_buffer_size(u32::try_from(recv).unwrap_or(u32::MAX))?;
        }
        let tcp = sock.connect(addr).await?;
        let (tcp, keep) = self.tune(tcp, &peer.id, purpose)?;
        let name = ServerName::try_from(SNI)
            .map_err(io::Error::other)?
            .to_owned();
        let mut tls = match self.client.connect(name, tcp).await {
            Ok(t) => t,
            Err(e) => {
                metrics::rejected(reject_reason(&e));
                return Err(e);
            }
        };
        let got = peer_id(
            tls.get_ref()
                .1
                .peer_certificates()
                .map(<[_]>::to_vec)
                .as_deref(),
        )?;
        if got != peer.id {
            metrics::rejected("wrong_peer");
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!("dialed {} but the certificate names {got}", peer.id),
            ));
        }
        write_small(
            &mut tls,
            &Hello {
                purpose: purpose.clone(),
            },
        )
        .await?;
        // A refused client certificate or a duplicate shows here, not in the handshake.
        let ans: Answer = match read_small(&mut tls).await {
            Ok(a) => a,
            Err(e) => {
                let reason = reject_reason(&e);
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    format!("{} refused the connection ({reason}): {e}", peer.id),
                ));
            }
        };
        if !ans.ok {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!("{} refused the connection: {}", peer.id, short(&ans.reason)),
            ));
        }
        Ok((tls, keep, Arc::new(AtomicBool::new(false))))
    }

    /// Handshake of one inbound connection; a refusal is counted and returns `None`.
    async fn serve(&self, tcp: TcpStream, permit: IpPermit) -> Option<Raw> {
        let tcp = match self.tune_inbound(tcp) {
            Ok(t) => t,
            Err(_) => return None,
        };
        let (tcp, keep) = tcp;
        let mut tls = match tokio::time::timeout(TLS_TIMEOUT, self.acceptor.accept(tcp)).await {
            Ok(Ok(t)) => t,
            Ok(Err(e)) => {
                metrics::rejected(reject_reason(&e));
                return None;
            }
            Err(_) => {
                metrics::rejected("tls_timeout");
                return None;
            }
        };
        // Authenticated: the live map bounds this peer from here on.
        drop(permit);
        let peer = match peer_id(
            tls.get_ref()
                .1
                .peer_certificates()
                .map(<[_]>::to_vec)
                .as_deref(),
        ) {
            Ok(p) => p,
            Err(_) => {
                metrics::rejected("no_identity");
                return None;
            }
        };
        let hello: Hello = match read_small(&mut tls).await {
            Ok(h) => h,
            Err(_) => {
                metrics::rejected("bad_hello");
                return None;
            }
        };
        // The guard clears the entry on every way out below, also when the timeout cancels us.
        let Some(guard) = self.register(&peer, &hello.purpose, &keep) else {
            let busy = Answer {
                ok: false,
                reason: "too many connections".into(),
            };
            let _ = write_small(&mut tls, &busy).await;
            return None;
        };
        let ok = Answer {
            ok: true,
            reason: String::new(),
        };
        if write_small(&mut tls, &ok).await.is_err() {
            return None;
        }
        {
            let (send, recv) = effective_buffers(&keep).unwrap_or((0, 0));
            tracing::info!(%peer, purpose = hello.purpose.label(), send, recv, requested = ?self.buffers, "socket buffers");
            let mut log = lock(&self.log);
            if log.len() >= 1024 {
                log.remove(0);
            }
            log.push(LinkBuffers {
                peer: peer.clone(),
                purpose: hello.purpose.clone(),
                send,
                recv,
                requested: self.buffers,
            });
        }
        Some(Raw {
            tls,
            sock: keep,
            peer,
            purpose: hello.purpose,
            guard,
        })
    }

    fn tune_inbound(&self, tcp: TcpStream) -> io::Result<(TcpStream, std::net::TcpStream)> {
        tcp.set_nodelay(true)?;
        keepalive(&tcp)?;
        let std = tcp.into_std()?;
        let keep = std.try_clone()?;
        Ok((TcpStream::from_std(std)?, keep))
    }
}

async fn accept_loop(
    inner: Arc<Inner>,
    listener: TcpListener,
    tx: mpsc::SyncSender<Raw>,
    mut stop: watch::Receiver<bool>,
) {
    let slots = Arc::new(tokio::sync::Semaphore::new(MAX_HANDSHAKES));
    let per_ip = Arc::new(IpLimiter::default());
    loop {
        tokio::select! {
            r = listener.accept() => {
                let (tcp, from) = match r {
                    Ok(x) => x,
                    Err(e) => {
                        // Out of descriptors or a reset before accept: do not spin.
                        tracing::warn!("accept failed: {e}");
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        continue;
                    }
                };
                let Some(permit) =
                    per_ip.try_acquire(from.ip().to_canonical(), MAX_HANDSHAKES_PER_IP)
                else {
                    metrics::rejected("ip_limit");
                    continue;
                };
                let Ok(slot) = Arc::clone(&slots).try_acquire_owned() else {
                    metrics::rejected("limit");
                    continue;
                };
                let (inner, tx) = (Arc::clone(&inner), tx.clone());
                tokio::spawn(async move {
                    let _slot = slot;
                    match tokio::time::timeout(ACCEPT_TIMEOUT, inner.serve(tcp, permit)).await {
                        Ok(Some(raw)) => {
                            // A full inbox drops the connection (and its registration).
                            let _ = tx.try_send(raw);
                        }
                        // Refused: already counted.
                        Ok(None) => {}
                        Err(_) => metrics::rejected("handshake"),
                    }
                });
            }
            _ = stop.changed() => break,
        }
    }
}

impl Transport for TlsTransport {
    fn connect(&self, peer: &Peer, purpose: Purpose) -> io::Result<Conn> {
        let inner = Arc::clone(&self.inner);
        let (tls, keep, dead) = self.handle.block_on(async {
            tokio::time::timeout(CONNECT_TIMEOUT, inner.dial(peer, &purpose))
                .await
                .unwrap_or_else(|_| {
                    Err(io::Error::new(io::ErrorKind::TimedOut, "connect timed out"))
                })
        })?;
        let m = LinkMetrics::new(&self.inner.me, &peer.id, "tls", purpose.label());
        let (rd, wr) = tokio::io::split(tls);
        Ok(build(
            &self.rt,
            &self.handle,
            rd,
            wr,
            keep,
            dead,
            None,
            peer.id.clone(),
            purpose,
            m,
        ))
    }

    fn accept(&self) -> io::Result<Conn> {
        if !self.listening {
            return Err(io::Error::new(
                io::ErrorKind::NotConnected,
                "this transport does not listen",
            ));
        }
        let rx = lock(&self.inbox);
        loop {
            if *self.inner.stop.borrow() {
                return Err(io::Error::new(
                    io::ErrorKind::NotConnected,
                    "transport shut down",
                ));
            }
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(raw) => return Ok(self.wrap(raw)),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(io::Error::new(
                        io::ErrorKind::NotConnected,
                        "transport shut down",
                    ));
                }
            }
        }
    }

    fn name(&self) -> &'static str {
        "tls"
    }

    fn local(&self) -> &NodeId {
        &self.inner.me
    }

    fn shutdown(&self) {
        let _ = self.inner.stop.send(true);
    }
}

impl std::fmt::Debug for TlsTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TlsTransport")
            .field("node", &self.inner.me)
            .field("local_addr", &self.local_addr)
            .field("anchors", &self.inner.nodes_anchors)
            .finish_non_exhaustive()
    }
}

/// What the two halves of a connection share.
struct Shared {
    closed: watch::Sender<bool>,
    dead: Arc<AtomicBool>,
    sock: std::net::TcpStream,
    /// Clears the acceptor's registration when the last half is gone.
    _guard: Option<LiveGuard>,
    /// Keeps the runtime alive while a half exists.
    _rt: Arc<Rt>,
}

impl Shared {
    fn close(&self) {
        self.dead.store(true, Ordering::SeqCst);
        let _ = self.closed.send(true);
        let _ = self.sock.shutdown(Shutdown::Both);
    }
}

#[allow(clippy::too_many_arguments)]
fn build<S>(
    rt: &Arc<Rt>,
    handle: &Handle,
    rd: ReadHalf<S>,
    wr: WriteHalf<S>,
    sock: std::net::TcpStream,
    dead: Arc<AtomicBool>,
    guard: Option<LiveGuard>,
    peer: NodeId,
    purpose: Purpose,
    m: LinkMetrics,
) -> Conn
where
    S: AsyncRead + AsyncWrite + Send + Unpin + 'static,
{
    let (closed, closed_rx) = watch::channel(false);
    let shared = Arc::new(Shared {
        closed,
        dead,
        sock,
        _guard: guard,
        _rt: Arc::clone(rt),
    });
    let rx = Rx {
        rd,
        closed: closed_rx.clone(),
        handle: handle.clone(),
        shared: Arc::clone(&shared),
    };
    let tx = Tx {
        wr: Some(wr),
        closed: closed_rx,
        handle: handle.clone(),
        shared: Arc::clone(&shared),
        m,
    };
    Conn {
        peer,
        purpose,
        rx: Box::new(rx),
        tx: Box::new(tx),
        close: Arc::new(move || shared.close()),
    }
}

fn aborted() -> io::Error {
    io::Error::from(io::ErrorKind::ConnectionAborted)
}

struct Rx<S> {
    rd: ReadHalf<S>,
    closed: watch::Receiver<bool>,
    handle: Handle,
    shared: Arc<Shared>,
}

impl<S: AsyncRead + AsyncWrite + Send + Unpin + 'static> Read for Rx<S> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if *self.closed.borrow() {
            return Err(aborted());
        }
        let (rd, closed) = (&mut self.rd, &mut self.closed);
        let r = self.handle.block_on(async {
            tokio::select! {
                r = rd.read(buf) => r,
                _ = closed.wait_for(|c| *c) => Err(aborted()),
            }
        });
        match r {
            // A peer that vanished without `close_notify` (a crash, a cut socket) is a reset,
            // not a clean end: only `close_notify` reads as `Ok(0)`.
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                self.shared.dead.store(true, Ordering::SeqCst);
                Err(io::Error::from(io::ErrorKind::ConnectionReset))
            }
            Ok(0) | Err(_) => {
                self.shared.dead.store(true, Ordering::SeqCst);
                r
            }
            ok => ok,
        }
    }
}

struct Tx<S: AsyncRead + AsyncWrite + Send + Unpin + 'static> {
    wr: Option<WriteHalf<S>>,
    closed: watch::Receiver<bool>,
    handle: Handle,
    shared: Arc<Shared>,
    m: LinkMetrics,
}

impl<S: AsyncRead + AsyncWrite + Send + Unpin + 'static> Write for Tx<S> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if *self.closed.borrow() {
            return Err(aborted());
        }
        let wr = self.wr.as_mut().ok_or_else(aborted)?;
        let closed = &mut self.closed;
        let r = self.handle.block_on(async {
            tokio::select! {
                r = wr.write_all(buf) => r,
                _ = closed.wait_for(|c| *c) => Err(aborted()),
            }
        });
        match r {
            Ok(()) => {
                self.m.bytes(buf.len());
                Ok(buf.len())
            }
            Err(e) => {
                self.shared.dead.store(true, Ordering::SeqCst);
                Err(e)
            }
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        let wr = self.wr.as_mut().ok_or_else(aborted)?;
        let closed = &mut self.closed;
        self.handle.block_on(async {
            tokio::select! {
                r = wr.flush() => r,
                _ = closed.wait_for(|c| *c) => Err(aborted()),
            }
        })
    }
}

impl<S: AsyncRead + AsyncWrite + Send + Unpin + 'static> Drop for Tx<S> {
    /// Sends `close_notify` so the peer reads a clean end of input.
    fn drop(&mut self) {
        if let Some(mut wr) = self.wr.take()
            && !*self.closed.borrow()
        {
            self.shared.dead.store(true, Ordering::SeqCst);
            self.handle.spawn(async move {
                let _ = wr.shutdown().await;
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_source_cannot_hold_more_than_its_share_of_handshakes() {
        let l = Arc::new(IpLimiter::default());
        let a: IpAddr = "10.0.0.1".parse().unwrap();
        let b: IpAddr = "10.0.0.2".parse().unwrap();
        let held: Vec<_> = (0..3).map(|_| l.try_acquire(a, 3).unwrap()).collect();
        assert!(l.try_acquire(a, 3).is_none());
        assert!(l.try_acquire(b, 3).is_some());
        drop(held);
        assert!(l.try_acquire(a, 3).is_some());
        // Slots given back leave no entry behind.
        assert!(lock(&l.counts).is_empty());
    }

    #[test]
    fn a_refusal_reason_is_cut() {
        let long = "x".repeat(10_000);
        assert_eq!(short(&long).len(), MAX_REASON);
        assert_eq!(short("duplicate"), "duplicate");
    }
}
