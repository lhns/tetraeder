//! Servers, clients and measurements for the three spike transports.

use crate::proto::{self, Header};
use crate::tls;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Transport {
    /// Plain TCP: the floor.
    Tcp,
    /// TCP with TLS 1.3 (rustls).
    Tls,
    /// QUIC (quinn), one bidirectional stream.
    Quic,
}

impl Transport {
    pub const ALL: [Transport; 3] = [Transport::Tcp, Transport::Tls, Transport::Quic];

    pub fn name(self) -> &'static str {
        match self {
            Transport::Tcp => "tcp",
            Transport::Tls => "tls",
            Transport::Quic => "quic",
        }
    }
}

type Reader = Box<dyn AsyncRead + Unpin + Send>;
type Writer = Box<dyn AsyncWrite + Unpin + Send>;

/// A bound server. `run` accepts connections until the task is dropped.
pub struct Server {
    pub transport: Transport,
    pub addr: SocketAddr,
    inner: Inner,
}

enum Inner {
    Tcp(TcpListener),
    Tls(TcpListener, tokio_rustls::TlsAcceptor),
    Quic(quinn::Endpoint),
}

fn quic_transport_config() -> Arc<quinn::TransportConfig> {
    // Large windows so flow control does not cap throughput on a fast link; same for both ends.
    let mut t = quinn::TransportConfig::default();
    t.stream_receive_window(quinn::VarInt::from_u32(32 << 20));
    t.receive_window(quinn::VarInt::from_u32(64 << 20));
    t.send_window(64 << 20);
    Arc::new(t)
}

pub async fn bind(transport: Transport, addr: SocketAddr) -> Result<Server> {
    let inner = match transport {
        Transport::Tcp => Inner::Tcp(TcpListener::bind(addr).await?),
        Transport::Tls => Inner::Tls(
            TcpListener::bind(addr).await?,
            tokio_rustls::TlsAcceptor::from(Arc::new(tls::server_config()?)),
        ),
        Transport::Quic => {
            let crypto = quinn::crypto::rustls::QuicServerConfig::try_from(tls::server_config()?)?;
            let mut cfg = quinn::ServerConfig::with_crypto(Arc::new(crypto));
            cfg.transport_config(quic_transport_config());
            Inner::Quic(quinn::Endpoint::server(cfg, addr)?)
        }
    };
    let addr = match &inner {
        Inner::Tcp(l) | Inner::Tls(l, _) => l.local_addr()?,
        Inner::Quic(e) => e.local_addr()?,
    };
    Ok(Server {
        transport,
        addr,
        inner,
    })
}

impl Server {
    pub async fn run(self) -> Result<()> {
        match self.inner {
            Inner::Tcp(l) => loop {
                let (s, _) = l.accept().await?;
                s.set_nodelay(true)?;
                tokio::spawn(async move {
                    let (r, w) = tokio::io::split(s);
                    let _ = proto::serve_stream(r, w).await;
                });
            },
            Inner::Tls(l, acceptor) => loop {
                let (s, _) = l.accept().await?;
                s.set_nodelay(true)?;
                let acceptor = acceptor.clone();
                tokio::spawn(async move {
                    if let Ok(s) = acceptor.accept(s).await {
                        let (r, w) = tokio::io::split(s);
                        let _ = proto::serve_stream(r, w).await;
                    }
                });
            },
            Inner::Quic(e) => {
                while let Some(incoming) = e.accept().await {
                    tokio::spawn(async move {
                        let Ok(conn) = incoming.await else { return };
                        while let Ok((w, r)) = conn.accept_bi().await {
                            tokio::spawn(async move {
                                let _ = proto::serve_stream(r, w).await;
                            });
                        }
                    });
                }
                Ok(())
            }
        }
    }
}

/// One open connection with a single bidirectional stream.
pub struct Client {
    r: Reader,
    w: Writer,
    // Keeps the QUIC endpoint and connection alive.
    _guard: Option<(quinn::Endpoint, quinn::Connection)>,
}

pub async fn connect(transport: Transport, addr: SocketAddr) -> Result<Client> {
    match transport {
        Transport::Tcp => {
            let s = TcpStream::connect(addr).await?;
            s.set_nodelay(true)?;
            let (r, w) = tokio::io::split(s);
            Ok(Client {
                r: Box::new(r),
                w: Box::new(w),
                _guard: None,
            })
        }
        Transport::Tls => {
            let s = TcpStream::connect(addr).await?;
            s.set_nodelay(true)?;
            let connector = tokio_rustls::TlsConnector::from(Arc::new(tls::client_config()?));
            let name = rustls::pki_types::ServerName::try_from(tls::SERVER_NAME)?;
            let (r, w) = tokio::io::split(connector.connect(name, s).await?);
            Ok(Client {
                r: Box::new(r),
                w: Box::new(w),
                _guard: None,
            })
        }
        Transport::Quic => {
            let local: SocketAddr = if addr.is_ipv4() {
                "0.0.0.0:0"
            } else {
                "[::]:0"
            }
            .parse()?;
            let mut endpoint = quinn::Endpoint::client(local)?;
            let mut cfg = quinn::ClientConfig::new(Arc::new(
                quinn::crypto::rustls::QuicClientConfig::try_from(tls::client_config()?)?,
            ));
            cfg.transport_config(quic_transport_config());
            endpoint.set_default_client_config(cfg);
            let conn = endpoint
                .connect(addr, tls::SERVER_NAME)?
                .await
                .context("QUIC handshake")?;
            let (w, r) = conn.open_bi().await?;
            Ok(Client {
                r: Box::new(r),
                w: Box::new(w),
                _guard: Some((endpoint, conn)),
            })
        }
    }
}

impl Client {
    /// Send `iters + warmup` echo requests of `size` bytes one after another; returns the round-trip
    /// time of each measured one. The reply is checked after the clock stops.
    pub async fn latency(
        &mut self,
        size: usize,
        warmup: usize,
        iters: usize,
    ) -> Result<Vec<Duration>> {
        proto::write_header(
            &mut self.w,
            Header {
                op: proto::OP_ECHO,
                size: size as u32,
                count: (warmup + iters) as u32,
            },
        )
        .await?;
        let mut out = Vec::with_capacity(iters);
        let mut msg = vec![0u8; size];
        let mut back = vec![0u8; size];
        for i in 0..warmup + iters {
            msg.iter_mut()
                .enumerate()
                .for_each(|(j, b)| *b = (i + j) as u8);
            let t0 = Instant::now();
            self.w.write_all(&msg).await?;
            self.w.flush().await?;
            self.r.read_exact(&mut back).await?;
            let dt = t0.elapsed();
            if back != msg {
                bail!("echo mismatch at request {i}");
            }
            if i >= warmup {
                out.push(dt);
            }
        }
        Ok(out)
    }

    /// Stream `total_bytes` in messages of `size`, one way, until the server acks. MiB per second.
    pub async fn throughput(&mut self, size: usize, total_bytes: usize) -> Result<f64> {
        let count = (total_bytes / size).max(1);
        proto::write_header(
            &mut self.w,
            Header {
                op: proto::OP_SINK,
                size: size as u32,
                count: count as u32,
            },
        )
        .await?;
        let msg = vec![0xABu8; size];
        let t0 = Instant::now();
        for _ in 0..count {
            self.w.write_all(&msg).await?;
        }
        self.w.flush().await?;
        let mut ack = [0u8; 1];
        self.r.read_exact(&mut ack).await?;
        let secs = t0.elapsed().as_secs_f64();
        Ok((count * size) as f64 / (1024.0 * 1024.0) / secs)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Row {
    pub transport: Transport,
    pub size: usize,
    pub iters: usize,
    pub min_us: f64,
    pub p50_us: f64,
    pub p99_us: f64,
    pub mean_us: f64,
    pub max_us: f64,
    pub throughput_mib_s: f64,
}

fn percentile(sorted: &[Duration], q: f64) -> f64 {
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx].as_secs_f64() * 1e6
}

pub fn latency_stats(mut d: Vec<Duration>) -> (f64, f64, f64, f64, f64) {
    d.sort();
    let mean = d.iter().map(Duration::as_secs_f64).sum::<f64>() / d.len() as f64 * 1e6;
    (
        percentile(&d, 0.0),
        percentile(&d, 0.5),
        percentile(&d, 0.99),
        mean,
        percentile(&d, 1.0),
    )
}

pub struct Plan {
    pub sizes: Vec<usize>,
    pub iters: usize,
    pub warmup: usize,
    pub throughput_bytes: usize,
}

/// Measure one transport against a running server, for every size.
pub async fn measure(transport: Transport, addr: SocketAddr, plan: &Plan) -> Result<Vec<Row>> {
    let mut client = connect(transport, addr)
        .await
        .with_context(|| format!("{} connect to {addr}", transport.name()))?;
    let mut rows = Vec::new();
    for &size in &plan.sizes {
        // Large messages need far fewer samples for the same wall time.
        let iters = if size >= 256 * 1024 {
            (plan.iters / 5).max(20)
        } else {
            plan.iters
        };
        let samples = client.latency(size, plan.warmup, iters).await?;
        let (min_us, p50_us, p99_us, mean_us, max_us) = latency_stats(samples);
        let throughput_mib_s = client.throughput(size, plan.throughput_bytes).await?;
        rows.push(Row {
            transport,
            size,
            iters,
            min_us,
            p50_us,
            p99_us,
            mean_us,
            max_us,
            throughput_mib_s,
        });
    }
    Ok(rows)
}

pub fn table(rows: &[Row]) -> String {
    let mut s = format!(
        "{:<6} {:>9} {:>6} {:>10} {:>10} {:>10} {:>10} {:>12}\n",
        "xport", "size", "iters", "min us", "p50 us", "p99 us", "mean us", "MiB/s"
    );
    for r in rows {
        s += &format!(
            "{:<6} {:>9} {:>6} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>12.1}\n",
            r.transport.name(),
            r.size,
            r.iters,
            r.min_us,
            r.p50_us,
            r.p99_us,
            r.mean_us,
            r.throughput_mib_s
        );
    }
    s
}
