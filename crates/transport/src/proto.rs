//! Wire protocol of the spike: a 12-byte header, then `count` messages of `size` bytes.
//!
//! `Echo`: the server returns every message (a dispatch/combine round trip).
//! `Sink`: the server reads `count` messages and answers with one ack byte (one-way throughput).

use std::io;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

pub const OP_ECHO: u32 = 1;
pub const OP_SINK: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub op: u32,
    pub size: u32,
    pub count: u32,
}

pub async fn write_header<W: AsyncWrite + Unpin>(w: &mut W, h: Header) -> io::Result<()> {
    let mut b = [0u8; 12];
    b[..4].copy_from_slice(&h.op.to_be_bytes());
    b[4..8].copy_from_slice(&h.size.to_be_bytes());
    b[8..].copy_from_slice(&h.count.to_be_bytes());
    w.write_all(&b).await
}

/// `None` on a clean end of stream before a header.
pub async fn read_header<R: AsyncRead + Unpin>(r: &mut R) -> io::Result<Option<Header>> {
    let mut b = [0u8; 12];
    match r.read_exact(&mut b).await {
        Ok(_) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let u = |i: usize| u32::from_be_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
    Ok(Some(Header {
        op: u(0),
        size: u(4),
        count: u(8),
    }))
}

/// Serve requests on one bidirectional stream until the peer closes it.
pub async fn serve_stream<R: AsyncRead + Unpin, W: AsyncWrite + Unpin>(
    mut r: R,
    mut w: W,
) -> io::Result<()> {
    let mut buf = Vec::new();
    while let Some(h) = read_header(&mut r).await? {
        buf.resize(h.size as usize, 0);
        match h.op {
            OP_ECHO => {
                for _ in 0..h.count {
                    r.read_exact(&mut buf).await?;
                    w.write_all(&buf).await?;
                    w.flush().await?;
                }
            }
            OP_SINK => {
                for _ in 0..h.count {
                    r.read_exact(&mut buf).await?;
                }
                w.write_all(&[1]).await?;
                w.flush().await?;
            }
            op => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown op {op}"),
                ));
            }
        }
    }
    Ok(())
}
