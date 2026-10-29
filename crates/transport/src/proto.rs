//! Wire protocol of the spike: a 12-byte header, then `count` messages of `size` bytes.
//!
//! `Echo`: the server returns every message (a dispatch/combine round trip).
//! `Sink`: the server reads `count` messages and answers with one ack byte (one-way throughput).

use std::io;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

pub const OP_ECHO: u32 = 1;
pub const OP_SINK: u32 = 2;
/// Largest message size accepted from a peer; the server allocates it.
pub const MAX_MESSAGE_SIZE: u32 = 1 << 30;

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

/// `None` only when the stream ends before the first header byte; a header cut short is an
/// `UnexpectedEof` error, and a size above `MAX_MESSAGE_SIZE` is `InvalidData`.
pub async fn read_header<R: AsyncRead + Unpin>(r: &mut R) -> io::Result<Option<Header>> {
    let mut b = [0u8; 12];
    let mut got = 0;
    while got < b.len() {
        let n = r.read(&mut b[got..]).await?;
        if n == 0 {
            return if got == 0 {
                Ok(None)
            } else {
                Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "stream ended inside a header",
                ))
            };
        }
        got += n;
    }
    let u = |i: usize| u32::from_be_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
    let h = Header {
        op: u(0),
        size: u(4),
        count: u(8),
    };
    if h.size > MAX_MESSAGE_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("message size {} exceeds {MAX_MESSAGE_SIZE}", h.size),
        ));
    }
    Ok(Some(h))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(op: u32, size: u32, count: u32) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend(op.to_be_bytes());
        v.extend(size.to_be_bytes());
        v.extend(count.to_be_bytes());
        v
    }

    #[tokio::test]
    async fn header_roundtrip_and_clean_eof() {
        let mut buf = Vec::new();
        let h = Header {
            op: OP_ECHO,
            size: 64,
            count: 3,
        };
        write_header(&mut buf, h).await.unwrap();
        let mut r = &buf[..];
        assert_eq!(read_header(&mut r).await.unwrap(), Some(h));
        assert_eq!(read_header(&mut r).await.unwrap(), None);
    }

    #[tokio::test]
    async fn truncated_header_is_an_error() {
        let full = bytes(OP_SINK, 8, 1);
        for cut in 1..12 {
            let mut r = &full[..cut];
            let e = read_header(&mut r).await.unwrap_err();
            assert_eq!(e.kind(), io::ErrorKind::UnexpectedEof, "cut {cut}");
        }
    }

    #[tokio::test]
    async fn oversized_message_is_rejected() {
        let ok = bytes(OP_SINK, MAX_MESSAGE_SIZE, 1);
        assert!(read_header(&mut &ok[..]).await.unwrap().is_some());
        let big = bytes(OP_SINK, MAX_MESSAGE_SIZE + 1, 1);
        let e = read_header(&mut &big[..]).await.unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::InvalidData);
    }
}
