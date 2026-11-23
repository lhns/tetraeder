//! The worker frame format (moved unchanged from `tetraeder::remote_backend`, ADR 0091 step 1).
//!
//! Frame: `u32` LE length of the rest, `u32` LE header length, a `serde_json` header, then a raw
//! payload (the logits of a `decode` reply as little-endian f32, snapshot bytes). The header may
//! carry a W3C `traceparent` field; see [`traceparent`].

use serde::Serialize;
use serde::de::DeserializeOwned;
use std::io::{Read, Write};

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

/// The W3C `traceparent` of a frame, if its JSON header has a `traceparent` string. Headers of
/// every call may carry one (ADR 0091); a frame without it comes from an older peer.
pub fn traceparent(f: &Frame) -> Option<String> {
    let v: serde_json::Value = serde_json::from_slice(&f.header).ok()?;
    v.get("traceparent")?.as_str().map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    #[serde(tag = "call", rename_all = "snake_case")]
    enum Head {
        Info,
        SeqRm { seq: u32, keep: usize },
    }

    // hardware: none
    #[test]
    fn frames_round_trip() {
        let mut buf = Vec::new();
        write_frame(&mut buf, &Head::SeqRm { seq: 3, keep: 7 }, &[1, 2, 3]).unwrap();
        write_frame(&mut buf, &Head::Info, &[]).unwrap();
        let mut r = buf.as_slice();
        let f = read_frame(&mut r).unwrap().unwrap();
        assert_eq!(f.payload, [1, 2, 3]);
        assert_eq!(
            parse_header::<Head>(&f).unwrap(),
            Head::SeqRm { seq: 3, keep: 7 }
        );
        let f = read_frame(&mut r).unwrap().unwrap();
        assert_eq!(parse_header::<Head>(&f).unwrap(), Head::Info);
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

    /// Splitmix64: a seeded generator, so the property cases are the same on every run.
    fn next(s: &mut u64) -> u64 {
        *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    // hardware: none
    #[test]
    fn any_payload_round_trips_and_a_stream_of_frames_splits_back() {
        let mut s = 7;
        let mut sent = Vec::new();
        let mut buf = Vec::new();
        for _ in 0..200 {
            let n = (next(&mut s) % 300) as usize;
            let payload: Vec<u8> = (0..n).map(|_| next(&mut s) as u8).collect();
            let head = Head::SeqRm {
                seq: next(&mut s) as u32,
                keep: (next(&mut s) % 100_000) as usize,
            };
            write_frame(&mut buf, &head, &payload).unwrap();
            sent.push((head, payload));
        }
        let mut r = buf.as_slice();
        for (head, payload) in &sent {
            let f = read_frame(&mut r).unwrap().unwrap();
            assert_eq!(&parse_header::<Head>(&f).unwrap(), head);
            assert_eq!(&f.payload, payload);
        }
        assert!(read_frame(&mut r).unwrap().is_none());
    }

    // hardware: none
    #[test]
    fn a_cut_or_flipped_stream_is_an_error_or_a_frame_never_a_panic() {
        let mut s = 99;
        let mut buf = Vec::new();
        write_frame(&mut buf, &Head::SeqRm { seq: 1, keep: 2 }, &[5; 40]).unwrap();
        for cut in 0..buf.len() {
            let mut r = &buf[..cut];
            let got = read_frame(&mut r);
            assert!(matches!(got, Ok(None) | Err(_)), "cut {cut}");
        }
        for _ in 0..500 {
            let mut b = buf.clone();
            let i = (next(&mut s) as usize) % b.len();
            b[i] ^= 1 << (next(&mut s) % 8);
            let mut r = b.as_slice();
            let _ = read_frame(&mut r);
        }
    }

    // hardware: none
    #[test]
    fn the_traceparent_is_read_from_the_header() {
        let tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
        let mut buf = Vec::new();
        let head = serde_json::json!({"call": "info", "traceparent": tp});
        write_frame(&mut buf, &head, &[]).unwrap();
        write_frame(&mut buf, &Head::Info, &[]).unwrap();
        let mut r = buf.as_slice();
        let f = read_frame(&mut r).unwrap().unwrap();
        assert_eq!(traceparent(&f).as_deref(), Some(tp));
        assert_eq!(traceparent(&read_frame(&mut r).unwrap().unwrap()), None);
    }
}
