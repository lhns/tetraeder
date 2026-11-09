//! The `tetraeder-routing` v1 trace (ADR 0050 (c)): one JSON header line, then binary
//! little-endian records.
//!
//! Record: kind (`u8`: 0 layer step, 1 hot-set refresh), step (`u32`); a layer step goes on
//! with layer (`u16`), `n_tokens` (`u32`), flags (`u8`: bit 0 predicted, bit 1 predicted2), the
//! routed ids (`n_tokens * n_used`, token-major, `id_bytes` each), then the predicted ids and the
//! depth-2 predicted ids when flagged (same count and width). A refresh record marks the point
//! where the engine rebuilt the VRAM hot set from the routing counts so far.

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

pub const FORMAT: &str = "tetraeder-routing";
pub const VERSION: u32 = 1;

const KIND_STEP: u8 = 0;
const KIND_REFRESH: u8 = 1;
const FLAG_PREDICTED: u8 = 1;
const FLAG_PREDICTED2: u8 = 2;

/// What a trace needs to be replayed against other tier sizes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceHeader {
    pub format: String,
    pub version: u32,
    pub model: String,
    pub n_layer: u32,
    pub n_expert: u32,
    pub n_used: u32,
    /// Bytes per id on the wire: 1 when `n_expert <= 256`, else 2.
    pub id_bytes: u8,
    /// Slots per layer of the recorded run.
    pub vram_slots: u32,
    pub ram_slots: u32,
    /// Prefetch depth of the run, 0 when prefetch was off.
    pub prefetch_depth: u32,
    /// Where the VRAM hot set came from: `first_prefill`, `refresh` or `profile`.
    pub hot_source: String,
    pub ubatch: u32,
    pub prompt_hash: String,
    /// Bytes of one expert (gate, up and down slab) per layer.
    pub expert_bytes: Vec<u64>,
    /// Routing counts the run started from (`LoadOptions::routing_profile`), `[layer][expert]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<Vec<Vec<u64>>>,
}

impl TraceHeader {
    /// A header with the wire width derived from `n_expert`.
    pub fn new(n_layer: u32, n_expert: u32, n_used: u32) -> Self {
        Self {
            format: FORMAT.into(),
            version: VERSION,
            model: String::new(),
            n_layer,
            n_expert,
            n_used,
            id_bytes: id_bytes_for(n_expert),
            vram_slots: 0,
            ram_slots: 0,
            prefetch_depth: 0,
            hot_source: "first_prefill".into(),
            ubatch: 0,
            prompt_hash: String::new(),
            expert_bytes: vec![0; n_layer as usize],
            profile: None,
        }
    }

    fn check(&self) -> io::Result<()> {
        if self.format != FORMAT || self.version != VERSION {
            return Err(bad(format!(
                "not a {FORMAT} v{VERSION} trace ({} v{})",
                self.format, self.version
            )));
        }
        if self.id_bytes != id_bytes_for(self.n_expert) || self.n_used == 0 {
            return Err(bad("inconsistent header (id_bytes or n_used)".into()));
        }
        if self.expert_bytes.len() != self.n_layer as usize {
            return Err(bad("expert_bytes must have one entry per layer".into()));
        }
        Ok(())
    }
}

fn id_bytes_for(n_expert: u32) -> u8 {
    if n_expert <= 256 { 1 } else { 2 }
}

fn bad(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

/// One routed layer step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerStep {
    pub step: u32,
    pub layer: u16,
    pub n_tokens: u32,
    /// `[n_used, n_tokens]`, token-major.
    pub ids: Vec<u16>,
    /// The experts of the next layer that this layer predicted, same layout as `ids`.
    pub predicted: Option<Vec<u16>>,
    /// The experts of the layer after next that this layer predicted.
    pub predicted2: Option<Vec<u16>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Record {
    Step(LayerStep),
    /// The VRAM hot set was rebuilt after step `step`.
    HotRefresh {
        step: u32,
    },
}

pub struct TraceWriter<W: Write> {
    out: W,
    header: TraceHeader,
}

impl TraceWriter<BufWriter<File>> {
    pub fn create(path: &Path, header: &TraceHeader) -> io::Result<Self> {
        Self::new(BufWriter::new(File::create(path)?), header.clone())
    }
}

impl<W: Write> TraceWriter<W> {
    pub fn new(mut out: W, header: TraceHeader) -> io::Result<Self> {
        header.check()?;
        serde_json::to_writer(&mut out, &header).map_err(io::Error::other)?;
        out.write_all(b"\n")?;
        Ok(Self { out, header })
    }

    /// One layer step. `ids` and the predictions are `[n_used, n_tokens]`; ids outside
    /// `0..n_expert` are an error.
    pub fn layer_step(
        &mut self,
        step: u32,
        layer: u16,
        n_tokens: u32,
        ids: &[i32],
        predicted: Option<&[i32]>,
        predicted2: Option<&[i32]>,
    ) -> io::Result<()> {
        let n = n_tokens as usize * self.header.n_used as usize;
        for v in std::iter::once(ids).chain(predicted).chain(predicted2) {
            if v.len() != n {
                return Err(bad(format!("{} ids for {n} expected", v.len())));
            }
        }
        let flags = u8::from(predicted.is_some()) * FLAG_PREDICTED
            + u8::from(predicted2.is_some()) * FLAG_PREDICTED2;
        self.out.write_all(&[KIND_STEP])?;
        self.out.write_all(&step.to_le_bytes())?;
        self.out.write_all(&layer.to_le_bytes())?;
        self.out.write_all(&n_tokens.to_le_bytes())?;
        self.out.write_all(&[flags])?;
        for v in std::iter::once(ids).chain(predicted).chain(predicted2) {
            self.write_ids(v)?;
        }
        Ok(())
    }

    fn write_ids(&mut self, ids: &[i32]) -> io::Result<()> {
        let mut buf = Vec::with_capacity(ids.len() * usize::from(self.header.id_bytes));
        for &e in ids {
            let e = u32::try_from(e)
                .ok()
                .filter(|&e| e < self.header.n_expert)
                .ok_or_else(|| bad(format!("expert id {e} out of range")))?;
            if self.header.id_bytes == 1 {
                buf.push(e as u8);
            } else {
                buf.extend_from_slice(&(e as u16).to_le_bytes());
            }
        }
        self.out.write_all(&buf)
    }

    /// The engine rebuilt the VRAM hot set after step `step`.
    pub fn hot_refresh(&mut self, step: u32) -> io::Result<()> {
        self.out.write_all(&[KIND_REFRESH])?;
        self.out.write_all(&step.to_le_bytes())
    }

    pub fn finish(mut self) -> io::Result<()> {
        self.out.flush()
    }
}

pub struct TraceReader<R: Read> {
    input: R,
    header: TraceHeader,
}

impl TraceReader<BufReader<File>> {
    pub fn open(path: &Path) -> io::Result<Self> {
        Self::new(BufReader::new(File::open(path)?))
    }
}

impl<R: Read> TraceReader<R> {
    pub fn new(mut input: R) -> io::Result<Self> {
        // The header line, read byte by byte so no record byte is consumed.
        let mut line = Vec::new();
        let mut b = [0u8; 1];
        loop {
            input.read_exact(&mut b)?;
            if b[0] == b'\n' {
                break;
            }
            line.push(b[0]);
            if line.len() > 64 << 20 {
                return Err(bad("trace header too long".into()));
            }
        }
        let header: TraceHeader =
            serde_json::from_slice(&line).map_err(|e| bad(format!("trace header: {e}")))?;
        header.check()?;
        Ok(Self { input, header })
    }

    pub fn header(&self) -> &TraceHeader {
        &self.header
    }

    fn read_ids(&mut self, n: usize) -> io::Result<Vec<u16>> {
        let w = usize::from(self.header.id_bytes);
        let mut raw = vec![0u8; n * w];
        self.input.read_exact(&mut raw)?;
        let ids: Vec<u16> = if w == 1 {
            raw.iter().map(|&b| u16::from(b)).collect()
        } else {
            raw.chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect()
        };
        if ids.iter().any(|&e| u32::from(e) >= self.header.n_expert) {
            return Err(bad("expert id out of range".into()));
        }
        Ok(ids)
    }

    fn read_u32(&mut self) -> io::Result<u32> {
        let mut b = [0u8; 4];
        self.input.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    fn record(&mut self, kind: u8) -> io::Result<Record> {
        let step = self.read_u32()?;
        match kind {
            KIND_REFRESH => Ok(Record::HotRefresh { step }),
            KIND_STEP => {
                let mut b = [0u8; 2];
                self.input.read_exact(&mut b)?;
                let layer = u16::from_le_bytes(b);
                let n_tokens = self.read_u32()?;
                let mut f = [0u8; 1];
                self.input.read_exact(&mut f)?;
                let n = n_tokens as usize * self.header.n_used as usize;
                let ids = self.read_ids(n)?;
                let predicted = (f[0] & FLAG_PREDICTED != 0)
                    .then(|| self.read_ids(n))
                    .transpose()?;
                let predicted2 = (f[0] & FLAG_PREDICTED2 != 0)
                    .then(|| self.read_ids(n))
                    .transpose()?;
                Ok(Record::Step(LayerStep {
                    step,
                    layer,
                    n_tokens,
                    ids,
                    predicted,
                    predicted2,
                }))
            }
            k => Err(bad(format!("unknown record kind {k}"))),
        }
    }
}

impl<R: Read> Iterator for TraceReader<R> {
    type Item = io::Result<Record>;

    /// A clean end of file at a record boundary ends the iteration; anywhere else it is an
    /// error (a truncated trace).
    fn next(&mut self) -> Option<Self::Item> {
        let mut kind = [0u8; 1];
        loop {
            match self.input.read(&mut kind) {
                Ok(0) => return None,
                Ok(_) => break,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Some(Err(e)),
            }
        }
        Some(self.record(kind[0]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(n_expert: u32) -> TraceHeader {
        let mut h = TraceHeader::new(3, n_expert, 2);
        h.model = "test".into();
        h.vram_slots = 4;
        h.ram_slots = 8;
        h.expert_bytes = vec![100, 200, 300];
        h
    }

    fn write_sample(n_expert: u32) -> Vec<u8> {
        let big = n_expert as i32 - 1;
        let mut w = TraceWriter::new(Vec::new(), header(n_expert)).unwrap();
        w.layer_step(0, 0, 2, &[0, big, 1, 2], Some(&[3, 4, 5, 6]), None)
            .unwrap();
        w.layer_step(0, 1, 2, &[7, 8, 9, big], None, Some(&[1, 1, 2, 2]))
            .unwrap();
        w.hot_refresh(0).unwrap();
        w.layer_step(1, 2, 1, &[5, big], Some(&[1, 2]), Some(&[big, 0]))
            .unwrap();
        w.out
    }

    fn roundtrip(n_expert: u32) {
        let bytes = write_sample(n_expert);
        let big = (n_expert - 1) as u16;
        let r = TraceReader::new(bytes.as_slice()).unwrap();
        assert_eq!(r.header(), &header(n_expert));
        assert_eq!(r.header().id_bytes, if n_expert <= 256 { 1 } else { 2 });
        let recs: Vec<Record> = r.map(Result::unwrap).collect();
        assert_eq!(recs.len(), 4);
        assert_eq!(
            recs[0],
            Record::Step(LayerStep {
                step: 0,
                layer: 0,
                n_tokens: 2,
                ids: vec![0, big, 1, 2],
                predicted: Some(vec![3, 4, 5, 6]),
                predicted2: None,
            })
        );
        assert!(matches!(&recs[1], Record::Step(s) if s.predicted.is_none()
            && s.predicted2 == Some(vec![1, 1, 2, 2])));
        assert_eq!(recs[2], Record::HotRefresh { step: 0 });
        assert!(
            matches!(&recs[3], Record::Step(s) if s.step == 1 && s.layer == 2
            && s.predicted2 == Some(vec![big, 0]))
        );
    }

    #[test]
    fn roundtrip_u8_ids() {
        roundtrip(128);
    }

    #[test]
    fn roundtrip_u16_ids() {
        roundtrip(1000);
    }

    #[test]
    fn truncated_file_is_an_error_not_a_panic() {
        let bytes = write_sample(128);
        let full = TraceReader::new(bytes.as_slice()).unwrap().count();
        let header_len = bytes.iter().position(|&b| b == b'\n').unwrap() + 1;
        let mut errors = 0;
        for cut in 0..bytes.len() {
            match TraceReader::new(&bytes[..cut]) {
                // A cut inside the header fails to open.
                Err(_) => assert!(cut < header_len),
                Ok(r) => {
                    let got: Vec<_> = r.collect();
                    if got.iter().any(Result::is_err) {
                        errors += 1;
                    } else {
                        // A cut on a record boundary: fewer records, never more.
                        assert!(got.len() < full);
                    }
                }
            }
        }
        assert!(errors > 0);
    }

    #[test]
    fn rejects_bad_ids_and_lengths() {
        let mut w = TraceWriter::new(Vec::new(), header(16)).unwrap();
        assert!(w.layer_step(0, 0, 1, &[0, 16], None, None).is_err());
        assert!(w.layer_step(0, 0, 1, &[-1, 2], None, None).is_err());
        assert!(w.layer_step(0, 0, 1, &[0, 1, 2], None, None).is_err());
    }

    #[test]
    fn rejects_foreign_headers() {
        assert!(TraceReader::new(&b"{}\n"[..]).is_err());
        assert!(TraceReader::new(&b"not json\n"[..]).is_err());
    }
}
