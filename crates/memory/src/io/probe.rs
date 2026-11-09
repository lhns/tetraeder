//! Measures a disk's read rate at one queue depth through the I/O service (ADR 0056,
//! `tetraeder-bench disk-curve`).
//!
//! `depth` workers each keep one chunk in flight, closed loop: when a chunk lands the next one,
//! at a random 4 KiB aligned offset of the file, is queued into the same buffer. The rate is the
//! bytes landed over the elapsed time; the latency of a chunk is the unit's time from submit to
//! landing (queueing included, so mean latency is `depth * chunk / rate` by Little's law).
//! The destinations are plain buffers owned here, which is why this lives beside `HostDst`.

use super::service::{IoService, ServiceOptions};
use super::{Deadline, Part, UnitId};
use crate::dio::{ALIGN, Range};
use crate::reader::HostDst;
use std::io;
use std::path::Path;
use std::time::{Duration, Instant};

/// One queue depth of the curve.
#[derive(Debug, Clone, PartialEq)]
pub struct DepthPoint {
    pub depth: usize,
    pub chunks: u64,
    pub bytes_per_s: f64,
    pub p50_s: f64,
    pub p99_s: f64,
}

/// Tiny deterministic generator (xorshift64).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Reads random aligned chunks of `chunk_bytes` from `path` for `seconds` at queue depth `depth`.
/// `buffered`: through the page cache (a filesystem that refuses direct I/O, tests). Errors when
/// the file is smaller than a chunk.
pub fn measure_depth(
    path: &Path,
    depth: usize,
    chunk_bytes: usize,
    seconds: f64,
    buffered: bool,
) -> io::Result<DepthPoint> {
    let depth = depth.max(1);
    let chunk = chunk_bytes.max(1).next_multiple_of(ALIGN);
    let size = std::fs::metadata(path)?.len();
    if size < chunk as u64 {
        return Err(io::Error::other(format!(
            "{} is smaller than one chunk of {chunk} bytes",
            path.display()
        )));
    }
    let svc = IoService::new(
        &[path.to_path_buf()],
        ServiceOptions {
            workers: depth,
            chunk_bytes: chunk,
            buffered,
        },
    )?;
    // Offsets are multiples of ALIGN with the whole chunk inside the file.
    let offsets = (size - chunk as u64) / ALIGN as u64 + 1;
    let mut rng = Rng((0x9e37_79b9_7f4a_7c15 ^ ((depth as u64) << 32)) | 1);
    let mut bufs: Vec<Vec<u8>> = (0..depth).map(|_| vec![0u8; chunk]).collect();
    let mut seq = 0u64;
    let mut submit = |svc: &IoService, buf: &mut [u8], rng: &mut Rng| -> UnitId {
        seq += 1;
        svc.submit(
            Deadline {
                order: seq,
                urgent: false,
            },
            vec![Part {
                shard: 0,
                range: Range {
                    offset: (rng.next() % offsets) * ALIGN as u64,
                    len: chunk,
                },
                // SAFETY: `buf` is a Vec of exactly `chunk` bytes that is neither moved, resized
                // nor touched until the unit has been collected below (or the service dropped).
                dst: unsafe { HostDst::new(buf.as_mut_ptr(), buf.len()) },
            }],
        )
    };
    let mut open: Vec<(UnitId, usize)> = Vec::with_capacity(depth);
    for (i, b) in bufs.iter_mut().enumerate() {
        open.push((submit(&svc, b, &mut rng), i));
    }
    let start = Instant::now();
    let limit = Duration::from_secs_f64(seconds.max(0.0));
    let mut took: Vec<f64> = Vec::new();
    let mut failure = None;
    while start.elapsed() < limit && failure.is_none() {
        let ids: Vec<UnitId> = open.iter().map(|o| o.0).collect();
        for (id, res) in svc.wait_any(&ids) {
            let Some(pos) = open.iter().position(|o| o.0 == id) else {
                continue;
            };
            let buf = open[pos].1;
            match res {
                Ok(d) => {
                    took.push(d.took.as_secs_f64());
                    open[pos].0 = submit(&svc, &mut bufs[buf], &mut rng);
                }
                Err(e) => {
                    failure.get_or_insert(e);
                    open.swap_remove(pos);
                }
            }
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    // Land what is still in flight before the buffers go.
    let rest: Vec<UnitId> = open.iter().map(|o| o.0).collect();
    let _ = svc.wait_all(&rest);
    if let Some(e) = failure {
        return Err(e);
    }
    took.sort_by(f64::total_cmp);
    let at = |q: f64| {
        took.get(((took.len() as f64 * q) as usize).min(took.len().saturating_sub(1)))
            .copied()
            .unwrap_or(0.0)
    };
    Ok(DepthPoint {
        depth,
        chunks: took.len() as u64,
        bytes_per_s: took.len() as f64 * chunk as f64 / elapsed.max(1e-9),
        p50_s: at(0.5),
        p99_s: at(0.99),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn file(len: usize) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("tetraeder-probe-{}", std::process::id()));
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&vec![7u8; len])
            .unwrap();
        path
    }

    #[test]
    fn a_depth_reads_and_reports_a_positive_rate() {
        let path = file(1 << 20);
        for depth in [1, 4] {
            let p = measure_depth(&path, depth, 64 << 10, 0.15, true).unwrap();
            assert_eq!(p.depth, depth);
            assert!(p.chunks > 0, "{p:?}");
            assert!(p.bytes_per_s > 0.0, "{p:?}");
            assert!(p.p50_s <= p.p99_s, "{p:?}");
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_file_smaller_than_a_chunk_is_an_error() {
        let path =
            std::env::temp_dir().join(format!("tetraeder-probe-small-{}", std::process::id()));
        std::fs::write(&path, [1u8; 100]).unwrap();
        assert!(measure_depth(&path, 1, 4096, 0.1, true).is_err());
        let _ = std::fs::remove_file(path);
    }
}
