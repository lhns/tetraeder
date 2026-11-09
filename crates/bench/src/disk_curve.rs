//! `tetraeder-bench disk-curve`: a disk's read rate by queue depth (ADR 0056, section 4).
//!
//! Random 4 KiB aligned chunks of one file through the I/O service with `workers = depth`,
//! direct I/O (the same reads the builder makes). Per depth: bytes per second and the p50 and
//! p99 latency of a chunk. With `--load-threads N` the same runs beside N spinning threads, the
//! starvation case of ADR 0054. The output carries `provenance` and a `queue_curve` to paste
//! into a hardware graph as the disk's curve.

use crate::env;
use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tetraeder_memory::io::probe::{DepthPoint, measure_depth};

pub struct Args<'a> {
    pub file: &'a Path,
    pub chunk_bytes: usize,
    pub depths: &'a [usize],
    pub seconds: f64,
    pub load_threads: usize,
    /// Read through the page cache (tests, filesystems that refuse direct I/O). Without it a
    /// direct open that fails falls back to buffered and the output says so.
    pub buffered: bool,
}

/// Spinning threads that stop when dropped.
struct Load {
    stop: Arc<AtomicBool>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

impl Load {
    fn start(n: usize) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let threads = (0..n)
            .map(|_| {
                let stop = Arc::clone(&stop);
                std::thread::spawn(move || {
                    let mut x = 1u64;
                    while !stop.load(Ordering::Relaxed) {
                        for _ in 0..10_000 {
                            x = std::hint::black_box(
                                x.wrapping_mul(6364136223846793005).wrapping_add(1),
                            );
                        }
                    }
                })
            })
            .collect();
        Self { stop, threads }
    }
}

impl Drop for Load {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

/// Runs the curve and returns the output document.
pub fn measure(a: &Args) -> Result<Value> {
    if a.depths.is_empty() {
        bail!("no queue depths given");
    }
    let _load = (a.load_threads > 0).then(|| Load::start(a.load_threads));
    let mut buffered = a.buffered;
    let mut points: Vec<DepthPoint> = Vec::new();
    for &depth in a.depths {
        let run = |buffered| measure_depth(a.file, depth, a.chunk_bytes, a.seconds, buffered);
        let p = match run(buffered) {
            Ok(p) => p,
            Err(e) if !buffered => {
                eprintln!("disk-curve: direct reads failed ({e}); falling back to buffered");
                buffered = true;
                run(true).with_context(|| format!("read {}", a.file.display()))?
            }
            Err(e) => return Err(e).with_context(|| format!("read {}", a.file.display())),
        };
        points.push(p);
    }
    let chunk = a.chunk_bytes.max(1).next_multiple_of(4096);
    Ok(json!({
        "tool": "tetraeder-bench disk-curve",
        "adr": "0056",
        "file": a.file.display().to_string(),
        "chunk_bytes": chunk,
        "seconds": a.seconds,
        "load_threads": a.load_threads,
        "buffered": buffered,
        "provenance": env::capture(),
        "points": points.iter().map(|p| json!({
            "depth": p.depth,
            "chunks": p.chunks,
            "bytes_per_s": p.bytes_per_s,
            "p50_ms": p.p50_s * 1e3,
            "p99_ms": p.p99_s * 1e3,
        })).collect::<Vec<_>>(),
        "queue_chunk_bytes": chunk,
        "queue_curve": points.iter().map(|p| json!({
            "depth": p.depth,
            "bytes_per_s": p.bytes_per_s,
        })).collect::<Vec<_>>(),
    }))
}

/// [`measure`] and write the document to `out`.
pub fn run(a: &Args, out: &Path) -> Result<()> {
    let doc = measure(a)?;
    if let Some(dir) = out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(out, serde_json::to_string_pretty(&doc)?)
        .with_context(|| format!("write {}", out.display()))
}
