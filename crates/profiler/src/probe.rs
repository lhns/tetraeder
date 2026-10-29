//! Bandwidth probes. Each runs `repeats` times and reports the median and spread.

use crate::graph::Measurement;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

const CHUNK: usize = 1 << 20;
const ALIGN: usize = 4096;

/// Removes the probe file on drop, also on error paths.
struct TempFile(PathBuf);

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Opens `path` for unbuffered reads (page cache bypassed).
fn open_direct(path: &Path) -> std::io::Result<File> {
    let mut o = OpenOptions::new();
    o.read(true);
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.custom_flags(libc::O_DIRECT);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
        o.custom_flags(FILE_FLAG_NO_BUFFERING);
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        return Err(std::io::Error::other(
            "direct reads not implemented on this OS",
        ));
    }
    #[allow(unreachable_code)]
    o.open(path)
}

/// Sequential-read bandwidth (bytes per second) of the volume holding `dir`.
///
/// Writes a `bytes`-sized temp file (opt-in at the caller), then reads it `repeats` times with
/// O_DIRECT on Linux and FILE_FLAG_NO_BUFFERING on Windows. Fails on filesystems without direct IO
/// (tmpfs), which the caller records as a note.
pub fn probe_disk_read(dir: &Path, bytes: u64, repeats: usize) -> Result<Measurement, String> {
    let chunks = (bytes as usize).div_ceil(CHUNK).max(1);
    static SEQ: AtomicU32 = AtomicU32::new(0);
    let path = dir.join(format!(
        "tetraeder-probe-{}-{}.tmp",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let guard = TempFile(path.clone());
    let err = |what: &str, e: std::io::Error| format!("{what} {}: {e}", path.display());

    {
        let mut f = File::create(&path).map_err(|e| err("create", e))?;
        let mut buf = vec![0u8; CHUNK];
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        for _ in 0..chunks {
            // Incompressible content, so compressing filesystems and controllers cannot cheat.
            for w in buf.chunks_exact_mut(8) {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                w.copy_from_slice(&x.to_le_bytes());
            }
            f.write_all(&buf).map_err(|e| err("write", e))?;
        }
        f.sync_all().map_err(|e| err("sync", e))?;
    }

    let mut raw = vec![0u8; CHUNK + ALIGN];
    let off = raw.as_ptr().align_offset(ALIGN);
    let buf = &mut raw[off..off + CHUNK];
    let total = (chunks * CHUNK) as f64;
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats.max(1) {
        let mut f = open_direct(&path).map_err(|e| err("open direct", e))?;
        let t = Instant::now();
        let mut got = 0usize;
        loop {
            let n = f.read(buf).map_err(|e| err("read", e))?;
            if n == 0 {
                break;
            }
            got += n;
        }
        let secs = t.elapsed().as_secs_f64();
        if got == 0 || secs <= 0.0 {
            return Err(format!("empty read of {}", path.display()));
        }
        samples.push(got.min(total as usize) as f64 / secs);
    }
    drop(guard);
    Measurement::from_samples(&samples).ok_or_else(|| "no samples".into())
}

/// Aggregate sequential read bandwidth of RAM: `threads` threads each sum their own buffer.
pub fn probe_ram_read(
    threads: usize,
    bytes_per_thread: usize,
    repeats: usize,
) -> Option<Measurement> {
    let threads = threads.max(1);
    let words = (bytes_per_thread / 8).max(1);
    let bufs: Vec<Vec<u64>> = (0..threads).map(|i| vec![i as u64 + 1; words]).collect();
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats.max(1) {
        let t = Instant::now();
        let sum: u64 = std::thread::scope(|s| {
            let hs: Vec<_> = bufs
                .iter()
                .map(|b| s.spawn(move || b.iter().fold(0u64, |a, &v| a.wrapping_add(v))))
                .collect();
            hs.into_iter()
                .map(|h| h.join().unwrap_or(0))
                .fold(0u64, u64::wrapping_add)
        });
        std::hint::black_box(sum);
        let secs = t.elapsed().as_secs_f64();
        if secs > 0.0 {
            samples.push((threads * words * 8) as f64 / secs);
        }
    }
    Measurement::from_samples(&samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ram_probe_reports_positive_bandwidth() {
        let m = probe_ram_read(2, 4 << 20, 3).unwrap();
        assert!(m.value > 1e7, "{m:?}");
        assert_eq!(m.samples, 3);
    }

    #[test]
    fn disk_probe_cleans_up_and_reports_or_explains() {
        let dir = std::env::temp_dir().join(format!("tetraeder-probe-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        match probe_disk_read(&dir, 8 << 20, 3) {
            Ok(m) => {
                assert!(m.value > 1e6, "{m:?}");
                assert_eq!(m.samples, 3);
            }
            // tmpfs has no O_DIRECT; the error must say what failed.
            Err(e) => assert!(e.contains("tetraeder-probe"), "{e}"),
        }
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
        std::fs::remove_dir(&dir).unwrap();
    }
}
