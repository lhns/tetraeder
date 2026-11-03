//! Load-time format conversion with a per-tensor disk cache (ADR 0047).
//!
//! A tensor whose target format differs from its type in the file is dequantized to f32 with
//! ggml's type traits and quantized with `ggml_quantize_chunk`, rows split over threads. The
//! result goes to a cache file named by a hash of the source, the tensor, the target format, the
//! importance matrix (none yet) and the ggml pin. The tensor index then points at the cache file
//! as an extra shard, so loading and expert streaming read it like any other shard.
//!
//! Cache layout: `<dir>/<64 hex digits>.bin`, the raw bytes of one tensor in the target type
//! (no header), written to `<key>.<pid>.tmp` and renamed.

use crate::gguf::{Gguf, GgufError, TensorInfo};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, UNIX_EPOCH};
use tetraeder_ggml_sys::ggml::{self, GgmlError, Type};
use tetraeder_plan::Format;
use tetraeder_telemetry::Telemetry;

/// The ggml pin the cache keys include: a new pin may change a quantizer's output.
const PIN: &str = include_str!("../../../vendor/PIN");

/// Working set of one chunk: bytes of f32 rows across all threads.
const CHUNK_F32_BYTES: usize = 64 << 20;

/// The ggml type of a plan format, if this ggml build has it.
pub fn ggml_type(f: Format) -> Option<Type> {
    Type::from_name(&format!("{f:?}"))
}

/// The plan format of a ggml type, if the plan knows it.
pub fn format_of(t: Type) -> Option<Format> {
    let name = t.name();
    Format::ALL
        .into_iter()
        .find(|f| format!("{f:?}").eq_ignore_ascii_case(&name))
}

#[derive(Debug, thiserror::Error)]
pub enum ConvertError {
    #[error("tensor `{0}` is not in the model")]
    Unknown(String),
    #[error("cannot convert `{tensor}`: {why}")]
    Refused { tensor: String, why: String },
    #[error(transparent)]
    Gguf(#[from] GgufError),
    #[error(transparent)]
    Ggml(#[from] GgmlError),
    #[error("conversion I/O on `{path}`: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
}

type Result<T> = std::result::Result<T, ConvertError>;

fn io(path: &Path) -> impl Fn(std::io::Error) -> ConvertError + '_ {
    move |source| ConvertError::Io {
        path: path.display().to_string(),
        source,
    }
}

#[derive(Clone, Default)]
pub struct ConvertOptions {
    /// Cache directory; `None`: see [`cache_dir`].
    pub cache: Option<PathBuf>,
    /// Worker threads (at least 1).
    pub threads: usize,
    /// Conversion metrics go here when set.
    pub telemetry: Option<Arc<Telemetry>>,
}

/// What one [`convert`] call did.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ConvertStats {
    /// Tensors computed (cache misses).
    pub converted: u64,
    /// Tensors found in the cache.
    pub cache_hits: u64,
    /// Bytes written to the cache.
    pub bytes_written: u64,
    /// Wall time of the call.
    pub seconds: f64,
}

/// The cache directory: `explicit`, else `$TETRAEDER_CACHE_DIR`, else `.tetraeder-cache` next to
/// the model's first shard.
pub fn cache_dir(explicit: Option<&Path>, first_shard: &Path) -> PathBuf {
    if let Some(d) = explicit {
        return d.to_path_buf();
    }
    if let Some(d) = std::env::var_os("TETRAEDER_CACHE_DIR").filter(|d| !d.is_empty()) {
        return PathBuf::from(d);
    }
    first_shard
        .parent()
        .unwrap_or(Path::new(""))
        .join(".tetraeder-cache")
}

struct Job {
    src: TensorInfo,
    format: Format,
    ty: Type,
}

fn refuse(tensor: &str, why: impl Into<String>) -> ConvertError {
    ConvertError::Refused {
        tensor: tensor.into(),
        why: why.into(),
    }
}

/// Checks every entry and returns the jobs whose target differs from the file's type.
fn plan_jobs(g: &Gguf, formats: &BTreeMap<String, Format>) -> Result<Vec<Job>> {
    let mut jobs = Vec::new();
    for (name, &format) in formats {
        let t = g
            .tensor(name)
            .map_err(|_| ConvertError::Unknown(name.clone()))?;
        let ty = ggml_type(format)
            .ok_or_else(|| refuse(name, format!("ggml has no type {format:?}")))?;
        if ty == t.ty {
            continue;
        }
        if t.ne[1] * t.ne[2] * t.ne[3] == 1 {
            return Err(refuse(name, "1-D tensors keep their format"));
        }
        if name.contains("ffn_gate_inp") {
            return Err(refuse(name, "the router keeps its format"));
        }
        let from = format_of(t.ty).ok_or_else(|| {
            refuse(
                name,
                format!("source type {} is not a plan format", t.ty.name()),
            )
        })?;
        if !from.at_least(format) {
            return Err(refuse(
                name,
                format!("{format:?} is more precise than the source {from:?}"),
            ));
        }
        if ty.requires_imatrix() {
            return Err(refuse(
                name,
                format!("{format:?} needs an importance matrix"),
            ));
        }
        if t.ne[0] % ty.block_size() as u64 != 0 {
            return Err(refuse(
                name,
                format!(
                    "row length {} is not a multiple of the {format:?} block size {}",
                    t.ne[0],
                    ty.block_size()
                ),
            ));
        }
        jobs.push(Job {
            src: t.clone(),
            format,
            ty,
        });
    }
    Ok(jobs)
}

/// SHA-256 over each shard's header bytes (up to its data section), length and modification
/// time: a copied or re-downloaded file converts again.
fn fingerprint(g: &Gguf) -> Result<[u8; 32]> {
    let mut h = Sha256::new();
    for (path, &data) in g.paths.iter().zip(&g.data_offsets) {
        let mut f = File::open(path).map_err(io(path))?;
        let meta = f.metadata().map_err(io(path))?;
        let mut head = vec![0u8; data as usize];
        f.read_exact(&mut head).map_err(io(path))?;
        let mtime = meta
            .modified()
            .ok()
            .and_then(|m| m.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_nanos());
        h.update((head.len() as u64).to_le_bytes());
        h.update(&head);
        h.update(meta.len().to_le_bytes());
        h.update(mtime.to_le_bytes());
    }
    Ok(h.finalize().into())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The cache key of one converted tensor.
fn cache_key(fingerprint: &[u8; 32], tensor: &str, format: Format) -> String {
    let pin = PIN
        .lines()
        .find_map(|l| l.trim().strip_prefix("commit="))
        .expect("vendor/PIN has a commit= line");
    let mut h = Sha256::new();
    for part in [
        fingerprint.as_slice(),
        tensor.as_bytes(),
        format!("{format:?}").as_bytes(),
        b"imatrix:none",
        pin.as_bytes(),
    ] {
        h.update((part.len() as u64).to_le_bytes());
        h.update(part);
    }
    hex(&h.finalize())
}

/// Converts the tensors of `formats` whose target differs from their type, through the cache,
/// and points `g` at the results. Everything is checked before the first conversion. Call it once
/// per index: the source fingerprint covers the index's shards as they are at the call.
pub fn convert(
    g: &mut Gguf,
    formats: &BTreeMap<String, Format>,
    opts: &ConvertOptions,
) -> Result<ConvertStats> {
    let start = Instant::now();
    let jobs = plan_jobs(g, formats)?;
    let mut stats = ConvertStats::default();
    if jobs.is_empty() {
        return Ok(stats);
    }
    let dir = cache_dir(opts.cache.as_deref(), &g.paths[0]);
    let dir = std::path::absolute(&dir).unwrap_or(dir);
    let fp = fingerprint(g)?;
    let threads = opts.threads.max(1);
    for job in jobs {
        let name = job.src.name.as_str();
        let rows = job.src.ne[1] * job.src.ne[2] * job.src.ne[3];
        let nbytes = job.ty.row_size(job.src.ne[0] as i64) as u64 * rows;
        let path = dir.join(format!("{}.bin", cache_key(&fp, name, job.format)));
        let hit = path.metadata().is_ok_and(|m| m.len() == nbytes);
        if !hit {
            let t0 = Instant::now();
            std::fs::create_dir_all(&dir).map_err(io(&dir))?;
            convert_to_file(&g.paths[job.src.shard], &job, threads, &path)?;
            let d = t0.elapsed();
            stats.converted += 1;
            stats.bytes_written += nbytes;
            tracing::debug!(tensor = name, format = ?job.format, secs = d.as_secs_f64(), "converted");
            if let Some(t) = &opts.telemetry {
                t.metrics()
                    .convert_duration(d, &format!("{:?}", job.format));
                t.metrics().convert_bytes(nbytes);
            }
        } else {
            stats.cache_hits += 1;
        }
        if let Some(t) = &opts.telemetry {
            t.metrics().convert_cache(hit);
        }
        g.retarget(name, job.ty, path)?;
    }
    stats.seconds = start.elapsed().as_secs_f64();
    tracing::info!(
        converted = stats.converted,
        cache_hits = stats.cache_hits,
        bytes_written = stats.bytes_written,
        seconds = stats.seconds,
        threads,
        "format conversion"
    );
    Ok(stats)
}

/// Converts one tensor from `src` into `path` (through a temporary file).
fn convert_to_file(src: &Path, job: &Job, threads: usize, path: &Path) -> Result<()> {
    let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
    let res = (|| {
        let mut out = File::create(&tmp).map_err(io(&tmp))?;
        convert_rows(src, job, threads, &mut out, &tmp)?;
        out.sync_all().map_err(io(&tmp))?;
        drop(out);
        std::fs::rename(&tmp, path).map_err(io(path))
    })();
    if res.is_err() {
        std::fs::remove_file(&tmp).ok();
    }
    res
}

/// Streams the tensor in chunks of rows: read, convert on `threads` threads, write.
fn convert_rows(src: &Path, job: &Job, threads: usize, out: &mut File, dst: &Path) -> Result<()> {
    let t = &job.src;
    let n_per_row = t.ne[0] as usize;
    let rows = (t.ne[1] * t.ne[2] * t.ne[3]) as usize;
    let (src_row, dst_row) = (
        t.ty.row_size(n_per_row as i64),
        job.ty.row_size(n_per_row as i64),
    );
    let chunk = (CHUNK_F32_BYTES / (4 * n_per_row)).max(threads).min(rows);
    let mut f = File::open(src).map_err(io(src))?;
    f.seek(SeekFrom::Start(t.offset)).map_err(io(src))?;
    let mut src_buf = vec![0u8; chunk * src_row];
    let mut dst_buf = vec![0u8; chunk * dst_row];
    let mut done = 0;
    while done < rows {
        let n = chunk.min(rows - done);
        f.read_exact(&mut src_buf[..n * src_row]).map_err(io(src))?;
        let per = n.div_ceil(threads);
        let work = |s: &[u8], d: &mut [u8]| -> Result<()> {
            let r = s.len() / src_row;
            let mut f32s = vec![0f32; r * n_per_row];
            ggml::to_f32(t.ty, s, &mut f32s)?;
            ggml::quantize(job.ty, &f32s, r, n_per_row, None, d)?;
            Ok(())
        };
        let pieces = src_buf[..n * src_row]
            .chunks(per * src_row)
            .zip(dst_buf[..n * dst_row].chunks_mut(per * dst_row));
        let work = &work;
        std::thread::scope(|sc| {
            let handles: Vec<_> = pieces.map(|(s, d)| sc.spawn(move || work(s, d))).collect();
            handles
                .into_iter()
                .try_for_each(|h| h.join().expect("conversion thread panicked"))
        })?;
        out.write_all(&dst_buf[..n * dst_row]).map_err(io(dst))?;
        done += n;
    }
    Ok(())
}
