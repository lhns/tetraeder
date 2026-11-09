//! Measured noise bands per device and GPU class, from the A/A (`role = "noise"`) pairs of committed results.
//!
//! `tetraeder-bench noise update` scans `bench/results` and writes `bench/noise/<class>.json`. A run reads the file
//! of its class and uses its band when it has enough sessions; otherwise it falls back to the spec's provisional
//! `[[noise_band]]` and records which one it used (`band_source` in each pair result).
//!
//! One session is one results file with an A/A pair. Its statistic is `d = max over metrics |median paired ratio - 1|`,
//! the quantity the gate compares with the band. The band is `clamp(margin * p95(d), floor, ceil)` over the sessions,
//! p95 by nearest rank (so the maximum below 20 sessions).

use crate::run::{PairResult, Results, Status};
use crate::spec::{Device, Role};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const SCHEMA: u32 = 1;
/// Fewer sessions than this: the measured file is ignored and the provisional band stays.
pub const MIN_SESSIONS: usize = 3;
pub const QUANTILE: f64 = 0.95;
pub const MARGIN: f64 = 1.25;
pub const FLOOR: f64 = 0.02;
pub const CEIL: f64 = 0.25;

/// One A/A pair of one results file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Session {
    /// Repo-relative path of the results file; the key that makes `update` idempotent.
    pub file: String,
    pub date: String,
    pub host_id: String,
    pub gpu: String,
    pub driver: String,
    pub pair: String,
    pub threads: u32,
    pub repeats: usize,
    /// Median paired ratio per metric.
    pub ratios: BTreeMap<String, f64>,
    /// `max |ratio - 1|` over the metrics.
    pub d: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NoiseFile {
    pub schema: u32,
    pub class: String,
    pub device: Device,
    pub updated: String,
    pub quantile: f64,
    pub margin: f64,
    pub floor: f64,
    pub ceil: f64,
    /// Sessions behind the band.
    pub n: usize,
    pub n_hosts: usize,
    /// p95 of `d` over the sessions (nearest rank).
    pub d_p95: f64,
    /// The allowed drop of a gate's paired ratio (and the +- window of an A/A pair).
    pub band: f64,
    pub sessions: Vec<Session>,
}

/// The band a run uses and where it came from.
#[derive(Debug, Clone, PartialEq)]
pub struct Chosen {
    pub class: String,
    pub band: f64,
    /// `measured <class> n=<n>` or `provisional <class>` (+ why the measured one was not used).
    pub source: String,
}

/// Class of a device: `cpu`, or the GPU name without vendor words, e.g. "NVIDIA GeForce RTX 3090" -> `rtx-3090`.
pub fn class_slug(device: Device, gpu_name: Option<&str>) -> String {
    if device != Device::Cuda {
        return "cpu".into();
    }
    let lower = gpu_name.unwrap_or("").to_lowercase();
    let words: Vec<&str> = lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty() && !matches!(*w, "nvidia" | "geforce"))
        .collect();
    if words.is_empty() {
        "cuda-unknown".into()
    } else {
        words.join("-")
    }
}

/// Nearest-rank quantile of a non-empty slice.
fn nearest_rank(v: &[f64], q: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let rank = ((q * s.len() as f64).ceil() as usize).clamp(1, s.len());
    s[rank - 1]
}

/// Band and p95 of `d` for a set of sessions; `None` without sessions.
pub fn band_of(sessions: &[Session]) -> Option<(f64, f64)> {
    if sessions.is_empty() {
        return None;
    }
    let d: Vec<f64> = sessions.iter().map(|s| s.d).collect();
    let p95 = nearest_rank(&d, QUANTILE);
    Some(((MARGIN * p95).clamp(FLOOR, CEIL), p95))
}

pub fn file_path(dir: &Path, class: &str) -> PathBuf {
    dir.join(format!("{class}.json"))
}

/// The measured file of a class: `Ok(None)` when absent, an error when it exists but cannot be used (a corrupt
/// committed file must not silently turn the gate back to the provisional band).
pub fn load(dir: &Path, class: &str) -> Result<Option<NoiseFile>> {
    let path = file_path(dir, class);
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("read {}", path.display())),
    };
    let f: NoiseFile =
        serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
    ensure!(
        f.schema == SCHEMA,
        "{}: schema {}",
        path.display(),
        f.schema
    );
    ensure!(
        f.class == class && f.band.is_finite() && f.band > 0.0 && f.band < 1.0,
        "{}: class or band out of range",
        path.display()
    );
    Ok(Some(f))
}

/// Measured band when the file has `MIN_SESSIONS` sessions, else the provisional one (`None` if the spec has none).
pub fn choose(
    class: &str,
    provisional: Option<(&str, f64)>,
    measured: Option<&NoiseFile>,
) -> Option<Chosen> {
    match measured {
        Some(f) if f.n >= MIN_SESSIONS => Some(Chosen {
            class: class.to_string(),
            band: f.band,
            source: format!("measured {class} n={}", f.n),
        }),
        other => {
            let (spec_class, band) = provisional?;
            let why = match other {
                Some(f) => format!(" (measured n={} < {MIN_SESSIONS})", f.n),
                None => String::new(),
            };
            Some(Chosen {
                class: spec_class.to_string(),
                band,
                source: format!("provisional {spec_class}{why}"),
            })
        }
    }
}

/// Sessions of one results file, keyed by class. Skipped and failed pairs carry no ratio and are left out.
pub fn sessions_of(r: &Results, file: &str) -> Vec<(String, Device, Session)> {
    let gpu = r.env.gpus.first();
    r.pairs
        .iter()
        .filter(|p| p.role == Role::Noise && matches!(p.status, Status::Ok | Status::Noisy))
        .filter_map(|p| {
            session_of(
                r,
                p,
                file,
                gpu.map(|g| (g.name.as_str(), g.driver.as_str())),
            )
        })
        .collect()
}

fn session_of(
    r: &Results,
    p: &PairResult,
    file: &str,
    gpu: Option<(&str, &str)>,
) -> Option<(String, Device, Session)> {
    let ratios: BTreeMap<String, f64> = p
        .metrics
        .iter()
        .map(|(k, m)| (k.clone(), m.ratio.median))
        .filter(|(_, v)| v.is_finite())
        .collect();
    let d = ratios.values().map(|v| (v - 1.0).abs()).reduce(f64::max)?;
    let (gpu_name, driver) = gpu.unwrap_or(("", ""));
    let threads = p.samples.first().map_or(0, |s| s.threads);
    Some((
        class_slug(p.device, Some(gpu_name)),
        p.device,
        Session {
            file: file.to_string(),
            date: r.env.started_utc.get(..10).unwrap_or("").to_string(),
            host_id: r.env.host_id.clone(),
            gpu: gpu_name.to_string(),
            driver: driver.to_string(),
            pair: p.id.clone(),
            threads,
            repeats: r.repeats,
            ratios,
            d,
        },
    ))
}

/// Merge new sessions into a class file (existing sessions stay, a session already present by file and pair is not
/// added twice) and recompute the band.
pub fn merge(
    old: Option<NoiseFile>,
    class: &str,
    device: Device,
    new: Vec<Session>,
    updated: &str,
) -> NoiseFile {
    let mut sessions = old.map(|f| f.sessions).unwrap_or_default();
    for s in new {
        if !sessions
            .iter()
            .any(|o| o.file == s.file && o.pair == s.pair)
        {
            sessions.push(s);
        }
    }
    sessions.sort_by(|a, b| (&a.date, &a.file, &a.pair).cmp(&(&b.date, &b.file, &b.pair)));
    let (band, d_p95) = band_of(&sessions).unwrap_or((FLOOR, 0.0));
    let mut hosts: Vec<&str> = sessions.iter().map(|s| s.host_id.as_str()).collect();
    hosts.sort_unstable();
    hosts.dedup();
    NoiseFile {
        schema: SCHEMA,
        class: class.to_string(),
        device,
        updated: updated.to_string(),
        quantile: QUANTILE,
        margin: MARGIN,
        floor: FLOOR,
        ceil: CEIL,
        n: sessions.len(),
        n_hosts: hosts.len(),
        d_p95,
        band,
        sessions,
    }
}

fn json_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            json_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "json") {
            out.push(p);
        }
    }
}

/// Scan `results` for harness results with A/A pairs and write one file per class into `out`. `rel_to` is the
/// directory the stored file names are relative to (the repo root). Returns the files written.
pub fn update(results: &Path, out: &Path, rel_to: &Path, updated: &str) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    json_files(results, &mut files);
    files.sort();
    let mut by_class: BTreeMap<String, (Device, Vec<Session>)> = BTreeMap::new();
    for path in files {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        // other JSON files live here too (stage results, parity, env): only a harness result counts
        let Ok(r) = serde_json::from_str::<Results>(&text) else {
            continue;
        };
        let rel = path.strip_prefix(rel_to).unwrap_or(&path);
        let rel = rel.to_string_lossy().replace('\\', "/");
        for (class, device, s) in sessions_of(&r, &rel) {
            by_class
                .entry(class)
                .or_insert((device, Vec::new()))
                .1
                .push(s);
        }
    }
    std::fs::create_dir_all(out).with_context(|| format!("create {}", out.display()))?;
    let mut written = Vec::new();
    for (class, (device, sessions)) in by_class {
        let old = load(out, &class)?;
        let f = merge(old, &class, device, sessions, updated);
        let path = file_path(out, &class);
        let mut text = serde_json::to_string_pretty(&f)?;
        text.push('\n');
        std::fs::write(&path, text).with_context(|| format!("write {}", path.display()))?;
        written.push(path);
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session(file: &str, host: &str, d: f64) -> Session {
        Session {
            file: file.into(),
            date: "2026-11-09".into(),
            host_id: host.into(),
            gpu: "NVIDIA GeForce RTX 3090".into(),
            driver: "570".into(),
            pair: "noise-cuda-dense".into(),
            threads: 8,
            repeats: 8,
            ratios: BTreeMap::from([("pp_ts".to_string(), 1.0 + d)]),
            d,
        }
    }

    fn file_with(ds: &[f64]) -> NoiseFile {
        let s = ds
            .iter()
            .enumerate()
            .map(|(i, d)| session(&format!("r{i}.json"), &format!("h{i}"), *d))
            .collect();
        merge(None, "rtx-3090", Device::Cuda, s, "2026-11-09")
    }

    #[test]
    fn slugs_drop_vendor_words() {
        assert_eq!(
            class_slug(Device::Cuda, Some("NVIDIA GeForce RTX 3090")),
            "rtx-3090"
        );
        assert_eq!(
            class_slug(Device::Cuda, Some("NVIDIA RTX A6000")),
            "rtx-a6000"
        );
        assert_eq!(class_slug(Device::Cuda, None), "cuda-unknown");
        assert_eq!(
            class_slug(Device::Cpu, Some("NVIDIA GeForce RTX 3090")),
            "cpu"
        );
    }

    #[test]
    fn band_is_margin_times_the_p95_clamped() {
        // below 20 sessions the nearest-rank p95 is the maximum
        let f = file_with(&[0.01, 0.03, 0.02]);
        assert_eq!(f.n, 3);
        assert!((f.d_p95 - 0.03).abs() < 1e-12);
        assert!((f.band - 0.0375).abs() < 1e-12);
        // quiet box: the floor
        assert!((file_with(&[0.001, 0.002, 0.003]).band - FLOOR).abs() < 1e-12);
        // wild box: the ceiling
        assert!((file_with(&[0.5, 0.5, 0.5]).band - CEIL).abs() < 1e-12);
        // 20 sessions: the 19th value, not the maximum
        let mut ds: Vec<f64> = (1..=20).map(|i| f64::from(i) * 0.001).collect();
        ds[19] = 0.2;
        assert!((file_with(&ds).d_p95 - 0.019).abs() < 1e-12);
    }

    #[test]
    fn merge_does_not_double_count_and_counts_hosts() {
        let a = file_with(&[0.01, 0.02]);
        let again = merge(
            Some(a.clone()),
            "rtx-3090",
            Device::Cuda,
            a.sessions.clone(),
            "2026-11-10",
        );
        assert_eq!(again.n, 2);
        assert_eq!(again.n_hosts, 2);
        let more = merge(
            Some(a),
            "rtx-3090",
            Device::Cuda,
            vec![session("r9.json", "h0", 0.05)],
            "2026-11-10",
        );
        assert_eq!((more.n, more.n_hosts), (3, 2));
        assert!((more.d_p95 - 0.05).abs() < 1e-12);
    }

    #[test]
    fn measured_band_wins_only_with_enough_sessions() {
        let prov = Some(("rtx-3090", 0.05));
        let enough = file_with(&[0.01, 0.01, 0.01]);
        let c = choose("rtx-3090", prov, Some(&enough)).unwrap();
        assert_eq!(c.source, "measured rtx-3090 n=3");
        assert!((c.band - 0.02).abs() < 1e-12);

        let few = file_with(&[0.01, 0.01]);
        let c = choose("rtx-3090", prov, Some(&few)).unwrap();
        assert_eq!(c.band, 0.05);
        assert_eq!(c.source, "provisional rtx-3090 (measured n=2 < 3)");

        let c = choose("rtx-3090", prov, None).unwrap();
        assert_eq!((c.band, c.source.as_str()), (0.05, "provisional rtx-3090"));
        assert_eq!(choose("rtx-3090", None, None), None);
        // a measured band needs no provisional one
        assert!(choose("rtx-3090", None, Some(&enough)).is_some());
    }

    #[test]
    fn load_reads_absent_valid_and_corrupt_files() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(load(dir.path(), "rtx-3090").unwrap(), None);
        let f = file_with(&[0.01, 0.02, 0.03]);
        std::fs::write(
            file_path(dir.path(), "rtx-3090"),
            serde_json::to_string(&f).unwrap(),
        )
        .unwrap();
        assert_eq!(load(dir.path(), "rtx-3090").unwrap(), Some(f));
        std::fs::write(file_path(dir.path(), "rtx-4090"), "{ not json").unwrap();
        assert!(load(dir.path(), "rtx-4090").is_err());
        // a file of another class under this name is refused
        let f = file_with(&[0.01]);
        std::fs::write(
            file_path(dir.path(), "rtx-4090"),
            serde_json::to_string(&f).unwrap(),
        )
        .unwrap();
        assert!(load(dir.path(), "rtx-4090").is_err());
    }
}
