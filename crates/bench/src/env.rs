//! Environment capture: every comparison needs host id, GPU, driver, clocks and throttle reasons.

use serde::{Deserialize, Serialize};
use std::process::Command;
use std::sync::OnceLock;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GpuSnap {
    pub name: String,
    pub driver: String,
    pub memory_mb: Option<f64>,
    pub clocks_sm_mhz: Option<f64>,
    pub clocks_mem_mhz: Option<f64>,
    pub clocks_max_sm_mhz: Option<f64>,
    /// Hex bitmask of active throttle reasons, "0x0000000000000000" when none.
    pub throttle_reasons: String,
    pub power_w: Option<f64>,
    pub temperature_c: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnvInfo {
    /// Stable id of the box: `CONTAINER_ID` (Vast), else the hostname.
    pub host_id: String,
    pub hostname: String,
    pub os: String,
    pub arch: String,
    pub cpu: String,
    pub cpus: usize,
    pub ram_gb: Option<f64>,
    pub gpus: Vec<GpuSnap>,
    pub git_commit: String,
    pub git_dirty: Option<bool>,
    pub cuda_env: String,
    pub started_utc: String,
}

const GPU_FIELDS: &str = "name,driver_version,memory.total,clocks.sm,clocks.mem,clocks.max.sm,{THROTTLE},power.draw,temperature.gpu";

fn run(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn num(s: &str) -> Option<f64> {
    s.trim().parse().ok()
}

/// Parse `nvidia-smi --query-gpu=<GPU_FIELDS> --format=csv,noheader,nounits` output, one GPU per line.
pub fn parse_gpu_csv(text: &str) -> Vec<GpuSnap> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| {
            let f: Vec<&str> = l.split(',').map(str::trim).collect();
            (f.len() >= 9).then(|| GpuSnap {
                name: f[0].to_string(),
                driver: f[1].to_string(),
                memory_mb: num(f[2]),
                clocks_sm_mhz: num(f[3]),
                clocks_mem_mhz: num(f[4]),
                clocks_max_sm_mhz: num(f[5]),
                throttle_reasons: f[6].to_string(),
                power_w: num(f[7]),
                temperature_c: num(f[8]),
            })
        })
        .collect()
}

/// Current state of all GPUs; empty without nvidia-smi or a device.
pub fn gpu_snapshot() -> Vec<GpuSnap> {
    // Newer drivers renamed clocks_throttle_reasons.active to clocks_event_reasons.active.
    // The field name that worked is cached, so a sample runs nvidia-smi once.
    static THROTTLE_FIELD: OnceLock<&'static str> = OnceLock::new();
    let query = |name: &str| {
        let q = format!("--query-gpu={}", GPU_FIELDS.replace("{THROTTLE}", name));
        let out = run("nvidia-smi", &[&q, "--format=csv,noheader,nounits"])?;
        Some(parse_gpu_csv(&out)).filter(|g| !g.is_empty())
    };
    if let Some(name) = THROTTLE_FIELD.get() {
        return query(name).unwrap_or_default();
    }
    for name in [
        "clocks_event_reasons.active",
        "clocks_throttle_reasons.active",
    ] {
        if let Some(gpus) = query(name) {
            let _ = THROTTLE_FIELD.set(name);
            return gpus;
        }
    }
    Vec::new()
}

fn hostname() -> String {
    ["HOSTNAME", "COMPUTERNAME"]
        .iter()
        .filter_map(|k| std::env::var(k).ok())
        .find(|v| !v.is_empty())
        .or_else(|| run("hostname", &[]))
        .unwrap_or_else(|| "unknown".into())
}

fn cpu_model() -> String {
    if let Ok(t) = std::fs::read_to_string("/proc/cpuinfo")
        && let Some(l) = t.lines().find(|l| l.starts_with("model name"))
    {
        return l
            .split_once(':')
            .map(|x| x.1.trim().to_string())
            .unwrap_or_default();
    }
    std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| "unknown".into())
}

fn ram_gb() -> Option<f64> {
    let t = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kb: f64 = t
        .lines()
        .find(|l| l.starts_with("MemTotal"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()?;
    Some((kb / 1048576.0 * 10.0).round() / 10.0)
}

fn git_commit() -> (String, Option<bool>) {
    let commit = run("git", &["rev-parse", "HEAD"])
        .or_else(|| std::env::var("TETRAEDER_GIT_COMMIT").ok())
        .unwrap_or_else(|| "unknown".into());
    let dirty = run("git", &["status", "--porcelain"]).map(|s| !s.is_empty());
    (commit, dirty)
}

pub fn capture() -> EnvInfo {
    let hostname = hostname();
    let host_id = std::env::var("CONTAINER_ID")
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| hostname.clone());
    let (git_commit, git_dirty) = git_commit();
    EnvInfo {
        host_id,
        hostname,
        os: std::env::consts::OS.into(),
        arch: std::env::consts::ARCH.into(),
        cpu: cpu_model(),
        cpus: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        ram_gb: ram_gb(),
        gpus: gpu_snapshot(),
        git_commit,
        git_dirty,
        cuda_env: std::env::var("CUDA_ENV_STATUS").unwrap_or_default(),
        started_utc: utc_now(),
    }
}

/// (year, month, day, hour, minute, second) in UTC.
pub fn utc_parts(unix_secs: u64) -> (i64, u32, u32, u32, u32, u32) {
    let days = (unix_secs / 86400) as i64;
    let rem = unix_secs % 86400;
    // Civil-from-days (Howard Hinnant).
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = yoe + era * 400 + i64::from(m <= 2);
    (
        y,
        m,
        d,
        (rem / 3600) as u32,
        (rem % 3600 / 60) as u32,
        (rem % 60) as u32,
    )
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn utc_now() -> String {
    let (y, mo, d, h, mi, s) = utc_parts(now_secs());
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

/// `HHMMSS` (UTC), distinguishes several runs of one spec on the same day.
pub fn time_name() -> String {
    let (.., h, mi, s) = utc_parts(now_secs());
    format!("{h:02}{mi:02}{s:02}")
}

/// `YYYYMMDD`, the name of the nightly results directory (UTC).
pub fn date_dir_name() -> String {
    let (y, mo, d, ..) = utc_parts(now_secs());
    format!("{y:04}{mo:02}{d:02}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nvidia_smi_csv() {
        let csv = "NVIDIA GeForce RTX 4090, 550.54.14, 24564, 2520, 10501, 3105, 0x0000000000000004, 312.45, 61\n";
        let g = parse_gpu_csv(csv);
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].name, "NVIDIA GeForce RTX 4090");
        assert_eq!(g[0].clocks_sm_mhz, Some(2520.0));
        assert_eq!(g[0].throttle_reasons, "0x0000000000000004");
        assert_eq!(g[0].power_w, Some(312.45));
    }

    #[test]
    fn tolerates_not_supported_fields_and_blank_lines() {
        let csv = "GPU, 1.0, 100, [N/A], [N/A], [N/A], [N/A], [N/A], 40\n\n";
        let g = parse_gpu_csv(csv);
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].clocks_sm_mhz, None);
        assert!(parse_gpu_csv("").is_empty());
    }

    #[test]
    fn civil_date() {
        assert_eq!(utc_parts(0), (1970, 1, 1, 0, 0, 0));
        assert_eq!(utc_parts(1_782_864_000 + 3661), (2026, 7, 1, 1, 1, 1));
        assert_eq!(utc_parts(951_782_400).0, 2000); // 2000-02-29
        assert_eq!(utc_parts(951_782_400).1, 2);
        assert_eq!(utc_parts(951_782_400).2, 29);
    }
}
