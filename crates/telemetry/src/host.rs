use std::path::{Path, PathBuf};
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    Gpu,
}

/// One reading of one device. The CPU device carries system RAM as its memory.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceSample {
    /// Stable id used as the `tetraeder.device` attribute: `cpu`, `gpu0`, ...
    pub id: String,
    pub name: String,
    pub kind: DeviceKind,
    /// Busy fraction, 0 to 1.
    pub busy: f64,
    pub mem_used: u64,
    pub mem_total: u64,
    pub power_w: Option<f64>,
    pub temp_c: Option<f64>,
}

/// A source of device readings.
pub trait HostMetrics: Send {
    fn name(&self) -> &'static str;
    fn sample(&mut self) -> Vec<DeviceSample>;
}

/// CPU and RAM through sysinfo (Windows and Linux).
pub struct SysinfoHost {
    sys: System,
}

impl SysinfoHost {
    /// Errors when the platform backend cannot start.
    pub fn new() -> Result<Self, sysinfo::Error> {
        let mut sys = System::new_with_specifics(
            RefreshKind::nothing()
                .with_cpu(CpuRefreshKind::nothing().with_usage())
                .with_memory(MemoryRefreshKind::nothing().with_ram()),
        )?;
        // CPU usage is a delta between two refreshes.
        sys.refresh_cpu_usage();
        Ok(Self { sys })
    }
}

impl HostMetrics for SysinfoHost {
    fn name(&self) -> &'static str {
        "sysinfo"
    }

    fn sample(&mut self) -> Vec<DeviceSample> {
        self.sys.refresh_cpu_usage();
        self.sys.refresh_memory();
        let name = self
            .sys
            .cpus()
            .first()
            .map(|c| c.brand().trim().to_owned())
            .unwrap_or_default();
        vec![DeviceSample {
            id: "cpu0".into(),
            name,
            kind: DeviceKind::Cpu,
            busy: f64::from(self.sys.global_cpu_usage()) / 100.0,
            mem_used: self.sys.used_memory(),
            mem_total: self.sys.total_memory(),
            power_w: None,
            temp_c: None,
        }]
    }
}

/// NVIDIA GPUs through NVML, loaded at runtime.
#[cfg(feature = "nvml")]
pub struct NvmlHost {
    nvml: nvml_wrapper::Nvml,
}

#[cfg(feature = "nvml")]
impl NvmlHost {
    /// `None` when the NVML library or a driver is missing.
    pub fn new() -> Option<Self> {
        let nvml = nvml_wrapper::Nvml::init().ok()?;
        Some(Self { nvml })
    }
}

#[cfg(feature = "nvml")]
impl HostMetrics for NvmlHost {
    fn name(&self) -> &'static str {
        "nvml"
    }

    fn sample(&mut self) -> Vec<DeviceSample> {
        use nvml_wrapper::enum_wrappers::device::TemperatureSensor;
        let count = self.nvml.device_count().unwrap_or(0);
        (0..count)
            .filter_map(|i| {
                let d = self.nvml.device_by_index(i).ok()?;
                let mem = d.memory_info().ok()?;
                Some(DeviceSample {
                    id: format!("gpu{i}"),
                    name: d.name().unwrap_or_default(),
                    kind: DeviceKind::Gpu,
                    busy: d
                        .utilization_rates()
                        .map_or(0.0, |u| f64::from(u.gpu) / 100.0),
                    mem_used: mem.used,
                    mem_total: mem.total,
                    power_w: d.power_usage().ok().map(|mw| f64::from(mw) / 1000.0),
                    temp_c: d.temperature(TemperatureSensor::Gpu).ok().map(f64::from),
                })
            })
            .collect()
    }
}

/// The cgroup's CPU limit in cores: v2 `cpu.max` (`<quota|max> <period>`), else v1
/// `cpu.cfs_quota_us` over `cpu.cfs_period_us`; `None` when unlimited or not on Linux.
pub fn cgroup_cpu_quota() -> Option<f64> {
    cgroup_cpu_quota_at(Path::new("/sys/fs/cgroup"))
}

/// [`cgroup_cpu_quota`] under another cgroup mount, for tests on fake files.
pub fn cgroup_cpu_quota_at(root: &Path) -> Option<f64> {
    let read = |p: PathBuf| std::fs::read_to_string(p).ok();
    let v1 = || {
        ["cpu", "cpu,cpuacct"].iter().find_map(|d| {
            let q = read(root.join(d).join("cpu.cfs_quota_us"))?;
            let p = read(root.join(d).join("cpu.cfs_period_us"))?;
            Some(format!("{} {}", q.trim(), p.trim()))
        })
    };
    parse_cpu_max(&read(root.join("cpu.max")).or_else(v1)?)
}

/// The host's online CPUs from `/sys/devices/system/cpu/online`, which a container shows in full
/// (unlike `available_parallelism`, which respects the quota); `None` when unreadable.
pub fn online_cpus() -> Option<usize> {
    online_cpus_at(Path::new("/sys/devices/system/cpu/online"))
}

/// [`online_cpus`] from another file, for tests.
pub fn online_cpus_at(path: &Path) -> Option<usize> {
    parse_cpu_list(&std::fs::read_to_string(path).ok()?)
}

/// Counts a kernel CPU list such as `0-63,128-191` or `0,2`; `None` when malformed or empty.
fn parse_cpu_list(text: &str) -> Option<usize> {
    let mut n = 0;
    for part in text.trim().split(',') {
        n += match part.split_once('-') {
            Some((a, b)) => {
                let (a, b): (usize, usize) = (a.parse().ok()?, b.parse().ok()?);
                b.checked_sub(a)? + 1
            }
            None => {
                part.parse::<usize>().ok()?;
                1
            }
        };
    }
    (n > 0).then_some(n)
}

/// `<quota> <period>` in microseconds as cores, rounded to 0.01; `None` for `max` or `-1`.
fn parse_cpu_max(text: &str) -> Option<f64> {
    let mut f = text.split_whitespace();
    let q: f64 = f.next()?.parse().ok()?;
    let p: f64 = f.next().unwrap_or("100000").parse().ok()?;
    (q > 0.0 && p > 0.0).then(|| (q / p * 100.0).round() / 100.0)
}

/// Every source available on this machine: sysinfo always, NVML when built in and present.
pub fn default_sources() -> Vec<Box<dyn HostMetrics>> {
    let mut v: Vec<Box<dyn HostMetrics>> = Vec::new();
    if let Ok(h) = SysinfoHost::new() {
        v.push(Box::new(h));
    }
    #[cfg(feature = "nvml")]
    if let Some(n) = NvmlHost::new() {
        v.push(Box::new(n));
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sysinfo_reports_cpu_and_ram() {
        let mut h = SysinfoHost::new().unwrap();
        let s = h.sample();
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].id, "cpu0");
        assert!(s[0].mem_total > 0 && s[0].mem_used <= s[0].mem_total);
        assert!((0.0..=1.0).contains(&s[0].busy));
    }

    #[test]
    fn cpu_max_parses_v2_and_v1_forms() {
        assert_eq!(parse_cpu_max("2304000 100000\n"), Some(23.04));
        assert_eq!(parse_cpu_max("max 100000\n"), None);
        assert_eq!(parse_cpu_max("-1 100000"), None);
        assert_eq!(parse_cpu_max("150000"), Some(1.5));
        assert_eq!(parse_cpu_max(""), None);
    }

    /// A fresh directory with `files` (path relative to it, content).
    fn fake(name: &str, files: &[(&str, &str)]) -> PathBuf {
        let d = std::env::temp_dir().join(format!("tetraeder-cg-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        for (p, c) in files {
            let f = d.join(p);
            std::fs::create_dir_all(f.parent().unwrap()).unwrap();
            std::fs::write(f, c).unwrap();
        }
        d
    }

    #[test]
    fn cgroup_quota_from_fake_files() {
        let v2 = fake("v2", &[("cpu.max", "2304000 100000\n")]);
        assert_eq!(cgroup_cpu_quota_at(&v2), Some(23.04));
        let v2max = fake("v2max", &[("cpu.max", "max 100000\n")]);
        assert_eq!(cgroup_cpu_quota_at(&v2max), None);
        let v1 = fake(
            "v1",
            &[
                ("cpu,cpuacct/cpu.cfs_quota_us", "1200000\n"),
                ("cpu,cpuacct/cpu.cfs_period_us", "100000\n"),
            ],
        );
        assert_eq!(cgroup_cpu_quota_at(&v1), Some(12.0));
        let v1max = fake(
            "v1max",
            &[
                ("cpu/cpu.cfs_quota_us", "-1\n"),
                ("cpu/cpu.cfs_period_us", "100000\n"),
            ],
        );
        assert_eq!(cgroup_cpu_quota_at(&v1max), None);
        // v2 wins over v1 when both exist.
        let both = fake(
            "both",
            &[
                ("cpu.max", "800000 100000"),
                ("cpu/cpu.cfs_quota_us", "1200000"),
                ("cpu/cpu.cfs_period_us", "100000"),
            ],
        );
        assert_eq!(cgroup_cpu_quota_at(&both), Some(8.0));
        let missing = fake("missing", &[]);
        assert_eq!(cgroup_cpu_quota_at(&missing), None);
        for d in [v2, v2max, v1, v1max, both, missing] {
            let _ = std::fs::remove_dir_all(d);
        }
    }

    #[test]
    fn online_cpus_from_cpu_lists() {
        assert_eq!(parse_cpu_list("0-127\n"), Some(128));
        assert_eq!(parse_cpu_list("0-63,128-191"), Some(128));
        assert_eq!(parse_cpu_list("0,2,4-5"), Some(4));
        assert_eq!(parse_cpu_list("0"), Some(1));
        assert_eq!(parse_cpu_list(""), None);
        assert_eq!(parse_cpu_list("5-3"), None);
        let d = fake("online", &[("online", "0-47\n")]);
        assert_eq!(online_cpus_at(&d.join("online")), Some(48));
        assert_eq!(online_cpus_at(&d.join("absent")), None);
        let _ = std::fs::remove_dir_all(d);
    }

    #[cfg(feature = "nvml")]
    #[test]
    fn nvml_absent_is_graceful() {
        // Skips without a GPU: either no library (None) or readings per device.
        if let Some(mut n) = NvmlHost::new() {
            let _ = n.sample();
        }
    }
}
