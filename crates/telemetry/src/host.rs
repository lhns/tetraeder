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

    #[cfg(feature = "nvml")]
    #[test]
    fn nvml_absent_is_graceful() {
        // Skips without a GPU: either no library (None) or readings per device.
        if let Some(mut n) = NvmlHost::new() {
            let _ = n.sample();
        }
    }
}
