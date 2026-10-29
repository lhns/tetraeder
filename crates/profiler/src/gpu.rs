//! GPU discovery. Real code behind the `nvml` feature; without it no GPUs are reported.

use crate::graph::Pcie;

#[derive(Debug, Clone, PartialEq)]
pub struct GpuInfo {
    pub index: u32,
    pub name: String,
    pub vram_bytes: u64,
    pub pcie: Option<Pcie>,
    /// Indices of GPUs NVML reports as reachable without crossing the CPU's host bridge or
    /// system interconnect (same PCIe switch or NVLink).
    pub p2p_peers: Vec<u32>,
}

/// Returns the GPUs and any notes (driver missing, NVML errors).
#[cfg(feature = "nvml")]
pub fn discover_gpus() -> (Vec<GpuInfo>, Vec<String>) {
    use nvml_wrapper::Nvml;
    let nvml = match Nvml::init() {
        Ok(n) => n,
        Err(e) => return (Vec::new(), vec![format!("nvml: init failed: {e}")]),
    };
    let count = match nvml.device_count() {
        Ok(c) => c,
        Err(e) => return (Vec::new(), vec![format!("nvml: device_count failed: {e}")]),
    };
    let mut notes = Vec::new();
    let mut gpus = Vec::new();
    for index in 0..count {
        let dev = match nvml.device_by_index(index) {
            Ok(d) => d,
            Err(e) => {
                notes.push(format!("nvml: device {index}: {e}"));
                continue;
            }
        };
        let name = dev.name().unwrap_or_else(|_| format!("gpu{index}"));
        let vram_bytes = dev.memory_info().map(|m| m.total).unwrap_or(0);
        let pcie = (|| {
            Some(Pcie {
                gen_current: dev.current_pcie_link_gen().ok()?,
                width_current: dev.current_pcie_link_width().ok()?,
                gen_max: dev.max_pcie_link_gen().ok()?,
                width_max: dev.max_pcie_link_width().ok()?,
            })
        })();
        let p2p_peers = p2p_peers(&nvml, &dev, index, count);
        gpus.push(GpuInfo {
            index,
            name,
            vram_bytes,
            pcie,
            p2p_peers,
        });
    }
    (gpus, notes)
}

/// Peers sharing at most a host bridge with `dev`. NVML exposes topology on Linux only.
#[cfg(all(feature = "nvml", target_os = "linux"))]
fn p2p_peers(
    nvml: &nvml_wrapper::Nvml,
    dev: &nvml_wrapper::Device<'_>,
    index: u32,
    count: u32,
) -> Vec<u32> {
    use nvml_wrapper::enum_wrappers::device::TopologyLevel;
    (0..count)
        .filter(|&o| o != index)
        .filter(|&o| {
            nvml.device_by_index(o)
                .and_then(|od| dev.topology_common_ancestor(od))
                .is_ok_and(|level| {
                    matches!(
                        level,
                        TopologyLevel::Internal
                            | TopologyLevel::Single
                            | TopologyLevel::Multiple
                            | TopologyLevel::HostBridge
                    )
                })
        })
        .collect()
}

#[cfg(all(feature = "nvml", not(target_os = "linux")))]
fn p2p_peers(_: &nvml_wrapper::Nvml, _: &nvml_wrapper::Device<'_>, _: u32, _: u32) -> Vec<u32> {
    Vec::new()
}

#[cfg(not(feature = "nvml"))]
pub fn discover_gpus() -> (Vec<GpuInfo>, Vec<String>) {
    (Vec::new(), Vec::new())
}

/// Theoretical per-lane PCIe payload bandwidth in bytes per second after encoding overhead.
pub fn pcie_lane_bytes_per_s(generation: u32) -> f64 {
    match generation {
        1 => 250.0e6,
        2 => 500.0e6,
        3 => 984.6e6,
        4 => 1969.0e6,
        5 => 3938.0e6,
        6 => 7563.0e6,
        _ => 984.6e6,
    }
}

/// Prior for a GPU's host link: 80% of the theoretical rate at the current width and generation.
pub fn pcie_prior_bytes_per_s(p: &Pcie) -> f64 {
    0.8 * pcie_lane_bytes_per_s(p.gen_current) * f64::from(p.width_current)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcie_prior_gen4_x16() {
        let p = Pcie {
            gen_current: 4,
            width_current: 16,
            gen_max: 4,
            width_max: 16,
        };
        let v = pcie_prior_bytes_per_s(&p);
        assert!((v - 0.8 * 1969.0e6 * 16.0).abs() < 1.0);
        assert!(v > 24.0e9 && v < 26.0e9);
    }

    #[test]
    fn pcie_prior_scales_with_width_and_gen() {
        let x4 = Pcie {
            gen_current: 3,
            width_current: 4,
            gen_max: 4,
            width_max: 8,
        };
        let x8 = Pcie {
            width_current: 8,
            ..x4
        };
        assert!((pcie_prior_bytes_per_s(&x8) / pcie_prior_bytes_per_s(&x4) - 2.0).abs() < 1e-12);
    }
}
