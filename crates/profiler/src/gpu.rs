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

/// The GPUs a process with `CUDA_VISIBLE_DEVICES=cvd` sees, numbered as ggml numbers them (the
/// position in the list). Only index lists are read: an entry that is not a number (a UUID)
/// keeps every GPU, with a note. Like CUDA, the list stops at a negative, unknown or repeated
/// index; an empty list shows none. `None` (variable unset) keeps all.
pub fn filter_visible(all: Vec<GpuInfo>, cvd: Option<&str>) -> (Vec<GpuInfo>, Vec<String>) {
    let Some(spec) = cvd else {
        return (all, Vec::new());
    };
    let mut order: Vec<u32> = Vec::new();
    let mut notes = Vec::new();
    for tok in spec.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        let Ok(i) = tok.parse::<i64>() else {
            notes.push(format!(
                "CUDA_VISIBLE_DEVICES entry {tok:?} is not an index; all GPUs kept"
            ));
            return (all, notes);
        };
        match u32::try_from(i) {
            Ok(i) if all.iter().any(|g| g.index == i) && !order.contains(&i) => order.push(i),
            _ => break,
        }
    }
    notes.push(format!(
        "CUDA_VISIBLE_DEVICES={spec}: {} of {} GPUs visible",
        order.len(),
        all.len()
    ));
    let pos = |nvml: u32| order.iter().position(|&o| o == nvml).map(|i| i as u32);
    let out = order
        .iter()
        .filter_map(|&o| all.iter().find(|g| g.index == o))
        .map(|g| GpuInfo {
            index: pos(g.index).unwrap_or(0),
            p2p_peers: g.p2p_peers.iter().filter_map(|&p| pos(p)).collect(),
            ..g.clone()
        })
        .collect();
    (out, notes)
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

    fn gpu(index: u32, peers: &[u32]) -> GpuInfo {
        GpuInfo {
            index,
            name: format!("g{index}"),
            vram_bytes: 1,
            pcie: None,
            p2p_peers: peers.to_vec(),
        }
    }

    #[test]
    fn visible_gpus_are_renumbered_in_list_order() {
        let all = || vec![gpu(0, &[1]), gpu(1, &[0]), gpu(2, &[3]), gpu(3, &[2])];
        assert_eq!(filter_visible(all(), None).0.len(), 4);
        let (v, notes) = filter_visible(all(), Some("0,1"));
        assert_eq!(v.iter().map(|g| g.index).collect::<Vec<_>>(), [0, 1]);
        assert_eq!(v[0].p2p_peers, [1]);
        assert!(notes[0].contains("2 of 4"));
        // Order is the list's; peers outside it are dropped.
        let (v, _) = filter_visible(all(), Some("3,1,2"));
        let names: Vec<_> = v.iter().map(|g| g.name.as_str()).collect();
        assert_eq!(names, ["g3", "g1", "g2"]);
        assert_eq!(v[0].p2p_peers, [2]);
        assert!(v[1].p2p_peers.is_empty());
        assert_eq!(filter_visible(all(), Some("")).0.len(), 0);
        assert_eq!(filter_visible(all(), Some("-1")).0.len(), 0);
        assert_eq!(filter_visible(all(), Some("1,9,2")).0.len(), 1);
        let (v, notes) = filter_visible(all(), Some("GPU-abc"));
        assert_eq!(v.len(), 4);
        assert!(notes[0].contains("not an index"));
    }

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
