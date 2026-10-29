//! Assembles the hardware graph from the OS, probes and NVML.

use crate::gpu::{GpuInfo, discover_gpus, pcie_prior_bytes_per_s};
use crate::graph::{Device, DeviceKind, HardwareGraph, Link, LinkKind, Measurement, Tier};
use crate::probe::{probe_disk_read, probe_ram_read};
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Hook for measurements that need ggml.
/// Still missing: an implementation backed by `ggml-sys` (pinned-memory host-to-device copy loop).
pub trait DeviceProbe: Send + Sync {
    /// Host-to-device copy bandwidth in bytes per second for the GPU at NVML index `gpu_index`.
    fn host_to_device(&self, gpu_index: u32) -> Option<Measurement>;
}

#[derive(Clone)]
pub struct DiscoverOpts {
    /// Repeats per probe; the result is the median.
    pub repeats: usize,
    pub probe_ram: bool,
    /// Writes a temp file, so it is opt-in.
    pub probe_disk: bool,
    /// Directory (and so the volume) the disk probe uses; default is the temp dir.
    pub disk_probe_dir: Option<PathBuf>,
    pub disk_probe_bytes: u64,
    pub gpus: bool,
    pub device_probe: Option<Arc<dyn DeviceProbe>>,
}

impl Default for DiscoverOpts {
    fn default() -> Self {
        Self {
            repeats: 5,
            probe_ram: true,
            probe_disk: false,
            disk_probe_dir: None,
            disk_probe_bytes: 64 << 20,
            gpus: true,
            device_probe: None,
        }
    }
}

/// NUMA nodes under a sysfs node directory (`/sys/devices/system/node`); 1 when absent
/// (Windows, containers without sysfs).
pub fn numa_node_count(node_dir: &Path) -> u32 {
    let n = std::fs::read_dir(node_dir)
        .map(|rd| {
            rd.filter_map(Result::ok)
                .filter(|e| {
                    let name = e.file_name();
                    let name = name.to_string_lossy();
                    name.strip_prefix("node")
                        .is_some_and(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))
                })
                .count() as u32
        })
        .unwrap_or(0);
    n.max(1)
}

const PSEUDO_FS: &[&str] = &[
    "tmpfs",
    "devtmpfs",
    "squashfs",
    "proc",
    "sysfs",
    "cgroup",
    "cgroup2",
    "devpts",
    "ramfs",
    "efivarfs",
    "fuse.lxcfs",
];

/// Prior sequential read bandwidth and latency per media type, until a probe replaces them.
fn disk_prior(media: &str) -> (f64, f64) {
    match media {
        "ssd" => (3.0e9, 1.0e-4),
        "hdd" => (1.5e8, 1.0e-2),
        _ => (5.0e8, 1.0e-3),
    }
}

pub fn discover(opts: &DiscoverOpts) -> HardwareGraph {
    let mut g = HardwareGraph::default();
    discover_host(opts, &mut g);
    discover_disks(opts, &mut g);
    if opts.gpus {
        let (gpus, notes) = discover_gpus();
        g.notes.extend(notes);
        add_gpus(opts, &gpus, &mut g);
    }
    g
}

fn discover_host(opts: &DiscoverOpts, g: &mut HardwareGraph) {
    use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};
    let sys = System::new_with_specifics(
        RefreshKind::nothing()
            .with_cpu(CpuRefreshKind::nothing())
            .with_memory(MemoryRefreshKind::nothing().with_ram()),
    );
    // Without sysinfo the graph still needs its cpu0 (links point at it), with unknown sizes.
    let sys = sys
        .inspect_err(|e| g.notes.push(format!("sysinfo: host: {e}")))
        .ok();
    let name = sys
        .as_ref()
        .and_then(|s| s.cpus().first())
        .map(|c| c.brand().trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown cpu".into());
    let threads = sys.as_ref().map_or(1, |s| s.cpus().len().max(1) as u32);
    let cores = System::physical_core_count().ok().map(|c| c as u32);
    let mut cpu = Device::new(
        "cpu0",
        DeviceKind::Cpu,
        &name,
        Tier::Ram,
        sys.as_ref().map_or(0, System::total_memory),
    );
    cpu.cores = cores;
    cpu.threads = Some(threads);
    cpu.numa_nodes = Some(numa_node_count(Path::new("/sys/devices/system/node")));
    if opts.probe_ram {
        let t = (cores.unwrap_or(threads).max(1) as usize).min(8);
        cpu.bandwidth_bytes_per_s = probe_ram_read(t, 64 << 20, opts.repeats);
    }
    g.devices.push(cpu);
}

fn discover_disks(opts: &DiscoverOpts, g: &mut HardwareGraph) {
    use sysinfo::{DiskKind, Disks};
    let disks = match Disks::new_with_refreshed_list() {
        Ok(d) => d,
        Err(e) => {
            g.notes.push(format!("sysinfo: disks: {e}"));
            return;
        }
    };
    let mut vols: Vec<(String, PathBuf, u64, &'static str)> = Vec::new();
    for d in disks.iter() {
        let fs = d.file_system().to_string_lossy().to_lowercase();
        if d.total_space() == 0 || PSEUDO_FS.contains(&fs.as_str()) {
            continue;
        }
        let media = match d.kind() {
            DiskKind::SSD => "ssd",
            DiskKind::HDD => "hdd",
            _ => "unknown",
        };
        vols.push((
            d.name().to_string_lossy().into_owned(),
            d.mount_point().to_path_buf(),
            d.total_space(),
            media,
        ));
    }

    // The probe measures the volume holding the probe dir: the longest matching mount point.
    let probe_dir = opts
        .disk_probe_dir
        .clone()
        .unwrap_or_else(std::env::temp_dir);
    let probe_vol = vols
        .iter()
        .enumerate()
        .filter(|(_, v)| probe_dir.starts_with(&v.1))
        .max_by_key(|(_, v)| v.1.as_os_str().len())
        .map(|(i, _)| i);
    let mut probed: Option<(usize, Measurement)> = None;
    if opts.probe_disk {
        match probe_disk_read(&probe_dir, opts.disk_probe_bytes, opts.repeats) {
            Ok(m) => match probe_vol {
                Some(i) => probed = Some((i, m)),
                None => g.notes.push(format!(
                    "disk probe on {} matched no listed volume",
                    probe_dir.display()
                )),
            },
            Err(e) => g.notes.push(format!("disk probe failed: {e}")),
        }
    }

    let keys: Vec<&OsStr> = vols
        .iter()
        .map(|v| volume_key(&v.0, &v.1, cfg!(windows)))
        .collect();
    let mut seen: Vec<&OsStr> = Vec::new();
    for (i, (name, mount, total, media)) in vols.iter().enumerate() {
        // Subvolumes and bind mounts list one device several times; keep the first, but a probe
        // result on a later duplicate still applies to it.
        let dup = seen.iter().position(|k| *k == keys[i]);
        let idx = dup.unwrap_or(seen.len());
        let id = format!("disk{idx}");
        if dup.is_none() {
            seen.push(keys[i]);
            let mut dev = Device::new(&id, DeviceKind::Disk, name, Tier::Nvme, *total);
            dev.mount = Some(mount.to_string_lossy().into_owned());
            dev.media = Some((*media).into());
            g.devices.push(dev);
            let (beta, alpha) = disk_prior(media);
            g.links.push(Link {
                id: format!("storage-{id}"),
                a: id.clone(),
                b: "cpu0".into(),
                kind: LinkKind::Storage,
                alpha_s: Measurement::prior(alpha),
                beta_bytes_per_s: Measurement::prior(beta),
                lane_group: None,
            });
        }
        if let Some((pi, m)) = probed
            && pi == i
        {
            if let Some(dev) = g.devices.iter_mut().find(|d| d.id == id) {
                dev.bandwidth_bytes_per_s = Some(m);
            }
            if let Some(l) = g.links.iter_mut().find(|l| l.a == id) {
                l.beta_bytes_per_s = m;
            }
        }
    }
}

/// Identity of a volume for deduplication. Linux subvolumes and bind mounts share a device name
/// under several mount points; Windows volumes can share an empty or equal label, but each has
/// its own mount point.
fn volume_key<'a>(name: &'a str, mount: &'a Path, by_mount: bool) -> &'a OsStr {
    if by_mount {
        mount.as_os_str()
    } else {
        OsStr::new(name)
    }
}

fn add_gpus(opts: &DiscoverOpts, gpus: &[GpuInfo], g: &mut HardwareGraph) {
    let mut host_beta: Vec<(u32, f64)> = Vec::new();
    for gpu in gpus {
        let id = format!("gpu{}", gpu.index);
        let mut dev = Device::new(&id, DeviceKind::Gpu, &gpu.name, Tier::Vram, gpu.vram_bytes);
        dev.index = Some(gpu.index);
        dev.pcie = gpu.pcie;
        g.devices.push(dev);

        let measured = opts
            .device_probe
            .as_ref()
            .and_then(|p| p.host_to_device(gpu.index));
        let beta = measured.unwrap_or_else(|| {
            Measurement::prior(gpu.pcie.as_ref().map_or(1.0e10, pcie_prior_bytes_per_s))
        });
        if let Some(p) = &gpu.pcie
            && (p.gen_current < p.gen_max || p.width_current < p.width_max)
        {
            g.notes.push(format!(
                "{id}: PCIe link is gen{} x{} now, max gen{} x{} (may be idle power saving)",
                p.gen_current, p.width_current, p.gen_max, p.width_max
            ));
        }
        host_beta.push((gpu.index, beta.value));
        g.links.push(Link {
            id: format!("pcie-{id}"),
            a: id,
            b: "cpu0".into(),
            kind: LinkKind::Pcie,
            alpha_s: Measurement::prior(1.0e-5),
            beta_bytes_per_s: beta,
            lane_group: None,
        });
    }
    for gpu in gpus {
        for &peer in gpu.p2p_peers.iter().filter(|&&p| p > gpu.index) {
            let b0 = host_beta.iter().find(|h| h.0 == gpu.index).map(|h| h.1);
            let b1 = host_beta.iter().find(|h| h.0 == peer).map(|h| h.1);
            let beta = b0.zip(b1).map_or(1.0e10, |(a, b)| a.min(b));
            g.links.push(Link {
                id: format!("p2p-gpu{}-gpu{peer}", gpu.index),
                a: format!("gpu{}", gpu.index),
                b: format!("gpu{peer}"),
                kind: LinkKind::P2p,
                alpha_s: Measurement::prior(1.0e-5),
                beta_bytes_per_s: Measurement::prior(beta),
                lane_group: None,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numa_count_from_sysfs_fixture() {
        let dir = std::env::temp_dir().join(format!("tetraeder-numa-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for n in ["node0", "node1", "possible", "online", "nodeX"] {
            std::fs::create_dir_all(dir.join(n)).unwrap();
        }
        assert_eq!(numa_node_count(&dir), 2);
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(numa_node_count(&dir), 1, "missing dir means one node");
    }

    struct FakeProbe;
    impl DeviceProbe for FakeProbe {
        fn host_to_device(&self, _: u32) -> Option<Measurement> {
            Some(Measurement::prior(7.0e9))
        }
    }

    #[test]
    fn gpus_become_devices_links_and_p2p() {
        let pcie = |w| crate::graph::Pcie {
            gen_current: 4,
            width_current: w,
            gen_max: 4,
            width_max: 16,
        };
        let gpus = vec![
            GpuInfo {
                index: 0,
                name: "A".into(),
                vram_bytes: 24 << 30,
                pcie: Some(pcie(16)),
                p2p_peers: vec![1],
            },
            GpuInfo {
                index: 1,
                name: "B".into(),
                vram_bytes: 8 << 30,
                pcie: Some(pcie(8)),
                p2p_peers: vec![0],
            },
        ];
        let mut g = HardwareGraph::default();
        g.devices.push(Device::new(
            "cpu0",
            DeviceKind::Cpu,
            "c",
            Tier::Ram,
            1 << 30,
        ));
        add_gpus(&DiscoverOpts::default(), &gpus, &mut g);
        g.validate().unwrap();
        assert_eq!(
            g.devices
                .iter()
                .filter(|d| d.kind == DeviceKind::Gpu)
                .count(),
            2
        );
        assert_eq!(
            g.links.iter().filter(|l| l.kind == LinkKind::P2p).count(),
            1
        );
        let l1 = g.link_between("gpu1", "cpu0").unwrap();
        assert!((l1.beta_bytes_per_s.value - 0.8 * 1969.0e6 * 8.0).abs() < 1.0);
        let p2p = g.link_between("gpu0", "gpu1").unwrap();
        assert_eq!(p2p.beta_bytes_per_s.value, l1.beta_bytes_per_s.value);
        assert_eq!(g.notes.len(), 1, "only gpu1 is below its max width");

        let opts = DiscoverOpts {
            device_probe: Some(Arc::new(FakeProbe)),
            ..Default::default()
        };
        let mut g2 = HardwareGraph::default();
        add_gpus(&opts, &gpus, &mut g2);
        assert_eq!(
            g2.link_between("gpu0", "cpu0")
                .unwrap()
                .beta_bytes_per_s
                .value,
            7.0e9
        );
    }

    #[test]
    fn volume_key_separates_windows_volumes_and_merges_linux_subvolumes() {
        let (c, d) = (Path::new("C:\\"), Path::new("D:\\"));
        assert_ne!(volume_key("", c, true), volume_key("", d, true));
        assert_eq!(
            volume_key("/dev/sda2", Path::new("/"), false),
            volume_key("/dev/sda2", Path::new("/home"), false)
        );
    }

    /// Runs the real discovery on whatever machine this is.
    #[test]
    #[ignore = "real hardware probes; cargo test -- --ignored"]
    fn real_discovery_on_this_machine() {
        let opts = DiscoverOpts {
            repeats: 3,
            probe_disk: true,
            disk_probe_bytes: 16 << 20,
            ..Default::default()
        };
        let g = discover(&opts);
        g.validate().unwrap();
        let cpu = g.device("cpu0").unwrap();
        assert!(cpu.capacity_bytes > 256 << 20, "RAM size");
        assert!(cpu.threads.unwrap() >= 1);
        assert!(cpu.numa_nodes.unwrap() >= 1);
        let ram = cpu.bandwidth_bytes_per_s.unwrap();
        assert_eq!(ram.samples, 3);
        assert!(ram.value > 1.0e7);
        let toml = g.to_toml().unwrap();
        assert_eq!(HardwareGraph::from_toml(&toml).unwrap(), g);
        if std::env::var_os("TETRAEDER_PRINT_DISCOVERY").is_some() {
            println!("{toml}");
        }
    }
}
