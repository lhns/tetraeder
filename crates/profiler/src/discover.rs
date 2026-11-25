//! Assembles the hardware graph from the OS, probes and NVML.

use crate::gpu::{GpuInfo, discover_gpus, filter_visible, pcie_prior_bytes_per_s};
use crate::graph::{Device, DeviceKind, HardwareGraph, Link, LinkKind, Measurement, Tier};
use crate::pcie::{PcieClass, PcieTopology, lane_groups};
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
    /// Where the node keeps its models (and so its expert files). The volume holding it is the
    /// builder disk, named [`BUILDER_DISK`]; without it every disk keeps its `diskN` name.
    pub models_dir: Option<PathBuf>,
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
            models_dir: None,
            gpus: true,
            device_probe: None,
        }
    }
}

/// Id of the disk that holds the node's expert files; the planner reads the builder's experts
/// from the device of this id (`tetraeder_planner::score::BUILDER_DISK`).
pub const BUILDER_DISK: &str = "nvme0";

/// A mounted volume as the OS lists it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Volume {
    pub name: String,
    pub mount: PathBuf,
    pub total_bytes: u64,
    /// `ssd`, `hdd` or `unknown`.
    pub media: String,
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
        // Rank i of a plan is ggml's device i, which is the i-th visible GPU.
        let cvd = std::env::var("CUDA_VISIBLE_DEVICES").ok();
        let total = gpus.len();
        let (gpus, notes) = filter_visible(gpus, cvd.as_deref());
        g.notes.extend(notes);
        add_gpus(opts, &gpus, &mut g);
        if gpus.len() == total {
            add_pcie_topology(&gpus, &mut g);
        } else {
            g.notes
                .push("pcie topology: not read while CUDA_VISIBLE_DEVICES hides GPUs".into());
        }
    }
    g
}

fn add_pcie_topology(gpus: &[GpuInfo], g: &mut HardwareGraph) {
    if cfg!(target_os = "linux") {
        apply_topology(&crate::pcie::pcie_topology(Path::new("/sys")), gpus, g);
    } else {
        g.notes
            .push("pcie topology: only read on Linux (sysfs), no lane groups".into());
    }
}

/// Matches NVML GPUs to the topology's GPU-class devices (both in PCI bus order), fills missing
/// `Device.pcie` and adds the lane groups that contain a GPU, tagging those GPUs' host links.
fn apply_topology(t: &PcieTopology, gpus: &[GpuInfo], g: &mut HardwareGraph) {
    let pci_gpus: Vec<&str> = t
        .devices
        .iter()
        .filter(|d| d.class == PcieClass::Gpu)
        .map(|d| d.bdf.as_str())
        .collect();
    if pci_gpus.len() != gpus.len() {
        if !gpus.is_empty() {
            g.notes.push(format!(
                "pcie topology: {} GPUs from NVML but {} PCI display devices, not matched",
                gpus.len(),
                pci_gpus.len()
            ));
        }
        return;
    }
    let mut by_index: Vec<&GpuInfo> = gpus.iter().collect();
    by_index.sort_by_key(|x| x.index);
    let bdf_to_id: Vec<(&str, String)> = pci_gpus
        .iter()
        .zip(&by_index)
        .map(|(b, x)| (*b, format!("gpu{}", x.index)))
        .collect();
    for (bdf, id) in &bdf_to_id {
        let link = t
            .devices
            .iter()
            .find(|d| d.bdf == *bdf)
            .and_then(|d| d.link);
        if let Some(dev) = g.devices.iter_mut().find(|d| d.id == *id)
            && dev.pcie.is_none()
        {
            dev.pcie = link;
        }
    }
    for (lg, members) in lane_groups(t) {
        let ids: Vec<&String> = bdf_to_id
            .iter()
            .filter(|(b, _)| members.iter().any(|m| m == b))
            .map(|(_, id)| id)
            .collect();
        if ids.is_empty() {
            continue;
        }
        for id in ids {
            let link_id = format!("pcie-{id}");
            if let Some(l) = g.links.iter_mut().find(|l| l.id == link_id) {
                l.lane_group = Some(lg.id.clone());
            }
        }
        g.lane_groups.push(lg);
    }
    halve_staged_in_shared_lanes(g);
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
    let mut vols: Vec<Volume> = Vec::new();
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
        vols.push(Volume {
            name: d.name().to_string_lossy().into_owned(),
            mount: d.mount_point().to_path_buf(),
            total_bytes: d.total_space(),
            media: media.into(),
        });
    }
    let probe_dir = opts
        .disk_probe_dir
        .clone()
        .unwrap_or_else(std::env::temp_dir);
    let probed = if opts.probe_disk {
        match probe_disk_read(&probe_dir, opts.disk_probe_bytes, opts.repeats) {
            Ok(m) => Some(m),
            Err(e) => {
                g.notes.push(format!("disk probe failed: {e}"));
                None
            }
        }
    } else {
        None
    };
    add_disks(g, &vols, opts.models_dir.as_deref(), &probe_dir, probed);
}

/// Index of the volume holding `dir`: the longest matching mount point. A relative `dir` is
/// made absolute first (`canonicalize` would add a `\?\` prefix on Windows that matches no
/// mount point).
fn volume_of(vols: &[Volume], dir: &Path) -> Option<usize> {
    let dir = std::path::absolute(dir).unwrap_or_else(|_| dir.to_path_buf());
    vols.iter()
        .enumerate()
        .filter(|(_, v)| dir.starts_with(&v.mount))
        .max_by_key(|(_, v)| v.mount.as_os_str().len())
        .map(|(i, _)| i)
}

/// Adds one disk device and storage link per distinct volume. The volume holding `models_dir`
/// is the builder disk (`nvme0`), the others are `diskN`. `probed` is the read rate measured on
/// the volume holding `probe_dir`.
pub fn add_disks(
    g: &mut HardwareGraph,
    vols: &[Volume],
    models_dir: Option<&Path>,
    probe_dir: &Path,
    probed: Option<Measurement>,
) {
    let probe_vol = volume_of(vols, probe_dir);
    let mut probed_at: Option<(usize, Measurement)> = None;
    if let Some(m) = probed {
        match probe_vol {
            Some(i) => probed_at = Some((i, m)),
            None => g.notes.push(format!(
                "disk probe on {} matched no listed volume",
                probe_dir.display()
            )),
        }
    }
    let keys: Vec<&OsStr> = vols
        .iter()
        .map(|v| volume_key(&v.name, &v.mount, cfg!(windows)))
        .collect();
    let models_key = match models_dir {
        None => {
            g.notes.push(format!(
                "no models dir: no disk is named {BUILDER_DISK}, the planner homes experts in RAM"
            ));
            None
        }
        Some(d) => {
            let i = volume_of(vols, d);
            if i.is_none() {
                g.notes.push(format!(
                    "models dir {} matched no listed volume: no disk is named {BUILDER_DISK}",
                    d.display()
                ));
            }
            i.map(|i| keys[i])
        }
    };
    let mut seen: Vec<&OsStr> = Vec::new();
    for (i, v) in vols.iter().enumerate() {
        // Subvolumes and bind mounts list one device several times; keep the first, but a probe
        // result on a later duplicate still applies to it.
        let dup = seen.iter().position(|k| *k == keys[i]);
        let idx = dup.unwrap_or(seen.len());
        let builder = models_key == Some(keys[i]);
        let id = if builder {
            BUILDER_DISK.to_owned()
        } else {
            format!("disk{idx}")
        };
        if dup.is_none() {
            seen.push(keys[i]);
            let mut dev = Device::new(&id, DeviceKind::Disk, &v.name, Tier::Nvme, v.total_bytes);
            dev.mount = Some(v.mount.to_string_lossy().into_owned());
            dev.media = Some(v.media.clone());
            g.devices.push(dev);
            let (beta, alpha) = disk_prior(&v.media);
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
        if let Some((pi, m)) = probed_at
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
    // Every other GPU pair stages through host memory (ADR 0070). Prior: twice the larger host
    // alpha, the smaller host beta; `apply_topology` halves it for two GPUs in one lane group.
    for (i, gpu) in gpus.iter().enumerate() {
        for other in &gpus[i + 1..] {
            let (lo, hi) = if gpu.index < other.index {
                (gpu, other)
            } else {
                (other, gpu)
            };
            if lo.p2p_peers.contains(&hi.index) {
                continue;
            }
            let host = |index: u32| g.link_between(&format!("gpu{index}"), "cpu0").cloned();
            let (Some(a), Some(b)) = (host(lo.index), host(hi.index)) else {
                continue;
            };
            let alpha = 2.0 * a.alpha_s.value.max(b.alpha_s.value);
            let beta = a.beta_bytes_per_s.value.min(b.beta_bytes_per_s.value);
            g.links.push(Link {
                id: format!("staged-gpu{}-gpu{}", lo.index, hi.index),
                a: format!("gpu{}", lo.index),
                b: format!("gpu{}", hi.index),
                kind: LinkKind::HostStaged,
                alpha_s: Measurement::prior(alpha),
                beta_bytes_per_s: Measurement::prior(beta),
                lane_group: None,
            });
        }
    }
}

/// Halves the prior beta of staged GPU pairs whose two host links share a lane group: both
/// transfers cross the same lanes twice (ADR 0070).
fn halve_staged_in_shared_lanes(g: &mut HardwareGraph) {
    let lane_of = |g: &HardwareGraph, dev: &str| {
        g.link_between(dev, "cpu0")
            .and_then(|l| l.lane_group.clone())
    };
    let shared: Vec<String> = g
        .links
        .iter()
        .filter(|l| l.kind == LinkKind::HostStaged)
        .filter(|l| {
            let (x, y) = (lane_of(g, &l.a), lane_of(g, &l.b));
            x.is_some() && x == y
        })
        .map(|l| l.id.clone())
        .collect();
    for l in g.links.iter_mut().filter(|l| shared.contains(&l.id)) {
        l.beta_bytes_per_s.value /= 2.0;
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

    fn vol(name: &str, mount: &Path, media: &str) -> Volume {
        Volume {
            name: name.into(),
            mount: mount.to_path_buf(),
            total_bytes: 1 << 40,
            media: media.into(),
        }
    }

    fn root() -> PathBuf {
        std::env::current_dir()
            .unwrap()
            .ancestors()
            .last()
            .unwrap()
            .to_path_buf()
    }

    #[test]
    fn the_volume_with_the_models_dir_is_the_builder_disk() {
        let r = root();
        let (m1, m2) = (r.join("mnt").join("a"), r.join("mnt").join("b"));
        let vols = vec![
            vol("overlay", &r, "unknown"),
            vol("/dev/a", &m1, "ssd"),
            vol("/dev/b", &m2, "ssd"),
        ];
        let mut g = HardwareGraph::default();
        add_disks(&mut g, &vols, Some(&m2.join("models")), &r, None);
        let ids: Vec<&str> = g.devices.iter().map(|d| d.id.as_str()).collect();
        assert_eq!(ids, ["disk0", "disk1", BUILDER_DISK]);
        let links: Vec<&str> = g.links.iter().map(|l| l.a.as_str()).collect();
        assert_eq!(links, ids, "one storage link per disk, from the disk");
        assert!(g.notes.is_empty(), "{:?}", g.notes);
    }

    #[test]
    fn a_probe_on_the_models_volume_is_the_builder_disks_rate() {
        let r = root();
        let m = r.join("mnt").join("m");
        let vols = vec![vol("overlay", &r, "unknown"), vol("/dev/m", &m, "ssd")];
        let mut g = HardwareGraph::default();
        let rate = Measurement::prior(2.0e9);
        add_disks(&mut g, &vols, Some(&m), &m.join("tmp"), Some(rate));
        assert_eq!(g.device(BUILDER_DISK).unwrap().disk_rate(1), Some(2.0e9));
        assert_eq!(g.device("disk0").unwrap().disk_rate(1), None);
    }

    #[test]
    fn without_a_models_dir_no_disk_is_the_builder_disk() {
        let r = root();
        let vols = vec![vol("overlay", &r, "unknown")];
        let mut g = HardwareGraph::default();
        add_disks(&mut g, &vols, None, &r, None);
        assert_eq!(g.devices[0].id, "disk0");
        assert!(g.notes.iter().any(|n| n.contains("no models dir")));
        // A models dir on no listed volume says so too.
        let mut g = HardwareGraph::default();
        add_disks(&mut g, &[], Some(Path::new("models")), &r, None);
        assert!(g.devices.is_empty());
        assert!(
            g.notes
                .iter()
                .any(|n| n.contains("matched no listed volume"))
        );
    }

    #[test]
    fn a_relative_models_dir_is_made_absolute() {
        let vols = vec![vol("cwd", &root(), "ssd")];
        let mut g = HardwareGraph::default();
        add_disks(&mut g, &vols, Some(Path::new(".hf_cache")), &root(), None);
        assert_eq!(g.devices[0].id, BUILDER_DISK);
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
        assert!(g.links.iter().all(|l| l.kind != LinkKind::HostStaged));
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
    fn topology_fills_pcie_and_lane_groups() {
        use crate::graph::Pcie;
        use crate::pcie::PcieDevice;
        let link = |w| {
            Some(Pcie {
                gen_current: 4,
                width_current: w,
                gen_max: 4,
                width_max: 16,
            })
        };
        let dev = |bdf: &str, class, up: &[&str], w| PcieDevice {
            bdf: bdf.into(),
            class,
            upstream: up.iter().map(|s| s.to_string()).collect(),
            link: link(w),
        };
        let t = PcieTopology {
            devices: vec![
                dev("0000:01:00.0", PcieClass::Other, &["0000:00:01.0"], 8),
                dev(
                    "0000:03:00.0",
                    PcieClass::Gpu,
                    &["0000:01:00.0", "0000:00:01.0"],
                    8,
                ),
                dev(
                    "0000:04:00.0",
                    PcieClass::Gpu,
                    &["0000:01:00.0", "0000:00:01.0"],
                    8,
                ),
                dev(
                    "0000:05:00.0",
                    PcieClass::Nvme,
                    &["0000:01:00.0", "0000:00:01.0"],
                    4,
                ),
            ],
        };
        let gpus: Vec<GpuInfo> = (0..2)
            .map(|index| GpuInfo {
                index,
                name: "G".into(),
                vram_bytes: 1 << 30,
                pcie: None,
                p2p_peers: Vec::new(),
            })
            .collect();
        let mut g = HardwareGraph::default();
        g.devices.push(Device::new(
            "cpu0",
            DeviceKind::Cpu,
            "c",
            Tier::Ram,
            1 << 30,
        ));
        add_gpus(&DiscoverOpts::default(), &gpus, &mut g);
        apply_topology(&t, &gpus, &mut g);
        g.validate().unwrap();
        assert_eq!(g.device("gpu1").unwrap().pcie.unwrap().width_current, 8);
        assert_eq!(g.lane_groups.len(), 1);
        assert_eq!(g.lane_groups[0].id, "pcie-0000:01:00.0");
        assert_eq!(
            g.link_between("gpu0", "cpu0")
                .unwrap()
                .lane_group
                .as_deref(),
            Some("pcie-0000:01:00.0")
        );
        // No peer access: a staged link with the prior; both GPUs share a lane group, so halved.
        let staged = g.link_between("gpu0", "gpu1").unwrap();
        assert_eq!(staged.kind, LinkKind::HostStaged);
        let host = g.link_between("gpu0", "cpu0").unwrap();
        assert_eq!(staged.alpha_s.value, 2.0 * host.alpha_s.value);
        assert_eq!(
            staged.beta_bytes_per_s.value,
            host.beta_bytes_per_s.value / 2.0
        );
        let mut g2 = HardwareGraph::default();
        apply_topology(&t, &gpus[..1], &mut g2);
        assert_eq!(g2.notes.len(), 1, "count mismatch is noted");
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
