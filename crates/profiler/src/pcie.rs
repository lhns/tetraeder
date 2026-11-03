//! PCIe topology from Linux sysfs: upstream bridge chain and link state of each device, and the
//! lane groups of devices behind one bridge (ADR 0040, "PCIe topology").
//!
//! The reader takes the sysfs root as a path so tests can use a fake tree. It finds each device's
//! real directory by walking `<root>/devices` without following symlinks, so it works on a tree
//! built from plain directories too.

use crate::gpu::pcie_prior_bytes_per_s;
use crate::graph::{LaneGroup, Pcie};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcieClass {
    Gpu,
    Nvme,
    Other,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PcieDevice {
    pub bdf: String,
    pub class: PcieClass,
    /// Bridge BDFs above this device, closest first; the last one is the root port.
    pub upstream: Vec<String>,
    pub link: Option<Pcie>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct PcieTopology {
    pub devices: Vec<PcieDevice>,
}

/// `0000:01:00.0`: domain:bus:device.function.
fn is_bdf(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() == 12
        && b[4] == b':'
        && b[7] == b':'
        && b[10] == b'.'
        && b.iter()
            .enumerate()
            .all(|(i, c)| matches!(i, 4 | 7 | 10) || c.is_ascii_hexdigit())
}

/// BDF directory -> path, for every PCI device directory below `dir`. Symlinks are not followed.
fn walk(dir: &Path, out: &mut HashMap<String, PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.filter_map(Result::ok) {
        if !e.file_type().is_ok_and(|t| t.is_dir()) {
            continue;
        }
        let name = e.file_name().to_string_lossy().into_owned();
        if is_bdf(&name) {
            out.insert(name, e.path());
        }
        walk(&e.path(), out);
    }
}

fn read_trim(p: &Path) -> Option<String> {
    std::fs::read_to_string(p)
        .ok()
        .map(|s| s.trim().to_string())
}

/// "16.0 GT/s PCIe" -> 4.
fn parse_gen(s: &str) -> Option<u32> {
    let gt: f64 = s.split_whitespace().next()?.parse().ok()?;
    [2.5, 5.0, 8.0, 16.0, 32.0, 64.0]
        .iter()
        .position(|&g| (g - gt).abs() < 0.05)
        .map(|i| i as u32 + 1)
}

fn parse_link(dir: &Path) -> Option<Pcie> {
    let get = |name: &str| read_trim(&dir.join(name));
    Some(Pcie {
        gen_current: parse_gen(&get("current_link_speed")?)?,
        width_current: get("current_link_width")?.parse().ok()?,
        gen_max: parse_gen(&get("max_link_speed")?)?,
        width_max: get("max_link_width")?.parse().ok()?,
    })
}

fn parse_class(dir: &Path) -> PcieClass {
    let code = read_trim(&dir.join("class"))
        .and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok())
        .unwrap_or(0);
    match code >> 8 {
        0x0300..=0x03ff => PcieClass::Gpu,
        0x0108 => PcieClass::Nvme,
        _ => PcieClass::Other,
    }
}

/// Reads `<sys_root>/bus/pci/devices/*`. A missing tree gives an empty topology.
pub fn pcie_topology(sys_root: &Path) -> PcieTopology {
    let Ok(rd) = std::fs::read_dir(sys_root.join("bus/pci/devices")) else {
        return PcieTopology::default();
    };
    let mut bdfs: Vec<String> = rd
        .filter_map(Result::ok)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| is_bdf(n))
        .collect();
    bdfs.sort();
    let mut real = HashMap::new();
    walk(&sys_root.join("devices"), &mut real);
    let devices = bdfs
        .into_iter()
        .map(|bdf| {
            let (dir, upstream) = match real.get(&bdf) {
                Some(p) => {
                    let up = p
                        .ancestors()
                        .skip(1)
                        .filter_map(|a| a.file_name()?.to_str())
                        .take_while(|n| is_bdf(n))
                        .map(str::to_string)
                        .collect();
                    (p.clone(), up)
                }
                None => (sys_root.join("bus/pci/devices").join(&bdf), Vec::new()),
            };
            PcieDevice {
                class: parse_class(&dir),
                link: parse_link(&dir),
                bdf,
                upstream,
            }
        })
        .collect();
    PcieTopology { devices }
}

/// GPUs and NVMe drives that share a non-root bridge form a group. The group is keyed by the
/// topmost non-root bridge above them (the one whose uplink all their traffic crosses) and has
/// that bridge's link bandwidth, in the units of `pcie_prior_bytes_per_s`. Groups with fewer than
/// two members, or whose bridge link is unknown, are dropped.
pub fn lane_groups(t: &PcieTopology) -> Vec<(LaneGroup, Vec<String>)> {
    let mut groups: Vec<(String, Vec<String>)> = Vec::new();
    for d in t.devices.iter().filter(|d| d.class != PcieClass::Other) {
        // The last entry is the root port, which is not shared lane capacity.
        let Some(top) = d.upstream.len().checked_sub(2).map(|i| &d.upstream[i]) else {
            continue;
        };
        match groups.iter_mut().find(|g| g.0 == *top) {
            Some(g) => g.1.push(d.bdf.clone()),
            None => groups.push((top.clone(), vec![d.bdf.clone()])),
        }
    }
    groups
        .into_iter()
        .filter(|g| g.1.len() >= 2)
        .filter_map(|(bridge, members)| {
            let link = t.devices.iter().find(|d| d.bdf == bridge)?.link?;
            Some((
                LaneGroup {
                    id: format!("pcie-{bridge}"),
                    capacity_bytes_per_s: pcie_prior_bytes_per_s(&link),
                },
                members,
            ))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Windows cannot hold ':' in directory names, so the fake sysfs tree tests are unix only; the
    // grouping logic is covered everywhere by `lane_groups_from_a_literal_topology`.
    #[cfg(unix)]
    use std::fs;

    #[cfg(unix)]
    const SW: &str = "pci0000:00/0000:00:01.0/0000:01:00.0";

    /// (path under devices/, class, current speed, current width, max speed, max width).
    #[cfg(unix)]
    type Node = (
        String,
        &'static str,
        &'static str,
        &'static str,
        &'static str,
        &'static str,
    );

    #[cfg(unix)]
    fn nodes() -> Vec<Node> {
        let g4 = "16.0 GT/s PCIe";
        let n = |p: &str, c, s, w, ms, mw| (p.to_string(), c, s, w, ms, mw);
        vec![
            n("pci0000:00/0000:00:01.0", "0x060400", g4, "16", g4, "16"),
            n(SW, "0x060400", g4, "8", g4, "16"),
            n(&format!("{SW}/0000:02:00.0"), "0x060400", g4, "8", g4, "8"),
            n(&format!("{SW}/0000:02:08.0"), "0x060400", g4, "4", g4, "4"),
            n(
                &format!("{SW}/0000:02:00.0/0000:03:00.0"),
                "0x030200",
                "8.0 GT/s PCIe",
                "8",
                g4,
                "16",
            ),
            n(
                &format!("{SW}/0000:02:08.0/0000:04:00.0"),
                "0x010802",
                g4,
                "4",
                g4,
                "4",
            ),
            n("pci0000:00/0000:00:02.0", "0x060400", g4, "16", g4, "16"),
            n(
                "pci0000:00/0000:00:02.0/0000:05:00.0",
                "0x030000",
                g4,
                "16",
                g4,
                "16",
            ),
        ]
    }

    /// Root port 00:01.0 -> switch 01:00.0 -> ports 02:00.0, 02:08.0 -> GPU 03:00.0 and NVMe
    /// 04:00.0; GPU 05:00.0 directly behind root port 00:02.0.
    #[cfg(unix)]
    fn build(root: &Path, symlinks: bool) {
        fs::create_dir_all(root.join("bus/pci/devices")).unwrap();
        for (path, class, speed, width, max_speed, max_width) in nodes() {
            let dir = root.join("devices").join(&path);
            fs::create_dir_all(&dir).unwrap();
            for (f, v) in [
                ("class", class),
                ("current_link_speed", speed),
                ("current_link_width", width),
                ("max_link_speed", max_speed),
                ("max_link_width", max_width),
            ] {
                fs::write(dir.join(f), format!("{v}\n")).unwrap();
            }
            let entry = root
                .join("bus/pci/devices")
                .join(path.rsplit('/').next().unwrap());
            if symlinks {
                #[cfg(unix)]
                std::os::unix::fs::symlink(&dir, &entry).unwrap();
            } else {
                fs::create_dir_all(entry).unwrap();
            }
        }
    }

    #[cfg(unix)]
    fn check(symlinks: bool) {
        let root = std::env::temp_dir().join(format!(
            "tetraeder-pcie-test-{}-{symlinks}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        build(&root, symlinks);
        let t = pcie_topology(&root);
        let _ = fs::remove_dir_all(&root);
        assert_eq!(t.devices.len(), 8);
        let get = |b: &str| t.devices.iter().find(|d| d.bdf == b).unwrap();
        let gpu = get("0000:03:00.0");
        assert_eq!(gpu.class, PcieClass::Gpu);
        assert_eq!(
            gpu.upstream,
            ["0000:02:00.0", "0000:01:00.0", "0000:00:01.0"]
        );
        let l = gpu.link.unwrap();
        assert_eq!(
            (l.gen_current, l.width_current, l.gen_max, l.width_max),
            (3, 8, 4, 16)
        );
        assert_eq!(get("0000:04:00.0").class, PcieClass::Nvme);
        assert_eq!(get("0000:01:00.0").class, PcieClass::Other);
        assert_eq!(get("0000:05:00.0").upstream, ["0000:00:02.0"]);

        let groups = lane_groups(&t);
        assert_eq!(
            groups.len(),
            1,
            "the lone GPU behind a root port forms no group"
        );
        let (g, members) = &groups[0];
        assert_eq!(g.id, "pcie-0000:01:00.0");
        assert_eq!(members, &["0000:03:00.0", "0000:04:00.0"]);
        // switch uplink: gen4 x8 at 80% of the theoretical rate
        assert!((g.capacity_bytes_per_s - 0.8 * 1969.0e6 * 8.0).abs() < 1.0);
    }

    #[cfg(unix)]
    #[test]
    fn topology_from_plain_directory_tree() {
        check(false);
    }

    #[cfg(unix)]
    #[test]
    fn topology_from_symlinked_tree() {
        check(true);
    }

    #[test]
    fn lane_groups_from_a_literal_topology() {
        let g4 = Pcie {
            gen_current: 4,
            width_current: 8,
            gen_max: 4,
            width_max: 16,
        };
        let dev = |bdf: &str, class, up: &[&str]| PcieDevice {
            bdf: bdf.into(),
            class,
            upstream: up.iter().map(|s| s.to_string()).collect(),
            link: Some(g4),
        };
        let t = PcieTopology {
            devices: vec![
                dev("0000:01:00.0", PcieClass::Other, &["0000:00:01.0"]),
                dev(
                    "0000:03:00.0",
                    PcieClass::Gpu,
                    &["0000:02:00.0", "0000:01:00.0", "0000:00:01.0"],
                ),
                dev(
                    "0000:04:00.0",
                    PcieClass::Nvme,
                    &["0000:02:08.0", "0000:01:00.0", "0000:00:01.0"],
                ),
                dev("0000:05:00.0", PcieClass::Gpu, &["0000:00:02.0"]),
            ],
        };
        let groups = lane_groups(&t);
        assert_eq!(
            groups.len(),
            1,
            "the lone GPU behind a root port forms no group"
        );
        assert_eq!(groups[0].0.id, "pcie-0000:01:00.0");
        assert_eq!(groups[0].1, ["0000:03:00.0", "0000:04:00.0"]);
        assert!((groups[0].0.capacity_bytes_per_s - 0.8 * 1969.0e6 * 8.0).abs() < 1.0);
    }

    #[test]
    fn missing_tree_is_empty_and_speeds_parse() {
        assert!(
            pcie_topology(Path::new("/definitely/not/here"))
                .devices
                .is_empty()
        );
        assert_eq!(parse_gen("2.5 GT/s PCIe"), Some(1));
        assert_eq!(parse_gen("64.0 GT/s PCIe"), Some(6));
        assert_eq!(parse_gen("Unknown"), None);
    }
}
