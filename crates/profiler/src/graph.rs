//! Hardware graph: devices with tiers, links with alpha/beta, shared-lane groups.
//!
//! Every number carries its provenance (`Source`) and, when measured, the spread over repeats.

use serde::{Deserialize, Serialize};

/// Where a number came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    Measured,
    /// Spec-sheet or default value, used until a probe replaces it.
    Prior,
}

/// A scalar with provenance. For measured values `value` is the median over `samples` repeats and
/// `spread` is `(max - min) / median`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Measurement {
    pub value: f64,
    #[serde(default)]
    pub spread: f64,
    #[serde(default)]
    pub samples: u32,
    pub source: Source,
}

impl Measurement {
    pub fn prior(value: f64) -> Self {
        Self {
            value,
            spread: 0.0,
            samples: 0,
            source: Source::Prior,
        }
    }

    /// Median and relative spread of `samples`; `None` when empty.
    pub fn from_samples(samples: &[f64]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }
        let mut s = samples.to_vec();
        s.sort_by(f64::total_cmp);
        let n = s.len();
        let median = if n % 2 == 1 {
            s[n / 2]
        } else {
            (s[n / 2 - 1] + s[n / 2]) / 2.0
        };
        let spread = if median > 0.0 {
            (s[n - 1] - s[0]) / median
        } else {
            0.0
        };
        Some(Self {
            value: median,
            spread,
            samples: n as u32,
            source: Source::Measured,
        })
    }
}

/// Memory tiers of the plan (docs/plan.md, "Memory tiers").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    Vram,
    Ram,
    /// Local persistent storage.
    Nvme,
    Remote,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceKind {
    Cpu,
    Gpu,
    Disk,
}

/// Peak compute and fixed per-op cost: the roofline prior (measured curves replace it later).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ComputeSpec {
    pub flops_per_s: f64,
    /// Launch and sync cost per op with CUDA graphs (or the CPU equivalent).
    pub c0_graph_s: f64,
    /// Launch and sync cost per op without graphs.
    pub c0_eager_s: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pcie {
    pub gen_current: u32,
    pub width_current: u32,
    pub gen_max: u32,
    pub width_max: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Device {
    pub id: String,
    pub kind: DeviceKind,
    pub name: String,
    pub tier: Tier,
    pub capacity_bytes: u64,
    /// Sequential bandwidth of the tier this device holds (device memory, RAM, disk read).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bandwidth_bytes_per_s: Option<Measurement>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compute: Option<ComputeSpec>,
    /// CPU: physical cores.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cores: Option<u32>,
    /// CPU: logical threads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threads: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub numa_nodes: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pcie: Option<Pcie>,
    /// GPU: NVML index.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
    /// Disk: mount point of the volume.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mount: Option<String>,
    /// Disk: `ssd`, `hdd` or `unknown`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media: Option<String>,
}

impl Device {
    pub fn new(id: &str, kind: DeviceKind, name: &str, tier: Tier, capacity_bytes: u64) -> Self {
        Self {
            id: id.into(),
            kind,
            name: name.into(),
            tier,
            capacity_bytes,
            bandwidth_bytes_per_s: None,
            compute: None,
            cores: None,
            threads: None,
            numa_nodes: None,
            pcie: None,
            index: None,
            mount: None,
            media: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkKind {
    Pcie,
    /// GPU to GPU, direct (NVLink or PCIe peer-to-peer).
    P2p,
    Storage,
    Network,
}

/// Undirected link with latency alpha (seconds) and bandwidth beta (bytes per second).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Link {
    pub id: String,
    pub a: String,
    pub b: String,
    pub kind: LinkKind,
    pub alpha_s: Measurement,
    pub beta_bytes_per_s: Measurement,
    /// Links naming the same group share its lanes: their transfers contend.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lane_group: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LaneGroup {
    pub id: String,
    /// Aggregate bandwidth all member links can use together.
    pub capacity_bytes_per_s: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct HardwareGraph {
    #[serde(default)]
    pub devices: Vec<Device>,
    #[serde(default)]
    pub links: Vec<Link>,
    #[serde(default)]
    pub lane_groups: Vec<LaneGroup>,
    /// Probe failures and caveats (e.g. a filesystem without O_DIRECT).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

impl HardwareGraph {
    pub fn device(&self, id: &str) -> Option<&Device> {
        self.devices.iter().find(|d| d.id == id)
    }

    /// The link joining `a` and `b`, in either direction.
    pub fn link_between(&self, a: &str, b: &str) -> Option<&Link> {
        self.links
            .iter()
            .find(|l| (l.a == a && l.b == b) || (l.a == b && l.b == a))
    }

    pub fn lane_group(&self, id: &str) -> Option<&LaneGroup> {
        self.lane_groups.iter().find(|g| g.id == id)
    }

    pub fn to_toml(&self) -> Result<String, toml::ser::Error> {
        toml::to_string_pretty(self)
    }

    pub fn from_toml(s: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(s)
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }

    /// Structural check: unique ids, link endpoints and lane groups exist.
    pub fn validate(&self) -> Result<(), String> {
        for (i, d) in self.devices.iter().enumerate() {
            if self.devices[..i].iter().any(|o| o.id == d.id) {
                return Err(format!("duplicate device id {}", d.id));
            }
        }
        for l in &self.links {
            for end in [&l.a, &l.b] {
                if self.device(end).is_none() {
                    return Err(format!("link {} names unknown device {end}", l.id));
                }
            }
            if let Some(g) = &l.lane_group
                && self.lane_group(g).is_none()
            {
                return Err(format!("link {} names unknown lane group {g}", l.id));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("../fixtures/two_gpu_host.toml");

    #[test]
    fn median_and_spread() {
        let m = Measurement::from_samples(&[10.0, 30.0, 20.0]).unwrap();
        assert_eq!(m.value, 20.0);
        assert!((m.spread - 1.0).abs() < 1e-12);
        assert_eq!(m.samples, 3);
        let m = Measurement::from_samples(&[1.0, 2.0, 3.0, 10.0]).unwrap();
        assert_eq!(m.value, 2.5);
        assert!(Measurement::from_samples(&[]).is_none());
    }

    #[test]
    fn fixture_roundtrips_in_toml_and_json() {
        let g = HardwareGraph::from_toml(FIXTURE).unwrap();
        g.validate().unwrap();
        assert_eq!(g.devices.len(), 4);
        assert_eq!(g.link_between("gpu1", "cpu0").unwrap().id, "pcie-gpu1");
        assert_eq!(g.lane_group("chipset").unwrap().capacity_bytes_per_s, 8.0e9);
        let again = HardwareGraph::from_toml(&g.to_toml().unwrap()).unwrap();
        assert_eq!(g, again);
        let from_json = HardwareGraph::from_json(&g.to_json().unwrap()).unwrap();
        assert_eq!(g, from_json);
    }

    #[test]
    fn validate_rejects_dangling_link() {
        let mut g = HardwareGraph::from_toml(FIXTURE).unwrap();
        g.links[0].b = "nope".into();
        assert!(g.validate().unwrap_err().contains("unknown device"));
    }
}
