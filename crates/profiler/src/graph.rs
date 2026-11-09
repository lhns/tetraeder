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

/// CPU rates at one thread count (a point of the thread curve).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThreadPoint {
    pub threads: u32,
    pub flops_per_s: f64,
    pub bytes_per_s: f64,
}

/// Disk read rate at one queue depth (a point of the queue curve, ADR 0056).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QueuePoint {
    pub depth: u32,
    pub bytes_per_s: f64,
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
    /// Disk: the chunk size the queue curve was measured with. Kept before the array-of-tables
    /// fields so TOML output stays valid.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queue_chunk_bytes: Option<u64>,
    /// CPU: measured or fitted rates per thread count; empty means use `compute` and
    /// `bandwidth_bytes_per_s` scaled by threads.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub thread_curve: Vec<ThreadPoint>,
    /// Disk: measured read bytes per second by queue depth at `queue_chunk_bytes` per read
    /// (`tetraeder-bench disk-curve`); empty means use `bandwidth_bytes_per_s`. Measured, never
    /// fitted: anchors cannot separate disk time from CPU time.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub queue_curve: Vec<QueuePoint>,
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
            thread_curve: Vec::new(),
            queue_curve: Vec::new(),
            queue_chunk_bytes: None,
        }
    }

    /// Disk read bytes per second at queue depth `depth`. From the queue curve when present:
    /// piecewise linear between points, proportional below the first, flat above the last (like
    /// [`Device::cpu_rates`]). Otherwise the device's `bandwidth_bytes_per_s`; `None` without one.
    pub fn disk_rate(&self, depth: u32) -> Option<f64> {
        if self.queue_curve.is_empty() {
            return self.bandwidth_bytes_per_s.map(|b| b.value);
        }
        let mut pts = self.queue_curve.clone();
        pts.sort_by_key(|p| p.depth);
        let d = f64::from(depth);
        let first = pts[0];
        let last = pts[pts.len() - 1];
        if d <= f64::from(first.depth) {
            return Some(first.bytes_per_s * d / f64::from(first.depth.max(1)));
        }
        if d >= f64::from(last.depth) {
            return Some(last.bytes_per_s);
        }
        let i = pts.iter().position(|p| d <= f64::from(p.depth))?;
        let (a, b) = (pts[i - 1], pts[i]);
        let w = (d - f64::from(a.depth)) / f64::from(b.depth - a.depth);
        Some(a.bytes_per_s + (b.bytes_per_s - a.bytes_per_s) * w)
    }

    /// `(flops/s, bytes/s)` at `t` threads. From the thread curve when present: piecewise linear
    /// between points, proportional below the first, flat above the last. Otherwise the device's
    /// `compute` and bandwidth scaled by `min(t, threads) / threads`. `None` when neither exists.
    pub fn cpu_rates(&self, t: u32) -> Option<(f64, f64)> {
        if self.thread_curve.is_empty() {
            let f = self.compute?.flops_per_s;
            let b = self.bandwidth_bytes_per_s?.value;
            let scale = match self.threads {
                Some(n) if n > 0 => f64::from(t.min(n)) / f64::from(n),
                _ => 1.0,
            };
            return Some((f * scale, b * scale));
        }
        let mut pts = self.thread_curve.clone();
        pts.sort_by_key(|p| p.threads);
        let t = f64::from(t);
        let first = pts[0];
        let last = pts[pts.len() - 1];
        if t <= f64::from(first.threads) {
            let k = t / f64::from(first.threads.max(1));
            return Some((first.flops_per_s * k, first.bytes_per_s * k));
        }
        if t >= f64::from(last.threads) {
            return Some((last.flops_per_s, last.bytes_per_s));
        }
        let i = pts.iter().position(|p| t <= f64::from(p.threads))?;
        let (a, b) = (pts[i - 1], pts[i]);
        let w = (t - f64::from(a.threads)) / f64::from(b.threads - a.threads);
        Some((
            a.flops_per_s + (b.flops_per_s - a.flops_per_s) * w,
            a.bytes_per_s + (b.bytes_per_s - a.bytes_per_s) * w,
        ))
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
    fn cpu_rates_from_curve_and_fallback() {
        let mut d = Device::new("cpu0", DeviceKind::Cpu, "c", Tier::Ram, 1 << 30);
        assert!(d.cpu_rates(4).is_none());
        d.threads = Some(8);
        d.compute = Some(ComputeSpec {
            flops_per_s: 8e11,
            c0_graph_s: 1e-6,
            c0_eager_s: 1e-6,
        });
        d.bandwidth_bytes_per_s = Some(Measurement::prior(8e10));
        assert_eq!(d.cpu_rates(4), Some((4e11, 4e10)));
        assert_eq!(d.cpu_rates(16), Some((8e11, 8e10)));
        d.thread_curve = vec![
            ThreadPoint {
                threads: 8,
                flops_per_s: 6e11,
                bytes_per_s: 5e10,
            },
            ThreadPoint {
                threads: 4,
                flops_per_s: 4e11,
                bytes_per_s: 4e10,
            },
        ];
        assert_eq!(d.cpu_rates(2), Some((2e11, 2e10)));
        assert_eq!(d.cpu_rates(6), Some((5e11, 4.5e10)));
        assert_eq!(d.cpu_rates(32), Some((6e11, 5e10)));
        let again = HardwareGraph {
            devices: vec![d],
            ..Default::default()
        };
        assert_eq!(
            HardwareGraph::from_toml(&again.to_toml().unwrap()).unwrap(),
            again
        );
    }

    #[test]
    fn disk_rate_from_curve_and_fallback() {
        let mut d = Device::new("nvme0", DeviceKind::Disk, "n", Tier::Nvme, 1 << 40);
        assert_eq!(d.disk_rate(4), None);
        d.bandwidth_bytes_per_s = Some(Measurement::prior(3e9));
        assert_eq!(d.disk_rate(1), Some(3e9));
        assert_eq!(d.disk_rate(64), Some(3e9));
        // unsorted on purpose
        d.queue_curve = vec![
            QueuePoint {
                depth: 4,
                bytes_per_s: 3e9,
            },
            QueuePoint {
                depth: 2,
                bytes_per_s: 2e9,
            },
        ];
        d.queue_chunk_bytes = Some(1 << 20);
        assert_eq!(d.disk_rate(1), Some(1e9));
        assert_eq!(d.disk_rate(2), Some(2e9));
        assert_eq!(d.disk_rate(3), Some(2.5e9));
        assert_eq!(d.disk_rate(32), Some(3e9));
        let again = HardwareGraph {
            devices: vec![d],
            ..Default::default()
        };
        assert_eq!(
            HardwareGraph::from_toml(&again.to_toml().unwrap()).unwrap(),
            again
        );
    }

    #[test]
    fn validate_rejects_dangling_link() {
        let mut g = HardwareGraph::from_toml(FIXTURE).unwrap();
        g.links[0].b = "nope".into();
        assert!(g.validate().unwrap_err().contains("unknown device"));
    }
}
