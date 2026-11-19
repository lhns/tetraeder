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
    /// Rate of the routed-expert matmuls (`mul_mat_id`), fitted from MoE anchors (ADR 0070);
    /// `None` prices them at `flops_per_s`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expert_flops_per_s: Option<f64>,
}

impl ComputeSpec {
    /// The rate of the routed-expert matmuls: the fitted one, else `flops_per_s`.
    pub fn expert_flops(&self) -> f64 {
        self.expert_flops_per_s.unwrap_or(self.flops_per_s)
    }
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

/// Bytes per second at queue depth `depth` from a queue curve: piecewise linear between points
/// (in any order), proportional below the first, flat above the last. `None` for an empty curve.
/// The one formula of [`Device::disk_rate`], the planner and the I/O simulator (ADR 0056).
pub fn curve_rate(curve: &[QueuePoint], depth: u32) -> Option<f64> {
    let mut pts = curve.to_vec();
    pts.sort_by_key(|p| p.depth);
    let first = *pts.first()?;
    let last = pts[pts.len() - 1];
    let d = f64::from(depth);
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
        curve_rate(&self.queue_curve, depth)
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
    /// GPU to GPU without peer access: transfers stage through host memory (ADR 0070).
    HostStaged,
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

/// A collective the executor issues between ranks (ADR 0069, 0070).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CollectiveOp {
    /// `comm-probe` writes the variant names as they are; both spellings are read.
    #[serde(alias = "AllReduce")]
    AllReduce,
    /// One-way copy; `ranks` of its curve are `[src, dst]`.
    #[serde(alias = "Copy")]
    Copy,
}

/// Measured (or prior) time of a collective over a rank set by message size (`tetraeder
/// comm-probe`, ADR 0070). Never fitted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollectiveCurve {
    pub op: CollectiveOp,
    /// Device ids: a set for all-reduce, `[src, dst]` for a copy.
    pub ranks: Vec<String>,
    /// The path the executor used: `nccl`, `internal` or `fallback`; `prior` for formula points.
    pub comm: String,
    /// `(bytes, seconds)`.
    pub points: Vec<(f64, f64)>,
}

/// Seconds at `bytes` from `(bytes, seconds)` points in any order: piecewise linear between
/// points, flat (latency) below the first, linear in bytes above the last along the last
/// segment (proportional from the origin with a single point). `None` for no points.
pub fn curve_time(points: &[(f64, f64)], bytes: f64) -> Option<f64> {
    let mut pts = points.to_vec();
    pts.sort_by(|a, b| a.0.total_cmp(&b.0));
    let first = *pts.first()?;
    let last = pts[pts.len() - 1];
    if bytes <= first.0 {
        return Some(first.1);
    }
    if bytes >= last.0 {
        if pts.len() == 1 {
            return Some(last.1 * bytes / last.0);
        }
        let prev = pts[pts.len() - 2];
        let slope = ((last.1 - prev.1) / (last.0 - prev.0)).max(0.0);
        return Some(last.1 + slope * (bytes - last.0));
    }
    let i = pts.iter().position(|p| bytes <= p.0)?;
    let (a, b) = (pts[i - 1], pts[i]);
    Some(a.1 + (b.1 - a.1) * (bytes - a.0) / (b.0 - a.0))
}

/// Fixed costs of the multi-rank executor, fitted from anchors (ADR 0070); absent means the
/// priors of [`RangeCosts::PRIOR`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RangeCosts {
    /// Launch of one range as a captured CUDA graph.
    pub c_range_graph_s: f64,
    /// Launch of one range op by op.
    pub c_range_eager_s: f64,
    /// Input upload, end-of-step sync and readback.
    pub c_step_s: f64,
    /// Share of the probe's zero-byte all-reduce time that an all-reduce inside a step pays
    /// (ADR 0070, 2026-11-18): the probe syncs every rank around each call, a step does not. The
    /// byte-proportional part of the curve stays as measured. 1 (the default) is the curve as is.
    #[serde(default = "one")]
    pub allreduce_latency_share: f64,
    /// In-step latency `a` of an all-reduce (ADR 0078). When set, an all-reduce over `p` ranks
    /// in a step costs the curve's byte part `t(n) - t(0)` plus `a + 2 (p - 1) b`, `b` being
    /// `allreduce_step_s`, and `allreduce_latency_share` is not used. `None`: graphs fitted
    /// before cost model 3.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allreduce_latency_s: Option<f64>,
    /// Latency `b` per ring step (ADR 0078); fitted only when anchors have all-reduces over two
    /// or more group sizes, else `None` (0: every group pays the anchored latency).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allreduce_step_s: Option<f64>,
}

fn one() -> f64 {
    1.0
}

impl RangeCosts {
    pub const PRIOR: RangeCosts = RangeCosts {
        c_range_graph_s: 3.0e-5,
        c_range_eager_s: 1.0e-5,
        c_step_s: 2.0e-4,
        allreduce_latency_share: 1.0,
        allreduce_latency_s: None,
        allreduce_step_s: None,
    };
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct HardwareGraph {
    /// Fitted range and step costs of the multi-rank model; `None` uses the priors.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range_costs: Option<RangeCosts>,
    #[serde(default)]
    pub devices: Vec<Device>,
    #[serde(default)]
    pub links: Vec<Link>,
    #[serde(default)]
    pub lane_groups: Vec<LaneGroup>,
    /// Probe failures and caveats (e.g. a filesystem without O_DIRECT).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
    /// Collective curves between GPUs (ADR 0070).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub collectives: Vec<CollectiveCurve>,
}

impl HardwareGraph {
    /// Time of a collective of `bytes` over `ranks`: the matching curve interpolated by
    /// [`curve_time`]. Ranks match as a set for all-reduce and as an ordered pair for a copy.
    /// `None` without a matching curve.
    pub fn collective_time(&self, op: CollectiveOp, ranks: &[String], bytes: f64) -> Option<f64> {
        let same = |c: &CollectiveCurve| match op {
            CollectiveOp::AllReduce => {
                c.ranks.len() == ranks.len() && ranks.iter().all(|r| c.ranks.contains(r))
            }
            CollectiveOp::Copy => c.ranks == ranks,
        };
        let c = self.collectives.iter().find(|c| c.op == op && same(c))?;
        curve_time(&c.points, bytes)
    }

    /// The range and step costs in use.
    pub fn range_costs_or_prior(&self) -> RangeCosts {
        self.range_costs.unwrap_or(RangeCosts::PRIOR)
    }

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
        for c in &self.collectives {
            if let Some(d) = c.ranks.iter().find(|r| self.device(r).is_none()) {
                return Err(format!("collective curve names unknown device {d}"));
            }
            if c.op == CollectiveOp::Copy && c.ranks.len() != 2 {
                return Err("a copy curve names exactly two devices".to_string());
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
            expert_flops_per_s: None,
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

    fn curve(op: CollectiveOp, ranks: &[&str], points: &[(f64, f64)]) -> CollectiveCurve {
        CollectiveCurve {
            op,
            ranks: ranks.iter().map(|r| r.to_string()).collect(),
            comm: "internal".into(),
            points: points.to_vec(),
        }
    }

    fn ids(v: &[&str]) -> Vec<String> {
        v.iter().map(|r| r.to_string()).collect()
    }

    #[test]
    fn collective_time_interpolates_and_extrapolates() {
        let mut g = HardwareGraph::from_toml(FIXTURE).unwrap();
        // unsorted on purpose
        g.collectives = vec![
            curve(
                CollectiveOp::AllReduce,
                &["gpu0", "gpu1"],
                &[(4000.0, 3e-5), (1000.0, 1e-5), (2000.0, 1e-5)],
            ),
            curve(CollectiveOp::Copy, &["gpu0", "gpu1"], &[(1000.0, 2e-5)]),
        ];
        g.validate().unwrap();
        let ar = |b: f64| g.collective_time(CollectiveOp::AllReduce, &ids(&["gpu1", "gpu0"]), b);
        // flat below the first point (latency), exact at points, linear between
        assert_eq!(ar(10.0), Some(1e-5));
        assert_eq!(ar(1000.0), Some(1e-5));
        assert_eq!(ar(2000.0), Some(1e-5));
        assert!((ar(3000.0).unwrap() - 2e-5).abs() < 1e-18);
        // above the last point: the last segment's slope (1e-5 per 1000 bytes)
        assert!((ar(6000.0).unwrap() - 5e-5).abs() < 1e-18);
        // all-reduce matches as a set, a copy as an ordered pair
        let cp = |a: &[&str], b: f64| g.collective_time(CollectiveOp::Copy, &ids(a), b);
        assert_eq!(cp(&["gpu0", "gpu1"], 1000.0), Some(2e-5));
        assert_eq!(cp(&["gpu1", "gpu0"], 1000.0), None);
        // one point: proportional above it
        assert!((cp(&["gpu0", "gpu1"], 3000.0).unwrap() - 6e-5).abs() < 1e-18);
        assert_eq!(
            g.collective_time(CollectiveOp::AllReduce, &ids(&["gpu0"]), 1.0),
            None
        );
    }

    #[test]
    fn collectives_and_range_costs_roundtrip() {
        let mut g = HardwareGraph::from_toml(FIXTURE).unwrap();
        assert_eq!(g.range_costs_or_prior(), RangeCosts::PRIOR);
        g.collectives = vec![curve(
            CollectiveOp::AllReduce,
            &["gpu0", "gpu1"],
            &[(4096.0, 1e-5), (65536.0, 2e-5)],
        )];
        g.range_costs = Some(RangeCosts {
            c_range_graph_s: 1e-5,
            c_range_eager_s: 2e-5,
            c_step_s: 3e-4,
            allreduce_latency_share: 0.5,
            allreduce_latency_s: Some(1.2e-5),
            allreduce_step_s: Some(1e-6),
        });
        assert_eq!(HardwareGraph::from_toml(&g.to_toml().unwrap()).unwrap(), g);
        // range costs written before the latency share read it as 1
        let rc: RangeCosts = serde_json::from_str(
            r#"{"c_range_graph_s":1e-5,"c_range_eager_s":2e-5,"c_step_s":3e-4}"#,
        )
        .unwrap();
        assert_eq!(rc.allreduce_latency_share, 1.0);
        assert_eq!((rc.allreduce_latency_s, rc.allreduce_step_s), (None, None));
        assert_eq!(HardwareGraph::from_json(&g.to_json().unwrap()).unwrap(), g);
        // old files have neither field
        let old = HardwareGraph::from_toml(FIXTURE).unwrap();
        assert!(old.collectives.is_empty() && old.range_costs.is_none());
        let mut bad = g.clone();
        bad.collectives[0].ranks[0] = "nope".into();
        assert!(bad.validate().unwrap_err().contains("unknown device"));
    }

    #[test]
    fn validate_rejects_dangling_link() {
        let mut g = HardwareGraph::from_toml(FIXTURE).unwrap();
        g.links[0].b = "nope".into();
        assert!(g.validate().unwrap_err().contains("unknown device"));
    }
}
