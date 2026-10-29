//! Hardware discovery and measurement: per-op curves, tier bandwidth, link latency and bandwidth.
//!
//! `discover(&DiscoverOpts) -> HardwareGraph` runs on Windows and Linux. Probes repeat and report
//! the median with the spread. GPU discovery needs the `nvml` feature.

mod discover;
mod gpu;
mod graph;
mod probe;

pub use discover::{DeviceProbe, DiscoverOpts, discover, numa_node_count};
pub use gpu::{GpuInfo, discover_gpus, pcie_lane_bytes_per_s, pcie_prior_bytes_per_s};
pub use graph::{
    ComputeSpec, Device, DeviceKind, HardwareGraph, LaneGroup, Link, LinkKind, Measurement, Pcie,
    Source, Tier,
};
pub use probe::{probe_disk_read, probe_ram_read};
