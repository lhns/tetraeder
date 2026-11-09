//! The detector input. One JSON document per run; every field except `windows` may be omitted.
//!
//! Resource ids are strings with a kind prefix: `gpu:<dev>`, `cpu`, `ram`, `disk:<dev>`,
//! `link:<id>`, `host`. Device and link ids are the ones of the hardware graph.
//! See `fixtures/observation_example.json` for a documented example.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// One measured run at one concurrency.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    pub id: String,
    /// Batch size or concurrency. 1 means serialized stages, more means resources overlap.
    #[serde(default = "one")]
    pub batch: u32,
    /// Repeated measurement windows of the same steady state (8 or more give a useful interval).
    pub windows: Vec<Window>,
    #[serde(default)]
    pub env: EnvSignals,
    /// Calibrated expectations; without them only the environment and share checks run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected: Option<Expected>,
}

fn one() -> u32 {
    1
}

/// One window of steady-state decoding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Window {
    pub tokens: u64,
    pub wall_s: f64,
    /// Resource id to seconds the resource was busy in this window.
    pub busy_s: BTreeMap<String, f64>,
    /// Resource id to work units moved in this window: bytes for memory, disk and links, tokens
    /// handled for `host`. `bytes / busy_s` is the measured rate compared with `expected`.
    #[serde(default)]
    pub bytes: BTreeMap<String, f64>,
    /// Step time split in seconds: `compute`, `cpu_experts`, `expert_load_wait`, `transfer_wait`,
    /// `host`. Unknown keys are kept and ignored.
    #[serde(default)]
    pub split_s: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThrottleReason {
    Thermal,
    Power,
    Clocks,
    /// Software cap such as a power limit set with nvidia-smi.
    Sw,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Throttle {
    /// Device id of the GPU (resource `gpu:<device>`).
    pub device: String,
    pub reason: ThrottleReason,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PcieSignal {
    /// Device id of the GPU.
    pub device: String,
    /// Link id of its PCIe link (resource `link:<link>`). Empty means `pcie-<device>`.
    #[serde(default)]
    pub link: String,
    pub gen_current: u32,
    pub width_current: u32,
    pub gen_max: u32,
    pub width_max: u32,
}

impl PcieSignal {
    pub fn resource(&self) -> String {
        if self.link.is_empty() {
            format!("link:pcie-{}", self.device)
        } else {
            format!("link:{}", self.link)
        }
    }

    /// Bandwidth now over bandwidth at the maximum; each generation doubles it.
    pub fn bandwidth_ratio(&self) -> f64 {
        let width = f64::from(self.width_current) / f64::from(self.width_max.max(1));
        let generation = 2f64.powi(self.gen_current as i32 - self.gen_max as i32);
        width * generation
    }
}

/// Signals about the machine around the run. Absent signals mean "not measured".
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EnvSignals {
    #[serde(default)]
    pub throttle: Vec<Throttle>,
    #[serde(default)]
    pub pcie: Vec<PcieSignal>,
    /// CPU quota of the container or host, in cores.
    #[serde(default)]
    pub cpu_quota_cores: f64,
    /// Cores used by other processes.
    #[serde(default)]
    pub cpu_other_cores: f64,
    #[serde(default)]
    pub swap_in_bytes_per_s: f64,
    /// Major page faults per second of the engine's processes (pages read back from swap or file).
    #[serde(default)]
    pub major_faults_per_s: f64,
    /// Share of wall time in which some engine task stalled on memory (PSI `memory.pressure`
    /// "some" of the engine's cgroup). `None` when the kernel does not expose PSI.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_stall_share: Option<f64>,
    /// GPU device id to VRAM bytes held by other processes.
    #[serde(default)]
    pub vram_other_bytes: BTreeMap<String, u64>,
    #[serde(default)]
    pub vram_total_bytes: BTreeMap<String, u64>,
    /// Disk read rate of other processes.
    #[serde(default)]
    pub disk_other_read_bytes_per_s: f64,
    /// `io.max` read limit of the cgroup, if set.
    #[serde(default)]
    pub io_max_bytes_per_s: Option<f64>,
    #[serde(default)]
    pub expert_hit_rate: Option<f64>,
    #[serde(default)]
    pub slots_planned: Option<u32>,
    #[serde(default)]
    pub slots_actual: Option<u32>,
}

/// What the calibrated cost model expects.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Expected {
    /// Resource id to expected work units per busy second (same unit as `Window::bytes`).
    pub rates: BTreeMap<String, f64>,
    /// Relative error band from the calibration anchors; a rate this far below expected is a fault.
    #[serde(default = "default_band")]
    pub band: f64,
}

fn default_band() -> f64 {
    0.1
}
