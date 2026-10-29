//! Which plan fields each phase can execute (docs/plan.md, "What each phase can execute").

use serde::{Deserialize, Serialize};

/// Roadmap phase of the executing engine; 3 stands for both 3a and 3b.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Phase(pub u8);

/// Where a shard may compute relative to its home.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeOverride {
    /// Compute follows the weight buffer.
    None,
    /// Only experts may compute away from their home (streaming, own MoE builder).
    ExpertsOnly,
    Any,
}

/// Fields of the plan a phase can execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capabilities {
    /// Formats other than the unit's source format (load-time conversion).
    pub convert_formats: bool,
    /// Expert homes on disk devices and expert slot caches.
    pub expert_tiers: bool,
    pub compute_override: ComputeOverride,
    /// Tensor and expert parallel degrees above 1.
    pub tp_ep: bool,
    /// More than one replica and devices with tier `Remote`.
    pub multi_node: bool,
}

impl Phase {
    pub fn capabilities(self) -> Capabilities {
        let p = self.0;
        Capabilities {
            convert_formats: p >= 2,
            expert_tiers: p >= 2,
            compute_override: match p {
                0..=1 => ComputeOverride::None,
                2..=4 => ComputeOverride::ExpertsOnly,
                _ => ComputeOverride::Any,
            },
            tp_ep: p >= 5,
            multi_node: p >= 6,
        }
    }
}
