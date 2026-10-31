//! The placement plan as written by the planner or by hand (ADR 0027).

use crate::{Format, UnitId, UnitKind};
use serde::{Deserialize, Serialize};

/// Half-open range `[start, end)` of layers or experts, written `[start, end]` in TOML.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct Span(pub u32, pub u32);

impl Span {
    pub fn contains(self, x: u32) -> bool {
        self.0 <= x && x < self.1
    }

    pub fn len(self) -> u32 {
        self.1.saturating_sub(self.0)
    }

    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
}

/// Selects units: all three filters must match. An empty `kinds` matches every kind; embed and
/// head match only when `layers` is absent; non-expert units match only when `experts` is absent.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnitSel {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub kinds: Vec<UnitKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layers: Option<Span>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experts: Option<Span>,
}

impl UnitSel {
    pub fn matches(&self, u: UnitId) -> bool {
        (self.kinds.is_empty() || self.kinds.contains(&u.kind()))
            && match (self.layers, u.layer()) {
                (None, _) => true,
                (Some(s), Some(l)) => s.contains(l),
                (Some(_), None) => false,
            }
            && match (self.experts, u.expert()) {
                (None, _) => true,
                (Some(s), Some(e)) => s.contains(e),
                (Some(_), None) => false,
            }
    }
}

/// Where a set of units lives and runs. One home per shard; the shard count is `home.len()`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Assign {
    pub units: UnitSel,
    /// Device holding each shard's weights. Absent: open (partial plan).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub home: Option<Vec<String>>,
    /// Device computing each shard. Absent: each shard computes on its home.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compute: Option<Vec<String>>,
    /// Absent: open (partial plan).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<Format>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceBytes {
    pub device: String,
    pub bytes: u64,
}

/// Expert slot cache: `slots` expert slots per MoE layer of the stage on `device`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotCache {
    pub device: String,
    pub slots: u32,
}

/// Bytes a phase uses of a device's slot and workspace arenas (ADR 0031).
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct PhaseBytes {
    pub slots: u64,
    pub workspace: u64,
}

/// Per-phase split of the slot and workspace arenas on one device. Prefill borrows slot bytes
/// beyond the pinned hot set for workspace, decode gets them back. KV is never borrowed: each
/// phase stays within the stage's slot and workspace arenas on the device. A device without an
/// entry uses the arenas as sized (`expert_cache`, `workspace`) in both phases.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhaseBudget {
    pub device: String,
    pub prefill: PhaseBytes,
    pub decode: PhaseBytes,
}

/// One pipeline stage: a layer span computed by a device group.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stage {
    pub layers: Span,
    /// Compute device group. Empty: open (partial plan).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub devices: Vec<String>,
    /// Tensor parallel degree. Absent: open (partial plan).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tp: Option<u32>,
    /// Expert parallel degree. Absent: open (partial plan).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ep: Option<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub assign: Vec<Assign>,
    /// KV cache budget per device. Absent: open (partial plan).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv: Option<Vec<DeviceBytes>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub workspace: Vec<DeviceBytes>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub expert_cache: Vec<SlotCache>,
    /// Prefill and decode budgets within the slot and workspace arenas (ADR 0031).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub phase_budgets: Vec<PhaseBudget>,
}

/// A full copy of the model behind the router (data parallel).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Replica {
    pub stages: Vec<Stage>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Knobs {
    /// A prompt is cut into at most this many chunks that pipeline through the stages.
    #[serde(default = "one")]
    pub prefill_micro_batches: u32,
    #[serde(default = "yes")]
    pub cuda_graphs: bool,
    /// Sequences per step; absent: the scheduler decides.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_batch: Option<u32>,
}

fn one() -> u32 {
    1
}

fn yes() -> bool {
    true
}

impl Default for Knobs {
    fn default() -> Self {
        Self {
            prefill_micro_batches: 1,
            cuda_graphs: true,
            max_batch: None,
        }
    }
}

/// The placement plan: the contract between planner and executor (ADR 0006, 0027).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Plan {
    /// `ModelDesc::name` this plan places.
    pub model: String,
    #[serde(default)]
    pub knobs: Knobs,
    pub replicas: Vec<Replica>,
}

impl Plan {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selector_rules() {
        let all = UnitSel::default();
        assert!(all.matches(UnitId::Embed) && all.matches(UnitId::Expert(2, 5)));
        let layers = UnitSel {
            layers: Some(Span(0, 4)),
            ..Default::default()
        };
        assert!(layers.matches(UnitId::Attn(3)) && layers.matches(UnitId::Expert(3, 9)));
        assert!(!layers.matches(UnitId::Attn(4)) && !layers.matches(UnitId::Head));
        let experts = UnitSel {
            kinds: vec![UnitKind::Expert],
            layers: None,
            experts: Some(Span(0, 8)),
        };
        assert!(experts.matches(UnitId::Expert(40, 7)));
        assert!(!experts.matches(UnitId::Expert(40, 8)) && !experts.matches(UnitId::Mlp(1)));
    }
}
