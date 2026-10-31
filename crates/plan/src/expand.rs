//! The expanded form of a complete plan: one entry per shard plus byte buffers per device.
//! Validator, diff and migration DAG all work on it (ADR 0027).

use crate::{Format, Knobs, UnitId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Identity of one shard. The shard count is part of the key, so a degree change is a removal
/// plus an addition, never an in-place change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ShardKey {
    pub replica: u32,
    pub unit: UnitId,
    pub shard: u32,
    pub of: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Placement {
    pub stage: u32,
    pub home: String,
    /// Always set: the home when the plan names no compute device.
    pub compute: String,
    pub format: Format,
    /// `ceil(format.bytes(params) / of)`.
    pub bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BufferKind {
    Kv,
    Workspace,
    /// Expert slot cache: slots x MoE layers of the stage x largest expert bytes in the stage.
    SlotCache,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BufferKey {
    pub replica: u32,
    pub stage: u32,
    pub kind: BufferKind,
    pub device: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Expanded {
    pub shards: BTreeMap<ShardKey, Placement>,
    /// Bytes per buffer; entries with the same key in one stage are summed.
    pub buffers: BTreeMap<BufferKey, u64>,
    pub knobs: Knobs,
}

impl Expanded {
    /// Bytes held per device: shard weights at their home plus buffers (saturating).
    pub fn memory_per_device(&self) -> BTreeMap<String, u64> {
        let mut m: BTreeMap<String, u64> = BTreeMap::new();
        for p in self.shards.values() {
            let used = m.entry(p.home.clone()).or_default();
            *used = used.saturating_add(p.bytes);
        }
        for (k, b) in &self.buffers {
            let used = m.entry(k.device.clone()).or_default();
            *used = used.saturating_add(*b);
        }
        m
    }
}
