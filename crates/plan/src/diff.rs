//! Plan diff on the expanded form (`plan diff`).

use crate::{BufferKey, Expanded, Knobs, Placement, ShardKey};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum ShardChange {
    Add {
        key: ShardKey,
        to: Placement,
    },
    Remove {
        key: ShardKey,
        from: Placement,
    },
    /// Same key, different home, compute, format or stage.
    Change {
        key: ShardKey,
        from: Placement,
        to: Placement,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BufferChange {
    pub key: BufferKey,
    /// `None`: the buffer does not exist on that side.
    pub from: Option<u64>,
    pub to: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanDiff {
    /// In key order.
    pub shards: Vec<ShardChange>,
    /// In key order.
    pub buffers: Vec<BufferChange>,
    /// `(from, to)` when the knobs differ.
    pub knobs: Option<(Knobs, Knobs)>,
}

impl PlanDiff {
    pub fn is_empty(&self) -> bool {
        self.shards.is_empty() && self.buffers.is_empty() && self.knobs.is_none()
    }

    /// Weight bytes that must be written somewhere: added shards and changed shards whose home
    /// or format differs.
    pub fn moved_bytes(&self) -> u64 {
        todo!()
    }
}

pub fn diff(from: &Expanded, to: &Expanded) -> PlanDiff {
    let _ = (from, to);
    todo!()
}
