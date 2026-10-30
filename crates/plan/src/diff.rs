//! Plan diff on the expanded form (`plan diff`).

use crate::{BufferKey, Expanded, Knobs, Placement, ShardKey};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

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
        self.shards
            .iter()
            .map(|c| match c {
                ShardChange::Add { to, .. } => to.bytes,
                ShardChange::Change { from, to, .. }
                    if from.home != to.home || from.format != to.format =>
                {
                    to.bytes
                }
                _ => 0,
            })
            .sum()
    }
}

pub fn diff(from: &Expanded, to: &Expanded) -> PlanDiff {
    let keys: BTreeSet<&ShardKey> = from.shards.keys().chain(to.shards.keys()).collect();
    let shards = keys
        .into_iter()
        .filter_map(|k| match (from.shards.get(k), to.shards.get(k)) {
            (None, Some(t)) => Some(ShardChange::Add {
                key: *k,
                to: t.clone(),
            }),
            (Some(f), None) => Some(ShardChange::Remove {
                key: *k,
                from: f.clone(),
            }),
            (Some(f), Some(t)) if f != t => Some(ShardChange::Change {
                key: *k,
                from: f.clone(),
                to: t.clone(),
            }),
            _ => None,
        })
        .collect();
    let keys: BTreeSet<&BufferKey> = from.buffers.keys().chain(to.buffers.keys()).collect();
    let buffers = keys
        .into_iter()
        .filter_map(|k| {
            let (f, t) = (from.buffers.get(k).copied(), to.buffers.get(k).copied());
            (f != t).then(|| BufferChange {
                key: k.clone(),
                from: f,
                to: t,
            })
        })
        .collect();
    PlanDiff {
        shards,
        buffers,
        knobs: (from.knobs != to.knobs).then_some((from.knobs, to.knobs)),
    }
}
