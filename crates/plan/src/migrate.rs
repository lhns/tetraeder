//! Migration DAG skeleton (ADR 0006, 0009, 0027): a diff compiled into ordered steps.
//!
//! Drain first; per shard, load the new placement before freeing the old one (shards are
//! independent of each other); buffer and knob changes; then prepare and commit (two-phase).
//! Until Phase 7 the executor runs it drain-and-restart. Memory-aware ordering is Phase 7.

use crate::{BufferKey, Expanded, Knobs, Placement, PlanDiff, ShardKey};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Step {
    Drain,
    Load {
        key: ShardKey,
        to: Placement,
    },
    Free {
        key: ShardKey,
        from: Placement,
    },
    /// `None` removes the buffer.
    SetBuffer {
        key: BufferKey,
        bytes: Option<u64>,
    },
    SetKnobs(Knobs),
    Prepare,
    Commit,
}

/// Steps plus edges `(before, after)` as indices into `steps`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MigrationDag {
    pub steps: Vec<Step>,
    pub edges: Vec<(usize, usize)>,
}

/// What the executor holds while a migration runs. A shard key can briefly have two copies
/// (between its load and its free), so placements per key are a list.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct State {
    pub shards: BTreeMap<ShardKey, Vec<Placement>>,
    pub buffers: BTreeMap<BufferKey, u64>,
    pub knobs: Knobs,
}

impl From<&Expanded> for State {
    fn from(e: &Expanded) -> Self {
        Self {
            shards: e
                .shards
                .iter()
                .map(|(k, p)| (*k, vec![p.clone()]))
                .collect(),
            buffers: e.buffers.clone(),
            knobs: e.knobs,
        }
    }
}

impl State {
    /// Apply one step. Freeing a placement the state does not hold is an error.
    pub fn apply(&mut self, step: &Step) -> Result<(), String> {
        let _ = step;
        todo!()
    }
}

impl MigrationDag {
    /// Compile a diff. Every step lies on a path from the single `Drain` to the single `Commit`.
    pub fn from_diff(diff: &PlanDiff) -> Self {
        let _ = diff;
        todo!()
    }

    /// One topological order; `pick` chooses among the ready steps (index into the ready list,
    /// taken modulo its length), so a test can drive random orders. `None` if there is a cycle.
    pub fn order_with(&self, mut pick: impl FnMut(usize) -> usize) -> Option<Vec<usize>> {
        let _ = &mut pick;
        todo!()
    }
}
