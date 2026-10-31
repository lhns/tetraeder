//! Migration DAG skeleton (ADR 0006, 0009, 0027): a diff compiled into ordered steps.
//!
//! Drain first; per shard, load the new placement before freeing the old one (shards are
//! independent of each other); buffer and knob changes; then prepare and commit (two-phase).
//! Until Phase 7 the executor runs it drain-and-restart. Memory-aware ordering is Phase 7.

use crate::{
    BudgetKey, Budgets, BufferKey, Expanded, Knobs, Placement, PlanDiff, ShardChange, ShardKey,
};
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
    /// `None` removes the budgets.
    SetBudgets {
        key: BudgetKey,
        budgets: Option<Budgets>,
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
    pub budgets: BTreeMap<BudgetKey, Budgets>,
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
            budgets: e.budgets.clone(),
            knobs: e.knobs,
        }
    }
}

impl State {
    /// Apply one step. Freeing a placement the state does not hold is an error.
    pub fn apply(&mut self, step: &Step) -> Result<(), String> {
        match step {
            Step::Load { key, to } => self.shards.entry(*key).or_default().push(to.clone()),
            Step::Free { key, from } => {
                let list = self.shards.get_mut(key);
                let pos = list.as_ref().and_then(|l| l.iter().position(|p| p == from));
                let (Some(list), Some(pos)) = (list, pos) else {
                    return Err(format!("free of a placement not held: {key:?}"));
                };
                list.remove(pos);
                if list.is_empty() {
                    self.shards.remove(key);
                }
            }
            Step::SetBuffer { key, bytes } => match bytes {
                Some(b) => {
                    self.buffers.insert(key.clone(), *b);
                }
                None => {
                    self.buffers.remove(key);
                }
            },
            Step::SetBudgets { key, budgets } => match budgets {
                Some(b) => {
                    self.budgets.insert(key.clone(), *b);
                }
                None => {
                    self.budgets.remove(key);
                }
            },
            Step::SetKnobs(k) => self.knobs = *k,
            Step::Drain | Step::Prepare | Step::Commit => {}
        }
        Ok(())
    }
}

impl MigrationDag {
    /// Compile a diff. Every step lies on a path from the single `Drain` to the single `Commit`.
    pub fn from_diff(diff: &PlanDiff) -> Self {
        let mut steps = vec![Step::Drain];
        // (first, last) step of each independent chain
        let mut chains: Vec<(usize, usize)> = Vec::new();
        let mut edges = Vec::new();
        for c in &diff.shards {
            let at = steps.len();
            match c {
                ShardChange::Add { key, to } => steps.push(Step::Load {
                    key: *key,
                    to: to.clone(),
                }),
                ShardChange::Remove { key, from } => steps.push(Step::Free {
                    key: *key,
                    from: from.clone(),
                }),
                ShardChange::Change { key, from, to } => {
                    steps.push(Step::Load {
                        key: *key,
                        to: to.clone(),
                    });
                    steps.push(Step::Free {
                        key: *key,
                        from: from.clone(),
                    });
                    edges.push((at, at + 1));
                }
            }
            chains.push((at, steps.len() - 1));
        }
        for b in &diff.buffers {
            chains.push((steps.len(), steps.len()));
            steps.push(Step::SetBuffer {
                key: b.key.clone(),
                bytes: b.to,
            });
        }
        for b in &diff.budgets {
            chains.push((steps.len(), steps.len()));
            steps.push(Step::SetBudgets {
                key: b.key.clone(),
                budgets: b.to,
            });
        }
        if let Some((_, to)) = diff.knobs {
            chains.push((steps.len(), steps.len()));
            steps.push(Step::SetKnobs(to));
        }
        let prepare = steps.len();
        steps.push(Step::Prepare);
        steps.push(Step::Commit);
        for (first, last) in chains {
            edges.push((0, first));
            edges.push((last, prepare));
        }
        if prepare == 1 {
            edges.push((0, prepare));
        }
        edges.push((prepare, prepare + 1));
        Self { steps, edges }
    }

    /// One topological order; `pick` chooses among the ready steps (index into the ready list,
    /// taken modulo its length), so a test can drive random orders. `None` if there is a cycle.
    pub fn order_with(&self, mut pick: impl FnMut(usize) -> usize) -> Option<Vec<usize>> {
        let n = self.steps.len();
        let mut indegree = vec![0usize; n];
        let mut next: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(a, b) in &self.edges {
            indegree[b] += 1;
            next[a].push(b);
        }
        // Ascending index order, so the same picks give the same order.
        let mut ready: Vec<usize> = (0..n).filter(|&i| indegree[i] == 0).collect();
        let mut order = Vec::with_capacity(n);
        while !ready.is_empty() {
            let i = ready.remove(pick(ready.len()) % ready.len());
            order.push(i);
            for &j in &next[i] {
                indegree[j] -= 1;
                if indegree[j] == 0 {
                    let at = ready.partition_point(|&x| x < j);
                    ready.insert(at, j);
                }
            }
        }
        (order.len() == n).then_some(order)
    }
}
