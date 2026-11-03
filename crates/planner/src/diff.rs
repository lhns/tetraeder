//! `plan diff` (docs/plan.md "Explain, diff, what-if"): what moves between two plans and how
//! long the switch takes. Built on `tetraeder_plan::diff` and the migration DAG (ADR 0027); the
//! result is what the control loop needs as `Proposal.switch_cost_s` and `knobs_only`.
//!
//! Switch time, when more than knobs change (drain and restart, ADR 0009):
//! `DRAIN_S + transfer_s + RELOAD_S`.
//! - `transfer_s`: every `Load` step of the DAG is independent of the others (each only waits
//!   for the drain), so all transfers overlap and the time is set by the busiest shared
//!   resource: bytes through it over its bandwidth. A transfer of a changed home with the same
//!   format reads the old home; a new shard or a new format is converted from the model file,
//!   read from the first disk (the CPU RAM page cache when the box has no disk). The path runs
//!   source tier, links (through the CPU when no direct link), destination tier; links in a lane
//!   group also count against the group's capacity.
//! - [`DRAIN_S`] and [`RELOAD_S`] are estimates, not measurements: replace them with the
//!   telemetry of real switches once Phase 3a has any.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use tetraeder_plan::{
    BufferChange, Expanded, Format, Knobs, MigrationDag, Phase, Placement, Plan, ShardChange,
    ShardKey, Step, validate,
};
use tetraeder_profiler::{DeviceKind, HardwareGraph};

/// Time for in-flight requests to finish after admission stops.
pub const DRAIN_S: f64 = 5.0;
/// Engine start after a plan switch: context, allocation, graph capture, warm-up.
pub const RELOAD_S: f64 = 10.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MoveKind {
    /// A shard the old plan does not have.
    Add,
    /// A shard the new plan does not have: bytes are freed, none written.
    Remove,
    /// Same format, new home: copied from the old home.
    Move,
    /// New format: converted from the model file at the new home.
    Reformat,
    /// Only the compute device or stage changed: no bytes move.
    Recompute,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShardMove {
    pub unit: String,
    pub replica: u32,
    pub shard: u32,
    pub of: u32,
    pub kind: MoveKind,
    pub from: Option<String>,
    pub to: Option<String>,
    pub format_from: Option<Format>,
    pub format_to: Option<Format>,
    /// Bytes written at the destination (0 for `remove` and `recompute`).
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanSwitch {
    /// The plans are equal in every field the diff looks at.
    pub empty: bool,
    /// Only knobs differ: no reload.
    pub knobs_only: bool,
    pub moves: Vec<ShardMove>,
    pub moved_bytes: u64,
    pub freed_bytes: u64,
    /// Bytes written per destination device.
    pub by_destination: BTreeMap<String, u64>,
    pub buffers: Vec<BufferChange>,
    pub budgets_changed: usize,
    pub knobs: Option<(Knobs, Knobs)>,
    pub steps: usize,
    pub transfer_s: f64,
    pub drain_s: f64,
    pub reload_s: f64,
    /// Seconds of lost service: 0 for an empty or knobs-only change.
    pub switch_s: f64,
    /// The resource that sets `transfer_s`.
    pub bottleneck: Option<String>,
}

fn expand(
    p: &Plan,
    hw: &HardwareGraph,
    model: &tetraeder_plan::ModelDesc,
) -> Result<Expanded, String> {
    let reserve = BTreeMap::new();
    let ctx = tetraeder_plan::Context {
        hw,
        model,
        phase: Phase(9),
        reserve: &reserve,
        hints: None,
    };
    validate(p, &ctx).map_err(|i| format!("plan does not validate: {:?}", i.first()))
}

/// Bandwidth the tier of `dev` offers, bytes per second.
fn tier_bw(hw: &HardwareGraph, dev: &str) -> Option<f64> {
    hw.device(dev)?.bandwidth_bytes_per_s.map(|m| m.value)
}

/// Bytes through each resource of a transfer path: resource name -> (bytes, bandwidth).
struct Load(BTreeMap<String, (f64, f64)>);

impl Load {
    fn add(&mut self, name: String, bytes: u64, bw: Option<f64>) {
        if let Some(bw) = bw.filter(|b| *b > 0.0) {
            self.0.entry(name).or_insert((0.0, bw)).0 += bytes as f64;
        }
    }

    fn link(&mut self, hw: &HardwareGraph, a: &str, b: &str, bytes: u64) -> bool {
        let Some(l) = hw.link_between(a, b) else {
            return false;
        };
        self.add(
            format!("link:{}", l.id),
            bytes,
            Some(l.beta_bytes_per_s.value),
        );
        if let Some(g) = l.lane_group.as_deref().and_then(|g| hw.lane_group(g)) {
            self.add(
                format!("lanes:{}", g.id),
                bytes,
                Some(g.capacity_bytes_per_s),
            );
        }
        true
    }

    /// One transfer: source tier, links, destination tier.
    fn transfer(&mut self, hw: &HardwareGraph, src: &str, dst: &str, bytes: u64) {
        self.add(format!("dev:{src}"), bytes, tier_bw(hw, src));
        if src != dst && !self.link(hw, src, dst, bytes) {
            // No direct link: through the first CPU.
            if let Some(cpu) = hw.devices.iter().find(|d| d.kind == DeviceKind::Cpu) {
                self.link(hw, src, &cpu.id, bytes);
                self.link(hw, &cpu.id, dst, bytes);
            }
        }
        self.add(format!("dev:{dst}"), bytes, tier_bw(hw, dst));
    }
}

/// Source of a transfer: the old home to copy from, else the model file's disk, else RAM.
fn source<'a>(hw: &'a HardwareGraph, copy_from: Option<&'a str>) -> Option<&'a str> {
    copy_from.or_else(|| {
        [DeviceKind::Disk, DeviceKind::Cpu]
            .iter()
            .find_map(|k| hw.devices.iter().find(|d| d.kind == *k))
            .map(|d| d.id.as_str())
    })
}

/// What changes from `old` to `new` and how long the switch takes (module docs). Both plans must
/// validate against `hw` and `model` at the latest phase, without hints or reserve.
pub fn plan_diff(
    old: &Plan,
    new: &Plan,
    hw: &HardwareGraph,
    model: &tetraeder_plan::ModelDesc,
) -> Result<PlanSwitch, String> {
    let (a, b) = (expand(old, hw, model)?, expand(new, hw, model)?);
    let d = tetraeder_plan::diff(&a, &b);
    let dag = MigrationDag::from_diff(&d);
    let mut moves = Vec::new();
    // Shards the migration writes, with the old home to copy from (else the model file).
    let mut written: BTreeMap<ShardKey, Option<String>> = BTreeMap::new();
    let (mut freed, mut moved) = (0u64, 0u64);
    for c in &d.shards {
        let m = match c {
            ShardChange::Add { key, to } => {
                written.insert(*key, None);
                mv(key, MoveKind::Add, None, Some(to))
            }
            ShardChange::Remove { key, from } => {
                freed += from.bytes;
                mv(key, MoveKind::Remove, Some(from), None)
            }
            ShardChange::Change { key, from, to } => {
                let kind = if from.format != to.format {
                    written.insert(*key, None);
                    MoveKind::Reformat
                } else if from.home != to.home {
                    written.insert(*key, Some(from.home.clone()));
                    MoveKind::Move
                } else {
                    MoveKind::Recompute
                };
                mv(key, kind, Some(from), Some(to))
            }
        };
        moved += m.bytes;
        moves.push(m);
    }
    let mut by_destination: BTreeMap<String, u64> = BTreeMap::new();
    let mut load = Load(BTreeMap::new());
    for s in &dag.steps {
        let Step::Load { key, to } = s else { continue };
        // A change that keeps home and format loads nothing.
        let Some(copy) = written.get(key) else {
            continue;
        };
        *by_destination.entry(to.home.clone()).or_default() += to.bytes;
        if let Some(src) = source(hw, copy.as_deref()) {
            load.transfer(hw, src, &to.home, to.bytes);
        }
    }
    let (transfer_s, bottleneck) = load
        .0
        .iter()
        .map(|(n, (bytes, bw))| (bytes / bw, n))
        .max_by(|x, y| x.0.total_cmp(&y.0))
        .map_or((0.0, None), |(t, n)| (t, Some(n.clone())));
    let empty = d.is_empty();
    let knobs_only =
        d.shards.is_empty() && d.buffers.is_empty() && d.budgets.is_empty() && d.knobs.is_some();
    let (drain_s, reload_s) = if empty || knobs_only {
        (0.0, 0.0)
    } else {
        (DRAIN_S, RELOAD_S)
    };
    let transfer_s = if drain_s == 0.0 { 0.0 } else { transfer_s };
    Ok(PlanSwitch {
        empty,
        knobs_only,
        moves,
        moved_bytes: moved,
        freed_bytes: freed,
        by_destination,
        buffers: d.buffers.clone(),
        budgets_changed: d.budgets.len(),
        knobs: d.knobs,
        steps: dag.steps.len(),
        transfer_s,
        drain_s,
        reload_s,
        switch_s: drain_s + transfer_s + reload_s,
        bottleneck: if transfer_s > 0.0 { bottleneck } else { None },
    })
}

fn mv(
    key: &ShardKey,
    kind: MoveKind,
    from: Option<&Placement>,
    to: Option<&Placement>,
) -> ShardMove {
    let written = matches!(kind, MoveKind::Add | MoveKind::Move | MoveKind::Reformat);
    ShardMove {
        unit: key.unit.to_string(),
        replica: key.replica,
        shard: key.shard,
        of: key.of,
        kind,
        from: from.map(|p| p.home.clone()),
        to: to.map(|p| p.home.clone()),
        format_from: from.map(|p| p.format),
        format_to: to.map(|p| p.format),
        bytes: if written {
            to.map_or(0, |p| p.bytes)
        } else {
            0
        },
    }
}

impl PlanSwitch {
    /// A short human report.
    pub fn to_text(&self) -> String {
        use std::fmt::Write as _;
        let mut o = String::new();
        if self.empty {
            return "no change\n".into();
        }
        if self.knobs_only {
            return "knobs only: no reload\n".into();
        }
        let _ = writeln!(
            o,
            "{} shard changes, {:.2} GB written, {:.2} GB freed",
            self.moves.len(),
            self.moved_bytes as f64 / 1e9,
            self.freed_bytes as f64 / 1e9
        );
        for (d, b) in &self.by_destination {
            let _ = writeln!(o, "  to {d}: {:.2} GB", *b as f64 / 1e9);
        }
        let _ = writeln!(
            o,
            "switch {:.1} s = drain {:.1} + transfer {:.1} ({}) + reload {:.1}",
            self.switch_s,
            self.drain_s,
            self.transfer_s,
            self.bottleneck.as_deref().unwrap_or("none"),
            self.reload_s
        );
        o
    }
}
