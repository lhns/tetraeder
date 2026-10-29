//! Plan validator (ADR 0006, 0027): structure, exact-once coverage, degrees, memory, formats,
//! hard hints and the phase capability table. Collects every issue, not just the first.

use crate::{Expanded, Format, Hints, ModelDesc, Phase, Plan, UnitId};
use std::collections::BTreeMap;
use std::fmt;
use tetraeder_profiler::HardwareGraph;

/// Everything a plan is checked against.
#[derive(Debug, Clone, Copy)]
pub struct Context<'a> {
    pub hw: &'a HardwareGraph,
    pub model: &'a ModelDesc,
    /// Phase of the executing engine.
    pub phase: Phase,
    /// Measured reserve per device id (CUDA context, workspaces, compositor); missing ids
    /// reserve 0.
    pub reserve: &'a BTreeMap<String, u64>,
    pub hints: Option<&'a Hints>,
}

/// Where in the plan an issue sits: replica and stage indices when known.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct At {
    pub replica: u32,
    pub stage: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Issue {
    /// `plan.model` differs from `ModelDesc::name`.
    WrongModel {
        plan: String,
        model: String,
    },
    NoReplicas,
    /// Stage spans do not tile `0..layers` in order, or a span is empty.
    Spans {
        at: At,
        detail: String,
    },
    UnknownDevice {
        at: At,
        device: String,
    },
    /// A compute device that is not a CPU or GPU.
    CannotCompute {
        at: At,
        device: String,
    },
    /// A compute device outside the stage's device group.
    NotInGroup {
        at: At,
        device: String,
    },
    /// An assign rule that selects no unit of its stage.
    EmptySelection {
        at: At,
        rule: usize,
    },
    /// A unit selected by more than one rule, or in more than one stage.
    Duplicate {
        at: At,
        unit: UnitId,
    },
    /// Complete plans only: a unit no rule selects.
    Missing {
        replica: u32,
        unit: UnitId,
    },
    /// Wrong number of homes (or compute devices) for the unit's kind and the stage's degree.
    ShardCount {
        at: At,
        unit: UnitId,
        expected: String,
        got: usize,
    },
    /// Tensor-parallel shards of one unit compute on the same device.
    SharedShardDevice {
        at: At,
        unit: UnitId,
        device: String,
    },
    /// A degree does not divide what it splits (`what` is heads, kv_heads, experts or devices).
    Degree {
        at: At,
        degree: &'static str,
        value: u32,
        what: &'static str,
        n: u32,
    },
    /// With `ep > 1`, a layer's experts are not spread evenly over `ep` compute devices.
    ExpertSpread {
        at: At,
        layer: u32,
        detail: String,
    },
    /// Complete plans only: an open field (`what` names it).
    Open {
        at: At,
        what: &'static str,
    },
    MemoryExceeded {
        device: String,
        used: u64,
        capacity: u64,
        reserve: u64,
    },
    /// Format more precise than the source: conversion only goes downward (ADR 0024).
    Upconvert {
        unit: UnitId,
        source: Format,
        target: Format,
    },
    /// A slot cache in a stage of a dense model or on an unknown device.
    SlotCache {
        at: At,
        detail: String,
    },
    /// A home on a device that cannot compute and no compute device given.
    NoCompute {
        at: At,
        unit: UnitId,
    },
    /// A field the executing phase cannot run (`field` names it).
    Phase {
        phase: Phase,
        field: &'static str,
    },
    /// A hard hint does not hold.
    Hint {
        detail: String,
    },
}

impl fmt::Display for Issue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for Issue {}

/// Validate a complete plan; on success returns its expanded form.
pub fn validate(plan: &Plan, ctx: &Context) -> Result<Expanded, Vec<Issue>> {
    let _ = (plan, ctx);
    todo!()
}

/// Validate a partial plan: only the fields that are set.
pub fn validate_partial(plan: &Plan, ctx: &Context) -> Result<(), Vec<Issue>> {
    let _ = (plan, ctx);
    todo!()
}
