//! Placement plan IR, validator, diff and migration DAGs (ADR 0006, 0027).
//!
//! - [`ModelDesc`]: what gets placed, as units with sizes.
//! - [`Plan`]: replicas of pipeline stages with assign rules; TOML and JSON; open fields make a
//!   partial plan.
//! - [`Hints`]: limits and device and format rules (ADR 0025), a separate input.
//! - [`validate`] / [`validate_partial`]: checks against hardware, model, phase and hints;
//!   [`validate`] returns the [`Expanded`] form.
//! - [`diff`] and [`MigrationDag`]: what changes between two plans and in which order.

mod diff;
mod expand;
mod format;
mod hints;
mod kv;
mod migrate;
mod model;
mod phase;
mod plan;
mod quality;
#[cfg(any(test, feature = "testing"))]
pub mod testing;
mod validate;

pub use diff::{BudgetChange, BufferChange, PlanDiff, ShardChange, diff};
pub use expand::{BudgetKey, Budgets, BufferKey, BufferKind, Expanded, Placement, ShardKey};
pub use format::Format;
pub use hints::{
    Forbid, FormatFloor, Freeze, Hints, Level, Objective, Pin, Prefer, Stability, WorkloadHints,
};
pub use kv::KvFormat;
pub use migrate::{MigrationDag, State, Step};
pub use model::{ModelDesc, RoleSize, UnitId, UnitKind, UnitSize};
pub use phase::{Capabilities, ComputeOverride, Phase};
pub use plan::{
    Assign, DeviceBytes, Knobs, PhaseBudget, PhaseBytes, Plan, Replica, SlotCache, Span, Stage,
    UnitFormats, UnitSel,
};
pub use quality::{
    EstimateError, GroupEntry, QUALITY_SCHEMA, QualityError, QualityGroup, QualityIssue,
    QualityProfile,
};
pub use validate::{At, Context, Issue, validate, validate_partial};
