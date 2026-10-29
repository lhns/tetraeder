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
mod migrate;
mod model;
mod phase;
mod plan;
mod validate;

pub use diff::{BufferChange, PlanDiff, ShardChange, diff};
pub use expand::{BufferKey, BufferKind, Expanded, Placement, ShardKey};
pub use format::Format;
pub use hints::{
    Forbid, FormatFloor, Freeze, Hints, Level, Objective, Pin, Prefer, Stability, WorkloadHints,
};
pub use migrate::{MigrationDag, State, Step};
pub use model::{ModelDesc, UnitId, UnitKind, UnitSize};
pub use phase::{Capabilities, ComputeOverride, Phase};
pub use plan::{Assign, DeviceBytes, Knobs, Plan, Replica, SlotCache, Span, Stage, UnitSel};
pub use validate::{At, Context, Issue, validate, validate_partial};
