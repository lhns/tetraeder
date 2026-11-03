//! Discovery, membership, failover, replanning triggers and mTLS.

mod replan;

pub use replan::{
    Clock, Config, EngineControl, FakeClock, Outcome, OutcomeHook, PlanSource, Proposal, Replanner,
    Trigger,
};
