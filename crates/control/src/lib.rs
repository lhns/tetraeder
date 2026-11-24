//! Discovery, membership, failover, replanning triggers and mTLS.
//!
//! ADR 0091 step 3: `cluster` (the static node list), `membership` (the pure state machine),
//! `monitor` (the head's heartbeats), `node` (the daemon), `client` (profile and probe).

mod replan;

pub mod client;
pub mod cluster;
pub mod cluster_replan;
pub mod head;
pub mod kv;
pub mod membership;
pub mod metrics;
pub mod monitor;
pub mod node;
pub mod proto;
pub mod replica;
pub mod router;

pub use cluster::{ClusterConfig, NodeConfig};
pub use membership::{Membership, NodeEvent, NodeState};
pub use replan::{
    Clock, Config, EngineControl, FakeClock, Outcome, OutcomeHook, PlanSource, Proposal, Replanner,
    SystemClock, Trigger,
};
