//! Replanning a cluster when a node changes state (ADR 0091 step 6, ADR 0042).
//!
//! [`ClusterControl`] is the [`EngineControl`] of the head's replicas: it applies a plan by
//! deriving the spec of every remote replica and replacing the ones that changed, one replica at
//! a time. A replica is *drained* first (no new request, the in-flight ones finish), so a change
//! that was planned (a node asked to drain, a node that joined) fails no request; the others
//! keep serving throughout. A node that goes `Down` still fails the in-flight requests of the
//! replicas that span it (Case F); the replan then only gives those replicas a plan without it.
//!
//! [`ReplanWatcher`] turns the monitor's transitions into [`Trigger::NodeChange`] and runs the
//! [`Replanner`] on them.

use crate::head::{ClusterEngine, Specs, StartSpec};
use crate::monitor::Monitor;
use crate::replan::{EngineControl, Outcome, Replanner, Trigger};
use crate::{Clock, PlanSource};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tetraeder_plan::Plan;
use tetraeder_transport::NodeId;

/// The leader and spec of every replica of a plan (`head::specs_from_plan` on the cluster's
/// hardware graph). Only the remote replicas the head runs are replaced, and each keeps its
/// leader: its connection is to that node.
pub type Derive = Arc<dyn Fn(&Plan) -> Result<Vec<(u32, NodeId, StartSpec)>, String> + Send + Sync>;

pub struct ClusterControl {
    engine: Arc<ClusterEngine>,
    specs: Specs,
    derive: Derive,
    /// How long a planned drain waits for a replica's in-flight requests before cutting them.
    drain_timeout: Duration,
    /// How long a replaced replica has to join again.
    join_timeout: Duration,
    changed: Vec<u32>,
}

impl ClusterControl {
    pub fn new(
        engine: Arc<ClusterEngine>,
        specs: Specs,
        derive: Derive,
        drain_timeout: Duration,
        join_timeout: Duration,
    ) -> Self {
        Self {
            engine,
            specs,
            derive,
            drain_timeout,
            join_timeout,
            changed: Vec::new(),
        }
    }
}

impl EngineControl for ClusterControl {
    /// The engine as a whole keeps admitting: a drain is per replica, in `restart`.
    fn stop_admitting(&mut self) {}

    fn in_flight(&self) -> usize {
        self.specs
            .replicas()
            .into_iter()
            .map(|r| self.engine.outstanding(r) as usize)
            .sum()
    }

    fn wait_drained(&mut self, _timeout_s: f64) -> bool {
        true
    }

    fn restart(&mut self, plan: &Plan) -> Result<(), String> {
        let derived = (self.derive)(plan)?;
        self.changed.clear();
        for (r, leader, spec) in derived {
            let Some(old) = self.specs.spec(r) else {
                continue;
            };
            if self.specs.leader(r).as_ref() != Some(&leader) {
                return Err(format!(
                    "the plan moves replica {r} to node `{leader}`, which changes its leader"
                ));
            }
            if old == spec {
                continue;
            }
            // Planned: nothing in flight is lost, unless the drain times out (ADR 0042: the
            // restart then cuts what is left).
            self.engine.drain(r, self.drain_timeout);
            self.specs.set(r, spec);
            self.changed.push(r);
        }
        Ok(())
    }

    fn healthy(&mut self) -> bool {
        let t0 = Instant::now();
        loop {
            if self
                .changed
                .iter()
                .all(|&r| self.engine.is_up(r) && !self.engine.is_draining(r))
            {
                return true;
            }
            if t0.elapsed() >= self.join_timeout {
                return false;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    fn resume(&mut self) {
        // A replica that did not come back must not stay draining: it is out of the router
        // already (a leaving replica clears the flag); a replacement that never joined stops
        // holding requests back.
        for &r in &self.changed {
            self.engine.end_replace(r);
        }
        self.changed.clear();
    }

    fn supports_knobs(&self) -> bool {
        false
    }

    fn apply_knobs(&mut self, _plan: &Plan) -> Result<(), String> {
        Err("a cluster plan has no knobs-only change".into())
    }
}

/// Runs a [`Replanner`] on the node transitions of a [`Monitor`].
pub struct ReplanWatcher {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl ReplanWatcher {
    /// `make` builds the replanner on the watcher's thread (a replanner is not `Send`); its
    /// outcome hook (`Replanner::on_outcome`) sees every decision.
    pub fn start<P, E, C>(
        monitor: &Monitor,
        make: impl FnOnce() -> Replanner<P, E, C> + Send + 'static,
    ) -> Self
    where
        P: PlanSource,
        E: EngineControl,
        C: Clock,
    {
        let rx = monitor.subscribe();
        let stop = Arc::new(AtomicBool::new(false));
        let s = Arc::clone(&stop);
        let thread = std::thread::Builder::new()
            .name("cluster-replan".into())
            .spawn(move || {
                let mut replanner = make();
                while !s.load(Ordering::SeqCst) {
                    let Ok((node, state)) = rx.recv_timeout(Duration::from_millis(100)) else {
                        continue;
                    };
                    let outcome = replanner.on_trigger(Trigger::NodeChange {
                        node: node.0.clone(),
                        state: format!("{state:?}").to_ascii_lowercase(),
                    });
                    if let Outcome::RolledBack { error } = &outcome {
                        tracing::warn!(%node, "replan rolled back: {error}");
                    }
                }
            })
            .ok();
        Self { stop, thread }
    }

    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for ReplanWatcher {
    fn drop(&mut self) {
        self.stop();
    }
}
