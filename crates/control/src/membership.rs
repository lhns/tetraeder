//! Membership of the cluster (ADR 0091): a pure state machine, the clock injected as in
//! `replan.rs` (here a `Duration` since any fixed origin, so a simulation can run in simulated
//! time).
//!
//! A node is `Up` after a successful hello and a heartbeat, `Down` after a failed call (reset,
//! EOF, malformed frame, refused handshake) or after `node_down_after` without any frame from
//! it, and `Draining` once it said so. A `Down` node that says hello again is `Joining` until its
//! first heartbeat. Nothing is `Up` without a hello on the connection it speaks on.

use crate::cluster::ClusterConfig;
use std::time::Duration;
use tetraeder_transport::NodeId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeState {
    /// Configured, not yet (or no longer) heard from after a hello.
    Joining,
    Up,
    Draining,
    Down,
}

impl NodeState {
    /// The `to` attribute of `tetraeder.cluster.node.transitions`.
    pub fn label(self) -> &'static str {
        match self {
            Self::Joining => "joining",
            Self::Up => "up",
            Self::Draining => "draining",
            Self::Down => "down",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeEvent {
    /// The hello reply of a new connection arrived.
    Hello(NodeId),
    /// Any frame from the node after its hello, heartbeat replies included.
    Heard(NodeId),
    /// A call failed: reset, EOF, malformed frame, refused handshake.
    Failed(NodeId),
    /// The node reported that it drains.
    Drain(NodeId),
}

struct Entry {
    id: NodeId,
    state: NodeState,
    /// A hello was answered on the current connection.
    hello: bool,
    last: Option<Duration>,
}

/// Pure state machine; the caller supplies time.
pub struct Membership {
    down_after: Duration,
    nodes: Vec<Entry>,
}

pub type Transitions = Vec<(NodeId, NodeState)>;

impl Membership {
    /// Every configured node starts `Joining`.
    pub fn new(cfg: &ClusterConfig) -> Self {
        Self::with(&cfg.ids(), cfg.node_down_after())
    }

    pub fn with(ids: &[NodeId], down_after: Duration) -> Self {
        Self {
            down_after,
            nodes: ids
                .iter()
                .map(|id| Entry {
                    id: id.clone(),
                    state: NodeState::Joining,
                    hello: false,
                    last: None,
                })
                .collect(),
        }
    }

    fn entry(&mut self, id: &NodeId) -> Option<&mut Entry> {
        self.nodes.iter_mut().find(|e| &e.id == id)
    }

    fn set(e: &mut Entry, to: NodeState, out: &mut Transitions) {
        if e.state != to {
            e.state = to;
            out.push((e.id.clone(), to));
        }
    }

    pub fn on_event(&mut self, now: Duration, ev: NodeEvent) -> Transitions {
        let mut out = Vec::new();
        match ev {
            NodeEvent::Hello(id) => {
                if let Some(e) = self.entry(&id) {
                    e.hello = true;
                    e.last = Some(now);
                    if e.state == NodeState::Down {
                        Self::set(e, NodeState::Joining, &mut out);
                    }
                }
            }
            NodeEvent::Heard(id) => {
                if let Some(e) = self.entry(&id)
                    && e.hello
                {
                    e.last = Some(now);
                    if e.state == NodeState::Joining {
                        Self::set(e, NodeState::Up, &mut out);
                    }
                }
            }
            NodeEvent::Failed(id) => {
                if let Some(e) = self.entry(&id) {
                    e.hello = false;
                    Self::set(e, NodeState::Down, &mut out);
                }
            }
            NodeEvent::Drain(id) => {
                if let Some(e) = self.entry(&id)
                    && e.hello
                    && matches!(e.state, NodeState::Up | NodeState::Joining)
                {
                    Self::set(e, NodeState::Draining, &mut out);
                }
            }
        }
        out
    }

    /// Nodes silent for `node_down_after` become `Down`.
    pub fn tick(&mut self, now: Duration) -> Transitions {
        let mut out = Vec::new();
        let after = self.down_after;
        for e in &mut self.nodes {
            if e.state != NodeState::Down
                && let Some(last) = e.last
                && now.saturating_sub(last) >= after
            {
                e.hello = false;
                Self::set(e, NodeState::Down, &mut out);
            }
        }
        out
    }

    pub fn state(&self, id: &NodeId) -> Option<NodeState> {
        self.nodes.iter().find(|e| &e.id == id).map(|e| e.state)
    }

    /// When the last frame from `id` came, on the caller's clock.
    pub fn last_heard(&self, id: &NodeId) -> Option<Duration> {
        self.nodes.iter().find(|e| &e.id == id).and_then(|e| e.last)
    }

    pub fn states(&self) -> Vec<(NodeId, NodeState)> {
        self.nodes.iter().map(|e| (e.id.clone(), e.state)).collect()
    }

    pub fn down_after(&self) -> Duration {
        self.down_after
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn id(s: &str) -> NodeId {
        NodeId::new(s)
    }

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    fn m() -> Membership {
        Membership::with(&[id("a"), id("b")], ms(2000))
    }

    #[test]
    fn up_needs_hello_then_a_frame() {
        let mut m = m();
        assert_eq!(m.state(&id("a")), Some(NodeState::Joining));
        // A frame without a hello is ignored.
        assert!(m.on_event(ms(0), NodeEvent::Heard(id("a"))).is_empty());
        assert!(m.on_event(ms(1), NodeEvent::Hello(id("a"))).is_empty());
        assert_eq!(m.state(&id("a")), Some(NodeState::Joining));
        assert_eq!(
            m.on_event(ms(2), NodeEvent::Heard(id("a"))),
            vec![(id("a"), NodeState::Up)]
        );
        assert_eq!(m.state(&id("b")), Some(NodeState::Joining));
    }

    #[test]
    fn silence_and_failure_take_a_node_down_and_it_rejoins_by_hello() {
        let mut m = m();
        m.on_event(ms(0), NodeEvent::Hello(id("a")));
        m.on_event(ms(0), NodeEvent::Heard(id("a")));
        assert!(m.tick(ms(1999)).is_empty());
        assert_eq!(m.tick(ms(2000)), vec![(id("a"), NodeState::Down)]);
        // Frames of the dead connection do not bring it back.
        assert!(m.on_event(ms(2001), NodeEvent::Heard(id("a"))).is_empty());
        assert_eq!(m.state(&id("a")), Some(NodeState::Down));
        assert_eq!(
            m.on_event(ms(2100), NodeEvent::Hello(id("a"))),
            vec![(id("a"), NodeState::Joining)]
        );
        assert_eq!(
            m.on_event(ms(2200), NodeEvent::Heard(id("a"))),
            vec![(id("a"), NodeState::Up)]
        );
        assert_eq!(
            m.on_event(ms(2300), NodeEvent::Failed(id("a"))),
            vec![(id("a"), NodeState::Down)]
        );
        assert!(m.on_event(ms(2301), NodeEvent::Failed(id("a"))).is_empty());
    }

    #[test]
    fn draining_stays_until_down() {
        let mut m = m();
        m.on_event(ms(0), NodeEvent::Hello(id("a")));
        m.on_event(ms(0), NodeEvent::Heard(id("a")));
        assert_eq!(
            m.on_event(ms(10), NodeEvent::Drain(id("a"))),
            vec![(id("a"), NodeState::Draining)]
        );
        assert!(m.on_event(ms(20), NodeEvent::Heard(id("a"))).is_empty());
        assert_eq!(m.state(&id("a")), Some(NodeState::Draining));
        assert_eq!(m.tick(ms(2020)), vec![(id("a"), NodeState::Down)]);
        // Unknown nodes are ignored.
        assert!(m.on_event(ms(0), NodeEvent::Failed(id("zz"))).is_empty());
    }

    #[derive(Debug, Clone)]
    enum Op {
        Hello(usize),
        Heard(usize),
        Failed(usize),
        Drain(usize),
        Tick,
    }

    fn op() -> impl Strategy<Value = (u64, Op)> {
        (
            0u64..900,
            prop_oneof![
                (0usize..3).prop_map(Op::Hello),
                (0usize..3).prop_map(Op::Heard),
                (0usize..3).prop_map(Op::Failed),
                (0usize..3).prop_map(Op::Drain),
                Just(Op::Tick),
            ],
        )
    }

    proptest! {
        // Rule 5 style: over random sequences of events in simulated time, with a model of what
        // each node has said since its last failure.
        #[test]
        fn random_sequences_keep_the_invariants(ops in prop::collection::vec(op(), 1..200)) {
            let ids = [id("a"), id("b"), id("c")];
            let mut m = Membership::with(&ids, ms(2000));
            let mut now = Duration::ZERO;
            // Model: hello answered on the current connection, time of the last frame.
            let mut hello = [false; 3];
            let mut last: [Option<Duration>; 3] = [None; 3];
            for (dt, op) in ops {
                now += ms(dt);
                let before = m.states();
                let out = match op {
                    Op::Hello(i) => { hello[i] = true; last[i] = Some(now); m.on_event(now, NodeEvent::Hello(ids[i].clone())) }
                    Op::Heard(i) => { if hello[i] { last[i] = Some(now); } m.on_event(now, NodeEvent::Heard(ids[i].clone())) }
                    Op::Failed(i) => { hello[i] = false; m.on_event(now, NodeEvent::Failed(ids[i].clone())) }
                    Op::Drain(i) => m.on_event(now, NodeEvent::Drain(ids[i].clone())),
                    Op::Tick => {
                        let out = m.tick(now);
                        for i in 0..3 {
                            if last[i].is_some_and(|l| now - l >= ms(2000)) { hello[i] = false; }
                        }
                        out
                    }
                };
                let after = m.states();
                // Transitions are exactly the state differences, once per node.
                let changed: Vec<_> = before.iter().zip(&after).filter(|(b, a)| b.1 != a.1).map(|(_, a)| a.clone()).collect();
                prop_assert_eq!(&out, &changed);
                for (i, (_, st)) in after.iter().enumerate() {
                    // No node is Up or Draining without a hello on its connection.
                    if matches!(st, NodeState::Up | NodeState::Draining) {
                        prop_assert!(hello[i], "node {} is {:?} without a hello", i, st);
                    }
                    // After a tick, nobody that was silent for node_down_after is alive.
                    if matches!(op, Op::Tick) && last[i].is_some_and(|l| now - l >= ms(2000)) {
                        prop_assert_eq!(*st, NodeState::Down);
                    }
                    // A node never heard after a hello is not Up.
                    if *st == NodeState::Up {
                        prop_assert!(last[i].is_some());
                    }
                }
            }
        }
    }
}
