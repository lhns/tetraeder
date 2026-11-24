//! Membership and the node daemon on simulated nodes (ADR 0091 step 3, rules 4 and 5): a head's
//! monitor and daemons over `MemTransport`, with its fault scripts (close, blackhole, slow,
//! partition) and node kills. Timings are real but short (`node_down_after` 1 s); only the pure property test of
//! `membership.rs` runs in simulated time.
//!
//! Hardware label: none.

mod common;

use common::{EchoLauncher, config, id, node_graph_json, wait_until};
use proptest::prelude::*;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;
use tetraeder_control::ClusterConfig;
use tetraeder_control::membership::NodeState;
use tetraeder_control::monitor::Monitor;
use tetraeder_control::node::{Daemon, NodeOptions, RestartPolicy};
use tetraeder_control::{client, proto::NodeRequest};
use tetraeder_transport::{
    LinkFault, MemTransport, NodeId, Purpose, Switchboard, Transport, frame,
};

const DOWN_MS: u64 = 1000;
const HB: Duration = Duration::from_millis(DOWN_MS / 4);
const DOWN: Duration = Duration::from_millis(DOWN_MS);

fn cfg() -> ClusterConfig {
    config(DOWN_MS, &["a", "b", "c"], &[1, 2, 3])
}

fn daemon(
    board: &Arc<Switchboard>,
    cfg: &ClusterConfig,
    name: &str,
    l: Arc<EchoLauncher>,
) -> Daemon {
    let t: Arc<dyn Transport> = Arc::new(board.node(&id(name)));
    let dial = Arc::clone(&t);
    Daemon::start(
        t,
        NodeOptions {
            cfg: cfg.clone(),
            launcher: l,
            profile: Arc::new(|| Ok(node_graph_json())),
            dial: Arc::new(move |_| Ok(Arc::clone(&dial))),
            restart: RestartPolicy {
                backoff_base: Duration::from_millis(20),
                max_restarts: 3,
                window: Duration::from_secs(60),
            },
        },
    )
}

struct Sim {
    board: Arc<Switchboard>,
    cfg: ClusterConfig,
    daemons: Vec<(String, Daemon)>,
    monitor: Monitor,
    log: std::sync::mpsc::Receiver<(NodeId, NodeState)>,
}

fn sim() -> Sim {
    let board = MemTransport::switchboard();
    let cfg = cfg();
    let daemons = ["a", "b", "c"]
        .iter()
        .map(|n| {
            (
                (*n).to_owned(),
                daemon(&board, &cfg, n, Arc::new(EchoLauncher::default())),
            )
        })
        .collect();
    let head: Arc<dyn Transport> = Arc::new(board.node(&id("a")));
    let monitor = Monitor::start(head, cfg.clone());
    let log = monitor.subscribe();
    Sim {
        board,
        cfg,
        daemons,
        monitor,
        log,
    }
}

impl Sim {
    fn state(&self, n: &str) -> NodeState {
        self.monitor.state(&id(n)).unwrap()
    }

    fn wait_state(&self, n: &str, want: NodeState, within: Duration) -> Option<Duration> {
        wait_until(within, || self.state(n) == want)
    }

    fn all_up(&self) {
        for n in ["a", "b", "c"] {
            assert!(
                self.wait_state(n, NodeState::Up, Duration::from_secs(3))
                    .is_some(),
                "{n} did not come up"
            );
        }
    }

    fn kill(&mut self, n: &str) {
        let i = self.daemons.iter().position(|(x, _)| x == n).unwrap();
        let (_, mut d) = self.daemons.remove(i);
        d.stop();
    }

    fn restart(&mut self, n: &str) {
        let d = daemon(&self.board, &self.cfg, n, Arc::new(EchoLauncher::default()));
        self.daemons.push((n.to_owned(), d));
    }

    /// Per node, the transitions seen so far must never go from Down straight to Up, and an
    /// Up always follows a Joining or Draining-free path (no Up without a hello).
    fn check_log(&self) {
        let mut last = std::collections::HashMap::new();
        while let Ok((n, st)) = self.log.try_recv() {
            let prev = last.insert(n.clone(), st).unwrap_or(NodeState::Joining);
            assert!(
                !(st == NodeState::Up && prev == NodeState::Down),
                "{n}: Down to Up without a hello"
            );
        }
    }
}

// hardware: none
#[test]
fn nodes_come_up_after_hello_and_heartbeat() {
    let s = sim();
    s.all_up();
    s.check_log();
}

// hardware: none
#[test]
fn a_killed_and_restarted_node_rejoins_within_down_after_plus_one_heartbeat() {
    let mut s = sim();
    s.all_up();
    s.kill("b");
    // A killed process resets its connection: Down at once, well within node_down_after.
    let t = s
        .wait_state("b", NodeState::Down, DOWN)
        .expect("b not down in time");
    assert!(t < DOWN, "{t:?}");
    s.restart("b");
    let t0 = std::time::Instant::now();
    s.wait_state("b", NodeState::Up, DOWN + HB + Duration::from_millis(300))
        .expect("b did not rejoin");
    // The bound of ADR 0091 (a little scheduling slack on a loaded CI machine).
    assert!(
        t0.elapsed() <= DOWN + HB + Duration::from_millis(300),
        "{:?}",
        t0.elapsed()
    );
    assert_eq!(s.state("c"), NodeState::Up);
    s.check_log();
}

// hardware: none
#[test]
fn a_node_that_stays_dead_stays_down_and_the_rest_stay_up() {
    let mut s = sim();
    s.all_up();
    s.kill("c");
    s.wait_state("c", NodeState::Down, DOWN).unwrap();
    std::thread::sleep(DOWN * 2);
    assert_eq!(s.state("c"), NodeState::Down);
    assert_eq!(s.state("b"), NodeState::Up);
    assert_eq!(s.state("a"), NodeState::Up);
}

/// A fault on the link between the head and `b` (both directions for a partition).
fn inject(s: &Sim, f: LinkFault, both: bool) {
    s.board.fault(&id("a"), &id("b"), f);
    if both {
        s.board.fault(&id("b"), &id("a"), f);
    }
}

// hardware: none
#[test]
fn every_fault_takes_the_node_down_within_node_down_after_and_it_returns_when_lifted() {
    let s = sim();
    s.all_up();
    let faults = [
        ("close", LinkFault::Close, false),
        ("blackhole a to b", LinkFault::Blackhole, false),
        ("partition", LinkFault::Blackhole, true),
        (
            "slow",
            LinkFault::Slow {
                alpha: DOWN * 3,
                bytes_per_s: 1e9,
            },
            false,
        ),
    ];
    for (name, f, both) in faults {
        inject(&s, f, both);
        let t = s
            .wait_state("b", NodeState::Down, DOWN + Duration::from_millis(300))
            .unwrap_or_else(|| panic!("{name}: b not down"));
        assert!(t <= DOWN + Duration::from_millis(300), "{name}: {t:?}");
        // The others are not affected.
        assert_eq!(s.state("c"), NodeState::Up, "{name}");
        inject(&s, LinkFault::None, true);
        s.wait_state("b", NodeState::Up, DOWN * 3 + HB * 2)
            .unwrap_or_else(|| panic!("{name}: b did not return"));
    }
    s.check_log();
}

// hardware: none
#[test]
fn a_silent_node_whose_replies_are_dropped_is_found_by_silence_alone() {
    let s = sim();
    s.all_up();
    // b to a only: the daemon still receives and answers, nothing reaches the head.
    s.board.fault(&id("b"), &id("a"), LinkFault::Blackhole);
    s.wait_state("b", NodeState::Down, DOWN + Duration::from_millis(300))
        .expect("silence not detected");
    s.board.fault(&id("b"), &id("a"), LinkFault::None);
    s.wait_state("b", NodeState::Up, DOWN * 3 + HB * 2).unwrap();
}

// hardware: none
#[test]
fn drain_is_reported_and_new_assignments_are_refused() {
    let s = sim();
    s.all_up();
    let t = s.board.node(&id("a"));
    let peer = s.cfg.peer(&id("b")).unwrap();
    client::request(&t, &peer, &NodeRequest::Drain, &[]).unwrap();
    s.wait_state("b", NodeState::Draining, Duration::from_secs(2))
        .expect("drain not seen");
    let e = client::request(
        &t,
        &peer,
        &NodeRequest::Assign {
            replica: 0,
            segment: 0,
            plan: String::new(),
            model_sha256: String::new(),
        },
        &[],
    )
    .unwrap_err();
    assert!(e.contains("draining"), "{e}");
}

// hardware: none
#[test]
fn only_the_head_may_open_control_and_a_worker_node_may_only_probe() {
    let s = sim();
    s.all_up();
    // c is not the head: its Control connection to b is closed at once.
    let c = s.board.node(&id("c"));
    let mut conn = c
        .connect(&s.cfg.peer(&id("b")).unwrap(), Purpose::Control)
        .unwrap();
    assert!(matches!(frame::read_frame(&mut conn.rx), Ok(None) | Err(_)));
    // Its privileged requests on a probe connection are refused; probe frames are answered.
    let mut p = c
        .connect(
            &s.cfg.peer(&id("b")).unwrap(),
            Purpose::Probe { session: 0 },
        )
        .unwrap();
    let (r, _) = tetraeder_control::proto::call(&mut p, &NodeRequest::Profile, &[]).unwrap();
    assert!(
        matches!(r, tetraeder_control::proto::NodeReply::Error { .. }),
        "{r:?}"
    );
    let (r, _) =
        tetraeder_control::proto::call(&mut p, &NodeRequest::Probe { bytes: 3 }, &[1, 2, 3])
            .unwrap();
    assert!(
        matches!(r, tetraeder_control::proto::NodeReply::Probe),
        "{r:?}"
    );
}

// hardware: none
#[test]
fn assigned_workers_are_relayed_restarted_and_given_up_at_the_cap() {
    let board = MemTransport::switchboard();
    let cfg = config(DOWN_MS, &["a", "b"], &[1, 2]);
    let l = Arc::new(EchoLauncher::default());
    let d = daemon(&board, &cfg, "b", Arc::clone(&l));
    let head = board.node(&id("a"));
    let peer = cfg.peer(&id("b")).unwrap();
    let assign = NodeRequest::Assign {
        replica: 0,
        segment: 1,
        plan: "plan".into(),
        model_sha256: "00".into(),
    };
    client::request(&head, &peer, &assign, &[]).unwrap();
    assert_eq!(d.workers(), 1);
    let seg = Purpose::Segment {
        replica: 0,
        segment: 1,
    };
    let call = |head: &MemTransport, n: u32| {
        let mut c = head.connect(&peer, seg.clone()).unwrap();
        // The worker may still be starting: the first frame can be lost, so retry briefly.
        for _ in 0..50 {
            if frame::write_frame(&mut c.tx, &serde_json::json!({ "n": n }), b"hidden rows")
                .is_err()
            {
                return None;
            }
            if let Ok(Some(f)) = frame::read_frame(&mut c.rx) {
                return Some((f.payload, c));
            }
            std::thread::sleep(Duration::from_millis(20));
            c = head.connect(&peer, seg.clone()).ok()?;
        }
        None
    };
    let (payload, conn) = call(&head, 1).expect("first relay");
    assert_eq!(payload, b"hidden rows");
    // A crash ends the connection at once; the daemon restarts the worker.
    l.crash();
    wait_until(Duration::from_secs(3), || {
        d.worker_restarts(0, 1) == Some(1)
    })
    .expect("no restart");
    drop(conn);
    let (payload, _conn) = call(&head, 2).expect("relay after restart");
    assert_eq!(payload, b"hidden rows");
    assert_eq!(l.launches.load(Ordering::SeqCst), 2);
    // Repeated crashes reach the cap (3 restarts), then the assignment is given up.
    for _ in 0..4 {
        l.crash();
        std::thread::sleep(Duration::from_millis(150));
    }
    wait_until(Duration::from_secs(5), || d.workers() == 0).expect("not given up");
}

// hardware: none
#[test]
fn a_segment_connection_without_an_assignment_is_closed() {
    let board = MemTransport::switchboard();
    let cfg = config(DOWN_MS, &["a", "b"], &[1, 2]);
    let _d = daemon(&board, &cfg, "b", Arc::new(EchoLauncher::default()));
    let head = board.node(&id("a"));
    let mut c = head
        .connect(
            &cfg.peer(&id("b")).unwrap(),
            Purpose::Segment {
                replica: 9,
                segment: 9,
            },
        )
        .unwrap();
    assert!(matches!(frame::read_frame(&mut c.rx), Ok(None) | Err(_)));
}

#[derive(Debug, Clone, Copy)]
enum Step {
    Close,
    Blackhole,
    Partition,
    Slow,
    Heal,
    KillRestart,
}

fn step() -> impl Strategy<Value = Step> {
    prop_oneof![
        Just(Step::Close),
        Just(Step::Blackhole),
        Just(Step::Partition),
        Just(Step::Slow),
        Just(Step::Heal),
        Just(Step::KillRestart),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 3, failure_persistence: None, ..ProptestConfig::default() })]

    // Random sequences of faults on b and recoveries, in short real time: a faulty node is Down
    // within node_down_after, a healed one is Up again, the healthy nodes never flap, and no
    // node is ever Up straight after being Down.
    #[test]
    fn random_faults_keep_membership_consistent(steps in prop::collection::vec(step(), 1..4)) {
        let mut s = sim();
        s.all_up();
        let slack = Duration::from_millis(300);
        for st in steps {
            let faulty = match st {
                Step::Close => { inject(&s, LinkFault::Close, false); true }
                Step::Blackhole => { inject(&s, LinkFault::Blackhole, false); true }
                Step::Partition => { inject(&s, LinkFault::Blackhole, true); true }
                Step::Slow => {
                    inject(&s, LinkFault::Slow { alpha: DOWN * 3, bytes_per_s: 1e9 }, false);
                    true
                }
                Step::Heal => { inject(&s, LinkFault::None, true); false }
                Step::KillRestart => {
                    inject(&s, LinkFault::None, true);
                    s.kill("b");
                    s.restart("b");
                    false
                }
            };
            if faulty {
                prop_assert!(
                    s.wait_state("b", NodeState::Down, DOWN + slack).is_some(),
                    "{:?}: b not down within node_down_after", st
                );
            } else {
                prop_assert!(
                    s.wait_state("b", NodeState::Up, DOWN * 3 + HB * 2).is_some(),
                    "{:?}: b not back", st
                );
            }
            prop_assert_eq!(s.state("c"), NodeState::Up);
            prop_assert_eq!(s.state("a"), NodeState::Up);
        }
        s.check_log();
    }
}
