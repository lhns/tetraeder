//! A replica that spans two nodes, replanned on a node change (ADR 0091 step 6, ADR 0042) on
//! simulated nodes (rule 4). Replica 0 is led by node b with a segment on b and one on c;
//! replica 1 is led by node c. Closed-loop clients stream requests through the router.
//!
//! - Node c is told to drain: `Trigger::NodeChange` reaches the replanner, which derives a plan
//!   for replica 0 on b alone. Replica 0 is drained first (no new request, the in-flight ones
//!   finish) and then started again with its new spec: no request fails, the segment on c is
//!   released and replica 1 keeps serving.
//! - Node c is killed: its in-flight requests fail (Case F), the replan gives replica 0 a plan
//!   without c, and the replica joins again and serves.
//!
//! Hardware label: none.

mod common;

use common::{EchoLauncher, ReplicaLauncher, config, id, wait_until};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tetraeder_control::ClusterConfig;
use tetraeder_control::client;
use tetraeder_control::cluster_replan::{ClusterControl, Derive, ReplanWatcher};
use tetraeder_control::head::{
    ClusterEngine, ClusterHead, ReplicaTarget, SegmentAssign, StartSpec,
};
use tetraeder_control::monitor::Monitor;
use tetraeder_control::node::{Assignment, Daemon, Launcher, NodeOptions, RestartPolicy, WorkerIo};
use tetraeder_control::proto::NodeRequest;
use tetraeder_control::replica::RemoteEngine;
use tetraeder_control::{Config, PlanSource, Proposal, Replanner, SystemClock, Trigger};
use tetraeder_gateway::engine::{Engine, GenEvent, GenerateRequest, SamplingParams};
use tetraeder_gateway::fake::FakeEngine;
use tetraeder_plan::{Knobs, Plan};
use tetraeder_transport::{MemTransport, Switchboard, Transport};
use tokio_util::sync::CancellationToken;

const DOWN_MS: u64 = 1000;
const SHA: &str = "5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed";

/// A replica engine for replica launches, an echo worker for segments.
struct Mixed {
    replica: ReplicaLauncher,
    worker: EchoLauncher,
}

impl Launcher for Mixed {
    fn launch(&self, a: &Assignment) -> std::io::Result<WorkerIo> {
        if a.is_replica() {
            self.replica.launch(a)
        } else {
            self.worker.launch(a)
        }
    }
}

fn mixed() -> Arc<Mixed> {
    Arc::new(Mixed {
        replica: ReplicaLauncher::new(3, || {
            let mut e = FakeEngine::new(|_| "x".repeat(40));
            e.token_delay = Duration::from_millis(5);
            Arc::new(e)
        }),
        worker: EchoLauncher::default(),
    })
}

fn daemon(board: &Arc<Switchboard>, cfg: &ClusterConfig, name: &str, l: Arc<Mixed>) -> Daemon {
    let t: Arc<dyn Transport> = Arc::new(board.node(&id(name)));
    let dial = Arc::clone(&t);
    Daemon::start(
        t,
        NodeOptions {
            cfg: cfg.clone(),
            launcher: l,
            profile: Arc::new(|| Ok(String::new())),
            dial: Arc::new(move |_| Ok(Arc::clone(&dial))),
            restart: RestartPolicy {
                backoff_base: Duration::from_millis(20),
                max_restarts: 10,
                window: Duration::from_secs(60),
            },
        },
    )
}

/// Proposes the one-node plan when node c drains or goes down.
struct LeaveC;

impl PlanSource for LeaveC {
    fn propose(&mut self, _current: &Plan, t: &Trigger) -> Option<Proposal> {
        match t {
            Trigger::NodeChange { node, state }
                if node == "c" && (state == "draining" || state == "down") =>
            {
                Some(Proposal {
                    plan: plan("on-b"),
                    predicted_gain_s_per_s: 1.0,
                    switch_cost_s: 0.0,
                    knobs_only: false,
                })
            }
            _ => None,
        }
    }
}

fn plan(model: &str) -> Plan {
    Plan {
        model: model.into(),
        knobs: Knobs::default(),
        replicas: vec![],
    }
}

struct Rig {
    board: Arc<Switchboard>,
    engine: Arc<ClusterEngine>,
    head: ClusterHead,
    daemon_b: Daemon,
    daemon_c: Daemon,
    mixed_b: Arc<Mixed>,
    _monitor: Arc<Monitor>,
    _watcher: ReplanWatcher,
    t: Arc<dyn Transport>,
    cfg: ClusterConfig,
}

fn rig() -> Rig {
    let board = MemTransport::switchboard();
    let cfg = config(DOWN_MS, &["a", "b", "c"], &[1, 2, 3]);
    let (mixed_b, mixed_c) = (mixed(), mixed());
    let daemon_b = daemon(&board, &cfg, "b", Arc::clone(&mixed_b));
    let daemon_c = daemon(&board, &cfg, "c", Arc::clone(&mixed_c));
    let t: Arc<dyn Transport> = Arc::new(board.node(&id("a")));
    let monitor = Arc::new(Monitor::start(Arc::clone(&t), cfg.clone()));
    let front: Arc<dyn Engine> = Arc::new(FakeEngine::new(|_| "x".repeat(40)));
    let remotes: Vec<Arc<RemoteEngine>> = ["b", "c"]
        .iter()
        .enumerate()
        .map(|(i, n)| {
            RemoteEngine::new(
                Arc::clone(&front),
                Arc::clone(&t),
                cfg.peer(&id(n)).unwrap(),
                i as u32,
            )
        })
        .collect();
    let engine = Arc::new(ClusterEngine::new(
        Arc::clone(&front),
        remotes
            .iter()
            .map(|r| ReplicaTarget {
                engine: Arc::clone(r) as Arc<dyn Engine>,
                remote: Some(Arc::clone(r)),
                cap: 3,
            })
            .collect(),
        8,
        None,
    ));
    let across = StartSpec {
        plan: String::new(),
        model_sha256: SHA.into(),
        segments: ["b", "c"]
            .iter()
            .map(|n| SegmentAssign {
                peer: cfg.peer(&id(n)).unwrap(),
                plan: format!("segment on {n}"),
            })
            .collect(),
        predicted_s: vec![0.01, 0.01],
    };
    let alone = StartSpec {
        plan: String::new(),
        model_sha256: SHA.into(),
        segments: Vec::new(),
        predicted_s: Vec::new(),
    };
    let head = ClusterHead::start(
        &engine,
        &monitor,
        &t,
        vec![(0, id("b"), across), (1, id("c"), alone.clone())],
    );
    let derive: Derive = Arc::new(move |p: &Plan| {
        if p.model == "on-b" {
            Ok(vec![(0, id("b"), alone.clone())])
        } else {
            Err(format!("no spec for plan {}", p.model))
        }
    });
    let control = ClusterControl::new(
        Arc::clone(&engine),
        head.specs(),
        derive,
        Duration::from_secs(10),
        Duration::from_secs(20),
    );
    let watcher = ReplanWatcher::start(&monitor, move || {
        let cfg = Config {
            min_dwell_s: 0.0,
            drain_timeout_s: 10.0,
            ..Config::default()
        };
        Replanner::new(cfg, plan("across"), LeaveC, control, SystemClock::new())
    });
    Rig {
        board,
        engine,
        head,
        daemon_b,
        daemon_c,
        mixed_b,
        _monitor: monitor,
        _watcher: watcher,
        t,
        cfg,
    }
}

/// `(sent, ok, errors with the time they ended)`.
#[derive(Default)]
struct Tally {
    ok: usize,
    errors: Vec<(Duration, String)>,
}

fn clients(
    engine: &Arc<ClusterEngine>,
    n: usize,
    t0: Instant,
    stop: &Arc<AtomicBool>,
    tally: &Arc<Mutex<Tally>>,
) -> Vec<std::thread::JoinHandle<()>> {
    (0..n)
        .map(|c| {
            let (e, stop, tally) = (Arc::clone(engine), Arc::clone(stop), Arc::clone(tally));
            std::thread::spawn(move || {
                let mut turn = 0;
                while !stop.load(Ordering::SeqCst) {
                    turn += 1;
                    let prompt: Vec<u32> = format!("client {c} turn {turn} {}", "y".repeat(30))
                        .bytes()
                        .map(u32::from)
                        .collect();
                    let rx = e.generate(GenerateRequest {
                        model: "fake".into(),
                        prompt,
                        params: SamplingParams {
                            max_tokens: 100,
                            ..SamplingParams::default()
                        },
                        constraint: None,
                        cancel: CancellationToken::new(),
                    });
                    let mut rx = match rx {
                        Ok(rx) => rx,
                        Err(e) => {
                            tally
                                .lock()
                                .unwrap()
                                .errors
                                .push((t0.elapsed(), format!("generate: {e}")));
                            std::thread::sleep(Duration::from_millis(50));
                            continue;
                        }
                    };
                    let deadline = Instant::now() + Duration::from_secs(30);
                    loop {
                        match rx.try_recv() {
                            Ok(GenEvent::Finished { .. }) => {
                                tally.lock().unwrap().ok += 1;
                                break;
                            }
                            Ok(GenEvent::Error(m) | GenEvent::ContextExceeded(m)) => {
                                tally.lock().unwrap().errors.push((t0.elapsed(), m));
                                break;
                            }
                            Ok(_) => {}
                            Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                                assert!(Instant::now() < deadline, "a request never ended");
                                std::thread::sleep(Duration::from_millis(1));
                            }
                            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                                panic!("a request ended without a final event")
                            }
                        }
                    }
                }
            })
        })
        .collect()
}

// hardware: none
#[test]
fn a_planned_drain_of_a_segmented_replica_fails_no_request() {
    let rig = rig();
    assert!(
        wait_until(Duration::from_secs(20), || rig.engine.up_count() == 2).is_some(),
        "both replicas join"
    );
    assert_eq!(rig.head.specs().spec(0).unwrap().segments.len(), 2);
    let t0 = Instant::now();
    let (stop, tally) = (Arc::new(AtomicBool::new(false)), Arc::default());
    let hs = clients(&rig.engine, 6, t0, &stop, &tally);
    std::thread::sleep(Duration::from_millis(500));
    let before = rig.mixed_b.replica.launches.load(Ordering::SeqCst);

    // Node c is asked to drain (the operator's `tetraeder node --drain --id c`).
    let peer = rig.cfg.peer(&id("c")).unwrap();
    client::request(rig.t.as_ref(), &peer, &NodeRequest::Drain, &[]).unwrap();
    let took = wait_until(Duration::from_secs(30), || {
        rig.head
            .specs()
            .spec(0)
            .is_some_and(|s| s.segments.is_empty())
            && rig.engine.is_up(0)
            && !rig.engine.is_draining(0)
    })
    .expect("replica 0 is replanned onto b and joins again");
    std::thread::sleep(Duration::from_millis(500));
    stop.store(true, Ordering::SeqCst);
    for h in hs {
        h.join().unwrap();
    }
    let tally = tally.lock().unwrap();
    assert!(
        tally.errors.is_empty(),
        "a planned drain fails nothing: {:?}",
        tally.errors
    );
    assert_eq!(rig.engine.counts().failed_in_flight, 0);
    assert!(tally.ok > 10, "requests flowed: {}", tally.ok);
    assert!(
        rig.mixed_b.replica.launches.load(Ordering::SeqCst) > before,
        "the replica engine restarted with its new spec"
    );
    assert!(
        wait_until(Duration::from_secs(5), || rig.daemon_b.workers() == 1).is_some(),
        "the segment worker of the old plan on b is released (the replica engine is left)"
    );
    assert!(rig.engine.is_up(1), "replica 1 kept serving");
    println!(
        "planned drain segmented replica replanned_s {:.1} failed_requests 0 ok_requests {}",
        took.as_secs_f64(),
        tally.ok
    );
    let _ = (&rig.board, &rig.daemon_c);
}

// hardware: none
#[test]
fn a_node_that_goes_down_is_replanned_around() {
    let mut rig = rig();
    assert!(wait_until(Duration::from_secs(20), || rig.engine.up_count() == 2).is_some());
    let t0 = Instant::now();
    let (stop, tally) = (Arc::new(AtomicBool::new(false)), Arc::default());
    let hs = clients(&rig.engine, 6, t0, &stop, &tally);
    std::thread::sleep(Duration::from_millis(500));
    let killed = t0.elapsed();
    rig.daemon_c.stop();
    let took = wait_until(Duration::from_secs(40), || {
        rig.head
            .specs()
            .spec(0)
            .is_some_and(|s| s.segments.is_empty())
            && rig.engine.is_up(0)
    })
    .expect("replica 0 is replanned onto b and joins again");
    let ok_at_rejoin = tally.lock().unwrap().ok;
    std::thread::sleep(Duration::from_secs(1));
    stop.store(true, Ordering::SeqCst);
    for h in hs {
        h.join().unwrap();
    }
    let tally = tally.lock().unwrap();
    // Case F: what was in flight on a replica that spans c fails, within the detection bound;
    // nothing fails after it.
    let bound = killed + Duration::from_millis(DOWN_MS + 1000);
    for (at, m) in &tally.errors {
        assert!(
            *at <= bound + Duration::from_millis(500),
            "late error at {at:?}: {m}"
        );
    }
    assert!(
        tally.ok > ok_at_rejoin,
        "requests succeed after the replan ({} then {})",
        ok_at_rejoin,
        tally.ok
    );
    println!(
        "node down replanned_s {:.1} in_flight_errors {}",
        took.as_secs_f64(),
        tally.errors.len()
    );
    let _ = &rig.board;
}
