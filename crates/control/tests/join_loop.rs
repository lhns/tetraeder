//! The head's join loop (`cluster-rejoin-after-restore`, `cluster-drain-without-plan`, ADR 0091
//! History).
//!
//! Rejoin: a replica that takes longer to load than the head waits for its `Info` joins anyway
//! (`cluster-rejoin-after-restore`, ADR 0091 History). On the rehearsal's real GPUs an engine
//! takes 11 s to load, longer than the head's 5 s hello wait; the head's join loop then hung in
//! `RemoteEngine::connect` and never tried again. The fake engine of `case_f.rs` answers at
//! once, which is why that simulation passed.
//!
//! Three cases, each with an engine whose load time is `LOAD`, above the hello wait: the cold
//! start of the cluster, the replica engine killed (node and daemon stay up), and the daemon and
//! engine killed and the daemon started again. Each asserts the replica serves a request again
//! within `LOAD + SLACK` of the restore, without anyone asking.
//!
//! Drain: with no cluster plan, a node told to drain takes no new request on its replica, what
//! was in flight and what waited at the router finishes, and the others serve the rest.
//!
//! Hardware label: none.

mod common;

use common::{EchoLauncher, ReplicaLauncher, config, id, wait_until};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tetraeder_control::ClusterConfig;
use tetraeder_control::head::{ClusterEngine, ClusterHead, ReplicaTarget, StartSpec};
use tetraeder_control::monitor::Monitor;
use tetraeder_control::node::{Daemon, NodeOptions, RestartPolicy};
use tetraeder_control::replica::RemoteEngine;
use tetraeder_gateway::engine::{Engine, GenEvent, GenerateRequest, SamplingParams};
use tetraeder_gateway::fake::FakeEngine;
use tetraeder_transport::{MemTransport, Switchboard, Transport};
use tokio_util::sync::CancellationToken;

/// Longer than the head's wait for a replica's first answer (5 s).
const LOAD: Duration = Duration::from_millis(6500);
/// What the head may add on top of the load: one hello wait and a retry.
const SLACK: Duration = Duration::from_secs(10);
const DOWN_MS: u64 = 1000;

fn fake(letter: &'static str) -> Arc<dyn Engine> {
    fake_slow(letter, Duration::from_millis(2))
}

fn fake_slow(letter: &'static str, delay: Duration) -> Arc<dyn Engine> {
    let mut e = FakeEngine::new(move |_| letter.repeat(8));
    e.token_delay = delay;
    Arc::new(e)
}

fn daemon(
    board: &Arc<Switchboard>,
    cfg: &ClusterConfig,
    name: &str,
    l: Arc<dyn tetraeder_control::node::Launcher>,
) -> Daemon {
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

struct Rig {
    board: Arc<Switchboard>,
    cfg: ClusterConfig,
    engine: Arc<ClusterEngine>,
    launcher_b: Arc<ReplicaLauncher>,
    head_t: Arc<dyn Transport>,
    daemon_b: Option<Daemon>,
    _daemon_a: Daemon,
    _monitor: Arc<Monitor>,
    _head: ClusterHead,
}

/// The head on node a with its own replica 0, replica 1 on node b. `load` is the engine load
/// time of node b's replica from the start.
fn rig(load: Duration) -> Rig {
    let board = MemTransport::switchboard();
    let cfg = config(DOWN_MS, &["a", "b"], &[1, 2]);
    let launcher_b = Arc::new(ReplicaLauncher::new(2, || {
        fake_slow("B", Duration::from_millis(60))
    }));
    *launcher_b.load.lock().unwrap() = load;
    let daemon_a = daemon(&board, &cfg, "a", Arc::new(EchoLauncher::default()));
    let daemon_b = daemon(&board, &cfg, "b", Arc::clone(&launcher_b) as _);
    let head_t: Arc<dyn Transport> = Arc::new(board.node(&id("a")));
    let monitor = Arc::new(Monitor::start(Arc::clone(&head_t), cfg.clone()));
    let front = fake("A");
    let remote = RemoteEngine::new(
        Arc::clone(&front),
        Arc::clone(&head_t),
        cfg.peer(&id("b")).unwrap(),
        1,
    );
    let engine = Arc::new(ClusterEngine::new(
        Arc::clone(&front),
        vec![
            ReplicaTarget {
                engine: fake("A"),
                remote: None,
                cap: 2,
            },
            ReplicaTarget {
                engine: remote.clone() as Arc<dyn Engine>,
                remote: Some(remote),
                cap: 2,
            },
        ],
        8,
        None,
    ));
    engine.set_up(0, true);
    let head = ClusterHead::start(
        &engine,
        &monitor,
        &head_t,
        vec![(
            1,
            id("b"),
            StartSpec {
                plan: String::new(),
                model_sha256: "0".repeat(64),
                segments: Vec::new(),
                predicted_s: Vec::new(),
            },
        )],
    );
    Rig {
        board,
        cfg,
        engine,
        launcher_b,
        head_t,
        daemon_b: Some(daemon_b),
        _daemon_a: daemon_a,
        _monitor: monitor,
        _head: head,
    }
}

/// Runs pairs of requests until one is answered by replica 1 ('B'; a lone request goes to the
/// lowest replica with room); the time it took.
fn served_by_b(engine: &Arc<ClusterEngine>, within: Duration) -> Option<Duration> {
    let t0 = Instant::now();
    while t0.elapsed() < within {
        let pair = [request(engine), request(engine)];
        if pair.into_iter().map(finish).any(|d| d.0 == Some(b'B')) {
            return Some(t0.elapsed());
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    None
}

fn report(case: &str, took: Duration) {
    println!(
        "rejoin case={case} load_s={:.1} rejoin_s={:.1} bound_s={:.1}",
        LOAD.as_secs_f64(),
        took.as_secs_f64(),
        (LOAD + SLACK).as_secs_f64()
    );
}

// hardware: none
#[test]
fn a_cold_start_with_a_slow_engine_joins() {
    let rig = rig(LOAD);
    let took = wait_until(LOAD + SLACK, || rig.engine.is_up(1))
        .expect("replica 1 joins although its engine loads longer than the hello wait");
    assert!(served_by_b(&rig.engine, Duration::from_secs(10)).is_some());
    report("cold", took);
}

// hardware: none
#[test]
fn a_killed_engine_on_a_live_node_is_started_again_and_rejoins() {
    let rig = rig(Duration::ZERO);
    assert!(
        wait_until(Duration::from_secs(10), || rig.engine.is_up(1)).is_some(),
        "the replica joins first"
    );
    *rig.launcher_b.load.lock().unwrap() = LOAD;
    let t0 = Instant::now();
    rig.launcher_b.crash();
    assert!(
        wait_until(Duration::from_secs(5), || !rig.engine.is_up(1)).is_some(),
        "the loss is seen"
    );
    wait_until(LOAD + SLACK, || rig.engine.is_up(1))
        .expect("replica 1 rejoins without anyone asking");
    assert!(served_by_b(&rig.engine, Duration::from_secs(10)).is_some());
    report("kill_engine", t0.elapsed());
}

// hardware: none
#[test]
fn a_killed_daemon_and_engine_come_back_and_rejoin() {
    let mut rig = rig(Duration::ZERO);
    assert!(
        wait_until(Duration::from_secs(10), || rig.engine.is_up(1)).is_some(),
        "the replica joins first"
    );
    let mut d = rig.daemon_b.take().unwrap();
    d.stop();
    rig.launcher_b.crash();
    assert!(
        wait_until(Duration::from_secs(5), || !rig.engine.is_up(1)).is_some(),
        "the loss is seen"
    );
    // Held past the detection bound, as the rehearsal held it.
    std::thread::sleep(Duration::from_millis(DOWN_MS + 1500));
    *rig.launcher_b.load.lock().unwrap() = LOAD;
    let t0 = Instant::now();
    rig.daemon_b = Some(daemon(
        &rig.board,
        &rig.cfg,
        "b",
        Arc::clone(&rig.launcher_b) as _,
    ));
    wait_until(LOAD + SLACK, || rig.engine.is_up(1))
        .expect("replica 1 rejoins without anyone asking");
    assert!(served_by_b(&rig.engine, Duration::from_secs(10)).is_some());
    report("kill_daemon", t0.elapsed());
}

fn request(engine: &Arc<ClusterEngine>) -> tokio::sync::mpsc::Receiver<GenEvent> {
    engine
        .generate(GenerateRequest {
            model: "fake".into(),
            prompt: vec![1, 2, 3],
            params: SamplingParams {
                max_tokens: 8,
                ..SamplingParams::default()
            },
            constraint: None,
            cancel: CancellationToken::new(),
        })
        .expect("a replica is up")
}

/// The first letter of the answer and whether it ended without an error.
fn finish(mut rx: tokio::sync::mpsc::Receiver<GenEvent>) -> (Option<u8>, bool) {
    let mut first = None;
    let end = Instant::now() + Duration::from_secs(10);
    while Instant::now() < end {
        match rx.try_recv() {
            Ok(GenEvent::Token(t)) => {
                first.get_or_insert(t as u8);
            }
            Ok(GenEvent::Finished { .. }) => return (first, true),
            Ok(GenEvent::Error(_) | GenEvent::ContextExceeded(_)) => return (first, false),
            Ok(_) => {}
            Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                std::thread::sleep(Duration::from_millis(2));
            }
            Err(_) => break,
        }
    }
    (first, false)
}

// hardware: none
#[test]
fn a_drained_node_takes_no_new_request_and_finishes_what_it_has() {
    let rig = rig(Duration::ZERO);
    assert!(
        wait_until(Duration::from_secs(10), || rig.engine.is_up(1)).is_some(),
        "the replica joins first"
    );
    // Six requests at once: four fit (two per replica), two wait at the router.
    let before: Vec<_> = (0..6).map(|_| request(&rig.engine)).collect();
    assert!(wait_until(Duration::from_secs(2), || rig.engine.outstanding(1) > 0).is_some());
    let t0 = Instant::now();
    let peer = rig.cfg.peer(&id("b")).unwrap();
    tetraeder_control::client::request(
        rig.head_t.as_ref(),
        &peer,
        &tetraeder_control::proto::NodeRequest::Drain,
        &[],
    )
    .expect("the daemon takes the drain");
    wait_until(Duration::from_secs(5), || rig.engine.is_draining(1))
        .expect("the head stops sending to the draining replica");
    let took = t0.elapsed();
    let done: Vec<_> = before.into_iter().map(finish).collect();
    assert!(
        done.iter().all(|d| d.1),
        "nothing in flight or waiting fails: {done:?}"
    );
    // From now on nothing goes to the drained node.
    let after: Vec<_> = (0..10).map(|_| finish(request(&rig.engine))).collect();
    assert!(
        after.iter().all(|d| d.1 && d.0 == Some(b'A')),
        "new requests run on the other replica: {after:?}"
    );
    println!(
        "drain_without_plan drained_after_s={:.2} in_flight_ok={} later_on_other={}",
        took.as_secs_f64(),
        done.len(),
        after.len()
    );
}
