//! ADR 0091 Case F on simulated nodes (rules 4 and 5): two replicas of a fake engine, R_A led by
//! the head and R_B led by node b behind its daemon, concurrency 8 of streaming requests, and a
//! fault on node b: its worker killed, its daemon and worker killed, or both directions of its
//! link blackholed (the partition stand-in). The ADR's pass rules are the assertions:
//!
//! - every request in flight on R_B at the fault ends with one clean error within
//!   `node_down_after + 1 s`; no request on R_A fails;
//! - every request submitted from `node_down_after + 1 s` after the fault on succeeds, and
//!   requests waiting at the router move and succeed;
//! - each request ends exactly once (no event after its error; client and router counts agree);
//! - b rejoins and serves requests after it is restored, in the first mode without any action.
//!
//! Timings are real but short (`node_down_after` 1 s). Each mode prints one `case_f` line.
//!
//! Hardware label: none.

mod common;

use common::{EchoLauncher, ReplicaLauncher, config, id, wait_until};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tetraeder_control::ClusterConfig;
use tetraeder_control::head::{ClusterEngine, ClusterHead, ReplicaTarget, StartSpec};
use tetraeder_control::monitor::Monitor;
use tetraeder_control::node::{Daemon, NodeOptions, RestartPolicy};
use tetraeder_control::replica::RemoteEngine;
use tetraeder_gateway::engine::{Engine, GenEvent, GenerateRequest, SamplingParams};
use tetraeder_gateway::fake::FakeEngine;
use tetraeder_transport::{LinkFault, MemTransport, Switchboard, Transport};
use tokio_util::sync::CancellationToken;

const DOWN_MS: u64 = 1000;
/// `node_down_after + 1 s`.
const BOUND: Duration = Duration::from_millis(DOWN_MS + 1000);
const CAP: u32 = 3;
/// More clients than the two replicas have slots (6), so requests wait at the router.
const CLIENTS: usize = 12;
const MIN_WAITING: usize = 3;

/// An engine that can be made to fail every request, as a replica whose engine survives but
/// whose steps fail (a sticky device error, a bad model): the replica stays in the router and the
/// errors arrive as events.
struct Flaky {
    inner: Arc<dyn Engine>,
    failing: Arc<AtomicBool>,
}

impl Engine for Flaky {
    fn models(&self) -> Vec<tetraeder_gateway::engine::ModelInfo> {
        self.inner.models()
    }

    fn vocab(
        &self,
        model: &str,
    ) -> Result<Arc<tetraeder_gateway::engine::Vocab>, tetraeder_gateway::engine::EngineError> {
        self.inner.vocab(model)
    }

    fn tokenize(
        &self,
        model: &str,
        text: &str,
    ) -> Result<Vec<u32>, tetraeder_gateway::engine::EngineError> {
        self.inner.tokenize(model, text)
    }

    fn generate(
        &self,
        req: GenerateRequest,
    ) -> Result<tokio::sync::mpsc::Receiver<GenEvent>, tetraeder_gateway::engine::EngineError> {
        if self.failing.load(Ordering::SeqCst) {
            let (tx, rx) = tokio::sync::mpsc::channel(1);
            let _ = tx.try_send(GenEvent::Error("the engine's step failed".into()));
            return Ok(rx);
        }
        self.inner.generate(req)
    }
}

fn fake(letter: &'static str) -> Arc<dyn Engine> {
    let mut e = FakeEngine::new(move |_| letter.repeat(40));
    e.token_delay = Duration::from_millis(5);
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
    failing: Arc<AtomicBool>,
    board: Arc<Switchboard>,
    cfg: ClusterConfig,
    engine: Arc<ClusterEngine>,
    launcher_b: Arc<ReplicaLauncher>,
    daemon_a: Daemon,
    daemon_b: Option<Daemon>,
    _monitor: Arc<Monitor>,
    _head: ClusterHead,
}

fn rig() -> Rig {
    let board = MemTransport::switchboard();
    let cfg = config(DOWN_MS, &["a", "b"], &[1, 2]);
    let failing = Arc::new(AtomicBool::new(false));
    let f = Arc::clone(&failing);
    let launcher_b = Arc::new(ReplicaLauncher::new(CAP, move || {
        Arc::new(Flaky {
            inner: fake("B"),
            failing: Arc::clone(&f),
        })
    }));
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
                cap: CAP,
            },
            ReplicaTarget {
                engine: remote.clone() as Arc<dyn Engine>,
                remote: Some(remote),
                cap: CAP,
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
    assert!(
        wait_until(Duration::from_secs(10), || engine.up_count() == 2).is_some(),
        "both replicas join"
    );
    Rig {
        failing,
        board,
        cfg,
        engine,
        launcher_b,
        daemon_a,
        daemon_b: Some(daemon_b),
        _monitor: monitor,
        _head: head,
    }
}

#[derive(Debug)]
struct Rec {
    start: Duration,
    end: Duration,
    first: Option<u8>,
    error: Option<String>,
    terminals: usize,
    after_terminal: usize,
}

/// One closed-loop client: a streaming request after another until `stop`.
fn client(
    engine: &Arc<ClusterEngine>,
    n: usize,
    t0: Instant,
    stop: &AtomicBool,
    out: &Mutex<Vec<Rec>>,
) {
    let mut turn = 0;
    while !stop.load(Ordering::SeqCst) {
        turn += 1;
        let start = t0.elapsed();
        let prompt: Vec<u32> = format!("client {n} turn {turn} {}", "x".repeat(40))
            .bytes()
            .map(u32::from)
            .collect();
        let rx = engine.generate(GenerateRequest {
            model: "fake".into(),
            prompt,
            params: SamplingParams {
                max_tokens: 200,
                ..SamplingParams::default()
            },
            constraint: None,
            cancel: CancellationToken::new(),
        });
        let mut rx = match rx {
            Ok(rx) => rx,
            Err(e) => {
                out.lock().unwrap().push(Rec {
                    start,
                    end: t0.elapsed(),
                    first: None,
                    error: Some(format!("generate: {e}")),
                    terminals: 1,
                    after_terminal: 0,
                });
                std::thread::sleep(Duration::from_millis(50));
                continue;
            }
        };
        let mut rec = Rec {
            start,
            end: start,
            first: None,
            error: None,
            terminals: 0,
            after_terminal: 0,
        };
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            match rx.try_recv() {
                Ok(ev) => {
                    if rec.terminals > 0 {
                        rec.after_terminal += 1;
                    }
                    match ev {
                        GenEvent::Token(t) => {
                            rec.first.get_or_insert(t as u8);
                        }
                        GenEvent::Finished { .. } => {
                            rec.terminals += 1;
                            rec.end = t0.elapsed();
                        }
                        GenEvent::Error(m) | GenEvent::ContextExceeded(m) => {
                            rec.terminals += 1;
                            rec.end = t0.elapsed();
                            rec.error = Some(m);
                        }
                        GenEvent::Probs(_) => {}
                    }
                }
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                    if Instant::now() > deadline {
                        break; // ends silently: terminals stays 0
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => break,
            }
        }
        out.lock().unwrap().push(rec);
    }
}

#[derive(Clone, Copy, Debug)]
enum Fault {
    KillWorker,
    KillDaemon,
    Blackhole,
}

fn run(mode: Fault) {
    let mut rig = rig();
    let t0 = Instant::now();
    let stop = Arc::new(AtomicBool::new(false));
    let out = Arc::new(Mutex::new(Vec::new()));
    let clients: Vec<_> = (0..CLIENTS)
        .map(|n| {
            let (e, s, o) = (Arc::clone(&rig.engine), Arc::clone(&stop), Arc::clone(&out));
            std::thread::spawn(move || client(&e, n, t0, &s, &o))
        })
        .collect();
    // When R_B was in the router, for the failure message.
    let ups = Arc::new(Mutex::new(Vec::<(Duration, bool)>::new()));
    {
        let (e, s, u) = (Arc::clone(&rig.engine), Arc::clone(&stop), Arc::clone(&ups));
        std::thread::spawn(move || {
            while !s.load(Ordering::SeqCst) {
                let up = e.is_up(1);
                let mut g = u.lock().unwrap();
                if g.last().is_none_or(|l| l.1 != up) {
                    g.push((t0.elapsed(), up));
                }
                drop(g);
                std::thread::sleep(Duration::from_millis(10));
            }
        });
    }
    std::thread::sleep(Duration::from_millis(1500));
    // The rule about waiting requests needs some at the fault: with more clients than slots there
    // are, but a starved client thread can leave the queue empty for a moment, so the fault waits
    // for several (a gate, not luck).
    assert!(
        wait_until(Duration::from_secs(10), || rig.engine.queued()
            >= MIN_WAITING)
        .is_some(),
        "requests wait at the router before the fault"
    );

    let a = id("a");
    let b = id("b");
    let t_fault = t0.elapsed();
    match mode {
        Fault::KillWorker => rig.launcher_b.crash(),
        Fault::KillDaemon => {
            let mut d = rig.daemon_b.take().unwrap();
            d.stop();
        }
        Fault::Blackhole => {
            rig.board.fault(&a, &b, LinkFault::Blackhole);
            rig.board.fault(&b, &a, LinkFault::Blackhole);
        }
    }
    // Hold the fault past the detection bound so the "after" rule has requests to judge.
    let hold = BOUND + Duration::from_millis(1200);
    std::thread::sleep(hold);
    let t_restore = t0.elapsed();
    match mode {
        Fault::KillWorker => {}
        Fault::KillDaemon => {
            rig.daemon_b = Some(daemon(
                &rig.board,
                &rig.cfg,
                "b",
                Arc::clone(&rig.launcher_b) as _,
            ));
        }
        Fault::Blackhole => {
            rig.board.fault(&a, &b, LinkFault::None);
            rig.board.fault(&b, &a, LinkFault::None);
        }
    }
    let rejoin = wait_until(Duration::from_secs(10), || rig.engine.is_up(1))
        .expect("b rejoins without any manual action");
    let t_rejoined = t0.elapsed();
    // Serve for a while after the join so the rejoined replica is seen to take requests.
    std::thread::sleep(Duration::from_millis(2500));
    stop.store(true, Ordering::SeqCst);
    for c in clients {
        c.join().unwrap();
    }
    let recs = out.lock().unwrap();
    let counts = rig.engine.counts();
    let _ = &rig.daemon_a;

    let errors: Vec<&Rec> = recs.iter().filter(|r| r.error.is_some()).collect();
    for r in recs.iter() {
        assert_eq!(r.terminals, 1, "a request ends exactly once: {r:?}");
        assert_eq!(r.after_terminal, 0, "no event follows the end: {r:?}");
    }
    for r in &errors {
        let m = r.error.as_ref().unwrap();
        assert!(
            m.contains("replica 1"),
            "only requests on R_B fail, and the error names it: {m}"
        );
        assert!(
            r.end <= t_fault + BOUND,
            "an in-flight request ends within node_down_after + 1 s of the fault: ended {:?} after {:?}",
            r.end,
            t_fault
        );
        assert!(
            r.start < t_fault + BOUND,
            "only requests sent before the node was found down fail: {r:?}"
        );
    }
    assert!(
        !errors.is_empty(),
        "requests were in flight on R_B at the fault"
    );
    let late: Vec<&Rec> = recs
        .iter()
        .filter(|r| r.start >= t_fault + BOUND && r.start < t_restore)
        .collect();
    assert!(!late.is_empty(), "requests were submitted after the bound");
    assert!(
        late.iter().all(|r| r.error.is_none()),
        "requests submitted from node_down_after + 1 s on succeed"
    );
    if !matches!(mode, Fault::KillWorker) {
        // The node stays out until it is restored: what ran before that ran on R_A. (A request
        // that waited at the router may start on R_B once it is back.)
        assert!(
            late.iter()
                .filter(|r| r.end <= t_restore)
                .all(|r| r.first == Some(b'A')),
            "R_B in the router: {:?}",
            ups.lock().unwrap()
        );
    }
    assert!(
        recs.iter()
            .any(|r| r.start >= t_rejoined && r.first == Some(b'B') && r.error.is_none()),
        "R_B serves requests after it rejoined"
    );
    assert_eq!(
        counts.failed_in_flight,
        errors.len() as u64,
        "the router and the clients count the same failures"
    );
    assert!(counts.moved >= 1, "requests waiting at the router moved");
    let max_err = errors
        .iter()
        .map(|r| r.end.saturating_sub(t_fault))
        .max()
        .unwrap();
    println!(
        "case_f mode={mode:?} requests={} in_flight_errors={} max_error_s={:.3} bound_s={:.1} moved={} ok_after_bound={} rejoin_s={:.3}",
        recs.len(),
        errors.len(),
        max_err.as_secs_f64(),
        BOUND.as_secs_f64(),
        counts.moved,
        late.len(),
        (t_rejoined - t_restore)
            .as_secs_f64()
            .max(rejoin.as_secs_f64())
    );
}

// hardware: none
#[test]
fn case_f_kill_the_replica_worker() {
    run(Fault::KillWorker);
}

// hardware: none
#[test]
fn case_f_kill_the_daemon_and_the_worker() {
    run(Fault::KillDaemon);
}

// hardware: none
#[test]
fn case_f_blackhole_both_directions() {
    run(Fault::Blackhole);
}

// hardware: none
#[test]
fn case_f_the_engine_survives_and_its_requests_fail() {
    // The ADR's first mode as the daemon sees it when only the engine's steps fail: the node and
    // the replica process stay up, the replica stays in the router, and every request sent to
    // it ends with one error event until the engine works again.
    let rig = rig();
    let t0 = Instant::now();
    let stop = Arc::new(AtomicBool::new(false));
    let out = Arc::new(Mutex::new(Vec::new()));
    let clients: Vec<_> = (0..8)
        .map(|n| {
            let (e, s, o) = (Arc::clone(&rig.engine), Arc::clone(&stop), Arc::clone(&out));
            std::thread::spawn(move || client(&e, n, t0, &s, &o))
        })
        .collect();
    let left = Arc::new(AtomicBool::new(false));
    {
        let (e, s, l) = (
            Arc::clone(&rig.engine),
            Arc::clone(&stop),
            Arc::clone(&left),
        );
        std::thread::spawn(move || {
            while !s.load(Ordering::SeqCst) {
                if !e.is_up(1) {
                    l.store(true, Ordering::SeqCst);
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        });
    }
    std::thread::sleep(Duration::from_millis(1500));
    let t_fault = t0.elapsed();
    rig.failing.store(true, Ordering::SeqCst);
    std::thread::sleep(Duration::from_millis(1000));
    let t_restore = t0.elapsed();
    rig.failing.store(false, Ordering::SeqCst);
    std::thread::sleep(Duration::from_millis(2000));
    stop.store(true, Ordering::SeqCst);
    for c in clients {
        c.join().unwrap();
    }
    let recs = out.lock().unwrap();
    let counts = rig.engine.counts();
    for r in recs.iter() {
        assert_eq!(r.terminals, 1, "a request ends exactly once: {r:?}");
        assert_eq!(r.after_terminal, 0, "no event follows the end: {r:?}");
    }
    let errors: Vec<&Rec> = recs.iter().filter(|r| r.error.is_some()).collect();
    assert!(!errors.is_empty(), "requests sent to R_B failed");
    for r in &errors {
        assert!(
            r.end >= t_fault && r.end <= t_restore + Duration::from_millis(500),
            "an error only while the engine fails: {r:?}"
        );
        assert_eq!(r.first, None, "it failed before its first token");
    }
    assert!(
        !left.load(Ordering::SeqCst),
        "the node and the replica stayed up: R_B never left the router"
    );
    assert!(
        recs.iter()
            .filter(|r| r.start >= t_restore + Duration::from_millis(500))
            .all(|r| r.error.is_none()),
        "requests after the engine works again succeed"
    );
    assert_eq!(counts.failed_in_flight, errors.len() as u64);
    println!(
        "case_f mode=EngineErrors requests={} errors={} replica_left=false",
        recs.len(),
        errors.len()
    );
}

// hardware: none
#[test]
fn case_f_a_child_that_is_slow_to_be_reaped_does_not_delay_the_errors() {
    // The rehearsal's mode 1 (kill -9 of a replica engine on a 3090) ended its in-flight
    // requests 2.79 s after the fault where this simulation took 1 ms. A killed process closes
    // its pipes first and is reaped last (its GPU context goes in between), and the daemon waited
    // for the reap before it dropped the head's connection. The pipes are closed here at once and
    // `kill` returns after 3 s.
    let rig = rig();
    *rig.launcher_b.kill_delay.lock().unwrap() = Duration::from_secs(3);
    let t0 = Instant::now();
    let stop = Arc::new(AtomicBool::new(false));
    let out = Arc::new(Mutex::new(Vec::new()));
    let clients: Vec<_> = (0..CLIENTS)
        .map(|n| {
            let (e, s, o) = (Arc::clone(&rig.engine), Arc::clone(&stop), Arc::clone(&out));
            std::thread::spawn(move || client(&e, n, t0, &s, &o))
        })
        .collect();
    assert!(
        wait_until(Duration::from_secs(10), || rig.engine.queued()
            >= MIN_WAITING)
        .is_some(),
        "requests wait at the router before the fault"
    );
    let t_fault = t0.elapsed();
    rig.launcher_b.crash();
    std::thread::sleep(Duration::from_millis(1500));
    stop.store(true, Ordering::SeqCst);
    for c in clients {
        c.join().unwrap();
    }
    let recs = out.lock().unwrap();
    let errors: Vec<&Rec> = recs.iter().filter(|r| r.error.is_some()).collect();
    assert!(!errors.is_empty(), "requests were in flight on R_B");
    let late = errors
        .iter()
        .map(|r| r.end.saturating_sub(t_fault))
        .max()
        .unwrap();
    assert!(
        late < Duration::from_millis(500),
        "the in-flight errors came {late:?} after the kill, with the reap taking 3 s"
    );
}

/// Scheduler slack of a loaded CI machine on top of a computed bound: thread hops, the 1 ms
/// delivery poll and the clients' 1 ms receive poll.
const SLACK: Duration = Duration::from_millis(400);

/// Worst-case time from the fault to the in-flight error on the head, per mode, from the code
/// (ADR 0091 History, "failover bound"):
///
/// - mode 0, the engine killed, and mode 1, the daemon and the engine killed: no timer. The
///   daemon (mode 0) or the OS (mode 1) closes the head's replica connection when the engine's
///   pipe or the process ends, the head's reader sees the end and fails every request. Terms:
///   the close propagating (a one-way network delay, nil in the simulation).
/// - mode 2, a partition or a stopped process: nothing closes. The node is `Down` at the first
///   tick with `node_down_after` of silence, and that tick closes the watched connections.
///   `ClusterConfig::silent_detect_bound` is `node_down_after + tick + one pause step`.
fn bound(mode: Fault) -> Duration {
    let cfg = config(DOWN_MS, &["a", "b"], &[1, 2]);
    match mode {
        Fault::KillWorker | Fault::KillDaemon => Duration::ZERO,
        Fault::Blackhole => cfg.silent_detect_bound(),
    }
}

/// One fault at its worst phase, with requests in flight on R_B: the time to the last error.
fn error_after_fault(mode: Fault) -> Duration {
    let rig = rig();
    let t0 = Instant::now();
    let stop = Arc::new(AtomicBool::new(false));
    let out = Arc::new(Mutex::new(Vec::new()));
    let clients: Vec<_> = (0..CLIENTS)
        .map(|n| {
            let (e, s, o) = (Arc::clone(&rig.engine), Arc::clone(&stop), Arc::clone(&out));
            std::thread::spawn(move || client(&e, n, t0, &s, &o))
        })
        .collect();
    std::thread::sleep(Duration::from_millis(1500));
    let b = id("b");
    if matches!(mode, Fault::Blackhole) {
        // The worst phase of the heartbeat: the fault just after a reply, so the silence is
        // the whole of `node_down_after`.
        wait_until(Duration::from_secs(5), || {
            rig._monitor
                .silent_for(&b)
                .is_some_and(|d| d < Duration::from_millis(10))
        })
        .expect("a heartbeat reply arrives");
    }
    let t_fault = t0.elapsed();
    let mut rig = rig;
    match mode {
        Fault::KillWorker => rig.launcher_b.crash(),
        Fault::KillDaemon => rig.daemon_b.take().unwrap().stop(),
        Fault::Blackhole => {
            rig.board.fault(&id("a"), &b, LinkFault::Blackhole);
            rig.board.fault(&b, &id("a"), LinkFault::Blackhole);
        }
    }
    std::thread::sleep(Duration::from_secs(1) + bound(mode) + SLACK);
    stop.store(true, Ordering::SeqCst);
    for c in clients {
        c.join().unwrap();
    }
    let recs = out.lock().unwrap();
    let errors: Vec<&Rec> = recs
        .iter()
        .filter(|r| r.error.is_some() && r.end >= t_fault)
        .collect();
    assert!(!errors.is_empty(), "requests were in flight on R_B");
    errors
        .iter()
        .map(|r| r.end.saturating_sub(t_fault))
        .max()
        .unwrap()
}

fn assert_bound(mode: Fault) {
    let worst = error_after_fault(mode);
    let limit = bound(mode) + SLACK;
    println!(
        "case_f_bound mode={mode:?} worst_error_s={:.3} computed_bound_s={:.3} slack_s={:.1} three_s_rule={}",
        worst.as_secs_f64(),
        bound(mode).as_secs_f64(),
        SLACK.as_secs_f64(),
        limit < Duration::from_secs(3)
    );
    assert!(
        worst <= limit,
        "{mode:?}: the last in-flight error came {worst:?} after the fault, bound {limit:?}"
    );
}

// hardware: none
#[test]
fn case_f_bound_kill_the_engine() {
    assert_bound(Fault::KillWorker);
}

// hardware: none
#[test]
fn case_f_bound_kill_the_daemon_and_the_engine() {
    assert_bound(Fault::KillDaemon);
}

// hardware: none
#[test]
fn case_f_bound_partition_after_the_last_heartbeat() {
    assert_bound(Fault::Blackhole);
    // The production default (2 s) stays under the 3 s rule with the slack; the test runs the
    // shorter 1 s to keep the suite fast, the terms are the same.
    let prod = config(2000, &["a", "b"], &[1, 2]);
    assert!(prod.silent_detect_bound() + SLACK < Duration::from_secs(3));
}
