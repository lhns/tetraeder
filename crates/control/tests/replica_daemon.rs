//! The daemon's rules for the replica purposes (ADR 0091, step 5) and the replica engine under
//! its supervisor: who may open `Replica` and `Prefix`, `StartReplica` repeated, and a request
//! that outlives the head connection that started it.
//!
//! Hardware label: none.

mod common;

use common::{EchoLauncher, ReplicaLauncher, config, id, wait_until};
use std::io::Write as _;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;
use tetraeder_control::ClusterConfig;
use tetraeder_control::client;
use tetraeder_control::node::{Daemon, NodeOptions, REPLICA_SEGMENT, RestartPolicy};
use tetraeder_control::proto::{NodeReply, NodeRequest};
use tetraeder_control::replica::RemoteEngine;
use tetraeder_gateway::engine::{Engine, GenEvent, GenerateRequest, SamplingParams};
use tetraeder_gateway::fake::FakeEngine;
use tetraeder_transport::{MemTransport, Purpose, Switchboard, Transport};
use tokio_util::sync::CancellationToken;

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
                max_restarts: 3,
                window: Duration::from_secs(60),
            },
        },
    )
}

fn start(replica: u32, plan: &str) -> NodeRequest {
    NodeRequest::StartReplica {
        replica,
        plan: plan.into(),
        model_sha256: "0".repeat(64),
        segments: Vec::new(),
        predicted_s: Vec::new(),
    }
}

fn closed(c: &mut tetraeder_transport::Conn) -> bool {
    // A refused connection ends: a read returns end of input or an error.
    c.tx.write_all(&[0]).ok();
    let mut b = [0u8; 1];
    std::io::Read::read(&mut c.rx, &mut b).map_or(true, |n| n == 0)
}

// hardware: none
#[test]
fn the_replica_purposes_are_open_only_to_who_may_use_them() {
    let board = MemTransport::switchboard();
    let cfg = config(1000, &["a", "b", "c"], &[1, 2, 3]);
    let launcher = Arc::new(ReplicaLauncher::new(1, || Arc::new(FakeEngine::fixed("x"))));
    let _b = daemon(&board, &cfg, "b", Arc::clone(&launcher) as _);
    let ta = board.node(&id("a"));
    let tc = board.node(&id("c"));
    let peer = cfg.peer(&id("b")).unwrap();

    // Before StartReplica there is no engine to relay to.
    let mut c = ta.connect(&peer, Purpose::Replica { replica: 1 }).unwrap();
    assert!(
        closed(&mut c),
        "a replica connection without a start is closed"
    );
    // Another node than the head never reaches the engine.
    client::request(&ta, &peer, &start(1, "p"), &[]).unwrap();
    let mut c = tc.connect(&peer, Purpose::Replica { replica: 1 }).unwrap();
    assert!(closed(&mut c), "only the head opens a replica connection");
    // A prefix connection needs a handler.
    let mut c = tc.connect(&peer, Purpose::Prefix).unwrap();
    assert!(closed(&mut c), "no prefix handler: closed");
    // A node cannot reach the engine through the segment purpose, nor assign into its slot.
    let mut c = tc
        .connect(
            &peer,
            Purpose::Segment {
                replica: 1,
                segment: REPLICA_SEGMENT,
            },
        )
        .unwrap();
    assert!(closed(&mut c));
    let r = client::request(
        &ta,
        &peer,
        &NodeRequest::Assign {
            replica: 1,
            segment: REPLICA_SEGMENT,
            plan: String::new(),
            model_sha256: "0".repeat(64),
        },
        &[],
    );
    assert!(
        r.is_err(),
        "the replica slot is not assignable as a segment"
    );
    // And only the head starts one.
    let r = client::request(&tc, &peer, &start(2, "p"), &[]);
    assert!(
        r.is_err(),
        "a node other than the head cannot start a replica"
    );
}

// hardware: none
#[test]
fn starting_the_same_replica_again_keeps_the_running_engine() {
    let board = MemTransport::switchboard();
    let cfg = config(1000, &["a", "b"], &[1, 2]);
    let launcher = Arc::new(ReplicaLauncher::new(1, || Arc::new(FakeEngine::fixed("x"))));
    let d = daemon(&board, &cfg, "b", Arc::clone(&launcher) as _);
    let ta = board.node(&id("a"));
    let peer = cfg.peer(&id("b")).unwrap();
    for _ in 0..3 {
        let (r, _) = client::request(&ta, &peer, &start(1, "plan"), &[]).unwrap();
        assert!(matches!(r, NodeReply::Started));
    }
    assert!(
        wait_until(Duration::from_secs(2), || launcher
            .launches
            .load(Ordering::SeqCst)
            == 1)
        .is_some()
    );
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(launcher.launches.load(Ordering::SeqCst), 1, "one engine");
    assert_eq!(d.workers(), 1);
    // A different plan is a different replica: it restarts.
    client::request(&ta, &peer, &start(1, "other plan"), &[]).unwrap();
    assert!(
        wait_until(Duration::from_secs(2), || launcher
            .launches
            .load(Ordering::SeqCst)
            == 2)
        .is_some()
    );
}

fn ask(e: &Arc<RemoteEngine>, prompt: &str, max: u32) -> tokio::sync::mpsc::Receiver<GenEvent> {
    e.generate(GenerateRequest {
        model: "fake".into(),
        prompt: prompt.bytes().map(u32::from).collect(),
        params: SamplingParams {
            max_tokens: max,
            ..SamplingParams::default()
        },
        constraint: None,
        cancel: CancellationToken::new(),
    })
    .unwrap()
}

// hardware: none
#[test]
fn a_request_does_not_outlive_the_head_connection_that_started_it() {
    let board = MemTransport::switchboard();
    let cfg = config(1000, &["a", "b"], &[1, 2]);
    let mut slow = FakeEngine::new(|_| "z".repeat(500));
    slow.token_delay = Duration::from_millis(5);
    let slow = Arc::new(slow);
    let cancelled = Arc::clone(&slow.cancelled);
    let eng = Arc::clone(&slow);
    let launcher = Arc::new(ReplicaLauncher::new(2, move || Arc::clone(&eng) as _));
    let _b = daemon(&board, &cfg, "b", launcher as _);
    let ta: Arc<dyn Transport> = Arc::new(board.node(&id("a")));
    let peer = cfg.peer(&id("b")).unwrap();
    client::request(ta.as_ref(), &peer, &start(1, "p"), &[]).unwrap();
    let front: Arc<dyn Engine> = Arc::new(FakeEngine::fixed("x"));

    // The first head connection starts a long request and goes away (a partition that healed
    // after the head gave up): the engine, which outlives it, must not keep running it.
    let one = RemoteEngine::new(Arc::clone(&front), Arc::clone(&ta), peer.clone(), 1);
    one.connect(&|_| {}).unwrap();
    one.probe().unwrap();
    let mut rx = ask(&one, "long", 400);
    assert!(matches!(rx.blocking_recv(), Some(GenEvent::Token(_))));
    // The head dies without a word: only the transport knows.
    one.disconnect();
    assert!(matches!(
        rx.blocking_recv(),
        Some(GenEvent::Error(m)) if m.contains("replica 1")
    ));
    assert!(rx.blocking_recv().is_none(), "no event follows the error");
    // A new connection finds the engine free of the zombie.
    let two = RemoteEngine::new(front, ta, peer, 1);
    two.connect(&|_| {}).unwrap();
    assert!(
        wait_until(Duration::from_secs(2), || cancelled.load(Ordering::SeqCst)
            == 1)
        .is_some(),
        "the engine cancelled the request of the connection that ended"
    );
    let mut rx = ask(&two, "short", 3);
    let mut n = 0;
    while let Some(ev) = rx.blocking_recv() {
        if matches!(ev, GenEvent::Token(_)) {
            n += 1;
        }
    }
    assert_eq!(n, 3);
    let _ = EchoLauncher::default();
}
