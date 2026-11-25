//! Fixes from the security review of Phase 6 steps 3 to 5 (ADR 0091 History): who may use an
//! assigned segment worker, a prefix fetch the head announced, a leader that streams more tokens
//! than asked for, and a hardware graph with absurd numbers.
//!
//! Hardware label: none.

mod common;

use common::{EchoLauncher, config, id, node_graph_json, wait_until};
use std::sync::Arc;
use std::time::Duration;
use tetraeder_control::client::check_profile;
use tetraeder_control::head::{ClusterEngine, ReplicaTarget};
use tetraeder_control::kv::{PrefixGrants, PrefixKey, PrefixStore};
use tetraeder_control::node::{Daemon, NodeOptions, RestartPolicy};
use tetraeder_control::proto::NodeRequest;
use tetraeder_control::replica::{KvHooks, RemoteEngine, ReplicaServer, serve_conn};
use tetraeder_control::{ClusterConfig, client};
use tetraeder_gateway::engine::{
    Engine, EngineError, EngineStats, FinishReason, GenEvent, GenerateRequest, ModelInfo,
    SamplingParams, TokenId, Vocab,
};
use tetraeder_gateway::fake::FakeEngine;
use tetraeder_profiler::{HardwareGraph, Measurement};
use tetraeder_transport::metrics::rejected_total;
use tetraeder_transport::{MemTransport, Purpose, Switchboard, Transport, frame};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

fn script(_: &str) -> String {
    "y".repeat(4)
}

fn daemon(board: &Arc<Switchboard>, cfg: &ClusterConfig, name: &str) -> Daemon {
    let t: Arc<dyn Transport> = Arc::new(board.node(&id(name)));
    let dial = Arc::clone(&t);
    Daemon::start(
        t,
        NodeOptions {
            cfg: cfg.clone(),
            launcher: Arc::new(EchoLauncher::default()),
            profile: Arc::new(|| Ok(node_graph_json())),
            dial: Arc::new(move |_| Ok(Arc::clone(&dial))),
            restart: RestartPolicy::default(),
        },
    )
}

// hardware: none
#[test]
fn only_the_head_and_the_leader_may_use_a_segment_worker() {
    let board = MemTransport::switchboard();
    // a is the head, b hosts the worker, c leads the replica, d is another configured node.
    let cfg = config(1000, &["a", "b", "c", "d"], &[1, 2, 3, 4]);
    let _d = daemon(&board, &cfg, "b");
    let head = board.node(&id("a"));
    let peer = cfg.peer(&id("b")).unwrap();
    client::request(
        &head,
        &peer,
        &NodeRequest::Assign {
            replica: 0,
            segment: 0,
            plan: "plan".into(),
            model_sha256: "00".into(),
            leader: Some(id("c")),
        },
        &[],
    )
    .unwrap();
    let seg = Purpose::Segment {
        replica: 0,
        segment: 0,
    };
    let echoes = |t: &MemTransport| {
        let mut c = t.connect(&peer, seg.clone()).unwrap();
        frame::write_frame(&mut c.tx, &serde_json::json!({ "n": 1 }), b"rows").unwrap();
        matches!(frame::read_frame(&mut c.rx), Ok(Some(f)) if f.payload == b"rows")
    };
    let before = rejected_total("not_leader");
    assert!(echoes(&board.node(&id("c"))), "the leader uses its worker");
    assert!(
        !echoes(&board.node(&id("d"))),
        "another configured node is closed out, not relayed"
    );
    assert!(rejected_total("not_leader") > before);
    assert!(echoes(&head), "the head may too");
}

// hardware: none
#[test]
fn without_a_leader_only_the_head_may_use_a_segment_worker() {
    let board = MemTransport::switchboard();
    let cfg = config(1000, &["a", "b", "c"], &[1, 2, 3]);
    let _d = daemon(&board, &cfg, "b");
    let head = board.node(&id("a"));
    let peer = cfg.peer(&id("b")).unwrap();
    client::request(
        &head,
        &peer,
        &NodeRequest::Assign {
            replica: 0,
            segment: 0,
            plan: "plan".into(),
            model_sha256: "00".into(),
            leader: None,
        },
        &[],
    )
    .unwrap();
    let mut c = board
        .node(&id("c"))
        .connect(
            &peer,
            Purpose::Segment {
                replica: 0,
                segment: 0,
            },
        )
        .unwrap();
    let _ = frame::write_frame(&mut c.tx, &serde_json::json!({ "n": 1 }), b"rows");
    assert!(matches!(frame::read_frame(&mut c.rx), Ok(None) | Err(_)));
}

/// A leader that ignores `max_tokens`.
struct Chatty {
    inner: FakeEngine,
    tokens: TokenId,
}

impl Engine for Chatty {
    fn models(&self) -> Vec<ModelInfo> {
        self.inner.models()
    }
    fn vocab(&self, model: &str) -> Result<Arc<Vocab>, EngineError> {
        self.inner.vocab(model)
    }
    fn tokenize(&self, model: &str, text: &str) -> Result<Vec<TokenId>, EngineError> {
        self.inner.tokenize(model, text)
    }
    fn generate(&self, _req: GenerateRequest) -> Result<mpsc::Receiver<GenEvent>, EngineError> {
        let (tx, rx) = mpsc::channel(256);
        for t in 0..self.tokens {
            tx.try_send(GenEvent::Token(t)).unwrap();
        }
        tx.try_send(GenEvent::Finished {
            reason: FinishReason::EndOfGeneration,
            stats: EngineStats::default(),
        })
        .unwrap();
        Ok(rx)
    }
}

// hardware: none
#[test]
fn a_leader_that_streams_past_max_tokens_is_cut_off() {
    let chatty: Arc<dyn Engine> = Arc::new(Chatty {
        inner: FakeEngine::new(script),
        tokens: 50,
    });
    let engine = ClusterEngine::new(
        Arc::new(FakeEngine::new(script)),
        vec![ReplicaTarget {
            engine: chatty,
            remote: None,
            cap: 2,
        }],
        8,
        None,
    );
    engine.set_up(0, true);
    let mut rx = engine
        .generate(GenerateRequest {
            model: "fake".into(),
            prompt: vec![1, 2, 3],
            params: SamplingParams {
                max_tokens: 5,
                ..SamplingParams::default()
            },
            constraint: None,
            cancel: CancellationToken::new(),
        })
        .unwrap();
    let mut tokens = 0;
    let mut last = None;
    while let Some(ev) = rx.blocking_recv() {
        match ev {
            GenEvent::Token(_) => tokens += 1,
            other => last = Some(other),
        }
    }
    assert_eq!(tokens, 5);
    assert!(
        matches!(
            last,
            Some(GenEvent::Finished {
                reason: FinishReason::Length,
                ..
            })
        ),
        "{last:?}"
    );
}

struct NoStore;

impl PrefixStore for NoStore {
    fn kv_format(&self) -> String {
        "f16".into()
    }
    fn get(&self, _key: &PrefixKey) -> Option<(Vec<u32>, Vec<u8>)> {
        None
    }
    fn put(&self, _key: PrefixKey, _bytes: Vec<u8>) {}
}

// hardware: none
#[test]
fn the_heads_offer_reaches_the_replicas_grants() {
    let board = MemTransport::switchboard();
    let cfg = config(1000, &["a", "b"], &[1, 2]);
    let (ta, tb): (Arc<dyn Transport>, Arc<dyn Transport>) = (
        Arc::new(board.node(&id("a"))),
        Arc::new(board.node(&id("b"))),
    );
    let grants = Arc::new(PrefixGrants::default());
    let server = ReplicaServer::new(
        Arc::new(FakeEngine::new(script)),
        1,
        Some(KvHooks {
            store: Arc::new(NoStore),
            transport: Arc::clone(&tb),
            peers: Arc::new(|_| None),
            grants: Arc::clone(&grants),
        }),
    );
    let accept = Arc::clone(&tb);
    std::thread::spawn(move || {
        while let Ok(conn) = accept.accept() {
            let server = Arc::clone(&server);
            std::thread::spawn(move || serve_conn(&server, conn));
        }
    });
    let remote = RemoteEngine::new(
        Arc::new(FakeEngine::new(script)),
        ta,
        cfg.peer(&id("b")).unwrap(),
        0,
    );
    remote.connect(&|_| {}).unwrap();
    let key = PrefixKey {
        format: "f16".into(),
        tokens: vec![1, 2, 3],
    };
    assert!(remote.offer(id("c"), key.clone()));
    assert!(wait_until(Duration::from_secs(5), || grants.len() == 1).is_some());
    assert!(grants.take(&id("c"), &key, Duration::ZERO));
}

fn graph() -> HardwareGraph {
    HardwareGraph::from_toml(include_str!(
        "../../costmodel/fixtures/cluster/cluster_two_3090_lan.toml"
    ))
    .unwrap()
}

// hardware: none
#[test]
fn a_hardware_graph_with_absurd_numbers_is_refused() {
    check_profile(&graph()).unwrap();
    let mut nan = graph();
    nan.devices[0].bandwidth_bytes_per_s = Some(Measurement::prior(f64::NAN));
    assert!(check_profile(&nan).is_err());
    let mut negative = graph();
    negative.links[0].beta_bytes_per_s = Measurement::prior(-1.0);
    assert!(check_profile(&negative).is_err());
    let mut huge = graph();
    huge.devices[0].capacity_bytes = u64::MAX;
    assert!(check_profile(&huge).is_err());
    let mut many = graph();
    let d = many.devices[0].clone();
    many.devices.extend(std::iter::repeat_n(d, 300));
    assert!(check_profile(&many).is_err());
    let mut lanes = graph();
    lanes.lane_groups.push(tetraeder_profiler::LaneGroup {
        id: "x".into(),
        capacity_bytes_per_s: f64::INFINITY,
    });
    assert!(check_profile(&lanes).is_err());
}
