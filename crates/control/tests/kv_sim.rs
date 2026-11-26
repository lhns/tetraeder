//! Remote KV on simulated nodes (ADR 0091, step 5): the fetch rule against recompute, and a
//! prefix computed on one replica that is fetched or recomputed on the other as priced.
//!
//! Hardware label: none.

mod common;

use common::{EchoLauncher, ReplicaLauncher, config, id, wait_until};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tetraeder_control::head::{
    ClusterEngine, ClusterHead, FetchPolicy, KvInfo, ReplicaTarget, StartSpec,
};
use tetraeder_control::kv::{
    PrefixGrants, PrefixKey, PrefixStore, fetch_beats_recompute, serve_prefix,
};
use tetraeder_control::monitor::Monitor;
use tetraeder_control::node::{Daemon, NodeOptions, RestartPolicy};
use tetraeder_control::replica::{FetchFrom, KvHooks, RemoteEngine};
use tetraeder_gateway::engine::{Engine, GenEvent, GenerateRequest, SamplingParams};
use tetraeder_gateway::fake::FakeEngine;
use tetraeder_profiler::HardwareGraph;
use tetraeder_transport::{MemTransport, NodeId, Transport};
use tokio_util::sync::CancellationToken;

const KV_ROW_BYTES: u64 = 98_304;

fn graph(toml: &str) -> HardwareGraph {
    HardwareGraph::from_toml(toml).unwrap()
}

fn lan() -> HardwareGraph {
    graph(include_str!(
        "../../costmodel/fixtures/cluster/cluster_two_3090_lan.toml"
    ))
}

fn wan() -> HardwareGraph {
    graph(include_str!(
        "../../costmodel/fixtures/cluster/cluster_two_3090_wan.toml"
    ))
}

fn rule(hw: &HardwareGraph, tokens: u32, pp_tps: f64) -> bool {
    fetch_beats_recompute(
        hw,
        "a",
        "b",
        u64::from(tokens) * KV_ROW_BYTES,
        tokens,
        pp_tps,
        0.1,
        64,
    )
}

// hardware: none
#[test]
fn a_fetch_happens_only_when_it_is_predicted_cheaper_than_recompute() {
    let (lan, wan) = (lan(), wan());
    // A slow prefill makes recomputing 4096 tokens cost more than moving 400 MB; a fast one not.
    assert!(rule(&lan, 4096, 1.0));
    assert!(!rule(&lan, 4096, 1e9));
    // The decision flips once as the prefill rate grows, on either link.
    for hw in [&lan, &wan] {
        let flips = (0..60)
            .map(|i| rule(hw, 4096, 10f64.powf(f64::from(i) / 6.0 - 1.0)))
            .collect::<Vec<_>>()
            .windows(2)
            .filter(|w| w[0] != w[1])
            .count();
        assert!(
            flips <= 1,
            "fetch then recompute as the prefill gets faster"
        );
    }
    // A slower link never fetches where a faster one recomputes.
    for tokens in [64, 512, 4096, 32_768] {
        for i in 0..60 {
            let pp = 10f64.powf(f64::from(i) / 6.0 - 1.0);
            assert!(!rule(&wan, tokens, pp) || rule(&lan, tokens, pp));
        }
    }
    // Too short to matter (below the cache's restore gain): recompute whatever the rate.
    assert!(!rule(&lan, 32, 1.0));
    // No link between the nodes in the graph, or no prefill rate: recompute.
    let one = graph(include_str!(
        "../../costmodel/fixtures/two_3090_hoststaged.toml"
    ));
    assert!(!rule(&one, 4096, 1.0));
    assert!(!rule(&lan, 4096, 0.0));
}

struct Store {
    format: String,
    map: Mutex<HashMap<PrefixKey, Vec<u8>>>,
    served: AtomicUsize,
    puts: AtomicUsize,
}

impl Store {
    fn new(format: &str) -> Arc<Self> {
        Arc::new(Self {
            format: format.into(),
            map: Mutex::new(HashMap::new()),
            served: AtomicUsize::new(0),
            puts: AtomicUsize::new(0),
        })
    }
}

impl PrefixStore for Store {
    fn kv_format(&self) -> String {
        self.format.clone()
    }

    fn get(&self, key: &PrefixKey) -> Option<(Vec<u32>, Vec<u8>)> {
        let v = self.map.lock().unwrap().get(key).cloned()?;
        self.served.fetch_add(1, Ordering::SeqCst);
        Some((key.tokens.clone(), v))
    }

    fn put(&self, key: PrefixKey, bytes: Vec<u8>) -> bool {
        self.puts.fetch_add(1, Ordering::SeqCst);
        self.map.lock().unwrap().insert(key, bytes);
        true
    }

    fn row_bytes(&self) -> u64 {
        ROW_BYTES
    }
}

/// Bytes of one cached token of the fake stores; a snapshot of n tokens is n times this.
const ROW_BYTES: u64 = 20;
/// 50 tokens: a snapshot of 1000 bytes.
const PREFIX: &str = "a conversation about replicas and prefixes of text";

enum Policy {
    Fetch,
    Decline,
}

struct Fixed {
    /// Times the router asked: it only asks when a request goes to a replica other than the one
    /// holding the prefix, which is what these tests need to know (not a live count that a fast
    /// runner can see change).
    asked: Arc<AtomicUsize>,
    policy: Policy,
    node: NodeId,
    format: String,
}

impl FetchPolicy for Fixed {
    fn plan(&self, _holder: u32, _target: u32, tokens: &[u32], _kv: &KvInfo) -> Option<FetchFrom> {
        self.asked.fetch_add(1, Ordering::SeqCst);
        match self.policy {
            Policy::Decline => None,
            Policy::Fetch => Some(FetchFrom {
                node: self.node.clone(),
                key: PrefixKey {
                    format: self.format.clone(),
                    tokens: tokens.to_vec(),
                },
            }),
        }
    }
}

fn run_one(e: &Arc<ClusterEngine>, prompt: &str) -> Vec<GenEvent> {
    let mut rx = e
        .generate(GenerateRequest {
            model: "fake".into(),
            prompt: prompt.bytes().map(u32::from).collect(),
            params: SamplingParams {
                max_tokens: 400,
                ..SamplingParams::default()
            },
            constraint: None,
            cancel: CancellationToken::new(),
        })
        .unwrap();
    let mut out = Vec::new();
    while let Some(ev) = rx.blocking_recv() {
        out.push(ev);
    }
    out
}

fn script(p: &str) -> String {
    if p.contains("SLOW") {
        "x".repeat(300)
    } else {
        "y".repeat(4)
    }
}

/// Two replicas: R0 on the head (cap 1), R1 on node b behind its daemon. R0 holds the prefix
/// `P` (it ran it) and is busy with another request when the follow-up of `P` arrives, so the
/// follow-up goes to R1, which fetches the prefix from R0's node or recomputes it.
fn follow_up_on_the_other_replica(
    policy: Policy,
    holder_has_it: bool,
    holder_format: &str,
) -> (Arc<Store>, Arc<Store>, Vec<GenEvent>) {
    let asked = Arc::new(AtomicUsize::new(0));
    let board = MemTransport::switchboard();
    let cfg = config(1000, &["a", "b"], &[1, 2]);
    let (ta, tb): (Arc<dyn Transport>, Arc<dyn Transport>) = (
        Arc::new(board.node(&id("a"))),
        Arc::new(board.node(&id("b"))),
    );
    let opts = |launcher: Arc<dyn tetraeder_control::node::Launcher>| NodeOptions {
        cfg: cfg.clone(),
        launcher,
        profile: Arc::new(|| Ok(String::new())),
        dial: Arc::new(|_| Err(std::io::Error::other("no dial"))),
        restart: RestartPolicy::default(),
    };
    let (store_a, store_b) = (Store::new(holder_format), Store::new("f16"));
    let prefix = PREFIX;
    if holder_has_it {
        store_a.map.lock().unwrap().insert(
            PrefixKey {
                format: "f16".into(),
                tokens: prefix.bytes().map(u32::from).collect(),
            },
            vec![7u8; 1000],
        );
    }
    let launcher_b = Arc::new(ReplicaLauncher::new(2, || {
        let mut e = FakeEngine::new(script);
        e.token_delay = Duration::from_millis(1);
        Arc::new(e)
    }));
    {
        let (sb, tb, cfg) = (Arc::clone(&store_b), Arc::clone(&tb), cfg.clone());
        *launcher_b.kv.lock().unwrap() = Some(Arc::new(move || KvHooks {
            store: Arc::clone(&sb) as Arc<dyn PrefixStore>,
            transport: Arc::clone(&tb),
            peers: {
                let cfg = cfg.clone();
                Arc::new(move |n| cfg.peer(n))
            },
            grants: Arc::new(PrefixGrants::default()),
        }));
    }
    let daemon_a = Daemon::start(Arc::clone(&ta), opts(Arc::new(EchoLauncher::default())));
    // R0 runs on the head's node: the head announces fetches from it to these grants.
    let grants_a = Arc::new(PrefixGrants::default());
    {
        let (s, g) = (Arc::clone(&store_a), Arc::clone(&grants_a));
        daemon_a.set_prefix_handler(Arc::new(move |conn| {
            serve_prefix(conn, s.as_ref(), &g);
        }));
    }
    let _daemon_b = Daemon::start(Arc::clone(&tb), opts(Arc::clone(&launcher_b) as _));
    let monitor = Arc::new(Monitor::start(Arc::clone(&ta), cfg.clone()));
    let front: Arc<dyn Engine> = {
        let mut e = FakeEngine::new(script);
        e.token_delay = Duration::from_millis(5);
        Arc::new(e)
    };
    let remote = RemoteEngine::new(
        Arc::clone(&front),
        Arc::clone(&ta),
        cfg.peer(&id("b")).unwrap(),
        1,
    );
    let engine = Arc::new(ClusterEngine::new(
        Arc::clone(&front),
        vec![
            ReplicaTarget {
                engine: Arc::clone(&front),
                remote: None,
                cap: 1,
            },
            ReplicaTarget {
                engine: remote.clone() as Arc<dyn Engine>,
                remote: Some(remote),
                cap: 2,
            },
        ],
        8,
        Some(Arc::new(Fixed {
            asked: Arc::clone(&asked),
            policy,
            node: id("a"),
            format: "f16".into(),
        })),
    ));
    engine.set_local_grants(Arc::clone(&grants_a));
    engine.set_up(0, true);
    let _head = ClusterHead::start(
        &engine,
        &monitor,
        &ta,
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
    assert!(wait_until(Duration::from_secs(10), || engine.up_count() == 2).is_some());

    // R0 ran the prefix (least outstanding ties go to the lowest id).
    let first = run_one(&engine, prefix);
    assert!(matches!(first.last(), Some(GenEvent::Finished { .. })));
    // R0 gets busy; the follow-up finds it full.
    let busy = Arc::clone(&engine);
    let t = std::thread::spawn(move || run_one(&busy, "SLOW unrelated request"));
    assert!(wait_until(Duration::from_secs(2), || engine.outstanding(0) == 1).is_some());
    let events = run_one(&engine, &format!("{prefix} and one more turn"));
    assert_eq!(
        asked.load(Ordering::SeqCst),
        1,
        "the follow-up went to R1 while R0 held the prefix"
    );
    let _ = t.join();
    (store_a, store_b, events)
}

// hardware: none
#[test]
fn a_prefix_computed_on_one_replica_is_fetched_when_priced_cheaper() {
    let (a, b, events) = follow_up_on_the_other_replica(Policy::Fetch, true, "f16");
    assert!(matches!(events.last(), Some(GenEvent::Finished { .. })));
    assert_eq!(a.served.load(Ordering::SeqCst), 1, "R0's leader served it");
    assert_eq!(
        b.puts.load(Ordering::SeqCst),
        1,
        "R1 installed the snapshot"
    );
    let map = b.map.lock().unwrap();
    assert_eq!(map.values().next().map(Vec::len), Some(1000));
    println!("kv fetch=fetched bytes=1000 served=1 puts=1");
}

// hardware: none
#[test]
fn a_prefix_priced_dearer_than_recompute_is_recomputed() {
    let (a, b, events) = follow_up_on_the_other_replica(Policy::Decline, true, "f16");
    assert!(matches!(events.last(), Some(GenEvent::Finished { .. })));
    assert_eq!(a.served.load(Ordering::SeqCst), 0);
    assert_eq!(b.puts.load(Ordering::SeqCst), 0);
    println!("kv fetch=recomputed served=0 puts=0");
}

// hardware: none
#[test]
fn a_prefix_that_is_gone_or_in_another_format_is_recomputed() {
    // Evicted at the holder: `Gone`.
    let (_, b, events) = follow_up_on_the_other_replica(Policy::Fetch, false, "f16");
    assert!(matches!(events.last(), Some(GenEvent::Finished { .. })));
    assert_eq!(b.puts.load(Ordering::SeqCst), 0);
    // A holder whose snapshots are of another cut or format refuses.
    let (a, b, events) = follow_up_on_the_other_replica(Policy::Fetch, true, "f16@0-2,2-4");
    assert!(matches!(events.last(), Some(GenEvent::Finished { .. })));
    assert_eq!(a.served.load(Ordering::SeqCst), 0);
    assert_eq!(b.puts.load(Ordering::SeqCst), 0);
}
