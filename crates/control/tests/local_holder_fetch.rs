//! The head's own replica as the holder of a prefix a remote replica fetches
//! (`cluster-prefix-fetch-not-triggered`, ADR 0091 History).
//!
//! On the rehearsal's two boxes no fetch happened although the rule was priced to win. The
//! head's in-process replica never told the router its snapshot format and size, so the real
//! policy (which needs both, as `PricedFetch` does) declined, and nothing served the head's
//! cache. `kv_sim.rs` missed it because its policy ignores the holder's `KvInfo`.
//!
//! Here the policy declines without the holder's format and row size, the holder is registered
//! with `ClusterEngine::set_local_replica`, and the remote replica fetches: the fetch happens
//! (the holder's store serves it, the fetcher installs it) and the answer equals the
//! recomputed one.
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
use tetraeder_control::kv::{PrefixGrants, PrefixKey, PrefixStore, serve_prefix};
use tetraeder_control::monitor::Monitor;
use tetraeder_control::node::{Daemon, NodeOptions, RestartPolicy};
use tetraeder_control::replica::{FetchFrom, KvHooks, RemoteEngine};
use tetraeder_gateway::engine::{Engine, GenEvent, GenerateRequest, SamplingParams};
use tetraeder_gateway::fake::FakeEngine;
use tetraeder_transport::{MemTransport, Transport};
use tokio_util::sync::CancellationToken;

const ROW_BYTES: u64 = 20;
/// 50 tokens: a snapshot of 1000 bytes.
const PREFIX: &str = "a conversation about replicas and prefixes of text";

struct Store {
    map: Mutex<HashMap<PrefixKey, Vec<u8>>>,
    served: AtomicUsize,
    puts: AtomicUsize,
}

impl Store {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            map: Mutex::new(HashMap::new()),
            served: AtomicUsize::new(0),
            puts: AtomicUsize::new(0),
        })
    }
}

impl PrefixStore for Store {
    fn kv_format(&self) -> String {
        "f16".into()
    }

    fn get(&self, key: &PrefixKey) -> Option<(Vec<u32>, Vec<u8>)> {
        let v = self.map.lock().unwrap().get(key).cloned()?;
        self.served.fetch_add(1, Ordering::SeqCst);
        Some((key.tokens.clone(), v))
    }

    fn put(&self, key: PrefixKey, bytes: Vec<u8>) {
        self.puts.fetch_add(1, Ordering::SeqCst);
        self.map.lock().unwrap().insert(key, bytes);
    }

    fn row_bytes(&self) -> u64 {
        ROW_BYTES
    }
}

/// What `PricedFetch` needs of the holder: its format and the size of a cached token. It fetches
/// whenever it has both (the price is not what this test is about).
struct NeedsHolderInfo {
    asked: AtomicUsize,
    declined_blind: AtomicUsize,
}

impl FetchPolicy for NeedsHolderInfo {
    fn plan(&self, holder: u32, _target: u32, tokens: &[u32], kv: &KvInfo) -> Option<FetchFrom> {
        self.asked.fetch_add(1, Ordering::SeqCst);
        if kv.row_bytes == 0 || kv.format.is_empty() {
            self.declined_blind.fetch_add(1, Ordering::SeqCst);
            return None;
        }
        assert_eq!(holder, 0, "the head's replica holds the prefix");
        Some(FetchFrom {
            node: id("a"),
            key: PrefixKey {
                format: kv.format.clone(),
                tokens: tokens.to_vec(),
            },
        })
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

fn tokens(events: &[GenEvent]) -> Vec<u32> {
    events
        .iter()
        .filter_map(|e| match e {
            GenEvent::Token(t) => Some(*t),
            _ => None,
        })
        .collect()
}

fn script(p: &str) -> String {
    if p.contains("SLOW") {
        "x".repeat(300)
    } else {
        "y".repeat(4)
    }
}

struct Run {
    store_a: Arc<Store>,
    store_b: Arc<Store>,
    policy: Arc<NeedsHolderInfo>,
    follow_up: Vec<GenEvent>,
}

/// R0 on the head's node (cap 1) holds the prefix and is busy when the follow-up arrives, so it
/// goes to R1 on node b. `register`: the head registers R0 as a holder (`set_local_replica`);
/// without it only the grants are set, as the head's wiring did.
fn run(register: bool) -> Run {
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
    let (store_a, store_b) = (Store::new(), Store::new());
    store_a.map.lock().unwrap().insert(
        PrefixKey {
            format: "f16".into(),
            tokens: PREFIX.bytes().map(u32::from).collect(),
        },
        vec![7u8; 1000],
    );
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
    // The head's cache is served on node a's `Prefix` purpose, as the head process does it.
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
    let policy = Arc::new(NeedsHolderInfo {
        asked: AtomicUsize::new(0),
        declined_blind: AtomicUsize::new(0),
    });
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
        Some(Arc::clone(&policy) as Arc<dyn FetchPolicy>),
    ));
    if register {
        engine.set_local_replica(
            0,
            Arc::clone(&store_a) as Arc<dyn PrefixStore>,
            Arc::clone(&grants_a),
        );
    } else {
        engine.set_local_grants(Arc::clone(&grants_a));
    }
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
    // R0 ran the prefix.
    let first = run_one(&engine, PREFIX);
    assert!(matches!(first.last(), Some(GenEvent::Finished { .. })));
    // R0 gets busy; the follow-up finds it full and goes to R1.
    let busy = Arc::clone(&engine);
    let t = std::thread::spawn(move || run_one(&busy, "SLOW unrelated request"));
    assert!(wait_until(Duration::from_secs(2), || engine.outstanding(0) == 1).is_some());
    let follow_up = run_one(&engine, &format!("{PREFIX} and one more turn"));
    let _ = t.join();
    Run {
        store_a,
        store_b,
        policy,
        follow_up,
    }
}

// hardware: none
#[test]
fn the_heads_own_replica_as_holder_is_fetched_from() {
    let r = run(true);
    assert!(matches!(
        r.follow_up.last(),
        Some(GenEvent::Finished { .. })
    ));
    assert_eq!(r.policy.declined_blind.load(Ordering::SeqCst), 0);
    assert_eq!(
        r.store_a.served.load(Ordering::SeqCst),
        1,
        "the head's replica served the prefix"
    );
    assert_eq!(
        r.store_b.puts.load(Ordering::SeqCst),
        1,
        "the remote replica installed it"
    );
    // What the fake engine says to this prompt, fetched or not.
    let want: Vec<u32> = "y".repeat(4).bytes().map(u32::from).collect();
    assert_eq!(tokens(&r.follow_up), want, "tokens equal");
    println!("local_holder_fetch fetched=1 served=1 puts=1 tokens_equal=true");
}

// hardware: none
#[test]
fn without_the_holders_format_and_row_size_the_policy_recomputes() {
    let r = run(false);
    assert!(matches!(
        r.follow_up.last(),
        Some(GenEvent::Finished { .. })
    ));
    assert!(r.policy.declined_blind.load(Ordering::SeqCst) >= 1);
    assert_eq!(r.store_a.served.load(Ordering::SeqCst), 0);
    assert_eq!(r.store_b.puts.load(Ordering::SeqCst), 0);
}
