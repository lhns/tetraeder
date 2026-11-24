//! Node daemons over loopback TLS (ADR 0091 step 3): two daemons profile and probe each other,
//! with kernel-autotuned and with explicit socket buffers, `cluster.toml` pins the certificates,
//! and a killed and restarted daemon rejoins within `node_down_after` plus one heartbeat.
//!
//! Hardware label: none (loopback).

mod common;

use common::{EchoLauncher, id, node_graph_json, wait_until};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tetraeder_control::membership::NodeState;
use tetraeder_control::monitor::Monitor;
use tetraeder_control::node::{Daemon, NodeOptions, RestartPolicy};
use tetraeder_control::{ClusterConfig, client};
use tetraeder_transport::identity::{Generated, certs_from_pem, generate_self_signed};
use tetraeder_transport::{CertificateDer, Identity, NodeId, SocketBuffers, TlsTransport, Transport};

const DOWN_MS: u64 = 1000;
const DOWN: Duration = Duration::from_millis(DOWN_MS);
const HB: Duration = Duration::from_millis(DOWN_MS / 4);

struct Keys {
    gens: Vec<(String, Generated)>,
    trust: Vec<CertificateDer<'static>>,
    names: Vec<NodeId>,
    dir: PathBuf,
}

impl Keys {
    fn new(names: &[&str], tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("tetraeder-ctl-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let gens: Vec<_> = names
            .iter()
            .map(|n| ((*n).to_owned(), generate_self_signed(n).unwrap()))
            .collect();
        let mut trust = Vec::new();
        for (n, g) in &gens {
            std::fs::write(dir.join(format!("{n}.pem")), &g.cert_pem).unwrap();
            trust.extend(certs_from_pem(g.cert_pem.as_bytes()).unwrap());
        }
        Self {
            gens,
            trust,
            names: names.iter().map(|n| id(n)).collect(),
            dir,
        }
    }

    fn ident(&self, n: &str) -> Identity {
        let g = &self.gens.iter().find(|(x, _)| x == n).unwrap().1;
        Identity::from_pem(g.cert_pem.as_bytes(), g.key_pem.as_bytes()).unwrap()
    }

    fn transport(&self, n: &str, listen: Option<&str>, b: SocketBuffers) -> TlsTransport {
        TlsTransport::new(&self.ident(n), &self.trust, &self.names, listen, b).unwrap()
    }

    /// Writes and loads a `cluster.toml` for `ports`.
    fn config(&self, ports: &[u16]) -> ClusterConfig {
        let mut t = format!(
            "cluster = \"loop\"\nhead = \"{}\"\nnode_down_after_ms = {DOWN_MS}\n",
            self.names[0]
        );
        for (n, p) in self.names.iter().zip(ports) {
            t += &format!("[[nodes]]\nid = \"{n}\"\naddr = \"127.0.0.1:{p}\"\nanchor = \"{n}.pem\"\n");
        }
        let path = self.dir.join("cluster.toml");
        std::fs::write(&path, t).unwrap();
        ClusterConfig::load(&path).unwrap()
    }
}

impl Drop for Keys {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn daemon(keys: &Keys, cfg: &ClusterConfig, name: &str, t: TlsTransport) -> Daemon {
    let t = Arc::new(t);
    let (k, nm) = (keys.ident(name), name.to_owned());
    let (trust, names) = (keys.trust.clone(), keys.names.clone());
    let own = Arc::clone(&t);
    Daemon::start(
        t,
        NodeOptions {
            cfg: cfg.clone(),
            launcher: Arc::new(EchoLauncher::default()),
            profile: Arc::new(|| Ok(node_graph_json())),
            dial: Arc::new(move |buffers| {
                let _ = &nm;
                Ok(match buffers {
                    None => Arc::clone(&own) as Arc<dyn Transport>,
                    Some((send, recv)) => Arc::new(
                        TlsTransport::new(
                            &Identity {
                                cert: k.cert.clone(),
                                key: k.key.clone_key(),
                            },
                            &trust,
                            &names,
                            None,
                            SocketBuffers::Explicit { send, recv },
                        )
                        .map_err(std::io::Error::other)?,
                    ),
                })
            }),
            restart: RestartPolicy::default(),
        },
    )
}

// hardware: none
#[test]
fn two_daemons_profile_and_probe_each_other_with_both_buffer_settings() {
    let keys = Keys::new(&["a", "b"], "probe");
    let (ta, tb) = (
        keys.transport("a", Some("127.0.0.1:0"), SocketBuffers::Auto),
        keys.transport("b", Some("127.0.0.1:0"), SocketBuffers::Auto),
    );
    let ports = [ta.local_addr().unwrap().port(), tb.local_addr().unwrap().port()];
    let cfg = keys.config(&ports);
    // The anchors are the pinned certificates, each naming its node, and hold no key.
    assert_eq!(cfg.anchors().unwrap().len(), 2);
    let _da = daemon(&keys, &cfg, "a", ta);
    let _db = daemon(&keys, &cfg, "b", tb);
    let head = keys.transport("a", None, SocketBuffers::Auto);

    let sizes = [4096, 64 << 10, 1 << 20];
    let auto = client::probe_cluster(&head, &cfg, &sizes, 3, None).unwrap();
    assert_eq!(auto.buffers, "auto");
    assert_eq!(auto.curves.len(), 2, "both directions");
    assert_eq!(auto.links.len(), 1);
    let explicit = client::probe_cluster(&head, &cfg, &sizes, 3, Some((256 << 10, 256 << 10))).unwrap();
    assert_eq!(explicit.buffers, "262144,262144");
    for p in [&auto, &explicit] {
        for c in &p.curves {
            assert_eq!(c.points.len(), sizes.len());
            assert!(c.points.iter().all(|&(_, t)| t > 0.0), "{c:?}");
            // Bigger frames take longer.
            assert!(c.points[2].1 > c.points[0].1, "{c:?}");
        }
    }

    let hw = client::profile_cluster(&head, &cfg, Some(&auto)).unwrap();
    assert!(hw.devices.iter().all(|d| d.node.is_some()));
    assert!(hw.device("a/cpu0").is_some() && hw.device("b/cpu0").is_some());
    // The curves are keyed by node name with each node's cpu0, per direction.
    assert!(hw.network_curve("a", "b").is_some());
    assert!(hw.network_curve("b", "a").is_some());
    assert!(hw.network_link("a", "b").is_some());
    hw.validate().unwrap();
}

// hardware: none
#[test]
fn an_anchor_file_with_a_private_key_or_the_wrong_node_is_refused() {
    let keys = Keys::new(&["a", "b"], "anchor");
    let cfg = keys.config(&[1, 2]);
    let key_pem = &keys.gens[0].1.key_pem;
    std::fs::write(keys.dir.join("a.pem"), key_pem).unwrap();
    assert!(cfg.anchors().unwrap_err().contains("private key"));
    std::fs::write(keys.dir.join("a.pem"), &keys.gens[1].1.cert_pem).unwrap();
    assert!(cfg.anchors().unwrap_err().contains("certificate of `b`"));
}

// hardware: none
#[test]
fn a_killed_and_restarted_daemon_rejoins_within_down_after_plus_one_heartbeat() {
    let keys = Keys::new(&["a", "b"], "rejoin");
    let (ta, tb) = (
        keys.transport("a", Some("127.0.0.1:0"), SocketBuffers::Auto),
        keys.transport("b", Some("127.0.0.1:0"), SocketBuffers::Auto),
    );
    let ports = [ta.local_addr().unwrap().port(), tb.local_addr().unwrap().port()];
    let cfg = keys.config(&ports);
    let _da = daemon(&keys, &cfg, "a", ta);
    let mut db = daemon(&keys, &cfg, "b", tb);
    let monitor = Monitor::start(Arc::new(keys.transport("a", None, SocketBuffers::Auto)), cfg.clone());
    let up = |n: &str| monitor.state(&id(n)) == Some(NodeState::Up);
    wait_until(Duration::from_secs(10), || up("a") && up("b")).expect("not up");

    db.stop();
    drop(db);
    wait_until(DOWN, || monitor.state(&id("b")) == Some(NodeState::Down)).expect("b not down");

    // Same address again, as a restarted process. The old listener may need a moment to go.
    let t0 = Instant::now();
    let tb2 = loop {
        match TlsTransport::new(
            &keys.ident("b"),
            &keys.trust,
            &keys.names,
            Some(&format!("127.0.0.1:{}", ports[1])),
            SocketBuffers::Auto,
        ) {
            Ok(t) => break t,
            Err(e) => {
                assert!(t0.elapsed() < Duration::from_secs(5), "cannot rebind: {e:#}");
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    };
    let _db2 = daemon(&keys, &cfg, "b", tb2);
    let restarted = Instant::now();
    let slack = Duration::from_millis(500);
    wait_until(DOWN + HB + slack, || up("b")).expect("b did not rejoin");
    assert!(restarted.elapsed() <= DOWN + HB + slack, "{:?}", restarted.elapsed());
    assert!(up("a"));
}
