//! Pre-authentication flooding (ADR 0091, review of steps 3 to 5): Vast boxes cannot firewall
//! ports to peer addresses, so any host can open sockets to a node. One source must not take the
//! handshake slots of the others, a silent socket is dropped after the TLS timeout, and one peer
//! must not fill the live map for the rest.
//!
//! Hardware label: none (loopback; the first test needs a second loopback address and skips
//! without it).

use rustls::pki_types::CertificateDer;
use std::sync::Arc;
use std::time::Duration;
use tetraeder_transport::identity::{Generated, certs_from_pem, generate_self_signed};
use tetraeder_transport::metrics::rejected_total;
use tetraeder_transport::{
    Conn, Identity, NodeId, Peer, Purpose, SocketBuffers, TlsTransport, Transport,
};

fn id(s: &str) -> NodeId {
    NodeId::new(s)
}

fn ident(g: &Generated) -> Identity {
    Identity::from_pem(g.cert_pem.as_bytes(), g.key_pem.as_bytes()).unwrap()
}

fn anchors(gs: &[&Generated]) -> Vec<CertificateDer<'static>> {
    gs.iter()
        .flat_map(|g| certs_from_pem(g.cert_pem.as_bytes()).unwrap())
        .collect()
}

fn node(g: &Generated, trust: &[CertificateDer<'static>], nodes: &[&str]) -> TlsTransport {
    let nodes: Vec<NodeId> = nodes.iter().map(|n| id(n)).collect();
    TlsTransport::new(
        &ident(g),
        trust,
        &nodes,
        Some("127.0.0.1:0"),
        SocketBuffers::Auto,
    )
    .unwrap()
}

fn peer(of: &TlsTransport, name: &str) -> Peer {
    Peer {
        id: id(name),
        addr: of.local_addr().unwrap().to_string(),
    }
}

fn accept_in_background(t: &Arc<TlsTransport>) -> std::sync::mpsc::Receiver<std::io::Result<Conn>> {
    let (tx, rx) = std::sync::mpsc::channel();
    let t = Arc::clone(t);
    std::thread::spawn(move || {
        let _ = tx.send(t.accept());
    });
    rx
}

/// A socket bound to `source` (another loopback address) and connected to `to`; `None` when the
/// platform does not give out that address.
fn connect_from(source: &str, to: std::net::SocketAddr) -> Option<std::net::TcpStream> {
    let s = socket2::Socket::new(socket2::Domain::IPV4, socket2::Type::STREAM, None).ok()?;
    let src: std::net::SocketAddr = format!("{source}:0").parse().ok()?;
    s.bind(&src.into()).ok()?;
    s.connect(&to.into()).ok()?;
    Some(s.into())
}

// hardware: none
#[test]
fn silent_sockets_of_one_source_do_not_block_another_source() {
    let (ga, gb) = (
        generate_self_signed("a").unwrap(),
        generate_self_signed("b").unwrap(),
    );
    let trust = anchors(&[&ga, &gb]);
    let a = Arc::new(node(&ga, &trust, &["a", "b"]));
    let b = node(&gb, &trust, &["a", "b"]);
    let addr = a.local_addr().unwrap();
    // The attacker is 127.0.0.2, the real node 127.0.0.1.
    let Some(first) = connect_from("127.0.0.2", addr) else {
        eprintln!("skipped: no second loopback address");
        return;
    };
    let before = (rejected_total("ip_limit"), rejected_total("tls_timeout"));
    let mut silent = vec![first];
    silent.extend((0..40).filter_map(|_| connect_from("127.0.0.2", addr)));
    let t0 = std::time::Instant::now();
    while rejected_total("ip_limit") == before.0 {
        assert!(
            t0.elapsed() < Duration::from_secs(5),
            "the source was not capped"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    // The attacker still holds its slots; a real connect from another source goes through.
    let acc = accept_in_background(&a);
    b.connect(&peer(&a, "a"), Purpose::Control)
        .expect("a real node connects while another source is flooding");
    assert_eq!(
        acc.recv_timeout(Duration::from_secs(5))
            .unwrap()
            .unwrap()
            .peer,
        id("b")
    );
    // A socket that never starts TLS is dropped after 3 s, not 10.
    let t0 = std::time::Instant::now();
    while rejected_total("tls_timeout") == before.1 {
        assert!(
            t0.elapsed() < Duration::from_secs(8),
            "silent sockets were not timed out"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    drop(silent);
}

// hardware: none
#[test]
fn one_peer_cannot_fill_the_live_map_for_others() {
    let (ga, gb, gc) = (
        generate_self_signed("a").unwrap(),
        generate_self_signed("b").unwrap(),
        generate_self_signed("c").unwrap(),
    );
    let trust = anchors(&[&ga, &gb, &gc]);
    let a = Arc::new(node(&ga, &trust, &["a", "b", "c"]));
    let b = node(&gb, &trust, &["a", "b", "c"]);
    let c = node(&gc, &trust, &["a", "b", "c"]);
    // Take every inbound connection and keep it open.
    let drain = {
        let a = Arc::clone(&a);
        std::thread::spawn(move || {
            let mut held = Vec::new();
            while let Ok(conn) = a.accept() {
                held.push(conn);
            }
        })
    };
    let before = rejected_total("peer_limit");
    let mut kept = Vec::new();
    let mut refused = 0;
    for session in 0..80 {
        match b.connect(&peer(&a, "a"), Purpose::Probe { session }) {
            Ok(conn) => kept.push(conn),
            Err(e) => {
                assert!(e.to_string().contains("too many connections"), "{e}");
                refused += 1;
            }
        }
    }
    assert!(refused > 0, "one peer filled the map");
    assert!(rejected_total("peer_limit") > before);
    // Another peer is not affected.
    c.connect(&peer(&a, "a"), Purpose::Control)
        .expect("another peer connects");
    a.shutdown();
    drop(kept);
    let _ = drain.join();
}
