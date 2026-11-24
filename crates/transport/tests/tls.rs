//! `TlsTransport` over loopback (ADR 0091 step 1): frames in both directions, one connection per
//! purpose, mutual authentication against pinned self-signed and CA anchors, refusals counted,
//! socket buffers read back.
//!
//! Hardware label: none (loopback).

use rustls::pki_types::CertificateDer;
use std::io::{ErrorKind, Read, Write};
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;
use tetraeder_transport::identity::{
    Generated, certs_from_pem, check_node_id, generate_self_signed, testing,
};
use tetraeder_transport::metrics::rejected_total;
use tetraeder_transport::{
    Conn, Identity, NodeId, Peer, Purpose, SocketBuffers, TlsTransport, Transport,
    effective_buffers, frame,
};

/// The refusal is counted by the node that refused, which may be a moment after the dialer
/// saw its error.
fn refused_since(reason: &str, before: u64) {
    for _ in 0..200 {
        if rejected_total(reason) > before {
            return;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("no `{reason}` refusal was counted");
}

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

fn node(
    g: &Generated,
    trust: &[CertificateDer<'static>],
    nodes: &[&str],
    buffers: SocketBuffers,
) -> TlsTransport {
    let nodes: Vec<NodeId> = nodes.iter().map(|n| id(n)).collect();
    TlsTransport::new(&ident(g), trust, &nodes, Some("127.0.0.1:0"), buffers).unwrap()
}

fn peer(of: &TlsTransport, name: &str) -> Peer {
    Peer {
        id: id(name),
        addr: of.local_addr().unwrap().to_string(),
    }
}

/// Three nodes with self-signed certificates, each pinning all three.
fn trio() -> (TlsTransport, TlsTransport, TlsTransport) {
    let (a, b, c) = (
        generate_self_signed("a").unwrap(),
        generate_self_signed("b").unwrap(),
        generate_self_signed("c").unwrap(),
    );
    let trust = anchors(&[&a, &b, &c]);
    let n = ["a", "b", "c"];
    (
        node(&a, &trust, &n, SocketBuffers::Auto),
        node(&b, &trust, &n, SocketBuffers::Auto),
        node(&c, &trust, &n, SocketBuffers::Auto),
    )
}

/// `accept` on a thread, so a `connect` on the test thread can complete.
fn accept_in_background(t: &Arc<TlsTransport>) -> mpsc::Receiver<std::io::Result<Conn>> {
    let (tx, rx) = mpsc::channel();
    let t = Arc::clone(t);
    std::thread::spawn(move || {
        let _ = tx.send(t.accept());
    });
    rx
}

fn ping(c: &mut Conn, n: u32, payload: &[u8]) {
    frame::write_frame(&mut c.tx, &serde_json::json!({ "n": n }), payload).unwrap();
}

fn pong(c: &mut Conn) -> (u32, Vec<u8>) {
    let f = frame::read_frame(&mut c.rx).unwrap().unwrap();
    let v: serde_json::Value = frame::parse_header(&f).unwrap();
    (v["n"].as_u64().unwrap() as u32, f.payload)
}

// hardware: none
#[test]
fn frames_round_trip_over_tls_in_both_directions() {
    let (a, b, _c) = trio();
    let a = Arc::new(a);
    let acc = accept_in_background(&a);
    let mut out = b.connect(&peer(&a, "a"), Purpose::Control).unwrap();
    let mut inb = acc.recv_timeout(Duration::from_secs(10)).unwrap().unwrap();
    assert_eq!(inb.peer, id("b"));
    assert_eq!(inb.purpose, Purpose::Control);
    assert_eq!(out.peer, id("a"));
    // A 3 MB payload one way, a small one back, while another thread reads.
    let big: Vec<u8> = (0..3_000_000u32).map(|i| (i % 251) as u8).collect();
    let writer = std::thread::spawn({
        let big = big.clone();
        move || {
            ping(&mut out, 1, &big);
            let r = pong(&mut out);
            (out, r)
        }
    });
    let (n, got) = pong(&mut inb);
    assert_eq!((n, got), (1, big));
    ping(&mut inb, 2, b"ok");
    let (_out, (n, got)) = writer.join().unwrap();
    assert_eq!((n, got.as_slice()), (2, &b"ok"[..]));
    assert_eq!(a.name(), "tls");
    assert_eq!(a.local(), &id("a"));
}

// hardware: none
#[test]
fn one_connection_per_purpose_and_the_acceptor_learns_it() {
    let (a, b, _c) = trio();
    let a = Arc::new(a);
    let purposes = [
        Purpose::Control,
        Purpose::Segment {
            replica: 0,
            segment: 1,
        },
        Purpose::Replica { replica: 2 },
        Purpose::Prefix,
    ];
    let mut seen = Vec::new();
    let mut keep = Vec::new();
    for p in &purposes {
        let acc = accept_in_background(&a);
        keep.push(b.connect(&peer(&a, "a"), p.clone()).unwrap());
        let inb = acc.recv_timeout(Duration::from_secs(10)).unwrap().unwrap();
        seen.push(inb.purpose.clone());
        keep.push(inb);
    }
    assert_eq!(seen, purposes);
}

// hardware: none
#[test]
fn a_client_certificate_no_anchor_vouches_for_is_refused() {
    let (ga, gb) = (
        generate_self_signed("a").unwrap(),
        generate_self_signed("b").unwrap(),
    );
    // `d` is a stranger: a valid certificate of its own that a does not pin. It pins a.
    let gd = generate_self_signed("d").unwrap();
    let a = node(
        &ga,
        &anchors(&[&ga, &gb]),
        &["a", "b", "d"],
        SocketBuffers::Auto,
    );
    let d = node(&gd, &anchors(&[&ga, &gd]), &["a", "d"], SocketBuffers::Auto);
    let before = rejected_total("untrusted");
    let err = d.connect(&peer(&a, "a"), Purpose::Control).err();
    assert!(err.is_some(), "the stranger was let in");
    refused_since("untrusted", before);
}

// hardware: none
#[test]
fn a_server_the_dialer_does_not_pin_is_refused() {
    let (a, g_a, g_b) = {
        let (ga, gb) = (
            generate_self_signed("a").unwrap(),
            generate_self_signed("b").unwrap(),
        );
        let a = node(&ga, &anchors(&[&ga, &gb]), &["a", "b"], SocketBuffers::Auto);
        (a, ga, gb)
    };
    // b pins only itself, not a.
    let b = node(&g_b, &anchors(&[&g_b]), &["a", "b"], SocketBuffers::Auto);
    let _ = g_a;
    let before = rejected_total("untrusted");
    let err = b
        .connect(&peer(&a, "a"), Purpose::Control)
        .err()
        .expect("refused");
    assert!(
        matches!(
            err.kind(),
            ErrorKind::InvalidData | ErrorKind::PermissionDenied | ErrorKind::Other
        ),
        "{err:?}"
    );
    refused_since("untrusted", before);
}

// hardware: none
#[test]
fn a_node_the_cluster_does_not_list_is_refused_even_with_a_trusted_certificate() {
    let (ga, gb, gc) = (
        generate_self_signed("a").unwrap(),
        generate_self_signed("b").unwrap(),
        generate_self_signed("c").unwrap(),
    );
    let trust = anchors(&[&ga, &gb, &gc]);
    // a's cluster lists a and b only; c's certificate is pinned (a CA-like blanket) but unlisted.
    let a = node(&ga, &trust, &["a", "b"], SocketBuffers::Auto);
    let c = node(&gc, &trust, &["a", "b", "c"], SocketBuffers::Auto);
    let before = rejected_total("unknown_node");
    let err = c.connect(&peer(&a, "a"), Purpose::Control);
    assert!(err.is_err(), "unlisted node was let in");
    refused_since("unknown_node", before);
}

// hardware: none
#[test]
fn a_certificate_without_a_node_name_is_refused() {
    let ga = generate_self_signed("a").unwrap();
    let nameless = testing::nameless().unwrap();
    let trust = anchors(&[&ga, &nameless]);
    let a = node(&ga, &trust, &["a"], SocketBuffers::Auto);
    assert!(Identity::from_pem(nameless.cert_pem.as_bytes(), nameless.key_pem.as_bytes()).is_err());
    assert!(a.local_addr().is_some());
    assert!(check_node_id("a b").is_err() && check_node_id("").is_err());
    assert!(check_node_id("node-1.eu_west").is_ok());
}

// hardware: none
#[test]
fn dialing_the_wrong_node_is_refused() {
    let (a, b, _c) = trio();
    let before = rejected_total("wrong_peer");
    let err = b
        .connect(&peer(&a, "c"), Purpose::Control)
        .err()
        .expect("a answered for c");
    assert!(err.to_string().contains("names a"), "{err}");
    refused_since("wrong_peer", before);
}

// hardware: none
#[test]
fn a_ca_certificate_is_an_anchor_like_a_pinned_one() {
    let ca = testing::ca("tetraeder test ca").unwrap();
    let (ga, gb) = (ca.issue("a").unwrap(), ca.issue("b").unwrap());
    let trust = certs_from_pem(ca.cert_pem.as_bytes()).unwrap();
    let a = Arc::new(node(&ga, &trust, &["a", "b"], SocketBuffers::Auto));
    let b = node(&gb, &trust, &["a", "b"], SocketBuffers::Auto);
    let acc = accept_in_background(&a);
    let mut out = b.connect(&peer(&a, "a"), Purpose::Control).unwrap();
    let mut inb = acc.recv_timeout(Duration::from_secs(10)).unwrap().unwrap();
    ping(&mut out, 5, b"x");
    assert_eq!(pong(&mut inb), (5, b"x".to_vec()));
    // A node of another CA is refused.
    let other = testing::ca("other ca").unwrap();
    let gx = other.issue("b").unwrap();
    let x = node(
        &gx,
        &certs_from_pem(other.cert_pem.as_bytes()).unwrap(),
        &["a", "b"],
        SocketBuffers::Auto,
    );
    assert!(x.connect(&peer(&a, "a"), Purpose::Prefix).is_err());
}

// hardware: none
#[test]
fn effective_buffers_are_read_back_for_autotuned_and_explicit_sockets() {
    let (ga, gb) = (
        generate_self_signed("a").unwrap(),
        generate_self_signed("b").unwrap(),
    );
    let trust = anchors(&[&ga, &gb]);
    for setting in [
        SocketBuffers::Auto,
        SocketBuffers::Explicit {
            send: 256 * 1024,
            recv: 256 * 1024,
        },
    ] {
        let a = Arc::new(node(&ga, &trust, &["a", "b"], setting));
        let b = node(&gb, &trust, &["a", "b"], setting);
        let acc = accept_in_background(&a);
        let _out = b.connect(&peer(&a, "a"), Purpose::Control).unwrap();
        let _inb = acc.recv_timeout(Duration::from_secs(10)).unwrap().unwrap();
        for t in [&a, &Arc::new(b)] {
            let log = t.link_buffers();
            assert!(!log.is_empty(), "{setting:?}");
            for l in &log {
                assert!(l.send > 0 && l.recv > 0, "{l:?}");
                assert_eq!(l.requested, setting);
            }
        }
    }
    // The free function works on any std stream.
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let s = std::net::TcpStream::connect(l.local_addr().unwrap()).unwrap();
    let (send, recv) = effective_buffers(&s).unwrap();
    assert!(send > 0 && recv > 0);
}

// hardware: none
#[test]
fn close_from_another_thread_fails_a_blocked_read() {
    let (a, b, _c) = trio();
    let a = Arc::new(a);
    let acc = accept_in_background(&a);
    let out = b.connect(&peer(&a, "a"), Purpose::Control).unwrap();
    let mut inb = acc.recv_timeout(Duration::from_secs(10)).unwrap().unwrap();
    let close = Arc::clone(&inb.close);
    let (tx, rx) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        let mut buf = [0u8; 8];
        let _ = tx.send(inb.rx.read(&mut buf).is_err());
        inb
    });
    std::thread::sleep(Duration::from_millis(200));
    assert!(rx.try_recv().is_err(), "the read returned before the close");
    close();
    assert!(rx.recv_timeout(Duration::from_secs(5)).unwrap());
    let mut inb = reader.join().unwrap();
    assert!(inb.tx.write_all(b"x").is_err());
    drop(out);
}

// hardware: none
#[test]
fn a_peer_that_goes_away_reads_as_end_of_input() {
    let (a, b, _c) = trio();
    let a = Arc::new(a);
    let acc = accept_in_background(&a);
    let out = b.connect(&peer(&a, "a"), Purpose::Control).unwrap();
    let mut inb = acc.recv_timeout(Duration::from_secs(10)).unwrap().unwrap();
    let mut out = out;
    ping(&mut out, 1, b"last");
    drop(out);
    assert_eq!(pong(&mut inb), (1, b"last".to_vec()));
    assert!(frame::read_frame(&mut inb.rx).unwrap().is_none());
}

// hardware: none
#[test]
fn shutdown_ends_accept() {
    let (a, _b, _c) = trio();
    let a = Arc::new(a);
    let acc = accept_in_background(&a);
    std::thread::sleep(Duration::from_millis(100));
    a.shutdown();
    assert!(acc.recv_timeout(Duration::from_secs(5)).unwrap().is_err());
}

// hardware: none
#[test]
fn a_dial_only_node_cannot_accept_and_the_private_key_is_not_in_debug_output() {
    let ga = generate_self_signed("a").unwrap();
    let i = ident(&ga);
    let dbg = format!("{i:?}");
    assert!(dbg.contains("\"a\"") && !dbg.contains("PRIVATE") && !dbg.contains("key"));
    let t = TlsTransport::new(&i, &anchors(&[&ga]), &[id("a")], None, SocketBuffers::Auto).unwrap();
    assert!(t.accept().is_err());
    assert!(!format!("{t:?}").contains("PRIVATE"));
}
