//! Fixes from the security review of steps 1 and 2 (ADR 0091 step 3): a pinned leaf is not a CA,
//! a new connection for a purpose replaces the old one, a peer that vanishes reads as a reset,
//! node ids cannot be path components.
//!
//! Hardware label: none (loopback).

use rustls::pki_types::CertificateDer;
use std::io::{ErrorKind, Read};
use std::sync::Arc;
use std::time::Duration;
use tetraeder_transport::identity::{
    Generated, certs_from_pem, check_node_id, generate_self_signed, testing,
};
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

fn accept_in_background(
    t: &Arc<TlsTransport>,
) -> std::sync::mpsc::Receiver<std::io::Result<Conn>> {
    let (tx, rx) = std::sync::mpsc::channel();
    let t = Arc::clone(t);
    std::thread::spawn(move || {
        let _ = tx.send(t.accept());
    });
    rx
}

// hardware: none
#[test]
fn a_pinned_leaf_cannot_vouch_for_another_node() {
    // c pins a, b and itself as leaves. a's key signs a certificate naming b: refused.
    let (ga, gb, gc) = (
        generate_self_signed("a").unwrap(),
        generate_self_signed("b").unwrap(),
        generate_self_signed("c").unwrap(),
    );
    let forged = testing::forge_signed_by("a", &ga, "b").unwrap();
    let c = Arc::new(node(&gc, &anchors(&[&ga, &gb, &gc]), &["a", "b", "c"]));
    let fake_b = node(&forged, &anchors(&[&gc, &ga]), &["a", "b", "c"]);
    let acc = accept_in_background(&c);
    let err = fake_b.connect(&peer(&c, "c"), Purpose::Control).err();
    assert!(err.is_some(), "the forged certificate was accepted");
    assert!(acc.recv_timeout(Duration::from_millis(500)).is_err());
    // The real b is still accepted (the first background accept is still waiting).
    let real_b = node(&gb, &anchors(&[&gc, &gb]), &["a", "b", "c"]);
    real_b.connect(&peer(&c, "c"), Purpose::Control).unwrap();
    let got = acc.recv_timeout(Duration::from_secs(10)).unwrap().unwrap();
    assert_eq!(got.peer, id("b"));
}

// hardware: none
#[test]
fn a_ca_anchor_still_vouches_for_its_leaves() {
    let ca = testing::ca("test ca").unwrap();
    let (ga, gb) = (ca.issue("a").unwrap(), ca.issue("b").unwrap());
    let trust = certs_from_pem(ca.cert_pem.as_bytes()).unwrap();
    let a = Arc::new(node(&ga, &trust, &["a", "b"]));
    let b = node(&gb, &trust, &["a", "b"]);
    let acc = accept_in_background(&a);
    b.connect(&peer(&a, "a"), Purpose::Control).unwrap();
    let got = acc.recv_timeout(Duration::from_secs(10)).unwrap().unwrap();
    assert_eq!(got.peer, id("b"));
}

// hardware: none
#[test]
fn a_new_connection_for_a_purpose_replaces_the_old_one() {
    let (ga, gb) = (
        generate_self_signed("a").unwrap(),
        generate_self_signed("b").unwrap(),
    );
    let trust = anchors(&[&ga, &gb]);
    let a = Arc::new(node(&ga, &trust, &["a", "b"]));
    let b = node(&gb, &trust, &["a", "b"]);
    let acc = accept_in_background(&a);
    let _first = b.connect(&peer(&a, "a"), Purpose::Control).unwrap();
    let mut first_in = acc.recv_timeout(Duration::from_secs(10)).unwrap().unwrap();
    // The old connection is not known to be dead (a crashed peer, a dropped NAT entry); a new
    // authenticated one for the same purpose takes over and the old one is closed.
    let acc = accept_in_background(&a);
    let _second = b.connect(&peer(&a, "a"), Purpose::Control).unwrap();
    acc.recv_timeout(Duration::from_secs(10)).unwrap().unwrap();
    let mut buf = [0u8; 1];
    let end = first_in.rx.read(&mut buf);
    assert!(matches!(end, Ok(0) | Err(_)), "{end:?}");
}

// hardware: none
#[test]
fn a_peer_that_vanishes_without_close_notify_reads_as_a_reset() {
    let (ga, gb) = (
        generate_self_signed("a").unwrap(),
        generate_self_signed("b").unwrap(),
    );
    let trust = anchors(&[&ga, &gb]);
    let a = Arc::new(node(&ga, &trust, &["a", "b"]));
    let b = node(&gb, &trust, &["a", "b"]);
    let acc = accept_in_background(&a);
    let out = b.connect(&peer(&a, "a"), Purpose::Control).unwrap();
    let mut inb = acc.recv_timeout(Duration::from_secs(10)).unwrap().unwrap();
    (out.close)(); // the socket goes away with no close_notify
    let mut buf = [0u8; 1];
    let e = inb.rx.read(&mut buf).unwrap_err();
    assert_eq!(e.kind(), ErrorKind::ConnectionReset, "{e}");
}

// hardware: none
#[test]
fn node_ids_cannot_be_path_components() {
    for bad in [".", "..", "", "a/b", "a b"] {
        assert!(check_node_id(bad).is_err(), "`{bad}`");
    }
    assert!(check_node_id("a.b-c_1").is_ok());
    assert!(generate_self_signed("..").is_err());
}
