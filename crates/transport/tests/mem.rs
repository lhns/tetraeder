//! `MemTransport` (ADR 0091 step 1): frames over in-process pipes and the fault script.
//!
//! Hardware label: none.

use std::io::{ErrorKind, Read, Write};
use std::sync::mpsc;
use std::time::{Duration, Instant};
use tetraeder_transport::{Conn, LinkFault, MemTransport, NodeId, Peer, Purpose, Transport, frame};

fn id(s: &str) -> NodeId {
    NodeId::new(s)
}

fn peer(s: &str) -> Peer {
    Peer {
        id: id(s),
        addr: String::new(),
    }
}

/// `a` connects to `b` and `b` accepts.
fn pair() -> (std::sync::Arc<tetraeder_transport::Switchboard>, Conn, Conn) {
    let board = MemTransport::switchboard();
    let (a, b) = (board.node(&id("a")), board.node(&id("b")));
    let out = a.connect(&peer("b"), Purpose::Control).unwrap();
    let inb = b.accept().unwrap();
    (board, out, inb)
}

fn send(c: &mut Conn, n: u32, payload: &[u8]) {
    frame::write_frame(&mut c.tx, &serde_json::json!({ "n": n }), payload).unwrap();
}

fn recv(c: &mut Conn) -> Option<(u32, Vec<u8>)> {
    let f = frame::read_frame(&mut c.rx).ok()??;
    let v: serde_json::Value = frame::parse_header(&f).unwrap();
    Some((v["n"].as_u64().unwrap() as u32, f.payload))
}

/// Runs `f` on a thread and gives its result, or `None` if it does not return in `wait`.
fn within<T: Send + 'static>(wait: Duration, f: impl FnOnce() -> T + Send + 'static) -> Option<T> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(wait).ok()
}

// hardware: none
#[test]
fn frames_round_trip_both_ways_and_the_acceptor_sees_who_and_why() {
    let (_board, mut out, mut inb) = pair();
    assert_eq!(
        (inb.peer.clone(), inb.purpose.clone()),
        (id("a"), Purpose::Control)
    );
    assert_eq!(out.peer, id("b"));
    send(&mut out, 1, &[7; 100_000]);
    assert_eq!(recv(&mut inb), Some((1, vec![7; 100_000])));
    send(&mut inb, 2, b"re");
    assert_eq!(recv(&mut out), Some((2, b"re".to_vec())));
    let t = MemTransport::switchboard().node(&id("z"));
    assert_eq!((t.name(), t.local()), ("mem", &id("z")));
}

// hardware: none
#[test]
fn dropping_a_half_is_a_clean_end_for_the_peer() {
    let (_board, mut out, mut inb) = pair();
    send(&mut out, 1, b"last");
    drop(out);
    assert_eq!(recv(&mut inb), Some((1, b"last".to_vec())));
    assert_eq!(recv(&mut inb), None);
    assert_eq!(
        inb.tx.write(b"x").unwrap_err().kind(),
        ErrorKind::BrokenPipe
    );
}

// hardware: none
#[test]
fn close_fails_a_blocked_read_at_once() {
    let (_board, out, mut inb) = pair();
    let close = std::sync::Arc::clone(&inb.close);
    let h = std::thread::spawn(move || {
        let mut b = [0u8; 4];
        inb.rx.read(&mut b)
    });
    std::thread::sleep(Duration::from_millis(100));
    close();
    assert_eq!(
        h.join().unwrap().unwrap_err().kind(),
        ErrorKind::ConnectionAborted
    );
    drop(out);
}

// hardware: none
#[test]
fn unknown_nodes_and_duplicate_connections_are_refused() {
    let board = MemTransport::switchboard();
    let (a, b) = (board.node(&id("a")), board.node(&id("b")));
    assert_eq!(
        a.connect(&peer("nobody"), Purpose::Control)
            .err()
            .unwrap()
            .kind(),
        ErrorKind::ConnectionRefused
    );
    let first = a.connect(&peer("b"), Purpose::Prefix).unwrap();
    let inb = b.accept().unwrap();
    assert_eq!(
        a.connect(&peer("b"), Purpose::Prefix).err().unwrap().kind(),
        ErrorKind::AlreadyExists
    );
    // Another purpose, and the other direction, are separate connections.
    let _c = a.connect(&peer("b"), Purpose::Control).unwrap();
    let _d = b.connect(&peer("a"), Purpose::Prefix).unwrap();
    drop(first);
    drop(inb);
    let _again = a.connect(&peer("b"), Purpose::Prefix).unwrap();
}

// hardware: none
#[test]
fn fault_close_resets_open_connections_and_refuses_new_ones_until_lifted() {
    let (board, mut out, mut inb) = pair();
    send(&mut out, 1, b"x");
    board.fault(&id("a"), &id("b"), LinkFault::Close);
    // Both sides see a reset, at once.
    assert_eq!(
        out.tx.write(b"y").unwrap_err().kind(),
        ErrorKind::ConnectionReset
    );
    let mut buf = [0u8; 1];
    assert_eq!(
        inb.rx.read(&mut buf).unwrap_err().kind(),
        ErrorKind::ConnectionReset
    );
    let a = board.node(&id("a"));
    assert_eq!(
        a.connect(&peer("b"), Purpose::Control)
            .err()
            .unwrap()
            .kind(),
        ErrorKind::ConnectionRefused
    );
    // The node comes back: a new connection works, and the old one stays dead.
    board.fault(&id("a"), &id("b"), LinkFault::None);
    let mut out2 = a.connect(&peer("b"), Purpose::Control).unwrap();
    let mut in2 = board.node(&id("b")).accept().unwrap();
    send(&mut out2, 3, b"again");
    assert_eq!(recv(&mut in2), Some((3, b"again".to_vec())));
}

// hardware: none
#[test]
fn a_blocked_read_is_reset_by_a_close_fault() {
    let (board, _out, mut inb) = pair();
    let h = std::thread::spawn(move || {
        let mut b = [0u8; 1];
        inb.rx.read(&mut b)
    });
    std::thread::sleep(Duration::from_millis(100));
    board.fault(&id("b"), &id("a"), LinkFault::Close);
    assert_eq!(
        h.join().unwrap().unwrap_err().kind(),
        ErrorKind::ConnectionReset
    );
}

// hardware: none
#[test]
fn blackhole_passes_nothing_and_closes_nothing_in_one_direction() {
    let (board, mut out, mut inb) = pair();
    board.fault(&id("a"), &id("b"), LinkFault::Blackhole);
    send(&mut out, 1, b"lost");
    // b hears nothing, but is not told the link is down.
    let r = within(Duration::from_millis(300), move || {
        recv(&mut inb).map(|x| x.0)
    });
    assert!(r.is_none(), "a blackholed link delivered {r:?}");
    // The other direction still works.
    let (board, mut out, mut inb) = pair();
    board.fault(&id("a"), &id("b"), LinkFault::Blackhole);
    send(&mut inb, 2, b"back");
    assert_eq!(recv(&mut out), Some((2, b"back".to_vec())));
    // A partition is a blackhole both ways; lifting it lets new bytes through (the lost stay lost).
    board.fault(&id("b"), &id("a"), LinkFault::Blackhole);
    send(&mut inb, 3, b"lost too");
    board.fault(&id("a"), &id("b"), LinkFault::None);
    board.fault(&id("b"), &id("a"), LinkFault::None);
    send(&mut out, 4, b"healed");
    assert_eq!(recv(&mut inb), Some((4, b"healed".to_vec())));
}

// hardware: none
#[test]
fn slow_links_delay_each_write_in_order() {
    let (board, mut out, mut inb) = pair();
    board.fault(
        &id("a"),
        &id("b"),
        LinkFault::Slow {
            alpha: Duration::from_millis(60),
            bytes_per_s: 1_000_000.0,
        },
    );
    let t0 = Instant::now();
    send(&mut out, 1, &[0; 50_000]);
    send(&mut out, 2, b"s");
    assert_eq!(recv(&mut inb).map(|x| x.0), Some(1));
    let first = t0.elapsed();
    assert_eq!(recv(&mut inb).map(|x| x.0), Some(2));
    // alpha + 50 KB at 1 MB/s is about 110 ms for the first frame; the second follows it.
    assert!(first >= Duration::from_millis(100), "{first:?}");
    assert!(t0.elapsed() >= first);
    // The reverse direction is not slowed.
    let t1 = Instant::now();
    send(&mut inb, 3, b"fast");
    assert_eq!(recv(&mut out).map(|x| x.0), Some(3));
    assert!(t1.elapsed() < Duration::from_millis(50));
}

// hardware: none
#[test]
fn shutdown_ends_accept() {
    let board = MemTransport::switchboard();
    let b = std::sync::Arc::new(board.node(&id("b")));
    let b2 = std::sync::Arc::clone(&b);
    let h = std::thread::spawn(move || b2.accept().err().map(|e| e.kind()));
    std::thread::sleep(Duration::from_millis(100));
    b.shutdown();
    assert_eq!(h.join().unwrap(), Some(ErrorKind::NotConnected));
}

// hardware: none
#[test]
fn bytes_written_are_counted_per_sender() {
    // The counter itself is an OpenTelemetry instrument (no exporter here); this checks that
    // writing through the counting wrapper keeps the byte stream intact.
    let (_board, mut out, mut inb) = pair();
    out.tx.write_all(&[1, 2, 3]).unwrap();
    let mut b = [0u8; 3];
    inb.rx.read_exact(&mut b).unwrap();
    assert_eq!(b, [1, 2, 3]);
}
