//! The worker cases of `worker.rs` with the worker on a `MemTransport` instead of a child process
//! (ADR 0091 step 1): `RemoteBackend` keeps its threads, deadlines and supervision and only opens
//! its pipes over a transport. The same bodies run (`worker_cases/cases.rs`); a fault that kills
//! the child there closes the connection here, and the next call connects to a fresh `FakeKv`,
//! as a restarted worker would be. Link faults (reset, silent node) come from the switchboard.
//!
//! Hardware label: cpu (the tokenizer comes from the tiny fixture).

#[allow(clippy::all, clippy::pedantic)]
#[path = "../../scheduler/tests/fake/mod.rs"]
mod fake;

mod mode {
    use super::{CTX, FLOOR, fake::FakeKv};
    use std::io::BufReader;
    use std::sync::Arc;
    use std::time::Duration;
    use tetraeder::remote_backend::RemoteConfig;
    use tetraeder::worker_loop::{Fault, Flow, parse_fault, serve_loop};
    use tetraeder_scheduler::BatchBackend;
    use tetraeder_transport::{LinkFault, MemTransport, NodeId, Peer, Switchboard, Transport};

    pub fn id(s: &str) -> NodeId {
        NodeId::new(s)
    }

    /// A worker node on `board` serving a fresh `FakeKv` on every connection, as the daemon
    /// restarts a dead worker. `extra` takes the flags of the child-process worker.
    pub fn serve(board: &Arc<Switchboard>, vocab: usize, parallel: usize, extra: &[&str]) {
        let mut faults: Vec<(Fault, i32)> = Vec::new();
        let mut fail_every = None;
        let mut it = extra.iter();
        while let Some(flag) = it.next() {
            let val = it.next().expect("a value");
            match *flag {
                "--fake-fault" => faults.push(parse_fault(val).unwrap()),
                "--fake-fail-every" => fail_every = Some(val.parse().unwrap()),
                other => panic!("unknown flag {other}"),
            }
        }
        let node = board.node(&id("worker"));
        std::thread::spawn(move || {
            while let Ok(conn) = node.accept() {
                let faults = faults.clone();
                std::thread::spawn(move || {
                    let mut backend =
                        FakeKv::new(vocab, parallel, CTX, CTX).with_fail_every(fail_every);
                    let mut rx = BufReader::new(conn.rx);
                    let mut tx = conn.tx;
                    let flow = serve_loop(
                        &mut backend as &mut dyn BatchBackend,
                        &mut rx,
                        &mut tx,
                        &faults,
                    );
                    // A dead process closes its end; so does a worker that leaves the loop.
                    if !matches!(flow, Ok(Flow::Eof)) {
                        (conn.close)();
                    }
                });
            }
        });
    }

    pub fn config_on(board: &Arc<Switchboard>) -> RemoteConfig {
        let head = Arc::new(board.node(&id("head"))) as Arc<dyn Transport>;
        let peer = Peer {
            id: id("worker"),
            addr: String::new(),
        };
        let mut c = RemoteConfig::node(head, peer, 0, 0);
        c.deadline_floor = FLOOR;
        c.backoff_base = Duration::from_millis(10);
        c.startup_timeout = Duration::from_secs(60);
        c
    }

    pub fn worker_config(vocab: usize, parallel: usize, extra: &[&str]) -> RemoteConfig {
        let board = MemTransport::switchboard();
        serve(&board, vocab, parallel, extra);
        config_on(&board)
    }

    pub fn cut(board: &Arc<Switchboard>, f: LinkFault) {
        board.fault(&id("head"), &id("worker"), f);
        board.fault(&id("worker"), &id("head"), f);
    }
}

include!("worker_cases/cases.rs");

use tetraeder_transport::LinkFault;

// hardware: cpu
#[test]
fn a_node_that_is_not_there_is_a_startup_error() {
    let board = tetraeder_transport::MemTransport::switchboard();
    // The head exists, the worker node does not.
    let e = RemoteBackend::start(mode::config_on(&board))
        .err()
        .expect("startup error");
    assert!(e.contains("cannot connect to node worker"), "{e}");
}

// hardware: cpu
#[test]
fn a_reset_link_fails_the_call_and_the_backend_reconnects_when_it_is_back() {
    let board = tetraeder_transport::MemTransport::switchboard();
    mode::serve(&board, VOCAB, 4, &[]);
    let mut b = RemoteBackend::start(mode::config_on(&board)).unwrap();
    step(&mut b, &[(0, vec![1, 2, 3], false)]).unwrap();
    mode::cut(&board, LinkFault::Close);
    failed_with(step(&mut b, &[(0, vec![4], false)]), "worker decode");
    mode::cut(&board, LinkFault::None);
    // Sequence 0 held rows on the dead connection; it is lost, a free one works.
    assert_recovers(&mut b);
}

// hardware: cpu
#[test]
fn a_silent_node_is_given_up_on_at_the_deadline() {
    let board = tetraeder_transport::MemTransport::switchboard();
    mode::serve(&board, VOCAB, 4, &[]);
    let mut b = RemoteBackend::start(mode::config_on(&board)).unwrap();
    step(&mut b, &[(0, vec![1, 2, 3], false)]).unwrap();
    // A partition: nothing arrives, nothing is closed.
    mode::cut(&board, LinkFault::Blackhole);
    let t0 = Instant::now();
    failed_with(step(&mut b, &[(0, vec![4], false)]), "no reply within");
    assert!(t0.elapsed() >= FLOOR);
    mode::cut(&board, LinkFault::None);
    assert_recovers(&mut b);
}

// hardware: cpu
#[test]
fn a_slow_link_delays_calls_but_keeps_the_answers() {
    let board = tetraeder_transport::MemTransport::switchboard();
    mode::serve(&board, VOCAB, 4, &[]);
    let mut b = RemoteBackend::start(mode::config_on(&board)).unwrap();
    mode::cut(
        &board,
        LinkFault::Slow {
            alpha: Duration::from_millis(30),
            bytes_per_s: 10_000_000.0,
        },
    );
    let t0 = Instant::now();
    let rows = step(&mut b, &[(0, vec![1, 2, 3], false)]).unwrap();
    assert!(
        t0.elapsed() >= Duration::from_millis(60),
        "{:?}",
        t0.elapsed()
    );
    assert_eq!(rows[0], logits_for(VOCAB, &[1, 2, 3]));
    assert_eq!(b.restarts(), 0);
}

// hardware: cpu
#[test]
fn a_bad_request_header_is_an_error_reply_and_the_worker_serves_on() {
    use std::io::Write;
    use tetraeder::remote_backend::wire::{Reply, Request};
    use tetraeder::remote_backend::{encode_raw, parse_header, read_frame, write_frame};
    use tetraeder_transport::{Peer, Purpose, Transport as _};
    let board = tetraeder_transport::MemTransport::switchboard();
    mode::serve(&board, VOCAB, 4, &[]);
    let head = board.node(&mode::id("head"));
    let mut c = head
        .connect(
            &Peer {
                id: mode::id("worker"),
                addr: String::new(),
            },
            Purpose::Segment {
                replica: 0,
                segment: 0,
            },
        )
        .unwrap();
    c.tx.write_all(&encode_raw(b"not json", &[]).unwrap())
        .unwrap();
    c.tx.flush().unwrap();
    let f = read_frame(&mut c.rx).unwrap().expect("a reply");
    let r: Reply = parse_header(&f).unwrap();
    assert!(r.err.is_some(), "the bad header is the call's failure");
    // The same connection still works.
    write_frame(&mut c.tx, &Request::Info, &[]).unwrap();
    let f = read_frame(&mut c.rx).unwrap().expect("a reply");
    let r: Reply = parse_header(&f).unwrap();
    assert!(r.err.is_none() && r.info.is_some());
}
