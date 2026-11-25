//! A bulk probe frame on a slow link (ADR 0091, rehearsal 2): `cluster probe` sends frames up to
//! 64 MiB, and on a link where 32 MiB took 15 s the receiving daemon's 30 s rule for a probe
//! connection closed it before the frame had arrived (the clock ran between whole frames). The
//! limit is now 300 s and every byte received restarts it. Scaled down here (0.3 s against 20 s)
//! and with the `MemTransport` slow-link fault, which delivers a frame whole after its time, so
//! a frame takes five times the old limit.
//!
//! Hardware label: none.

mod common;

use common::{EchoLauncher, config, id};
use std::io::Write as _;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tetraeder_control::client;
use tetraeder_control::node::{Daemon, NodeOptions, RestartPolicy};
use tetraeder_control::proto::{NodeReply, NodeRequest};
use tetraeder_transport::{LinkFault, MemTransport, Purpose, Switchboard, Transport};

fn daemon(board: &Arc<Switchboard>, name: &str, idle: Duration) -> Daemon {
    let t: Arc<dyn Transport> = Arc::new(board.node(&id(name)));
    let dial = Arc::clone(&t);
    let d = Daemon::start(
        t,
        NodeOptions {
            cfg: config(1000, &["a", "b"], &[1, 2]),
            launcher: Arc::new(EchoLauncher::default()),
            profile: Arc::new(|| Ok(String::new())),
            dial: Arc::new(move |_| Ok(Arc::clone(&dial))),
            restart: RestartPolicy::default(),
        },
    );
    d.set_probe_idle(idle);
    d
}

const OLD: Duration = Duration::from_millis(300);
const NEW: Duration = Duration::from_secs(20);

fn probe_b_from_a(board: &Arc<Switchboard>, bytes: u64) -> Result<Duration, String> {
    let cfg = config(1000, &["a", "b"], &[1, 2]);
    let head = board.node(&id("a"));
    let t0 = Instant::now();
    let (r, _) = client::request(
        &head,
        &cfg.peer(&id("a")).unwrap(),
        &NodeRequest::ProbePeer {
            peer: id("b"),
            sizes: vec![bytes],
            repeats: 2,
            buffers: None,
        },
        &[],
    )?;
    assert!(matches!(r, NodeReply::Curve { .. }), "{r:?}");
    Ok(t0.elapsed())
}

/// A link that carries 1 MiB in 1.5 s, five times the old limit, one way (a to b).
fn slow_link(board: &Arc<Switchboard>) {
    board.fault(
        &id("a"),
        &id("b"),
        LinkFault::Slow {
            alpha: Duration::from_millis(5),
            bytes_per_s: (1 << 20) as f64 / 1.5,
        },
    );
}

// hardware: none
#[test]
fn a_bulk_probe_frame_on_a_slow_link_arrives_within_the_probe_limit() {
    let board = MemTransport::switchboard();
    let _a = daemon(&board, "a", NEW);
    let _b = daemon(&board, "b", NEW);
    slow_link(&board);
    let took = probe_b_from_a(&board, 1 << 20).expect("the probe completes");
    // Two repeats of a frame that takes 1.5 s each.
    assert!(took >= Duration::from_secs(3), "{took:?}");
}

// hardware: none
#[test]
fn the_old_limit_cuts_that_same_probe() {
    let board = MemTransport::switchboard();
    // With the limit as short as the frame takes to arrive this is the old 30 s rule against
    // rehearsal 2's link: the connection is closed before the frame has arrived.
    let _a = daemon(&board, "a", OLD);
    let _b = daemon(&board, "b", OLD);
    slow_link(&board);
    let e = probe_b_from_a(&board, 1 << 20).unwrap_err();
    assert!(e.contains("probe of b failed"), "{e}");
}

// hardware: none
#[test]
fn a_silent_probe_connection_is_still_closed_after_the_limit() {
    let board = MemTransport::switchboard();
    let _a = daemon(&board, "a", OLD);
    let _b = daemon(&board, "b", OLD);
    let cfg = config(1000, &["a", "b"], &[1, 2]);
    let ta = board.node(&id("a"));
    let mut c = ta
        .connect(&cfg.peer(&id("b")).unwrap(), Purpose::Probe { session: 1 })
        .unwrap();
    let t0 = Instant::now();
    std::thread::sleep(OLD * 4);
    // Closed by now: a write or the read that follows fails or ends.
    let _ = c.tx.write_all(&[0]);
    let mut b = [0u8; 1];
    let n = std::io::Read::read(&mut c.rx, &mut b).unwrap_or(0);
    assert_eq!(n, 0, "the connection ended");
    assert!(t0.elapsed() < Duration::from_secs(10));
}
