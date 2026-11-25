//! `CheckModel` (ADR 0091, Case S blockers): before anything starts, every node that runs a
//! part of a replica is asked whether it holds the model whole, so a missing or partial copy of
//! a split GGUF fails the head's start with the node named, not a worker that never comes up.
//!
//! Hardware label: none.

mod common;

use common::{config, id};
use std::io;
use std::sync::Arc;
use tetraeder_control::client;
use tetraeder_control::node::{
    Assignment, Daemon, Launcher, ModelStat, NodeOptions, RestartPolicy, WorkerIo,
};
use tetraeder_transport::{MemTransport, Switchboard, Transport};

/// A node that holds the model only when its id is in `have`.
struct Shelf {
    have: Vec<&'static str>,
}

impl Launcher for Shelf {
    fn launch(&self, _: &Assignment) -> io::Result<WorkerIo> {
        Err(io::Error::other("no worker here"))
    }

    fn check_model(&self, sha256: &str) -> io::Result<ModelStat> {
        if self.have.contains(&sha256) {
            Ok(ModelStat {
                shards: 5,
                bytes: 142,
            })
        } else {
            Err(io::Error::new(
                io::ErrorKind::NotFound,
                "model shard 3 of 5 is missing",
            ))
        }
    }
}

fn daemon(board: &Arc<Switchboard>, name: &str, have: Vec<&'static str>) -> Daemon {
    let t: Arc<dyn Transport> = Arc::new(board.node(&id(name)));
    let dial = Arc::clone(&t);
    Daemon::start(
        t,
        NodeOptions {
            cfg: config(1000, &["a", "b"], &[1, 2]),
            launcher: Arc::new(Shelf { have }),
            profile: Arc::new(|| Ok(String::new())),
            dial: Arc::new(move |_| Ok(Arc::clone(&dial))),
            restart: RestartPolicy::default(),
        },
    )
}

const SPLIT: &str = "5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed5eed";

// hardware: none
#[test]
fn the_head_learns_which_node_lacks_the_model_before_anything_starts() {
    let board = MemTransport::switchboard();
    let _a = daemon(&board, "a", vec![SPLIT]);
    let _b = daemon(&board, "b", vec![]);
    let cfg = config(1000, &["a", "b"], &[1, 2]);
    let head = board.node(&id("a"));
    // Node a holds it: fine on its own.
    assert!(
        client::check_model(&head, &cfg, &[id("a")], SPLIT)
            .unwrap()
            .is_empty()
    );
    // Node b does not: the error names the node, the model and the shard.
    let e = client::check_model(&head, &cfg, &[id("a"), id("b")], SPLIT).unwrap_err();
    assert!(
        e.contains("node `b`") && e.contains(SPLIT) && e.contains("shard 3 of 5"),
        "{e}"
    );
}

// hardware: none
#[test]
fn a_node_that_is_not_up_yet_is_reported_not_failed() {
    let board = MemTransport::switchboard();
    let _a = daemon(&board, "a", vec![SPLIT]);
    let cfg = config(1000, &["a", "b"], &[1, 2]);
    let head = board.node(&id("a"));
    // Node b has no daemon: the join loop of the head retries it, the check does not stop the start.
    let unreached = client::check_model(&head, &cfg, &[id("a"), id("b")], SPLIT).unwrap();
    assert_eq!(unreached, vec![id("b")]);
}
