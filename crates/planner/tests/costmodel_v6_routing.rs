//! Cost model 6 (ADR 0092), step 4: every builder prediction says where its routing came from
//! and a judged one refuses anything but a trace.
//!
//! Hardware label: none.

mod v6_common;

use std::path::Path;
use tetraeder_planner::RoutingSource;
use v6_common::{json, session};

#[test]
fn only_a_trace_is_judged() {
    assert!(RoutingSource::Trace.check_judged().is_ok());
    for r in [RoutingSource::Uniform, RoutingSource::Measured] {
        assert!(r.check_judged().is_err(), "{}", r.name());
    }
    assert_eq!(RoutingSource::Trace.name(), "trace");
}

// ---- routing provenance of node-predict

fn node_predict(extra: &[&str], out: &Path) -> std::process::Output {
    let dir = session().expect("bench results");
    let p = dir.join("predictions");
    std::process::Command::new(env!("CARGO_BIN_EXE_tetraeder-plan"))
        .arg("node-predict")
        .args(["--hw", p.join("d2/hw-D2.toml").to_str().unwrap()])
        .args(["--layout", p.join("layout.json").to_str().unwrap()])
        .args(["--pin", "16,52"])
        .args(extra)
        .args(["--out", out.to_str().unwrap()])
        .output()
        .unwrap()
}

#[test]
fn a_judged_node_prediction_needs_a_trace() {
    let Some(dir) = session() else { return };
    let tmp = std::env::temp_dir().join(format!("tetraeder-judged-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let out = tmp.join("pred.json");
    let no_trace = node_predict(&["--judged"], &out);
    assert!(!no_trace.status.success());
    assert!(String::from_utf8_lossy(&no_trace.stderr).contains("routing trace"));
    assert!(!out.exists());

    let replay = dir.join("predictions/d2/replay-D2.json");
    let judged = node_predict(
        &[
            "--judged",
            "--tier-replay",
            replay.to_str().unwrap(),
            "--trace-sha256",
            "abc123",
        ],
        &out,
    );
    assert!(
        judged.status.success(),
        "{}",
        String::from_utf8_lossy(&judged.stderr)
    );
    let doc = json(&out);
    assert_eq!(doc["routing"], "trace");
    assert_eq!(doc["routing_trace"]["sha256"], "abc123");
    assert_eq!(doc["pick"]["routing"], "trace");
    // without a trace and without --judged: marked uniform
    let uniform = node_predict(&[], &out);
    assert!(uniform.status.success());
    assert_eq!(json(&out)["routing"], "uniform");
    std::fs::remove_dir_all(&tmp).ok();
}
