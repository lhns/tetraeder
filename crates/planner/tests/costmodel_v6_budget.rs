//! Cost model 6 (ADR 0092), step 7: the planner's RAM budget under the balloon is measured; the
//! 59.9 GB the 64 GB arm really had rejects the pick that OOMed.
//!
//! Hardware label: none.

mod v6_common;

use tetraeder_planner::{Workload, single_node_runs};
use tetraeder_profiler::HardwareGraph;
use v6_common::{layout_of, session};

/// ADR 0092 section 4: the measured RAM budget of the 64 GB arm (59.9 GB) rejects the pick that
/// OOMed and accepts the one that ran.
#[test]
fn a_59_9_gb_budget_rejects_16_56_and_accepts_16_52() {
    let Some(dir) = session() else { return };
    let mut hw = HardwareGraph::from_toml(
        &std::fs::read_to_string(dir.join("predictions/hw-D.toml")).unwrap(),
    )
    .unwrap();
    hw.devices
        .iter_mut()
        .find(|d| d.id == "cpu0")
        .unwrap()
        .capacity_bytes = 59_900_000_000;
    let layout = layout_of(&dir);
    let w = Workload {
        prompt: 2048,
        r#gen: 128,
    };
    let (rejected, why) = single_node_runs(&hw, &layout, &w, Some((16, 56)), None).unwrap();
    assert!(rejected.is_empty(), "{rejected:?}");
    assert!(!why.is_empty());
    let (accepted, _) = single_node_runs(&hw, &layout, &w, Some((16, 52)), None).unwrap();
    assert!(!accepted.is_empty());
}
