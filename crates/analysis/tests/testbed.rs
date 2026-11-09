//! The detector on the real fault testbed runs (hardware: none; the observations are committed in
//! bench/results/20261109-runner/, 4-core GitHub runners, ADR 0041 addenda). A held-out check:
//! the thresholds stay tuned on the simulator.

use std::path::{Path, PathBuf};

use tetraeder_analysis::detect::FaultClass;
use tetraeder_analysis::score::score_dir;

fn run(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/results/20261109-runner")
        .join(name)
}

#[test]
fn first_testbed_run_singles_and_no_fault() {
    let s = score_dir(&run("detector-testbed")).unwrap();
    // Before the Swapping fix: 12/18 (mem 0/6).
    assert_eq!((s.single_top1.ok, s.single_top1.runs), (18, 18), "{s:#?}");
    assert_eq!((s.false_positives, s.no_fault.runs), (0, 6), "{s:#?}");
    // cpu+mem and disk+mem pairs name the memory fault. The disk cap of that run never bound in
    // a pair (cpu+disk 1.34 s/token, cpu alone 1.36; disk read at 4 GB/s against a 0.2 GB/s cap),
    // so no disk evidence exists there; the testbed now scales the pair cap with the partner's pace.
    for r in &s.runs {
        match r.label.as_str() {
            "cpu+mem" => assert!(r.ok, "{r:?}"),
            "disk+mem" => assert_eq!(r.found, vec![FaultClass::Swapping], "{r:?}"),
            "cpu+disk" => assert_eq!(r.found, vec![FaultClass::CpuCotenant], "{r:?}"),
            _ => {}
        }
    }
}

/// Second run: PSI and major faults sampled, one busy loop and a pace-scaled disk cap in pairs.
#[test]
fn second_testbed_run_meets_the_bars() {
    let s = score_dir(&run("detector-testbed-2")).unwrap();
    // With PSI as a gate (the first fix): single 12/18, pairs 2/6. Memory PSI was only 4% to 5%
    // at a 5x slowdown, and swap-in fell to 17 MB/s under the disk+mem io.max.
    assert_eq!((s.single_top1.ok, s.single_top1.runs), (18, 18), "{s:#?}");
    assert_eq!((s.false_positives, s.no_fault.runs), (0, 6), "{s:#?}");
    assert_eq!((s.pair_top3.ok, s.pair_top3.runs), (6, 6), "{s:#?}");
    assert!(s.pass);
}
