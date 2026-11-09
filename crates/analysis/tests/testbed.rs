//! The detector on the first real fault testbed run (hardware: none; the observations are committed
//! in bench/results/20261109-runner/detector-testbed, 4-core GitHub runner, ADR 0041 addendum).
//! A held-out check: the thresholds stay tuned on the simulator.

use std::path::Path;

use tetraeder_analysis::detect::FaultClass;
use tetraeder_analysis::score::score_dir;

#[test]
fn first_testbed_run_singles_and_no_fault() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/results/20261109-runner/detector-testbed");
    let s = score_dir(&dir).unwrap();
    // Before the Swapping fix: 12/18 (mem 0/6).
    assert_eq!((s.single_top1.ok, s.single_top1.runs), (18, 18), "{s:#?}");
    assert_eq!((s.false_positives, s.no_fault.runs), (0, 6), "{s:#?}");
    // cpu+mem pairs now name both. The disk cap of that run never bound in a pair (cpu+disk
    // 1.34 s/token, cpu alone 1.36; disk read at 4 GB/s against a 0.2 GB/s cap), so no disk
    // evidence exists there; the testbed now scales the pair cap with the partner's pace.
    for r in &s.runs {
        match r.label.as_str() {
            "cpu+mem" => assert!(r.ok, "{r:?}"),
            "disk+mem" => assert_eq!(r.found, vec![FaultClass::Swapping], "{r:?}"),
            "cpu+disk" => assert_eq!(r.found, vec![FaultClass::CpuCotenant], "{r:?}"),
            _ => {}
        }
    }
}
