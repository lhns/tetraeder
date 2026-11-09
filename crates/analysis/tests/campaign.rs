//! Phase 3a detector exit test on box results (hardware: none to score; the campaign itself needs a GPU box).
//! Scores every committed `bench/results/*/detector/` directory; skips when there is none.
//! The scorer itself is tested on campaign directories built from the simulator.

use std::path::{Path, PathBuf};

use tetraeder_analysis::detect::FaultClass;
use tetraeder_analysis::score::{campaign_dirs, score_dir};
use tetraeder_analysis::simfault::{SimSpec, simulate};
use tetraeder_profiler::HardwareGraph;

type Run = (
    &'static str,
    &'static [&'static str],
    &'static [&'static str],
);

fn fault(name: &str) -> FaultClass {
    match name {
        "cpu" => FaultClass::CpuCotenant,
        "disk" => FaultClass::DiskSlow,
        "mem" => FaultClass::Swapping,
        _ => FaultClass::VramPressure,
    }
}

/// A campaign directory in the real file format. Each run is (label, injected faults, labelled faults).
fn campaign(tag: &str, runs: &[Run]) -> PathBuf {
    let hw = HardwareGraph::from_toml(include_str!("../fixtures/rtx3090_box.toml")).unwrap();
    let dir = std::env::temp_dir().join(format!("tetraeder-detector-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut manifest = String::new();
    for (n, (label, injected, labelled)) in runs.iter().enumerate() {
        let classes: Vec<_> = injected.iter().map(|f| fault(f)).collect();
        let mut obs = simulate(&hw, &SimSpec::new(n as u64, &classes)).unwrap();
        obs.id = format!("run-{n:02}");
        let json = serde_json::to_string(&obs).unwrap();
        std::fs::write(dir.join(format!("obs-{n:02}.json")), json).unwrap();
        let faults = serde_json::to_string(labelled).unwrap();
        manifest +=
            &format!("{{\"id\":\"run-{n:02}\",\"label\":\"{label}\",\"faults\":{faults}}}\n");
    }
    std::fs::write(dir.join("manifest.jsonl"), manifest).unwrap();
    dir
}

fn good_runs() -> Vec<Run> {
    let mut v: Vec<Run> = vec![("calib", &[], &[])];
    for _ in 0..10 {
        v.push(("none", &[], &[]));
    }
    for _ in 0..4 {
        v.push(("cpu", &["cpu"], &["cpu"]));
        v.push(("disk", &["disk"], &["disk"]));
        v.push(("vram", &["vram"], &["vram"]));
    }
    for _ in 0..3 {
        v.push(("cpu+disk", &["cpu", "disk"], &["cpu", "disk"]));
        v.push(("cpu+vram", &["cpu", "vram"], &["cpu", "vram"]));
    }
    v
}

#[test]
fn scorer_passes_a_clean_campaign() {
    let dir = campaign("pass", &good_runs());
    let s = score_dir(&dir).unwrap();
    assert!(s.pass, "{s:#?}");
    assert_eq!(
        (s.single_top1.runs, s.no_fault.runs, s.pair_top3.runs),
        (12, 10, 6)
    );
    assert_eq!(s.false_positives, 0);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The CPU-runner testbed (scripts/fault_testbed.sh) labels RAM pressure "mem".
#[test]
fn scorer_reads_the_mem_label_of_the_fault_testbed() {
    let mut runs: Vec<Run> = vec![("calib", &[], &[])];
    for _ in 0..4 {
        runs.push(("mem", &["mem"], &["mem"]));
    }
    runs.push(("cpu+mem", &["cpu", "mem"], &["cpu", "mem"]));
    let dir = campaign("mem", &runs);
    let s = score_dir(&dir).unwrap();
    assert_eq!((s.single_top1.runs, s.single_top1.ok), (4, 4), "{s:#?}");
    assert_eq!(s.pair_top3.ok, 1, "{s:#?}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn scorer_fails_mislabelled_singles_and_missed_pairs() {
    // The observations carry a disk fault; the manifest says cpu, so top-1 misses.
    let mut runs = good_runs();
    runs.retain(|r| r.0 != "cpu");
    for _ in 0..4 {
        runs.push(("cpu", &["disk"], &["cpu"]));
    }
    // A pair whose second fault was never injected.
    runs.push(("cpu+disk", &["cpu"], &["cpu", "disk"]));
    let dir = campaign("fail", &runs);
    let s = score_dir(&dir).unwrap();
    assert!(!s.single_pass && !s.pair_pass && s.no_fault_pass, "{s:#?}");
    assert!(!s.pass);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn scorer_fails_on_two_false_positives() {
    let mut runs = good_runs();
    runs.retain(|r| r.0 != "none");
    for _ in 0..8 {
        runs.push(("none", &[], &[]));
    }
    for _ in 0..2 {
        runs.push(("none", &["cpu"], &[]));
    }
    let dir = campaign("fp", &runs);
    let s = score_dir(&dir).unwrap();
    assert_eq!(s.false_positives, 2, "{s:#?}");
    assert!(!s.no_fault_pass && !s.pass);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn scorer_errors_on_a_missing_observation() {
    let dir = campaign("missing", &good_runs());
    std::fs::remove_file(dir.join("obs-05.json")).unwrap();
    assert!(score_dir(&dir).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

/// The exit test: every committed campaign meets the bars. Skips cleanly when none is committed.
#[test]
fn committed_campaigns_meet_the_exit_bars() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/results");
    let dirs = campaign_dirs(&root);
    if dirs.is_empty() {
        eprintln!("skip: no bench/results/*/detector/manifest.jsonl committed yet");
        return;
    }
    for dir in dirs {
        let s = score_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        println!(
            "{}: single {}/{}, false positives {}/{}, pairs {}/{}",
            dir.display(),
            s.single_top1.ok,
            s.single_top1.runs,
            s.false_positives,
            s.no_fault.runs,
            s.pair_top3.ok,
            s.pair_top3.runs
        );
        assert!(s.pass, "{}: {s:#?}", dir.display());
    }
}
