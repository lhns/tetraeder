//! Rule 4: every simulated fixture times every applicable fault, with seeded window noise.

use std::path::Path;

use tetraeder_analysis::detect::{FaultClass, detect};
use tetraeder_analysis::simfault::{SimSpec, applicable_faults, simulate};
use tetraeder_profiler::HardwareGraph;

const SEEDS: u64 = 10;

fn fixtures() -> Vec<(String, HardwareGraph)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut out = Vec::new();
    for dir in [root.join("fixtures"), root.join("../costmodel/fixtures")] {
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.extension().is_some_and(|x| x == "toml") {
                let hw = HardwareGraph::from_toml(&std::fs::read_to_string(&p).unwrap()).unwrap();
                hw.validate().unwrap();
                out.push((p.file_name().unwrap().to_string_lossy().into_owned(), hw));
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    assert!(
        out.len() >= 3,
        "expected the costmodel fixtures and rtx3090_box"
    );
    out
}

fn classes(hw: &HardwareGraph, faults: &[FaultClass], seed: u64) -> Vec<FaultClass> {
    let obs = simulate(hw, &SimSpec::new(seed, faults)).unwrap();
    detect(&obs).faults.iter().map(|f| f.class).collect()
}

#[test]
fn single_faults_are_top_1() {
    for (name, hw) in fixtures() {
        for c in applicable_faults(&hw) {
            let ok = (0..SEEDS)
                .filter(|s| classes(&hw, &[c], *s).first() == Some(&c))
                .count();
            println!("single {name} {c:?}: {ok}/{SEEDS}");
            assert!(ok >= 9, "{name} {c:?}: {ok}/{SEEDS}");
        }
    }
}

#[test]
fn single_faults_are_the_only_finding() {
    for (name, hw) in fixtures() {
        for c in applicable_faults(&hw) {
            for s in 0..SEEDS {
                let got = classes(&hw, &[c], s);
                assert_eq!(got, vec![c], "{name} {c:?} seed {s}");
            }
        }
    }
}

#[test]
fn no_fault_runs_have_at_most_one_false_positive() {
    for (name, hw) in fixtures() {
        let bad = (0..SEEDS)
            .filter(|s| !classes(&hw, &[], *s).is_empty())
            .count();
        println!("no fault {name}: {bad}/{SEEDS} with a fault");
        assert!(bad <= 1, "{name}: {bad}");
    }
}

#[test]
fn fault_pairs_are_both_in_the_top_3() {
    for (name, hw) in fixtures() {
        let all = applicable_faults(&hw);
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                let ok = (0..SEEDS)
                    .filter(|s| {
                        let got = classes(&hw, &[*a, *b], *s);
                        let top: Vec<_> = got.iter().take(3).collect();
                        top.contains(&a) && top.contains(&b)
                    })
                    .count();
                println!("pair {name} {a:?}+{b:?}: {ok}/{SEEDS}");
                assert!(ok >= 9, "{name} {a:?}+{b:?}: {ok}/{SEEDS}");
            }
        }
    }
}

#[test]
fn faults_never_repeat_a_cause() {
    for (name, hw) in fixtures() {
        let all = applicable_faults(&hw);
        for s in 0..SEEDS {
            let obs = simulate(&hw, &SimSpec::new(s, &all)).unwrap();
            let r = detect(&obs);
            let mut res: Vec<_> = r.faults.iter().map(|f| f.resource.clone()).collect();
            let n = res.len();
            res.sort();
            res.dedup();
            // Only vram_pressure may share a resource (the GPU) with another fault.
            assert!(n - res.len() <= 1, "{name} seed {s}: {:?}", r.faults);
        }
    }
}

#[test]
fn detection_is_deterministic() {
    for (_, hw) in fixtures() {
        let all = applicable_faults(&hw);
        let obs = simulate(&hw, &SimSpec::new(4, &all)).unwrap();
        assert_eq!(detect(&obs), detect(&obs));
        assert_eq!(obs, simulate(&hw, &SimSpec::new(4, &all)).unwrap());
    }
}
