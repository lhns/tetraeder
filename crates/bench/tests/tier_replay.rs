//! The replay output the planner reads (ADR 0056): the fields `planner::score::TierOutcome`
//! expects are present, the trace is fingerprinted, and every committed `tier-replay.json` is
//! current against a fresh replay of the trace it names.
//!
//! Hardware label: none (pure host code).

use serde_json::Value;
use std::path::Path;
use tetraeder_bench::replay;
use tetraeder_memory::replay::REPLAY_VERSION;
use tetraeder_memory::replay::sim::{Policy, VramPolicy};
use tetraeder_memory::replay::trace::{TraceHeader, TraceWriter};

/// What `planner::score::TierOutcome` reads from one outcome. No dependency on the planner:
/// this sample is the contract, and its test lives in the planner too.
const PLANNER_SAMPLE: &str = r#"{
  "vram_slots": 2, "ram_slots": 4, "policy": "lru", "prefetch": false,
  "decode": {"vram_share": 0.2, "full_hit_share": 0.1, "sync_miss_bytes_per_token": 3.0,
             "prefetch_bytes_per_token": 4.0, "disk_bytes_per_token": 7.0,
             "ram_hit_rate_engine_mean": 0.5},
  "prefill": {"vram_share": 0.1, "full_hit_share": 0.0, "sync_miss_bytes_per_token": 3.0,
              "prefetch_bytes_per_token": 4.0, "disk_bytes_per_token": 7.0,
              "ram_hit_rate_engine_mean": 0.5}
}"#;

fn write_trace(path: &Path) {
    let mut h = TraceHeader::new(2, 8, 2);
    h.expert_bytes = vec![1000; 2];
    let mut w = TraceWriter::create(path, &h).unwrap();
    // a 4-token prefill, a hot-set refresh, then 12 decode steps over two layers
    for l in 0..2u16 {
        w.layer_step(0, l, 4, &[0, 1, 2, 3, 4, 5, 6, 7], None, None)
            .unwrap();
    }
    w.hot_refresh(0).unwrap();
    for s in 1..=12u32 {
        for l in 0..2u16 {
            let e = ((s + u32::from(l)) % 8) as i32;
            w.layer_step(s, l, 1, &[e, (e + 3) % 8], None, None)
                .unwrap();
        }
    }
    w.finish().unwrap();
}

#[test]
fn replay_output_has_the_fields_the_planner_reads() {
    let dir = tempfile::tempdir().unwrap();
    let (trace, out) = (
        dir.path().join("t.trace"),
        dir.path().join("tier-replay.json"),
    );
    write_trace(&trace);
    let n = replay::run(
        &trace,
        &[0, 2],
        &[2, 4],
        &[Policy::Lru],
        &[VramPolicy::Static],
        &out,
    )
    .unwrap();
    assert_eq!(n, 4);
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    assert_eq!(doc["replay_version"], REPLAY_VERSION);
    let bytes = std::fs::read(&trace).unwrap();
    assert_eq!(
        doc["trace"]["fnv1a64"].as_str().unwrap(),
        format!("{:016x}", replay::fnv1a64(&bytes))
    );
    assert_eq!(doc["trace"]["file"], "t.trace");
    let want: Value = serde_json::from_str(PLANNER_SAMPLE).unwrap();
    for o in doc["outcomes"].as_array().unwrap() {
        for k in ["vram_slots", "ram_slots", "policy", "prefetch"] {
            assert_eq!(
                std::mem::discriminant(&o[k]),
                std::mem::discriminant(&want[k]),
                "{k}"
            );
        }
        for phase in ["decode", "prefill"] {
            for (k, v) in want[phase].as_object().unwrap() {
                assert!(o[phase][k].is_number(), "{phase}.{k} in {o}");
                assert!(v.is_number());
            }
            let share = o[phase]["full_hit_share"].as_f64().unwrap();
            assert!((0.0..=1.0).contains(&share));
        }
        // a full hit needs every pair in VRAM, so it cannot exceed the pair share
        assert!(
            o["decode"]["full_hit_share"].as_f64().unwrap()
                <= o["decode"]["vram_share"].as_f64().unwrap() + 1e-12
        );
    }
    // with no VRAM slots nothing is a full hit; the file is deterministic
    let first = &doc["outcomes"][0];
    assert_eq!(first["vram_slots"], 0);
    assert_eq!(first["decode"]["full_hit_share"], 0.0);
    let again = dir.path().join("again.json");
    replay::run(
        &trace,
        &[0, 2],
        &[2, 4],
        &[Policy::Lru],
        &[VramPolicy::Static],
        &again,
    )
    .unwrap();
    let doc2: Value = serde_json::from_str(&std::fs::read_to_string(&again).unwrap()).unwrap();
    assert_eq!(doc["outcomes"], doc2["outcomes"]);
}

/// `replay --timed`: a timing per outcome beside the byte outcomes, every read landed, and the
/// byte outcomes are the same as without a time model.
#[test]
fn timed_replay_writes_a_timing_for_every_outcome() {
    let dir = tempfile::tempdir().unwrap();
    let (trace, model, plain, timed) = (
        dir.path().join("t.trace"),
        dir.path().join("time-model.json"),
        dir.path().join("plain.json"),
        dir.path().join("timed.json"),
    );
    write_trace(&trace);
    std::fs::write(
        &model,
        include_str!("../../memory/tests/data/time-model.json"),
    )
    .unwrap();
    let grid = (&[0, 2][..], &[2, 4][..], &[Policy::Lru][..]);
    let st = &[VramPolicy::Static][..];
    replay::run(&trace, grid.0, grid.1, grid.2, st, &plain).unwrap();
    replay::run_timed(&trace, grid.0, grid.1, grid.2, st, &timed, Some(&model)).unwrap();
    let read =
        |p: &Path| -> Value { serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap() };
    let (a, b) = (read(&plain), read(&timed));
    assert_eq!(a["outcomes"], b["outcomes"]);
    assert_eq!(b["time_model"]["path"], "service_overlap");
    let t = b["timed"].as_array().unwrap();
    assert_eq!(t.len(), b["outcomes"].as_array().unwrap().len());
    for (o, e) in b["outcomes"].as_array().unwrap().iter().zip(t) {
        assert_eq!(e["vram_slots"], o["vram_slots"]);
        assert_eq!(e["ram_slots"], o["ram_slots"]);
        let tm = &e["timing"];
        assert_eq!(tm["units_submitted"], tm["units_landed"]);
        assert_eq!(tm["decode"]["tokens"], 12);
        assert!(tm["decode"]["step_s_mean"].as_f64().unwrap() > 0.0);
    }
    assert!(a.get("timed").is_none());
}

/// Every committed `tier-replay.json` whose directory also holds the trace it names replays to
/// exactly the committed outcomes. Skips cleanly when none is committed.
#[test]
fn committed_tier_replays_are_current() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/results");
    let mut checked = 0;
    for day in std::fs::read_dir(&root).into_iter().flatten().flatten() {
        for d in std::fs::read_dir(day.path())
            .into_iter()
            .flatten()
            .flatten()
        {
            let file = d.path().join("tier-replay.json");
            let Ok(text) = std::fs::read_to_string(&file) else {
                continue;
            };
            let doc: Value = serde_json::from_str(&text).unwrap();
            let Some(name) = doc["trace"]["file"].as_str() else {
                continue;
            };
            let trace = d.path().join(name);
            if !trace.is_file() {
                eprintln!("skip {}: trace {name} not committed", file.display());
                continue;
            }
            let list = |k: &str| -> Vec<u32> {
                doc["grid"][k]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u32)
                    .collect()
            };
            let policies: Vec<Policy> = doc["grid"]["policies"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| Policy::parse(p.as_str().unwrap()).unwrap())
                .collect();
            let tmp = tempfile::tempdir().unwrap();
            let out = tmp.path().join("fresh.json");
            replay::run(
                &trace,
                &list("vram_slots"),
                &list("ram_slots"),
                &policies,
                &[VramPolicy::Static],
                &out,
            )
            .unwrap();
            let fresh: Value =
                serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
            assert_eq!(
                doc["trace"]["fnv1a64"],
                fresh["trace"]["fnv1a64"],
                "{}: the trace changed",
                file.display()
            );
            assert_eq!(
                doc["replay_version"],
                fresh["replay_version"],
                "{}: replay_version {} committed, {} now; replay again",
                file.display(),
                doc["replay_version"],
                fresh["replay_version"]
            );
            assert_eq!(
                doc["outcomes"],
                fresh["outcomes"],
                "{}: outcomes differ from a fresh replay",
                file.display()
            );
            checked += 1;
        }
    }
    if checked == 0 {
        eprintln!("skip: no tier-replay.json with its trace committed");
    }
}

/// `replay --decode-window` (ADR 0050 (c) addendum 2026-11-14): windows of the decode bytes on
/// every outcome, global policies included; without the flag the outcomes are as before.
#[test]
fn decode_windows_split_the_decode_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let (trace, plain, windowed) = (
        dir.path().join("t.trace"),
        dir.path().join("plain.json"),
        dir.path().join("windowed.json"),
    );
    write_trace(&trace);
    let policies = [
        Policy::Lru,
        Policy::GlobalLru,
        Policy::GlobalFreq { share: 0.5 },
        Policy::GlobalBelady,
    ];
    let st = &[VramPolicy::Static][..];
    replay::run(&trace, &[0, 2], &[2], &policies, st, &plain).unwrap();
    let extras = replay::Extras {
        decode_window: Some(5),
        ..replay::Extras::default()
    };
    replay::run_with(&trace, &[0, 2], &[2], &policies, st, &windowed, &extras).unwrap();
    let read =
        |p: &Path| -> Value { serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap() };
    let (a, b) = (read(&plain), read(&windowed));
    for (x, y) in a["outcomes"]
        .as_array()
        .unwrap()
        .iter()
        .zip(b["outcomes"].as_array().unwrap())
    {
        assert!(x.get("decode_windows").is_none());
        let w: Vec<f64> = y["decode_windows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        // 12 decode tokens in windows of 5, 5 and 2
        assert_eq!(w.len(), 3, "{y}");
        let total = (w[0] * 5.0 + w[1] * 5.0 + w[2] * 2.0) / 12.0;
        let want = y["decode"]["disk_bytes_per_token"].as_f64().unwrap();
        assert!((total - want).abs() < 1e-9, "{total} {want}");
        let mut y = y.clone();
        y.as_object_mut().unwrap().remove("decode_windows");
        assert_eq!(x, &y);
    }
    assert_eq!(replay::decode_windows(&[4, 2, 6], 2), vec![3.0, 6.0]);
}
