//! `tetraeder-bench disk-curve` (ADR 0056) on a temp file with buffered reads: one point per
//! depth, positive rates, a queue curve ready for a hardware graph, and provenance.
//!
//! Hardware label: none (a local temp file; the real curve comes from a box stage).

use serde_json::Value;
use tetraeder_bench::disk_curve::{Args, measure, run};

fn args<'a>(file: &'a std::path::Path, depths: &'a [usize], load_threads: usize) -> Args<'a> {
    Args {
        file,
        chunk_bytes: 64 << 10,
        depths,
        seconds: 0.1,
        load_threads,
        buffered: true,
    }
}

#[test]
fn one_point_per_depth_with_positive_rates() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("shard.bin");
    std::fs::write(&file, vec![3u8; 2 << 20]).unwrap();
    let depths = [1usize, 2, 4];
    let doc = measure(&args(&file, &depths, 0)).unwrap();
    let points = doc["points"].as_array().unwrap();
    assert_eq!(points.len(), depths.len());
    for (p, d) in points.iter().zip(depths) {
        assert_eq!(p["depth"], d);
        assert!(p["bytes_per_s"].as_f64().unwrap() > 0.0, "{p}");
        assert!(p["p50_ms"].as_f64().unwrap() <= p["p99_ms"].as_f64().unwrap());
    }
    let curve = doc["queue_curve"].as_array().unwrap();
    assert_eq!(curve.len(), depths.len());
    assert_eq!(doc["queue_chunk_bytes"], 64 << 10);
    assert!(doc["provenance"]["host_id"].is_string());
    assert_eq!(doc["buffered"], true);
    assert_eq!(doc["load_threads"], 0);
}

#[test]
fn a_loaded_run_writes_its_file() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("shard.bin");
    std::fs::write(&file, vec![3u8; 1 << 20]).unwrap();
    let out = dir.path().join("out/curve.json");
    run(&args(&file, &[2], 1), &out).unwrap();
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    assert_eq!(doc["load_threads"], 1);
    assert_eq!(doc["points"].as_array().unwrap().len(), 1);
}

#[test]
fn no_depths_or_a_missing_file_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("nope.bin");
    assert!(measure(&args(&file, &[], 0)).is_err());
    assert!(measure(&args(&file, &[1], 0)).is_err());
}
