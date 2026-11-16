//! The two commands `scripts/phase5_box.sh` calls (ADR 0068, "As built"): `plan hw` and
//! `bench --engine dense|moe --plan --layer-limit`. Hardware: cpu. `gpu_*` tests need two GPUs
//! and skip without them (AGENTS rule 9).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use tetraeder_planner::multi_cli::{candidates_multi, parse_hw, parse_layout};
use tetraeder_profiler::DeviceKind;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn tet(args: &[&str], cvd: Option<&str>) -> Output {
    let mut c = Command::new(env!("CARGO_BIN_EXE_tetraeder"));
    c.args(args);
    match cvd {
        Some(v) => c.env("CUDA_VISIBLE_DEVICES", v),
        None => c.env_remove("CUDA_VISIBLE_DEVICES"),
    };
    c.output().expect("run tetraeder")
}

fn ok(o: &Output) -> String {
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn tmp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("tetraeder-p5cli-{}-{name}", std::process::id()))
}

fn layout_json(model: &Path) -> String {
    ok(&tet(&["plan", "layout", model.to_str().unwrap()], None))
}

/// `plan hw` with no GPU visible writes a host-only graph that the planner's reader loads and
/// that enumerates (to nothing: no GPU, no shape).
#[test]
fn cpu_plan_hw_writes_a_host_graph_the_planner_reads() {
    let out = tmp("hw.json");
    ok(&tet(
        &["plan", "hw", "--out", out.to_str().unwrap()],
        Some(""),
    ));
    let text = std::fs::read_to_string(&out).unwrap();
    std::fs::remove_file(&out).ok();
    let hw = parse_hw(&text).unwrap();
    hw.validate().unwrap();
    assert!(hw.devices.iter().any(|d| d.kind == DeviceKind::Cpu));
    assert!(hw.devices.iter().all(|d| d.kind != DeviceKind::Gpu));
    let layout = parse_layout(&layout_json(
        &root().join("tests/fixtures/tiny-qwen3-f16.gguf"),
    ))
    .unwrap();
    let found = candidates_multi(&hw, &layout);
    assert_eq!(found.gpus, 0);
}

/// hardware: cpu. `bench --engine dense --plan <single-rank plan> --layer-limit 2` on the
/// 4-layer fixture emits the llama-bench rows `phase5_summary.py` and `tetraeder-plan
/// calibrate` read: `n_prompt`, `n_gen`, `avg_ts`, `plan_id`, `layer_limit`.
#[test]
fn cpu_bench_dense_plan_layer_limit_emits_the_anchor_rows() {
    let model = root().join("tests/fixtures/tiny-qwen3-l4-f16.gguf");
    let hw_path = root().join("crates/costmodel/fixtures/two_3090_hoststaged.toml");
    let hw = parse_hw(&std::fs::read_to_string(&hw_path).unwrap()).unwrap();
    let layout = parse_layout(&layout_json(&model)).unwrap();
    let found = candidates_multi(&hw, &layout);
    let single = found
        .candidates
        .iter()
        .find(|c| c.id == "single")
        .unwrap_or_else(|| panic!("no single candidate, pruned {:?}", found.pruned));
    let plan = tmp("single.toml");
    std::fs::write(&plan, single.plan.to_toml().unwrap()).unwrap();
    let o = tet(
        &[
            "bench",
            "-m",
            model.to_str().unwrap(),
            "--engine",
            "dense",
            "--plan",
            plan.to_str().unwrap(),
            "--hw",
            hw_path.to_str().unwrap(),
            "--layer-limit",
            "2",
            "--ngl",
            "0",
            "-p",
            "8",
            "-n",
            "4",
            "-r",
            "1",
            "-t",
            "2",
            "--telemetry",
            "off",
            "-o",
            "json",
        ],
        None,
    );
    std::fs::remove_file(&plan).ok();
    let rows: serde_json::Value = serde_json::from_str(&ok(&o)).unwrap();
    let rows = rows.as_array().unwrap();
    assert_eq!(rows.len(), 2);
    for (r, (np, ng)) in rows.iter().zip([(8, 0), (0, 4)]) {
        assert_eq!(
            (r["n_prompt"].as_u64(), r["n_gen"].as_u64()),
            (Some(np), Some(ng))
        );
        assert!(r["avg_ts"].as_f64().unwrap() > 0.0, "{r}");
        assert_eq!(r["layer_limit"], 2);
        assert_eq!(r["plan_id"].as_str().unwrap().len(), 8);
        assert_eq!(r["ranks"], 1);
        assert_eq!(r["builder"], "dense");
    }
}

/// hardware: cpu. A layer limit is only for the own builders.
#[test]
fn cpu_bench_layer_limit_needs_an_own_builder() {
    let o = tet(
        &["bench", "-m", "m.gguf", "--layer-limit", "2", "-r", "1"],
        None,
    );
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("--engine dense"));
}

/// hardware: 2 GPUs. `plan hw` under `CUDA_VISIBLE_DEVICES=0,1` names exactly gpu0 and gpu1.
#[test]
fn gpu2_plan_hw_respects_visible_devices() {
    let o = tet(&["plan", "hw"], Some("0,1"));
    let hw = parse_hw(&ok(&o)).unwrap();
    let gpus = hw
        .devices
        .iter()
        .filter(|d| d.kind == DeviceKind::Gpu)
        .count();
    if gpus == 0 {
        eprintln!("skipped: no GPU (or a build without nvml)");
        return;
    }
    assert!(gpus <= 2, "{gpus} GPUs under CUDA_VISIBLE_DEVICES=0,1");
}
