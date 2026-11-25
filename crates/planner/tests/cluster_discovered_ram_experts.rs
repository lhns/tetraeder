//! Case S needs experts in RAM computed on the CPU (ADR 0091). A discovered CPU has no compute
//! spec, and the rehearsal fix made `choose` skip every placement the graph cannot price, so a
//! graph as `tetraeder cluster profile` produces it (two 3090 nodes, the CPU rate missing) had
//! no spanning candidate with RAM experts for the 235B. Discovery now fills a per-core CPU prior
//! (`fill_from_priors`), and names the disk that holds the models dir `nvme0` (`add_disks`), which
//! the planner reads the builder's experts from. This test builds each node's graph from the
//! discovery code (recorded volumes, no edit after the merge), runs the planner on it with 64 GB
//! RAM budgets per node and the 235B layout, and checks the pick against the confirmed shape of
//! ADR 0092 (ubatch 2048, experts in VRAM and RAM on both nodes, and on each node's `nvme0`).
//! The test merges the node graphs itself and skips the wire's `check_profile`; the volumes and
//! the node's CPU and GPUs are the rehearsal's recorded `a/hw.json`.
//!
//! Hardware label: none (the graph is a copy of the rehearsal's `a/hw.json`).

use std::path::PathBuf;
use tetraeder_plan::{Hints, Phase, Plan};
use tetraeder_planner::layout::fixtures;
use tetraeder_planner::{ClusterCandidate, Priors, enumerate_cluster, fill_from_priors};
use tetraeder_profiler::{DeviceKind, HardwareGraph, LinkKind, Volume, add_disks};

const HW: &str = include_str!("../fixtures/phase6-rehearsal/hw.json");
/// The Case S RAM budget per node (`EMULATE_RAM_GB=64`).
const RAM: u64 = 64 << 30;

/// The file system root of the machine running the test, so the mount points below are
/// absolute on Windows and Linux alike.
fn root() -> PathBuf {
    std::env::current_dir()
        .unwrap()
        .ancestors()
        .last()
        .unwrap()
        .to_path_buf()
}

/// The volumes the rehearsal's nodes listed (container overlay, two bind mounts), as recorded in
/// `hw.json` (the mount points are stand-ins that are absolute on both platforms); the models
/// live on the 80 GiB overlay, as on the rehearsal's boxes.
fn recorded_volumes() -> Vec<Volume> {
    let v = |name: &str, mount: PathBuf, gib: u64, media: &str| Volume {
        name: name.into(),
        mount,
        total_bytes: gib << 30,
        media: media.into(),
    };
    vec![
        v("overlay", root(), 80, "unknown"),
        v("/dev/loop8", root().join("mnt").join("loop8"), 773, "ssd"),
        v(
            "/dev/mapper/ubuntu--vg-ubuntu--lv",
            root().join("mnt").join("lv"),
            914,
            "unknown",
        ),
    ]
}

/// One node's graph as the daemon builds it for `NodeRequest::Profile`: the rehearsal node's CPU
/// and GPUs (a recorded probe, with the 64 GB budget the box reports), its disks from the
/// discovery code with the models dir, the priors filled, and the JSON round trip of the wire.
fn node_graph(all: &HardwareGraph, node: &str, models: bool) -> HardwareGraph {
    let mut g = all.node_graph(Some(node));
    g.devices.retain(|d| d.kind != DeviceKind::Disk);
    g.links.retain(|l| l.kind != LinkKind::Storage);
    for d in g.devices.iter_mut().filter(|d| d.kind == DeviceKind::Cpu) {
        d.capacity_bytes = RAM;
        // The fixture was written before discovery filled the CPU rate.
        d.compute = None;
    }
    add_disks(
        &mut g,
        &recorded_volumes(),
        models.then(|| root().join("hf")).as_deref(),
        &root().join("tmp"),
        None,
    );
    fill_from_priors(&mut g, &Priors::builtin());
    HardwareGraph::from_json(&g.to_json().unwrap()).unwrap()
}

/// The graph `tetraeder cluster profile` writes for two nodes: each node's graph, merged as
/// `client::profile_cluster` does, no edit after that.
fn discovered(models: bool) -> HardwareGraph {
    let all = tetraeder_planner::multi_cli::parse_hw(HW).unwrap();
    let nodes: Vec<(String, HardwareGraph)> = ["a", "b"]
        .iter()
        .map(|n| (n.to_string(), node_graph(&all, n, models)))
        .collect();
    let links = all
        .links
        .iter()
        .filter(|l| l.kind == LinkKind::Network)
        .cloned()
        .collect();
    let hw = HardwareGraph::merge_cluster(&nodes, links, all.collectives.clone());
    hw.validate().unwrap();
    hw
}

/// Stages of replica 0 that hold experts in both VRAM and RAM slots.
fn ram_expert_stages(plan: &Plan) -> usize {
    plan.replicas[0]
        .stages
        .iter()
        .filter(|s| {
            s.expert_cache
                .iter()
                .any(|e| e.device.ends_with("gpu0") && e.slots > 0)
                && s.expert_cache
                    .iter()
                    .any(|e| e.device.ends_with("cpu0") && e.slots > 0)
        })
        .count()
}

fn spanning_with_ram_experts(cands: &[ClusterCandidate]) -> Vec<&ClusterCandidate> {
    cands
        .iter()
        .filter(|c| c.plan.replicas[0].stages.len() == 2 && ram_expert_stages(&c.plan) == 2)
        .collect()
}

/// The discovered graph with the CPU rate cleared: a node whose CPU rate is unknown.
fn without_cpu_rate() -> HardwareGraph {
    let mut hw = discovered(true);
    for d in hw.devices.iter_mut().filter(|d| d.kind == DeviceKind::Cpu) {
        d.compute = None;
    }
    hw
}

// hardware: none
#[test]
fn a_graph_without_a_cpu_rate_has_no_ram_expert_candidate() {
    // What the rehearsal fix left: the placements with RAM experts are skipped as unpriced.
    let hw = without_cpu_rate();
    let (cands, _) = enumerate_cluster(&hw, &fixtures::moe_235b(), &Hints::default(), Phase(6));
    assert!(spanning_with_ram_experts(&cands).is_empty());
}

// hardware: none
#[test]
fn without_a_models_dir_no_node_has_a_builder_disk_and_the_235b_has_no_candidate() {
    // What the pre-exit check found: discovery named the disks `disk0..`, the planner homed
    // every expert in RAM and the validator rejected the 235B with MemoryExceeded.
    let hw = discovered(false);
    assert!(hw.device("a/nvme0").is_none() && hw.device("a/disk0").is_some());
    let (cands, _) = enumerate_cluster(&hw, &fixtures::moe_235b(), &Hints::default(), Phase(6));
    assert!(spanning_with_ram_experts(&cands).is_empty());
}

// hardware: none
#[test]
fn discovery_gives_the_235b_a_spanning_candidate_with_ram_experts() {
    let hw = discovered(true);
    // No edit after the merge: the builder disk is the discovery's `nvme0`, with a rate.
    for n in ["a", "b"] {
        let d = hw.device(&format!("{n}/nvme0")).unwrap();
        assert!(d.disk_rate(1).is_some(), "{n}/nvme0 has no read rate");
        assert!(hw.device(&format!("{n}/cpu0")).unwrap().compute.is_some());
    }
    let layout = fixtures::moe_235b();
    let (cands, pruned) = enumerate_cluster(&hw, &layout, &Hints::default(), Phase(6));
    let ram = spanning_with_ram_experts(&cands);
    assert!(
        !ram.is_empty(),
        "no spanning candidate with RAM experts; {} candidates, pruned: {:?}",
        cands.len(),
        pruned.iter().take(5).collect::<Vec<_>>()
    );
    // The pick is the best candidate; the confirmation (ADR 0092) ran ubatch 2048 with the
    // experts of each span in VRAM and RAM.
    let pick = &cands[0];
    println!(
        "case_s_candidate pick={} stages={} ubatch={:?} micro_batches={} ram_expert_stages={} \
         pp_tps={:.1} tg_tps={:.2} note={}",
        pick.id,
        pick.plan.replicas[0].stages.len(),
        pick.plan.knobs.ubatch,
        pick.plan.knobs.prefill_micro_batches,
        ram_expert_stages(&pick.plan),
        pick.prediction.pp_tps,
        pick.prediction.tg_tps,
        pick.notes.join(" ")
    );
    let plan = serde_json::to_string(&pick.plan).unwrap();
    for n in ["a", "b"] {
        assert!(
            plan.contains(&format!("{n}/nvme0")),
            "the pick keeps experts on {n}'s builder disk"
        );
    }
    assert_eq!(
        pick.plan.replicas[0].stages.len(),
        2,
        "the pick spans both nodes"
    );
    assert_eq!(
        ram_expert_stages(&pick.plan),
        2,
        "the pick has RAM experts on both nodes"
    );
    assert_eq!(pick.plan.knobs.ubatch, Some(2048), "the confirmed ubatch");
}

// hardware: none
#[test]
fn a_skipped_unpriced_placement_is_named_in_the_candidate_notes() {
    // The 30B on the same graph: placements with RAM experts are feasible but unpriced, the
    // others compete, and the candidate says what was left out and why.
    let hw = without_cpu_rate();
    let layout = tetraeder_planner::multi_cli::parse_layout(include_str!(
        "../fixtures/phase6-rehearsal/layout.json"
    ))
    .unwrap();
    let (cands, _) = enumerate_cluster(&hw, &layout, &Hints::default(), Phase(6));
    assert!(!cands.is_empty());
    assert!(
        cands
            .iter()
            .any(|c| c.notes.iter().any(|n| n.contains("skipped, not priced"))),
        "{:?}",
        cands[0].notes
    );
}

// ---- 235B link feasibility (rehearsal 2, ADR 0091 answer 10) --------------------------------

/// The five link probes of rehearsal 2 (`bench/results/phase6-rehearsal-2-20261125/link/`), frames
/// 4 KiB to 4 MiB, in the order they were taken.
const PROBES: [(&str, &str); 5] = [
    (
        "start",
        include_str!("../fixtures/phase6-rehearsal/link-probes/probe-start.json"),
    ),
    (
        "mid-1",
        include_str!("../fixtures/phase6-rehearsal/link-probes/probe-mid-1.json"),
    ),
    (
        "mid-2",
        include_str!("../fixtures/phase6-rehearsal/link-probes/probe-mid-2.json"),
    ),
    (
        "mid-3",
        include_str!("../fixtures/phase6-rehearsal/link-probes/probe-mid-3.json"),
    ),
    (
        "end",
        include_str!("../fixtures/phase6-rehearsal/link-probes/probe-end.json"),
    ),
];

/// The graph with a probe's network links and `a`/`b` copy curves in place of its own, nothing
/// else changed (what `tetraeder cluster merge --swap-link` does).
fn with_probe(hw: &HardwareGraph, probe: &str) -> HardwareGraph {
    #[derive(serde::Deserialize)]
    struct Probe {
        links: Vec<tetraeder_profiler::Link>,
        curves: Vec<tetraeder_profiler::CollectiveCurve>,
    }
    let p: Probe = serde_json::from_str(probe).unwrap();
    let mut hw = hw.clone();
    let on_probe = |c: &tetraeder_profiler::CollectiveCurve| {
        c.ranks.len() == 2
            && c.ranks
                .iter()
                .all(|r| r.starts_with("a/") || r.starts_with("b/"))
    };
    hw.collectives.retain(|c| !on_probe(c));
    hw.collectives.extend(p.curves);
    hw.links.retain(|l| l.kind != LinkKind::Network);
    hw.links.extend(p.links);
    hw.validate().unwrap();
    hw
}

fn priced(hw: &HardwareGraph, plan: &Plan) -> (f64, f64) {
    let w = tetraeder_planner::Workload {
        prompt: 2048,
        r#gen: 128,
    };
    let p = tetraeder_planner::cluster::predict_cluster_plan(hw, &fixtures::moe_235b(), plan, &w)
        .unwrap();
    (p.pp_tps, p.tg_tps)
}

// hardware: none
#[test]
fn the_235b_pick_priced_on_each_rehearsal_2_probe() {
    let base = discovered(true);
    let layout = fixtures::moe_235b();
    // The pick is chosen on the graph with the start probe, as a session freezes it, and kept.
    let start = with_probe(&base, PROBES[0].1);
    let (cands, _) = enumerate_cluster(&start, &layout, &Hints::default(), Phase(6));
    let pick = &cands[0];
    let stages = &pick.plan.replicas[0].stages;
    assert_eq!(stages.len(), 2);
    // The probe's curve is the graph's after the swap, point for point.
    let c = start.network_curve("a", "b").unwrap();
    assert_eq!(c.points.len(), 6);
    assert_eq!(c.points[0].0, 4096.0);

    let mut rows = Vec::new();
    for (name, probe) in PROBES {
        let (pp, tg) = priced(&with_probe(&base, probe), &pick.plan);
        rows.push((name, pp, tg));
    }
    let (pp0, tg0) = (rows[0].1, rows[0].2);
    let mut max_pp = 0.0f64;
    let mut max_tg = 0.0f64;
    for (name, pp, tg) in &rows {
        let (mp, mt) = ((pp / pp0 - 1.0).abs(), (tg / tg0 - 1.0).abs());
        max_pp = max_pp.max(mp);
        max_tg = max_tg.max(mt);
        println!(
            "link_feasibility probe={name} pp_tps={pp:.2} tg_tps={tg:.3} pp_move={mp:.4} tg_move={mt:.4}"
        );
    }
    println!(
        "link_feasibility pick={} leader={} stages={:?} max_pp_move={max_pp:.4} max_tg_move={max_tg:.4}",
        pick.id,
        stages[0].devices.first().cloned().unwrap_or_default(),
        stages
            .iter()
            .map(|s| (s.layers, s.devices.clone()))
            .collect::<Vec<_>>()
    );
    // The pick's leader is node `a`, the head of the rehearsal's cluster: a head-led plan.
    assert!(
        stages[0].devices[0].starts_with("a/"),
        "{:?}",
        stages[0].devices
    );
    // Priced on the start probe (pp rides on the curve extrapolated from 4 MiB to the 32 MiB
    // hidden state of a micro-batch, tg on the small frames).
    assert!(
        (pp0 - 53.6).abs() < 0.5 && (tg0 - 3.96).abs() < 0.05,
        "{pp0} {tg0}"
    );
    // The link rule (ADR 0091 answer 10) would pass: the frozen pick moves under 0.08 in both
    // rates across the five probes, though the largest frames of the 30B split moved 3.5 times.
    assert!(max_pp < 0.08 && max_tg < 0.08, "{max_pp} {max_tg}");
    // The swap took: the five prices are not one price.
    assert!(
        rows.iter()
            .any(|r| (r.1 / pp0 - 1.0).abs() > 1e-6 || (r.2 / tg0 - 1.0).abs() > 1e-6)
    );
}
