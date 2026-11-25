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
use tetraeder_planner::{
    ClusterCandidate, Priors, enumerate_cluster, enumerate_cluster_covering, fill_from_priors,
};
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
fn node_graph(all: &HardwareGraph, node: &str, models: bool, ram: u64) -> HardwareGraph {
    let mut g = all.node_graph(Some(node));
    g.devices.retain(|d| d.kind != DeviceKind::Disk);
    g.links.retain(|l| l.kind != LinkKind::Storage);
    for d in g.devices.iter_mut().filter(|d| d.kind == DeviceKind::Cpu) {
        d.capacity_bytes = ram;
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
    discovered_at(models, RAM)
}

/// [`discovered`] with `ram` bytes of RAM budget on each node.
fn discovered_at(models: bool, ram: u64) -> HardwareGraph {
    let all = tetraeder_planner::multi_cli::parse_hw(HW).unwrap();
    let nodes: Vec<(String, HardwareGraph)> = ["a", "b"]
        .iter()
        .map(|n| (n.to_string(), node_graph(&all, n, models, ram)))
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

// ---- covering candidates under the measured budget (ADR 0091 History 2026-11-25) --------------

/// The measured RAM budget of Case S boxes (59.4 to 59.9 GB under the 64 GB balloon).
const MEASURED_RAM: u64 = 59_400_000_000;

/// The committed 235B layout of the confirmed segment session (real tensor sizes).
fn real_235b() -> tetraeder_planner::layout::ModelLayout {
    tetraeder_planner::multi_cli::parse_layout(include_str!(
        "../../../bench/results/segment-confirm-20261125b/layout.json"
    ))
    .unwrap()
}

/// What the planner and the budgets say of one stage: node, layers, VRAM and RAM slots, and the
/// footprint estimate (VRAM, host) of its builder placement.
struct Seg {
    node: String,
    layers: (u32, u32),
    v: u32,
    r: u32,
    gpu: u64,
    host: u64,
}

fn segs(plan: &Plan, layout: &tetraeder_planner::layout::ModelLayout) -> Vec<Seg> {
    let n = plan.replicas[0].stages.len();
    plan.replicas[0]
        .stages
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let slots = |end: &str| {
                s.expert_cache
                    .iter()
                    .find(|e| e.device.ends_with(end))
                    .map_or(0, |e| e.slots)
            };
            let (v, r) = (slots("gpu0"), slots("cpu0"));
            let sub = tetraeder_planner::segment_layout(layout, s.layers, i == 0, i + 1 == n);
            let w = tetraeder_planner::Workload {
                prompt: 2048,
                r#gen: 128,
            };
            let f = tetraeder_planner::footprint(
                &sub,
                tetraeder_planner::Placement::Builder {
                    vram_slots: v,
                    ram_slots: r,
                },
                plan.knobs.ubatch.unwrap(),
                &w,
                tetraeder_plan::KvFormat::default(),
            );
            Seg {
                node: s.devices[0].split('/').next().unwrap().to_string(),
                layers: (s.layers.0, s.layers.1),
                v,
                r,
                gpu: f.gpu,
                host: f.host,
            }
        })
        .collect()
}

fn show(c: &ClusterCandidate, layout: &tetraeder_planner::layout::ModelLayout) -> String {
    let parts: Vec<String> = segs(&c.plan, layout)
        .iter()
        .map(|s| {
            format!(
                "{} {}..{} v{}+r{} gpu {:.2} GB host {:.2} GB covering {}",
                s.node,
                s.layers.0,
                s.layers.1,
                s.v,
                s.r,
                s.gpu as f64 / 1e9,
                s.host as f64 / 1e9,
                s.v + s.r >= layout.n_expert
            )
        })
        .collect();
    format!(
        "{} ub {:?} pp {:.1} tg {:.2}: {}",
        c.id,
        c.plan.knobs.ubatch,
        c.prediction.pp_tps,
        c.prediction.tg_tps,
        parts.join(" | ")
    )
}

// hardware: none
#[test]
fn covering_candidates_of_two_3090_nodes_at_the_measured_budget() {
    let hw = discovered_at(true, MEASURED_RAM);
    let layout = real_235b();
    let gpu = 24u64 << 30;
    let (all, _) = enumerate_cluster(&hw, &layout, &Hints::default(), Phase(6));
    println!("case_s_pick unrestricted {}", show(&all[0], &layout));
    let (cands, pruned) = enumerate_cluster_covering(&hw, &layout, &Hints::default(), Phase(6));
    println!(
        "case_s_pick covering candidates={} pruned={}",
        cands.len(),
        pruned.len()
    );
    for c in cands.iter().take(5) {
        println!("case_s_pick covering {}", show(c, &layout));
    }
    // The rehearsal's synthetic 235B layout (the earlier tests above) for the record.
    let synth = fixtures::moe_235b();
    let (c2, _) = enumerate_cluster_covering(&hw, &synth, &Hints::default(), Phase(6));
    println!(
        "case_s_pick synthetic layout covering candidates={} first={}",
        c2.len(),
        c2.first().map_or(String::new(), |c| show(c, &synth))
    );
    assert!(!cands.is_empty(), "no covering spanning candidate");
    for c in &cands {
        for s in segs(&c.plan, &layout) {
            assert!(
                s.v + s.r >= layout.n_expert,
                "{} {} holds {}+{} of {} experts",
                c.id,
                s.node,
                s.v,
                s.r,
                layout.n_expert
            );
            // The planner's margin of 1% on the RAM budget (the host estimate is exact).
            assert!(s.host <= MEASURED_RAM / 100 * 99, "{}: {}", c.id, s.host);
        }
    }
    // The pick's VRAM estimate (a head-last plan: the head is on the last stage, as `segs`
    // assumes) fits the card.
    for s in segs(&cands[0].plan, &layout) {
        assert!(s.gpu <= gpu / 100 * 99, "{}: {}", s.node, s.gpu);
    }
    // The pick spans both nodes, led by the head `a`, on the confirmed ubatch.
    let pick = &cands[0];
    assert_eq!(pick.plan.replicas[0].stages.len(), 2);
    assert!(pick.plan.replicas[0].stages[0].devices[0].starts_with("a/"));
    assert_eq!(pick.plan.knobs.ubatch, Some(2048));
}

// hardware: none
#[test]
fn no_covering_candidate_when_the_budget_cannot_hold_the_experts_and_the_reason_says_why() {
    // 40 GB per node cannot hold 94 layers of 128 experts between VRAM and RAM on two 24 GiB cards.
    let hw = discovered_at(true, 40_000_000_000);
    let layout = real_235b();
    let (cands, pruned) = enumerate_cluster_covering(&hw, &layout, &Hints::default(), Phase(6));
    assert!(cands.is_empty());
    let why = pruned
        .iter()
        .find(|(_, w)| w.contains("no covering placement"))
        .expect("a pruned reason names the covering need");
    println!("case_s_pick no covering at 40 GB: {} {}", why.0, why.1);
    assert!(why.1.contains("needs at least"), "{}", why.1);
    // The unrestricted planner still finds plans there (non-covering ones, disk reads in prefill).
    let (all, _) = enumerate_cluster(&hw, &layout, &Hints::default(), Phase(6));
    assert!(!all.is_empty());
    assert!(
        segs(&all[0].plan, &layout)
            .iter()
            .any(|s| s.v + s.r < layout.n_expert)
    );
}
