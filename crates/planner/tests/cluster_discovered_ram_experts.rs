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
    prepare_cluster_gpus,
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
        "link_feasibility pick={} micro={} leader={} stages={:?} max_pp_move={max_pp:.4} max_tg_move={max_tg:.4}",
        pick.id,
        pick.plan.knobs.prefill_micro_batches,
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
    // Priced on the start probe (pp rides on the curve extrapolated from 4 MiB to the 32 MiB hidden
    // state of the one batch of 2048, tg on the small frames). Cost model 6 now prices each
    // micro-batch as a batch of its own (the offload copy and the weight reads once per
    // micro-batch): the pick that was ubatch 2048 with four micro-batches at 53.6 tok/s (pp move
    // 5.8%) is one batch of 2048 at 40.1 tok/s (2026-11-26).
    assert_eq!(pick.plan.knobs.prefill_micro_batches, 1);
    assert!(
        (pp0 - 40.1).abs() < 0.5 && (tg0 - 3.96).abs() < 0.05,
        "{pp0} {tg0}"
    );
    // On priors the pick's pp moves 18.7% across the five probes, over 0.08, because one batch of
    // 2048 sends one 32 MiB frame that nothing hides (the gate prices on the fitted rates instead:
    // see the fitted tests below); tg stays under.
    assert!(
        (max_pp - 0.187).abs() < 0.01 && max_tg < 0.08,
        "{max_pp} {max_tg}"
    );
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
        "{} ub {:?} micro {} pp {:.1} tg {:.2}: {}",
        c.id,
        c.plan.knobs.ubatch,
        c.plan.knobs.prefill_micro_batches,
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
    // No plan the validator rejects for memory (`planner-footprint-validator-gap`).
    assert!(
        !pruned
            .iter()
            .any(|(_, why)| why.starts_with("invalid") && why.contains("MemoryExceeded")),
        "{pruned:?}"
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

// ---- link feasibility on fitted rates (ADR 0091 History 2026-11-26) ------------------------------

/// A node graph fitted from the anchors of the confirmed segment session (a 3090 box,
/// `segment-confirm-20261125b`): the compute rates the Case S link gate prices with.
const FITTED: &str =
    include_str!("../../../bench/results/segment-confirm-20261125b/predictions/fitted.toml");

/// The three probes of rehearsal 1 (`phase6-rehearsal-20261125/a/`): 64 MiB frames, a to b in
/// about 1.3 s with autotuned buffers; the explicit-buffer runs slow only b to a.
const R1_PROBES: [(&str, &str); 3] = [
    (
        "auto",
        include_str!("../../../bench/results/phase6-rehearsal-20261125/a/probe-auto.json"),
    ),
    (
        "explicit-4m",
        include_str!(
            "../../../bench/results/phase6-rehearsal-20261125/a/probe-explicit-4194304.json"
        ),
    ),
    (
        "explicit-16m",
        include_str!(
            "../../../bench/results/phase6-rehearsal-20261125/a/probe-explicit-16777216.json"
        ),
    ),
];

/// Rehearsal 1's probes with only the a to b curve of each: the explicit-buffer runs slowed b to a
/// 6 to 7 times (a setting of the probe, see the README of that session), which is no link
/// variance; the like-for-like set keeps the autotuned run's b to a curve and link and takes a to b
/// from each run (64 MiB in 1.342, 1.313 and 1.282 s, a 4.6% spread).
fn r1_a_to_b_only() -> Vec<(String, String)> {
    let auto: serde_json::Value = serde_json::from_str(R1_PROBES[0].1).unwrap();
    let ab = |c: &serde_json::Value| c["ranks"][0] == "a/cpu0";
    R1_PROBES
        .iter()
        .map(|(n, p)| {
            let v: serde_json::Value = serde_json::from_str(p).unwrap();
            let mut mixed = auto.clone();
            let from = v["curves"]
                .as_array()
                .unwrap()
                .iter()
                .find(|c| ab(c))
                .unwrap();
            let slot = mixed["curves"]
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .find(|c| ab(c))
                .unwrap();
            *slot = from.clone();
            ((*n).to_string(), mixed.to_string())
        })
        .collect()
}

/// The graph with the fitted node graph in place of each node's own rates, as
/// `tetraeder cluster merge --node a=F --node b=F` does (`hw_merge::merge_fitted`): a fitted
/// device replaces the fields of `<node>/<id>` except its identity and a GPU's capacity, a fitted
/// link its alpha and beta; the links of the network and the curves stay.
fn with_fitted(hw: &HardwareGraph) -> HardwareGraph {
    const KEEP: [&str; 8] = [
        "id",
        "name",
        "kind",
        "tier",
        "node",
        "index",
        "numa_nodes",
        "cores",
    ];
    let fitted: serde_json::Value =
        serde_json::from_str(&HardwareGraph::from_toml(FITTED).unwrap().to_json().unwrap())
            .unwrap();
    let mut v: serde_json::Value = serde_json::from_str(&hw.to_json().unwrap()).unwrap();
    let s = |x: &serde_json::Value, k: &str| x.get(k).and_then(|y| y.as_str()).map(str::to_owned);
    for node in ["a", "b"] {
        for fd in fitted["devices"].as_array().unwrap() {
            let id = format!("{node}/{}", s(fd, "id").unwrap());
            let dst = v["devices"]
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .find(|d| s(d, "id").as_deref() == Some(&id))
                .unwrap_or_else(|| panic!("no device {id}"));
            let gpu = s(dst, "kind").as_deref() == Some("gpu");
            for (k, x) in fd.as_object().unwrap() {
                if !(KEEP.contains(&k.as_str()) || (k == "capacity_bytes" && gpu)) {
                    dst[k] = x.clone();
                }
            }
        }
        for fl in fitted["links"].as_array().unwrap() {
            let (a, b) = (
                format!("{node}/{}", s(fl, "a").unwrap()),
                format!("{node}/{}", s(fl, "b").unwrap()),
            );
            let dst = v["links"]
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .find(|l| {
                    s(l, "kind") == s(fl, "kind")
                        && ((s(l, "a").as_deref() == Some(&a) && s(l, "b").as_deref() == Some(&b))
                            || (s(l, "a").as_deref() == Some(&b)
                                && s(l, "b").as_deref() == Some(&a)))
                })
                .unwrap_or_else(|| panic!("no link {a} {b}"));
            for k in ["alpha_s", "beta_bytes_per_s"] {
                dst[k] = fl[k].clone();
            }
        }
    }
    HardwareGraph::from_json(&serde_json::to_string(&v).unwrap()).unwrap()
}

/// One row per probe: (name, pp, tg) of `plan` priced on `base` with the probe swapped in.
fn rates_by_probe(
    base: &HardwareGraph,
    layout: &tetraeder_planner::layout::ModelLayout,
    plan: &Plan,
    probes: &[(&str, &str)],
) -> Vec<(String, f64, f64)> {
    let w = tetraeder_planner::Workload {
        prompt: 2048,
        r#gen: 128,
    };
    probes
        .iter()
        .map(|(n, p)| {
            let hw = with_probe(base, p);
            let r =
                tetraeder_planner::cluster::predict_cluster_plan(&hw, layout, plan, &w).unwrap();
            ((*n).to_string(), r.pp_tps, r.tg_tps)
        })
        .collect()
}

/// Prints the table and returns (max pp move, max tg move) against the first probe.
fn report(tag: &str, rows: &[(String, f64, f64)]) -> (f64, f64) {
    let (pp0, tg0) = (rows[0].1, rows[0].2);
    let (mut mp, mut mt) = (0.0f64, 0.0f64);
    for (n, pp, tg) in rows {
        let (dp, dt) = ((pp / pp0 - 1.0).abs(), (tg / tg0 - 1.0).abs());
        mp = mp.max(dp);
        mt = mt.max(dt);
        println!(
            "fitted_feasibility {tag} probe={n} pp_tps={pp:.2} tg_tps={tg:.3} pp_move={dp:.4} tg_move={dt:.4}"
        );
    }
    println!("fitted_feasibility {tag} max_pp_move={mp:.4} max_tg_move={mt:.4}");
    (mp, mt)
}

// hardware: none
#[test]
fn the_fitted_graph_replaces_the_discovered_rates_and_keeps_the_nodes_identity() {
    let base = discovered_at(true, MEASURED_RAM);
    let fit = with_fitted(&base);
    for n in ["a", "b"] {
        let (c0, c1) = (
            base.device(&format!("{n}/cpu0")).unwrap(),
            fit.device(&format!("{n}/cpu0")).unwrap(),
        );
        assert_ne!(c0.compute, c1.compute, "{n} cpu rate is the fitted one");
        let g = fit.device(&format!("{n}/gpu0")).unwrap();
        assert_eq!(
            g.compute.as_ref().unwrap().flops_per_s,
            37_191_779_652_309.76
        );
        assert_eq!(
            g.capacity_bytes,
            base.device(&format!("{n}/gpu0")).unwrap().capacity_bytes
        );
    }
    assert_eq!(fit.network_curve("a", "b"), base.network_curve("a", "b"));
}

/// The covering pick of Case S (a 0..44, b 44..94, 32 + 96 slots), chosen on the discovered
/// graph with the start probe, as the plan stage does.
fn covering_pick() -> (ClusterCandidate, tetraeder_planner::layout::ModelLayout) {
    let hw = with_probe(&discovered_at(true, MEASURED_RAM), PROBES[0].1);
    let layout = real_235b();
    let (cands, _) = enumerate_cluster_covering(&hw, &layout, &Hints::default(), Phase(6));
    (cands.into_iter().next().unwrap(), layout)
}

// hardware: none
#[test]
fn the_covering_pick_is_a_0_44_b_44_94_with_32_and_96_slots() {
    let (pick, layout) = covering_pick();
    let s: Vec<(String, (u32, u32), u32, u32)> = segs(&pick.plan, &layout)
        .into_iter()
        .map(|s| (s.node, s.layers, s.v, s.r))
        .collect();
    assert_eq!(
        s,
        vec![
            ("a".to_string(), (0, 44), 32, 96),
            ("b".to_string(), (44, 94), 32, 96)
        ]
    );
}

/// The covering pick is byte-identical to the committed snapshot: plan and prediction (tree node
/// `planner-footprint-validator-gap`; a judged Case S run is pending on this pick).
// hardware: none
#[test]
fn the_covering_pick_matches_the_committed_snapshot() {
    let (pick, _) = covering_pick();
    let text = format!(
        "{}\npp {:.6} tg {:.6} request_s {:.6}\n",
        serde_json::to_string_pretty(&pick.plan).unwrap(),
        pick.prediction.pp_tps,
        pick.prediction.tg_tps,
        pick.prediction.request_s
    );
    let path = format!(
        "{}/tests/snapshots/case_s_covering_pick.txt",
        env!("CARGO_MANIFEST_DIR")
    );
    let want = std::fs::read_to_string(&path).unwrap_or_default();
    assert_eq!(
        want.replace("\r\n", "\n"),
        text,
        "the Case S pick changed; if on purpose, write this to {path}:\n<<<SNAP\n{text}SNAP>>>"
    );
}

// hardware: none
#[test]
fn the_covering_pick_priced_on_fitted_rates_on_each_rehearsal_probe() {
    let (pick, layout) = covering_pick();
    let fit = with_fitted(&discovered_at(true, MEASURED_RAM));
    let r2 = rates_by_probe(&fit, &layout, &pick.plan, &PROBES);
    let (mp, mt) = report("covering r2", &r2);
    // The same plan cut into four micro-batches (the pick before the per-micro-batch copy).
    let mut plan4 = pick.plan.clone();
    plan4.knobs.prefill_micro_batches = 4;
    let r2m4 = rates_by_probe(&fit, &layout, &plan4, &PROBES);
    report("covering 4 micro-batches r2", &r2m4);
    let r1 = rates_by_probe(&fit, &layout, &pick.plan, &R1_PROBES);
    let (mp1, mt1) = report("covering r1 as probed", &r1);
    // The like-for-like set: the explicit-buffer runs only differ from the auto one in b to a.
    let mixed = r1_a_to_b_only();
    let mixed: Vec<(&str, &str)> = mixed
        .iter()
        .map(|(n, p)| (n.as_str(), p.as_str()))
        .collect();
    let r1ab = rates_by_probe(&fit, &layout, &pick.plan, &mixed);
    let (mpab, mtab) = report("covering r1 a to b only", &r1ab);
    // The swap took: five prices, not one.
    assert!(r2.iter().any(|r| (r.1 / r2[0].1 - 1.0).abs() > 1e-6));
    // Recorded in the rehearsal 2 README ("235B link feasibility on fitted rates", cost model 6
    // with the offload copy per micro-batch): rehearsal 2's links move the fitted prediction's pp by
    // up to 43%, rehearsal 1's by up to 24%, both far over the gate's 0.08; tg moves under 4%.
    assert!(
        (r2[0].1 - 81.4).abs() < 0.5 && (r2[0].2 - 5.417).abs() < 0.01,
        "{:?}",
        r2[0]
    );
    assert!(
        (mp - 0.428).abs() < 0.01 && (mt - 0.0165).abs() < 0.002,
        "{mp} {mt}"
    );
    assert!(
        (r1[0].1 - 196.0).abs() < 1.0 && (r1[0].2 - 4.650).abs() < 0.01,
        "{:?}",
        r1[0]
    );
    assert!(
        (mp1 - 0.2433).abs() < 0.01 && (mt1 - 0.0345).abs() < 0.003,
        "{mp1} {mt1}"
    );
    assert!(mp > 0.08, "the fitted gate would abort rehearsal 2's link");
    // Rehearsal 1 as probed moves 24% only because the explicit buffers cut b to a; with a to b
    // alone (a 4.6% spread of the 64 MiB point) the move is MPAB.
    assert!(mp1 > 0.2, "{mp1}");
    assert!(mpab < 0.08 && mtab < 0.08, "{mpab} {mtab}");
}

// hardware: none
#[test]
fn the_synthetic_235b_pick_priced_on_fitted_rates_on_each_rehearsal_probe() {
    let base = discovered(true);
    let layout = fixtures::moe_235b();
    let start = with_probe(&base, PROBES[0].1);
    let (cands, _) = enumerate_cluster(&start, &layout, &Hints::default(), Phase(6));
    let fit = with_fitted(&base);
    let r2 = rates_by_probe(&fit, &layout, &cands[0].plan, &PROBES);
    let (mp, _) = report("synthetic r2", &r2);
    let r1 = rates_by_probe(&fit, &layout, &cands[0].plan, &R1_PROBES);
    let (mp1, _) = report("synthetic r1", &r1);
    // The synthetic layout's pick (c00, a 0..40, b 40..94) moves as the covering pick does.
    assert!(
        (mp - 0.428).abs() < 0.01 && (mp1 - 0.2434).abs() < 0.01,
        "{mp} {mp1}"
    );
}

// hardware: none
#[test]
fn the_covering_pick_is_one_batch_of_2048_since_each_micro_batch_pays_the_offload_copy() {
    // Before the copy was charged per micro-batch the pick was ubatch 2048 with four micro-batches
    // (pp 59.5 on priors); with the copy and the weight reads once per 512-token batch it is one
    // batch of 2048 (pp 56.3), the regime cost model 6 was confirmed on. The anchors follow the pick
    // (`plans/anchor-step.txt` is the micro-batch of its first step).
    let (pick, _) = covering_pick();
    assert_eq!(pick.plan.knobs.ubatch, Some(2048));
    assert_eq!(pick.plan.knobs.prefill_micro_batches, 1);
}

// hardware: none
#[test]
fn a_covering_plan_with_four_micro_batches_pays_the_offload_copy_in_each() {
    // ADR 0092 section 5 and the pipeline: a covering segment copies its RAM-resident weights to the
    // GPU once per batch, so four micro-batches of 512 tokens copy four times what one ubatch of
    // 2048 does.
    let (pick, layout) = covering_pick();
    // The same plan with four micro-batches (what the pick was before the copy was charged per
    // micro-batch): the planner's own pick is the one batch of 2048, see the next test.
    let mut plan4 = pick.plan.clone();
    plan4.knobs.prefill_micro_batches = 4;
    let fit = with_fitted(&discovered_at(true, MEASURED_RAM));
    let f = tetraeder_planner::cluster::cluster_features(&fit, &layout, &plan4).unwrap();
    assert_eq!(f.prefill_micro_batches, 4);
    let copy_s = |seg: usize, ubatch: u32| -> f64 {
        let mut run = f.segments[seg].run.clone();
        run.ubatch = ubatch;
        let g = fit.node_graph(f.segments[seg].node.as_deref());
        tetraeder_costmodel::predict::predict(&g, &run, 2048, 1)
            .unwrap()
            .pp_breakdown
            .iter()
            .filter(|e| e.0.starts_with("offload_copy:"))
            .map(|e| e.1)
            .sum()
    };
    let mut free = fit.clone();
    free.collectives.clear();
    for l in &mut free.links {
        if l.kind == LinkKind::Network {
            l.alpha_s.value = 0.0;
            l.beta_bytes_per_s.value = f64::INFINITY;
        }
    }
    let t = tetraeder_costmodel::cluster::predict_cluster(&free, &f, 2048, 8).unwrap();
    let mut copying = 0;
    for seg in 0..f.segments.len() {
        let (one, four) = (copy_s(seg, 2048), copy_s(seg, 512));
        println!("micro_batch_copy seg={seg} one_ubatch_s={one:.3} four_micro_batches_s={four:.3}");
        if one.abs() < 1e-9 {
            continue; // the head segment on the leader holds no RAM experts
        }
        copying += 1;
        assert!(one > 0.1, "segment {seg} copies its RAM experts: {one}");
        assert!((four / one - 4.0).abs() < 1e-6, "{four} vs 4 x {one}");
        // With free links the prompt takes at least the copies of its slowest segment's four batches.
        assert!(
            2048.0 / t.pp_tps >= four * 0.999,
            "{} vs {four}",
            2048.0 / t.pp_tps
        );
    }
    assert_eq!(copying, 2, "both nodes' segments copy");
}

// ---- A-class cards under a per-node GPU cap (ADR 0091 History 2026-11-26, answer 9) -----------

/// The A6000 reports 48 GiB (51.5 GB); the A40 is 48 GB.
const CARD_48: u64 = 51_539_607_552;
const CAP: u64 = 24 << 30;

/// The discovered graph with node `node`'s GPU replaced by a card `cluster profile` has no spec
/// for: its name and capacity, no compute, no bandwidth (what the A6000 box reported).
fn with_bare_gpu(hw: &HardwareGraph, node: &str, name: &str, bytes: u64) -> HardwareGraph {
    let mut hw = hw.clone();
    let d = hw
        .devices
        .iter_mut()
        .find(|d| d.id == format!("{node}/gpu0"))
        .unwrap();
    d.name = name.into();
    d.capacity_bytes = bytes;
    d.compute = None;
    d.bandwidth_bytes_per_s = None;
    hw
}

fn covering(hw: &HardwareGraph) -> Vec<ClusterCandidate> {
    enumerate_cluster_covering(hw, &real_235b(), &Hints::default(), Phase(6)).0
}

fn check_pick(hw: &HardwareGraph, label: &str) {
    let layout = real_235b();
    let cands = covering(hw);
    assert!(!cands.is_empty(), "{label}: no covering candidate");
    let pick = &cands[0];
    println!(
        "gpu_cap_pick {label} candidates={} {}",
        cands.len(),
        show(pick, &layout)
    );
    assert_eq!(pick.plan.replicas[0].stages.len(), 2);
    for s in segs(&pick.plan, &layout) {
        assert!(s.v + s.r >= layout.n_expert, "{label}: {}", s.node);
        assert!(s.gpu <= CAP / 100 * 99, "{label}: {} {}", s.node, s.gpu);
    }
}

// hardware: none
#[test]
fn an_uncapped_unspecced_a6000_node_has_no_covering_plan() {
    // What the plan stage of phase6-exit-s-20261126 found.
    let base = discovered_at(true, MEASURED_RAM);
    let hw = with_bare_gpu(&base, "b", "NVIDIA RTX A6000", CARD_48);
    assert!(covering(&hw).is_empty());
}

// hardware: none
#[test]
fn a_3090_and_an_a6000_under_the_cap_have_a_covering_pick() {
    let base = discovered_at(true, MEASURED_RAM);
    let mut hw = with_bare_gpu(&base, "b", "NVIDIA RTX A6000", CARD_48);
    let notes = prepare_cluster_gpus(&mut hw, &Priors::builtin(), Some(CAP));
    let g = hw.device("b/gpu0").unwrap();
    assert_eq!(g.capacity_bytes, CAP);
    assert!(g.compute.is_some() && g.bandwidth_bytes_per_s.is_some());
    // The 3090 is already at 24 GiB and keeps its numbers.
    assert_eq!(hw.device("a/gpu0").unwrap(), base.device("a/gpu0").unwrap());
    // The cap is in the graph.
    assert!(
        hw.notes
            .iter()
            .any(|n| n.contains("b/gpu0") && n.contains("capped")),
        "{notes:?}"
    );
    hw.validate().unwrap();
    check_pick(&hw, "3090+a6000");
}

// hardware: none
#[test]
fn an_a6000_head_and_a_3090_under_the_cap_have_a_covering_pick() {
    // The orientation of phase6-exit-s-20261126: the A6000 on node a.
    let base = discovered_at(true, MEASURED_RAM);
    let mut hw = with_bare_gpu(&base, "a", "NVIDIA RTX A6000", CARD_48);
    prepare_cluster_gpus(&mut hw, &Priors::builtin(), Some(CAP));
    check_pick(&hw, "a6000+3090");
}

// hardware: none
#[test]
fn an_a40_pair_under_the_cap_has_a_covering_pick() {
    let base = discovered_at(true, MEASURED_RAM);
    let mut hw = with_bare_gpu(&base, "a", "NVIDIA A40", CARD_48);
    hw = with_bare_gpu(&hw, "b", "NVIDIA A40", CARD_48);
    prepare_cluster_gpus(&mut hw, &Priors::builtin(), Some(CAP));
    for n in ["a", "b"] {
        assert_eq!(hw.device(&format!("{n}/gpu0")).unwrap().capacity_bytes, CAP);
    }
    check_pick(&hw, "a40+a40");
}

// hardware: none
#[test]
fn the_gpu_class_table_covers_the_a_class_and_ada_cards() {
    let p = Priors::builtin();
    for (name, bw) in [
        ("NVIDIA RTX A6000", 7.68e11),
        ("NVIDIA A40", 6.96e11),
        ("NVIDIA L40", 8.64e11),
        ("NVIDIA L40S", 8.64e11),
        ("NVIDIA GeForce RTX 3090", 9.36e11),
        ("NVIDIA GeForce RTX 4090", 1.008e12),
    ] {
        let base = discovered_at(true, MEASURED_RAM);
        let mut hw = with_bare_gpu(&base, "a", name, CARD_48);
        let notes = prepare_cluster_gpus(&mut hw, &p, None);
        let g = hw.device("a/gpu0").unwrap();
        assert_eq!(g.bandwidth_bytes_per_s.unwrap().value, bw, "{name}");
        assert!(g.compute.unwrap().flops_per_s > 0.0, "{name}");
        assert!(!notes.iter().any(|n| n.contains("conservative")), "{name}");
        // No cap asked: the capacity stays.
        assert_eq!(g.capacity_bytes, CARD_48, "{name}");
    }
}

// hardware: none
#[test]
fn an_a4000_is_not_an_a40_and_an_l4_is_not_an_l40() {
    let p = Priors::builtin();
    let min_bw = p
        .gpu
        .iter()
        .map(|c| c.bandwidth_bytes_per_s)
        .fold(f64::MAX, f64::min);
    for name in ["NVIDIA RTX A4000", "NVIDIA L4"] {
        let base = discovered_at(true, MEASURED_RAM);
        let mut hw = with_bare_gpu(&base, "a", name, CARD_48);
        let notes = prepare_cluster_gpus(&mut hw, &p, None);
        let g = hw.device("a/gpu0").unwrap();
        assert_eq!(g.bandwidth_bytes_per_s.unwrap().value, min_bw, "{name}");
        assert!(notes.iter().any(|n| n.contains("conservative")), "{name}");
    }
}

// hardware: none
#[test]
fn a_gpu_not_in_the_table_gets_the_most_conservative_prior_and_a_note() {
    let p = Priors::builtin();
    let base = discovered_at(true, MEASURED_RAM);
    let mut hw = with_bare_gpu(&base, "a", "Mystery Card 9000", CARD_48);
    let notes = prepare_cluster_gpus(&mut hw, &p, Some(CAP));
    let g = hw.device("a/gpu0").unwrap();
    let min_bw = p
        .gpu
        .iter()
        .map(|c| c.bandwidth_bytes_per_s)
        .fold(f64::MAX, f64::min);
    let min_fl = p.gpu.iter().map(|c| c.flops_per_s).fold(f64::MAX, f64::min);
    assert_eq!(g.bandwidth_bytes_per_s.unwrap().value, min_bw);
    assert_eq!(g.compute.unwrap().flops_per_s, min_fl);
    assert!(
        notes
            .iter()
            .any(|n| n.contains("a/gpu0") && n.contains("conservative")),
        "{notes:?}"
    );
    assert!(hw.notes.iter().any(|n| n.contains("conservative")));
}

// hardware: none
#[test]
fn without_a_cap_and_with_specced_gpus_the_graph_is_unchanged() {
    // Single-node and 3090 graphs: nothing to fill, nothing to cap.
    let base = discovered_at(true, MEASURED_RAM);
    let mut hw = base.clone();
    let notes = prepare_cluster_gpus(&mut hw, &Priors::builtin(), None);
    assert_eq!(hw, base);
    assert!(notes.is_empty());
    // The cap at the cards' own size changes nothing either.
    prepare_cluster_gpus(&mut hw, &Priors::builtin(), Some(CAP));
    assert_eq!(hw, base);
}
