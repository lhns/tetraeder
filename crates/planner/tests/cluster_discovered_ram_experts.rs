//! Case S needs experts in RAM computed on the CPU (ADR 0091). A discovered CPU has no compute
//! spec, and the rehearsal fix made `choose` skip every placement the graph cannot price, so a
//! graph as `tetraeder cluster profile` produces it (two 3090 nodes, the CPU rate missing) had
//! no spanning candidate with RAM experts for the 235B. Discovery now fills a per-core CPU prior
//! (`fill_from_priors`). This test runs the planner on the rehearsal's discovered graph with
//! 64 GB RAM budgets per node and the 235B layout, and checks the pick against the confirmed
//! shape of ADR 0092 (ubatch 2048, experts in VRAM and RAM on both nodes).
//!
//! Hardware label: none (the graph is a copy of the rehearsal's `a/hw.json`).

use tetraeder_plan::{Hints, Phase, Plan};
use tetraeder_planner::layout::fixtures;
use tetraeder_planner::{ClusterCandidate, Priors, enumerate_cluster, fill_from_priors};
use tetraeder_profiler::{DeviceKind, HardwareGraph};

const HW: &str = include_str!("../fixtures/phase6-rehearsal/hw.json");
/// The Case S RAM budget per node (`EMULATE_RAM_GB=64`).
const RAM: u64 = 64 << 30;

/// The graph `tetraeder cluster profile` wrote for two 3090 boxes, with the 64 GB budgets and
/// the one thing a judged session adds (`scripts/sp_make_hw.py`): each node's experts live on a
/// builder disk named `nvme0` (discovery names disks `disk0`, `disk1`, ...; the planner reads
/// the builder's experts from `nvme0`, and without it homes every expert in RAM, which a 235B
/// does not fit). The disk rates are the committed NVMe prior.
fn discovered() -> HardwareGraph {
    let mut hw = tetraeder_planner::multi_cli::parse_hw(HW).unwrap();
    let prior = HardwareGraph::from_toml(include_str!("../fixtures/rtx3090_prior.toml")).unwrap();
    for node in ["a", "b"] {
        hw.devices
            .retain(|d| !d.id.starts_with(&format!("{node}/disk")));
        hw.links
            .retain(|l| !l.id.starts_with(&format!("{node}/storage-disk")));
        let mut disk = prior.device("nvme0").unwrap().clone();
        disk.id = format!("{node}/nvme0");
        disk.node = Some(node.to_owned());
        hw.devices.push(disk);
        let mut link = prior
            .links
            .iter()
            .find(|l| l.id == "nvme-cpu0")
            .unwrap()
            .clone();
        link.id = format!("{node}/nvme-cpu0");
        link.a = format!("{node}/nvme0");
        link.b = format!("{node}/cpu0");
        hw.links.push(link);
    }
    for d in hw.devices.iter_mut().filter(|d| d.kind == DeviceKind::Cpu) {
        d.capacity_bytes = RAM;
        // The fixture was written before discovery filled the CPU rate.
        d.compute = None;
    }
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

// hardware: none
#[test]
fn a_graph_without_a_cpu_rate_has_no_ram_expert_candidate() {
    // What the rehearsal fix left: the placements with RAM experts are skipped as unpriced.
    let hw = discovered();
    let (cands, _) = enumerate_cluster(&hw, &fixtures::moe_235b(), &Hints::default(), Phase(6));
    assert!(spanning_with_ram_experts(&cands).is_empty());
}

// hardware: none
#[test]
fn discovery_gives_the_235b_a_spanning_candidate_with_ram_experts() {
    let mut hw = discovered();
    let notes = fill_from_priors(&mut hw, &Priors::builtin());
    assert_eq!(
        notes
            .iter()
            .filter(|n| n.contains("cpu0: compute is a prior"))
            .count(),
        2,
        "{notes:?}"
    );
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
    let hw = discovered();
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
