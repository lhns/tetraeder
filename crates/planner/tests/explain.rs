//! Tests for explain, diff and what-if (docs/plan.md "Explain, diff, what-if") on the simulated
//! hardware fixtures. Rule 4: every fixture, no hardware needed.

use tetraeder_plan::*;
use tetraeder_planner::cli;
use tetraeder_planner::diff::{DRAIN_S, RELOAD_S};
use tetraeder_planner::explain::{Explanation, Term};
use tetraeder_planner::whatif::{
    Change, Kind, Priors, default_candidates, rank_purchases, what_if,
};
use tetraeder_planner::{
    BitsQuality, NullQuality, QualityModel, SearchInput, SearchResult, explain, plan_diff, search,
};
use tetraeder_profiler::{DeviceKind, HardwareGraph, Measurement, Source};

const TWO_GPU: &str = include_str!("../../costmodel/fixtures/two_gpu_cpu_nvme.toml");
const WAN: &str = include_str!("../../costmodel/fixtures/wan_two_node.toml");
const BOX3090: &str = include_str!("../../analysis/fixtures/rtx3090_box.toml");
const PRIOR3090: &str = include_str!("../fixtures/rtx3090_prior.toml");
const SMALL_MOE: &str = include_str!("../../plan/fixtures/model_small_moe.toml");
const SMALL_MOE_PLAN: &str = include_str!("../../plan/fixtures/plan_small_moe_phase2.toml");

fn fixtures() -> Vec<(&'static str, HardwareGraph)> {
    [
        ("two_gpu_cpu_nvme", TWO_GPU),
        ("wan_two_node", WAN),
        ("rtx3090_box", BOX3090),
        ("rtx3090_prior", PRIOR3090),
    ]
    .iter()
    .map(|(n, t)| (*n, HardwareGraph::from_toml(t).unwrap()))
    .collect()
}

fn size(params: u64) -> UnitSize {
    UnitSize {
        params,
        flops_per_token: 2.0 * params as f64,
        source_format: Format::F16,
    }
}

fn dense(layers: u32, attn: u64, mlp: u64, io: u64) -> ModelDesc {
    ModelDesc {
        name: "dense".into(),
        layers,
        heads: 32,
        kv_heads: 8,
        experts: 0,
        experts_used: 0,
        embed: size(io),
        attn: size(attn),
        mlp: size(mlp),
        expert: None,
        head: size(io),
        kv_bytes_per_token_per_layer: 4096,
        activation_bytes_per_token: 8192,
    }
}

fn dense_small() -> ModelDesc {
    dense(8, 20_000_000, 60_000_000, 50_000_000)
}

/// 57 GB at F16: does not fit any single GPU of the fixtures.
fn dense_big() -> ModelDesc {
    dense(40, 200_000_000, 500_000_000, 200_000_000)
}

fn moe_small() -> ModelDesc {
    toml::from_str(SMALL_MOE).unwrap()
}

fn models() -> Vec<(&'static str, ModelDesc)> {
    vec![
        ("dense_small", dense_small()),
        ("moe_small", moe_small()),
        ("dense_big", dense_big()),
    ]
}

struct Env {
    hw: HardwareGraph,
    model: ModelDesc,
    hints: Hints,
    q: Box<dyn QualityModel>,
}

impl Env {
    fn new(hw: &HardwareGraph, model: &ModelDesc) -> Self {
        Self {
            hw: hw.clone(),
            model: model.clone(),
            hints: Hints::default(),
            q: Box::new(BitsQuality::new(model, 1.0)),
        }
    }

    /// Every unit keeps its source format: all plans tie on quality and speed decides, so
    /// hardware changes show in the chosen plan's speed.
    fn null(hw: &HardwareGraph, model: &ModelDesc) -> Self {
        Self {
            q: Box::new(NullQuality::new(model)),
            ..Self::new(hw, model)
        }
    }

    fn input(&self) -> SearchInput<'_> {
        SearchInput {
            hw: &self.hw,
            model: &self.model,
            hints: &self.hints,
            phase: Phase(3),
            workload: cli::WORKLOAD,
            quality: self.q.as_ref(),
            reserve: cli::default_reserve(&self.hw),
            routing: None,
            linux: true,
        }
    }

    fn solve(&self) -> (SearchResult, Explanation) {
        let inp = self.input();
        let res = search(&inp);
        let e = explain(&inp, &res);
        (res, e)
    }
}

// ------------------------------------------------------------------------------- explain

#[test]
fn explain_names_a_binding_resource_and_a_runner_up_on_every_fixture() {
    let mut checked = 0;
    for (hn, hw) in fixtures() {
        for (mn, m) in models() {
            let env = Env::new(&hw, &m);
            let (res, e) = env.solve();
            if res.choice.chosen.is_none() {
                continue;
            }
            let c = e.chosen.as_ref().unwrap();
            assert!(c.tok_s > 0.0 && c.ttft_s > 0.0, "{hn} {mn}");
            assert!(e.feasible && e.failing.is_empty(), "{hn} {mn}");
            // stage and device shares
            assert_eq!(e.stages.len(), res.scored[c.index].stage_times.len());
            let total: f64 = e.stages.iter().map(|s| s.share).sum();
            assert!((total - 1.0).abs() < 1e-9, "{hn} {mn}: {total}");
            assert!(!e.devices.is_empty(), "{hn} {mn}");
            let b = e.binding.as_ref().unwrap_or_else(|| panic!("{hn} {mn}"));
            assert!(b.share > 0.0 && b.share <= 1.0 + 1e-9);
            let max_dev = e.devices.iter().map(|d| d.share).fold(0.0, f64::max);
            let max_link = e.links.iter().map(|l| l.share).fold(0.0, f64::max);
            assert!((b.share - max_dev.max(max_link)).abs() < 1e-12, "{hn} {mn}");
            // formats cover the model
            assert!(!e.formats.is_empty(), "{hn} {mn}");
            let units: u32 = e.formats.iter().map(|f| f.units).sum();
            assert_eq!(units as usize, m.units().count(), "{hn} {mn}");
            let loss: f64 = e.formats.iter().map(|f| f.quality_loss).sum();
            assert!(
                (loss + e.base_kl - c.quality_loss).abs() < 1e-9,
                "{hn} {mn}: {loss} {} {}",
                e.base_kl,
                c.quality_loss
            );
            // front has the chosen point marked
            assert_eq!(e.front.iter().filter(|p| p.chosen).count(), 1, "{hn} {mn}");
            // runner-up with a reason and both numbers
            if res.scored.len() >= 2 {
                let r = e.runner_up.as_ref().unwrap_or_else(|| panic!("{hn} {mn}"));
                assert_ne!(r.plan.index, c.index);
                assert!(!r.reason.is_empty());
                match r.lost_on {
                    Term::Quality | Term::Cost => assert!(r.runner_value > r.chosen_value),
                    Term::Speed => assert!(r.runner_value <= r.chosen_value),
                    t => panic!("{hn} {mn}: unexpected {t:?}"),
                }
                checked += 1;
            }
            let text = e.to_text();
            assert!(
                text.contains("binding:") && text.contains("front"),
                "{text}"
            );
        }
    }
    assert!(checked >= 8, "only {checked} combinations had a runner-up");
}

#[test]
fn infeasible_limits_are_listed_and_the_runner_up_lost_on_a_limit() {
    let hw = HardwareGraph::from_toml(BOX3090).unwrap();
    let mut env = Env::new(&hw, &dense_small());
    env.hints.target_tok_s = Some(1.0e9);
    let (res, e) = env.solve();
    assert!(!res.choice.feasible && !e.feasible);
    assert!(e.failing.iter().any(|f| f.starts_with("target_tok_s")));
    let r = e.runner_up.clone().expect("a runner-up exists");
    assert_eq!(r.lost_on, Term::Limit);
    assert!(r.reason.contains("target_tok_s"));
    assert!(e.to_text().contains("limits not met"));
}

#[test]
fn a_limit_failing_plan_is_the_runner_up_when_nothing_else_is_feasible() {
    let hw = HardwareGraph::from_toml(BOX3090).unwrap();
    let mut env = Env::new(&hw, &dense_small());
    let (res, _) = env.solve();
    // A target only the fastest plan reaches.
    let fastest = res.scored.iter().map(|s| s.tok_s).fold(0.0, f64::max);
    let second = res
        .scored
        .iter()
        .map(|s| s.tok_s)
        .filter(|t| *t < fastest)
        .fold(0.0, f64::max);
    assert!(second > 0.0);
    let target = (fastest + second) / 2.0;
    env.hints.target_tok_s = Some(target);
    let (res, e) = env.solve();
    assert!(res.choice.feasible);
    let reaching = res.scored.iter().filter(|s| s.tok_s >= target).count();
    let r = e.runner_up.unwrap();
    if reaching == 1 {
        assert_eq!(r.lost_on, Term::Limit);
    } else {
        assert_ne!(r.lost_on, Term::Limit);
    }
}

#[test]
fn explanation_roundtrips_through_json() {
    let hw = HardwareGraph::from_toml(TWO_GPU).unwrap();
    let (_, e) = Env::new(&hw, &moe_small()).solve();
    // serde_json parses floats within an ulp, so compare the structure
    let back: Explanation = serde_json::from_str(&serde_json::to_string(&e).unwrap()).unwrap();
    assert_eq!(
        back.chosen.as_ref().unwrap().index,
        e.chosen.as_ref().unwrap().index
    );
    assert_eq!(
        back.binding.as_ref().unwrap().resource,
        e.binding.as_ref().unwrap().resource
    );
    assert_eq!(back.formats.len(), e.formats.len());
    assert_eq!(back.front.len(), e.front.len());
    assert_eq!(back.runner_up.unwrap().reason, e.runner_up.unwrap().reason);
}

// ---------------------------------------------------------------------------------- diff

fn small_moe_plan() -> Plan {
    Plan::from_toml(SMALL_MOE_PLAN).unwrap()
}

/// The fixture plan with experts `[k, 8)` of stage 1 on the NVMe instead of `[4, 8)`.
fn nvme_from(k: u32) -> Plan {
    let mut p = small_moe_plan();
    let a = &mut p.replicas[0].stages[1].assign;
    a[1].units.experts = Some(Span(0, k));
    a[2].units.experts = Some(Span(k, 8));
    p
}

fn two_gpu_env() -> (HardwareGraph, ModelDesc) {
    (HardwareGraph::from_toml(TWO_GPU).unwrap(), moe_small())
}

#[test]
fn the_diff_of_a_plan_with_itself_is_empty() {
    let (hw, m) = two_gpu_env();
    let p = small_moe_plan();
    let s = plan_diff(&p, &p, &hw, &m).unwrap();
    assert!(s.empty && !s.knobs_only && s.moves.is_empty());
    assert_eq!((s.moved_bytes, s.switch_s), (0, 0.0));
    // and for every plan the search finds
    for (hn, hw) in fixtures() {
        let env = Env::new(&hw, &dense_small());
        let (res, _) = env.solve();
        for sc in res.scored.iter().take(3) {
            let s = plan_diff(&sc.plan, &sc.plan, &hw, &env.model).unwrap();
            assert!(s.empty && s.switch_s == 0.0, "{hn}");
        }
    }
}

#[test]
fn a_knobs_only_change_is_flagged_and_costs_no_reload() {
    let (hw, m) = two_gpu_env();
    let a = small_moe_plan();
    let mut b = a.clone();
    b.knobs.prefill_micro_batches = 2;
    let s = plan_diff(&a, &b, &hw, &m).unwrap();
    assert!(s.knobs_only && !s.empty);
    assert_eq!((s.moved_bytes, s.switch_s, s.moves.len()), (0, 0.0, 0));
    let (from, to) = s.knobs.unwrap();
    assert_eq!(
        (from.prefill_micro_batches, to.prefill_micro_batches),
        (4, 2)
    );
    // moving shards is not knobs-only
    let s = plan_diff(&a, &nvme_from(3), &hw, &m).unwrap();
    assert!(!s.knobs_only && !s.empty);
    assert!(s.switch_s >= DRAIN_S + RELOAD_S);
}

#[test]
fn switch_time_grows_with_the_bytes_moved() {
    let (hw, m) = two_gpu_env();
    let base = nvme_from(4);
    let mut last = (0u64, 0.0f64);
    for k in [3, 2, 1] {
        let s = plan_diff(&base, &nvme_from(k), &hw, &m).unwrap();
        assert!(s.moved_bytes > last.0, "k={k}");
        assert!(s.switch_s > last.1, "k={k}: {} <= {}", s.switch_s, last.1);
        assert_eq!(s.by_destination.values().sum::<u64>(), s.moved_bytes);
        assert!(s.bottleneck.is_some());
        last = (s.moved_bytes, s.switch_s);
    }
    assert!(last.1 > DRAIN_S + RELOAD_S);
}

#[test]
fn moves_name_units_destinations_and_bytes() {
    let (hw, m) = two_gpu_env();
    let s = plan_diff(&nvme_from(4), &nvme_from(2), &hw, &m).unwrap();
    // experts 2 and 3 of layers 2 and 3 leave gpu1 for the NVMe, converted to the NVMe format
    assert_eq!(s.moves.len(), 4);
    assert!(s.moves.iter().all(|mv| {
        mv.to.as_deref() == Some("nvme0") && mv.from.as_deref() == Some("gpu1") && mv.bytes > 0
    }));
    assert_eq!(s.freed_bytes, 0);
    let t = s.to_text();
    assert!(t.contains("nvme0") && t.contains("switch"), "{t}");
}

#[test]
fn diffs_between_searched_plans_are_consistent() {
    for (hn, hw) in fixtures() {
        let env = Env::new(&hw, &moe_small());
        let (res, _) = env.solve();
        let Some(c) = res.choice.chosen else { continue };
        for sc in res.scored.iter().take(6) {
            let s = plan_diff(&sc.plan, &res.scored[c].plan, &hw, &env.model).unwrap();
            assert_eq!(
                s.by_destination.values().sum::<u64>(),
                s.moved_bytes,
                "{hn}"
            );
            assert_eq!(s.moves.iter().map(|m| m.bytes).sum::<u64>(), s.moved_bytes);
            assert!(s.switch_s.is_finite() && s.switch_s >= 0.0, "{hn}");
            if s.empty {
                assert_eq!(s.switch_s, 0.0);
            }
        }
    }
}

// -------------------------------------------------------------------------------- what-if

#[test]
fn priors_cover_the_required_classes_and_are_priors() {
    let p = Priors::builtin();
    for c in ["3090", "4090", "4060ti16", "a100"] {
        let g = p.gpu_class(c).unwrap_or_else(|| panic!("{c}"));
        assert!(g.capacity_bytes > 0 && g.bandwidth_bytes_per_s > 0.0 && g.flops_per_s > 0.0);
        assert!(g.c0_graph_s > 0.0 && g.pcie_gen >= 3 && g.pcie_width >= 8 && g.price_eur > 0.0);
    }
    assert!(p.ram_class("ddr4").is_some() && p.ram_class("DDR5").is_some());
    for c in ["gen3", "gen4", "gen5"] {
        assert!(p.nvme_class(c).is_some());
    }
    assert!(p.net_class("10gbe").is_some() && p.net_class("25GbE").is_some());
    // devices built from the priors say so
    let hw = HardwareGraph::from_toml(BOX3090).unwrap();
    let after = tetraeder_planner::apply(&hw, &["gpu:4090".parse().unwrap()]).unwrap();
    let g = after.device("gpu1").unwrap();
    assert_eq!(g.bandwidth_bytes_per_s.unwrap().source, Source::Prior);
    let new_links = after.links.iter().filter(|l| l.id.starts_with("pcie-gpu1"));
    assert!(new_links.count() >= 2);
    assert!(
        after
            .links
            .iter()
            .filter(|l| l.id.starts_with("pcie-gpu1"))
            .all(|l| l.beta_bytes_per_s.source == Source::Prior)
    );
}

#[test]
fn changes_parse_and_print() {
    for (s, canon) in [
        ("gpu:4090", "gpu:rtx4090"),
        ("gpu:4060ti:2", "gpu:rtx4060ti16:2"),
        ("ram:256G", "ram:256G"),
        ("ram:128G:ddr5:8", "ram:128G:ddr5:8"),
        ("nvme:gen4", "nvme:gen4"),
        ("net:25GbE", "net:25gbe"),
        ("remove:gpu1", "remove:gpu1"),
    ] {
        let c: Change = s.parse().unwrap_or_else(|e| panic!("{s}: {e}"));
        assert_eq!(c.to_string(), canon);
        assert_eq!(canon.parse::<Change>().unwrap(), c);
    }
    for bad in [
        "gpu:9999",
        "ram:lots",
        "nvme",
        "disk:gen4",
        "gpu:4090:0",
        "remove:",
    ] {
        assert!(bad.parse::<Change>().is_err(), "{bad}");
    }
    assert!("gpu:4090".parse::<Change>().unwrap().price_eur().unwrap() > 0.0);
    assert!(
        "remove:gpu1"
            .parse::<Change>()
            .unwrap()
            .price_eur()
            .is_none()
    );
    let Change::Add(a) = "ram:256G".parse::<Change>().unwrap() else {
        panic!()
    };
    assert_eq!((a.kind, a.bytes), (Kind::Ram, Some(256 << 30)));
}

fn apply_str(hw: &HardwareGraph, changes: &[&str]) -> Result<HardwareGraph, String> {
    let cs: Vec<Change> = changes.iter().map(|c| c.parse().unwrap()).collect();
    tetraeder_planner::apply(hw, &cs)
}

#[test]
fn apply_edits_a_copy_of_the_graph() {
    let hw = HardwareGraph::from_toml(BOX3090).unwrap();
    // GPU: new device linked to the CPU and to the other GPUs
    let a = apply_str(&hw, &["gpu:4090", "gpu:3090"]).unwrap();
    assert_eq!(a.devices.len(), hw.devices.len() + 2);
    assert!(a.link_between("gpu1", "cpu0").is_some());
    assert!(a.link_between("gpu1", "gpu2").is_some() && a.link_between("gpu0", "gpu1").is_some());
    assert_eq!(a.device("gpu2").unwrap().index, Some(2));
    assert!(a.validate().is_ok());
    assert_eq!(hw.devices.len(), 3, "the input is untouched");
    // RAM adds capacity; with a class it sets the bandwidth
    let a = apply_str(&hw, &["ram:256G"]).unwrap();
    assert_eq!(
        a.device("cpu0").unwrap().capacity_bytes,
        64424509440 + (256 << 30)
    );
    let a = apply_str(&hw, &["ram:64G:ddr5:8"]).unwrap();
    assert_eq!(
        a.device("cpu0")
            .unwrap()
            .bandwidth_bytes_per_s
            .unwrap()
            .value,
        8.0 * 4.48e10
    );
    // NVMe stripes with the existing drive
    let a = apply_str(&hw, &["nvme:gen4"]).unwrap();
    let d = a.device("nvme0").unwrap();
    assert_eq!(d.bandwidth_bytes_per_s.unwrap().value, 3.0e9 + 7.0e9);
    assert_eq!(
        a.link_between("nvme0", "cpu0")
            .unwrap()
            .beta_bytes_per_s
            .value,
        10.0e9
    );
    assert!(a.lane_group("chipset").unwrap().capacity_bytes_per_s > 3.5e9);
    // without a disk it adds one
    let no_disk = apply_str(&hw, &["remove:nvme0", "nvme:gen5"]).unwrap();
    assert_eq!(no_disk.device("nvme0").unwrap().kind, DeviceKind::Disk);
    assert!(no_disk.link_between("nvme0", "cpu0").is_some());
    // removal drops links; errors
    let r = apply_str(&hw, &["remove:gpu0"]).unwrap();
    assert!(r.device("gpu0").is_none() && r.links.iter().all(|l| l.a != "gpu0" && l.b != "gpu0"));
    assert!(apply_str(&hw, &["remove:gpu9"]).is_err());
    assert!(
        apply_str(&hw, &["net:25gbe"]).is_err(),
        "no network link on a single box"
    );
    let wan = HardwareGraph::from_toml(WAN).unwrap();
    let up = apply_str(&wan, &["net:25gbe"]).unwrap();
    assert!(
        up.links
            .iter()
            .filter(|l| l.kind == tetraeder_profiler::LinkKind::Network)
            .all(|l| l.beta_bytes_per_s.value == 3.125e9)
    );
}

#[test]
fn a_new_device_of_a_measured_class_gets_the_measured_ratio() {
    let mut hw = HardwareGraph::from_toml(BOX3090).unwrap();
    let spec = Priors::builtin()
        .gpu_class("3090")
        .unwrap()
        .bandwidth_bytes_per_s;
    // not measured: spec as is
    let a = apply_str(&hw, &["gpu:3090"]).unwrap();
    assert_eq!(
        a.device("gpu1")
            .unwrap()
            .bandwidth_bytes_per_s
            .unwrap()
            .value,
        spec
    );
    // measured at 90% of spec on gpu0 (the fixture name says RTX 3090)
    let gpu = hw.devices.iter_mut().find(|d| d.id == "gpu0").unwrap();
    gpu.bandwidth_bytes_per_s = Measurement::from_samples(&[0.9 * spec]);
    let a = apply_str(&hw, &["gpu:3090", "gpu:4090"]).unwrap();
    let new = a.device("gpu1").unwrap().bandwidth_bytes_per_s.unwrap();
    assert!((new.value / spec - 0.9).abs() < 1e-9);
    assert_eq!(new.source, Source::Prior);
    assert!(a.notes.iter().any(|n| n.contains("scaled by 0.900")));
    // another class is not scaled
    let other = a
        .device("gpu2")
        .unwrap()
        .bandwidth_bytes_per_s
        .unwrap()
        .value;
    assert_eq!(
        other,
        Priors::builtin()
            .gpu_class("4090")
            .unwrap()
            .bandwidth_bytes_per_s
    );
}

fn what(hw: &HardwareGraph, m: &ModelDesc, changes: &[&str]) -> tetraeder_planner::WhatIf {
    let env = Env::null(hw, m);
    let cs: Vec<Change> = changes.iter().map(|c| c.parse().unwrap()).collect();
    what_if(&env.input(), &cs).unwrap()
}

#[test]
fn a_second_gpu_raises_the_speed_when_the_model_does_not_fit() {
    for (name, hw) in [
        ("rtx3090_box", HardwareGraph::from_toml(BOX3090).unwrap()),
        (
            "two_gpu_cpu_nvme",
            HardwareGraph::from_toml(TWO_GPU).unwrap(),
        ),
    ] {
        let w = what(&hw, &dense_big(), &["gpu:a100"]);
        assert!(w.before.tok_s > 0.0, "{name}");
        assert!(w.after.tok_s > w.before.tok_s, "{name}: {}", w.to_text());
        assert!(w.gain.tok_s > 0.0 && w.gain.ratio.unwrap() > 1.0);
        assert_eq!(w.changes, vec!["gpu:a100_40"]);
    }
}

#[test]
fn removing_the_only_gpu_lowers_the_speed() {
    let hw = HardwareGraph::from_toml(BOX3090).unwrap();
    for m in [dense_small(), dense_big()] {
        let w = what(&hw, &m, &["remove:gpu0"]);
        assert!(w.after.tok_s > 0.0);
        assert!(w.after.tok_s < w.before.tok_s, "{}", w.to_text());
        assert!(w.gain.tok_s < 0.0);
    }
}

#[test]
fn what_if_drops_hints_that_name_a_removed_device() {
    let hw = HardwareGraph::from_toml(BOX3090).unwrap();
    let mut env = Env::null(&hw, &dense_small());
    env.hints.pin.push(Pin {
        units: UnitSel::default(),
        device: "gpu0".into(),
    });
    env.hints.cost.insert("gpu0".into(), 3.0);
    let w = what_if(&env.input(), &["remove:gpu0".parse().unwrap()]).unwrap();
    assert!(w.after.tok_s > 0.0 && w.after.tok_s < w.before.tok_s);
    assert!(what_if(&env.input(), &["remove:nope".parse().unwrap()]).is_err());
}

#[test]
fn more_ram_never_lowers_the_speed() {
    let hw = HardwareGraph::from_toml(BOX3090).unwrap();
    let w = what(&hw, &dense_big(), &["ram:256G"]);
    assert!(w.after.tok_s >= w.before.tok_s * 0.999, "{}", w.to_text());
}

#[test]
fn rank_purchases_is_deterministic_and_sorted() {
    let hw = HardwareGraph::from_toml(BOX3090).unwrap();
    let env = Env::null(&hw, &dense_big());
    let cands = default_candidates();
    let a = rank_purchases(&env.input(), &cands);
    let b = rank_purchases(&env.input(), &cands);
    assert_eq!(a, b);
    assert_eq!(a.ranked.len() + a.skipped.len(), cands.len());
    assert!(a.skipped.iter().any(|(c, _)| c == "net:25gbe"));
    assert!(
        a.ranked
            .windows(2)
            .all(|w| w[0].gain_tok_s >= w[1].gain_tok_s)
    );
    let gpu = a.ranked.iter().find(|p| p.change == "gpu:rtx4090").unwrap();
    assert!(gpu.gain_tok_s > 0.0);
    assert!(gpu.price_eur.unwrap() > 0.0 && gpu.gain_per_1000_eur.unwrap() > 0.0);
    // the order of the candidates does not change the ranking
    let mut rev = cands.clone();
    rev.reverse();
    assert_eq!(rank_purchases(&env.input(), &rev).ranked, a.ranked);
}

// ----------------------------------------------------------------------------- CLI

#[test]
fn explain_cmd_reports_text_and_json() {
    let model = toml::to_string(&dense_small()).unwrap();
    let text = cli::explain_cmd(BOX3090, &model, None, 3, true, false).unwrap();
    assert!(
        text.contains("binding:") && text.contains("runner-up:"),
        "{text}"
    );
    let json = cli::explain_cmd(BOX3090, &model, None, 3, true, true).unwrap();
    let e: Explanation = serde_json::from_str(&json).unwrap();
    assert!(e.binding.is_some() && e.runner_up.is_some());
    let h = Hints {
        target_tok_s: Some(1.0e9),
        ..Default::default()
    };
    let text =
        cli::explain_cmd(BOX3090, &model, Some(&h.to_toml().unwrap()), 3, true, false).unwrap();
    assert!(text.contains("limits not met"), "{text}");
    assert!(cli::explain_cmd("devices = 3", &model, None, 3, true, false).is_err());
    assert!(cli::explain_cmd(BOX3090, "x", None, 3, true, false).is_err());
}

#[test]
fn diff_cmd_reports_text_and_json() {
    let a = SMALL_MOE_PLAN;
    let b = nvme_from(2).to_toml().unwrap();
    let t = cli::diff_cmd(TWO_GPU, SMALL_MOE, a, &b, false).unwrap();
    assert!(t.contains("switch"), "{t}");
    let j = cli::diff_cmd(TWO_GPU, SMALL_MOE, a, &b, true).unwrap();
    let s: tetraeder_planner::PlanSwitch = serde_json::from_str(&j).unwrap();
    assert!(!s.empty && s.moved_bytes > 0);
    assert_eq!(
        cli::diff_cmd(TWO_GPU, SMALL_MOE, a, a, false).unwrap(),
        "no change\n"
    );
    assert!(cli::diff_cmd(TWO_GPU, SMALL_MOE, a, "model = 3", false).is_err());
}

#[test]
fn what_if_cmd_reports_changes_and_a_ranking() {
    let model = toml::to_string(&dense_big()).unwrap();
    let s = |x: &[&str]| x.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let t = cli::what_if_cmd(
        BOX3090,
        &model,
        None,
        3,
        true,
        &s(&["gpu:4090", "ram:256G"]),
        &[],
        false,
    )
    .unwrap();
    assert!(t.contains("gain:") && t.contains("gpu:rtx4090"), "{t}");
    let small = toml::to_string(&dense_small()).unwrap();
    let j = cli::what_if_cmd(BOX3090, &small, None, 3, true, &[], &s(&["gpu0"]), true).unwrap();
    let w: tetraeder_planner::WhatIf = serde_json::from_str(&j).unwrap();
    assert!(w.gain.tok_s < 0.0);
    let r = cli::what_if_cmd(BOX3090, &model, None, 3, true, &[], &[], false).unwrap();
    assert!(
        r.contains("gpu:rtx4090") && r.contains("per 1000 EUR"),
        "{r}"
    );
    assert!(
        cli::what_if_cmd(
            BOX3090,
            &model,
            None,
            3,
            true,
            &s(&["gpu:nope"]),
            &[],
            false
        )
        .is_err()
    );
}
