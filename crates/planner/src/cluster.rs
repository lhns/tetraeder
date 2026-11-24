//! Plans across nodes (ADR 0091, build step 2): which layers each node runs, priced with
//! today's single-node model plus the network hops. Pure, offline, deterministic.
//!
//! A cluster candidate is one replica whose stages sit on different nodes, one stage per node,
//! each a builder placement of its node (ADR 0035: dense weights and VRAM expert slots on the
//! GPU, RAM expert slots on the host, the experts' backing store on the node's disk). Every
//! candidate computes on every usable node; the single-node plan is only a reference
//! ([`single_node_references`]), never judged (ADR 0091, Case S).
//!
//! Per segment the planner slices the model layout (its layers, the embedding only in the first
//! segment, the output only where the head computes) and runs today's builder footprint and
//! feature functions on the node's own graph, so a plan on one node predicts as before.

use crate::builder::{feasible_vram_tiers, ram_slot_options};
use crate::candidate::{
    BUILDER_IO, Budget, CPU, CUDA_RESERVE, GPU, HOST_RESERVE, Placement, Workload, footprint,
};
use crate::layout::{LayerBytes, ModelLayout};
use crate::score::{BUILDER_DISK, READER_DEPTH, builder_features, builder_features_tier, features};
use crate::serve::uniform_tier;
use std::collections::BTreeMap;
use tetraeder_costmodel::cluster::{
    ClusterFeatures, ClusterPrediction, SegmentFeatures, predict_cluster,
};
use tetraeder_costmodel::predict::{ReadPath, RunFeatures};
use tetraeder_plan::{
    Assign, Context, DeviceBytes, Hints, Knobs, KvFormat, Phase, Plan, Replica, Segment, SlotCache,
    Span, Stage, UnitKind, UnitSel, segment_plan, segments, validate,
};
use tetraeder_profiler::{DeviceKind, HardwareGraph};

/// Candidates kept after thinning (the best of every order and head placement stay).
pub const MAX_CLUSTER_CANDIDATES: usize = 40;
/// Prefill micro-batches the planner may pick for a plan across nodes (ADR 0077).
pub const MICRO_BATCHES: [u32; 2] = [1, 4];
/// Physical batches of cluster candidates (as the builder's, ADR 0040).
pub const CLUSTER_UBATCHES: [u32; 2] = [256, 2048];

/// One plan across nodes with its analytic prediction (cost model 5, uncalibrated).
#[derive(Debug, Clone)]
pub struct ClusterCandidate {
    pub id: String,
    pub plan: Plan,
    pub notes: Vec<String>,
    pub prediction: ClusterPrediction,
}

/// Key of a cached segment choice: node index, first and last layer, embed, head, ubatch (plus
/// one when the leader's GPU also holds the head's worker).
type ChoiceKey = (usize, u32, u32, bool, bool, u32);
/// A plan found: request seconds, order of the nodes, head placement, plan, prediction, note.
type Found = (f64, Vec<usize>, HeadAt, Plan, ClusterPrediction, String);

/// What a segment runs on its node.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Choice {
    placement: Placement,
    request_s: f64,
}

/// The ids and capacities of one node, as the planner sees them.
struct Node {
    name: Option<String>,
    graph: HardwareGraph,
    /// Merged-graph ids: GPU, CPU and the disk the builder reads from.
    gpu: Option<String>,
    cpu: String,
    disk: Option<String>,
    gpu_bytes: u64,
    ram_bytes: u64,
    threads: u32,
}

fn node_of(hw: &HardwareGraph, name: Option<&str>) -> Result<Node, String> {
    let label = name.unwrap_or("local");
    let cpu = hw
        .device_in_node(name, CPU)
        .filter(|d| d.kind == DeviceKind::Cpu)
        .ok_or_else(|| format!("node {label} has no {CPU}"))?;
    let gpu = hw
        .device_in_node(name, GPU)
        .filter(|d| d.kind == DeviceKind::Gpu);
    let disk = hw
        .device_in_node(name, BUILDER_DISK)
        .filter(|d| d.kind == DeviceKind::Disk);
    Ok(Node {
        name: name.map(str::to_string),
        graph: hw.node_graph(name),
        gpu: gpu.map(|d| d.id.clone()),
        cpu: cpu.id.clone(),
        disk: disk.map(|d| d.id.clone()),
        gpu_bytes: gpu.map_or(0, |d| d.capacity_bytes),
        ram_bytes: cpu.capacity_bytes,
        threads: cpu.threads.unwrap_or(1).max(1),
    })
}

/// The layout of a segment: its layers, the embedding only when `embed`, the output head only
/// when `head`. The whole model with both is the layout itself.
pub fn segment_layout(layout: &ModelLayout, layers: Span, embed: bool, head: bool) -> ModelLayout {
    let all: &[LayerBytes] = &layout.layers;
    let part = all[layers.0 as usize..layers.1 as usize].to_vec();
    ModelLayout {
        n_layer: part.len() as u32,
        layers: part,
        embed_bytes: if embed { layout.embed_bytes } else { 0 },
        output_bytes: if head { layout.output_bytes } else { 0 },
        ..layout.clone()
    }
}

/// GPU bytes a head-only segment takes next to the leader's own stage: the output weights, the
/// logits buffer and the CUDA reserve of its worker.
fn head_only_gpu_bytes(layout: &ModelLayout, ubatch: u32) -> u64 {
    layout.output_bytes
        + u64::from(ubatch) * (4 * u64::from(layout.n_vocab) + 64 * u64::from(layout.n_embd))
        + CUDA_RESERVE
}

/// Run features of a segment of `sub` on its node graph (ids as the node's own graph names
/// them). A builder placement reads its misses from the node's disk when it has one, under the
/// uniform routing prior; a CPU placement is today's `features`.
fn run_features(
    node: &Node,
    sub: &ModelLayout,
    placement: Placement,
    ubatch: u32,
    threads: u32,
) -> RunFeatures {
    match placement {
        Placement::Builder {
            vram_slots,
            ram_slots,
        } => {
            let has_disk = node.disk.is_some()
                && node
                    .graph
                    .device(BUILDER_DISK)
                    .is_some_and(|d| d.disk_rate(1).is_some());
            let knobs = (threads, ubatch, true);
            if has_disk && !sub.layers.is_empty() {
                let o = &uniform_tier(sub, &[(vram_slots, ram_slots)])[0];
                builder_features_tier(
                    sub,
                    vram_slots,
                    o,
                    knobs,
                    ReadPath::Reader,
                    (READER_DEPTH, READER_DEPTH),
                    BUILDER_DISK,
                )
            } else {
                let hit = f64::from(vram_slots) / f64::from(sub.n_expert.max(1));
                builder_features(sub, vram_slots, hit, knobs)
            }
        }
        p => features(
            sub,
            Some(p),
            threads,
            &["-ub".to_string(), ubatch.to_string()],
            &BTreeMap::new(),
        ),
    }
}

/// The best placement of `sub` on `node` for `ubatch`, by predicted request time, with `extra`
/// GPU bytes already taken. `Err` names what does not fit.
fn choose(
    node: &Node,
    sub: &ModelLayout,
    ubatch: u32,
    extra_gpu: u64,
    w: &Workload,
) -> Result<Choice, String> {
    let label = node.name.as_deref().unwrap_or("local");
    let threads = node.threads;
    let time = |p: Placement| {
        let f = run_features(node, sub, p, ubatch, threads);
        tetraeder_costmodel::predict::predict(&node.graph, &f, w.prompt, w.r#gen)
            .map(|p| p.request_s)
            .map_err(|e| e.to_string())
    };
    let mut feasible: Vec<Placement> = Vec::new();
    if node.gpu.is_some() {
        // A margin of 1%: the validator sizes units by their kind's mean, the footprint by bytes.
        let budget = Budget {
            gpu_bytes: node.gpu_bytes.saturating_sub(extra_gpu) / 100 * 99,
            ram_bytes: node.ram_bytes / 100 * 99,
            threads,
        };
        let mut dropped = Vec::new();
        for v in feasible_vram_tiers(&budget, sub, w, &mut dropped) {
            for r in ram_slot_options(sub, v) {
                let p = Placement::Builder {
                    vram_slots: v,
                    ram_slots: r,
                };
                let f = footprint(sub, p, ubatch, w, KvFormat::default());
                if f.gpu <= budget.gpu_bytes && f.host <= budget.ram_bytes {
                    feasible.push(p);
                }
            }
        }
    }
    if feasible.is_empty() {
        let f = footprint(sub, Placement::Cpu, ubatch, w, KvFormat::default());
        if f.host <= node.ram_bytes / 100 * 99 {
            feasible.push(Placement::Cpu);
        }
    }
    let mut best: Option<Choice> = None;
    for p in feasible {
        let request_s = time(p)?;
        if best.is_none_or(|b| request_s < b.request_s) {
            best = Some(Choice {
                placement: p,
                request_s,
            });
        }
    }
    best.ok_or_else(|| {
        format!(
            "node {label}: layers {} do not fit {} B of VRAM and {} B of RAM",
            sub.n_layer,
            node.gpu_bytes.saturating_sub(extra_gpu),
            node.ram_bytes
        )
    })
}

fn format_of(layout: &ModelLayout) -> tetraeder_plan::Format {
    layout.main_format()
}

/// The stage of one segment: merged-graph device ids, experts backed by the node's disk when it
/// has one (the builder's experts live in the file), `head_here` puts the output on this stage.
#[allow(clippy::too_many_arguments)]
fn stage_of(
    node: &Node,
    layout: &ModelLayout,
    sub: &ModelLayout,
    layers: Span,
    embed: bool,
    head_here: bool,
    choice: Choice,
    ubatch: u32,
    w: &Workload,
) -> Stage {
    let fmt = format_of(layout);
    let f = footprint(sub, choice.placement, ubatch, w, KvFormat::default());
    let on = |ids: &[&String]| ids.iter().map(|s| (*s).clone()).collect::<Vec<_>>();
    let assign = |kinds: Vec<UnitKind>, home: Vec<String>, compute: Option<Vec<String>>| Assign {
        units: UnitSel {
            kinds,
            ..Default::default()
        },
        home: Some(home),
        compute,
        format: Some(fmt),
        ..Default::default()
    };
    let main = match choice.placement {
        Placement::Cpu => &node.cpu,
        _ => node.gpu.as_ref().unwrap_or(&node.cpu),
    };
    let mut rules = Vec::new();
    if embed {
        rules.push(assign(vec![UnitKind::Embed], on(&[&node.cpu]), None));
    }
    let mut dense = vec![UnitKind::Attn, UnitKind::Mlp];
    if head_here {
        dense.push(UnitKind::Head);
    }
    rules.push(assign(dense, on(&[main]), None));
    if layout.is_moe() {
        let (home, compute) = match (&node.disk, choice.placement) {
            (Some(d), Placement::Builder { .. }) => (d, Some(on(&[&node.cpu]))),
            _ => (&node.cpu, None),
        };
        let kinds = vec![UnitKind::Expert];
        rules.push(assign(kinds, on(&[home]), compute));
    }
    let (devices, kv, workspace, expert_cache) = match (&node.gpu, choice.placement) {
        (
            Some(g),
            Placement::Builder {
                vram_slots,
                ram_slots,
            },
        ) => (
            on(&[g, &node.cpu]),
            vec![DeviceBytes {
                device: g.clone(),
                bytes: f.kv_gpu,
            }],
            vec![DeviceBytes {
                device: g.clone(),
                bytes: f.compute_gpu,
            }],
            vec![
                SlotCache {
                    device: g.clone(),
                    slots: vram_slots,
                },
                SlotCache {
                    device: node.cpu.clone(),
                    slots: ram_slots,
                },
            ],
        ),
        _ => (
            on(&[&node.cpu]),
            vec![DeviceBytes {
                device: node.cpu.clone(),
                bytes: f.kv_host,
            }],
            Vec::new(),
            Vec::new(),
        ),
    };
    Stage {
        layers,
        devices,
        tp: Some(1),
        ep: Some(1),
        assign: rules,
        kv: Some(kv),
        workspace,
        expert_cache,
        ..Default::default()
    }
}

/// Where the head unit computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HeadAt {
    /// On the last stage's node: the logits row crosses back to the leader.
    Last,
    /// On the leader's node, in its own worker: the hidden rows cross back.
    Leader,
}

/// A plan from an order of nodes and the first layer of every stage but the first.
#[allow(clippy::too_many_arguments)]
fn assemble(
    nodes: &[&Node],
    layout: &ModelLayout,
    cuts: &[u32],
    head: HeadAt,
    ubatch: u32,
    micro: u32,
    choices: &[Choice],
    w: &Workload,
) -> Plan {
    let m = nodes.len();
    let mut bounds = vec![0];
    bounds.extend_from_slice(cuts);
    bounds.push(layout.n_layer);
    let mut stages = Vec::with_capacity(m);
    for j in 0..m {
        let span = Span(bounds[j], bounds[j + 1]);
        let last = j + 1 == m;
        let head_here = last && head == HeadAt::Last;
        let sub = segment_layout(layout, span, j == 0, head_here);
        let mut st = stage_of(
            nodes[j],
            layout,
            &sub,
            span,
            j == 0,
            head_here,
            choices[j],
            ubatch,
            w,
        );
        if last && head == HeadAt::Leader {
            let leader = nodes[0];
            let main = leader.gpu.as_ref().unwrap_or(&leader.cpu).clone();
            st.assign.push(Assign {
                units: UnitSel {
                    kinds: vec![UnitKind::Head],
                    ..Default::default()
                },
                home: Some(vec![main.clone()]),
                compute: Some(vec![main]),
                format: Some(format_of(layout)),
                ..Default::default()
            });
        }
        stages.push(st);
    }
    Plan {
        model: layout.model.clone(),
        knobs: Knobs {
            prefill_micro_batches: micro,
            ubatch: Some(ubatch),
            op_offload: Some(true),
            ..Default::default()
        },
        replicas: vec![Replica { stages }],
    }
}

/// Placement and run settings of one segment, read back from its single-node plan.
fn read_stage(plan: &Plan) -> Result<Placement, String> {
    let st = &plan.replicas[0].stages[0];
    let slots = |id: &str| {
        st.expert_cache
            .iter()
            .find(|c| c.device == id)
            .map(|c| c.slots)
    };
    match (slots(GPU), slots(CPU)) {
        (Some(vram_slots), Some(ram_slots)) => Ok(Placement::Builder {
            vram_slots,
            ram_slots,
        }),
        (None, None) if !st.devices.iter().any(|d| d == GPU) => Ok(Placement::Cpu),
        _ => Err("a segment is a builder placement or all on the CPU".to_string()),
    }
}

/// The run features of every segment of a plan's replica 0 and the pipeline knobs.
pub fn cluster_features(
    hw: &HardwareGraph,
    layout: &ModelLayout,
    plan: &Plan,
) -> Result<ClusterFeatures, String> {
    let segs = segments(plan, 0, hw).map_err(|e| e.join("; "))?;
    let ubatch = plan.knobs.ubatch.unwrap_or(512);
    let mut out = Vec::with_capacity(segs.len());
    for s in &segs {
        let node = node_of(hw, s.node.as_deref())?;
        let sub = segment_layout(layout, s.layers, s.embed, s.head);
        let threads = plan.knobs.threads.unwrap_or(node.threads);
        let run = if s.stages.is_empty() {
            // The head alone: the output matmul on the GPU of the node.
            run_features(
                &node,
                &sub,
                Placement::Builder {
                    vram_slots: 0,
                    ram_slots: 0,
                },
                ubatch,
                threads,
            )
        } else if s.stages.len() > 1 {
            return Err(format!(
                "segment on {:?} has {} stages; step 2 prices one stage per node",
                s.node,
                s.stages.len()
            ));
        } else {
            let sp = segment_plan(plan, 0, s);
            run_features(&node, &sub, read_stage(&sp)?, ubatch, threads)
        };
        out.push(SegmentFeatures {
            node: s.node.clone(),
            run,
            embed: s.embed,
            head: s.head,
        });
    }
    Ok(ClusterFeatures {
        segments: out,
        n_embd: layout.n_embd,
        n_vocab: layout.n_vocab,
        prefill_micro_batches: plan.knobs.prefill_micro_batches,
    })
}

/// The cost model's prediction for a plan across nodes.
pub fn predict_cluster_plan(
    hw: &HardwareGraph,
    layout: &ModelLayout,
    plan: &Plan,
    w: &Workload,
) -> Result<ClusterPrediction, String> {
    let f = cluster_features(hw, layout, plan)?;
    predict_cluster(hw, &f, w.prompt, w.r#gen).map_err(|e| e.to_string())
}

fn reserve_of(hw: &HardwareGraph) -> BTreeMap<String, u64> {
    hw.devices
        .iter()
        .map(|d| {
            let r = match d.kind {
                DeviceKind::Gpu => CUDA_RESERVE,
                DeviceKind::Cpu => HOST_RESERVE + BUILDER_IO,
                DeviceKind::Disk => 0,
            };
            (d.id.clone(), r)
        })
        .collect()
}

fn permutations(items: &[usize]) -> Vec<Vec<usize>> {
    if items.len() <= 1 {
        return vec![items.to_vec()];
    }
    let mut out = Vec::new();
    for (i, &x) in items.iter().enumerate() {
        let mut rest = items.to_vec();
        rest.remove(i);
        for mut p in permutations(&rest) {
            p.insert(0, x);
            out.push(p);
        }
    }
    out
}

/// Strictly increasing cut vectors of `m - 1` points in `1..layers` on a grid of about `grid`
/// points per cut.
fn cut_vectors(layers: u32, m: usize, grid: u32) -> Vec<Vec<u32>> {
    let step = (layers / grid.max(1)).max(1);
    let points: Vec<u32> = (1..layers).filter(|x| x % step == 0).collect();
    let mut out: Vec<Vec<u32>> = vec![Vec::new()];
    for _ in 1..m {
        let mut next = Vec::new();
        for v in &out {
            let from = v.last().copied().unwrap_or(0);
            for &p in points.iter().filter(|&&p| p > from) {
                let mut w = v.clone();
                w.push(p);
                next.push(w);
            }
        }
        out = next;
    }
    out
}

/// The usable nodes of `hw` and why the others are not.
fn usable_nodes(
    hw: &HardwareGraph,
    hints: &Hints,
    pruned: &mut Vec<(String, String)>,
) -> Vec<Node> {
    let mut out = Vec::new();
    for n in hw.nodes() {
        let label = n.unwrap_or("local").to_string();
        if hints
            .forbid
            .iter()
            .any(|f| f.node.as_deref() == n && n.is_some() && f.units.is_none())
        {
            pruned.push((format!("node {label}"), "forbid node".into()));
            continue;
        }
        match node_of(hw, n) {
            Ok(node) => out.push(node),
            Err(e) => pruned.push((format!("node {label}"), e)),
        }
    }
    out
}

/// Candidate plans that compute on every usable node, best predicted first (ids `c00`...), and
/// what was pruned with the reason. One replica, one stage per node, in every order of the nodes
/// with the head unit on the last stage or on the leader, cut points on a grid, the best
/// placement per segment, micro-batches chosen by the prediction; every plan passes
/// `tetraeder_plan::validate` for `phase` and `hints`. MoE models only (open: dense segments).
/// The workload is `hints.workload.typical_prompt` (2048) and 128 generated tokens.
pub fn enumerate_cluster(
    hw: &HardwareGraph,
    layout: &ModelLayout,
    hints: &Hints,
    phase: Phase,
) -> (Vec<ClusterCandidate>, Vec<(String, String)>) {
    let w = Workload {
        prompt: hints
            .workload
            .as_ref()
            .and_then(|x| x.typical_prompt)
            .unwrap_or(2048),
        r#gen: 128,
    };
    let mut pruned = Vec::new();
    if !layout.is_moe() {
        pruned.push((
            "all plans".into(),
            "the cluster planner prices MoE builder segments only".into(),
        ));
        return (Vec::new(), pruned);
    }
    let nodes = usable_nodes(hw, hints, &mut pruned);
    let m = nodes.len();
    if m < 2 || layout.n_layer < m as u32 {
        pruned.push((
            "all plans".into(),
            format!(
                "{m} usable node(s) for {} layers: a plan must compute on every node; \
                 the single-node plan is a reference only",
                layout.n_layer
            ),
        ));
        return (Vec::new(), pruned);
    }
    let grid = if m == 2 { 47 } else { 8 };
    let cuts = cut_vectors(layout.n_layer, m, grid);
    let order: Vec<usize> = (0..m).collect();
    let mut cache: BTreeMap<ChoiceKey, Result<Choice, String>> = BTreeMap::new();
    let mut found: Vec<Found> = Vec::new();
    for perm in permutations(&order) {
        let ns: Vec<&Node> = perm.iter().map(|&i| &nodes[i]).collect();
        let names: Vec<String> = ns
            .iter()
            .map(|n| n.name.clone().unwrap_or_else(|| "local".into()))
            .collect();
        let tag = names.join(">");
        for head in [HeadAt::Last, HeadAt::Leader] {
            if head == HeadAt::Leader && ns[0].gpu.is_none() {
                continue;
            }
            for &ubatch in &CLUSTER_UBATCHES {
                'cut: for cut in &cuts {
                    let mut bounds = vec![0];
                    bounds.extend_from_slice(cut);
                    bounds.push(layout.n_layer);
                    let mut choices = Vec::with_capacity(m);
                    for j in 0..m {
                        let last = j + 1 == m;
                        let head_here = last && head == HeadAt::Last;
                        let extra = if j == 0 && head == HeadAt::Leader {
                            head_only_gpu_bytes(layout, ubatch)
                        } else {
                            0
                        };
                        let key = (
                            perm[j],
                            bounds[j],
                            bounds[j + 1],
                            j == 0,
                            head_here,
                            ubatch + u32::from(extra > 0),
                        );
                        let c = cache.entry(key).or_insert_with(|| {
                            let sub = segment_layout(
                                layout,
                                Span(bounds[j], bounds[j + 1]),
                                j == 0,
                                head_here,
                            );
                            choose(ns[j], &sub, ubatch, extra, &w)
                        });
                        match c {
                            Ok(c) => choices.push(*c),
                            Err(e) => {
                                pruned.push((
                                    format!("{tag} head {head:?} cut {cut:?} ub {ubatch}"),
                                    e.clone(),
                                ));
                                continue 'cut;
                            }
                        }
                    }
                    let mut best: Option<(f64, Plan, ClusterPrediction)> = None;
                    for micro in MICRO_BATCHES {
                        let plan = assemble(&ns, layout, cut, head, ubatch, micro, &choices, &w);
                        if let Ok(p) = predict_cluster_plan(hw, layout, &plan, &w)
                            && best.as_ref().is_none_or(|b| p.request_s < b.0)
                        {
                            best = Some((p.request_s, plan, p));
                        }
                    }
                    match best {
                        Some((t, plan, p)) => {
                            let note = format!(
                                "{tag}, cuts {cut:?}, head {head:?}, ubatch {ubatch}, micro {}: pp {:.1} tg {:.2} tok/s",
                                plan.knobs.prefill_micro_batches, p.pp_tps, p.tg_tps
                            );
                            found.push((t, perm.clone(), head, plan, p, note));
                        }
                        None => pruned.push((
                            format!("{tag} head {head:?} cut {cut:?} ub {ubatch}"),
                            "no prediction".into(),
                        )),
                    }
                }
            }
        }
    }
    found.sort_by(|a, b| a.0.total_cmp(&b.0));
    // Thin: the best plan of every order and head placement stays, then the best of the rest.
    let mut keep = vec![false; found.len()];
    let mut seen: Vec<(Vec<usize>, HeadAt)> = Vec::new();
    for (i, f) in found.iter().enumerate() {
        let k = (f.1.clone(), f.2);
        if !seen.contains(&k) {
            seen.push(k);
            keep[i] = true;
        }
    }
    let mut kept = keep.iter().filter(|k| **k).count();
    for k in keep.iter_mut().filter(|k| !**k) {
        if kept < MAX_CLUSTER_CANDIDATES {
            *k = true;
            kept += 1;
        }
    }
    let desc = layout.to_model_desc();
    let reserve = reserve_of(hw);
    let ctx = Context {
        hw,
        model: &desc,
        phase,
        reserve: &reserve,
        hints: Some(hints),
    };
    let mut out = Vec::new();
    for (i, (_, _, _, plan, prediction, note)) in found.into_iter().enumerate() {
        if !keep[i] {
            pruned.push((note, format!("thinned to {MAX_CLUSTER_CANDIDATES}")));
            continue;
        }
        match validate(&plan, &ctx) {
            Ok(_) => out.push(ClusterCandidate {
                id: String::new(),
                plan,
                notes: vec![note],
                prediction,
            }),
            Err(issues) => pruned.push((note, format!("invalid: {:?}", issues[0]))),
        }
    }
    for (i, c) in out.iter_mut().enumerate() {
        c.id = format!("c{i:02}");
    }
    (out, pruned)
}

/// The best plan on each single node, for reference only: never a judged candidate (ADR 0091).
/// A node whose memory cannot hold any placement is left out.
pub fn single_node_references(
    hw: &HardwareGraph,
    layout: &ModelLayout,
    hints: &Hints,
) -> Vec<ClusterCandidate> {
    let w = Workload {
        prompt: hints
            .workload
            .as_ref()
            .and_then(|x| x.typical_prompt)
            .unwrap_or(2048),
        r#gen: 128,
    };
    let mut pruned = Vec::new();
    let mut out = Vec::new();
    for node in usable_nodes(hw, hints, &mut pruned) {
        let mut best: Option<(f64, Plan, ClusterPrediction)> = None;
        for &ubatch in &CLUSTER_UBATCHES {
            let choice = choose(&node, layout, ubatch, 0, &w);
            let Ok(c) = choice else { continue };
            let plan = assemble(&[&node], layout, &[], HeadAt::Last, ubatch, 1, &[c], &w);
            if let Ok(p) = predict_cluster_plan(hw, layout, &plan, &w)
                && best.as_ref().is_none_or(|b| p.request_s < b.0)
            {
                best = Some((p.request_s, plan, p));
            }
        }
        if let Some((_, plan, prediction)) = best {
            let label = node.name.as_deref().unwrap_or("local");
            out.push(ClusterCandidate {
                id: format!("ref-{label}"),
                plan,
                notes: vec![format!("reference only: all layers on node {label}")],
                prediction,
            });
        }
    }
    out
}

/// Segments of a candidate, for reporting.
pub fn candidate_segments(hw: &HardwareGraph, c: &ClusterCandidate) -> Vec<Segment> {
    segments(&c.plan, 0, hw).unwrap_or_default()
}

/// A builder placement on one node with its prediction, for the single-node streaming measurement
/// (tree node `bench-235b-stream-predict`, ADR 0091 "named risk").
#[derive(Debug, Clone, serde::Serialize)]
pub struct NodeRun {
    pub vram_slots: u32,
    pub ram_slots: u32,
    pub ubatch: u32,
    pub pp_tps: f64,
    pub tg_tps: f64,
    pub request_s: f64,
    pub binding: String,
    pub tg_breakdown: Vec<(String, f64)>,
    pub pp_breakdown: Vec<(String, f64)>,
}

/// Every builder placement of the whole model on the only node of `hw` that fits, priced like a
/// segment of [`enumerate_cluster`] (reader path, the disk of the node), best predicted request
/// first. `pin` limits it to one slot pair. With `replay` (outcomes of `tetraeder-bench replay` on a
/// routing trace) decode is priced from the matching outcome (policy `lru`, prefetch on) and
/// prefill keeps the uniform prior, whose 2048-token reads a 32K-prefill trace does not describe;
/// a pair without an outcome is left out. The second value says what was dropped and why.
#[allow(clippy::type_complexity)]
pub fn single_node_runs(
    hw: &HardwareGraph,
    layout: &ModelLayout,
    w: &Workload,
    pin: Option<(u32, u32)>,
    replay: Option<&[crate::score::TierOutcome]>,
) -> Result<(Vec<NodeRun>, Vec<(String, String)>), String> {
    let node = node_of(hw, None)?;
    if node.gpu.is_none() || node.disk.is_none() {
        return Err("the node needs a GPU and the builder disk".into());
    }
    let sub = segment_layout(layout, Span(0, layout.n_layer), true, true);
    let budget = Budget {
        gpu_bytes: node.gpu_bytes / 100 * 99,
        ram_bytes: node.ram_bytes / 100 * 99,
        threads: node.threads,
    };
    let mut dropped = Vec::new();
    let pairs: Vec<(u32, u32)> = match pin {
        Some(p) => vec![p],
        None => feasible_vram_tiers(&budget, &sub, w, &mut dropped)
            .into_iter()
            .flat_map(|v| {
                ram_slot_options(&sub, v)
                    .into_iter()
                    .map(move |r| (v, r))
            })
            .collect(),
    };
    let mut out = Vec::new();
    for (v, r) in pairs {
        let mut tier = uniform_tier(&sub, &[(v, r)]).remove(0);
        if let Some(rep) = replay {
            let Some(t) = rep
                .iter()
                .find(|t| t.vram_slots == v && t.ram_slots == r && t.policy == "lru" && t.prefetch)
            else {
                dropped.push((format!("v{v} r{r}"), "no replay outcome".into()));
                continue;
            };
            tier.decode = t.decode.clone();
        }
        for &ubatch in &CLUSTER_UBATCHES {
            let p = Placement::Builder {
                vram_slots: v,
                ram_slots: r,
            };
            let f = footprint(&sub, p, ubatch, w, KvFormat::default());
            if f.gpu > budget.gpu_bytes || f.host > budget.ram_bytes {
                dropped.push((
                    format!("v{v} r{r} ub{ubatch}"),
                    format!(
                        "needs gpu {} B (budget {}) and host {} B (budget {})",
                        f.gpu, budget.gpu_bytes, f.host, budget.ram_bytes
                    ),
                ));
                continue;
            }
            let feat = builder_features_tier(
                &sub,
                v,
                &tier,
                (node.threads, ubatch, true),
                ReadPath::Reader,
                (READER_DEPTH, READER_DEPTH),
                BUILDER_DISK,
            );
            let pr = tetraeder_costmodel::predict::predict(&node.graph, &feat, w.prompt, w.r#gen)
                .map_err(|e| e.to_string())?;
            out.push(NodeRun {
                vram_slots: v,
                ram_slots: r,
                ubatch,
                pp_tps: pr.pp_tok_s,
                tg_tps: pr.tg_tok_s,
                request_s: pr.request_s,
                binding: pr.binding,
                tg_breakdown: pr.tg_breakdown,
                pp_breakdown: pr.pp_breakdown,
            });
        }
    }
    out.sort_by(|a, b| a.request_s.total_cmp(&b.request_s));
    Ok((out, dropped))
}
