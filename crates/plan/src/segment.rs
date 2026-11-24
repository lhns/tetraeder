//! Segments of a replica across nodes (ADR 0091): derived from the plan, never stored.
//!
//! A segment is a maximal run of consecutive stages on one node, plus a head-only segment when
//! the head unit computes on another node than the last stage. A one-node plan has one segment
//! that is the plan. [`segment_plan`] cuts the single-node plan a node's worker runs.

use crate::{Assign, Plan, Replica, Span, Stage, UnitId, UnitKind};
use std::ops::Range;
use tetraeder_profiler::HardwareGraph;

/// One worker's share of a replica.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    /// `None` is the local node.
    pub node: Option<String>,
    /// Stage indices of the replica; empty for a head-only segment (`start == end == n_stages`).
    pub stages: Range<usize>,
    /// Layers computed; `[L, L)` for a head-only segment.
    pub layers: Span,
    /// The segment holds the embedding: only the first.
    pub embed: bool,
    /// The segment computes the output head: exactly one.
    pub head: bool,
}

fn node_name(n: Option<&str>) -> String {
    n.map_or("local".to_string(), str::to_string)
}

/// The one node of a stage's device group.
fn stage_node(hw: &HardwareGraph, st: &Stage, s: usize) -> Result<Option<String>, String> {
    let mut node: Option<Option<String>> = None;
    for id in &st.devices {
        let d = hw
            .device(id)
            .ok_or_else(|| format!("stage {s}: unknown device {id}"))?;
        match &node {
            None => node = Some(d.node.clone()),
            Some(n) if *n == d.node => {}
            Some(n) => {
                return Err(format!(
                    "stage {s}: devices span nodes {} and {}",
                    node_name(n.as_deref()),
                    node_name(d.node.as_deref())
                ));
            }
        }
    }
    node.ok_or_else(|| format!("stage {s}: no devices"))
}

/// The node the head unit computes on, when the last stage's rules say so: the first rule that
/// selects it, compute device else home. `Ok(None)` when no rule selects it (a partial plan).
fn head_node(hw: &HardwareGraph, last: &Stage) -> Result<Option<Option<String>>, String> {
    let Some(rule) = last.assign.iter().find(|a| a.units.matches(UnitId::Head)) else {
        return Ok(None);
    };
    let node_of = |ds: &Option<Vec<String>>| -> Result<Option<Option<String>>, String> {
        let mut out: Option<Option<String>> = None;
        for id in ds.iter().flatten() {
            let d = hw
                .device(id)
                .ok_or_else(|| format!("head: unknown device {id}"))?;
            match &out {
                None => out = Some(d.node.clone()),
                Some(n) if *n == d.node => {}
                Some(_) => return Err("head: shards on different nodes".to_string()),
            }
        }
        Ok(out)
    };
    let (home, compute) = (node_of(&rule.home)?, node_of(&rule.compute)?);
    if let (Some(h), Some(c)) = (&home, &compute)
        && h != c
    {
        return Err("head: home and compute on different nodes".to_string());
    }
    Ok(compute.or(home))
}

/// The segments of one replica in order. `Err` lists what keeps a stage on no single node.
pub fn segments(
    plan: &Plan,
    replica: usize,
    hw: &HardwareGraph,
) -> Result<Vec<Segment>, Vec<String>> {
    let Some(rep) = plan.replicas.get(replica).filter(|r| !r.stages.is_empty()) else {
        return Err(vec![format!("replica {replica} has no stages")]);
    };
    let mut errs = Vec::new();
    let mut nodes = Vec::new();
    for (i, st) in rep.stages.iter().enumerate() {
        match stage_node(hw, st, i) {
            Ok(n) => nodes.push(n),
            Err(e) => errs.push(e),
        }
    }
    let n_stages = rep.stages.len();
    let last = &rep.stages[n_stages - 1];
    let head = match head_node(hw, last) {
        Ok(h) => h,
        Err(e) => {
            errs.push(e);
            None
        }
    };
    if !errs.is_empty() {
        return Err(errs);
    }
    let mut out: Vec<Segment> = Vec::new();
    for (i, st) in rep.stages.iter().enumerate() {
        match out.last_mut() {
            Some(s) if s.node == nodes[i] => {
                s.stages.end = i + 1;
                s.layers.1 = st.layers.1;
            }
            _ => out.push(Segment {
                node: nodes[i].clone(),
                stages: i..i + 1,
                layers: st.layers,
                embed: i == 0,
                head: false,
            }),
        }
    }
    let tail = out.last_mut().expect("at least one stage");
    tail.head = true;
    if let Some(h) = head
        && h != tail.node
    {
        let alone = last
            .assign
            .iter()
            .find(|a| a.units.matches(UnitId::Head))
            .is_some_and(|a| a.units.kinds == [UnitKind::Head]);
        if !alone {
            return Err(vec![
                "a head unit on another node needs a rule of its own (kinds = [head])".to_string(),
            ]);
        }
        tail.head = false;
        let end = last.layers.1;
        out.push(Segment {
            node: h,
            stages: n_stages..n_stages,
            layers: Span(end, end),
            embed: false,
            head: true,
        });
    }
    Ok(out)
}

fn strip(node: Option<&str>, id: &str) -> String {
    node.and_then(|n| id.strip_prefix(n).and_then(|r| r.strip_prefix('/')))
        .unwrap_or(id)
        .to_string()
}

fn strip_stage(st: &Stage, node: Option<&str>) -> Stage {
    let s = |id: &String| strip(node, id);
    let all = |v: &Vec<String>| v.iter().map(s).collect::<Vec<_>>();
    let bytes = |v: &[crate::DeviceBytes]| {
        v.iter()
            .map(|b| crate::DeviceBytes {
                device: s(&b.device),
                bytes: b.bytes,
            })
            .collect::<Vec<_>>()
    };
    Stage {
        devices: all(&st.devices),
        assign: st
            .assign
            .iter()
            .map(|a| Assign {
                home: a.home.as_ref().map(all),
                compute: a.compute.as_ref().map(all),
                ..a.clone()
            })
            .collect(),
        kv: st.kv.as_deref().map(bytes),
        workspace: bytes(&st.workspace),
        expert_cache: st
            .expert_cache
            .iter()
            .map(|c| crate::SlotCache {
                device: s(&c.device),
                slots: c.slots,
            })
            .collect(),
        phase_budgets: st
            .phase_budgets
            .iter()
            .map(|b| crate::PhaseBudget {
                device: s(&b.device),
                ..b.clone()
            })
            .collect(),
        ..st.clone()
    }
}

/// The plan a node's worker runs for `seg`: its stages with device ids without the node prefix
/// (layer spans stay absolute). When the head unit moved to its own segment, the last stage of
/// the others loses its head rule and the head-only segment gets one stage holding it.
pub fn segment_plan(plan: &Plan, replica: usize, seg: &Segment) -> Plan {
    let rep = &plan.replicas[replica];
    let node = seg.node.as_deref();
    let n = rep.stages.len();
    let head_apart = |rep: &Replica| {
        // A head rule on its own, whose compute node differs: the head-only segment exists when
        // `segments` made one, recognised by the caller's `seg.head` flags.
        rep.stages[n - 1]
            .assign
            .iter()
            .any(|a| a.units.kinds == [UnitKind::Head])
    };
    let stages = if seg.stages.is_empty() {
        let last = &rep.stages[n - 1];
        let heads: Vec<Assign> = last
            .assign
            .iter()
            .filter(|a| a.units.kinds == [UnitKind::Head])
            .cloned()
            .collect();
        let mut devices: Vec<String> = Vec::new();
        for d in heads
            .iter()
            .flat_map(|a| a.compute.iter().chain(&a.home).flatten())
        {
            if !devices.contains(d) {
                devices.push(d.clone());
            }
        }
        let st = Stage {
            layers: seg.layers,
            devices,
            tp: Some(1),
            ep: Some(1),
            assign: heads,
            ..Default::default()
        };
        vec![strip_stage(&st, node)]
    } else {
        seg.stages
            .clone()
            .map(|i| {
                let mut st = rep.stages[i].clone();
                if i == n - 1 && !seg.head && head_apart(rep) {
                    st.assign.retain(|a| a.units.kinds != [UnitKind::Head]);
                }
                strip_stage(&st, node)
            })
            .collect()
    };
    Plan {
        model: plan.model.clone(),
        knobs: plan.knobs,
        replicas: vec![Replica { stages }],
    }
}

/// What a segment's worker holds, read from the sub-plan [`segment_plan`] cut (ADR 0091).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentShape {
    /// Layers computed; `[L, L)` for a head-only segment.
    pub layers: Span,
    pub embed: bool,
    pub head: bool,
}

/// The shape of the one replica of a sub-plan of a model with `n_layer` layers. The embedding
/// belongs to a first stage that starts at layer 0 and the head to a last stage that ends at
/// `n_layer`; a rule without a layer filter also matches them in the stages between, so the
/// layer bounds decide first and the rules second (`segment_plan` removes the head rule of a
/// segment whose head went elsewhere).
pub fn segment_shape(plan: &Plan, n_layer: u32) -> Result<SegmentShape, String> {
    let [rep] = plan.replicas.as_slice() else {
        return Err(format!(
            "a segment plan has one replica, not {}",
            plan.replicas.len()
        ));
    };
    let (Some(first), Some(last)) = (rep.stages.first(), rep.stages.last()) else {
        return Err("a segment plan has no stages".to_string());
    };
    let holds = |st: &Stage, u: UnitId| st.assign.iter().any(|a| a.units.matches(u));
    let shape = SegmentShape {
        layers: Span(first.layers.0, last.layers.1),
        embed: first.layers.0 == 0 && holds(first, UnitId::Embed),
        head: last.layers.1 == n_layer && holds(last, UnitId::Head),
    };
    if shape.layers.1 > n_layer || shape.layers.0 > shape.layers.1 {
        return Err(format!(
            "layers {}..{} outside the model's {n_layer}",
            shape.layers.0, shape.layers.1
        ));
    }
    Ok(shape)
}
