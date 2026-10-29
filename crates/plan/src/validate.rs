//! Plan validator (ADR 0006, 0027): structure, exact-once coverage, degrees, memory, formats,
//! hard hints and the phase capability table. Collects every issue, not just the first.

use crate::{
    BufferKey, BufferKind, ComputeOverride, Expanded, Format, Hints, ModelDesc, Phase, Placement,
    Plan, Replica, ShardKey, Stage, UnitId, UnitKind,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use tetraeder_profiler::{DeviceKind, HardwareGraph, Tier};

/// Everything a plan is checked against.
#[derive(Debug, Clone, Copy)]
pub struct Context<'a> {
    pub hw: &'a HardwareGraph,
    pub model: &'a ModelDesc,
    /// Phase of the executing engine.
    pub phase: Phase,
    /// Measured reserve per device id (CUDA context, workspaces, compositor); missing ids
    /// reserve 0.
    pub reserve: &'a BTreeMap<String, u64>,
    pub hints: Option<&'a Hints>,
}

/// Where in the plan an issue sits: replica and stage indices when known.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct At {
    pub replica: u32,
    pub stage: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Issue {
    /// `plan.model` differs from `ModelDesc::name`.
    WrongModel {
        plan: String,
        model: String,
    },
    NoReplicas,
    /// Stage spans do not tile `0..layers` in order, or a span is empty.
    Spans {
        at: At,
        detail: String,
    },
    UnknownDevice {
        at: At,
        device: String,
    },
    /// A compute device that is not a CPU or GPU.
    CannotCompute {
        at: At,
        device: String,
    },
    /// A compute device outside the stage's device group.
    NotInGroup {
        at: At,
        device: String,
    },
    /// An assign rule that selects no unit of its stage.
    EmptySelection {
        at: At,
        rule: usize,
    },
    /// A unit selected by more than one rule, or in more than one stage.
    Duplicate {
        at: At,
        unit: UnitId,
    },
    /// Complete plans only: a unit no rule selects.
    Missing {
        replica: u32,
        unit: UnitId,
    },
    /// Wrong number of homes (or compute devices) for the unit's kind and the stage's degree.
    ShardCount {
        at: At,
        unit: UnitId,
        expected: String,
        got: usize,
    },
    /// Tensor-parallel shards of one unit compute on the same device.
    SharedShardDevice {
        at: At,
        unit: UnitId,
        device: String,
    },
    /// A degree does not divide what it splits (`what` is heads, kv_heads, experts or devices).
    Degree {
        at: At,
        degree: &'static str,
        value: u32,
        what: &'static str,
        n: u32,
    },
    /// With `ep > 1`, a layer's experts are not spread evenly over `ep` compute devices.
    ExpertSpread {
        at: At,
        layer: u32,
        detail: String,
    },
    /// Complete plans only: an open field (`what` names it).
    Open {
        at: At,
        what: &'static str,
    },
    MemoryExceeded {
        device: String,
        used: u64,
        capacity: u64,
        reserve: u64,
    },
    /// Format more precise than the source: conversion only goes downward (ADR 0024).
    Upconvert {
        unit: UnitId,
        source: Format,
        target: Format,
    },
    /// A slot cache in a stage of a dense model or on an unknown device.
    SlotCache {
        at: At,
        detail: String,
    },
    /// A home on a device that cannot compute and no compute device given.
    NoCompute {
        at: At,
        unit: UnitId,
    },
    /// A field the executing phase cannot run (`field` names it).
    Phase {
        phase: Phase,
        field: &'static str,
    },
    /// A hard hint does not hold.
    Hint {
        detail: String,
    },
}

impl fmt::Display for Issue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for Issue {}

/// Validate a complete plan; on success returns its expanded form.
pub fn validate(plan: &Plan, ctx: &Context) -> Result<Expanded, Vec<Issue>> {
    let (issues, expanded) = Checker::new(plan, ctx, true).run();
    if issues.is_empty() {
        Ok(expanded)
    } else {
        Err(issues)
    }
}

/// Validate a partial plan: only the fields that are set.
pub fn validate_partial(plan: &Plan, ctx: &Context) -> Result<(), Vec<Issue>> {
    let (issues, _) = Checker::new(plan, ctx, false).run();
    if issues.is_empty() {
        Ok(())
    } else {
        Err(issues)
    }
}

fn divides(d: u32, n: u32) -> bool {
    d != 0 && n.is_multiple_of(d)
}

struct Checker<'a> {
    plan: &'a Plan,
    ctx: Context<'a>,
    complete: bool,
    issues: Vec<Issue>,
    /// Phase fields hit so far, each once.
    fields: Vec<&'static str>,
    exp: Expanded,
    unknown: BTreeSet<(u32, Option<u32>, String)>,
    remote: bool,
}

impl<'a> Checker<'a> {
    fn new(plan: &'a Plan, ctx: &Context<'a>, complete: bool) -> Self {
        Self {
            plan,
            ctx: *ctx,
            complete,
            issues: Vec::new(),
            fields: Vec::new(),
            exp: Expanded {
                knobs: plan.knobs,
                ..Default::default()
            },
            unknown: BTreeSet::new(),
            remote: false,
        }
    }

    fn flag_if(&mut self, cond: bool, field: &'static str) {
        if cond && !self.fields.contains(&field) {
            self.fields.push(field);
        }
    }

    /// Note a device the plan names: unknown ids are issues, remote ones need Phase 6.
    fn named(&mut self, at: At, device: &str) {
        match self.ctx.hw.device(device) {
            None => {
                if self
                    .unknown
                    .insert((at.replica, at.stage, device.to_string()))
                {
                    self.issues.push(Issue::UnknownDevice {
                        at,
                        device: device.to_string(),
                    });
                }
            }
            Some(d) => self.remote |= d.tier == Tier::Remote,
        }
    }

    fn run(mut self) -> (Vec<Issue>, Expanded) {
        let (plan, ctx) = (self.plan, self.ctx);
        let multi_node = ctx.phase.capabilities().multi_node;
        if plan.model != ctx.model.name {
            self.issues.push(Issue::WrongModel {
                plan: plan.model.clone(),
                model: ctx.model.name.clone(),
            });
        }
        if plan.replicas.is_empty() {
            self.issues.push(Issue::NoReplicas);
        }
        self.flag_if(plan.replicas.len() > 1 && !multi_node, "multi_node");
        for (r, rep) in plan.replicas.iter().enumerate() {
            self.replica(r as u32, rep);
        }
        self.flag_if(self.remote && !multi_node, "multi_node");
        self.memory();
        self.hints();
        for field in std::mem::take(&mut self.fields) {
            self.issues.push(Issue::Phase {
                phase: ctx.phase,
                field,
            });
        }
        (self.issues, self.exp)
    }

    fn replica(&mut self, r: u32, rep: &Replica) {
        let model = self.ctx.model;
        self.spans(r, rep);
        let mut selected = BTreeSet::new();
        let n = rep.stages.len();
        for (s, st) in rep.stages.iter().enumerate() {
            self.stage(r, s as u32, (s == 0, s + 1 == n), st, &mut selected);
        }
        if self.complete {
            for u in model.units() {
                if !selected.contains(&u) {
                    self.issues.push(Issue::Missing {
                        replica: r,
                        unit: u,
                    });
                }
            }
        }
    }

    fn spans(&mut self, r: u32, rep: &Replica) {
        let layers = self.ctx.model.layers;
        let mut next = 0;
        for (s, st) in rep.stages.iter().enumerate() {
            let at = At {
                replica: r,
                stage: Some(s as u32),
            };
            if st.layers.is_empty() {
                let detail = format!("empty span {}..{}", st.layers.0, st.layers.1);
                self.issues.push(Issue::Spans { at, detail });
            }
            if st.layers.0 != next {
                let detail = format!("starts at layer {}, expected {next}", st.layers.0);
                self.issues.push(Issue::Spans { at, detail });
            }
            next = st.layers.1;
        }
        if next != layers || rep.stages.is_empty() {
            let at = At {
                replica: r,
                stage: None,
            };
            let detail = format!("stages end at layer {next}, model has {layers}");
            self.issues.push(Issue::Spans { at, detail });
        }
    }

    fn degrees(&mut self, at: At, st: &Stage) {
        let model = self.ctx.model;
        let devices = st.devices.len() as u32;
        let mut bad = |degree, value, what, n| {
            self.issues.push(Issue::Degree {
                at,
                degree,
                value,
                what,
                n,
            });
        };
        if let Some(tp) = st.tp {
            for (what, n) in [("heads", model.heads), ("kv_heads", model.kv_heads)] {
                if !divides(tp, n) {
                    bad("tp", tp, what, n);
                }
            }
            if devices > 0 && !divides(tp, devices) {
                bad("tp", tp, "devices", devices);
            }
        }
        if let Some(ep) = st.ep {
            if (model.experts == 0 && ep > 1) || !divides(ep, model.experts) {
                bad("ep", ep, "experts", model.experts);
            }
            if devices > 0 && !divides(ep, devices) {
                bad("ep", ep, "devices", devices);
            }
        }
    }

    fn stage(
        &mut self,
        r: u32,
        s: u32,
        (first, last): (bool, bool),
        st: &Stage,
        selected: &mut BTreeSet<UnitId>,
    ) {
        let ctx = self.ctx;
        let (model, hw) = (ctx.model, ctx.hw);
        let caps = ctx.phase.capabilities();
        let at = At {
            replica: r,
            stage: Some(s),
        };

        for d in &st.devices {
            self.named(at, d);
        }
        for a in &st.assign {
            for d in a.home.iter().chain(&a.compute).flatten() {
                self.named(at, d);
            }
        }
        for d in st.kv.iter().flatten().chain(&st.workspace) {
            self.named(at, &d.device);
        }
        for c in &st.expert_cache {
            self.named(at, &c.device);
        }

        if self.complete {
            let mut open = Vec::new();
            for (is_open, what) in [
                (st.devices.is_empty(), "devices"),
                (st.tp.is_none(), "tp"),
                (st.ep.is_none(), "ep"),
                (st.kv.is_none(), "kv"),
            ] {
                if is_open {
                    open.push(what);
                }
            }
            for a in &st.assign {
                if a.home.is_none() {
                    open.push("home");
                }
                if a.format.is_none() {
                    open.push("format");
                }
            }
            self.issues
                .extend(open.into_iter().map(|what| Issue::Open { at, what }));
        }

        self.degrees(at, st);
        self.flag_if(
            (st.tp.unwrap_or(1) > 1 || st.ep.unwrap_or(1) > 1) && !caps.tp_ep,
            "tp_ep",
        );
        self.flag_if(
            !st.expert_cache.is_empty() && !caps.expert_tiers,
            "expert_tiers",
        );

        let rules = st.assign.len();
        let mut first_unit: Vec<Option<UnitId>> = vec![None; rules];
        // Once per (rule, unit kind): the format check (true) and the shard count check (false).
        let mut checked: BTreeSet<(usize, UnitKind, bool)> = BTreeSet::new();
        let mut max_expert = 0u64;
        let mut expert_compute: BTreeMap<(u32, u32), String> = BTreeMap::new();

        for u in model.units() {
            let in_stage = match u {
                UnitId::Embed => first,
                UnitId::Head => last,
                _ => u.layer().is_some_and(|l| st.layers.contains(l)),
            };
            if !in_stage {
                continue;
            }
            let hit: Vec<usize> = (0..rules)
                .filter(|&i| st.assign[i].units.matches(u))
                .collect();
            for &i in &hit {
                first_unit[i].get_or_insert(u);
            }
            let Some(&fi) = hit.first() else { continue };
            let fresh = selected.insert(u);
            if hit.len() > 1 || !fresh {
                self.issues.push(Issue::Duplicate { at, unit: u });
            }
            let Some(size) = model.size(u) else { continue };
            if !fresh {
                continue;
            }
            let a = &st.assign[fi];
            let kind = u.kind();

            for (i, home) in a.home.iter().flatten().enumerate() {
                let comp = a.compute.as_ref().and_then(|c| c.get(i)).unwrap_or(home);
                let disk = hw.device(home).is_some_and(|d| d.kind == DeviceKind::Disk);
                self.flag_if(
                    disk && (!caps.expert_tiers || (kind != UnitKind::Expert && ctx.phase.0 < 5)),
                    "expert_tiers",
                );
                let moved = match caps.compute_override {
                    ComputeOverride::None => true,
                    ComputeOverride::ExpertsOnly => kind != UnitKind::Expert,
                    ComputeOverride::Any => false,
                };
                self.flag_if(comp != home && moved, "compute_override");
            }
            if let Some(f) = a.format {
                self.flag_if(
                    f != size.source_format && !caps.convert_formats,
                    "format_conversion",
                );
                if !size.source_format.at_least(f) && checked.insert((fi, kind, true)) {
                    self.issues.push(Issue::Upconvert {
                        unit: u,
                        source: size.source_format,
                        target: f,
                    });
                }
            }
            let Some(home) = &a.home else { continue };
            let of = home.len();
            if checked.insert((fi, kind, false)) {
                let (ok, expected) = match (kind, st.tp) {
                    (UnitKind::Expert, _) => (of == 1, "1".to_string()),
                    (UnitKind::Attn | UnitKind::Mlp, Some(tp)) => {
                        (of == tp as usize, format!("tp = {tp}"))
                    }
                    (UnitKind::Embed | UnitKind::Head, Some(tp)) => {
                        (of == 1 || of == tp as usize, format!("1 or tp = {tp}"))
                    }
                    (_, None) => (of >= 1, "at least 1".to_string()),
                };
                if !ok || of == 0 {
                    self.issues.push(Issue::ShardCount {
                        at,
                        unit: u,
                        expected,
                        got: of,
                    });
                }
            }
            let (Some(format), true) = (a.format, of > 0) else {
                continue;
            };
            let bytes = size.bytes(format).div_ceil(of as u64);
            for (i, h) in home.iter().enumerate() {
                let comp = a.compute.as_ref().and_then(|c| c.get(i)).unwrap_or(h);
                let key = ShardKey {
                    replica: r,
                    unit: u,
                    shard: i as u32,
                    of: of as u32,
                };
                let placement = Placement {
                    stage: s,
                    home: h.clone(),
                    compute: comp.clone(),
                    format,
                    bytes,
                };
                if i == 0
                    && let UnitId::Expert(l, e) = u
                {
                    max_expert = max_expert.max(bytes);
                    expert_compute.insert((l, e), comp.clone());
                }
                self.exp.shards.insert(key, placement);
            }
        }

        self.rule_checks(at, st, &first_unit);
        self.buffers(r, s, st, max_expert);
        self.expert_spread(at, st, &expert_compute);
    }

    /// Checks that depend on the rule, not the unit; `first_unit` is the first unit it selects.
    fn rule_checks(&mut self, at: At, st: &Stage, first_unit: &[Option<UnitId>]) {
        let hw = self.ctx.hw;
        for (i, a) in st.assign.iter().enumerate() {
            let Some(fu) = first_unit[i] else {
                self.issues.push(Issue::EmptySelection { at, rule: i });
                continue;
            };
            // (device, given as compute device)
            let mut eff: Vec<(&str, bool)> = Vec::new();
            if let Some(c) = &a.compute {
                if let Some(h) = &a.home
                    && h.len() != c.len()
                {
                    self.issues.push(Issue::ShardCount {
                        at,
                        unit: fu,
                        expected: format!("{} compute devices, as many as homes", h.len()),
                        got: c.len(),
                    });
                }
                eff.extend(c.iter().map(|d| (d.as_str(), true)));
            } else if let Some(h) = &a.home {
                eff.extend(h.iter().map(|d| (d.as_str(), false)));
            }
            for &(d, given) in &eff {
                let Some(dev) = hw.device(d) else { continue };
                if dev.kind == DeviceKind::Disk {
                    self.issues.push(if given {
                        Issue::CannotCompute {
                            at,
                            device: d.to_string(),
                        }
                    } else {
                        Issue::NoCompute { at, unit: fu }
                    });
                } else if !st.devices.is_empty() && !st.devices.iter().any(|g| g == d) {
                    self.issues.push(Issue::NotInGroup {
                        at,
                        device: d.to_string(),
                    });
                }
            }
            if a.home.as_ref().is_some_and(|h| h.len() > 1) {
                let mut seen = BTreeSet::new();
                if let Some((d, _)) = eff.iter().find(|(d, _)| !seen.insert(*d)) {
                    self.issues.push(Issue::SharedShardDevice {
                        at,
                        unit: fu,
                        device: d.to_string(),
                    });
                }
            }
        }
    }

    fn buffers(&mut self, r: u32, s: u32, st: &Stage, max_expert: u64) {
        let at = At {
            replica: r,
            stage: Some(s),
        };
        let moe = self.ctx.model.is_moe();
        let layers = u64::from(st.layers.len());
        let mut add = |kind, device: &str, bytes: u64| {
            let key = BufferKey {
                replica: r,
                stage: s,
                kind,
                device: device.to_string(),
            };
            *self.exp.buffers.entry(key).or_default() += bytes;
        };
        for d in st.kv.iter().flatten() {
            add(BufferKind::Kv, &d.device, d.bytes);
        }
        for d in &st.workspace {
            add(BufferKind::Workspace, &d.device, d.bytes);
        }
        for c in &st.expert_cache {
            if moe {
                let bytes = u64::from(c.slots) * layers * max_expert;
                add(BufferKind::SlotCache, &c.device, bytes);
            } else {
                self.issues.push(Issue::SlotCache {
                    at,
                    detail: format!("slot cache on {} in a dense model", c.device),
                });
            }
        }
    }

    fn expert_spread(&mut self, at: At, st: &Stage, compute: &BTreeMap<(u32, u32), String>) {
        let model = self.ctx.model;
        let Some(ep) = st.ep.filter(|&ep| ep > 1) else {
            return;
        };
        if !divides(ep, model.experts) {
            return;
        }
        for l in st.layers.0..st.layers.1.min(model.layers) {
            let mut counts: BTreeMap<&str, u32> = BTreeMap::new();
            for e in 0..model.experts {
                if let Some(d) = compute.get(&(l, e)) {
                    *counts.entry(d).or_default() += 1;
                }
            }
            if counts.values().sum::<u32>() != model.experts {
                continue;
            }
            let each = model.experts / ep;
            if counts.len() != ep as usize || counts.values().any(|&c| c != each) {
                self.issues.push(Issue::ExpertSpread {
                    at,
                    layer: l,
                    detail: format!("want {each} experts on each of {ep} devices, got {counts:?}"),
                });
            }
        }
    }

    fn memory(&mut self) {
        for (dev, used) in self.exp.memory_per_device() {
            let Some(d) = self.ctx.hw.device(&dev) else {
                continue;
            };
            let reserve = self.ctx.reserve.get(&dev).copied().unwrap_or(0);
            if used > d.capacity_bytes.saturating_sub(reserve) {
                self.issues.push(Issue::MemoryExceeded {
                    device: dev,
                    used,
                    capacity: d.capacity_bytes,
                    reserve,
                });
            }
        }
    }

    fn hints(&mut self) {
        let Some(h) = self.ctx.hints else { return };
        let hw = self.ctx.hw;
        let shards = &self.exp.shards;
        let mut out = Vec::new();
        let mut report = |label: String, bad: Vec<UnitId>| {
            if let Some(first) = bad.first() {
                out.push(Issue::Hint {
                    detail: format!("{label}: {} shards violate it, first {first}", bad.len()),
                });
            }
        };
        let violating = |f: &dyn Fn(&ShardKey, &Placement) -> bool| -> Vec<UnitId> {
            shards
                .iter()
                .filter(|(k, p)| f(k, p))
                .map(|(k, _)| k.unit)
                .collect()
        };
        for p in &h.pin {
            let bad = violating(&|k, pl| {
                p.units.matches(k.unit) && pl.home != p.device && pl.compute != p.device
            });
            report(format!("pin to {}", p.device), bad);
        }
        for f in &h.forbid {
            let bad = violating(&|k, pl| {
                f.units.as_ref().is_none_or(|u| u.matches(k.unit))
                    && (f
                        .device
                        .as_ref()
                        .is_some_and(|d| &pl.home == d || &pl.compute == d)
                        || f.tier
                            .is_some_and(|t| hw.device(&pl.home).is_some_and(|d| d.tier == t)))
            });
            report(format!("forbid {:?} {:?}", f.device, f.tier), bad);
        }
        if let Some(min) = h.min_format {
            let bad = violating(&|_, pl| !pl.format.at_least(min));
            report(format!("min_format {min:?}"), bad);
        }
        for f in &h.format_floor {
            let bad = violating(&|k, pl| f.units.matches(k.unit) && !pl.format.at_least(f.format));
            report(format!("format_floor {:?}", f.format), bad);
        }
        self.issues.extend(out);
    }
}
