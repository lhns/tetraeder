//! Solving one skeleton: expert tiers, memory fit by format knapsack, plan construction.

use super::cost::{StageShape, hop};
use super::prep::Prep;
use std::collections::BTreeMap;
use tetraeder_plan::{
    Assign, DeviceBytes, Format, Knobs, Plan, Replica, SlotCache, Span, Stage, UnitId, UnitKind,
    UnitSel,
};
use tetraeder_profiler::DeviceKind;

/// Where cold experts of a GPU stage compute: on the CPU (the weights stay in RAM) or on the
/// GPU (streamed from RAM or NVMe through slots).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    Cpu,
    Stream,
}

/// A pipeline stage's device group. `devs.len() > 1` only with `tp > 1` or `ep > 1`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Group {
    pub devs: Vec<String>,
    pub tp: u32,
    pub ep: u32,
}

impl Group {
    pub fn single(d: &str) -> Self {
        Self {
            devs: vec![d.to_string()],
            tp: 1,
            ep: 1,
        }
    }

    pub fn primary(&self) -> &str {
        &self.devs[0]
    }

    /// Devices holding attention and MLP shards.
    fn shards(&self) -> &[String] {
        if self.tp > 1 {
            &self.devs
        } else {
            &self.devs[..1]
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Skel {
    pub groups: Vec<Group>,
    pub mode: Mode,
}

impl Skel {
    pub fn describe(&self, moe: bool) -> String {
        let g: Vec<String> = self
            .groups
            .iter()
            .map(|g| {
                if g.devs.len() == 1 {
                    g.devs[0].clone()
                } else {
                    let kind = if g.tp > 1 { "tp" } else { "ep" };
                    format!("{kind}{}[{}]", g.devs.len(), g.devs.join("+"))
                }
            })
            .collect();
        let mut s = g.join(">");
        if moe {
            s += match self.mode {
                Mode::Cpu => " experts=cpu",
                Mode::Stream => " experts=stream",
            };
        }
        s
    }
}

/// `(home, compute)` of the hot, RAM and NVMe tier of a stage; `None` when unusable.
pub(crate) type Tiers = [Option<(String, String)>; 3];

/// Tier availability of one stage, per layer so a layer span can be checked.
pub(crate) struct StageTiers {
    pub tiers: Tiers,
    /// Per tier, per layer: the rules allow every expert of the layer there.
    ok: [Vec<bool>; 3],
}

impl StageTiers {
    pub fn for_span(&self, span: Span) -> Tiers {
        let mut out: Tiers = [None, None, None];
        for (t, slot) in out.iter_mut().enumerate() {
            if (span.0..span.1).all(|l| self.ok[t][l as usize]) {
                slot.clone_from(&self.tiers[t]);
            }
        }
        out
    }
}

pub(crate) fn stage_tiers(p: &Prep, sk: &Skel, s: usize) -> StageTiers {
    let g = &sk.groups[s];
    let d = g.primary();
    let n = p.layers as usize;
    let mut tiers: Tiers = [None, None, None];
    if p.experts > 0 && g.devs.len() == 1 {
        tiers[0] = Some((d.to_string(), d.to_string()));
        let gpu = p.kind(d) == Some(DeviceKind::Gpu);
        if let Some(cpu) = p.cpu.as_deref()
            && cpu != d
            && hop(p.hw, d, cpu).is_some()
        {
            match sk.mode {
                Mode::Cpu => tiers[1] = Some((cpu.to_string(), cpu.to_string())),
                Mode::Stream
                    if gpu && p.caps.compute_override != tetraeder_plan::ComputeOverride::None =>
                {
                    tiers[1] = Some((cpu.to_string(), d.to_string()));
                }
                Mode::Stream => {}
            }
        }
        if p.caps.expert_tiers
            && let (Some(cpu), Some(nvme)) = (p.cpu.as_deref(), p.nvme.as_deref())
            && hop(p.hw, nvme, cpu).is_some()
            && hop(p.hw, d, cpu).is_some()
        {
            tiers[2] = Some((nvme.to_string(), d.to_string()));
        }
    }
    let mut ok: [Vec<bool>; 3] = [vec![true; n], vec![true; n], vec![true; n]];
    if !p.rules.trivial() {
        for (t, flags) in ok.iter_mut().enumerate() {
            let Some((home, comp)) = &tiers[t] else {
                continue;
            };
            for (l, flag) in flags.iter_mut().enumerate() {
                *flag = (0..p.model.experts)
                    .all(|e| p.rules.ok(UnitId::Expert(l as u32, e), home, comp));
            }
        }
    }
    StageTiers { tiers, ok }
}

/// Experts per layer in each tier, filling tier 0 first. `None` when they do not all fit.
pub(crate) fn fill(
    avail: [Option<u64>; 3],
    layer_bytes: u64,
    experts: usize,
) -> Option<[usize; 3]> {
    let mut rem = experts;
    let mut out = [0usize; 3];
    for (t, a) in avail.iter().enumerate() {
        let Some(a) = a else { continue };
        let c = a
            .checked_div(layer_bytes)
            .map_or(rem, |fit| rem.min(fit as usize));
        out[t] = c;
        rem -= c;
    }
    (rem == 0).then_some(out)
}

/// Slots of the expert cache of a streaming GPU stage (at least top-k, ADR 0034).
pub(crate) fn slots_for(p: &Prep) -> u32 {
    p.model.experts_used.clamp(1, p.model.experts.max(1))
}

fn expert_bytes(p: &Prep, f: Format) -> u64 {
    p.model.expert.as_ref().map_or(0, |e| e.bytes(f))
}

/// One stage's decision.
#[derive(Debug, Clone)]
pub(crate) struct StageSol {
    pub span: Span,
    pub counts: [usize; 3],
    pub slots: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct Solved {
    pub bounds: Vec<u32>,
    pub sel: Vec<usize>,
    pub stages: Vec<StageSol>,
}

pub(crate) enum Overflow {
    Dense(usize),
    Experts(usize),
}

fn add(used: &mut BTreeMap<String, u64>, d: &str, b: u64) {
    *used.entry(d.to_string()).or_default() += b;
}

/// Memory use at the formats `sel` and the expert tiers that result; `Err` names the stage that
/// overflows. Mirrors the validator's byte rules (shards `ceil(bytes / of)`, slot arena
/// `slots x layers x largest expert`).
pub(crate) fn place(
    p: &Prep,
    sk: &Skel,
    bounds: &[u32],
    sel: &[usize],
    tiers: &[StageTiers],
) -> Result<Vec<StageSol>, Overflow> {
    let n = sk.groups.len();
    let mut used: BTreeMap<String, u64> = BTreeMap::new();
    let fmt = |i: usize| p.items[i].opts[sel[i]].fmt;
    for (s, g) in sk.groups.iter().enumerate() {
        let span = Span(bounds[s], bounds[s + 1]);
        let shards = g.shards();
        let of = shards.len() as u64;
        for l in span.0..span.1 {
            let a = p.model.attn.bytes(fmt(p.attn_item(l))).div_ceil(of);
            let m = p.model.mlp.bytes(fmt(p.mlp_item(l))).div_ceil(of);
            for d in shards {
                add(&mut used, d, a + m);
            }
        }
        if s == 0 {
            add(
                &mut used,
                g.primary(),
                p.model.embed.bytes(fmt(p.embed_item())),
            );
        }
        if s + 1 == n {
            add(
                &mut used,
                g.primary(),
                p.model.head.bytes(fmt(p.head_item())),
            );
        }
        let kv = (p.kv_layer * u64::from(span.len())).div_ceil(of);
        for d in shards {
            add(&mut used, d, kv);
        }
        for d in &g.devs {
            add(&mut used, d, p.ws);
        }
    }
    for (s, g) in sk.groups.iter().enumerate() {
        for d in &g.devs {
            if used.get(d).copied().unwrap_or(0) > p.cap(d) {
                return Err(Overflow::Dense(s));
            }
        }
    }
    let mut out = Vec::with_capacity(n);
    for (s, g) in sk.groups.iter().enumerate() {
        let span = Span(bounds[s], bounds[s + 1]);
        if p.experts == 0 {
            out.push(StageSol {
                span,
                counts: [0; 3],
                slots: 0,
            });
            continue;
        }
        let e = p.experts;
        let layer_bytes: u64 = (span.0..span.1)
            .map(|l| expert_bytes(p, fmt(p.expert_item(l))))
            .sum();
        if g.devs.len() > 1 {
            let each = layer_bytes * (e / g.devs.len()) as u64;
            for d in &g.devs {
                add(&mut used, d, each);
                if used[d] > p.cap(d) {
                    return Err(Overflow::Experts(s));
                }
            }
            out.push(StageSol {
                span,
                counts: [e, 0, 0],
                slots: 0,
            });
            continue;
        }
        let d = g.primary();
        let t = tiers[s].for_span(span);
        let max_b = (span.0..span.1)
            .map(|l| expert_bytes(p, fmt(p.expert_item(l))))
            .max()
            .unwrap_or(0);
        let gpu = p.kind(d) == Some(DeviceKind::Gpu) && p.caps.expert_tiers;
        let streams = t[2].is_some() || t[1].as_ref().is_some_and(|x| x.0 != x.1 && x.1 == d);
        let slots = if streams && gpu { slots_for(p) } else { 0 };
        let slot_bytes = u64::from(slots) * u64::from(span.len()) * max_b;
        let free = |dev: &str, used: &BTreeMap<String, u64>| {
            p.cap(dev)
                .saturating_sub(used.get(dev).copied().unwrap_or(0))
        };
        let cpu_helper = t[1].as_ref().is_some_and(|x| x.1 != d);
        let mut avail = [None, None, None];
        if t[0].is_some() {
            avail[0] = Some(free(d, &used).saturating_sub(slot_bytes));
        }
        if let Some((home, _)) = &t[1] {
            let ws = if cpu_helper { p.ws } else { 0 };
            avail[1] = Some(free(home, &used).saturating_sub(ws));
        }
        if let Some((home, _)) = &t[2] {
            avail[2] = Some(free(home, &used));
        }
        let Some(counts) = fill(avail, layer_bytes, e) else {
            return Err(Overflow::Experts(s));
        };
        if let Some((home, _)) = &t[0] {
            add(&mut used, home, counts[0] as u64 * layer_bytes);
        }
        if let Some((home, _)) = &t[1] {
            add(&mut used, home, counts[1] as u64 * layer_bytes);
            if cpu_helper && counts[1] > 0 {
                add(&mut used, home, p.ws);
            }
        }
        if let Some((home, _)) = &t[2] {
            add(&mut used, home, counts[2] as u64 * layer_bytes);
        }
        let slots = if counts[2] > 0 || (counts[1] > 0 && !cpu_helper) {
            slots
        } else {
            0
        };
        if slots > 0 {
            add(&mut used, d, slot_bytes);
        }
        out.push(StageSol {
            span,
            counts,
            slots,
        });
    }
    Ok(out)
}

/// The item with the cheapest next step (loss per byte saved) among `items`.
pub(crate) fn best_step(p: &Prep, sel: &[usize], items: &[usize]) -> Option<usize> {
    items
        .iter()
        .copied()
        .filter_map(|i| {
            let o = &p.items[i].opts;
            let s = sel[i];
            (s + 1 < o.len())
                .then(|| (o[s + 1].loss - o[s].loss) / (o[s].bytes - o[s + 1].bytes) as f64)
                .map(|e| (e, i))
        })
        .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)))
        .map(|x| x.1)
}

fn stage_items(
    p: &Prep,
    sk: &Skel,
    bounds: &[u32],
    s: usize,
    dense: bool,
    experts: bool,
) -> Vec<usize> {
    let mut v = Vec::new();
    for l in bounds[s]..bounds[s + 1] {
        if dense {
            v.push(p.attn_item(l));
            v.push(p.mlp_item(l));
        }
        if experts && p.experts > 0 {
            v.push(p.expert_item(l));
        }
    }
    if dense && s == 0 {
        v.push(p.embed_item());
    }
    if dense && s + 1 == sk.groups.len() {
        v.push(p.head_item());
    }
    v
}

/// Compress items by the greedy knapsack until the skeleton's memory fits. `sel` starts at the
/// most precise formats and only goes down.
pub(crate) fn fit(
    p: &Prep,
    sk: &Skel,
    bounds: &[u32],
    sel: &mut [usize],
    tiers: &[StageTiers],
) -> Result<Vec<StageSol>, String> {
    loop {
        let ov = match place(p, sk, bounds, sel, tiers) {
            Ok(v) => return Ok(v),
            Err(ov) => ov,
        };
        let (s, first, what) = match ov {
            Overflow::Dense(s) => (
                s,
                stage_items(p, sk, bounds, s, true, false),
                "weights and KV",
            ),
            Overflow::Experts(s) => (s, stage_items(p, sk, bounds, s, false, true), "experts"),
        };
        let pick = best_step(p, sel, &first)
            .or_else(|| best_step(p, sel, &stage_items(p, sk, bounds, s, true, true)));
        match pick {
            Some(i) => sel[i] += 1,
            None => {
                return Err(format!(
                    "stage {s} ({}): {what} do not fit at the smallest allowed formats",
                    sk.groups[s].devs.join("+")
                ));
            }
        }
    }
}

pub(crate) fn total_bytes(p: &Prep, sel: &[usize]) -> u64 {
    p.items
        .iter()
        .zip(sel)
        .map(|(it, &s)| it.opts[s].bytes)
        .sum()
}

pub(crate) fn min_total_bytes(p: &Prep) -> u64 {
    p.items
        .iter()
        .map(|it| it.opts[it.opts.len() - 1].bytes)
        .sum()
}

/// `sel` after taking the global compression steps until the total is at most `target` bytes.
pub(crate) fn compress_to(p: &Prep, sel: &[usize], target: f64) -> Vec<usize> {
    let mut out = sel.to_vec();
    let mut total = total_bytes(p, &out) as f64;
    for &(i, step) in &p.path {
        if total <= target {
            break;
        }
        if out[i] == step {
            let o = &p.items[i].opts;
            total -= (o[step].bytes - o[step + 1].bytes) as f64;
            out[i] += 1;
        }
    }
    out
}

/// Quality loss of a format selection: the model's base KL plus the item losses.
pub(crate) fn loss_of(p: &Prep, sel: &[usize]) -> f64 {
    p.base_kl
        + p.items
            .iter()
            .zip(sel)
            .map(|(it, &s)| it.opts[s].loss)
            .sum::<f64>()
}

/// The StageShape of stage `s` for the formats `sel` and the tier counts `counts`.
pub(crate) fn shape(
    p: &Prep,
    sk: &Skel,
    bounds: &[u32],
    s: usize,
    sel: &[usize],
    counts: [usize; 3],
) -> StageShape {
    let g = &sk.groups[s];
    let n = sk.groups.len();
    let span = Span(bounds[s], bounds[s + 1]);
    let layers = f64::from(span.len());
    let of = g.shards().len() as f64;
    let bytes = |i: usize| p.items[i].opts[sel[i]].bytes as f64;
    let dense: f64 = (span.0..span.1)
        .map(|l| bytes(p.attn_item(l)) + bytes(p.mlp_item(l)))
        .sum();
    let eb = if p.experts > 0 {
        (span.0..span.1)
            .map(|l| bytes(p.expert_item(l)))
            .sum::<f64>()
            / (layers * p.experts as f64)
    } else {
        0.0
    };
    let m = p.model;
    let last = s + 1 == n;
    StageShape {
        group: g.devs.clone(),
        tp: g.tp,
        ep: g.ep,
        layers: span.len(),
        nb: dense / layers / of,
        nf: (m.attn.flops_per_token + m.mlp.flops_per_token) / of,
        kv_read: p.read_ctx * m.kv_bytes_per_token_per_layer as f64 / of,
        eb,
        ef: m.expert.as_ref().map_or(0.0, |e| e.flops_per_token),
        counts,
        stream: sk.mode == Mode::Stream,
        cpu: p.cpu.clone(),
        nvme: p.nvme.clone(),
        head_bytes: if last { bytes(p.head_item()) } else { 0.0 },
        head_flops: if last { m.head.flops_per_token } else { 0.0 },
    }
}

/// True when any part of the plan computes on a CPU or streams weights, so CUDA graphs are off.
pub(crate) fn uses_host(p: &Prep, sk: &Skel, sols: &[StageSol]) -> bool {
    sk.groups.iter().zip(sols).any(|(g, s)| {
        g.devs.iter().any(|d| p.kind(d) == Some(DeviceKind::Cpu)) || s.counts[1] + s.counts[2] > 0
    })
}

fn runs<T: PartialEq + Copy>(xs: &[T]) -> Vec<(u32, u32, T)> {
    let mut out: Vec<(u32, u32, T)> = Vec::new();
    for (i, &x) in xs.iter().enumerate() {
        match out.last_mut() {
            Some(r) if r.2 == x => r.1 = i as u32 + 1,
            _ => out.push((i as u32, i as u32 + 1, x)),
        }
    }
    out
}

/// The complete plan of a solved skeleton.
pub(crate) fn build_plan(
    p: &Prep,
    sk: &Skel,
    sv: &Solved,
    graphs: bool,
    all_tiers: &[StageTiers],
) -> Plan {
    let n = sk.groups.len();
    let fmt = |i: usize| p.items[i].opts[sv.sel[i]].fmt;
    let mut stages = Vec::new();
    for (s, g) in sk.groups.iter().enumerate() {
        let sol = &sv.stages[s];
        let span = sol.span;
        let tiers = all_tiers[s].for_span(span);
        let home: Vec<String> = g.shards().to_vec();
        let mut devices = g.devs.clone();
        let cpu_helper = tiers[1]
            .as_ref()
            .filter(|t| sol.counts[1] > 0 && t.1 != g.primary())
            .map(|t| t.1.clone());
        if let Some(c) = &cpu_helper
            && !devices.contains(c)
        {
            devices.push(c.clone());
        }
        let mut assign = Vec::new();
        let layer_runs = |item: &dyn Fn(u32) -> usize| -> Vec<(Span, Format)> {
            let fs: Vec<Format> = (span.0..span.1).map(|l| fmt(item(l))).collect();
            runs(&fs)
                .into_iter()
                .map(|(a, b, f)| (Span(span.0 + a, span.0 + b), f))
                .collect()
        };
        let attn_f = |l: u32| p.attn_item(l);
        let mlp_f = |l: u32| p.mlp_item(l);
        let dense_kinds: [(UnitKind, &dyn Fn(u32) -> usize); 2] =
            [(UnitKind::Attn, &attn_f), (UnitKind::Mlp, &mlp_f)];
        for (kind, item) in dense_kinds {
            for (sp, f) in layer_runs(item) {
                assign.push(Assign {
                    units: UnitSel {
                        kinds: vec![kind],
                        layers: Some(sp),
                        experts: None,
                    },
                    home: Some(home.clone()),
                    compute: None,
                    format: Some(f),
                });
            }
        }
        for (cond, kind, item) in [
            (s == 0, UnitKind::Embed, p.embed_item()),
            (s + 1 == n, UnitKind::Head, p.head_item()),
        ] {
            if cond {
                assign.push(Assign {
                    units: UnitSel {
                        kinds: vec![kind],
                        layers: None,
                        experts: None,
                    },
                    home: Some(vec![g.primary().to_string()]),
                    compute: None,
                    format: Some(fmt(item)),
                });
            }
        }
        if p.experts > 0 {
            let e = p.experts;
            // Tier of every expert id.
            let mut tier_of = vec![0usize; e];
            let mut place_of: Vec<(String, Option<String>)> = Vec::new();
            if g.devs.len() > 1 {
                let each = e / g.devs.len();
                for (id, t) in tier_of.iter_mut().enumerate() {
                    *t = id / each;
                }
                place_of = g.devs.iter().map(|d| (d.clone(), None)).collect();
            } else {
                let [h0, h1, _] = sol.counts;
                for (rank, &id) in p.order.iter().enumerate() {
                    tier_of[id] = if rank < h0 {
                        0
                    } else if rank < h0 + h1 {
                        1
                    } else {
                        2
                    };
                }
                for t in &tiers {
                    place_of.push(match t {
                        Some((h, c)) => (h.clone(), Some(c.clone())),
                        None => (String::new(), None),
                    });
                }
            }
            for (sp, f) in layer_runs(&|l| p.expert_item(l)) {
                for (a, b, t) in runs(&tier_of) {
                    let (h, c) = place_of[t].clone();
                    assign.push(Assign {
                        units: UnitSel {
                            kinds: vec![UnitKind::Expert],
                            layers: Some(sp),
                            experts: Some(Span(a, b)),
                        },
                        home: Some(vec![h]),
                        compute: c.map(|c| vec![c]),
                        format: Some(f),
                    });
                }
            }
        }
        let of = home.len() as u64;
        let kv = (p.kv_layer * u64::from(span.len())).div_ceil(of);
        let mut workspace: Vec<DeviceBytes> = g
            .devs
            .iter()
            .map(|d| DeviceBytes {
                device: d.clone(),
                bytes: p.ws,
            })
            .collect();
        if let Some(c) = &cpu_helper
            && !g.devs.contains(c)
        {
            workspace.push(DeviceBytes {
                device: c.clone(),
                bytes: p.ws,
            });
        }
        stages.push(Stage {
            layers: span,
            devices,
            tp: Some(g.tp),
            ep: Some(g.ep),
            assign,
            kv: Some(
                home.iter()
                    .map(|d| DeviceBytes {
                        device: d.clone(),
                        bytes: kv,
                    })
                    .collect(),
            ),
            workspace,
            expert_cache: if sol.slots > 0 {
                vec![SlotCache {
                    device: g.primary().to_string(),
                    slots: sol.slots,
                }]
            } else {
                Vec::new()
            },
            phase_budgets: Vec::new(),
        });
    }
    Plan {
        model: p.model.name.clone(),
        knobs: Knobs {
            cuda_graphs: graphs,
            ..Knobs::default()
        },
        replicas: vec![Replica { stages }],
    }
}
