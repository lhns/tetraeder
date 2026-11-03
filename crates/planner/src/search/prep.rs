//! What the search computes once per call: workload numbers, the routing distribution, the
//! format ladder of every item, the usable devices and the hard-hint rules.

use super::SearchInput;
use std::collections::BTreeMap;
use tetraeder_costmodel::kv_bytes;
use tetraeder_plan::{Capabilities, Format, Hints, ModelDesc, Phase, UnitId, UnitKind};
use tetraeder_profiler::{DeviceKind, HardwareGraph, Tier};

/// One format choice of an item: its bytes and quality loss.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Opt {
    pub fmt: Format,
    pub bytes: u64,
    pub loss: f64,
}

/// Units that share one format: embed, head, and per layer attention, MLP and the experts.
/// `opts` is the Pareto ladder: strictly fewer bytes and strictly more loss down the list;
/// index 0 is the most precise allowed format.
#[derive(Debug, Clone)]
pub(crate) struct Item {
    pub opts: Vec<Opt>,
}

/// The hard hints (`pin`, `forbid`) as placement predicates, the same rule the validator applies.
pub(crate) struct Rules<'a> {
    pub hw: &'a HardwareGraph,
    pub hints: &'a Hints,
}

impl Rules<'_> {
    pub fn trivial(&self) -> bool {
        self.hints.pin.is_empty() && self.hints.forbid.is_empty()
    }

    /// May `u` live on `home` and compute on `compute`?
    pub fn ok(&self, u: UnitId, home: &str, compute: &str) -> bool {
        let tier = self.hw.device(home).map(|d| d.tier);
        let forbidden = self.hints.forbid.iter().any(|f| {
            f.units.as_ref().is_none_or(|s| s.matches(u))
                && (f
                    .device
                    .as_deref()
                    .is_some_and(|d| d == home || d == compute)
                    || f.tier.is_some_and(|t| tier == Some(t)))
        });
        !forbidden
            && self
                .hints
                .pin
                .iter()
                .all(|p| !p.units.matches(u) || p.device == home || p.device == compute)
    }

    /// A device no unit may use at all.
    pub fn banned(&self, dev: &str) -> bool {
        self.hints
            .forbid
            .iter()
            .any(|f| f.units.is_none() && f.device.as_deref() == Some(dev))
    }
}

pub(crate) struct Prep<'a> {
    pub hw: &'a HardwareGraph,
    pub model: &'a ModelDesc,
    pub hints: &'a Hints,
    pub phase: Phase,
    pub caps: Capabilities,
    pub rules: Rules<'a>,
    pub reserve: &'a BTreeMap<String, u64>,
    pub prompt: u32,
    pub gen_tokens: u32,
    pub conc: u32,
    pub linux: bool,
    /// Activation bytes per token between stages.
    pub act: f64,
    /// KV bytes per layer for `conc` sequences of `prompt + gen` tokens.
    pub kv_layer: u64,
    pub ws: u64,
    /// Context a decode token attends over, on average.
    pub read_ctx: f64,
    pub layers: u32,
    pub experts: usize,
    /// Expert ids, most selected first (ties by id).
    pub order: Vec<usize>,
    /// Prefix sums over ranks of the per-token selection probability.
    pub cum_p: Vec<f64>,
    /// Prefix sums over ranks of P(expert is touched by a step of 1 token / of `prompt` tokens).
    pub cum_d1: Vec<f64>,
    pub cum_dm: Vec<f64>,
    pub items: Vec<Item>,
    pub base_kl: f64,
    /// Usable compute devices (GPU and CPU), in hardware order.
    pub compute: Vec<String>,
    pub cpu: Option<String>,
    pub nvme: Option<String>,
    /// Devices left out and why.
    pub notes: Vec<(String, String)>,
    /// Prefix sums over layers of the most precise and the smallest attn + mlp bytes, and of the
    /// experts' bytes.
    pub pre_dense_max: Vec<u64>,
    pub pre_dense_min: Vec<u64>,
    pub pre_exp_max: Vec<u64>,
    pub pre_exp_min: Vec<u64>,
    /// Global greedy order of compression steps: `(item, step index)`.
    pub path: Vec<(usize, usize)>,
}

impl<'a> Prep<'a> {
    pub fn embed_item(&self) -> usize {
        0
    }
    pub fn head_item(&self) -> usize {
        1
    }
    pub fn attn_item(&self, l: u32) -> usize {
        2 + l as usize
    }
    pub fn mlp_item(&self, l: u32) -> usize {
        2 + self.layers as usize + l as usize
    }
    pub fn expert_item(&self, l: u32) -> usize {
        2 + 2 * self.layers as usize + l as usize
    }

    /// Usable bytes of a device: capacity minus the measured reserve.
    pub fn cap(&self, dev: &str) -> u64 {
        let c = self.hw.device(dev).map_or(0, |d| d.capacity_bytes);
        c.saturating_sub(self.reserve.get(dev).copied().unwrap_or(0))
    }

    pub fn kind(&self, dev: &str) -> Option<DeviceKind> {
        self.hw.device(dev).map(|d| d.kind)
    }

    pub fn new(inp: &'a SearchInput<'a>) -> Result<Self, String> {
        let (hw, model, hints) = (inp.hw, inp.model, inp.hints);
        let caps = inp.phase.capabilities();
        let rules = Rules { hw, hints };
        let (mut prompt, gen_tokens, mut conc) = inp.workload;
        if let Some(w) = &hints.workload {
            prompt = w.typical_prompt.unwrap_or(prompt);
            conc = w.concurrency.unwrap_or(conc);
        }
        let (prompt, conc) = (prompt.max(1), conc.max(1));
        let gen_tokens = gen_tokens.max(1);
        let layers = model.layers;
        let experts = model.experts as usize;

        // Routing: probability that a token selects expert e; uniform when unknown.
        let k = f64::from(model.experts_used.min(model.experts));
        let mut prob = vec![if experts > 0 { k / experts as f64 } else { 0.0 }; experts];
        if let Some(r) = inp.routing {
            let sum: f64 = r.iter().sum();
            if r.len() == experts && sum > 0.0 && r.iter().all(|x| x.is_finite() && *x >= 0.0) {
                prob = r.iter().map(|x| (k * x / sum).min(1.0)).collect();
            }
        }
        let mut order: Vec<usize> = (0..experts).collect();
        order.sort_by(|&a, &b| prob[b].total_cmp(&prob[a]).then(a.cmp(&b)));
        let prefix = |f: &dyn Fn(f64) -> f64| -> Vec<f64> {
            let mut v = vec![0.0];
            for &e in &order {
                v.push(v.last().copied().unwrap_or(0.0) + f(prob[e]));
            }
            v
        };
        let touched = |m: f64| move |p: f64| 1.0 - (1.0 - p.min(1.0)).powf(m);
        let cum_p = prefix(&|p| p);
        let cum_d1 = prefix(&touched(1.0));
        let cum_dm = prefix(&touched(f64::from(prompt)));

        // Items and their format ladders.
        let floor_ok = |u: UnitId, f: Format| {
            hints.min_format.is_none_or(|m| f.at_least(m))
                && hints
                    .format_floor
                    .iter()
                    .all(|ff| !ff.units.matches(u) || f.at_least(ff.format))
        };
        let mut items = Vec::new();
        let mut shapes: Vec<(UnitKind, u32)> = vec![(UnitKind::Embed, 0), (UnitKind::Head, 0)];
        shapes.extend((0..layers).map(|l| (UnitKind::Attn, l)));
        shapes.extend((0..layers).map(|l| (UnitKind::Mlp, l)));
        if model.is_moe() {
            shapes.extend((0..layers).map(|l| (UnitKind::Expert, l)));
        }
        for (kind, l) in shapes {
            let units: Vec<UnitId> = match kind {
                UnitKind::Embed => vec![UnitId::Embed],
                UnitKind::Head => vec![UnitId::Head],
                UnitKind::Attn => vec![UnitId::Attn(l)],
                UnitKind::Mlp => vec![UnitId::Mlp(l)],
                UnitKind::Expert => (0..model.experts).map(|e| UnitId::Expert(l, e)).collect(),
            };
            let size = match kind {
                UnitKind::Embed => &model.embed,
                UnitKind::Head => &model.head,
                UnitKind::Attn => &model.attn,
                UnitKind::Mlp => &model.mlp,
                UnitKind::Expert => model.expert.as_ref().ok_or("no expert size")?,
            };
            let mut opts: Vec<Opt> = Vec::new();
            for f in Format::ALL {
                if !size.source_format.at_least(f)
                    || (f != size.source_format && !caps.convert_formats)
                    || !units.iter().all(|&u| floor_ok(u, f))
                {
                    continue;
                }
                let mut loss = 0.0;
                let mut known = true;
                for &u in &units {
                    match inp.quality.loss(u, f) {
                        Some(x) => loss += x,
                        None => {
                            known = false;
                            break;
                        }
                    }
                }
                if known {
                    opts.push(Opt {
                        fmt: f,
                        bytes: units.len() as u64 * size.bytes(f),
                        loss,
                    });
                }
            }
            opts.sort_by(|a, b| b.bytes.cmp(&a.bytes).then(a.loss.total_cmp(&b.loss)));
            let mut ladder: Vec<Opt> = Vec::new();
            for o in opts {
                if ladder.last().is_some_and(|l| l.bytes == o.bytes) {
                    continue;
                }
                while ladder.last().is_some_and(|l| l.loss >= o.loss) {
                    ladder.pop();
                }
                ladder.push(o);
            }
            if ladder.is_empty() {
                let name = match kind {
                    UnitKind::Embed => "embed".to_string(),
                    UnitKind::Head => "head".to_string(),
                    _ => units[0].to_string(),
                };
                return Err(format!(
                    "no allowed format for {name}: floors and known quality leave none"
                ));
            }
            items.push(Item { opts: ladder });
        }

        // Devices.
        let mut compute = Vec::new();
        let mut notes = Vec::new();
        let (mut cpu, mut nvme) = (None, None);
        for d in &hw.devices {
            let why = if rules.banned(&d.id) {
                Some("forbidden by a hint")
            } else if d.tier == Tier::Remote && !caps.multi_node {
                Some("remote tier needs a later phase")
            } else if d.kind != DeviceKind::Disk
                && (d.compute.is_none() || d.bandwidth_bytes_per_s.is_none())
            {
                Some("no compute or bandwidth numbers")
            } else {
                None
            };
            if let Some(w) = why {
                notes.push((format!("device {}", d.id), w.to_string()));
                continue;
            }
            match d.kind {
                DeviceKind::Gpu => compute.push(d.id.clone()),
                DeviceKind::Cpu => {
                    cpu.get_or_insert_with(|| d.id.clone());
                    compute.push(d.id.clone());
                }
                DeviceKind::Disk => {
                    nvme.get_or_insert_with(|| d.id.clone());
                }
            }
        }

        let one = |f: &dyn Fn(&Item) -> u64, kinds: &[UnitKind]| -> Vec<u64> {
            let mut v = vec![0u64];
            for l in 0..layers {
                let mut s = 0;
                for &k in kinds {
                    let i = match k {
                        UnitKind::Attn => 2 + l as usize,
                        UnitKind::Mlp => 2 + layers as usize + l as usize,
                        _ => 2 + 2 * layers as usize + l as usize,
                    };
                    s += f(&items[i]);
                }
                v.push(v[l as usize] + s);
            }
            v
        };
        let max_b = |it: &Item| it.opts[0].bytes;
        let min_b = |it: &Item| it.opts[it.opts.len() - 1].bytes;
        let pre_dense_max = one(&max_b, &[UnitKind::Attn, UnitKind::Mlp]);
        let pre_dense_min = one(&min_b, &[UnitKind::Attn, UnitKind::Mlp]);
        let (pre_exp_max, pre_exp_min) = if model.is_moe() {
            (
                one(&max_b, &[UnitKind::Expert]),
                one(&min_b, &[UnitKind::Expert]),
            )
        } else {
            (vec![0; layers as usize + 1], vec![0; layers as usize + 1])
        };

        // Greedy order of compression steps over all items (cheapest loss per byte first).
        let mut path = Vec::new();
        let mut at = vec![0usize; items.len()];
        let eff = |i: usize, s: usize| -> Option<f64> {
            let o = &items[i].opts;
            (s + 1 < o.len())
                .then(|| (o[s + 1].loss - o[s].loss) / (o[s].bytes - o[s + 1].bytes) as f64)
        };
        let mut heap: Vec<(f64, usize)> = (0..items.len())
            .filter_map(|i| eff(i, 0).map(|e| (e, i)))
            .collect();
        while !heap.is_empty() {
            let (pos, _) = heap
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.0.total_cmp(&b.1.0).then(a.1.1.cmp(&b.1.1)))
                .expect("heap is not empty");
            let (_, i) = heap.swap_remove(pos);
            path.push((i, at[i]));
            at[i] += 1;
            if let Some(e) = eff(i, at[i]) {
                heap.push((e, i));
            }
        }

        Ok(Self {
            hw,
            model,
            hints,
            phase: inp.phase,
            caps,
            rules,
            reserve: &inp.reserve,
            prompt,
            gen_tokens,
            conc,
            linux: inp.linux,
            act: model.activation_bytes_per_token as f64,
            kv_layer: kv_bytes(
                model.kv_bytes_per_token_per_layer,
                u64::from(conc),
                u64::from(prompt) + u64::from(gen_tokens),
            ),
            ws: (model.activation_bytes_per_token * 2048).max(1 << 20),
            read_ctx: f64::from(prompt) + f64::from(gen_tokens) / 2.0,
            layers,
            experts,
            order,
            cum_p,
            cum_d1,
            cum_dm,
            items,
            base_kl: inp.quality.base_kl(),
            compute,
            cpu,
            nvme,
            notes,
            pre_dense_max,
            pre_dense_min,
            pre_exp_max,
            pre_exp_min,
            path,
        })
    }
}
