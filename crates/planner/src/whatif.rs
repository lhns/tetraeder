//! `plan what-if` (docs/plan.md "Explain, diff, what-if"): solve for hardware you do not have.
//!
//! The priors database (`priors.toml`) holds spec-sheet classes. [`apply`] edits a copy of the
//! hardware graph; [`what_if`] searches before and after and reports the gain in the predicted
//! decode speed of the chosen plan.
//!
//! - `gpu:<class>[:n]` adds GPUs. Each links to the first CPU by PCIe at its class's generation
//!   and width, and to every other GPU by a PCIe link through the host (no peer-to-peer, so no
//!   tensor or expert parallel): latency is the sum of the two host links, bandwidth the smaller.
//! - `ram:<size>[:<class>[:<channels>]]` adds capacity to the first CPU. With a class the
//!   memory bandwidth becomes `channels x` the class's per-channel rate (a new platform).
//! - `nvme:<class>[:n]` adds a drive. The search uses one disk, so a new drive is striped
//!   (RAID 0) with the existing one: capacity and bandwidth add up, as do its link and lane
//!   group. A box without a disk gets one.
//! - `net:<class>` upgrades every network link to the class.
//! - `remove:<device id>` drops a device and its links; hints naming it are dropped.
//!
//! **Calibration transfer:** a new GPU of a class already on the box with a measured bandwidth
//! gets the measured/spec ratio of that device on its bandwidth, and the measured PCIe link of
//! that device. Compute rates carry no measurement flag in the graph, so they stay at spec.
//! Everything added stays `Source::Prior`.

use crate::explain::{Explanation, Summary, explain};
use crate::search::{SearchInput, search};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;
use std::str::FromStr;
use tetraeder_plan::Hints;
use tetraeder_profiler::{
    ComputeSpec, Device, DeviceKind, HardwareGraph, Link, LinkKind, Measurement, Pcie, Source, Tier,
};

const PRIORS_TOML: &str = include_str!("../priors.toml");

/// Reserve for a GPU the what-if adds (CUDA context and workspaces).
const NEW_GPU_RESERVE: u64 = 600 << 20;

#[derive(Debug, Clone, Deserialize)]
pub struct GpuClass {
    pub class: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    /// Lower-case fragments of a device name that mark a device of this class.
    #[serde(default)]
    pub match_name: Vec<String>,
    pub name: String,
    pub capacity_bytes: u64,
    pub bandwidth_bytes_per_s: f64,
    pub flops_per_s: f64,
    pub c0_graph_s: f64,
    pub c0_eager_s: f64,
    pub pcie_gen: u32,
    pub pcie_width: u32,
    pub price_eur: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RamClass {
    pub class: String,
    pub bandwidth_per_channel_bytes_per_s: f64,
    pub default_channels: u32,
    pub price_eur_per_gb: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NvmeClass {
    pub class: String,
    pub capacity_bytes: u64,
    pub bandwidth_bytes_per_s: f64,
    pub alpha_s: f64,
    pub price_eur: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NetClass {
    pub class: String,
    pub bandwidth_bytes_per_s: f64,
    pub alpha_s: f64,
    pub price_eur: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Priors {
    #[serde(default)]
    pub gpu: Vec<GpuClass>,
    #[serde(default)]
    pub ram: Vec<RamClass>,
    #[serde(default)]
    pub nvme: Vec<NvmeClass>,
    #[serde(default)]
    pub net: Vec<NetClass>,
}

/// Fills the compute and bandwidth a discovered GPU lacks from the class `priors` has for its
/// name (`Source::Prior`). CPUs are not filled. Returns a note per GPU it could not fill.
pub fn fill_from_priors(hw: &mut HardwareGraph, priors: &Priors) -> Vec<String> {
    let mut notes = Vec::new();
    for d in hw.devices.iter_mut().filter(|d| d.kind == DeviceKind::Gpu) {
        if d.compute.is_some() && d.bandwidth_bytes_per_s.is_some() {
            continue;
        }
        let lower = d.name.to_lowercase();
        let class = priors
            .gpu
            .iter()
            .find(|c| c.match_name.iter().any(|m| lower.contains(m)))
            .or_else(|| priors.gpu_class(&d.name));
        let Some(c) = class else {
            notes.push(format!("{}: no prior for GPU `{}`", d.id, d.name));
            continue;
        };
        d.compute.get_or_insert(ComputeSpec {
            flops_per_s: c.flops_per_s,
            c0_graph_s: c.c0_graph_s,
            c0_eager_s: c.c0_eager_s,
        });
        d.bandwidth_bytes_per_s
            .get_or_insert(Measurement::prior(c.bandwidth_bytes_per_s));
    }
    notes
}

fn norm(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect::<String>()
        .to_lowercase()
}

impl Priors {
    /// The database shipped in `priors.toml`.
    pub fn builtin() -> Self {
        toml::from_str(PRIORS_TOML).expect("priors.toml parses")
    }

    pub fn gpu_class(&self, name: &str) -> Option<&GpuClass> {
        let n = norm(name);
        self.gpu
            .iter()
            .find(|c| norm(&c.class) == n || c.aliases.iter().any(|a| norm(a) == n))
    }

    fn by_class<'a, T>(xs: &'a [T], class: &str, key: impl Fn(&T) -> &str) -> Option<&'a T> {
        xs.iter().find(|c| norm(key(c)) == norm(class))
    }

    pub fn ram_class(&self, c: &str) -> Option<&RamClass> {
        Self::by_class(&self.ram, c, |x| &x.class)
    }

    pub fn nvme_class(&self, c: &str) -> Option<&NvmeClass> {
        Self::by_class(&self.nvme, c, |x| &x.class)
    }

    pub fn net_class(&self, c: &str) -> Option<&NetClass> {
        Self::by_class(&self.net, c, |x| &x.class)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Gpu,
    Ram,
    Nvme,
    Net,
}

/// A hardware addition: a class with a count (GPU, NVMe) or a size (RAM).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Add {
    pub kind: Kind,
    /// Canonical class id; `None` for plain RAM capacity.
    pub class: Option<String>,
    pub count: u32,
    pub bytes: Option<u64>,
    pub channels: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Change {
    Add(Add),
    Remove(String),
}

fn parse_bytes(s: &str) -> Result<u64, String> {
    let t = s.trim().to_lowercase();
    let t = t.trim_end_matches('b').trim_end_matches('i');
    let (num, mult) = match t.chars().last() {
        Some('g') => (&t[..t.len() - 1], 1u64 << 30),
        Some('t') => (&t[..t.len() - 1], 1u64 << 40),
        Some('m') => (&t[..t.len() - 1], 1u64 << 20),
        _ => return Err(format!("bad size {s:?}, want e.g. 256G")),
    };
    let n: f64 = num.parse().map_err(|_| format!("bad size {s:?}"))?;
    if n <= 0.0 {
        return Err(format!("bad size {s:?}"));
    }
    Ok((n * mult as f64) as u64)
}

impl FromStr for Change {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        let p = Priors::builtin();
        let parts: Vec<&str> = s.split(':').collect();
        let count = |i: usize| -> Result<u32, String> {
            parts.get(i).map_or(Ok(1), |c| {
                c.parse::<u32>()
                    .ok()
                    .filter(|n| *n > 0)
                    .ok_or(format!("bad count {c:?} in {s:?}"))
            })
        };
        let unknown = |what: &str, c: &str| format!("unknown {what} class {c:?} in {s:?}");
        match parts.as_slice() {
            ["remove", id] if !id.is_empty() => Ok(Self::Remove((*id).to_string())),
            ["gpu", c, ..] => {
                let g = p.gpu_class(c).ok_or_else(|| unknown("gpu", c))?;
                Ok(Self::Add(Add {
                    kind: Kind::Gpu,
                    class: Some(g.class.clone()),
                    count: count(2)?,
                    bytes: None,
                    channels: None,
                }))
            }
            ["nvme", c, ..] => {
                let n = p.nvme_class(c).ok_or_else(|| unknown("nvme", c))?;
                Ok(Self::Add(Add {
                    kind: Kind::Nvme,
                    class: Some(n.class.clone()),
                    count: count(2)?,
                    bytes: None,
                    channels: None,
                }))
            }
            ["net", c] => {
                let n = p.net_class(c).ok_or_else(|| unknown("net", c))?;
                Ok(Self::Add(Add {
                    kind: Kind::Net,
                    class: Some(n.class.clone()),
                    count: 1,
                    bytes: None,
                    channels: None,
                }))
            }
            ["ram", size, rest @ ..] if rest.len() <= 2 => {
                let class = match rest.first() {
                    Some(c) => Some(
                        p.ram_class(c)
                            .ok_or_else(|| unknown("ram", c))?
                            .class
                            .clone(),
                    ),
                    None => None,
                };
                let channels = match rest.get(1) {
                    Some(c) => Some(
                        c.parse::<u32>()
                            .ok()
                            .filter(|n| *n > 0)
                            .ok_or(format!("bad channel count {c:?} in {s:?}"))?,
                    ),
                    None => None,
                };
                Ok(Self::Add(Add {
                    kind: Kind::Ram,
                    class,
                    count: 1,
                    bytes: Some(parse_bytes(size)?),
                    channels,
                }))
            }
            _ => Err(format!(
                "bad change {s:?}: want gpu:<class>[:n], ram:<size>[:<class>[:<channels>]], \
                 nvme:<class>[:n], net:<class> or remove:<device id>"
            )),
        }
    }
}

impl fmt::Display for Change {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Remove(id) => write!(f, "remove:{id}"),
            Self::Add(a) => {
                let class = a.class.as_deref().unwrap_or("");
                match a.kind {
                    Kind::Gpu => write!(f, "gpu:{class}")?,
                    Kind::Nvme => write!(f, "nvme:{class}")?,
                    Kind::Net => return write!(f, "net:{class}"),
                    Kind::Ram => {
                        write!(f, "ram:{}G", a.bytes.unwrap_or(0) >> 30)?;
                        if !class.is_empty() {
                            write!(f, ":{class}")?;
                            if let Some(c) = a.channels {
                                write!(f, ":{c}")?;
                            }
                        }
                        return Ok(());
                    }
                }
                if a.count > 1 {
                    write!(f, ":{}", a.count)?;
                }
                Ok(())
            }
        }
    }
}

impl Change {
    /// List price of the addition in EUR; `None` for a removal.
    pub fn price_eur(&self) -> Option<f64> {
        let Self::Add(a) = self else { return None };
        let p = Priors::builtin();
        let class = a.class.as_deref().unwrap_or("");
        let n = f64::from(a.count);
        match a.kind {
            Kind::Gpu => p.gpu_class(class).map(|c| c.price_eur * n),
            Kind::Nvme => p.nvme_class(class).map(|c| c.price_eur * n),
            Kind::Net => p.net_class(class).map(|c| c.price_eur),
            Kind::Ram => {
                let per_gb = p
                    .ram_class(if class.is_empty() { "ddr4" } else { class })
                    .map(|c| c.price_eur_per_gb)?;
                Some(per_gb * (a.bytes? >> 30) as f64)
            }
        }
    }
}

/// Practical PCIe bandwidth, bytes per second.
fn pcie_beta(gen_: u32, width: u32) -> f64 {
    let x16 = match gen_ {
        0..=3 => 12.5e9,
        4 => 25.0e9,
        _ => 50.0e9,
    };
    x16 * f64::from(width) / 16.0
}

fn next_id(hw: &HardwareGraph, prefix: &str) -> String {
    (0u32..)
        .map(|i| format!("{prefix}{i}"))
        .find(|id| hw.device(id).is_none())
        .expect("some id is free")
}

fn first_of(hw: &HardwareGraph, kind: DeviceKind) -> Option<String> {
    hw.devices
        .iter()
        .find(|d| d.kind == kind)
        .map(|d| d.id.clone())
}

/// Measured/spec bandwidth ratio and the measured PCIe link of a device of the same class on the
/// box (calibration transfer, module docs).
fn calibration(hw: &HardwareGraph, g: &GpuClass) -> Option<(String, f64, Option<Link>)> {
    let d = hw.devices.iter().find(|d| {
        d.kind == DeviceKind::Gpu
            && g.match_name
                .iter()
                .any(|m| d.name.to_lowercase().contains(m))
            && d.bandwidth_bytes_per_s
                .is_some_and(|b| b.source == Source::Measured)
    })?;
    let ratio = d.bandwidth_bytes_per_s?.value / g.bandwidth_bytes_per_s;
    let link = hw
        .links
        .iter()
        .find(|l| {
            l.kind == LinkKind::Pcie
                && (l.a == d.id || l.b == d.id)
                && l.beta_bytes_per_s.source == Source::Measured
        })
        .cloned();
    Some((d.id.clone(), ratio, link))
}

fn add_gpu(hw: &mut HardwareGraph, g: &GpuClass) {
    let id = next_id(hw, "gpu");
    let cal = calibration(hw, g);
    let mut bw = Measurement::prior(g.bandwidth_bytes_per_s);
    if let Some((from, ratio, _)) = &cal {
        bw.value *= ratio;
        hw.notes.push(format!(
            "what-if: {id} ({}) bandwidth scaled by {ratio:.3} measured on {from}",
            g.class
        ));
    }
    let index = hw
        .devices
        .iter()
        .filter_map(|d| d.index)
        .max()
        .map_or(0, |m| m + 1);
    let mut d = Device::new(&id, DeviceKind::Gpu, &g.name, Tier::Vram, g.capacity_bytes);
    d.bandwidth_bytes_per_s = Some(bw);
    d.compute = Some(ComputeSpec {
        flops_per_s: g.flops_per_s,
        c0_graph_s: g.c0_graph_s,
        c0_eager_s: g.c0_eager_s,
    });
    d.pcie = Some(Pcie {
        gen_current: g.pcie_gen,
        width_current: g.pcie_width,
        gen_max: g.pcie_gen,
        width_max: g.pcie_width,
    });
    d.index = Some(index);
    let others: Vec<String> = hw
        .devices
        .iter()
        .filter(|x| x.kind == DeviceKind::Gpu)
        .map(|x| x.id.clone())
        .collect();
    let cpu = first_of(hw, DeviceKind::Cpu);
    let host = |hw: &HardwareGraph, dev: &str| -> (f64, f64) {
        cpu.as_deref()
            .and_then(|c| hw.link_between(dev, c))
            .map_or((1.0e-5, pcie_beta(g.pcie_gen, g.pcie_width)), |l| {
                (l.alpha_s.value, l.beta_bytes_per_s.value)
            })
    };
    let own = match &cal {
        Some((_, _, Some(l))) => (l.alpha_s.value, l.beta_bytes_per_s.value),
        _ => (1.0e-5, pcie_beta(g.pcie_gen, g.pcie_width)),
    };
    let mut links = Vec::new();
    if let Some(c) = &cpu {
        links.push(Link {
            id: format!("pcie-{id}"),
            a: id.clone(),
            b: c.clone(),
            kind: LinkKind::Pcie,
            alpha_s: Measurement::prior(own.0),
            beta_bytes_per_s: Measurement::prior(own.1),
            lane_group: None,
        });
    }
    for o in &others {
        let (a2, b2) = host(hw, o);
        links.push(Link {
            id: format!("pcie-{id}-{o}"),
            a: id.clone(),
            b: o.clone(),
            kind: LinkKind::Pcie,
            alpha_s: Measurement::prior(own.0 + a2),
            beta_bytes_per_s: Measurement::prior(own.1.min(b2)),
            lane_group: None,
        });
    }
    hw.devices.push(d);
    hw.links.extend(links);
}

fn add_nvme(hw: &mut HardwareGraph, n: &NvmeClass, count: u32) -> Result<(), String> {
    let bw = n.bandwidth_bytes_per_s * f64::from(count);
    let cap = n.capacity_bytes * u64::from(count);
    if let Some(disk) = hw.devices.iter().position(|d| d.kind == DeviceKind::Disk) {
        let id = hw.devices[disk].id.clone();
        let d = &mut hw.devices[disk];
        d.capacity_bytes += cap;
        let old = d.bandwidth_bytes_per_s.map_or(0.0, |b| b.value);
        d.bandwidth_bytes_per_s = Some(Measurement::prior(old + bw));
        let mut groups = Vec::new();
        for l in hw
            .links
            .iter_mut()
            .filter(|l| l.kind == LinkKind::Storage && (l.a == id || l.b == id))
        {
            l.beta_bytes_per_s = Measurement::prior(l.beta_bytes_per_s.value + bw);
            groups.extend(l.lane_group.clone());
        }
        for g in hw.lane_groups.iter_mut().filter(|g| groups.contains(&g.id)) {
            g.capacity_bytes_per_s += bw;
        }
        hw.notes.push(format!(
            "what-if: {count} x nvme:{} striped with {id}",
            n.class
        ));
        return Ok(());
    }
    let cpu = first_of(hw, DeviceKind::Cpu).ok_or("no CPU to attach an NVMe drive to")?;
    let id = next_id(hw, "nvme");
    let mut d = Device::new(&id, DeviceKind::Disk, "what-if NVMe", Tier::Nvme, cap);
    d.bandwidth_bytes_per_s = Some(Measurement::prior(bw));
    d.media = Some("ssd".into());
    hw.devices.push(d);
    hw.links.push(Link {
        id: format!("nvme-{id}"),
        a: id,
        b: cpu,
        kind: LinkKind::Storage,
        alpha_s: Measurement::prior(n.alpha_s),
        beta_bytes_per_s: Measurement::prior(bw),
        lane_group: None,
    });
    Ok(())
}

fn add_ram(hw: &mut HardwareGraph, a: &Add, p: &Priors) -> Result<(), String> {
    let cpu = hw
        .devices
        .iter_mut()
        .find(|d| d.kind == DeviceKind::Cpu)
        .ok_or("no CPU to add RAM to")?;
    cpu.capacity_bytes += a.bytes.unwrap_or(0);
    if let Some(c) = a.class.as_deref().and_then(|c| p.ram_class(c)) {
        let ch = a.channels.unwrap_or(c.default_channels);
        let new = c.bandwidth_per_channel_bytes_per_s * f64::from(ch);
        let old = cpu.bandwidth_bytes_per_s.map_or(new, |b| b.value);
        cpu.bandwidth_bytes_per_s = Some(Measurement::prior(new));
        for t in &mut cpu.thread_curve {
            t.bytes_per_s *= new / old;
        }
    }
    Ok(())
}

/// A copy of `hw` with the changes applied in order (module docs).
pub fn apply(hw: &HardwareGraph, changes: &[Change]) -> Result<HardwareGraph, String> {
    let p = Priors::builtin();
    let mut out = hw.clone();
    for c in changes {
        match c {
            Change::Remove(id) => {
                if out.device(id).is_none() {
                    return Err(format!("remove:{id}: no such device"));
                }
                out.devices.retain(|d| &d.id != id);
                out.links.retain(|l| &l.a != id && &l.b != id);
            }
            Change::Add(a) => {
                let class = a.class.as_deref().unwrap_or("");
                match a.kind {
                    Kind::Gpu => {
                        let g = p
                            .gpu_class(class)
                            .ok_or(format!("unknown gpu class {class}"))?;
                        for _ in 0..a.count {
                            add_gpu(&mut out, g);
                        }
                    }
                    Kind::Nvme => {
                        let n = p
                            .nvme_class(class)
                            .ok_or(format!("unknown nvme class {class}"))?;
                        add_nvme(&mut out, n, a.count)?;
                    }
                    Kind::Ram => add_ram(&mut out, a, &p)?,
                    Kind::Net => {
                        let n = p
                            .net_class(class)
                            .ok_or(format!("unknown net class {class}"))?;
                        let mut any = false;
                        for l in out.links.iter_mut().filter(|l| l.kind == LinkKind::Network) {
                            l.beta_bytes_per_s = Measurement::prior(n.bandwidth_bytes_per_s);
                            l.alpha_s = Measurement::prior(n.alpha_s);
                            any = true;
                        }
                        if !any {
                            return Err(format!("net:{class}: the hardware has no network link"));
                        }
                    }
                }
            }
        }
    }
    out.validate()?;
    Ok(out)
}

/// Hints without the rules that name a removed device.
fn strip_hints(h: &Hints, gone: &BTreeSet<String>) -> Hints {
    let mut h = h.clone();
    h.pin.retain(|p| !gone.contains(&p.device));
    h.forbid
        .retain(|f| f.device.as_ref().is_none_or(|d| !gone.contains(d)));
    h.prefer.retain(|p| !gone.contains(&p.device));
    h.cost.retain(|d, _| !gone.contains(d));
    h
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Gain {
    /// After minus before, tok/s.
    pub tok_s: f64,
    /// After over before; `None` when there was no plan before.
    pub ratio: Option<f64>,
    pub quality_loss: f64,
    pub ttft_s: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WhatIf {
    pub changes: Vec<String>,
    pub before: Summary,
    pub after: Summary,
    pub gain: Gain,
    /// Priors and calibration transfers the new hardware used.
    pub notes: Vec<String>,
}

impl WhatIf {
    pub fn to_text(&self) -> String {
        let b = &self.before;
        let a = &self.after;
        let mut o = format!(
            "{}\nbefore: {:.2} tok/s, KL {:.5}, cost {:.2}, binding {}\nafter:  {:.2} tok/s, KL {:.5}, cost {:.2}, binding {}\n",
            self.changes.join(" "),
            b.tok_s,
            b.quality_loss,
            b.resource_cost,
            b.binding.as_deref().unwrap_or("-"),
            a.tok_s,
            a.quality_loss,
            a.resource_cost,
            a.binding.as_deref().unwrap_or("-"),
        );
        match self.gain.ratio {
            Some(r) => {
                o += &format!(
                    "gain: {:+.2} tok/s ({:+.0}%)\n",
                    self.gain.tok_s,
                    (r - 1.0) * 100.0
                )
            }
            None => o += &format!("gain: {:+.2} tok/s\n", self.gain.tok_s),
        }
        for n in &self.notes {
            o += &format!("note: {n}\n");
        }
        o
    }
}

fn gain(before: &Summary, after: &Summary) -> Gain {
    Gain {
        tok_s: after.tok_s - before.tok_s,
        ratio: (before.tok_s > 0.0).then(|| after.tok_s / before.tok_s),
        quality_loss: after.quality_loss - before.quality_loss,
        ttft_s: after.ttft_s - before.ttft_s,
    }
}

/// Search and explain the input with the changes applied.
fn explain_after(
    inp: &SearchInput,
    changes: &[Change],
) -> Result<(Explanation, Vec<String>), String> {
    let hw = apply(inp.hw, changes)?;
    let gone: BTreeSet<String> = changes
        .iter()
        .filter_map(|c| match c {
            Change::Remove(id) => Some(id.clone()),
            Change::Add(_) => None,
        })
        .collect();
    let hints = strip_hints(inp.hints, &gone);
    let mut reserve = inp.reserve.clone();
    reserve.retain(|d, _| hw.device(d).is_some());
    for d in &hw.devices {
        if d.kind == DeviceKind::Gpu {
            reserve.entry(d.id.clone()).or_insert(NEW_GPU_RESERVE);
        }
    }
    let after = SearchInput {
        hw: &hw,
        model: inp.model,
        hints: &hints,
        phase: inp.phase,
        workload: inp.workload,
        quality: inp.quality,
        reserve,
        routing: inp.routing,
        linux: inp.linux,
    };
    let res = search(&after);
    let notes = hw
        .notes
        .iter()
        .filter(|n| n.starts_with("what-if:") && !inp.hw.notes.contains(n))
        .cloned()
        .collect();
    Ok((explain(&after, &res), notes))
}

/// Predicted speed before and after the changes.
pub fn what_if(inp: &SearchInput, changes: &[Change]) -> Result<WhatIf, String> {
    let before = explain(inp, &search(inp)).summary();
    what_if_from(inp, &before, changes)
}

fn what_if_from(inp: &SearchInput, before: &Summary, changes: &[Change]) -> Result<WhatIf, String> {
    let (after, notes) = explain_after(inp, changes)?;
    let after = after.summary();
    Ok(WhatIf {
        changes: changes.iter().map(ToString::to_string).collect(),
        gain: gain(before, &after),
        before: before.clone(),
        after,
        notes,
    })
}

/// The purchases `rank_purchases` considers by default: one addition each.
pub fn default_candidates() -> Vec<Change> {
    [
        "gpu:rtx3090",
        "gpu:rtx4090",
        "gpu:rtx4060ti16",
        "gpu:a100_40",
        "ram:64G",
        "ram:128G",
        "ram:256G",
        "nvme:gen4",
        "nvme:gen5",
        "net:25gbe",
    ]
    .iter()
    .map(|s| s.parse().expect("default candidates parse"))
    .collect()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Purchase {
    pub change: String,
    pub after: Summary,
    pub gain_tok_s: f64,
    pub price_eur: Option<f64>,
    /// tok/s gained per 1000 EUR.
    pub gain_per_1000_eur: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ranking {
    pub before: Summary,
    /// Largest gain first; ties by change string.
    pub ranked: Vec<Purchase>,
    /// Candidates that do not apply to the hardware, with the reason.
    pub skipped: Vec<(String, String)>,
}

/// Rank single additions by the predicted gain in decode tok/s (and per euro).
pub fn rank_purchases(inp: &SearchInput, candidates: &[Change]) -> Ranking {
    let before = explain(inp, &search(inp)).summary();
    let mut ranked = Vec::new();
    let mut skipped = Vec::new();
    for c in candidates {
        match what_if_from(inp, &before, std::slice::from_ref(c)) {
            Ok(w) => {
                let price = c.price_eur();
                ranked.push(Purchase {
                    change: c.to_string(),
                    gain_per_1000_eur: price
                        .filter(|p| *p > 0.0)
                        .map(|p| w.gain.tok_s / p * 1000.0),
                    price_eur: price,
                    gain_tok_s: w.gain.tok_s,
                    after: w.after,
                });
            }
            Err(e) => skipped.push((c.to_string(), e)),
        }
    }
    ranked.sort_by(|a, b| {
        b.gain_tok_s
            .total_cmp(&a.gain_tok_s)
            .then_with(|| a.change.cmp(&b.change))
    });
    Ranking {
        before,
        ranked,
        skipped,
    }
}
