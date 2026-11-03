//! `plan explain` (docs/plan.md "Explain, diff, what-if"): why the search chose its plan.
//!
//! Shares are fractions of the summed stage times of the chosen plan (compute plus hops), the
//! same numbers the search scored with. A device that helps a stage (the CPU holding streamed
//! experts) is counted busy for the whole stage time, so device shares can add up to more than 1.

use crate::objective::{
    SPEED_BAND, Scored, cmp_loss, devices_used, has_speed_limit, objective_order, violations,
};
use crate::search::{SearchInput, SearchResult};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use tetraeder_plan::{Context, Format, UnitKind, validate};

/// The numbers of one plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanNumbers {
    /// Index into `SearchResult::scored`.
    pub index: usize,
    pub desc: String,
    /// Decode speed at batch 1.
    pub tok_s: f64,
    pub ttft_s: f64,
    pub tpot_s: Option<f64>,
    pub simulated: bool,
    /// KL divergence.
    pub quality_loss: f64,
    pub resource_cost: f64,
    pub devices: Vec<String>,
}

impl PlanNumbers {
    fn of(index: usize, s: &Scored) -> Self {
        Self {
            index,
            desc: s.desc.clone(),
            tok_s: s.tok_s,
            ttft_s: s.ttft_s,
            tpot_s: s.tpot_s,
            simulated: s.simulated,
            quality_loss: s.quality_loss,
            resource_cost: s.resource_cost,
            devices: devices_used(&s.plan),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageShare {
    pub stage: u32,
    pub layers: (u32, u32),
    pub devices: Vec<String>,
    pub compute_s: f64,
    pub hop_s: f64,
    /// `(compute_s + hop_s)` over the sum of all stages.
    pub share: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceShare {
    pub device: String,
    /// Compute time of the stages the device takes part in.
    pub time_s: f64,
    pub share: f64,
    pub stages: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinkShare {
    pub from: String,
    pub to: String,
    pub time_s: f64,
    pub share: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceKind {
    Compute,
    Link,
}

/// The resource with the largest share of the token time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Binding {
    /// A device id, or `a->b` for a link.
    pub resource: String,
    pub kind: ResourceKind,
    pub time_s: f64,
    pub share: f64,
}

/// The term a plan lost on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Term {
    Speed,
    Quality,
    Cost,
    /// The runner-up fails a hard limit (or misses it by more).
    Limit,
    /// The runner-up is better by the objective order; the `tradeoff` hint picked another point.
    Tradeoff,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunnerUp {
    pub plan: PlanNumbers,
    pub lost_on: Term,
    /// The chosen plan's value of the term (tok/s, KL, cost; 0 for `limit` and `tradeoff`).
    pub chosen_value: f64,
    pub runner_value: f64,
    pub reason: String,
}

/// Units of one kind stored in one format.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FormatGroup {
    pub kind: UnitKind,
    pub format: Format,
    pub units: u32,
    pub bytes: u64,
    /// KL the quality model charges for these units at this format.
    pub quality_loss: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrontPoint {
    pub index: usize,
    pub tok_s: f64,
    pub quality_loss: f64,
    pub resource_cost: f64,
    pub chosen: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Explanation {
    pub model: String,
    pub phase: u8,
    pub plans_found: usize,
    pub feasible: bool,
    /// Each limit the chosen plan misses; empty when feasible.
    pub failing: Vec<String>,
    pub chosen: Option<PlanNumbers>,
    pub stages: Vec<StageShare>,
    pub devices: Vec<DeviceShare>,
    pub links: Vec<LinkShare>,
    pub binding: Option<Binding>,
    pub runner_up: Option<RunnerUp>,
    pub formats: Vec<FormatGroup>,
    /// Fastest first.
    pub front: Vec<FrontPoint>,
    /// Quality model base KL, added to every plan's loss.
    pub base_kl: f64,
}

/// The headline numbers, for before and after comparisons.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Summary {
    pub feasible: bool,
    pub tok_s: f64,
    pub ttft_s: f64,
    pub quality_loss: f64,
    pub resource_cost: f64,
    pub binding: Option<String>,
    pub desc: String,
}

impl Explanation {
    /// Zeros when no plan was found.
    pub fn summary(&self) -> Summary {
        let c = self.chosen.as_ref();
        Summary {
            feasible: self.feasible,
            tok_s: c.map_or(0.0, |c| c.tok_s),
            ttft_s: c.map_or(0.0, |c| c.ttft_s),
            quality_loss: c.map_or(0.0, |c| c.quality_loss),
            resource_cost: c.map_or(0.0, |c| c.resource_cost),
            binding: self.binding.as_ref().map(|b| b.resource.clone()),
            desc: c.map_or(String::new(), |c| c.desc.clone()),
        }
    }

    /// A short human report.
    pub fn to_text(&self) -> String {
        let mut o = String::new();
        let Some(c) = &self.chosen else {
            let _ = writeln!(o, "{}: no plan found", self.model);
            for f in &self.failing {
                let _ = writeln!(o, "  {f}");
            }
            return o;
        };
        let _ = writeln!(
            o,
            "{} (phase {}): {} of {} plans scored",
            self.model, self.phase, c.desc, self.plans_found
        );
        let _ = writeln!(
            o,
            "decode {:.2} tok/s, TTFT {:.0} ms, KL {:.5}, cost {:.2}, devices {}",
            c.tok_s,
            c.ttft_s * 1e3,
            c.quality_loss,
            c.resource_cost,
            c.devices.join(", ")
        );
        if !self.feasible {
            let _ = writeln!(o, "limits not met (closest plan shown):");
            for f in &self.failing {
                let _ = writeln!(o, "  {f}");
            }
        }
        let _ = writeln!(o, "stages:");
        for s in &self.stages {
            let _ = writeln!(
                o,
                "  {} layers {}..{} on {}: {:.2} ms compute + {:.2} ms hop ({:.0}%)",
                s.stage,
                s.layers.0,
                s.layers.1,
                s.devices.join("+"),
                s.compute_s * 1e3,
                s.hop_s * 1e3,
                s.share * 100.0
            );
        }
        if let Some(b) = &self.binding {
            let _ = writeln!(
                o,
                "binding: {} ({:?}, {:.0}% of the token time)",
                b.resource,
                b.kind,
                b.share * 100.0
            );
        }
        if let Some(r) = &self.runner_up {
            let _ = writeln!(
                o,
                "runner-up: {} ({:.2} tok/s, KL {:.5}, cost {:.2}) lost on {:?}: {}",
                r.plan.desc,
                r.plan.tok_s,
                r.plan.quality_loss,
                r.plan.resource_cost,
                r.lost_on,
                r.reason
            );
        }
        let _ = writeln!(o, "formats:");
        for f in &self.formats {
            let _ = writeln!(
                o,
                "  {:?} {:?}: {} units, {:.2} GB, KL {:.5}",
                f.kind,
                f.format,
                f.units,
                f.bytes as f64 / 1e9,
                f.quality_loss
            );
        }
        let _ = writeln!(o, "front (speed vs quality):");
        for p in &self.front {
            let _ = writeln!(
                o,
                "  {} {:.2} tok/s, KL {:.5}, cost {:.2}",
                if p.chosen { "*" } else { " " },
                p.tok_s,
                p.quality_loss,
                p.resource_cost
            );
        }
        o
    }
}

/// Sum of the relative shortfalls, how far a plan is from meeting the limits.
fn shortfall(s: &Scored, h: &tetraeder_plan::Hints) -> f64 {
    violations(s, h).iter().map(|v| v.1).sum()
}

/// Why `r` is behind `c`: the decisive term in the objective order, with both values.
fn classify(c: &Scored, r: &Scored, hints: &tetraeder_plan::Hints) -> (Term, f64, f64, String) {
    let (cl, rl) = (c.quality_loss, r.quality_loss);
    let by_quality = (
        Term::Quality,
        cl,
        rl,
        format!("quality loss KL {rl:.5} vs {cl:.5}"),
    );
    let by_cost = (
        Term::Cost,
        c.resource_cost,
        r.resource_cost,
        format!(
            "resource cost {:.2} vs {:.2}",
            r.resource_cost, c.resource_cost
        ),
    );
    let by_speed = (
        Term::Speed,
        c.tok_s,
        r.tok_s,
        format!("speed {:.2} vs {:.2} tok/s", r.tok_s, c.tok_s),
    );
    let tradeoff = (
        Term::Tradeoff,
        0.0,
        0.0,
        "better by the objective order; the tradeoff hint picked another point of the front"
            .to_string(),
    );
    let cmp = cmp_loss(cl, rl);
    if cmp.is_lt() {
        return by_quality;
    }
    if cmp.is_gt() {
        return tradeoff;
    }
    let cheaper = cmp_loss(c.resource_cost, r.resource_cost).is_lt();
    if has_speed_limit(hints) {
        if cheaper {
            by_cost
        } else if c.tok_s >= r.tok_s {
            by_speed
        } else {
            tradeoff
        }
    } else if r.tok_s < c.tok_s * (1.0 - SPEED_BAND) {
        by_speed
    } else if cheaper {
        by_cost
    } else if c.tok_s >= r.tok_s {
        by_speed
    } else {
        tradeoff
    }
}

fn runner_up(res: &SearchResult, hints: &tetraeder_plan::Hints) -> Option<RunnerUp> {
    let c = res.choice.chosen?;
    let others: Vec<usize> = (0..res.scored.len()).filter(|&i| i != c).collect();
    let ok: Vec<usize> = others
        .iter()
        .copied()
        .filter(|&i| violations(&res.scored[i], hints).is_empty())
        .collect();
    let chosen = &res.scored[c];
    let (r, lost) = if res.choice.feasible && !ok.is_empty() {
        let r = objective_order(&res.scored, &ok, hints)[0];
        (r, classify(chosen, &res.scored[r], hints))
    } else {
        let r = others.iter().copied().min_by(|&a, &b| {
            shortfall(&res.scored[a], hints)
                .total_cmp(&shortfall(&res.scored[b], hints))
                .then(a.cmp(&b))
        })?;
        let why: Vec<String> = violations(&res.scored[r], hints)
            .into_iter()
            .map(|v| v.0)
            .collect();
        let reason = format!("fails {}", why.join("; "));
        (r, (Term::Limit, 0.0, 0.0, reason))
    };
    Some(RunnerUp {
        plan: PlanNumbers::of(r, &res.scored[r]),
        lost_on: lost.0,
        chosen_value: lost.1,
        runner_value: lost.2,
        reason: lost.3,
    })
}

/// Units per kind and format of the chosen plan, with bytes and the quality model's loss.
fn format_groups(inp: &SearchInput, s: &Scored) -> Vec<FormatGroup> {
    let ctx = Context {
        hw: inp.hw,
        model: inp.model,
        phase: inp.phase,
        reserve: &inp.reserve,
        hints: Some(inp.hints),
    };
    let Ok(e) = validate(&s.plan, &ctx) else {
        return Vec::new();
    };
    let mut m: BTreeMap<(u8, String), FormatGroup> = BTreeMap::new();
    for (k, p) in &e.shards {
        let kind = k.unit.kind();
        let g = m
            .entry((kind as u8, format!("{:?}", p.format)))
            .or_insert(FormatGroup {
                kind,
                format: p.format,
                units: 0,
                bytes: 0,
                quality_loss: 0.0,
            });
        g.bytes += p.bytes;
        if k.shard == 0 {
            g.units += 1;
            g.quality_loss += inp.quality.loss(k.unit, p.format).unwrap_or(0.0);
        }
    }
    m.into_values().collect()
}

pub fn explain(inp: &SearchInput, res: &SearchResult) -> Explanation {
    let ch = &res.choice;
    let chosen = ch.chosen.map(|i| (i, &res.scored[i]));
    let mut stages = Vec::new();
    let mut devices: Vec<DeviceShare> = Vec::new();
    let mut links: Vec<LinkShare> = Vec::new();
    let mut binding: Option<Binding> = None;
    if let Some((_, s)) = chosen {
        let total: f64 = s.stage_times.iter().map(|t| t.compute_s + t.hop_s).sum();
        let share = |t: f64| if total > 0.0 { t / total } else { 0.0 };
        let n = s.stage_times.len();
        for (k, t) in s.stage_times.iter().enumerate() {
            stages.push(StageShare {
                stage: t.stage,
                layers: (t.layers.0, t.layers.1),
                devices: t.devices.clone(),
                compute_s: t.compute_s,
                hop_s: t.hop_s,
                share: share(t.compute_s + t.hop_s),
            });
            for d in &t.devices {
                match devices.iter_mut().find(|x| &x.device == d) {
                    Some(x) => {
                        x.time_s += t.compute_s;
                        x.stages.push(t.stage);
                    }
                    None => devices.push(DeviceShare {
                        device: d.clone(),
                        time_s: t.compute_s,
                        share: 0.0,
                        stages: vec![t.stage],
                    }),
                }
            }
            let (from, to) = (&t.devices[0], &s.stage_times[(k + 1) % n].devices[0]);
            if from != to {
                links.push(LinkShare {
                    from: from.clone(),
                    to: to.clone(),
                    time_s: t.hop_s,
                    share: share(t.hop_s),
                });
            }
        }
        for d in &mut devices {
            d.share = share(d.time_s);
        }
        // Largest share; the first listed wins ties (devices before links, a stage's primary
        // before its helpers).
        for (resource, kind, time_s, sh) in devices
            .iter()
            .map(|d| (d.device.clone(), ResourceKind::Compute, d.time_s, d.share))
            .chain(links.iter().map(|l| {
                (
                    format!("{}->{}", l.from, l.to),
                    ResourceKind::Link,
                    l.time_s,
                    l.share,
                )
            }))
        {
            if binding.as_ref().is_none_or(|b| sh > b.share) {
                binding = Some(Binding {
                    resource,
                    kind,
                    time_s,
                    share: sh,
                });
            }
        }
    }
    Explanation {
        model: inp.model.name.clone(),
        phase: inp.phase.0,
        plans_found: res.scored.len(),
        feasible: ch.feasible,
        failing: ch.failing.clone(),
        chosen: chosen.map(|(i, s)| PlanNumbers::of(i, s)),
        stages,
        devices,
        links,
        binding,
        runner_up: runner_up(res, inp.hints),
        formats: chosen.map_or(Vec::new(), |(_, s)| format_groups(inp, s)),
        front: ch
            .front
            .iter()
            .map(|&i| FrontPoint {
                index: i,
                tok_s: res.scored[i].tok_s,
                quality_loss: res.scored[i].quality_loss,
                resource_cost: res.scored[i].resource_cost,
                chosen: ch.chosen == Some(i),
            })
            .collect(),
        base_kl: inp.quality.base_kl(),
    }
}
