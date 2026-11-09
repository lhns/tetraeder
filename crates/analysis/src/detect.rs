//! The detector (ADR 0041, ADR 0013): environment first, then regime and shares with intervals,
//! then the residual check against the calibrated rates.
//!
//! `faults` holds what is wrong, ranked by the step time it costs. `limits` holds every
//! resource's share of the step and is never a finding. All thresholds are named constants,
//! tuned on simulated fixtures only.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::observation::{EnvSignals, Expected, Observation};

/// A resource must hold this share of the step for an environment fault on it to matter.
pub const MATERIAL_SHARE: f64 = 0.10;
/// A resource below this share is ignored by the residual check: its slowdown costs too little.
pub const RESIDUAL_MIN_SHARE: f64 = 0.05;
/// CPU used by other processes, as a fraction of the quota, that counts as a co-tenant.
pub const CPU_OTHER_FRAC: f64 = 0.15;
/// The same, when the rate residual already shows the CPU is slow (weaker evidence is enough).
pub const CPU_OTHER_FRAC_RESIDUAL: f64 = 0.05;
/// VRAM held by others, as a fraction of the device, that matters.
pub const VRAM_OTHER_FRAC: f64 = 0.05;
/// Fewer VRAM slots than planned by this fraction is a deficit.
pub const SLOT_DEFICIT: f64 = 0.10;
/// An expert hit rate below this is low.
pub const HIT_RATE_LOW: f64 = 0.75;
/// Swap-in (or major-fault) rate that is paging, not a stray page: 5 MB/s. Low because the rate
/// falls with the pace it slows (and under an io.max that also caps swap-in); the rule also needs
/// a stall.
pub const SWAP_IN_MIN_BPS: f64 = 5e6;
/// PSI memory stall share that marks major faults as refaults.
pub const PSI_MIN_SHARE: f64 = 0.01;
/// Bytes per major fault (one base page).
pub const PAGE_BYTES: f64 = 4096.0;
/// Without a calibrated disk rate, other readers above this are contention: 100 MB/s.
pub const DISK_OTHER_MIN_BPS: f64 = 100e6;
/// Cost assumed for a throttled GPU when no calibrated rate says how slow it is.
pub const THROTTLE_DEFAULT_LOSS: f64 = 0.3;
/// Share of a stalled resource's time assumed lost to swapping when neither PSI nor a
/// calibrated rate says how much.
pub const SWAP_DEFAULT_LOSS: f64 = 0.5;
/// Batch sizes up to this run the stages one after another.
pub const SERIALIZED_MAX_BATCH: u32 = 1;
/// Band used when the observation carries no valid one.
pub const DEFAULT_BAND: f64 = 0.1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Regime {
    /// Batch 1 through a split: throughput is the inverse of the summed demands.
    Serialized,
    /// Resources overlap: throughput is bounded by the busiest.
    Load,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FaultClass {
    GpuThrottle,
    CpuCotenant,
    DiskSlow,
    VramPressure,
    PcieDegraded,
    LinkSlow,
    Swapping,
    ModelMismatch,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fault {
    pub class: FaultClass,
    pub resource: String,
    /// Estimated share of the step time this fault costs.
    pub cost_share: f64,
    pub evidence: Vec<String>,
}

/// A resource's share with a 95% interval over windows.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Limit {
    pub resource: String,
    pub share: f64,
    pub ci_low: f64,
    pub ci_high: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// The top share's interval overlaps the second's.
    Balanced,
    Limited(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub regime: Regime,
    pub faults: Vec<Fault>,
    pub limits: Vec<Limit>,
    pub verdict: Verdict,
    pub notes: Vec<String>,
}

/// Run all checks in the ADR 0013 order. Deterministic.
pub fn detect(obs: &Observation) -> Report {
    let regime = if obs.batch <= SERIALIZED_MAX_BATCH {
        Regime::Serialized
    } else {
        Regime::Load
    };
    let mut notes = Vec::new();
    let agg = Agg::of(obs);
    if obs.windows.len() < 2 {
        notes.push("fewer than 2 windows: intervals have zero width".into());
    }
    if obs.expected.is_none() {
        notes.push("no calibrated rates: residual and rate checks skipped".into());
    }
    let limits = limits(obs, regime);
    let verdict = match limits.as_slice() {
        [top, second, ..] if top.ci_low <= second.ci_high => Verdict::Balanced,
        [top, ..] => Verdict::Limited(top.resource.clone()),
        [] => Verdict::Balanced,
    };
    let mut faults = Vec::new();
    let mut claimed = BTreeSet::new();
    env_checks(obs, &agg, &mut faults, &mut claimed);
    residual_checks(obs, &agg, &mut faults, &claimed);
    faults.sort_by(|a, b| {
        b.cost_share
            .total_cmp(&a.cost_share)
            .then(a.class.cmp(&b.class))
            .then(a.resource.cmp(&b.resource))
    });
    Report {
        regime,
        faults,
        limits,
        verdict,
        notes,
    }
}

/// Windows pooled: sums of wall time, busy time, work units and step split.
struct Agg {
    wall: f64,
    busy: BTreeMap<String, f64>,
    units: BTreeMap<String, f64>,
    split: BTreeMap<String, f64>,
}

impl Agg {
    fn of(obs: &Observation) -> Self {
        let mut a = Agg {
            wall: 0.0,
            busy: BTreeMap::new(),
            units: BTreeMap::new(),
            split: BTreeMap::new(),
        };
        for w in &obs.windows {
            a.wall += w.wall_s;
            for (k, v) in &w.busy_s {
                *a.busy.entry(k.clone()).or_default() += v;
            }
            for (k, v) in &w.bytes {
                *a.units.entry(k.clone()).or_default() += v;
            }
            for (k, v) in &w.split_s {
                *a.split.entry(k.clone()).or_default() += v;
            }
        }
        a
    }

    /// Share of the step the resource was busy.
    fn occ(&self, r: &str) -> f64 {
        if self.wall > 0.0 {
            self.busy.get(r).copied().unwrap_or(0.0) / self.wall
        } else {
            0.0
        }
    }

    fn split_share(&self, key: &str) -> f64 {
        if self.wall > 0.0 {
            self.split.get(key).copied().unwrap_or(0.0) / self.wall
        } else {
            0.0
        }
    }

    /// Measured rate over expected rate; `None` when either is unknown.
    fn rate_ratio(&self, exp: Option<&Expected>, r: &str) -> Option<f64> {
        let rate = *exp?.rates.get(r)?;
        let busy = *self.busy.get(r)?;
        let units = *self.units.get(r)?;
        (rate > 0.0 && busy > 0.0).then(|| units / busy / rate)
    }

    /// Seconds beyond what the expected rate needs for the same work, as a share of the step.
    fn excess_share(&self, exp: Option<&Expected>, r: &str) -> Option<f64> {
        let rate = *exp?.rates.get(r)?;
        let busy = *self.busy.get(r)?;
        let units = *self.units.get(r)?;
        (rate > 0.0 && self.wall > 0.0).then(|| ((busy - units / rate) / self.wall).max(0.0))
    }
}

fn band(exp: Option<&Expected>) -> f64 {
    exp.map_or(DEFAULT_BAND, |e| {
        if e.band > 0.0 { e.band } else { DEFAULT_BAND }
    })
}

/// True when the resource runs slower than calibrated by more than the band. With no calibrated
/// rate the signal alone decides, so this returns true.
fn slow_or_unknown(agg: &Agg, exp: Option<&Expected>, r: &str) -> bool {
    agg.rate_ratio(exp, r).is_none_or(|q| q < 1.0 - band(exp))
}

fn slowdown_evidence(agg: &Agg, exp: Option<&Expected>, r: &str) -> Option<String> {
    agg.rate_ratio(exp, r)
        .map(|q| format!("{r} runs at {q:.2} of its calibrated rate"))
}

fn env_checks(
    obs: &Observation,
    agg: &Agg,
    faults: &mut Vec<Fault>,
    claimed: &mut BTreeSet<String>,
) {
    let env = &obs.env;
    let exp = obs.expected.as_ref();

    // GPU throttle reasons.
    let mut by_gpu: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for t in &env.throttle {
        let reason = serde_json::to_value(t.reason)
            .ok()
            .and_then(|v| v.as_str().map(str::to_owned))
            .unwrap_or_default();
        by_gpu.entry(t.device.clone()).or_default().push(reason);
    }
    for (dev, reasons) in by_gpu {
        let r = format!("gpu:{dev}");
        if agg.occ(&r) < MATERIAL_SHARE || !slow_or_unknown(agg, exp, &r) {
            continue;
        }
        let cost = agg
            .excess_share(exp, &r)
            .unwrap_or(agg.occ(&r) * THROTTLE_DEFAULT_LOSS);
        let mut evidence = vec![format!("throttle reasons: {}", reasons.join(", "))];
        evidence.extend(slowdown_evidence(agg, exp, &r));
        claimed.insert(r.clone());
        faults.push(Fault {
            class: FaultClass::GpuThrottle,
            resource: r,
            cost_share: cost,
            evidence,
        });
    }

    // PCIe link below its maximum width or generation, on a link that carries the step.
    for p in &env.pcie {
        let ratio = p.bandwidth_ratio();
        let r = p.resource();
        if ratio >= 1.0 || agg.occ(&r) < MATERIAL_SHARE || !slow_or_unknown(agg, exp, &r) {
            continue;
        }
        let cost = agg
            .excess_share(exp, &r)
            .unwrap_or(agg.occ(&r) * (1.0 - ratio));
        let mut evidence = vec![format!(
            "{} runs gen{} x{}, maximum gen{} x{}",
            p.device, p.gen_current, p.width_current, p.gen_max, p.width_max
        )];
        evidence.extend(slowdown_evidence(agg, exp, &r));
        claimed.insert(r.clone());
        faults.push(Fault {
            class: FaultClass::PcieDegraded,
            resource: r,
            cost_share: cost,
            evidence,
        });
    }

    // CPU co-tenant.
    if env.cpu_quota_cores > 0.0 && agg.busy.contains_key("cpu") {
        let frac = env.cpu_other_cores / env.cpu_quota_cores;
        if frac >= CPU_OTHER_FRAC
            && agg.occ("cpu") >= MATERIAL_SHARE
            && slow_or_unknown(agg, exp, "cpu")
        {
            let cost = agg
                .excess_share(exp, "cpu")
                .unwrap_or(agg.occ("cpu") * frac.min(1.0));
            let mut evidence = vec![format!(
                "other processes use {:.1} of {:.1} CPU cores",
                env.cpu_other_cores, env.cpu_quota_cores
            )];
            evidence.extend(slowdown_evidence(agg, exp, "cpu"));
            claimed.insert("cpu".into());
            faults.push(Fault {
                class: FaultClass::CpuCotenant,
                resource: "cpu".into(),
                cost_share: cost,
                evidence,
            });
        }
    }

    // Disk with a visible cause: io.max or other readers.
    let disks: Vec<String> = agg
        .busy
        .keys()
        .filter(|k| k.starts_with("disk:"))
        .cloned()
        .collect();
    for r in disks {
        let mut evidence = Vec::new();
        if let Some(m) = env.io_max_bytes_per_s {
            evidence.push(format!("io.max limits reads to {:.0} MB/s", m / 1e6));
        }
        if env.disk_other_read_bytes_per_s >= DISK_OTHER_MIN_BPS {
            evidence.push(format!(
                "other processes read {:.0} MB/s",
                env.disk_other_read_bytes_per_s / 1e6
            ));
        }
        if evidence.is_empty() || agg.occ(&r) < MATERIAL_SHARE || !slow_or_unknown(agg, exp, &r) {
            continue;
        }
        evidence.extend(slowdown_evidence(agg, exp, &r));
        let cost = agg.excess_share(exp, &r).unwrap_or(0.0);
        claimed.insert(r.clone());
        faults.push(Fault {
            class: FaultClass::DiskSlow,
            resource: r,
            cost_share: cost,
            evidence,
        });
    }

    // VRAM held by others while the expert slots shrink or the hit rate drops.
    let pressured = env
        .vram_other_bytes
        .iter()
        .filter_map(|(dev, other)| {
            let total = *env.vram_total_bytes.get(dev)?;
            (total > 0).then(|| (dev, *other as f64 / total as f64))
        })
        .filter(|(_, f)| *f > VRAM_OTHER_FRAC)
        .max_by(|a, b| a.1.total_cmp(&b.1));
    if let Some((dev, frac)) = pressured {
        let deficit = match (env.slots_planned, env.slots_actual) {
            (Some(p), Some(a)) if p > 0 => 1.0 - f64::from(a) / f64::from(p),
            _ => 0.0,
        };
        let hit_low = env.expert_hit_rate.is_some_and(|h| h < HIT_RATE_LOW);
        let waits = agg.split_share("expert_load_wait") + agg.occ("cpu");
        if (deficit > SLOT_DEFICIT || hit_low) && waits >= MATERIAL_SHARE {
            let loss = if deficit > 0.0 {
                deficit
            } else {
                1.0 - env.expert_hit_rate.unwrap_or(1.0)
            };
            let mut evidence = vec![format!(
                "other processes hold {:.0}% of {dev} VRAM",
                frac * 100.0
            )];
            if let (Some(p), Some(a)) = (env.slots_planned, env.slots_actual) {
                evidence.push(format!("{a} of {p} planned expert slots fit"));
            }
            if let Some(h) = env.expert_hit_rate {
                evidence.push(format!("expert hit rate {h:.2}"));
            }
            faults.push(Fault {
                class: FaultClass::VramPressure,
                resource: format!("gpu:{dev}"),
                cost_share: loss * waits,
                evidence,
            });
        }
    }

    swapping(env, agg, exp, faults, claimed);
}

/// Swapping: pages come back from swap (or refault from the file) while the engine stalls on
/// them. The stall has no resource of its own: it lands in whatever touches the pages, compute
/// on the CPU and host work (a memory-capped CPU run: 0.5 s/token in compute, 0.002 in host).
/// PSI says how long tasks stalled on memory; without it, the CPU or host must be material and
/// slower than calibrated (or uncalibrated).
fn swapping(
    env: &EnvSignals,
    agg: &Agg,
    exp: Option<&Expected>,
    faults: &mut Vec<Fault>,
    claimed: &mut BTreeSet<String>,
) {
    let faulted_bps = env.major_faults_per_s * PAGE_BYTES;
    let psi = env.memory_stall_share.unwrap_or(0.0);
    // Major faults alone also count first reads of a mapped file; a memory stall in PSI says
    // they are refaults.
    let paging = env.swap_in_bytes_per_s >= SWAP_IN_MIN_BPS
        || (psi >= PSI_MIN_SHARE && faulted_bps >= SWAP_IN_MIN_BPS);
    if !paging {
        return;
    }
    let mut evidence = vec![format!(
        "swap-in {:.0} MB/s, major faults {:.0} MB/s",
        env.swap_in_bytes_per_s / 1e6,
        faulted_bps / 1e6
    )];
    let stalled: Vec<&str> = ["cpu", "host"]
        .into_iter()
        .filter(|r| agg.occ(r) >= MATERIAL_SHARE && slow_or_unknown(agg, exp, r))
        .collect();
    if psi < MATERIAL_SHARE && stalled.is_empty() {
        return;
    }
    if env.memory_stall_share.is_some() {
        evidence.push(format!(
            "tasks stalled on memory {:.0}% of the time",
            psi * 100.0
        ));
    }
    // PSI counts only some paging stalls (5% on the testbed at a 5x slowdown), so it is a floor.
    let estimate: f64 = stalled
        .iter()
        .map(|r| {
            agg.excess_share(exp, r)
                .unwrap_or(agg.occ(r) * SWAP_DEFAULT_LOSS)
        })
        .sum();
    let cost = psi.max(estimate);
    for r in ["cpu", "host"] {
        evidence.extend(slowdown_evidence(agg, exp, r));
        claimed.insert(r.into());
    }
    faults.push(Fault {
        class: FaultClass::Swapping,
        resource: "ram".into(),
        cost_share: cost.min(1.0),
        evidence,
    });
}

/// Resources slower than calibrated beyond the band that no environment fault explains.
fn residual_checks(
    obs: &Observation,
    agg: &Agg,
    faults: &mut Vec<Fault>,
    claimed: &BTreeSet<String>,
) {
    let exp = obs.expected.as_ref();
    let Some(e) = exp else { return };
    for r in agg.busy.keys() {
        if claimed.contains(r) || agg.occ(r) < RESIDUAL_MIN_SHARE {
            continue;
        }
        let Some(q) = agg.rate_ratio(exp, r) else {
            continue;
        };
        if q >= 1.0 - band(exp) {
            continue;
        }
        let cost = agg.excess_share(exp, r).unwrap_or(0.0);
        let mut evidence = vec![format!(
            "{r} runs at {q:.2} of its calibrated rate (band {:.2})",
            band(Some(e))
        )];
        let class = if r.starts_with("disk:") {
            if let Some(m) = obs.env.io_max_bytes_per_s {
                evidence.push(format!("io.max is {:.0} MB/s", m / 1e6));
            }
            if obs.env.disk_other_read_bytes_per_s > 0.0 {
                evidence.push(format!(
                    "other processes read {:.0} MB/s",
                    obs.env.disk_other_read_bytes_per_s / 1e6
                ));
            }
            FaultClass::DiskSlow
        } else if r.starts_with("link:") {
            FaultClass::LinkSlow
        } else if let Some(dev) = r.strip_prefix("gpu:")
            && obs.env.throttle.iter().any(|t| t.device == dev)
        {
            evidence.push("GPU reports throttle reasons".into());
            FaultClass::GpuThrottle
        } else if r == "cpu"
            && obs.env.cpu_quota_cores > 0.0
            && obs.env.cpu_other_cores / obs.env.cpu_quota_cores >= CPU_OTHER_FRAC_RESIDUAL
        {
            evidence.push(format!(
                "other processes use {:.1} CPU cores",
                obs.env.cpu_other_cores
            ));
            FaultClass::CpuCotenant
        } else {
            FaultClass::ModelMismatch
        };
        faults.push(Fault {
            class,
            resource: r.clone(),
            cost_share: cost,
            evidence,
        });
    }
}

/// Two-sided 95% Student t quantiles for 1 to 30 degrees of freedom.
const T95: [f64; 30] = [
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
    2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
    2.052, 2.048, 2.045, 2.042,
];

/// Mean and 95% t-interval half width of `xs`; zero width for fewer than 2 samples.
fn mean_ci(xs: &[f64]) -> (f64, f64) {
    let n = xs.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mean = xs.iter().sum::<f64>() / n as f64;
    if n < 2 {
        return (mean, 0.0);
    }
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let t = T95.get(n - 2).copied().unwrap_or(1.96);
    (mean, t * (var / n as f64).sqrt())
}

/// Shares per window, then the interval over windows. Serialized: share of the summed demands.
/// Load: utilization against the busiest resource.
fn limits(obs: &Observation, regime: Regime) -> Vec<Limit> {
    let resources: BTreeSet<&String> = obs.windows.iter().flat_map(|w| w.busy_s.keys()).collect();
    let mut per_resource: BTreeMap<&String, Vec<f64>> = BTreeMap::new();
    for w in &obs.windows {
        let denom = match regime {
            Regime::Serialized => w.busy_s.values().sum::<f64>(),
            Regime::Load => w.busy_s.values().copied().fold(0.0, f64::max),
        };
        if denom <= 0.0 {
            continue;
        }
        for r in &resources {
            let busy = w.busy_s.get(*r).copied().unwrap_or(0.0);
            per_resource.entry(r).or_default().push(busy / denom);
        }
    }
    let mut out: Vec<Limit> = per_resource
        .into_iter()
        .map(|(r, xs)| {
            let (share, half) = mean_ci(&xs);
            Limit {
                resource: r.clone(),
                share,
                ci_low: share - half,
                ci_high: share + half,
            }
        })
        .collect();
    out.sort_by(|a, b| {
        b.share
            .total_cmp(&a.share)
            .then(a.resource.cmp(&b.resource))
    });
    out
}
