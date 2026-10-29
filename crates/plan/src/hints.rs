//! Limits and hints (ADR 0025, docs/plan.md "Hints"): planner input next to the plan.
//!
//! The validator enforces the hard device and format rules (`pin`, `forbid`, `min_format`,
//! `format_floor`); everything else steers the planner. A partial plan is a `Plan` with open
//! fields, not a hint.

use crate::{Format, UnitSel};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use tetraeder_profiler::Tier;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Objective {
    Latency,
    Throughput,
}

/// Units must be homed or computed on `device`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pin {
    pub units: UnitSel,
    pub device: String,
}

/// Units (all when `units` is absent) must not be homed or computed on `device`, nor homed on a
/// device of `tier`. At least one of `device` and `tier` is set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Forbid {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub units: Option<UnitSel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier: Option<Tier>,
}

/// Units must use a format at least as precise as `format`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FormatFloor {
    pub units: UnitSel,
    pub format: Format,
}

/// Soft cost term: a positive weight attracts placement to the device.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Prefer {
    pub device: String,
    pub weight: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Level {
    Low,
    Medium,
    High,
}

/// Seeds the workload profile until telemetry measures it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct WorkloadHints {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_reuse: Option<Level>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub typical_prompt: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub concurrency: Option<u32>,
}

/// Local wall-clock window, `"HH:MM"`, in which no replanning happens.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Freeze {
    pub from: String,
    pub to: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Stability {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_switches_per_hour: Option<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub freeze: Vec<Freeze>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Hints {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective: Option<Objective>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_tok_s: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p95_ttft_ms: Option<f64>,
    /// KL divergence from the full-precision logits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_quality_loss: Option<f64>,
    /// Floor for every unit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_format: Option<Format>,
    /// Floors for selected units, e.g. layers 0..2 at Q8_0.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub format_floor: Vec<FormatFloor>,
    /// 0 picks the fastest plan on the front, 1 the best-quality one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tradeoff: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pin: Vec<Pin>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub forbid: Vec<Forbid>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prefer: Vec<Prefer>,
    /// Resource cost per device id.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub cost: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_watts: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workload: Option<WorkloadHints>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stability: Option<Stability>,
    /// Advisor actions applied without asking, e.g. `batch`, `kv_budget`, `hot_set`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub auto_apply: Vec<String>,
}

impl Hints {
    pub fn to_toml(&self) -> Result<String, toml::ser::Error> {
        toml::to_string_pretty(self)
    }

    pub fn from_toml(s: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(s)
    }

    /// Range checks on the values themselves (independent of any plan).
    pub fn check(&self) -> Result<(), Vec<String>> {
        let mut e = Vec::new();
        let positive = |name: &str, v: Option<f64>, e: &mut Vec<String>| {
            if let Some(v) = v
                && !(v > 0.0 && v.is_finite())
            {
                e.push(format!("{name} must be positive, got {v}"));
            }
        };
        positive("target_tok_s", self.target_tok_s, &mut e);
        positive("p95_ttft_ms", self.p95_ttft_ms, &mut e);
        positive("max_watts", self.max_watts, &mut e);
        if let Some(q) = self.max_quality_loss
            && !(q >= 0.0 && q.is_finite())
        {
            e.push(format!("max_quality_loss must be at least 0, got {q}"));
        }
        if let Some(t) = self.tradeoff
            && !(0.0..=1.0).contains(&t)
        {
            e.push(format!("tradeoff must be in [0, 1], got {t}"));
        }
        for f in &self.forbid {
            if f.device.is_none() && f.tier.is_none() {
                e.push("forbid needs a device or a tier".into());
            }
        }
        for (d, c) in &self.cost {
            if !(*c >= 0.0 && c.is_finite()) {
                e.push(format!("cost of {d} must be at least 0, got {c}"));
            }
        }
        if e.is_empty() { Ok(()) } else { Err(e) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hints_table_examples_parse() {
        let h = Hints::from_toml(
            r#"
objective = "latency"
p95_ttft_ms = 2000.0
target_tok_s = 20.0
max_quality_loss = 0.02
min_format = "Q4_K"
tradeoff = 0.7
max_watts = 900.0
auto_apply = ["batch", "kv_budget", "hot_set"]

[[format_floor]]
units = { layers = [0, 2] }
format = "Q8_0"

[[pin]]
units = { layers = [0, 8] }
device = "gpu0"

[[forbid]]
units = { kinds = ["embed", "attn", "mlp", "head"] }
tier = "nvme"

[[prefer]]
device = "gpu0"
weight = 0.2

[cost]
gpu1 = 0.3

[workload]
prefix_reuse = "high"
typical_prompt = 30000
concurrency = 4

[stability]
max_switches_per_hour = 1
freeze = [{ from = "09:00", to = "18:00" }]
"#,
        )
        .unwrap();
        h.check().unwrap();
        assert_eq!(Hints::from_toml(&h.to_toml().unwrap()).unwrap(), h);
    }

    #[test]
    fn check_rejects_out_of_range() {
        let h = Hints {
            tradeoff: Some(1.5),
            target_tok_s: Some(-1.0),
            forbid: vec![Forbid {
                units: None,
                device: None,
                tier: None,
            }],
            ..Default::default()
        };
        assert_eq!(h.check().unwrap_err().len(), 3);
    }
}
