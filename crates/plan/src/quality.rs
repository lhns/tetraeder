//! Quality profile: measured KL cost of each (tensor group, format) choice (ADR 0042), and the
//! additive estimate of a plan's KL from it.

use crate::{Format, ModelDesc, Plan, UnitKind, UnitSel};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// Schema number of the profile file; a breaking change raises it.
pub const QUALITY_SCHEMA: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfileModel {
    pub name: String,
    pub arch: String,
    pub source: String,
    pub source_format: Format,
    pub fingerprint: String,
    pub layers: u32,
    pub experts: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Calibration {
    pub text: String,
    pub sha256: String,
    pub ctx: u32,
    pub chunks: u32,
    pub scored_tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImatrixRef {
    pub sha256: String,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Reference {
    pub format: Format,
}

/// The whole model at the base format.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BaseEntry {
    pub format: Format,
    pub kl: f64,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroupEntry {
    pub format: Format,
    /// The group's size in this format.
    pub bytes: u64,
    pub kl: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ppl: Option<f64>,
}

/// Units (and roles within them) measured together.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityGroup {
    pub id: String,
    pub units: UnitSel,
    /// Empty: every role of the selected units.
    #[serde(default)]
    pub roles: Vec<String>,
    pub params: u64,
    pub entries: Vec<GroupEntry>,
}

/// Measured KL of a whole mix against its estimate (the additivity test of ADR 0024).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityCheck {
    pub name: String,
    pub formats: BTreeMap<String, Format>,
    pub estimate: f64,
    pub measured: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationCost {
    pub seconds: f64,
    pub gpu: String,
    pub host: String,
    pub threads: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityProfile {
    pub schema: u32,
    pub model: ProfileModel,
    /// ggml pin tag.
    pub ggml: String,
    pub calibration: Calibration,
    #[serde(default)]
    pub imatrix: Option<ImatrixRef>,
    pub reference: Reference,
    pub base: BaseEntry,
    /// Layers per group of the per-layer-span groups.
    pub layer_span: u32,
    pub groups: Vec<QualityGroup>,
    #[serde(default)]
    pub checks: Vec<QualityCheck>,
    #[serde(default)]
    pub cost: Option<CalibrationCost>,
}

#[derive(Debug)]
pub enum QualityError {
    Json(serde_json::Error),
    Schema { found: u32 },
}

impl fmt::Display for QualityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(e) => write!(f, "quality profile: {e}"),
            Self::Schema { found } => {
                write!(
                    f,
                    "quality profile schema {found}, expected {QUALITY_SCHEMA}"
                )
            }
        }
    }
}

impl std::error::Error for QualityError {}

#[derive(Debug, Clone, PartialEq)]
pub enum QualityIssue {
    Schema {
        found: u32,
    },
    DuplicateGroup {
        id: String,
    },
    /// A role none of the group's unit kinds has.
    UnknownRole {
        group: String,
        role: String,
    },
    /// An empty layer or expert span, or one beyond the model.
    SpanOutsideModel {
        group: String,
        detail: String,
    },
    DuplicateFormat {
        group: String,
        format: Format,
    },
    /// `group` is `None` for the base entry.
    BadKl {
        group: Option<String>,
        format: Format,
        kl: f64,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum EstimateError {
    NoReplica,
    /// The plan puts params of `group` in `format`, which the profile never measured.
    MissingEntry {
        group: String,
        format: Format,
    },
}

impl fmt::Display for EstimateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoReplica => write!(f, "plan has no replica"),
            Self::MissingEntry { group, format } => {
                write!(f, "no measurement for group {group} in {format:?}")
            }
        }
    }
}

impl std::error::Error for EstimateError {}

impl QualityProfile {
    pub fn from_json(s: &str) -> Result<Self, QualityError> {
        #[derive(Deserialize)]
        struct Head {
            schema: u32,
        }
        let head: Head = serde_json::from_str(s).map_err(QualityError::Json)?;
        if head.schema != QUALITY_SCHEMA {
            return Err(QualityError::Schema { found: head.schema });
        }
        serde_json::from_str(s).map_err(QualityError::Json)
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn group(&self, id: &str) -> Option<&QualityGroup> {
        self.groups.iter().find(|g| g.id == id)
    }

    /// Consistency of the profile and its fit to `desc`.
    pub fn validate(&self, desc: &ModelDesc) -> Vec<QualityIssue> {
        let mut out = Vec::new();
        if self.schema != QUALITY_SCHEMA {
            out.push(QualityIssue::Schema { found: self.schema });
        }
        let bad_kl = |kl: f64| !kl.is_finite() || kl < 0.0;
        if bad_kl(self.base.kl) {
            out.push(QualityIssue::BadKl {
                group: None,
                format: self.base.format,
                kl: self.base.kl,
            });
        }
        let mut ids = BTreeSet::new();
        for g in &self.groups {
            let id = || g.id.clone();
            if !ids.insert(&g.id) {
                out.push(QualityIssue::DuplicateGroup { id: id() });
            }
            let kinds: &[UnitKind] = if g.units.kinds.is_empty() {
                &UnitKind::ALL
            } else {
                &g.units.kinds
            };
            for role in &g.roles {
                if !kinds.iter().any(|k| k.roles().contains(&role.as_str())) {
                    out.push(QualityIssue::UnknownRole {
                        group: id(),
                        role: role.clone(),
                    });
                }
            }
            for (what, span, n) in [
                ("layers", g.units.layers, desc.layers),
                ("experts", g.units.experts, desc.experts),
            ] {
                if let Some(s) = span.filter(|s| s.is_empty() || s.1 > n) {
                    out.push(QualityIssue::SpanOutsideModel {
                        group: id(),
                        detail: format!("{what} {}..{}, model has {n}", s.0, s.1),
                    });
                }
            }
            let mut seen = Vec::new();
            for e in &g.entries {
                if seen.contains(&e.format) {
                    out.push(QualityIssue::DuplicateFormat {
                        group: id(),
                        format: e.format,
                    });
                }
                seen.push(e.format);
                if bad_kl(e.kl) {
                    out.push(QualityIssue::BadKl {
                        group: Some(id()),
                        format: e.format,
                        kl: e.kl,
                    });
                }
            }
        }
        out
    }

    /// Estimated KL of replica 0 of `plan`: `base.kl + sum w * (kl(group, format) - base.kl)`,
    /// `w` being the share of a group's params the plan puts in that format. A group covers the
    /// units its selector matches and, in a unit with described roles, its roles (all when
    /// empty); a unit without roles counts as a whole. Formats at least as precise as the base
    /// count as the base; roles outside every group add nothing. Params come from `desc`, so
    /// `w` sums to 1 over each group.
    pub fn estimate(&self, plan: &Plan, desc: &ModelDesc) -> Result<f64, EstimateError> {
        if plan.replicas.is_empty() {
            return Err(EstimateError::NoReplica);
        }
        let units = plan.unit_formats(desc, 0);
        let mut kl = self.base.kl;
        for g in &self.groups {
            let mut by_format: BTreeMap<Format, u64> = BTreeMap::new();
            for (u, uf) in units.iter().filter(|(u, _)| g.units.matches(**u)) {
                let Some(size) = desc.size(*u) else { continue };
                if size.roles.is_empty() {
                    *by_format.entry(uf.format).or_default() += size.params;
                    continue;
                }
                for r in &size.roles {
                    if g.roles.is_empty() || g.roles.contains(&r.name) {
                        let f = uf.roles.get(&r.name).copied().unwrap_or(uf.format);
                        *by_format.entry(f).or_default() += r.params;
                    }
                }
            }
            let total: u64 = by_format.values().sum();
            for (format, params) in by_format {
                if total == 0 || format.at_least(self.base.format) {
                    continue;
                }
                let entry = g.entries.iter().find(|e| e.format == format);
                let entry = entry.ok_or_else(|| EstimateError::MissingEntry {
                    group: g.id.clone(),
                    format,
                })?;
                kl += params as f64 / total as f64 * (entry.kl - self.base.kl);
            }
        }
        Ok(kl)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Assign, Replica, Span, Stage};

    const PROFILE: &str = include_str!("../fixtures/quality_small_moe.json");
    const MODEL: &str = include_str!("../fixtures/model_small_moe.toml");

    fn profile() -> QualityProfile {
        QualityProfile::from_json(PROFILE).unwrap()
    }

    fn desc() -> ModelDesc {
        toml::from_str(MODEL).unwrap()
    }

    /// One stage over all layers: embed at Q8_0, head at `head`, attention, mlp and experts at
    /// `rest`, with `expert_roles` overriding expert roles.
    fn plan(head: Format, rest: Format, expert_roles: &[(&str, Format)]) -> Plan {
        let rule = |kinds: &[UnitKind], format, roles: &[(&str, Format)]| Assign {
            units: UnitSel {
                kinds: kinds.to_vec(),
                ..Default::default()
            },
            format: Some(format),
            role_formats: roles.iter().map(|(r, f)| (r.to_string(), *f)).collect(),
            ..Default::default()
        };
        Plan {
            model: "small-moe".into(),
            knobs: Default::default(),
            replicas: vec![Replica {
                stages: vec![Stage {
                    layers: Span(0, 4),
                    assign: vec![
                        rule(&[UnitKind::Embed], Format::Q8_0, &[]),
                        rule(&[UnitKind::Head], head, &[]),
                        rule(&[UnitKind::Attn, UnitKind::Mlp], rest, &[]),
                        rule(&[UnitKind::Expert], rest, expert_roles),
                    ],
                    ..Default::default()
                }],
            }],
        }
    }

    fn kl(group: &str, format: Format) -> f64 {
        let p = profile();
        let entries = &p.group(group).unwrap().entries;
        entries.iter().find(|e| e.format == format).unwrap().kl
    }

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "{a} != {b}");
    }

    #[test]
    fn fixture_loads_validates_and_roundtrips() {
        let p = profile();
        assert_eq!(p.groups.len(), 7);
        assert!(p.validate(&desc()).is_empty());
        assert_eq!(QualityProfile::from_json(&p.to_json().unwrap()).unwrap(), p);
        // Unknown fields are ignored, optional ones may be absent.
        let mut v: serde_json::Value = serde_json::from_str(PROFILE).unwrap();
        v["future"] = 1.into();
        v.as_object_mut().unwrap().remove("cost");
        v.as_object_mut().unwrap().remove("checks");
        assert!(QualityProfile::from_json(&v.to_string()).is_ok());
    }

    #[test]
    fn other_schema_is_rejected() {
        let s = PROFILE.replacen("\"schema\": 1", "\"schema\": 2", 1);
        assert!(matches!(
            QualityProfile::from_json(&s),
            Err(QualityError::Schema { found: 2 })
        ));
    }

    #[test]
    fn estimate_uniform_group() {
        let (p, d) = (profile(), desc());
        let base = plan(Format::Q8_0, Format::Q8_0, &[]);
        close(p.estimate(&base, &d).unwrap(), p.base.kl);
        // The whole head group at Q4_K.
        let e = p.estimate(&plan(Format::Q4_K, Format::Q8_0, &[]), &d);
        close(e.unwrap(), kl("head", Format::Q4_K));
    }

    #[test]
    fn estimate_half_and_half_group() {
        let (p, d) = (profile(), desc());
        // expert.gateup covers gate and up, equal in params: gate at Q4_K is half of it.
        let roles = [("expert.gate", Format::Q4_K)];
        let e = p.estimate(&plan(Format::Q8_0, Format::Q8_0, &roles), &d);
        close(
            e.unwrap(),
            p.base.kl + 0.5 * (kl("expert.gateup", Format::Q4_K) - p.base.kl),
        );
    }

    #[test]
    fn estimate_precise_formats_and_uncovered_roles_add_nothing() {
        let (p, d) = (profile(), desc());
        let e = p.estimate(&plan(Format::F16, Format::F16, &[]), &d);
        close(e.unwrap(), p.base.kl);
        // attn.o and mlp.down are in no group.
        let mut pl = plan(Format::Q8_0, Format::Q8_0, &[]);
        let attn = &mut pl.replicas[0].stages[0].assign[2];
        attn.role_formats.insert("attn.o".into(), Format::Q4_K);
        attn.role_formats.insert("mlp.down".into(), Format::Q4_K);
        close(p.estimate(&pl, &d).unwrap(), p.base.kl);
    }

    #[test]
    fn estimate_missing_entry_is_an_error() {
        let (p, d) = (profile(), desc());
        let e = p.estimate(&plan(Format::Q3_K, Format::Q8_0, &[]), &d);
        assert_eq!(
            e,
            Err(EstimateError::MissingEntry {
                group: "head".into(),
                format: Format::Q3_K
            })
        );
    }

    #[test]
    fn validate_catches_bad_role_and_span() {
        let d = desc();
        let mut p = profile();
        p.groups[2].roles.push("mlp.up".into());
        p.groups[3].units.layers = Some(Span(2, 9));
        let found = p.validate(&d);
        assert!(found.iter().any(|i| matches!(i,
            QualityIssue::UnknownRole { group, role }
                if group == "attn.qk.0" && role == "mlp.up")));
        assert!(found.iter().any(|i| matches!(i,
            QualityIssue::SpanOutsideModel { group, .. } if group == "attn.qk.1")));

        let mut p = profile();
        p.groups[1].id = "embed".into();
        let f = p.groups[0].entries[0].format;
        p.groups[0].entries[1].format = f;
        p.groups[0].entries[0].kl = f64::NAN;
        let found = p.validate(&d);
        assert!(
            found
                .iter()
                .any(|i| matches!(i, QualityIssue::DuplicateGroup { .. }))
        );
        assert!(
            found
                .iter()
                .any(|i| matches!(i, QualityIssue::DuplicateFormat { .. }))
        );
        assert!(
            found
                .iter()
                .any(|i| matches!(i, QualityIssue::BadKl { .. }))
        );
    }
}
