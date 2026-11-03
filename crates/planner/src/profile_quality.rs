//! The measured quality profile (ADR 0048) as a planner quality model, and the format mix the
//! planner picks from it at a byte budget (Phase 3a, ADR 0040 "Quality").
//!
//! Two views of one profile:
//! - [`ProfileQuality`] implements [`QualityModel`] per unit for the search: a group's KL delta
//!   at a format is spread over the units that hold it by their share of the group's params.
//! - The group view: [`estimate_mix`] (the ADR 0048 sum for `{group id: format}`),
//!   [`Correction`] (estimate to measured) and [`mix_at_budget`] (the knapsack).
//!
//! The summed estimate is not the measured KL of a whole mix (on MoE models single groups share a
//! rerouting floor, so the sum overestimates light mixes). [`Correction`] is a power law
//! `measured = exp(a) * estimate^b`, least squares in log space over the profile's `checks`.
//! It is strictly increasing, so it changes no ranking of mixes, only the absolute numbers.

use crate::quality::QualityModel;
use std::collections::BTreeMap;
use std::fmt;
use tetraeder_plan::{
    EstimateError, Format, ModelDesc, QualityGroup, QualityProfile, RoleSize, UnitId,
};

/// Estimated KL of a mix `{group id: format}`: `base.kl + sum (kl(group, format) - base.kl)`.
/// Groups missing from the mix and formats at least as precise as the base count as the base.
pub fn estimate_mix(
    profile: &QualityProfile,
    mix: &BTreeMap<String, Format>,
) -> Result<f64, EstimateError> {
    let mut kl = profile.base.kl;
    for g in &profile.groups {
        let Some(&format) = mix.get(&g.id) else {
            continue;
        };
        if format.at_least(profile.base.format) {
            continue;
        }
        let e = g.entries.iter().find(|e| e.format == format);
        let e = e.ok_or_else(|| EstimateError::MissingEntry {
            group: g.id.clone(),
            format,
        })?;
        kl += e.kl - profile.base.kl;
    }
    Ok(kl)
}

/// Monotone map from the summed estimate to the measured KL: `exp(a) * estimate^b`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Correction {
    pub a: f64,
    pub b: f64,
}

impl Correction {
    pub const IDENTITY: Self = Self { a: 0.0, b: 1.0 };

    /// Least-squares line through `(ln estimate, ln measured)`. Points with a non-positive
    /// coordinate are dropped. Fewer than two usable points, equal estimates or a non-positive
    /// slope (not monotone) give the identity. Deterministic: closed form over the points in
    /// file order.
    pub fn fit(points: &[(f64, f64)]) -> Self {
        let pts: Vec<(f64, f64)> = points
            .iter()
            .filter(|(e, m)| *e > 0.0 && *m > 0.0 && e.is_finite() && m.is_finite())
            .map(|(e, m)| (e.ln(), m.ln()))
            .collect();
        let n = pts.len() as f64;
        if pts.len() < 2 {
            return Self::IDENTITY;
        }
        let mx = pts.iter().map(|p| p.0).sum::<f64>() / n;
        let my = pts.iter().map(|p| p.1).sum::<f64>() / n;
        let sxx: f64 = pts.iter().map(|p| (p.0 - mx).powi(2)).sum();
        let sxy: f64 = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
        if sxx <= 0.0 || sxy <= 0.0 {
            return Self::IDENTITY;
        }
        let b = sxy / sxx;
        Self { a: my - b * mx, b }
    }

    /// Fit on a profile's checks, with the estimate recomputed from each check's formats.
    pub fn from_profile(profile: &QualityProfile) -> Self {
        Self::fit(&check_points(profile))
    }

    pub fn apply(&self, estimate: f64) -> f64 {
        if estimate <= 0.0 {
            0.0
        } else {
            (self.a + self.b * estimate.ln()).exp()
        }
    }
}

/// `(estimate, measured)` of every check whose formats the profile can estimate.
fn check_points(profile: &QualityProfile) -> Vec<(f64, f64)> {
    profile
        .checks
        .iter()
        .filter_map(|c| Some((estimate_mix(profile, &c.formats).ok()?, c.measured)))
        .collect()
}

/// Largest relative error `|measured - predicted| / measured` over the profile's checks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Band {
    /// The summed estimate as is.
    pub raw: f64,
    /// The corrected estimate, on the checks the correction was fitted on (in-sample).
    pub corrected: f64,
    /// The corrected estimate with each check left out of its own fit; `None` below three
    /// checks.
    pub leave_one_out: Option<f64>,
}

fn max_rel(points: &[(f64, f64)], f: impl Fn(f64) -> f64) -> f64 {
    points
        .iter()
        .map(|&(e, m)| (m - f(e)).abs() / m)
        .fold(0.0, f64::max)
}

impl Band {
    pub fn of(profile: &QualityProfile) -> Self {
        let pts = check_points(profile);
        let c = Correction::fit(&pts);
        let loo = (pts.len() >= 3).then(|| {
            (0..pts.len())
                .map(|i| {
                    let rest: Vec<_> = pts
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j != i)
                        .map(|(_, p)| *p)
                        .collect();
                    let c = Correction::fit(&rest);
                    max_rel(&pts[i..=i], |e| c.apply(e))
                })
                .fold(0.0, f64::max)
        });
        Self {
            raw: max_rel(&pts, |e| e),
            corrected: max_rel(&pts, |e| c.apply(e)),
            leave_one_out: loo,
        }
    }
}

/// The planner's choice of one format per profile group.
#[derive(Debug, Clone, PartialEq)]
pub struct Mix {
    /// Profile group id (`attn.v@0-6`, `embed`) to format.
    pub formats: BTreeMap<String, Format>,
    /// Model bytes: tensors outside every group plus the groups in their chosen formats.
    pub bytes: u64,
    /// ADR 0048 sum.
    pub estimate: f64,
    /// [`Correction`] of the estimate.
    pub corrected: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MixError {
    /// The group has no entry at the base format, so the bytes outside the groups are unknown.
    NoBaseEntry {
        group: String,
    },
    BudgetTooSmall {
        budget: u64,
        smallest: u64,
    },
}

impl fmt::Display for MixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoBaseEntry { group } => {
                write!(f, "group {group} has no entry at the base format")
            }
            Self::BudgetTooSmall { budget, smallest } => {
                write!(
                    f,
                    "budget {budget} below the smallest mix ({smallest} bytes)"
                )
            }
        }
    }
}

impl std::error::Error for MixError {}

type Point = (u64, f64, Format);

/// Lower convex hull of `(bytes, kl, format)` by bytes ascending: more bytes always mean less KL
/// and the KL drop per byte falls along the hull.
fn hull(g: &QualityGroup) -> Vec<Point> {
    let mut pts: Vec<Point> = g
        .entries
        .iter()
        .map(|e| (e.bytes, e.kl, e.format))
        .collect();
    pts.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let mut h: Vec<Point> = Vec::new();
    for p in pts {
        if h.last().is_some_and(|l| p.1 >= l.1) {
            continue;
        }
        while h.len() >= 2 {
            let (a, b) = (h[h.len() - 2], h[h.len() - 1]);
            let cross =
                (b.0 as f64 - a.0 as f64) * (p.1 - a.1) - (b.1 - a.1) * (p.0 as f64 - a.0 as f64);
            if cross <= 0.0 {
                h.pop();
            } else {
                break;
            }
        }
        h.push(p);
    }
    h
}

/// Formats per group within `budget` bytes. Every group starts at its smallest format; the hull
/// step with the largest KL drop per byte saved is taken while it fits (least loss per byte
/// first, the order of the search's greedy ladder); then what is left buys any single upgrade
/// that still fits. The correction is increasing, so it does not reorder the steps and the
/// order uses the raw deltas. `estimate` and `corrected` are reported for the result.
///
/// This is the same procedure as the reference knapsack of `scripts/quality/compare_mix.py`,
/// including its tie-breaks, so equal profiles give equal mixes.
pub fn mix_at_budget(profile: &QualityProfile, budget: u64) -> Result<Mix, MixError> {
    let base = &profile.base;
    let mut base_group_bytes = 0u64;
    for g in &profile.groups {
        let e = g.entries.iter().find(|e| e.format == base.format);
        base_group_bytes += e
            .ok_or_else(|| MixError::NoBaseEntry {
                group: g.id.clone(),
            })?
            .bytes;
    }
    let rest = base.bytes.saturating_sub(base_group_bytes);
    let hulls: Vec<Vec<Point>> = profile.groups.iter().map(hull).collect();
    let mut pos = vec![0usize; hulls.len()];
    let mut total = rest
        + hulls
            .iter()
            .filter_map(|h| h.first())
            .map(|p| p.0)
            .sum::<u64>();
    if total > budget {
        return Err(MixError::BudgetTooSmall {
            budget,
            smallest: total,
        });
    }
    loop {
        let mut best: Option<(f64, usize, u64)> = None;
        for (gi, h) in hulls.iter().enumerate() {
            let i = pos[gi];
            if i + 1 < h.len() {
                let db = h[i + 1].0 - h[i].0;
                let dk = h[i].1 - h[i + 1].1;
                let ratio = dk / db as f64;
                if total + db <= budget && best.is_none_or(|b| ratio > b.0) {
                    best = Some((ratio, gi, db));
                }
            }
        }
        let Some((_, gi, db)) = best else { break };
        pos[gi] += 1;
        total += db;
    }
    let mut choice: Vec<Point> = hulls.iter().zip(&pos).map(|(h, &i)| h[i]).collect();
    for (gi, g) in profile.groups.iter().enumerate() {
        let cur = choice[gi];
        let mut by_kl: Vec<_> = g.entries.iter().collect();
        by_kl.sort_by(|a, b| a.kl.total_cmp(&b.kl));
        if let Some(e) = by_kl
            .into_iter()
            .find(|e| e.kl < cur.1 && total - cur.0 + e.bytes <= budget)
        {
            total = total - cur.0 + e.bytes;
            choice[gi] = (e.bytes, e.kl, e.format);
        }
    }
    let formats: BTreeMap<String, Format> = profile
        .groups
        .iter()
        .zip(&choice)
        .map(|(g, c)| (g.id.clone(), c.2))
        .collect();
    let estimate = estimate_mix(profile, &formats).expect("chosen formats have entries");
    Ok(Mix {
        corrected: Correction::from_profile(profile).apply(estimate),
        formats,
        bytes: total,
        estimate,
    })
}

/// Profile group of a GGUF tensor, by llama.cpp's names: `token_embd.weight` is `embed`,
/// `output.weight` is `head`, `blk.N.<suffix>.weight` is the group of its role class and layer
/// span, as `scripts/quality/qlib.py` builds the groups. Norms, the router and other tensors are
/// in no group.
pub fn group_of_tensor(name: &str, layers: u32, layer_span: u32) -> Option<String> {
    match name {
        "token_embd.weight" => return Some("embed".into()),
        "output.weight" => return Some("head".into()),
        _ => {}
    }
    let (layer, rest) = name.strip_prefix("blk.")?.split_once('.')?;
    let class = match rest.strip_suffix(".weight")? {
        "attn_q" | "attn_k" => "attn.q+k",
        "attn_v" => "attn.v",
        "attn_output" => "attn.o",
        "ffn_gate" | "ffn_up" => "mlp.gate+up",
        "ffn_down" => "mlp.down",
        "ffn_gate_exps" | "ffn_up_exps" => "expert.gate+up",
        "ffn_down_exps" => "expert.down",
        _ => return None,
    };
    let l: u32 = layer.parse().ok()?;
    let span = layer_span.max(1);
    let a = l / span * span;
    Some(format!("{class}@{a}-{}", (a + span).min(layers)))
}

/// `{tensor name: format}` (the `--mix` file of `scripts/quality/compare_mix.py`) for a mix over
/// the named tensors. Tensors in no group of the mix are left out and keep the source format.
pub fn tensor_formats<'a>(
    profile: &QualityProfile,
    mix: &Mix,
    tensors: impl IntoIterator<Item = &'a str>,
) -> BTreeMap<String, Format> {
    tensors
        .into_iter()
        .filter_map(|t| {
            let id = group_of_tensor(t, profile.model.layers, profile.layer_span)?;
            Some((t.to_string(), *mix.formats.get(&id)?))
        })
        .collect()
}

/// A profile as the search's quality model, for one model description.
#[derive(Debug, Clone)]
pub struct ProfileQuality {
    profile: QualityProfile,
    desc: ModelDesc,
    correction: Correction,
    /// Params of each group in `desc`, the denominator of a unit's share.
    totals: Vec<u64>,
}

/// Params `u` holds in group `g`; 0 if the group does not cover it.
fn params_in(g: &QualityGroup, desc: &ModelDesc, u: UnitId) -> u64 {
    if !g.units.matches(u) {
        return 0;
    }
    let Some(size) = desc.size(u) else { return 0 };
    if size.roles.is_empty() {
        return size.params;
    }
    let covered = |r: &&RoleSize| g.roles.is_empty() || g.roles.contains(&r.name);
    size.roles.iter().filter(covered).map(|r| r.params).sum()
}

impl ProfileQuality {
    pub fn new(profile: QualityProfile, desc: ModelDesc) -> Self {
        let totals = profile
            .groups
            .iter()
            .map(|g| desc.units().map(|u| params_in(g, &desc, u)).sum())
            .collect();
        let correction = Correction::from_profile(&profile);
        Self {
            profile,
            desc,
            correction,
            totals,
        }
    }

    pub fn profile(&self) -> &QualityProfile {
        &self.profile
    }

    pub fn correction(&self) -> Correction {
        self.correction
    }

    /// ADR 0048 sum for a mix of group formats.
    pub fn estimate(&self, mix: &BTreeMap<String, Format>) -> Result<f64, EstimateError> {
        estimate_mix(&self.profile, mix)
    }

    /// The estimate mapped to the measured scale.
    pub fn corrected(&self, mix: &BTreeMap<String, Format>) -> Result<f64, EstimateError> {
        Ok(self.correction.apply(self.estimate(mix)?))
    }

    /// Error band of the estimate and of the corrected estimate over the profile's checks.
    pub fn band(&self) -> Band {
        Band::of(&self.profile)
    }

    pub fn mix_at_budget(&self, budget: u64) -> Result<Mix, MixError> {
        mix_at_budget(&self.profile, budget)
    }
}

impl QualityModel for ProfileQuality {
    fn base_format(&self, _: UnitId) -> Format {
        self.profile.base.format
    }

    /// The unit's share of each covering group's delta. A format below the base that a covering
    /// group did not measure, or a unit no group covers, is unknown (`None`).
    fn loss(&self, u: UnitId, f: Format) -> Option<f64> {
        let base = &self.profile.base;
        if f.at_least(base.format) {
            return Some(0.0);
        }
        let mut loss = 0.0;
        let mut covered = false;
        for (g, &total) in self.profile.groups.iter().zip(&self.totals) {
            let params = params_in(g, &self.desc, u);
            if params == 0 || total == 0 {
                continue;
            }
            covered = true;
            let e = g.entries.iter().find(|e| e.format == f)?;
            loss += params as f64 / total as f64 * (e.kl - base.kl);
        }
        covered.then_some(loss)
    }

    fn base_kl(&self) -> f64 {
        self.profile.base.kl
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tetraeder_plan::GroupEntry;

    const SMALL: &str = include_str!("../../plan/fixtures/quality_small_moe.json");
    const MODEL: &str = include_str!("../../plan/fixtures/model_small_moe.toml");

    fn small() -> (QualityProfile, ModelDesc) {
        (
            QualityProfile::from_json(SMALL).unwrap(),
            toml::from_str(MODEL).unwrap(),
        )
    }

    /// The small fixture with a base-format entry per group (the fixture predates that rule).
    fn small_with_base() -> QualityProfile {
        let (mut p, _) = small();
        for g in &mut p.groups {
            let q6 = g.entries[0].bytes;
            g.entries.insert(
                0,
                GroupEntry {
                    format: p.base.format,
                    bytes: q6 * 85 / 66,
                    kl: p.base.kl,
                    ppl: None,
                },
            );
        }
        p
    }

    #[test]
    fn fit_recovers_a_power_law_and_is_monotone() {
        let pts: Vec<_> = [0.01, 0.05, 0.2, 1.0]
            .iter()
            .map(|&e: &f64| (e, 0.5 * e.powf(1.5)))
            .collect();
        let c = Correction::fit(&pts);
        assert!((c.b - 1.5).abs() < 1e-9 && (c.a - 0.5f64.ln()).abs() < 1e-9);
        assert!(c.apply(0.1) < c.apply(0.2));
        assert_eq!(c.apply(0.0), 0.0);
        assert_eq!(c.apply(-1.0), 0.0);
    }

    #[test]
    fn fit_falls_back_to_identity() {
        assert_eq!(Correction::fit(&[]), Correction::IDENTITY);
        assert_eq!(Correction::fit(&[(0.1, 0.2)]), Correction::IDENTITY);
        // Equal estimates, and a decreasing relation, are not usable.
        assert_eq!(
            Correction::fit(&[(0.1, 0.2), (0.1, 0.3)]),
            Correction::IDENTITY
        );
        assert_eq!(
            Correction::fit(&[(0.1, 0.3), (0.2, 0.2)]),
            Correction::IDENTITY
        );
        assert_eq!(Correction::IDENTITY.apply(0.25), 0.25);
    }

    #[test]
    fn band_of_a_profile_without_checks_is_zero() {
        let (p, _) = small();
        let b = Band::of(&p);
        assert_eq!((b.raw, b.corrected, b.leave_one_out), (0.0, 0.0, None));
    }

    #[test]
    fn estimate_mix_matches_the_adr_sum() {
        let (p, _) = small();
        let head = p.group("head").unwrap();
        let q4 = head.entries.iter().find(|e| e.format == Format::Q4_K);
        let mix: BTreeMap<_, _> = [("head".to_string(), Format::Q4_K)].into();
        let e = estimate_mix(&p, &mix).unwrap();
        assert!((e - q4.unwrap().kl).abs() < 1e-12);
        let bad: BTreeMap<_, _> = [("head".to_string(), Format::Q3_K)].into();
        assert!(estimate_mix(&p, &bad).is_err());
        assert_eq!(estimate_mix(&p, &BTreeMap::new()).unwrap(), p.base.kl);
    }

    #[test]
    fn mix_at_budget_fits_and_improves_with_bytes() {
        let p = small_with_base();
        let sum = |f: &dyn Fn(&QualityGroup) -> u64| p.groups.iter().map(f).sum::<u64>();
        let smallest = sum(&|g| g.entries.iter().map(|e| e.bytes).min().unwrap());
        let in_base = sum(&|g| {
            g.entries
                .iter()
                .find(|e| e.format == p.base.format)
                .unwrap()
                .bytes
        });
        let rest = p.base.bytes - in_base;
        let floor = rest + smallest;
        assert_eq!(
            mix_at_budget(&p, floor - 1),
            Err(MixError::BudgetTooSmall {
                budget: floor - 1,
                smallest: floor
            })
        );
        let mut last = f64::INFINITY;
        for extra in [0, 1 << 20, 1 << 22, 1 << 24, 1 << 26, 1 << 30] {
            let m = mix_at_budget(&p, floor + extra).unwrap();
            assert!(m.bytes <= floor + extra);
            assert!(m.estimate <= last + 1e-12, "{extra}");
            last = m.estimate;
        }
        // Room for everything: every group at the base.
        let m = mix_at_budget(&p, p.base.bytes).unwrap();
        assert!((m.estimate - p.base.kl).abs() < 1e-12);
        // Without base entries the bytes outside the groups are unknown.
        let (bare, _) = small();
        assert!(matches!(
            mix_at_budget(&bare, 1 << 40),
            Err(MixError::NoBaseEntry { .. })
        ));
    }

    #[test]
    fn tensor_names_map_to_groups() {
        let g = |n| group_of_tensor(n, 36, 12);
        assert_eq!(g("token_embd.weight").as_deref(), Some("embed"));
        assert_eq!(g("output.weight").as_deref(), Some("head"));
        assert_eq!(g("blk.13.attn_k.weight").as_deref(), Some("attn.q+k@12-24"));
        assert_eq!(
            g("blk.35.ffn_down_exps.weight").as_deref(),
            Some("expert.down@24-36")
        );
        assert_eq!(
            group_of_tensor("blk.38.attn_v.weight", 40, 6).as_deref(),
            Some("attn.v@36-40")
        );
        assert_eq!(g("blk.3.attn_q_norm.weight"), None);
        assert_eq!(g("blk.3.ffn_gate_inp.weight"), None);
        assert_eq!(g("output_norm.weight"), None);
    }

    #[test]
    fn unit_losses_are_shares_of_the_group_deltas() {
        let (p, d) = small();
        let q = ProfileQuality::new(p.clone(), d.clone());
        assert_eq!(q.base_kl(), p.base.kl);
        assert_eq!(q.base_format(UnitId::Head), Format::Q8_0);
        assert_eq!(q.loss(UnitId::Head, Format::F16), Some(0.0));
        let delta = |id: &str, f: Format| {
            let e = p.group(id).unwrap().entries.iter().find(|e| e.format == f);
            e.unwrap().kl - p.base.kl
        };
        let close = |a: f64, b: f64| assert!((a - b).abs() < 1e-12, "{a} != {b}");
        close(
            q.loss(UnitId::Head, Format::Q4_K).unwrap(),
            delta("head", Format::Q4_K),
        );
        // Attention of layer 0: q and k are 10M of the 20M params of the 2 layers of attn.qk.0;
        // v is 2M of the 8M params of attn.v (4 layers).
        close(
            q.loss(UnitId::Attn(0), Format::Q4_K).unwrap(),
            0.5 * delta("attn.qk.0", Format::Q4_K) + 0.25 * delta("attn.v", Format::Q4_K),
        );
        // One expert holds 1/8 of each expert group of its layer, a layer 1/4 of the group.
        let e = delta("expert.down", Format::Q6_K) + delta("expert.gateup", Format::Q6_K);
        close(
            q.loss(UnitId::Expert(1, 3), Format::Q6_K).unwrap(),
            e / 32.0,
        );
        // Q3_K was never measured; attn.o and mlp are in no group.
        assert_eq!(q.loss(UnitId::Head, Format::Q3_K), None);
        assert_eq!(q.loss(UnitId::Mlp(0), Format::Q4_K), None);
        // The units of a group sum to its delta.
        let sum: f64 = d
            .units()
            .filter_map(|u| match u {
                UnitId::Expert(..) => q.loss(u, Format::Q4_K),
                _ => None,
            })
            .sum();
        close(
            sum,
            delta("expert.down", Format::Q4_K) + delta("expert.gateup", Format::Q4_K),
        );
    }
}
