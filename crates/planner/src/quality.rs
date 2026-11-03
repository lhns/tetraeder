//! Quality model for the search (ADR 0024, 0025, 0048).
//!
//! A plan's quality loss is `base_kl + sum over units of loss(unit, format)`, the additive
//! estimate of ADR 0048: `loss` is the KL increase over the base format for the unit's share of
//! its measured group (the group's delta weighted by the unit's share of the group's params).
//! `None` means no measurement, so the search must not choose that format.
//!
//! The adapter for `tetraeder_plan::QualityProfile` (the ADR 0048 file) comes when that type is on
//! main; it implements [`QualityModel`] by spreading each group's delta over its units by params.

use tetraeder_plan::{Format, ModelDesc, Span, UnitId, UnitKind};

pub trait QualityModel {
    /// The format the base KL is measured at for this unit.
    fn base_format(&self, u: UnitId) -> Format;

    /// KL increase over the base for this unit stored as `f`; 0 for formats at least as precise
    /// as the base. `None`: unknown, must not be chosen.
    fn loss(&self, u: UnitId, f: Format) -> Option<f64>;

    /// KL of the model at the base formats.
    fn base_kl(&self) -> f64 {
        0.0
    }
}

/// Source format per unit kind of a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Sources {
    embed: Format,
    attn: Format,
    mlp: Format,
    expert: Format,
    head: Format,
}

impl Sources {
    fn of(m: &ModelDesc) -> Self {
        Self {
            embed: m.embed.source_format,
            attn: m.attn.source_format,
            mlp: m.mlp.source_format,
            expert: m.expert.map_or(m.mlp.source_format, |e| e.source_format),
            head: m.head.source_format,
        }
    }

    fn get(&self, kind: UnitKind) -> Format {
        match kind {
            UnitKind::Embed => self.embed,
            UnitKind::Attn => self.attn,
            UnitKind::Mlp => self.mlp,
            UnitKind::Expert => self.expert,
            UnitKind::Head => self.head,
        }
    }
}

/// Keeps every unit at its source format: only that format is allowed, at no loss.
#[derive(Debug, Clone)]
pub struct NullQuality {
    src: Sources,
}

impl NullQuality {
    pub fn new(model: &ModelDesc) -> Self {
        Self {
            src: Sources::of(model),
        }
    }
}

impl QualityModel for NullQuality {
    fn base_format(&self, u: UnitId) -> Format {
        self.src.get(u.kind())
    }

    fn loss(&self, u: UnitId, f: Format) -> Option<f64> {
        (f == self.base_format(u)).then_some(0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Entry {
    kind: UnitKind,
    layers: Span,
    format: Format,
    delta: f64,
}

/// Per (unit kind, layer span, format) deltas, for tests and demos. The base format of each kind
/// is the model's source format. A delta is the KL increase when every unit of the kind in the
/// span goes to the format; one unit gets its equal share (units of a kind are alike).
#[derive(Debug, Clone)]
pub struct TableQuality {
    src: Sources,
    layers: u32,
    experts: u32,
    base_kl: f64,
    entries: Vec<Entry>,
}

impl TableQuality {
    pub fn new(model: &ModelDesc, base_kl: f64) -> Self {
        Self {
            src: Sources::of(model),
            layers: model.layers,
            experts: model.experts,
            base_kl,
            entries: Vec::new(),
        }
    }

    /// Add a delta. `layers` is ignored for embed and head.
    pub fn with(mut self, kind: UnitKind, layers: Span, format: Format, delta: f64) -> Self {
        self.entries.push(Entry {
            kind,
            layers,
            format,
            delta,
        });
        self
    }

    fn units_in(&self, e: &Entry) -> u64 {
        let n = u64::from(e.layers.1.min(self.layers).saturating_sub(e.layers.0));
        match e.kind {
            UnitKind::Embed | UnitKind::Head => 1,
            UnitKind::Attn | UnitKind::Mlp => n,
            UnitKind::Expert => n * u64::from(self.experts),
        }
    }
}

impl QualityModel for TableQuality {
    fn base_format(&self, u: UnitId) -> Format {
        self.src.get(u.kind())
    }

    fn loss(&self, u: UnitId, f: Format) -> Option<f64> {
        if f.at_least(self.base_format(u)) {
            return Some(0.0);
        }
        let hit = self.entries.iter().find(|e| {
            e.kind == u.kind() && e.format == f && u.layer().is_none_or(|l| e.layers.contains(l))
        })?;
        let n = self.units_in(hit);
        (n > 0).then(|| hit.delta / n as f64)
    }

    fn base_kl(&self) -> f64 {
        self.base_kl
    }
}

/// Synthetic model: loss grows with the square of the relative bit loss and is spread over units
/// by params, so the whole model at `f` costs `scale * ((b0 - b) / b0)^2` when all kinds share a
/// base. Every format is known. For tests and demos.
#[derive(Debug, Clone)]
pub struct BitsQuality {
    src: Sources,
    total_params: f64,
    sizes: [(UnitKind, u64); 5],
    scale: f64,
}

impl BitsQuality {
    pub fn new(model: &ModelDesc, scale: f64) -> Self {
        let layers = u64::from(model.layers);
        let experts = u64::from(model.experts);
        let sizes = [
            (UnitKind::Embed, model.embed.params),
            (UnitKind::Attn, model.attn.params),
            (UnitKind::Mlp, model.mlp.params),
            (UnitKind::Expert, model.expert.map_or(0, |e| e.params)),
            (UnitKind::Head, model.head.params),
        ];
        let total = (sizes[0].1 + sizes[4].1) as f64
            + layers as f64 * (sizes[1].1 + sizes[2].1 + experts * sizes[3].1) as f64;
        Self {
            src: Sources::of(model),
            total_params: total.max(1.0),
            sizes,
            scale,
        }
    }
}

impl QualityModel for BitsQuality {
    fn base_format(&self, u: UnitId) -> Format {
        self.src.get(u.kind())
    }

    fn loss(&self, u: UnitId, f: Format) -> Option<f64> {
        let base = self.base_format(u);
        if f.at_least(base) {
            return Some(0.0);
        }
        let params = self.sizes.iter().find(|(k, _)| *k == u.kind())?.1 as f64;
        let x = (base.bits_per_weight() - f.bits_per_weight()) / base.bits_per_weight();
        Some(self.scale * params / self.total_params * x * x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> ModelDesc {
        toml::from_str(include_str!("../../plan/fixtures/model_small_moe.toml")).unwrap()
    }

    #[test]
    fn null_allows_only_the_source_format() {
        let m = model();
        let q = NullQuality::new(&m);
        assert_eq!(q.loss(UnitId::Attn(0), Format::F16), Some(0.0));
        assert_eq!(q.loss(UnitId::Attn(0), Format::Q4_K), None);
        assert_eq!(q.base_format(UnitId::Expert(1, 2)), Format::F16);
        assert_eq!(q.base_kl(), 0.0);
    }

    #[test]
    fn table_spreads_a_delta_over_the_units_of_its_span() {
        let m = model();
        let q = TableQuality::new(&m, 0.01)
            .with(UnitKind::Attn, Span(0, 2), Format::Q4_K, 0.04)
            .with(UnitKind::Expert, Span(0, 4), Format::Q4_K, 0.32)
            .with(UnitKind::Head, Span(0, 0), Format::Q4_K, 0.5);
        assert_eq!(q.loss(UnitId::Attn(1), Format::Q4_K), Some(0.02));
        assert_eq!(q.loss(UnitId::Attn(2), Format::Q4_K), None);
        assert_eq!(q.loss(UnitId::Expert(3, 7), Format::Q4_K), Some(0.01));
        assert_eq!(q.loss(UnitId::Head, Format::Q4_K), Some(0.5));
        assert_eq!(q.loss(UnitId::Head, Format::F32), Some(0.0));
        assert_eq!(q.loss(UnitId::Attn(0), Format::Q2_K), None);
        assert_eq!(q.base_kl(), 0.01);
    }

    #[test]
    fn bits_quality_sums_to_the_scale_for_a_uniform_drop() {
        let m = model();
        let q = BitsQuality::new(&m, 1.0);
        let all = m
            .units()
            .map(|u| q.loss(u, Format::Q8_0).unwrap())
            .sum::<f64>();
        let x = (16.0 - 8.5) / 16.0;
        assert!((all - x * x).abs() < 1e-12, "{all}");
        let low = m
            .units()
            .map(|u| q.loss(u, Format::Q2_K).unwrap())
            .sum::<f64>();
        assert!(low > all);
    }
}
