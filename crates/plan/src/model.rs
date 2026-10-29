//! What gets placed: the model as units with sizes (docs/plan.md, "The placement plan").

use crate::Format;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Size and cost of one unit of a kind; all units of a kind are alike.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct UnitSize {
    pub params: u64,
    pub flops_per_token: f64,
    /// Most precise format available on disk; conversion only goes downward (ADR 0024).
    pub source_format: Format,
}

impl UnitSize {
    pub fn bytes(&self, format: Format) -> u64 {
        format.bytes(self.params)
    }
}

/// Model shape and per-kind unit sizes. Every layer has attention and an MLP; an MoE model
/// (`experts > 0`) also has `experts` routed experts in every layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelDesc {
    pub name: String,
    pub layers: u32,
    pub heads: u32,
    pub kv_heads: u32,
    /// Routed experts per layer; 0 for a dense model.
    #[serde(default)]
    pub experts: u32,
    /// Experts each token uses per layer (top-k).
    #[serde(default)]
    pub experts_used: u32,
    pub embed: UnitSize,
    pub attn: UnitSize,
    /// Dense FFN, or router plus shared experts in an MoE layer.
    pub mlp: UnitSize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expert: Option<UnitSize>,
    pub head: UnitSize,
    pub kv_bytes_per_token_per_layer: u64,
    /// Activation bytes per token crossing a stage boundary.
    pub activation_bytes_per_token: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnitKind {
    Embed,
    Attn,
    Mlp,
    Expert,
    Head,
}

/// One unit of the model. Serialized as `embed`, `layer.3.attn`, `layer.3.mlp`,
/// `layer.3.expert.17` or `head`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum UnitId {
    Embed,
    Attn(u32),
    Mlp(u32),
    Expert(u32, u32),
    Head,
}

impl UnitId {
    pub fn kind(self) -> UnitKind {
        match self {
            Self::Embed => UnitKind::Embed,
            Self::Attn(_) => UnitKind::Attn,
            Self::Mlp(_) => UnitKind::Mlp,
            Self::Expert(..) => UnitKind::Expert,
            Self::Head => UnitKind::Head,
        }
    }

    pub fn layer(self) -> Option<u32> {
        match self {
            Self::Attn(l) | Self::Mlp(l) | Self::Expert(l, _) => Some(l),
            Self::Embed | Self::Head => None,
        }
    }

    pub fn expert(self) -> Option<u32> {
        match self {
            Self::Expert(_, e) => Some(e),
            _ => None,
        }
    }
}

impl fmt::Display for UnitId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Embed => write!(f, "embed"),
            Self::Attn(l) => write!(f, "layer.{l}.attn"),
            Self::Mlp(l) => write!(f, "layer.{l}.mlp"),
            Self::Expert(l, e) => write!(f, "layer.{l}.expert.{e}"),
            Self::Head => write!(f, "head"),
        }
    }
}

impl FromStr for UnitId {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        let bad = || format!("bad unit id {s:?}");
        let num = |p: &str| p.parse::<u32>().map_err(|_| bad());
        let parts: Vec<&str> = s.split('.').collect();
        match parts.as_slice() {
            ["embed"] => Ok(Self::Embed),
            ["head"] => Ok(Self::Head),
            ["layer", l, "attn"] => Ok(Self::Attn(num(l)?)),
            ["layer", l, "mlp"] => Ok(Self::Mlp(num(l)?)),
            ["layer", l, "expert", e] => Ok(Self::Expert(num(l)?, num(e)?)),
            _ => Err(bad()),
        }
    }
}

impl Serialize for UnitId {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for UnitId {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

impl ModelDesc {
    pub fn is_moe(&self) -> bool {
        self.experts > 0
    }

    /// Every unit in model order: embed, per layer attn, mlp, experts, then head.
    pub fn units(&self) -> impl Iterator<Item = UnitId> + '_ {
        let layers = (0..self.layers).flat_map(move |l| {
            [UnitId::Attn(l), UnitId::Mlp(l)]
                .into_iter()
                .chain((0..self.experts).map(move |e| UnitId::Expert(l, e)))
        });
        std::iter::once(UnitId::Embed)
            .chain(layers)
            .chain(std::iter::once(UnitId::Head))
    }

    /// Size of a unit, `None` if the model has no such unit.
    pub fn size(&self, u: UnitId) -> Option<&UnitSize> {
        let in_layers = |l: u32| l < self.layers;
        match u {
            UnitId::Embed => Some(&self.embed),
            UnitId::Head => Some(&self.head),
            UnitId::Attn(l) if in_layers(l) => Some(&self.attn),
            UnitId::Mlp(l) if in_layers(l) => Some(&self.mlp),
            UnitId::Expert(l, e) if in_layers(l) && e < self.experts => self.expert.as_ref(),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_ids_roundtrip_as_strings() {
        for u in [
            UnitId::Embed,
            UnitId::Attn(3),
            UnitId::Mlp(0),
            UnitId::Expert(47, 127),
            UnitId::Head,
        ] {
            assert_eq!(u.to_string().parse::<UnitId>().unwrap(), u);
        }
        assert!("layer.x.attn".parse::<UnitId>().is_err());
        assert!("layer.1.expert".parse::<UnitId>().is_err());
    }
}
