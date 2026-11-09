//! KV cache format of an attention unit (ADR 0050 b): K and V each in one ggml format.

use crate::Format;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Formats of the K and V cache rows. `Default` is F16/F16, what plans without the field mean.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct KvFormat {
    pub k: Format,
    pub v: Format,
}

impl Default for KvFormat {
    fn default() -> Self {
        Self::F16
    }
}

impl KvFormat {
    pub const F16: Self = Self::both(Format::F16);
    pub const BF16: Self = Self::both(Format::BF16);
    pub const Q8_0: Self = Self::both(Format::Q8_0);
    pub const Q4_0: Self = Self::both(Format::Q4_0);
    pub const Q8_0_Q4_0: Self = Self {
        k: Format::Q8_0,
        v: Format::Q4_0,
    };

    /// The pairs the flash-attention kernels are compiled for (CPU and CUDA, ADR 0050).
    pub const ALLOWED: [KvFormat; 5] = [
        Self::F16,
        Self::BF16,
        Self::Q8_0,
        Self::Q8_0_Q4_0,
        Self::Q4_0,
    ];

    pub const fn both(f: Format) -> Self {
        Self { k: f, v: f }
    }

    pub fn is_allowed(self) -> bool {
        Self::ALLOWED.contains(&self)
    }

    /// Bytes of K and V for one token of one layer, `n_embd_kv = kv_heads * head_dim`.
    pub fn bytes(self, n_embd_kv: u64) -> u64 {
        self.k.bytes(n_embd_kv) + self.v.bytes(n_embd_kv)
    }
}

fn lower(f: Format) -> String {
    format!("{f:?}").to_lowercase()
}

/// `q8_0/q4_0`: K then V, lower case; this is the `tetraeder.format` metric label.
impl fmt::Display for KvFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", lower(self.k), lower(self.v))
    }
}

/// `q8_0` means q8_0/q8_0; `q8_0/q4_0` names both.
impl FromStr for KvFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        let one = |n: &str| {
            Format::ALL
                .into_iter()
                .find(|f| lower(*f) == n.trim().to_lowercase())
                .ok_or_else(|| format!("unknown KV format {n:?}"))
        };
        let kv = match s.split_once('/') {
            Some((k, v)) => Self {
                k: one(k)?,
                v: one(v)?,
            },
            None => Self::both(one(s)?),
        };
        Ok(kv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_and_parse_roundtrip() {
        for kv in KvFormat::ALLOWED {
            assert_eq!(kv.to_string().parse::<KvFormat>().unwrap(), kv);
        }
        assert_eq!(KvFormat::Q8_0_Q4_0.to_string(), "q8_0/q4_0");
        assert_eq!("q8_0".parse::<KvFormat>().unwrap(), KvFormat::Q8_0);
        assert!("nope".parse::<KvFormat>().is_err());
    }
}
