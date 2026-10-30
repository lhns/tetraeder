//! Weight formats per unit (ADR 0024) with their ggml block layout.

use serde::{Deserialize, Serialize};

/// A ggml weight format. Names are ggml's, upper case, as in GGUF tooling.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Format {
    F32,
    F16,
    BF16,
    Q8_0,
    Q6_K,
    Q5_K,
    Q5_1,
    Q5_0,
    Q4_K,
    Q4_1,
    Q4_0,
    IQ4_XS,
    IQ4_NL,
    MXFP4,
    NVFP4,
    Q3_K,
    IQ3_S,
    IQ3_XXS,
    Q2_K,
    IQ2_S,
    IQ2_XS,
    IQ2_XXS,
    TQ2_0,
    Q2_0,
    TQ1_0,
    IQ1_M,
    IQ1_S,
    Q1_0,
}

impl Format {
    pub const ALL: [Format; 28] = [
        Self::F32,
        Self::F16,
        Self::BF16,
        Self::Q8_0,
        Self::Q6_K,
        Self::Q5_K,
        Self::Q5_1,
        Self::Q5_0,
        Self::Q4_K,
        Self::Q4_1,
        Self::Q4_0,
        Self::IQ4_XS,
        Self::IQ4_NL,
        Self::MXFP4,
        Self::NVFP4,
        Self::Q3_K,
        Self::IQ3_S,
        Self::IQ3_XXS,
        Self::Q2_K,
        Self::IQ2_S,
        Self::IQ2_XS,
        Self::IQ2_XXS,
        Self::TQ2_0,
        Self::Q2_0,
        Self::TQ1_0,
        Self::IQ1_M,
        Self::IQ1_S,
        Self::Q1_0,
    ];

    /// `(elements per block, bytes per block)`, from the static asserts in ggml-common.h at the
    /// pin (b11976).
    pub const fn block(self) -> (u64, u64) {
        match self {
            Self::F32 => (1, 4),
            Self::F16 | Self::BF16 => (1, 2),
            Self::Q8_0 => (32, 34),
            Self::Q6_K => (256, 210),
            Self::Q5_K => (256, 176),
            Self::Q5_1 => (32, 24),
            Self::Q5_0 => (32, 22),
            Self::Q4_K => (256, 144),
            Self::Q4_1 => (32, 20),
            Self::Q4_0 | Self::IQ4_NL => (32, 18),
            Self::IQ4_XS => (256, 136),
            Self::MXFP4 => (32, 17),
            Self::NVFP4 => (64, 36),
            Self::Q3_K | Self::IQ3_S => (256, 110),
            Self::IQ3_XXS => (256, 98),
            Self::Q2_K => (256, 84),
            Self::IQ2_S => (256, 82),
            Self::IQ2_XS => (256, 74),
            Self::IQ2_XXS | Self::TQ2_0 => (256, 66),
            Self::Q2_0 => (64, 18),
            Self::TQ1_0 => (256, 54),
            Self::IQ1_M => (256, 56),
            Self::IQ1_S => (256, 50),
            Self::Q1_0 => (128, 18),
        }
    }

    pub fn bits_per_weight(self) -> f64 {
        let (n, b) = self.block();
        (b * 8) as f64 / n as f64
    }

    /// Bytes of `params` weights in this format (whole blocks).
    pub fn bytes(self, params: u64) -> u64 {
        let (n, b) = self.block();
        params.div_ceil(n) * b
    }

    /// At least as many bits per weight as `other` (ADR 0027: the precision order).
    pub fn at_least(self, other: Format) -> bool {
        let (n, b) = self.block();
        let (m, c) = other.block();
        b * m >= c * n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_per_weight_match_ggml() {
        assert_eq!(Format::Q4_K.bits_per_weight(), 4.5);
        assert_eq!(Format::Q8_0.bits_per_weight(), 8.5);
        assert_eq!(Format::Q6_K.bits_per_weight(), 6.5625);
        assert_eq!(Format::IQ1_S.bits_per_weight(), 1.5625);
        assert_eq!(Format::IQ1_M.bits_per_weight(), 1.75);
        assert_eq!(Format::F16.bits_per_weight(), 16.0);
    }

    #[test]
    fn bytes_round_up_to_blocks() {
        assert_eq!(Format::Q4_K.bytes(256), 144);
        assert_eq!(Format::Q4_K.bytes(257), 288);
        assert_eq!(Format::F32.bytes(3), 12);
        assert_eq!(Format::Q8_0.bytes(0), 0);
    }

    #[test]
    fn precision_order() {
        assert!(Format::Q8_0.at_least(Format::Q4_K));
        assert!(!Format::Q4_K.at_least(Format::Q8_0));
        assert!(Format::Q4_K.at_least(Format::Q4_0));
        assert!(Format::Q4_0.at_least(Format::Q4_K));
        assert!(Format::F16.at_least(Format::BF16));
    }

    #[test]
    fn names_roundtrip_through_serde() {
        for f in Format::ALL {
            let s = serde_json::to_string(&f).unwrap();
            assert_eq!(serde_json::from_str::<Format>(&s).unwrap(), f);
        }
        assert_eq!(
            serde_json::to_string(&Format::IQ4_XS).unwrap(),
            "\"IQ4_XS\""
        );
    }
}
