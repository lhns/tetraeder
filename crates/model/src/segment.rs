//! A segment of the layers: the share of a model one node runs (ADR 0091, step 4).
//!
//! A segment that does not hold the embedding takes the hidden rows of every input token as its
//! input (F32, `n_embd` per token); one that does not hold the head returns the hidden rows of
//! every input token instead of logits. Nothing else in a builder knows about nodes.

use crate::qwen3moe::{ModelError, Result};
use std::ops::Range;

/// Which part of a model a builder loads and runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentSpec {
    /// Layers computed, `[start, end)`; empty for an embedding-only or head-only segment.
    pub layers: Range<u32>,
    /// Holds the token embedding: takes tokens, else hidden rows.
    pub embed: bool,
    /// Holds the output norm and projection: returns logits, else hidden rows.
    pub head: bool,
}

impl SegmentSpec {
    /// The whole model: the same as no segment.
    pub fn whole(n_layer: u32) -> Self {
        Self {
            layers: 0..n_layer,
            embed: true,
            head: true,
        }
    }

    /// Whether this is the whole model of `n_layer` layers.
    pub fn is_whole(&self, n_layer: u32) -> bool {
        *self == Self::whole(n_layer)
    }

    /// Layers the segment holds.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Checks the segment against a model of `n_layer` layers.
    pub fn validate(&self, n_layer: u32) -> Result<()> {
        let bad = |why: &str| Err(ModelError::Invalid(format!("segment {self:?}: {why}")));
        if self.layers.start > self.layers.end || self.layers.end > n_layer {
            return bad(&format!("layers outside 0..{n_layer}"));
        }
        if self.layers.is_empty() && !self.embed && !self.head {
            return bad("holds nothing");
        }
        if self.embed && self.layers.start != 0 && !self.layers.is_empty() {
            return bad("the embedding goes with layer 0");
        }
        if self.head && self.layers.end != n_layer && !self.layers.is_empty() {
            return bad("the head goes with the last layer");
        }
        Ok(())
    }

    /// Floats per row the segment's reply holds: `n_embd` without the head, else `n_vocab`.
    pub fn out_width(&self, n_embd: usize, n_vocab: usize) -> usize {
        if self.head { n_vocab } else { n_embd }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // hardware: none
    #[test]
    fn validation_follows_the_cut_rules() {
        let ok = |s: SegmentSpec| s.validate(4).is_ok();
        assert!(ok(SegmentSpec::whole(4)));
        assert!(ok(SegmentSpec {
            layers: 0..2,
            embed: true,
            head: false
        }));
        assert!(ok(SegmentSpec {
            layers: 2..4,
            embed: false,
            head: true
        }));
        // Embedding only, head only.
        assert!(ok(SegmentSpec {
            layers: 0..0,
            embed: true,
            head: false
        }));
        assert!(ok(SegmentSpec {
            layers: 4..4,
            embed: false,
            head: true
        }));
        assert!(!ok(SegmentSpec {
            layers: 1..3,
            embed: true,
            head: false
        }));
        assert!(!ok(SegmentSpec {
            layers: 1..3,
            embed: false,
            head: true
        }));
        assert!(!ok(SegmentSpec {
            layers: 1..5,
            embed: false,
            head: false
        }));
        assert!(!ok(SegmentSpec {
            layers: 2..2,
            embed: false,
            head: false
        }));
    }
}
