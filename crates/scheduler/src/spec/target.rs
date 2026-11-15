use super::{SpecError, Target};
use crate::backend::{BatchBackend, SeqId, SeqInput};
use tetraeder_model::Qwen3Moe;

impl Target for Qwen3Moe {
    fn n_vocab(&self) -> usize {
        Qwen3Moe::n_vocab(self)
    }

    fn n_past(&self) -> usize {
        Qwen3Moe::n_past(self)
    }

    fn eval(&mut self, tokens: &[i32]) -> Result<Vec<f32>, SpecError> {
        Ok(Qwen3Moe::eval(self, tokens)?)
    }

    fn rewind(&mut self, n_past: usize) -> Result<(), SpecError> {
        Ok(Qwen3Moe::rewind(self, n_past)?)
    }
}

/// One sequence of a [`BatchBackend`] as a [`Target`] (ADR 0065): `eval` is a step of that
/// sequence alone with logits of every row, `rewind` is `seq_rm`. The serving loop does not go
/// through it (it puts the verify rows into the shared step); it lets `Speculator` and its tests
/// run on any batch backend.
pub struct SeqTarget<'a> {
    pub backend: &'a mut dyn BatchBackend,
    pub seq: SeqId,
}

impl Target for SeqTarget<'_> {
    fn n_vocab(&self) -> usize {
        self.backend.n_vocab()
    }

    fn n_past(&self) -> usize {
        self.backend.seq_len(self.seq)
    }

    fn eval(&mut self, tokens: &[i32]) -> Result<Vec<f32>, SpecError> {
        let out = self.backend.decode(&[SeqInput {
            seq: self.seq,
            tokens,
            all_logits: true,
        }])?;
        Ok(out.rows(0).to_vec())
    }

    fn rewind(&mut self, n_past: usize) -> Result<(), SpecError> {
        Ok(self.backend.seq_rm(self.seq, n_past)?)
    }
}
