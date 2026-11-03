use super::{SpecError, Target};
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
