use super::{Drafter, SpecError};

/// Prompt lookup: proposes what followed the most recent earlier occurrence of the history's
/// last m tokens, trying m from `max` down to `min`.
#[derive(Debug, Clone, Copy)]
pub struct NgramDrafter {
    pub min: usize,
    pub max: usize,
}

impl Default for NgramDrafter {
    fn default() -> Self {
        Self { min: 2, max: 4 }
    }
}

impl Drafter for NgramDrafter {
    fn draft(&mut self, history: &[i32], max_tokens: usize) -> Result<Vec<i32>, SpecError> {
        let n = history.len();
        for m in (self.min.max(1)..=self.max).rev() {
            if m >= n {
                continue;
            }
            let suffix = &history[n - m..];
            // The suffix itself starts at n - m; only earlier starts count.
            if let Some(s) = (0..n - m).rev().find(|&s| &history[s..s + m] == suffix) {
                let from = s + m;
                return Ok(history[from..(from + max_tokens).min(n)].to_vec());
            }
        }
        Ok(Vec::new())
    }

    fn method(&self) -> &'static str {
        "ngram"
    }
}
