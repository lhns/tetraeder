//! The dump of hidden rows the Case S exit passes between runs (ADR 0092): written by the exit
//! harness (`TETRAEDER_EXIT_HIDDEN_OUT`), read by its span runs and by `tetraeder cluster
//! judged-round`, which takes the frozen prompt's token ids from it.

use std::path::Path;

/// A dump of the hidden rows entering `layer` for a prompt and its greedy decode tokens: one row
/// per token, prompt first. The token ids travel with the rows, because a span without the head
/// cannot pick its decode tokens. Little-endian: magic, version, layer, n_embd, n_prompt,
/// n_tokens, the ids (i32), the rows (f32).
#[derive(Debug, Clone, PartialEq)]
pub struct HiddenFile {
    pub layer: u32,
    pub n_embd: usize,
    pub n_prompt: usize,
    /// The prompt and the decode tokens.
    pub tokens: Vec<i32>,
    /// `tokens.len() * n_embd` values.
    pub rows: Vec<f32>,
}

const MAGIC: &[u8; 8] = b"TTRHIDDN";
const HEADER: usize = 8 + 5 * 4;

impl HiddenFile {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(HEADER + 4 * (self.tokens.len() + self.rows.len()));
        b.extend_from_slice(MAGIC);
        for v in [
            1,
            self.layer,
            self.n_embd as u32,
            self.n_prompt as u32,
            self.tokens.len() as u32,
        ] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for t in &self.tokens {
            b.extend_from_slice(&t.to_le_bytes());
        }
        for r in &self.rows {
            b.extend_from_slice(&r.to_le_bytes());
        }
        b
    }

    pub fn from_bytes(b: &[u8]) -> Result<Self, String> {
        if b.len() < HEADER || &b[..8] != MAGIC {
            return Err("not a hidden-row file".into());
        }
        let u32_at = |i: usize| u32::from_le_bytes(b[8 + 4 * i..12 + 4 * i].try_into().unwrap());
        if u32_at(0) != 1 {
            return Err(format!("hidden-row file version {}", u32_at(0)));
        }
        let (layer, n_embd, n_prompt, n_tokens) = (
            u32_at(1),
            u32_at(2) as usize,
            u32_at(3) as usize,
            u32_at(4) as usize,
        );
        if n_prompt > n_tokens || b.len() != HEADER + 4 * (n_tokens + n_tokens * n_embd) {
            return Err("hidden-row file: sizes do not match its header".into());
        }
        let body = &b[HEADER..];
        let tokens = body[..4 * n_tokens]
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let rows = body[4 * n_tokens..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Ok(Self {
            layer,
            n_embd,
            n_prompt,
            tokens,
            rows,
        })
    }

    pub fn write(&self, path: &Path) -> std::io::Result<()> {
        std::fs::write(path, self.to_bytes())
    }

    pub fn read(path: &Path) -> Result<Self, String> {
        let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_bytes(&b).map_err(|e| format!("{}: {e}", path.display()))
    }
}
