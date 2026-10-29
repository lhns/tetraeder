//! Streaming helpers: UTF-8-safe piece assembly and top-k selection over logits.

use tetraeder_ggml_sys::llama::Token;

/// Buffers bytes of token pieces and releases only complete UTF-8. Invalid bytes become U+FFFD
/// as soon as they are known to be invalid; an incomplete tail waits for the next token.
#[derive(Debug, Default)]
pub(crate) struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    pub(crate) fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        let mut start = 0;
        loop {
            match std::str::from_utf8(&self.pending[start..]) {
                Ok(s) => {
                    out.push_str(s);
                    start = self.pending.len();
                    break;
                }
                Err(e) => {
                    let valid = start + e.valid_up_to();
                    // Valid by from_utf8's report.
                    out.push_str(&String::from_utf8_lossy(&self.pending[start..valid]));
                    match e.error_len() {
                        Some(n) => {
                            out.push('\u{FFFD}');
                            start = valid + n;
                        }
                        None => {
                            start = valid;
                            break;
                        }
                    }
                }
            }
        }
        self.pending.drain(..start);
        out
    }

    /// Releases the incomplete remainder, lossily.
    pub(crate) fn finish(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }
}

/// The `k` largest logits, best first; ties keep the lower token id first (greedy sampling's
/// choice). NaN logits are skipped.
pub(crate) fn top_k(logits: &[f32], k: usize) -> Vec<(Token, f32)> {
    let mut best: Vec<(Token, f32)> = Vec::with_capacity(k + 1);
    if k == 0 {
        return best;
    }
    for (i, &l) in logits.iter().enumerate() {
        if l.is_nan() || (best.len() == k && l <= best[k - 1].1) {
            continue;
        }
        let at = best.partition_point(|&(_, b)| b >= l);
        best.insert(at, (i as Token, l));
        best.truncate(k);
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_split_across_tokens_is_held_back() {
        let euro = "€".as_bytes(); // 3 bytes
        let mut s = Utf8Stream::default();
        assert_eq!(s.push(b"a"), "a");
        assert_eq!(s.push(&euro[..1]), "");
        assert_eq!(s.push(&euro[1..2]), "");
        assert_eq!(s.push(&euro[2..]), "€");
        assert_eq!(s.finish(), "");
    }

    #[test]
    fn invalid_bytes_become_replacement_and_tail_is_flushed_lossily() {
        let mut s = Utf8Stream::default();
        assert_eq!(s.push(&[b'x', 0xFF, b'y']), "x\u{FFFD}y");
        assert_eq!(s.push(&"€".as_bytes()[..2]), "");
        assert_eq!(s.finish(), "\u{FFFD}");
    }

    #[test]
    fn top_k_orders_and_breaks_ties_by_id() {
        let l = [1.0, 5.0, 3.0, 5.0, f32::NAN, 2.0];
        assert_eq!(top_k(&l, 3), vec![(1, 5.0), (3, 5.0), (2, 3.0)]);
        assert_eq!(top_k(&l, 0), vec![]);
        assert_eq!(top_k(&l, 10).len(), 5);
    }
}
