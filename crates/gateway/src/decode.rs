//! Incremental UTF-8 decoding and stop-sequence matching on streamed output.

/// Assembles UTF-8 text from token bytes that may split a character.
#[derive(Debug, Default)]
pub struct Utf8Decoder {
    pending: Vec<u8>,
}

impl Utf8Decoder {
    /// Appends bytes and returns the text that is complete so far.
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    return out;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    // Safe split: the prefix up to `valid` is UTF-8.
                    out.push_str(&String::from_utf8_lossy(&self.pending[..valid]));
                    match e.error_len() {
                        // Incomplete character at the end: wait for more bytes.
                        None => {
                            self.pending.drain(..valid);
                            return out;
                        }
                        // Invalid bytes: replace and continue.
                        Some(n) => {
                            out.push(char::REPLACEMENT_CHARACTER);
                            self.pending.drain(..valid + n);
                        }
                    }
                }
            }
        }
    }

    /// Flushes an incomplete trailing character as a replacement character.
    pub fn finish(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }
}

/// Finds caller stop sequences in streamed text, holding back a suffix that may start one.
#[derive(Debug, Default)]
pub struct StopMatcher {
    stops: Vec<String>,
    held: String,
}

impl StopMatcher {
    pub fn new(stops: Vec<String>) -> Self {
        Self {
            stops: stops.into_iter().filter(|s| !s.is_empty()).collect(),
            held: String::new(),
        }
    }

    /// Returns text safe to emit and, on a match, the matched stop sequence. After a match
    /// the text after the stop is discarded and the caller should end generation.
    pub fn push(&mut self, text: &str) -> (String, Option<String>) {
        if self.stops.is_empty() {
            return (text.to_owned(), None);
        }
        self.held.push_str(text);
        // Earliest match wins; on ties the longest stop.
        let hit = self
            .stops
            .iter()
            .filter_map(|s| self.held.find(s.as_str()).map(|i| (i, s)))
            .min_by(|a, b| a.0.cmp(&b.0).then(b.1.len().cmp(&a.1.len())));
        if let Some((i, s)) = hit {
            let out = self.held[..i].to_owned();
            let s = s.clone();
            self.held.clear();
            return (out, Some(s));
        }
        let keep = self.partial_suffix_len();
        let cut = self.held.len() - keep;
        let out = self.held[..cut].to_owned();
        self.held.drain(..cut);
        (out, None)
    }

    /// Releases held-back text at the end of generation.
    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.held)
    }

    /// Length of the longest suffix of `held` that is a proper prefix of some stop.
    fn partial_suffix_len(&self) -> usize {
        let h = &self.held;
        let mut best = 0;
        for s in &self.stops {
            let max = s.len().saturating_sub(1).min(h.len());
            for n in (1..=max).rev() {
                if n <= best {
                    break;
                }
                let start = h.len() - n;
                if h.is_char_boundary(start) && s.starts_with(&h[start..]) {
                    best = n;
                    break;
                }
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_split_across_tokens() {
        let mut d = Utf8Decoder::default();
        let e = "é".as_bytes();
        assert_eq!(d.push(&e[..1]), "");
        assert_eq!(d.push(&e[1..]), "é");
        assert_eq!(d.push(b"a\xff"), "a\u{fffd}");
        assert_eq!(d.push(&"€".as_bytes()[..2]), "");
        assert_eq!(d.finish(), "\u{fffd}");
    }

    #[test]
    fn stop_split_across_pushes() {
        let mut m = StopMatcher::new(vec!["END".into()]);
        assert_eq!(m.push("abc E"), ("abc ".into(), None));
        assert_eq!(m.push("N"), (String::new(), None));
        assert_eq!(m.push("Dxyz"), (String::new(), Some("END".into())));
    }

    #[test]
    fn false_start_is_released() {
        let mut m = StopMatcher::new(vec!["END".into()]);
        assert_eq!(m.push("EN"), (String::new(), None));
        assert_eq!(m.push("x"), ("ENx".into(), None));
        assert_eq!(m.push("E"), (String::new(), None));
        assert_eq!(m.finish(), "E");
    }

    #[test]
    fn earliest_stop_wins() {
        let mut m = StopMatcher::new(vec!["bc".into(), "a".into()]);
        assert_eq!(m.push("xabc"), ("x".into(), Some("a".into())));
    }
}
