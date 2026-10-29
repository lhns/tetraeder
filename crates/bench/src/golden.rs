//! Golden data: greedy token sequences plus top-k logits per position, kept in git under `bench/golden/`.
//!
//! One file per (model, backend). Generation comes later; this module defines the format, the
//! loader and the comparison with a tolerance. Full logits (151k vocab) are a nightly artifact, not in git.

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const SCHEMA: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Golden {
    pub schema: u32,
    /// Model as `org/name:file.gguf`.
    pub model: String,
    /// Engine that produced the data, e.g. `llama.cpp@<commit>`.
    pub engine: String,
    /// `cpu` or `cuda`: logits differ slightly per backend.
    pub backend: String,
    pub top_k: usize,
    pub cases: Vec<Case>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Case {
    pub id: String,
    pub prompt_tokens: Vec<u32>,
    /// Greedily generated tokens.
    pub tokens: Vec<u32>,
    /// `top[i]` = the top-k (token, logit) pairs, best first, that produced `tokens[i]`.
    pub top: Vec<Vec<TopLogit>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TopLogit {
    pub id: u32,
    pub logit: f32,
}

/// Comparison tolerance. Defaults suit same-backend, same-build checks; loosen per backend and split point.
#[derive(Debug, Clone, Copy)]
pub struct Tolerance {
    /// Largest accepted absolute logit difference for a token present in both top-k lists.
    pub logit_abs: f32,
}

impl Default for Tolerance {
    fn default() -> Self {
        Tolerance { logit_abs: 1e-3 }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Mismatch {
    CaseMissing {
        case: String,
    },
    /// First position where the greedy tokens differ; later positions are not comparable.
    TokenDiverged {
        case: String,
        pos: usize,
        expected: u32,
        actual: u32,
    },
    LengthDiffers {
        case: String,
        expected: usize,
        actual: usize,
    },
    /// A token of the expected top-k is absent from the actual list and not a tie at the list boundary.
    TopMissing {
        case: String,
        pos: usize,
        token: u32,
    },
    LogitOff {
        case: String,
        pos: usize,
        token: u32,
        expected: f32,
        actual: f32,
    },
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Comparison {
    pub mismatches: Vec<Mismatch>,
    pub positions: usize,
    pub max_logit_diff: f32,
}

impl Comparison {
    pub fn ok(&self) -> bool {
        self.mismatches.is_empty()
    }
}

impl Golden {
    pub fn load(path: &Path) -> Result<Golden> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let g: Golden =
            serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
        g.validate().with_context(|| path.display().to_string())?;
        Ok(g)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        self.validate()?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(self)? + "\n")
            .with_context(|| format!("write {}", path.display()))
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == SCHEMA,
            "unsupported golden schema {}",
            self.schema
        );
        ensure!(self.top_k >= 1, "top_k must be at least 1");
        for c in &self.cases {
            ensure!(
                c.top.len() == c.tokens.len(),
                "case `{}`: {} tokens but {} top-k lists",
                c.id,
                c.tokens.len(),
                c.top.len()
            );
            for (i, t) in c.top.iter().enumerate() {
                ensure!(
                    t.len() <= self.top_k && !t.is_empty(),
                    "case `{}` pos {i}: need 1..={} entries",
                    c.id,
                    self.top_k
                );
                ensure!(
                    t[0].id == c.tokens[i],
                    "case `{}` pos {i}: top-1 is not the greedy token",
                    c.id
                );
            }
        }
        Ok(())
    }

    /// Compare `actual` against `self` (the expected data). Greedy tokens must match exactly; logits
    /// of tokens in both top-k lists must agree within `tol`. A token missing from the actual list is
    /// accepted only when its expected logit is within `tol` of the actual list's last entry (a tie at the boundary).
    pub fn compare(&self, actual: &Golden, tol: Tolerance) -> Comparison {
        let mut cmp = Comparison::default();
        for exp in &self.cases {
            let Some(act) = actual.cases.iter().find(|c| c.id == exp.id) else {
                cmp.mismatches.push(Mismatch::CaseMissing {
                    case: exp.id.clone(),
                });
                continue;
            };
            if exp.tokens.len() != act.tokens.len() {
                cmp.mismatches.push(Mismatch::LengthDiffers {
                    case: exp.id.clone(),
                    expected: exp.tokens.len(),
                    actual: act.tokens.len(),
                });
            }
            for pos in 0..exp.tokens.len().min(act.tokens.len()) {
                if exp.tokens[pos] != act.tokens[pos] {
                    cmp.mismatches.push(Mismatch::TokenDiverged {
                        case: exp.id.clone(),
                        pos,
                        expected: exp.tokens[pos],
                        actual: act.tokens[pos],
                    });
                    break;
                }
                cmp.positions += 1;
                let (e, a) = (&exp.top[pos], &act.top[pos]);
                let boundary = a.last().map(|x| x.logit);
                for want in e {
                    match a.iter().find(|x| x.id == want.id) {
                        Some(got) => {
                            let d = (got.logit - want.logit).abs();
                            cmp.max_logit_diff = cmp.max_logit_diff.max(d);
                            if d > tol.logit_abs {
                                cmp.mismatches.push(Mismatch::LogitOff {
                                    case: exp.id.clone(),
                                    pos,
                                    token: want.id,
                                    expected: want.logit,
                                    actual: got.logit,
                                });
                            }
                        }
                        None if boundary
                            .is_some_and(|b| (want.logit - b).abs() <= tol.logit_abs) => {}
                        None => cmp.mismatches.push(Mismatch::TopMissing {
                            case: exp.id.clone(),
                            pos,
                            token: want.id,
                        }),
                    }
                }
            }
        }
        cmp
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tl(id: u32, logit: f32) -> TopLogit {
        TopLogit { id, logit }
    }

    fn sample() -> Golden {
        Golden {
            schema: SCHEMA,
            model: "org/name:f.gguf".into(),
            engine: "llama.cpp@abc".into(),
            backend: "cpu".into(),
            top_k: 3,
            cases: vec![Case {
                id: "hello".into(),
                prompt_tokens: vec![1, 2, 3],
                tokens: vec![10, 20],
                top: vec![
                    vec![tl(10, 5.0), tl(11, 4.0), tl(12, 3.0)],
                    vec![tl(20, 6.0), tl(21, 2.5), tl(22, 2.0)],
                ],
            }],
        }
    }

    #[test]
    fn roundtrip_through_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("sub").join("g.json");
        sample().save(&p).unwrap();
        assert_eq!(Golden::load(&p).unwrap(), sample());
    }

    #[test]
    fn shipped_example_loads() {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/golden/example.json");
        let g = Golden::load(&p).unwrap();
        assert!(g.compare(&g, Tolerance::default()).ok());
    }

    #[test]
    fn identical_data_matches() {
        let c = sample().compare(&sample(), Tolerance::default());
        assert!(c.ok());
        assert_eq!(c.positions, 2);
        assert_eq!(c.max_logit_diff, 0.0);
    }

    #[test]
    fn small_logit_noise_within_tolerance() {
        let mut a = sample();
        a.cases[0].top[0][1].logit += 5e-4;
        assert!(sample().compare(&a, Tolerance::default()).ok());
        a.cases[0].top[0][1].logit += 1e-2;
        let c = sample().compare(&a, Tolerance::default());
        assert!(matches!(
            c.mismatches[0],
            Mismatch::LogitOff {
                pos: 0,
                token: 11,
                ..
            }
        ));
        assert!(sample().compare(&a, Tolerance { logit_abs: 0.1 }).ok());
    }

    #[test]
    fn token_divergence_stops_the_case() {
        let mut a = sample();
        a.cases[0].tokens[1] = 99;
        a.cases[0].top[1][0].id = 99;
        let c = sample().compare(&a, Tolerance::default());
        assert_eq!(
            c.mismatches,
            vec![Mismatch::TokenDiverged {
                case: "hello".into(),
                pos: 1,
                expected: 20,
                actual: 99
            }]
        );
        assert_eq!(c.positions, 1);
    }

    #[test]
    fn boundary_tie_is_accepted_but_other_missing_is_not() {
        let mut a = sample();
        a.cases[0].top[0][2] = tl(13, 3.0004); // expected token 12 (3.0) tied with the new boundary entry
        assert!(sample().compare(&a, Tolerance::default()).ok());
        a.cases[0].top[0][2] = tl(13, 1.0);
        let c = sample().compare(&a, Tolerance::default());
        assert!(c.mismatches.contains(&Mismatch::TopMissing {
            case: "hello".into(),
            pos: 0,
            token: 12
        }));
    }

    #[test]
    fn missing_case_and_bad_files_are_reported() {
        let mut a = sample();
        a.cases.clear();
        assert_eq!(
            sample().compare(&a, Tolerance::default()).mismatches,
            vec![Mismatch::CaseMissing {
                case: "hello".into()
            }]
        );
        let mut bad = sample();
        bad.cases[0].top[0][0].id = 7;
        assert!(bad.validate().is_err());
        bad = sample();
        bad.cases[0].top.pop();
        assert!(bad.validate().is_err());
    }
}
