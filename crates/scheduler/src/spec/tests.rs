//! Loop tests on a toy target: speculative decoding must equal plain decoding token for token.

use super::policy::Adaptive;
use super::*;

const V: usize = 12;

fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// A deterministic pseudo-LM: the logits of a position depend on the two tokens before it
/// (inclusive) only, so the output falls into loops an n-gram drafter can find. It keeps its
/// own KV token list, like a real target, and checks the calls it gets.
struct Toy {
    kv: Vec<i32>,
    rewinds: Vec<usize>,
    evals: Vec<usize>,
}

impl Toy {
    fn new() -> Self {
        Self {
            kv: Vec::new(),
            rewinds: Vec::new(),
            evals: Vec::new(),
        }
    }

    fn row(&self, p: usize) -> Vec<f32> {
        let a = if p > 0 { self.kv[p - 1] } else { -1 };
        let b = self.kv[p];
        (0..V as u64)
            .map(|v| {
                let h = mix(((a as u64) << 32) ^ (b as u64) << 8 ^ v ^ 0x1234);
                (h % 1000) as f32 / 150.0
            })
            .collect()
    }
}

impl Target for Toy {
    fn n_vocab(&self) -> usize {
        V
    }
    fn n_past(&self) -> usize {
        self.kv.len()
    }
    fn eval(&mut self, tokens: &[i32]) -> Result<Vec<f32>, SpecError> {
        assert!(!tokens.is_empty() && tokens.len() <= MAX_VERIFY);
        self.evals.push(tokens.len());
        let mut out = Vec::new();
        for &t in tokens {
            self.kv.push(t);
            out.extend(self.row(self.kv.len() - 1));
        }
        Ok(out)
    }
    fn rewind(&mut self, n_past: usize) -> Result<(), SpecError> {
        assert!(n_past < self.kv.len(), "rewind must shorten the cache");
        self.rewinds.push(n_past);
        self.kv.truncate(n_past);
        Ok(())
    }
}

/// Proposes the plain continuation; with `every`, each `every`-th drafted token is wrong.
struct Oracle {
    plain: Vec<i32>,
    every: Option<usize>,
}

impl Drafter for Oracle {
    fn draft(&mut self, history: &[i32], max: usize) -> Result<Vec<i32>, SpecError> {
        assert_eq!(
            history,
            &self.plain[..history.len()],
            "history is plain's prefix"
        );
        let end = (history.len() + max).min(self.plain.len());
        let mut d = self.plain[history.len()..end].to_vec();
        if let Some(e) = self.every {
            for (i, t) in d.iter_mut().enumerate() {
                if i % e == e - 1 {
                    *t = (*t + 1) % V as i32;
                }
            }
        }
        Ok(d)
    }
    fn method(&self) -> &'static str {
        "oracle"
    }
}

struct Random(u64);

impl Drafter for Random {
    fn draft(&mut self, _: &[i32], max: usize) -> Result<Vec<i32>, SpecError> {
        Ok((0..max)
            .map(|_| {
                self.0 = mix(self.0.wrapping_add(1));
                (self.0 % V as u64) as i32
            })
            .collect())
    }
    fn method(&self) -> &'static str {
        "random"
    }
}

const PROMPT: [i32; 5] = [3, 1, 4, 1, 5];
const N: usize = 60;

/// Prefills the toy with all but the last prompt token, like the caller of `step` does.
fn start() -> (Toy, Vec<i32>) {
    let mut t = Toy::new();
    t.eval(&PROMPT[..PROMPT.len() - 1]).unwrap();
    (t, PROMPT.to_vec())
}

fn plain(s: Sampling, n: usize) -> Vec<i32> {
    let (mut t, mut h) = start();
    let mut sp = Speculator::new(DraftLen::Fixed(0), s, None);
    sp.generate(&mut t, None, &mut h, n).unwrap();
    assert_eq!(t.n_past(), h.len() - 1);
    h
}

fn spec(len: DraftLen, s: Sampling, d: &mut dyn Drafter) -> (Vec<i32>, SpecStats, Toy) {
    let (mut t, mut h) = start();
    let mut sp = Speculator::new(len, s, None);
    let mut total = 0;
    while total < N {
        let before = h.len();
        let e = sp.step(&mut t, Some(&mut *d), &mut h).unwrap();
        assert_eq!(h.len() - before, e);
        assert_eq!(t.n_past(), h.len() - 1, "target is one behind the history");
        assert_eq!(&t.kv[..], &h[..h.len() - 1], "KV holds exactly the history");
        total += e;
    }
    (h, sp.stats(), t)
}

fn check(len: DraftLen, s: Sampling, d: &mut dyn Drafter, want: &[i32]) -> SpecStats {
    let (h, st, t) = spec(len, s, d);
    // `want` runs past N so the oracle covers the overshoot; compare the common length.
    let n = h.len().min(want.len());
    assert!(n >= PROMPT.len() + N);
    assert_eq!(&h[..n], &want[..n], "{len:?} {s:?}");
    assert_eq!(st.emitted, st.accepted + st.steps);
    assert_eq!(st.emitted as usize, h.len() - PROMPT.len());
    assert_eq!(st.by_batch.iter().map(|b| b.steps).sum::<u64>(), st.steps);
    assert_eq!(
        st.drafted,
        t.evals.iter().skip(1).map(|n| (n - 1) as u64).sum::<u64>()
    );
    st
}

fn samplings() -> [Sampling; 2] {
    [
        Sampling::GREEDY,
        Sampling {
            temperature: 0.8,
            seed: 7,
        },
    ]
}

fn lens() -> Vec<DraftLen> {
    let mut v: Vec<_> = (1..=6).map(DraftLen::Fixed).collect();
    v.push(DraftLen::Adaptive { max: 6 });
    v
}

#[test]
fn every_drafter_equals_plain_decoding() {
    for s in samplings() {
        let want = plain(s, N + 8);
        for len in lens() {
            check(
                len,
                s,
                &mut Oracle {
                    plain: want.clone(),
                    every: None,
                },
                &want,
            );
            check(
                len,
                s,
                &mut Oracle {
                    plain: want.clone(),
                    every: Some(3),
                },
                &want,
            );
            check(len, s, &mut Random(5), &want);
            check(len, s, &mut NgramDrafter::default(), &want);
        }
    }
}

#[test]
fn oracle_accepts_everything_and_batches() {
    let want = plain(Sampling::GREEDY, N + 8);
    let st = check(
        DraftLen::Fixed(4),
        Sampling::GREEDY,
        &mut Oracle {
            plain: want.clone(),
            every: None,
        },
        &want,
    );
    assert_eq!(st.accepted, st.drafted);
    assert_eq!(st.emitted, 5 * st.steps);
    assert_eq!(st.by_batch[4].steps, st.steps);
}

#[test]
fn rejections_rewind_the_target() {
    let want = plain(Sampling::GREEDY, N + 8);
    let (_, st, t) = spec(
        DraftLen::Fixed(4),
        Sampling::GREEDY,
        &mut Oracle {
            plain: want,
            every: Some(3),
        },
    );
    // Every step drafts [ok, ok, wrong, ok]: two accepted, the rest rolled back.
    assert!(st.accepted < st.drafted);
    assert!(!t.rewinds.is_empty());
}

#[test]
fn no_drafter_is_a_one_row_step() {
    let (mut t, mut h) = start();
    let mut sp = Speculator::new(DraftLen::Fixed(6), Sampling::GREEDY, None);
    assert_eq!(sp.generate(&mut t, None, &mut h, 10).unwrap(), 10);
    assert!(t.evals.iter().skip(1).all(|&n| n == 1));
    assert!(t.rewinds.is_empty());
    assert_eq!(sp.stats().drafted, 0);
}

#[test]
fn step_checks_its_precondition() {
    let mut t = Toy::new();
    let mut h = vec![1, 2, 3];
    let mut sp = Speculator::new(DraftLen::Fixed(1), Sampling::GREEDY, None);
    assert!(sp.step(&mut t, None, &mut h).is_err());
}

#[test]
fn argmax_first_maximum_wins() {
    assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
    assert_eq!(argmax(&[5.0, 5.0]), 0);
    assert_eq!(argmax(&[f32::NAN, 1.0, 1.0]), 1);
    assert_eq!(argmax(&[-2.0]), 0);
}

#[test]
fn choose_is_deterministic_per_position_and_follows_the_distribution() {
    let l = [0.0, 2.0, 0.0, 1.0];
    let s = Sampling {
        temperature: 1.0,
        seed: 3,
    };
    assert_eq!(choose(&l, 9, &s), choose(&l, 9, &s));
    assert_eq!(choose(&l, 9, &Sampling::GREEDY), 1);
    let mut counts = [0; 4];
    for pos in 0..4000 {
        counts[choose(&l, pos, &s) as usize] += 1;
    }
    // softmax = [0.083, 0.610, 0.083, 0.224]
    assert!(
        (counts[1] as f64 / 4000.0 - 0.610).abs() < 0.03,
        "{counts:?}"
    );
    assert!(counts[3] > counts[0]);
}

#[test]
fn ngram_finds_the_most_recent_longest_match() {
    let mut d = NgramDrafter { min: 2, max: 4 };
    // Suffix [1,2,3] occurred before, followed by 7, 8.
    assert_eq!(
        d.draft(&[9, 1, 2, 3, 7, 8, 5, 1, 2, 3], 5).unwrap(),
        vec![7, 8, 5, 1, 2]
    );
    assert_eq!(
        d.draft(&[9, 1, 2, 3, 7, 8, 5, 1, 2, 3], 2).unwrap(),
        vec![7, 8]
    );
    // The most recent occurrence wins.
    assert_eq!(d.draft(&[1, 2, 4, 1, 2, 6, 0, 1, 2], 1).unwrap(), vec![6]);
    // A longer suffix beats a shorter, more recent one.
    assert_eq!(
        d.draft(&[5, 1, 2, 7, 0, 1, 2, 8, 5, 1, 2], 1).unwrap(),
        vec![7]
    );
    // Proposals stop at the end of the history.
    assert_eq!(d.draft(&[1, 2, 1, 2], 4).unwrap(), vec![1, 2]);
    // No earlier occurrence, or too short.
    assert!(d.draft(&[1, 2, 3, 4], 3).unwrap().is_empty());
    assert!(d.draft(&[1], 3).unwrap().is_empty());
    assert!(d.draft(&[], 3).unwrap().is_empty());
}

#[test]
fn chooser_goes_long_when_time_is_flat_and_short_when_steep_or_unlikely() {
    assert_eq!(choose_k(0.8, 6, |_| 1.0), 6);
    assert_eq!(choose_k(0.5, 6, |_| 1.0), 6);
    // Time proportional to rows: speculation never pays.
    assert_eq!(choose_k(0.8, 6, |n| n as f64), 0);
    // Low acceptance with a moderate cost per row.
    assert_eq!(choose_k(0.1, 6, |n| 1.0 + 0.5 * (n - 1) as f64), 0);
    // High acceptance with a mild cost per row.
    assert!(choose_k(0.9, 6, |n| 1.0 + 0.05 * (n - 1) as f64) >= 4);
    // Out-of-range estimates are clamped, max is capped to the verify batch.
    assert_eq!(choose_k(1.0, 20, |_| 1.0), MAX_VERIFY - 1);
}

#[test]
fn adaptive_estimates_acceptance_and_interpolates_times() {
    let mut a = Adaptive::new(6);
    assert!((a.acceptance() - 0.5).abs() < 1e-12);
    assert_eq!(a.time_at(3), 1.0);
    // Optimistic before a multi-row sample: flat at T(1).
    a.observe_time(1, 0.4);
    assert_eq!(a.time_at(5), 0.4);
    assert_eq!(a.next_k(), 6);
    a.observe_time(5, 0.8);
    assert!((a.time_at(3) - 0.6).abs() < 1e-12);
    assert!((a.time_at(7) - 1.0).abs() < 1e-12);
    for _ in 0..50 {
        a.observe(4, 4);
    }
    assert!((a.acceptance() - 0.95).abs() < 1e-12);
    for _ in 0..200 {
        a.observe(4, 0);
    }
    assert!((a.acceptance() - 0.05).abs() < 1e-12);
    assert_eq!(a.next_k(), 0);
}

#[test]
fn adaptive_with_rejected_drafts_still_equals_plain() {
    let want = plain(Sampling::GREEDY, N + 8);
    let (mut t, mut h) = start();
    let mut sp = Speculator::new(DraftLen::Adaptive { max: 6 }, Sampling::GREEDY, None);
    let mut d = Random(1);
    sp.generate(&mut t, Some(&mut d), &mut h, N).unwrap();
    assert_eq!(&h[..N], &want[..N]);
}
