//! A model whose memory cannot drop a suffix (a recurrent or hybrid one such as Qwen3.6, whose
//! libllama `seq_rm` returns false): prefix reuse must not fail the request (bench-strata-q36).
//! `FakeKv` stands in; the tiny fixtures hold no recurrent model, so libllama itself is not
//! exercised here.
//!
//! Hardware label: none (the tokenizer comes from the tiny fixture).

mod fake;

use fake::{FakeKv, Probe};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use tetraeder_gateway::engine::{
    Engine, FinishReason, GenEvent, GenerateRequest, SamplingParams, TokenConstraint, TokenMask,
};
use tetraeder_ggml_sys::llama::{Model, ModelParams};
use tetraeder_scheduler::{Loaded, Scheduler, SchedulerConfig};
use tokio_util::sync::CancellationToken;

const N_CTX_SEQ: usize = 3200;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3moe-f16.gguf")
}

/// One sequence, reuse on, over a `FakeKv` shaped by `shape`.
fn start(shape: impl FnOnce(FakeKv) -> FakeKv + Send + 'static) -> (Scheduler, Arc<Probe>) {
    let path = fixture();
    let mut sc = SchedulerConfig::new(&path);
    sc.parallel = 1;
    sc.step_tokens = 512;
    sc.cache_save_min = 32;
    sc.cache_restore_gain = 16;
    sc.prefix_reuse = true;
    let cell: Arc<std::sync::OnceLock<Arc<Probe>>> = Arc::default();
    let c = Arc::clone(&cell);
    let s = Scheduler::start(sc, move || {
        let tokenizer = Arc::new(
            Model::load(
                &path,
                &ModelParams {
                    vocab_only: true,
                    ..ModelParams::default()
                },
            )
            .map_err(|e| e.to_string())?,
        );
        let n_vocab = tokenizer.n_vocab() as usize;
        let kv = shape(FakeKv::new(n_vocab, 1, N_CTX_SEQ, N_CTX_SEQ));
        let _ = c.set(kv.probe());
        Ok(Loaded {
            backend: Box::new(kv),
            tokenizer,
        })
    })
    .expect("scheduler over FakeKv");
    (s, cell.get().cloned().expect("probe"))
}

/// Allows a few tokens, so the tiny vocabulary's end-of-generation tokens never end a run.
struct Only(Vec<u32>);

impl TokenConstraint for Only {
    fn mask(&mut self) -> Result<Option<TokenMask>, String> {
        let mut words = vec![0u32; 16];
        for &t in &self.0 {
            words[t as usize / 32] |= 1 << (t % 32);
        }
        Ok(Some(TokenMask { words }))
    }

    fn commit(&mut self, _: u32) -> Result<(), String> {
        Ok(())
    }
}

fn generate(s: &Scheduler, prompt: Vec<u32>, max_tokens: u32) -> Vec<GenEvent> {
    let mut rx = s
        .generate(GenerateRequest {
            model: s.models()[0].id.clone(),
            prompt,
            params: SamplingParams {
                max_tokens,
                temperature: Some(0.0),
                ..SamplingParams::default()
            },
            constraint: Some(Box::new(Only(vec![65, 66, 67]))),
            cancel: CancellationToken::new(),
        })
        .expect("generate");
    let mut out = Vec::new();
    while let Some(ev) = rx.blocking_recv() {
        out.push(ev);
    }
    out
}

/// The generated tokens and `cached_tokens` of a request that finished; panics on an error.
fn finished(events: &[GenEvent]) -> (Vec<u32>, u32) {
    let toks = events
        .iter()
        .filter_map(|e| match e {
            GenEvent::Token(t) => Some(*t),
            _ => None,
        })
        .collect();
    match events.last() {
        Some(GenEvent::Finished { reason, stats }) => {
            assert_eq!(*reason, FinishReason::Length);
            (toks, stats.cached_tokens)
        }
        other => panic!("not finished: {other:?}"),
    }
}

/// The first `share` tokens are common to all seeds.
fn prompt(n: usize, seed: u32, share: usize) -> Vec<u32> {
    (0..n)
        .map(|i| {
            if i < share {
                10 + (i as u32 * 7) % 150
            } else {
                10 + (seed * 31 + i as u32 * 13) % 150
            }
        })
        .collect()
}

/// The backend says it cannot cut: a diverging prompt clears the sequence and prefills in full,
/// and no cut is attempted. A prompt that continues the held sequence reuses it whole.
#[test]
fn a_backend_that_cannot_cut_reuses_only_what_it_keeps_whole() {
    let (s, probe) = start(|k| k.with_no_suffix_cut(true));
    let a = prompt(200, 1, 200);
    let (gen_a, _) = finished(&generate(&s, a.clone(), 8));

    let (_, cached) = finished(&generate(&s, prompt(200, 2, 100), 4));
    assert_eq!(cached, 0, "the shared 100 tokens need a cut");
    assert_eq!(probe.suffix_cuts.load(Ordering::SeqCst), 0);

    // The sequence now holds the second prompt and what it generated; continue it exactly.
    let b = prompt(200, 2, 100);
    let (gen_b, _) = finished(&generate(&s, b.clone(), 8));
    let mut c = b.clone();
    c.extend(&gen_b);
    c.extend([20, 21, 22]);
    let (_, cached) = finished(&generate(&s, c, 4));
    assert_eq!(cached as usize, b.len() + gen_b.len() - 1);
    assert_eq!(probe.suffix_cuts.load(Ordering::SeqCst), 0);
    drop(gen_a);
    s.shutdown().expect("shutdown");
}

/// The backend does not say so, but its cut fails at runtime: the request falls back to a full
/// prefill instead of failing.
#[test]
fn a_refused_cut_falls_back_to_a_full_prefill() {
    let (s, probe) = start(|k| k.with_no_suffix_cut(false));
    finished(&generate(&s, prompt(200, 1, 200), 8));
    let (toks, cached) = finished(&generate(&s, prompt(200, 2, 100), 4));
    assert_eq!(cached, 0);
    assert_eq!(toks.len(), 4);
    assert_eq!(probe.suffix_cuts.load(Ordering::SeqCst), 1);
    // And the slot keeps working afterwards.
    finished(&generate(&s, prompt(200, 3, 50), 4));
    s.shutdown().expect("shutdown");
}

/// A backend error ends the request with `GenEvent::Error`, never a bare finish.
#[test]
fn a_backend_error_reaches_the_client() {
    let (s, _) = start(|k| k.with_poison(Some(66)));
    let events = generate(&s, prompt(40, 1, 40), 8);
    assert!(
        matches!(events.last(), Some(GenEvent::Error(_))),
        "{events:?}"
    );
    s.shutdown().expect("shutdown");
}
