//! Speculation in the batching loop (ADR 0043, ADR 0065 "Speculative decoding"): greedy output
//! with `SchedulerConfig::spec` on equals output with it off, for the libllama and the dense
//! builder engines on the tiny dense fixture, with the n-gram drafter and a draft model (the same
//! tiny file: a draft that agrees with the target, so most drafts are accepted). With two
//! concurrent requests nothing speculates and the outputs equal the plain ones.
//!
//! The equality is bitwise where ggml's kernels give a verify step the logits of one-row steps;
//! on AVX2 and AVX-512 builds they differ in the last bits (ADR 0043), which flips a token only
//! at a numeric tie. The tiny model's top-2 margins are far above that.
//!
//! Hardware label (AGENTS rule 9): `cpu_*` tests need no GPU.

mod fake;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tetraeder_engine::{ContextConfig, Placement};
use tetraeder_gateway::engine::{
    Engine, FinishReason, GenEvent, GenerateRequest, SamplingParams, TokenId,
};
use tetraeder_model::{BackendChoice, DenseOptions};
use tetraeder_scheduler::spec::{
    DraftLen, MAX_VERIFY, NgramDrafter, Sampling, SeqTarget, SpecConfig, SpecMethod, Speculator,
    argmax,
};
use tetraeder_scheduler::{
    BackendError, BatchBackend, Loaded, Outputs, Scheduler, SchedulerConfig, SeqId, SeqInput,
    load_dense, load_llama,
};
use tokio_util::sync::CancellationToken;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3-f16.gguf")
}

#[derive(Default)]
struct Seen {
    /// Decodes with an input of more than one token that asked for every row's logits.
    verify_decodes: AtomicUsize,
    decodes: AtomicUsize,
    /// Decodes that held a verify input next to a one-token input of another sequence.
    verify_beside_decoder: AtomicUsize,
    widest: AtomicUsize,
    /// While set, the first decode waits (up to 30 s), so a test can queue several requests
    /// before the worker runs its first step.
    hold_first_decode: AtomicBool,
}

/// Passes everything through and records what the steps looked like.
struct Spy {
    inner: Box<dyn BatchBackend>,
    seen: Arc<Seen>,
}

impl BatchBackend for Spy {
    fn n_vocab(&self) -> usize {
        self.inner.n_vocab()
    }
    fn n_seq_max(&self) -> usize {
        self.inner.n_seq_max()
    }
    fn n_ctx_seq(&self) -> usize {
        self.inner.n_ctx_seq()
    }
    fn kv_capacity(&self) -> usize {
        self.inner.kv_capacity()
    }
    fn kv_free(&self) -> usize {
        self.inner.kv_free()
    }
    fn seq_len(&self, seq: SeqId) -> usize {
        self.inner.seq_len(seq)
    }
    fn seq_reserved(&self, seq: SeqId) -> usize {
        self.inner.seq_reserved(seq)
    }
    fn seq_reserve(&mut self, seq: SeqId, n: usize) -> Result<(), BackendError> {
        self.inner.seq_reserve(seq, n)
    }
    fn seq_rm(&mut self, seq: SeqId, keep: usize) -> Result<(), BackendError> {
        self.inner.seq_rm(seq, keep)
    }
    fn seq_cp(&mut self, src: SeqId, dst: SeqId, n: usize) -> Result<(), BackendError> {
        self.inner.seq_cp(src, dst, n)
    }
    fn decode(&mut self, inputs: &[SeqInput<'_>]) -> Result<Outputs<'_>, BackendError> {
        let s = &self.seen;
        if s.decodes.fetch_add(1, Ordering::Relaxed) == 0 {
            for _ in 0..3000 {
                if !s.hold_first_decode.load(Ordering::Acquire) {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }
        s.widest.fetch_max(inputs.len(), Ordering::Relaxed);
        let verify = inputs.iter().any(|i| i.all_logits && i.tokens.len() > 1);
        if verify {
            s.verify_decodes.fetch_add(1, Ordering::Relaxed);
            if inputs.iter().any(|i| i.tokens.len() == 1) {
                s.verify_beside_decoder.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.inner.decode(inputs)
    }
    fn seq_snapshot(&mut self, seq: SeqId) -> Result<Vec<u8>, BackendError> {
        self.inner.seq_snapshot(seq)
    }
    fn seq_restore(&mut self, seq: SeqId, bytes: &[u8], n: usize) -> Result<(), BackendError> {
        self.inner.seq_restore(seq, bytes, n)
    }
    fn kv_format(&self) -> String {
        self.inner.kv_format()
    }
    fn on_gpu(&self) -> bool {
        false
    }
    fn bytes_streamed_total(&self) -> Option<u64> {
        None
    }
    fn max_step_tokens(&self) -> usize {
        self.inner.max_step_tokens()
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    Llama,
    Dense,
}

fn spec_config(method: &str) -> Option<SpecConfig> {
    let method = match method {
        "off" => return None,
        "ngram" => SpecMethod::Ngram(NgramDrafter::default()),
        "draft" => SpecMethod::Draft {
            path: fixture(),
            n_ctx: 512,
            n_gpu_layers: 0,
            threads: 2,
            p_min: 0.0,
        },
        other => panic!("unknown method {other}"),
    };
    let mut c = SpecConfig::new(method);
    if matches!(c.method, SpecMethod::Ngram(_)) {
        // Draft whenever the history repeats, so the verify path runs often.
        c.len = DraftLen::Fixed(4);
    }
    Some(c)
}

fn load(kind: Kind, parallel: usize) -> Result<Loaded, String> {
    let path = fixture();
    match kind {
        Kind::Llama => load_llama(
            &path,
            &Placement {
                n_gpu_layers: 0,
                ..Placement::default()
            },
            &ContextConfig {
                n_ctx: 2048,
                n_batch: 64,
                n_ubatch: 64,
                threads: Some(2),
                threads_batch: Some(2),
                n_seq_max: parallel as u32,
                kv_unified: true,
                ..ContextConfig::default()
            },
            None,
        ),
        Kind::Dense => load_dense(
            &path,
            &DenseOptions {
                n_seq_max: parallel,
                n_ctx_pool: 2048,
                threads: Some(2),
                ..DenseOptions::new(BackendChoice::Cpu { threads: Some(2) })
            },
        ),
    }
}

fn start(kind: Kind, parallel: usize, spec: Option<SpecConfig>) -> (Arc<Scheduler>, Arc<Seen>) {
    let mut sc = SchedulerConfig::new(fixture());
    sc.parallel = parallel;
    sc.step_tokens = 64;
    sc.spec = spec;
    let seen = Arc::new(Seen::default());
    let seen2 = Arc::clone(&seen);
    let s = Scheduler::start(sc, move || {
        let Loaded { backend, tokenizer } = load(kind, parallel)?;
        Ok(Loaded {
            backend: Box::new(Spy {
                inner: backend,
                seen: seen2,
            }),
            tokenizer,
        })
    })
    .expect("scheduler");
    (Arc::new(s), seen)
}

/// A prompt that repeats itself, so the n-gram drafter finds matches.
fn prompt(id: u32, n: usize) -> Vec<TokenId> {
    (0..n as u32)
        .map(|i| 10 + (id * 31 + (i % 11) * 13) % 150)
        .collect()
}

fn generate(s: &Scheduler, prompt: Vec<TokenId>, max_tokens: u32) -> Vec<TokenId> {
    generate_at(s, prompt, max_tokens, 0.0)
}

fn generate_at(
    s: &Scheduler,
    prompt: Vec<TokenId>,
    max_tokens: u32,
    temperature: f32,
) -> Vec<TokenId> {
    drain(submit(s, prompt, max_tokens, temperature))
}

/// Queues a request and returns its event stream.
fn submit(
    s: &Scheduler,
    prompt: Vec<TokenId>,
    max_tokens: u32,
    temperature: f32,
) -> tokio::sync::mpsc::Receiver<GenEvent> {
    let id = s.models()[0].id.clone();
    s.generate(GenerateRequest {
        model: id,
        prompt,
        params: SamplingParams {
            max_tokens,
            temperature: Some(temperature),
            seed: Some(7),
            ..SamplingParams::default()
        },
        constraint: None,
        cancel: CancellationToken::new(),
    })
    .expect("generate")
}

fn drain(mut rx: tokio::sync::mpsc::Receiver<GenEvent>) -> Vec<TokenId> {
    let mut out = Vec::new();
    let mut finished = false;
    while let Some(ev) = rx.blocking_recv() {
        assert!(!finished, "event after the terminal one");
        match ev {
            GenEvent::Token(t) => out.push(t),
            GenEvent::Finished { reason, .. } => {
                assert!(
                    matches!(reason, FinishReason::Length | FinishReason::EndOfGeneration),
                    "{reason:?}"
                );
                finished = true;
            }
            GenEvent::Probs(_) => {}
            GenEvent::Error(e) | GenEvent::ContextExceeded(e) => panic!("{e}"),
        }
    }
    assert!(finished, "no terminal event");
    out
}

fn shutdown(s: Arc<Scheduler>) {
    Arc::try_unwrap(s)
        .unwrap_or_else(|_| panic!("a request thread still holds the scheduler"))
        .shutdown()
        .expect("shutdown");
}

/// Top-2 margin, relative to the largest logit, of the one-row steps' logits that choose
/// `want[i]` after `prompt` and `want[..i]`.
fn plain_margin(kind: Kind, prompt: &[TokenId], want: &[TokenId], i: usize) -> f32 {
    let Loaded { mut backend, .. } = load(kind, 1).expect("load");
    let toks: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let mut l = backend
        .decode(&[SeqInput {
            seq: 0,
            tokens: &toks,
            all_logits: false,
        }])
        .expect("prefill")
        .last(0)
        .to_vec();
    for &t in &want[..i] {
        l = backend
            .decode(&[SeqInput {
                seq: 0,
                tokens: &[t as i32],
                all_logits: false,
            }])
            .expect("decode")
            .last(0)
            .to_vec();
    }
    let top = argmax(&l);
    let second = l
        .iter()
        .enumerate()
        .filter(|&(j, _)| j != top)
        .map(|(_, &x)| x)
        .fold(f32::MIN, f32::max);
    let scale = l.iter().fold(0f32, |a, x| a.max(x.abs()));
    (l[top] - second) / scale
}

/// Top-2 margin below which plain greedy counts as a near-tie (ADR 0043, 0052): verification
/// batches are not bitwise batch-invariant on AVX2 and AVX-512 builds, and CI runners flipped a
/// token at a margin of 0.0044.
fn near_tie() -> f32 {
    let info = tetraeder_ggml_sys::cpu::backend_info().expect("CPU backend");
    if info.isa.iter().any(|f| f == "AVX512") {
        1e-2
    } else {
        1e-3
    }
}

/// `got` equals the plain output `want`, or first leaves it where plain greedy was a near-tie.
fn assert_agrees(kind: Kind, prompt: &[TokenId], got: &[TokenId], want: &[TokenId], what: &str) {
    let n = got.len().min(want.len());
    let Some(i) = (0..n).find(|&i| got[i] != want[i]) else {
        assert_eq!(got.len(), want.len(), "{what}: lengths differ");
        return;
    };
    let m = plain_margin(kind, prompt, want, i);
    eprintln!("{what}: leaves plain greedy at token {i}, plain top-2 margin {m}");
    assert!(
        m < near_tie(),
        "{what}: speculation changed the output at token {i}, where plain greedy had a margin of {m}"
    );
}

fn one_request(kind: Kind, method: &str) -> (Vec<TokenId>, Arc<Seen>) {
    let (s, seen) = start(kind, 2, spec_config(method));
    let out = generate(&s, prompt(1, 40), 120);
    shutdown(s);
    (out, seen)
}

fn check_exact(kind: Kind, method: &str) {
    let (plain, plain_seen) = one_request(kind, "off");
    assert!(
        plain.len() >= 100,
        "the tiny model stopped early: {}",
        plain.len()
    );
    assert_eq!(plain_seen.verify_decodes.load(Ordering::Relaxed), 0);
    // The margin helper runs on the real backends (it matters only when a run diverges).
    assert!(plain_margin(kind, &prompt(1, 40), &plain, 10) >= 0.0);
    let (spec, seen) = one_request(kind, method);
    assert_agrees(
        kind,
        &prompt(1, 40),
        &spec,
        &plain,
        &format!("{kind:?} {method}"),
    );
    let verifies = seen.verify_decodes.load(Ordering::Relaxed);
    assert!(verifies > 0, "{kind:?} {method}: nothing speculated");
    // Accepted drafts save steps.
    let (d, p) = (
        seen.decodes.load(Ordering::Relaxed),
        plain_seen.decodes.load(Ordering::Relaxed),
    );
    eprintln!("{kind:?} {method}: {verifies} verify steps, {d} decodes against {p} plain");
    if method == "draft" {
        assert!(d < p, "the draft model saved no steps: {d} against {p}");
    }
}

/// hardware: cpu.
#[test]
fn cpu_llama_ngram_equals_plain() {
    check_exact(Kind::Llama, "ngram");
}

/// hardware: cpu.
#[test]
fn cpu_llama_draft_model_equals_plain() {
    check_exact(Kind::Llama, "draft");
}

/// hardware: cpu.
#[test]
fn cpu_dense_ngram_equals_plain() {
    check_exact(Kind::Dense, "ngram");
}

/// hardware: cpu.
#[test]
fn cpu_dense_draft_model_equals_plain() {
    check_exact(Kind::Dense, "draft");
}

/// A sampled request never speculates, and its seeded output is the same with speculation on.
/// hardware: cpu.
#[test]
fn cpu_sampled_requests_do_not_speculate() {
    let (s, _) = start(Kind::Dense, 2, None);
    let plain = generate_at(&s, prompt(1, 40), 60, 0.8);
    shutdown(s);
    let (s, seen) = start(Kind::Dense, 2, spec_config("draft"));
    let got = generate_at(&s, prompt(1, 40), 60, 0.8);
    shutdown(s);
    assert_eq!(got, plain);
    assert_eq!(seen.verify_decodes.load(Ordering::Relaxed), 0);
}

/// Two requests decoding together: no verify rows next to another sequence's decode row, and
/// both outputs equal the plain ones.
fn check_concurrent(kind: Kind) {
    let (s, _) = start(kind, 2, None);
    let alone = [
        generate(&s, prompt(2, 20), 100),
        generate(&s, prompt(3, 25), 100),
    ];
    shutdown(s);

    let (s, seen) = start(kind, 2, spec_config("draft"));
    // Both requests are queued before the worker runs its first step, so they overlap.
    seen.hold_first_decode.store(true, Ordering::Release);
    let handles: Vec<_> = [(2, 20), (3, 25)]
        .into_iter()
        .map(|(id, n)| {
            let rx = submit(&s, prompt(id, n), 100, 0.0);
            std::thread::spawn(move || drain(rx))
        })
        .collect();
    seen.hold_first_decode.store(false, Ordering::Release);
    let got: Vec<_> = handles
        .into_iter()
        .map(|h| h.join().expect("request thread"))
        .collect();
    shutdown(s);
    for (i, (n, g)) in [(20, &got[0]), (25, &got[1])].into_iter().enumerate() {
        let p = prompt(2 + i as u32, n);
        assert_agrees(kind, &p, g, &alone[i], &format!("{kind:?} request {i}"));
    }
    assert_eq!(
        seen.verify_beside_decoder.load(Ordering::Relaxed),
        0,
        "{kind:?}: a sequence speculated next to another decoding one"
    );
    assert!(seen.widest.load(Ordering::Relaxed) > 1, "never batched");
}

/// hardware: cpu.
#[test]
fn cpu_llama_two_requests_do_not_speculate() {
    check_concurrent(Kind::Llama);
}

/// hardware: cpu.
#[test]
fn cpu_dense_two_requests_do_not_speculate() {
    check_concurrent(Kind::Dense);
}

/// `SeqTarget` lets `Speculator` run on a batch backend: its output equals plain greedy decoding
/// of the same sequence, and the rewind leaves exactly the accepted prefix in the sequence.
#[test]
fn seq_target_speculates_like_plain_decoding_on_fake_kv() {
    const NV: usize = 40;
    let greedy = |kv: &mut fake::FakeKv, toks: &mut Vec<i32>, n: usize| {
        for _ in 0..n {
            let out = kv
                .decode(&[SeqInput {
                    seq: 0,
                    tokens: &toks[kv.seq_len(0)..],
                    all_logits: false,
                }])
                .unwrap();
            let next = tetraeder_scheduler::spec::argmax(out.last(0)) as i32;
            toks.push(next);
        }
    };
    let mut plain_kv = fake::FakeKv::new(NV, 1, 2000, 2000);
    let mut plain: Vec<i32> = (0..12).map(|i| (i * 7 % NV as i32).max(1)).collect();
    plain_kv.seq_reserve(0, 400).unwrap();
    greedy(&mut plain_kv, &mut plain, 60);

    let mut kv = fake::FakeKv::new(NV, 1, 2000, 2000);
    kv.seq_reserve(0, 400).unwrap();
    let mut hist: Vec<i32> = plain[..12].to_vec();
    // Prefill all but the last token, as the speculator expects.
    kv.decode(&[SeqInput {
        seq: 0,
        tokens: &hist[..11],
        all_logits: false,
    }])
    .unwrap();
    let mut sp = Speculator::new(DraftLen::Fixed(MAX_VERIFY - 1), Sampling::GREEDY, None);
    let mut d = NgramDrafter::default();
    while hist.len() < 60 {
        let mut t = SeqTarget {
            backend: &mut kv,
            seq: 0,
        };
        sp.step(&mut t, Some(&mut d), &mut hist).unwrap();
        assert_eq!(
            kv.seq_len(0),
            hist.len() - 1,
            "KV is the history minus the pending token"
        );
    }
    let n = hist.len().min(plain.len());
    assert_eq!(hist[..n], plain[..n]);
    assert!(sp.stats().steps > 0);
}
