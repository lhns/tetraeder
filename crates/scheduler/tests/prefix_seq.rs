//! Prefix reuse from busy sequences (`seq_cp`) and `prefix_reuse: false` (ADR 0065, brief C3b),
//! through the gateway `Engine` trait on `Scheduler` over `FakeKv`, and `seq_cp` on libllama.
//!
//! Hardware label: none for the FakeKv tests (the tokenizer comes from the tiny fixture); the
//! libllama test needs cpu.

mod fake;

use fake::{FakeKv, Probe};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use tetraeder_engine::{ContextConfig, Engine as LlamaEngine, Placement};
use tetraeder_gateway::engine::{
    Engine, FinishReason, GenEvent, GenerateRequest, SamplingParams, TokenConstraint, TokenMask,
};
use tetraeder_ggml_sys::llama::{Model, ModelParams};
use tetraeder_scheduler::{
    BatchBackend, LlamaBackend, Loaded, Scheduler, SchedulerConfig, SeqInput,
};
use tokio_util::sync::CancellationToken;

const N_CTX_SEQ: usize = 3200;
/// Batched and single-row kernels differ by CPU variant (see engine's multi_seq.rs): equal
/// within this or 1e-3 of the largest logit.
const TOL_CPU: f32 = 1e-4;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3moe-f16.gguf")
}

fn start(prefix_reuse: bool, parallel: usize) -> (Arc<Scheduler>, Arc<Probe>) {
    let path = fixture();
    let mut sc = SchedulerConfig::new(&path);
    sc.parallel = parallel;
    sc.step_tokens = 512;
    sc.cache_save_min = 32;
    sc.cache_restore_gain = 16;
    sc.prefix_reuse = prefix_reuse;
    let probe_cell: Arc<std::sync::OnceLock<Arc<Probe>>> = Arc::default();
    let cell = Arc::clone(&probe_cell);
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
        let kv = FakeKv::new(n_vocab, parallel, N_CTX_SEQ, N_CTX_SEQ * parallel);
        let _ = cell.set(kv.probe());
        Ok(Loaded {
            backend: Box::new(kv),
            tokenizer,
        })
    })
    .expect("scheduler over FakeKv");
    (Arc::new(s), probe_cell.get().cloned().expect("probe"))
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

type Events = tokio::sync::mpsc::Receiver<GenEvent>;

fn generate(s: &Scheduler, prompt: Vec<u32>, max_tokens: u32) -> Events {
    s.generate(GenerateRequest {
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
    .expect("generate")
}

/// Reads to the end; returns the generated tokens and `cached_tokens`.
fn finish(mut rx: Events) -> (Vec<u32>, u32) {
    let mut toks = Vec::new();
    loop {
        match rx.blocking_recv().expect("terminal event") {
            GenEvent::Token(t) => toks.push(t),
            GenEvent::Finished { reason, stats } => {
                assert_eq!(reason, FinishReason::Length);
                return (toks, stats.cached_tokens);
            }
            GenEvent::Probs(_) => {}
            GenEvent::Error(e) | GenEvent::ContextExceeded(e) => panic!("{e}"),
        }
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

fn wait_for_decodes(probe: &Probe, n: u32) {
    let t0 = Instant::now();
    while probe.decodes.load(Ordering::SeqCst) < n {
        assert!(
            t0.elapsed() < Duration::from_secs(30),
            "first never decodes"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn shutdown(s: Arc<Scheduler>) {
    Arc::try_unwrap(s)
        .ok()
        .expect("no other handle")
        .shutdown()
        .expect("shutdown");
}

/// The second request arrives while the first still decodes (its reader is not reading, so it
/// is paused with a full channel and stays busy): it copies the shared prefix.
#[test]
fn a_busy_sequence_lends_its_prefix() {
    let (s, probe) = start(true, 2);
    let first = generate(&s, prompt(1000, 1, 1000), 400);
    wait_for_decodes(&probe, 4);
    let (_, cached) = finish(generate(&s, prompt(1050, 2, 1000), 5));
    assert!(cached >= 1000, "cached {cached}");
    let cps = probe.seq_cps.lock().unwrap().clone();
    assert!(
        cps.iter().any(|&(src, dst, n)| src != dst && n >= 1000),
        "no seq_cp of the shared prefix: {cps:?}"
    );
    let (toks, _) = finish(first);
    assert_eq!(toks.len(), 400);
    shutdown(s);
}

/// The copy does not change what the new request generates: same tokens as without reuse.
#[test]
fn a_copied_prefix_generates_what_a_fresh_prefill_does() {
    let run = |reuse: bool| {
        let (s, probe) = start(reuse, 2);
        let first = generate(&s, prompt(600, 1, 600), 300);
        wait_for_decodes(&probe, 3);
        let out = finish(generate(&s, prompt(640, 2, 600), 20));
        drop(first);
        shutdown(s);
        out
    };
    let (with, cached) = run(true);
    let (without, none) = run(false);
    assert!(cached >= 600, "cached {cached}");
    assert_eq!(none, 0);
    assert_eq!(with, without);
}

/// Five turns of one conversation: with reuse off nothing is cached or copied; with it on,
/// later turns hit.
#[test]
fn no_prefix_reuse_caches_nothing_over_a_conversation() {
    for reuse in [false, true] {
        let (s, probe) = start(reuse, 2);
        let mut cached = Vec::new();
        for turn in 1..=5 {
            let (_, c) = finish(generate(&s, prompt(100 * turn, 0, 100 * turn), 8));
            cached.push(c);
        }
        if reuse {
            assert!(cached[1..].iter().all(|&c| c >= 100), "{cached:?}");
        } else {
            assert!(cached.iter().all(|&c| c == 0), "{cached:?}");
            assert!(probe.seq_cps.lock().unwrap().is_empty());
        }
        shutdown(s);
    }
}

/// An identical prompt repeated after other prompts filled the ids: nothing is reused, from
/// idle sequences or from the RAM cache.
#[test]
fn no_prefix_reuse_ignores_idle_and_saved_sequences() {
    let (s, probe) = start(false, 2);
    for seed in 1..=4 {
        let (_, c) = finish(generate(&s, prompt(200, seed, 0), 4));
        assert_eq!(c, 0);
    }
    let (_, c) = finish(generate(&s, prompt(200, 4, 0), 4));
    assert_eq!(c, 0);
    assert!(probe.seq_cps.lock().unwrap().is_empty());
    shutdown(s);
}

fn llama(n_seq_max: u32) -> LlamaBackend {
    let cfg = ContextConfig {
        n_ctx: 1024,
        n_batch: 64,
        n_ubatch: 64,
        threads: Some(2),
        n_seq_max,
        kv_unified: true,
        ..ContextConfig::default()
    };
    let e = LlamaEngine::load(fixture(), &Placement::cpu(), &cfg, None).expect("fixture loads");
    LlamaBackend::new(e, false)
}

fn last_logits(b: &mut LlamaBackend, seq: u32, toks: &[i32]) -> Vec<f32> {
    let out = b
        .decode(&[SeqInput {
            seq,
            tokens: toks,
            all_logits: false,
        }])
        .expect("decode");
    out.rows(0).to_vec()
}

/// On libllama: a sequence copied from a busy one continues like a fresh prefill of the same
/// tokens.
#[test]
fn llama_seq_cp_continuation_equals_a_fresh_prefill() {
    let p: Vec<i32> = (0..25).map(|i| 4 + (8 * 31 + i * 7) % 200).collect();
    let extra = [11, 12, 13];
    let all: Vec<i32> = p.iter().chain(&extra).copied().collect();

    let mut fresh = llama(1);
    fresh.seq_reserve(0, 64).unwrap();
    let want = last_logits(&mut fresh, 0, &all);

    let mut b = llama(2);
    b.seq_reserve(0, 64).unwrap();
    b.seq_reserve(1, 64).unwrap();
    last_logits(&mut b, 0, &p);
    b.seq_cp(0, 1, p.len()).unwrap();
    let got = last_logits(&mut b, 1, &extra);
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let tol = TOL_CPU.max(1e-3 * scale);
    let worst = got
        .iter()
        .zip(&want)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max);
    assert!(worst <= tol, "max |diff| {worst} > {tol}");
}
