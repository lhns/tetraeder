//! Context shift and slot actions (ADR 0090) over `FakeKv`, whose logits depend on the tokens
//! only, so what is checked is the scheduler's bookkeeping: positions, reuse and files.
//!
//! Hardware label: none (the tokenizer comes from the tiny fixture).

mod fake;

use fake::FakeKv;
use std::path::PathBuf;
use std::sync::Arc;
use tetraeder_gateway::engine::{
    Engine, EngineError, FinishReason, GenEvent, GenerateRequest, SamplingExtras, SamplingParams,
    SlotAction, SlotReply, TokenConstraint, TokenMask,
};
use tetraeder_ggml_sys::llama::{Model, ModelParams};
use tetraeder_scheduler::{Loaded, Scheduler, SchedulerConfig};
use tokio_util::sync::CancellationToken;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3moe-f16.gguf")
}

struct Setup {
    parallel: usize,
    n_ctx: usize,
    shift_backend: bool,
    context_shift: bool,
    slot_dir: Option<PathBuf>,
}

impl Default for Setup {
    fn default() -> Self {
        Self {
            parallel: 2,
            n_ctx: 64,
            shift_backend: true,
            context_shift: false,
            slot_dir: None,
        }
    }
}

fn start(s: Setup) -> Result<Scheduler, String> {
    let path = fixture();
    let mut sc = SchedulerConfig::new(&path);
    sc.parallel = s.parallel;
    sc.step_tokens = 64;
    sc.context_shift = s.context_shift;
    sc.slot_dir = s.slot_dir.clone();
    Scheduler::start(sc, move || {
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
        let mut kv = FakeKv::new(n_vocab, s.parallel, s.n_ctx, s.n_ctx * s.parallel);
        if s.shift_backend {
            kv = kv.with_shift();
        }
        Ok(Loaded {
            backend: Box::new(kv),
            tokenizer,
        })
    })
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

fn run(s: &Scheduler, prompt: Vec<u32>, max_tokens: u32, extra: SamplingExtras) -> Vec<GenEvent> {
    let mut rx = s
        .generate(GenerateRequest {
            model: s.models()[0].id.clone(),
            prompt,
            params: SamplingParams {
                max_tokens,
                temperature: Some(0.0),
                extra,
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

fn finished(events: &[GenEvent]) -> (FinishReason, u32, u32) {
    match events.last() {
        Some(GenEvent::Finished { reason, stats }) => {
            (*reason, stats.generated_tokens, stats.cached_tokens)
        }
        other => panic!("not finished: {other:?}"),
    }
}

fn prompt(n: usize) -> Vec<u32> {
    (0..n).map(|i| 10 + (i as u32 * 7) % 150).collect()
}

fn temp_dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("tetraeder-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&d).expect("temp dir");
    d
}

#[test]
fn without_context_shift_a_full_context_ends_the_request() {
    let s = start(Setup::default()).unwrap();
    let (reason, generated, _) = finished(&run(&s, prompt(10), 200, SamplingExtras::default()));
    assert_eq!(reason, FinishReason::Length);
    assert!(generated < 64, "{generated} tokens in a context of 64");
}

#[test]
fn context_shift_goes_on_past_the_context() {
    let s = start(Setup {
        context_shift: true,
        ..Setup::default()
    })
    .unwrap();
    let (reason, generated, _) = finished(&run(&s, prompt(10), 200, SamplingExtras::default()));
    assert_eq!((reason, generated), (FinishReason::Length, 200));
    // with n_keep and n_discard from the request
    let extra = SamplingExtras {
        n_keep: Some(4),
        n_discard: Some(8),
        ..SamplingExtras::default()
    };
    let (reason, generated, _) = finished(&run(&s, prompt(10), 150, extra));
    assert_eq!((reason, generated), (FinishReason::Length, 150));
}

#[test]
fn a_shifted_sequence_is_not_kept_for_reuse() {
    let s = start(Setup {
        context_shift: true,
        parallel: 1,
        ..Setup::default()
    })
    .unwrap();
    let p = prompt(10);
    finished(&run(&s, p.clone(), 100, SamplingExtras::default()));
    // the same prompt again finds nothing to reuse: the sequence was freed, not left idle
    let (_, _, cached) = finished(&run(&s, p, 4, SamplingExtras::default()));
    assert_eq!(cached, 0);
}

#[test]
fn context_shift_is_disabled_with_a_warning_on_a_backend_that_cannot_shift() {
    // As llama-server ("ctx_shift is not supported by this context, it will be disabled"): the
    // server starts and a full context ends the request.
    let s = start(Setup {
        context_shift: true,
        shift_backend: false,
        ..Setup::default()
    })
    .expect("started");
    let (reason, generated, _) = finished(&run(&s, prompt(10), 200, SamplingExtras::default()));
    assert_eq!(reason, FinishReason::Length);
    assert!(generated < 64, "{generated} tokens in a context of 64");
}

fn slot(s: &Scheduler, id: i32, a: SlotAction) -> Result<SlotReply, EngineError> {
    s.slot_action(id, a)?
        .blocking_recv()
        .map_err(|_| EngineError::Other("the engine worker has stopped".into()))?
}

#[test]
fn slots_save_erase_and_restore_a_sequence() {
    let dir = temp_dir("slots");
    let s = start(Setup {
        slot_dir: Some(dir.clone()),
        ..Setup::default()
    })
    .unwrap();
    assert!(s.slot_actions());
    let pin = |i| SamplingExtras {
        slot: Some(i),
        ..SamplingExtras::default()
    };
    let p = prompt(20);
    finished(&run(&s, p.clone(), 5, pin(1)));

    // slot 1 holds the prompt and what was generated but the last token; slot 0 is empty
    let SlotReply::Saved { tokens, bytes, .. } = slot(
        &s,
        1,
        SlotAction::Save {
            filename: "one.bin".into(),
        },
    )
    .unwrap() else {
        panic!("not a save")
    };
    assert!((20..=25).contains(&tokens), "{tokens} tokens saved");
    assert_eq!(
        std::fs::metadata(dir.join("one.bin")).unwrap().len(),
        bytes,
        "n_written is the file's size"
    );
    let SlotReply::Saved { tokens: empty, .. } = slot(
        &s,
        0,
        SlotAction::Save {
            filename: "zero.bin".into(),
        },
    )
    .unwrap() else {
        panic!("not a save")
    };
    assert_eq!(empty, 0);

    // erase, then the same prompt finds nothing
    assert_eq!(
        slot(&s, 1, SlotAction::Erase).unwrap(),
        SlotReply::Erased { tokens }
    );
    assert_eq!(finished(&run(&s, p.clone(), 3, pin(1))).2, 0);
    assert_eq!(slot(&s, 1, SlotAction::Erase).map(|_| ()), Ok(()));

    // restore into the other slot: the prompt is a hit there
    let SlotReply::Restored { tokens: back, .. } = slot(
        &s,
        0,
        SlotAction::Restore {
            filename: "one.bin".into(),
        },
    )
    .unwrap() else {
        panic!("not a restore")
    };
    assert_eq!(back, tokens);
    let (_, _, cached) = finished(&run(&s, p, 3, pin(0)));
    assert!(
        cached >= 19,
        "{cached} tokens reused from the restored slot"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn slot_actions_refuse_what_llama_server_refuses() {
    let dir = temp_dir("slots-err");
    let s = start(Setup {
        slot_dir: Some(dir.clone()),
        ..Setup::default()
    })
    .unwrap();
    let invalid = |r: Result<SlotReply, EngineError>| match r {
        Err(EngineError::Invalid(m)) => m,
        other => panic!("not refused: {other:?}"),
    };
    assert_eq!(invalid(slot(&s, 2, SlotAction::Erase)), "Invalid slot ID");
    assert_eq!(invalid(slot(&s, -1, SlotAction::Erase)), "Invalid slot ID");
    let m = invalid(slot(
        &s,
        0,
        SlotAction::Restore {
            filename: "missing.bin".into(),
        },
    ));
    assert!(m.starts_with("Unable to restore slot"), "{m}");
    std::fs::write(dir.join("junk.bin"), b"not a slot file at all").unwrap();
    let m = invalid(slot(
        &s,
        0,
        SlotAction::Restore {
            filename: "junk.bin".into(),
        },
    ));
    assert!(m.contains("not a slot save file"), "{m}");
    // a slot out of bounds wraps around, as in llama-server's `get_slot_by_id`
    let ev = run(
        &s,
        prompt(5),
        2,
        SamplingExtras {
            slot: Some(5),
            ..SamplingExtras::default()
        },
    );
    finished(&ev);
    std::fs::remove_dir_all(&dir).unwrap();

    let off = start(Setup::default()).unwrap();
    assert!(!off.slot_actions());
}

/// Events of `rx` until it ends, or `None` after `secs` seconds.
fn collect_within(
    rx: &mut tokio::sync::mpsc::Receiver<GenEvent>,
    secs: u64,
) -> Option<Vec<GenEvent>> {
    let end = std::time::Instant::now() + std::time::Duration::from_secs(secs);
    let mut out = Vec::new();
    while std::time::Instant::now() < end {
        match rx.try_recv() {
            Ok(ev) => out.push(ev),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => return Some(out),
        }
    }
    None
}

fn submit(
    s: &Scheduler,
    prompt: Vec<u32>,
    max_tokens: u32,
    slot: Option<usize>,
) -> tokio::sync::mpsc::Receiver<GenEvent> {
    s.generate(GenerateRequest {
        model: s.models()[0].id.clone(),
        prompt,
        params: SamplingParams {
            max_tokens,
            temperature: Some(0.0),
            extra: SamplingExtras {
                slot,
                ..SamplingExtras::default()
            },
            ..SamplingParams::default()
        },
        constraint: Some(Box::new(Only(vec![65, 66, 67]))),
        cancel: CancellationToken::new(),
    })
    .expect("generate")
}

/// A request pinned to a running slot waits for it; the ones behind it do not wait with it.
#[test]
fn a_request_pinned_to_a_busy_slot_does_not_hold_up_the_others() {
    let s = start(Setup::default()).unwrap();
    // never read: after its channel fills the request is paused and its slot stays busy
    let a = submit(&s, prompt(8), 60, Some(0));
    std::thread::sleep(std::time::Duration::from_millis(300));
    let mut b = submit(&s, prompt(9), 3, Some(0));
    let mut c = submit(&s, prompt(10), 3, None);
    let c_events =
        collect_within(&mut c, 10).expect("the unpinned request ran beside the waiting one");
    assert!(matches!(c_events.last(), Some(GenEvent::Finished { .. })));
    assert!(
        b.try_recv().is_err(),
        "the pinned request is still waiting for slot 0"
    );
    // the first request goes away; the pinned one takes the slot
    drop(a);
    let b_events =
        collect_within(&mut b, 10).expect("the pinned request ran once its slot was free");
    assert!(matches!(b_events.last(), Some(GenEvent::Finished { .. })));
}
