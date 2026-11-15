//! The worker driven step by step over `FakeKv`, for what needs a look inside it (review fixes
//! after C3b and C4): admission checks, retrying a failed step, `reserved_to`, the RAM cache.
//!
//! Hardware label: none.

use super::*;

#[path = "../../tests/fake/mod.rs"]
mod fake;
use fake::FakeKv;

const N_VOCAB: usize = 300;

type Rx = mpsc::Receiver<GenEvent>;

fn worker(kv: FakeKv) -> Worker {
    let parallel = kv.n_seq_max();
    let vocab = Arc::new(Vocab {
        tokens: vec![b"a".to_vec(); N_VOCAB],
        special: Vec::new(),
        eog: Vec::new(),
    });
    let defaults = SampleParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
    };
    Worker::new(
        Box::new(kv),
        PrefixCache::new(1 << 20),
        32,
        16,
        true,
        vocab,
        defaults,
        None,
        false,
        parallel,
        512,
    )
}

fn req(prompt: &[TokenId], max: u32) -> (Req, Rx) {
    let (tx, rx) = mpsc::channel(10_000);
    let params = SamplingParams {
        max_tokens: max,
        temperature: Some(0.0),
        ..SamplingParams::default()
    };
    (
        Req::new(prompt.to_vec(), params, None, CancellationToken::new(), tx),
        rx,
    )
}

fn submit(w: &mut Worker, prompt: &[TokenId], max: u32) -> Rx {
    let (r, rx) = req(prompt, max);
    w.waiting.push_back(r);
    rx
}

fn pump(w: &mut Worker) {
    for _ in 0..100_000 {
        w.flush_drains();
        if !w.has_work() {
            return;
        }
        w.admit();
        w.sweep();
        w.step();
    }
    panic!("the worker did not finish");
}

fn events(rx: &mut Rx) -> Vec<GenEvent> {
    let mut out = Vec::new();
    while let Ok(e) = rx.try_recv() {
        out.push(e);
    }
    out
}

fn tokens(evs: &[GenEvent]) -> usize {
    evs.iter()
        .filter(|e| matches!(e, GenEvent::Token(_)))
        .count()
}

fn prompt(seed: u32, n: usize) -> Vec<TokenId> {
    (0..n as u32)
        .map(|i| 4 + (seed * 31 + i * 7) % 200)
        .collect()
}

#[test]
fn an_out_of_vocabulary_token_fails_only_its_request() {
    let mut w = worker(FakeKv::new(N_VOCAB, 2, 3200, 4000));
    let mut bad = prompt(1, 20);
    bad[7] = N_VOCAB as u32 + 5;
    let mut a = submit(&mut w, &bad, 5);
    let mut b = submit(&mut w, &prompt(2, 20), 5);
    pump(&mut w);
    let (ea, eb) = (events(&mut a), events(&mut b));
    match ea.as_slice() {
        [GenEvent::Error(m)] => assert!(m.contains("outside the vocabulary"), "{m}"),
        other => panic!("{other:?}"),
    }
    assert_eq!(tokens(&eb), 5);
    assert!(matches!(eb.last(), Some(GenEvent::Finished { .. })));
}

/// The backend rejects one input of a step that holds two: the other request still finishes.
#[test]
fn a_failed_step_is_retried_so_only_the_offender_fails() {
    let kv = FakeKv::new(N_VOCAB, 3, 3200, 6000).with_poison(Some(250));
    let mut w = worker(kv);
    let mut bad = prompt(1, 20);
    bad[3] = 250;
    let mut a = submit(&mut w, &bad, 5);
    let mut b = submit(&mut w, &prompt(2, 20), 8);
    let mut c = submit(&mut w, &prompt(3, 20), 8);
    pump(&mut w);
    assert!(matches!(events(&mut a).as_slice(), [GenEvent::Error(_)]));
    for rx in [&mut b, &mut c] {
        let e = events(rx);
        assert_eq!(tokens(&e), 8, "{e:?}");
        assert!(matches!(e.last(), Some(GenEvent::Finished { .. })));
    }
}

/// A preempted request that filled the context ends with `Length`, not "prompt does not fit".
#[test]
fn a_requeued_request_that_filled_the_context_ends_with_length() {
    let mut w = worker(FakeKv::new(N_VOCAB, 1, 64, 64));
    let (mut r, mut rx) = req(&prompt(1, 64), 10);
    r.stats.generated_tokens = 3;
    w.waiting.push_back(r);
    w.admit();
    let e = events(&mut rx);
    assert!(
        matches!(
            e.last(),
            Some(GenEvent::Finished {
                reason: FinishReason::Length,
                ..
            })
        ),
        "{e:?}"
    );
    // A fresh prompt that long still does not fit.
    let mut rx = submit(&mut w, &prompt(2, 64), 10);
    w.admit();
    assert!(matches!(events(&mut rx).as_slice(), [GenEvent::Error(_)]));
}

/// After the idle prefix is cut, `reserved_to` is what the backend holds (the dense backend
/// shrinks a reservation to the rows it keeps).
#[test]
fn reserved_to_follows_the_backend_after_a_prefix_cut() {
    let kv = FakeKv::new(N_VOCAB, 2, 3200, 8000).with_shrink_on_rm(Some(256));
    let mut w = worker(kv);
    let first = prompt(1, 600);
    let mut a = submit(&mut w, &first, 1);
    pump(&mut w);
    assert_eq!(tokens(&events(&mut a)), 1);
    let mut second = first[..300].to_vec();
    second.extend(prompt(9, 100));
    let _b = submit(&mut w, &second, 50);
    w.admit();
    let (id, held) = w
        .slots
        .iter()
        .enumerate()
        .find_map(|(i, s)| match s {
            Slot::Live(s) => Some((i, s.reserved_to)),
            _ => None,
        })
        .expect("admitted");
    assert_eq!(held, w.backend.seq_reserved(id as SeqId));
    assert!(held < 400 + GROW);
}

/// A saved sequence whose bytes cannot be restored is dropped (retrying would fail again), and
/// the request still runs, from an empty sequence. Entries are kept only on KvFull.
#[test]
fn a_corrupt_saved_entry_is_dropped_and_the_request_runs() {
    let mut w = worker(FakeKv::new(N_VOCAB, 1, 3200, 4000));
    let saved = prompt(4, 200);
    // Too short for 200 positions: the backend rejects it.
    w.cache.insert(
        &w.backend.kv_format(),
        saved.iter().map(|&t| t as i32).collect(),
        vec![0; 10],
    );
    let mut longer = saved.clone();
    longer.extend(prompt(5, 10));
    let mut rx = submit(&mut w, &longer, 3);
    w.admit();
    assert_eq!(w.cache.len(), 0, "the unusable entry was kept");
    pump(&mut w);
    assert_eq!(tokens(&events(&mut rx)), 3);
}
