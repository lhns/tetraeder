//! The worker driven step by step over `FakeKv`, for what needs a look inside it (review fixes
//! after C3b and C4): admission checks, retrying a failed step, `reserved_to`, the RAM cache.
//!
//! Hardware label: none.

use super::*;
use std::sync::atomic::Ordering;

#[path = "../../tests/fake/mod.rs"]
mod fake;
use fake::FakeKv;

const N_VOCAB: usize = 300;

type Rx = mpsc::Receiver<GenEvent>;

fn worker(kv: FakeKv) -> Worker {
    worker_with(kv, 512)
}

fn worker_with(kv: FakeKv, step_tokens: usize) -> Worker {
    worker_cache(kv, step_tokens, PrefixCache::new(1 << 20))
}

fn worker_cache(kv: FakeKv, step_tokens: usize, cache: PrefixCache) -> Worker {
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
        cache,
        32,
        16,
        true,
        vocab,
        defaults,
        None,
        false,
        parallel,
        step_tokens,
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
    // A fresh prompt that long still does not fit: the client's error.
    let mut rx = submit(&mut w, &prompt(2, 64), 10);
    w.admit();
    assert!(matches!(
        events(&mut rx).as_slice(),
        [GenEvent::ContextExceeded(_)]
    ));
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

/// A failed busy copy marks the request, not the worker: when that request ends before it is
/// placed again (cancelled here, by the copy itself), the next request still gets a busy copy.
#[test]
fn a_failed_busy_copy_does_not_stop_the_next_request_from_copying() {
    let cancel = CancellationToken::new();
    let kv = FakeKv::new(N_VOCAB, 3, 3200, 6000).with_fail_cp_cancelling(cancel.clone());
    let probe = kv.probe();
    let mut w = worker(kv);
    let first = prompt(1, 100);
    let _a = submit(&mut w, &first, 200);
    for _ in 0..3 {
        w.admit();
        w.step();
    }
    // B shares 80 tokens with the live first request; its copy fails and cancels it.
    let mut shared = first[..80].to_vec();
    shared.extend(prompt(7, 10));
    let (tx, mut rb) = mpsc::channel(100);
    let params = SamplingParams {
        max_tokens: 5,
        temperature: Some(0.0),
        ..SamplingParams::default()
    };
    w.waiting
        .push_back(Req::new(shared, params, None, cancel, tx));
    w.admit();
    assert!(w.waiting.is_empty(), "the cancelled request left the queue");
    assert!(matches!(
        events(&mut rb).last(),
        Some(GenEvent::Finished {
            reason: FinishReason::Cancelled,
            ..
        })
    ));
    assert!(probe.seq_cps.lock().unwrap().is_empty());
    // C shares 70 tokens with the first request and copies them.
    let mut shared = first[..70].to_vec();
    shared.extend(prompt(8, 10));
    let _c = submit(&mut w, &shared, 5);
    w.admit();
    assert_eq!(probe.seq_cps.lock().unwrap().len(), 1, "no busy copy");
}

/// A request whose prompt shares only a chat-template header with the idle sequences.
fn chat(seed: u32) -> Vec<TokenId> {
    let mut p = vec![1, 2, 3];
    p.extend(prompt(seed, 40));
    p
}

/// Runs `waves` waves of `parallel` requests, each to its end, and checks every one finished.
fn run_waves(w: &mut Worker, waves: u32, parallel: u32) {
    for wave in 0..waves {
        let mut rxs: Vec<Rx> = (0..parallel)
            .map(|i| submit(w, &chat(wave * parallel + i), 10))
            .collect();
        pump(w);
        for rx in &mut rxs {
            assert_eq!(tokens(&events(rx)), 10);
        }
    }
}

/// When the pool has no rows for a new request, the oldest idle sequence is saved and evicted.
#[test]
fn under_kv_pressure_an_idle_sequence_is_still_saved() {
    // Each request reserves 43 + 10 rows: four idle ones and one new fit, a second new does not.
    let kv = FakeKv::new(N_VOCAB, 4, 3200, 300);
    let probe = kv.probe();
    let mut w = worker_with(kv, 2048);
    run_waves(&mut w, 2, 4);
    assert!(probe.snapshots.load(Ordering::SeqCst) > 0);
    assert!(!w.cache.is_empty(), "the evicted sequences are in RAM");
    // A saved prompt comes back from RAM.
    let mut rx = submit(&mut w, &chat(0), 10);
    pump(&mut w);
    assert_eq!(tokens(&events(&mut rx)), 10);
}

/// Prompts that wait together are prefilled in one step under the default budget, as
/// llama-server takes them in one `n_batch`; 512 split four 405-token prompts over four steps.
#[test]
fn prompts_waiting_together_share_one_step_under_the_default_budget() {
    let kv = FakeKv::new(N_VOCAB, 4, 3200, 20_000);
    let probe = kv.probe();
    let mut w = worker_with(kv, STEP_TOKENS);
    let _rxs: Vec<Rx> = (0..4).map(|i| submit(&mut w, &prompt(i, 405), 2)).collect();
    w.admit();
    assert!(w.step());
    assert_eq!(probe.max_inputs.load(Ordering::SeqCst), 4);
    assert_eq!(probe.max_step_tokens.load(Ordering::SeqCst), 4 * 405);
}

/// The tokens of `run_waves`, then of the first prompt again (restored from RAM), with a private
/// snapshot pool; the pool counters and the tokens the last request took from the cache.
fn pooled_run(
    spares: usize,
) -> (
    Vec<GenEvent>,
    tetraeder_model::snapshot_pool::PoolStats,
    u32,
) {
    use tetraeder_model::snapshot_pool::{PoolConfig, SnapshotPool};
    let pool = Arc::new(SnapshotPool::new(PoolConfig {
        spares,
        thp: false,
        min_bytes: 1,
        round: 64,
    }));
    let kv = FakeKv::new(N_VOCAB, 4, 3200, 300);
    let mut w = worker_cache(kv, 2048, PrefixCache::with_pool(1 << 20, Arc::clone(&pool)));
    run_waves(&mut w, 2, 4);
    assert!(!w.cache.is_empty(), "the evicted sequences are in RAM");
    let before = pool.stats().recycled;
    let mut rx = submit(&mut w, &chat(0), 10);
    pump(&mut w);
    let evs = events(&mut rx);
    let cached = evs
        .iter()
        .find_map(|e| match e {
            GenEvent::Finished { stats, .. } => Some(stats.cached_tokens),
            _ => None,
        })
        .unwrap();
    // Released, unless the preparer had already filled every spare. The preparer runs on its own
    // thread, so wait for it to settle (CI saw the check race it).
    let settled = |s: &tetraeder_model::snapshot_pool::PoolStats| {
        spares == 0 || s.recycled > before || s.ready == spares
    };
    let t0 = std::time::Instant::now();
    let mut s = pool.stats();
    while !settled(&s) && t0.elapsed() < std::time::Duration::from_secs(2) {
        std::thread::sleep(std::time::Duration::from_millis(5));
        s = pool.stats();
    }
    assert!(settled(&s), "{s:?}");
    let toks = evs
        .into_iter()
        .filter(|e| matches!(e, GenEvent::Token(_)))
        .collect();
    (toks, s, cached)
}

/// A restored snapshot's buffer goes back to the pool, and what a request generates does not
/// depend on the pool (ADR 0074).
#[test]
fn the_snapshot_pool_takes_restored_buffers_back_and_changes_no_output() {
    let (off, s0, c0) = pooled_run(0);
    let (on, s2, c2) = pooled_run(2);
    assert!(
        c0 > 3 && c2 > 3,
        "the prompt came back from RAM ({c0}, {c2})"
    );
    assert_eq!((on, c2), (off, c0), "the pool changed the result");
    assert_eq!((s0.recycled, s0.ready), (0, 0));
    assert!(s2.ready > 0, "{s2:?}");
}

/// A request that may generate 32000 tokens hints the snapshot pool for the prompt plus a
/// stretch of generation, not for the whole context.
#[test]
fn the_size_hint_does_not_follow_a_huge_max_tokens() {
    use tetraeder_model::snapshot_pool::{PoolConfig, SnapshotPool};
    let pool = Arc::new(SnapshotPool::new(PoolConfig {
        spares: 2,
        thp: false,
        min_bytes: 1,
        round: 64,
    }));
    let kv = FakeKv::new(N_VOCAB, 2, 3200, 4000);
    let mut w = worker_cache(kv, 2048, PrefixCache::with_pool(1 << 20, Arc::clone(&pool)));
    let _rx = submit(&mut w, &prompt(3, 100), 32_000);
    w.admit();
    let row = w.backend.kv_row_bytes();
    let target = pool.target();
    assert!(target >= 100 * row, "the prompt is hinted ({target})");
    assert!(
        target <= (100 + EXPECT_AHEAD) * row + 64,
        "hinted {target} bytes, more than prompt + {EXPECT_AHEAD} rows"
    );
}

/// The finish reason and generated tokens of one request's events; panics on an error.
fn finished(evs: &[GenEvent]) -> (FinishReason, usize) {
    match evs.last() {
        Some(GenEvent::Finished { reason, stats }) => (*reason, stats.generated_tokens as usize),
        other => panic!("{other:?}"),
    }
}

/// Session 20261117l: a Claude Code conversation in a 49152-row pool laid out as one run per
/// sequence (ADR 0067). A side request lands after the conversation's run, the conversation
/// moves behind it, and its next prompt of 33586 tokens cannot grow the run where it lies.
/// Alone, with most of the pool free, it must still be served.
#[test]
fn a_prompt_that_fits_the_pool_is_served_when_its_own_run_cannot_grow() {
    let kv = FakeKv::new(N_VOCAB, 4, 49_152, 49_152).with_chunked();
    let mut w = worker_with(kv, 2048);
    for (seed, n) in [(1, 12_000), (2, 4_000), (1, 20_000)] {
        let mut rx = submit(&mut w, &prompt(seed, n), 1);
        pump(&mut w);
        assert_eq!(finished(&events(&mut rx)).1, 1);
    }
    let mut rx = submit(&mut w, &prompt(1, 33_586), 1100);
    pump(&mut w);
    assert_eq!(finished(&events(&mut rx)), (FinishReason::Length, 1100));
}

/// The same layout while generating: a lone sequence whose run cannot grow in place is moved
/// (slid down on the device, or saved, freed and placed again by a layout that cannot) instead
/// of ending early with `Length`.
#[test]
fn a_lone_sequence_whose_run_cannot_grow_keeps_generating() {
    let kv = FakeKv::new(N_VOCAB, 4, 8192, 8192).with_chunked();
    let mut w = worker_with(kv, 2048);
    for (seed, n) in [(1, 1000), (2, 1000), (1, 2500)] {
        let mut rx = submit(&mut w, &prompt(seed, n), 1);
        pump(&mut w);
        assert_eq!(finished(&events(&mut rx)).1, 1);
    }
    let mut rx = submit(&mut w, &prompt(1, 5000), 3000);
    pump(&mut w);
    assert_eq!(finished(&events(&mut rx)), (FinishReason::Length, 3000));
}

/// The slot of `w` holding a live request, with its position count.
fn live_slot(w: &Worker) -> Option<(usize, usize)> {
    w.slots.iter().enumerate().find_map(|(i, s)| match s {
        Slot::Live(s) => Some((i, s.fed)),
        _ => None,
    })
}

/// ttft4-20261119 #42 and #102: a new conversation sharing a short prefix with the idle one
/// went to the free slot, whose rows only the idle sequence's eviction could give. Admission
/// evicted that sequence (saving it), the copy from it failed, and the request restored it
/// from RAM again. It now reuses the idle sequence in place: one save, no restore.
#[test]
fn a_free_slot_copy_never_evicts_its_own_source() {
    let kv = FakeKv::new(N_VOCAB, 2, 8000, 1000);
    let probe = kv.probe();
    let mut w = worker_with(kv, 2048);
    let first = prompt(1, 600);
    let mut a = submit(&mut w, &first, 1);
    pump(&mut w);
    assert_eq!(tokens(&events(&mut a)), 1);
    let mut other = first[..100].to_vec();
    other.extend(prompt(9, 500));
    let _b = submit(&mut w, &other, 1);
    w.admit();
    let rec = w.last_admit.clone().expect("admitted");
    assert_eq!(
        (rec.source, rec.common, rec.seq),
        ("live", 100, 0),
        "{rec:?}"
    );
    assert_eq!(live_slot(&w), Some((0, 100)));
    assert!(probe.seq_cps.lock().unwrap().is_empty());
    assert_eq!(probe.snapshots.load(Ordering::SeqCst), 1);
    assert_eq!(w.cache.len(), 1, "the saved sequence stays in RAM");
    assert_eq!(rec.saved, vec![(0, 600)]);
    assert_eq!(rec.attempts, 2);
}

/// ttft4-20261119 #24 and #53: a continuation whose prompt fits where its idle sequence lies,
/// but not with the stretch of generation admission reserves too, evicted (saved) the other
/// conversation first. Admission now takes the prompt alone before evicting anything.
#[test]
fn admission_shrinks_its_reservation_before_evicting_an_idle_sequence() {
    let kv = FakeKv::new(N_VOCAB, 2, 8000, 3000);
    let probe = kv.probe();
    let mut w = worker_with(kv, 2048);
    for seed in [1, 2] {
        let mut rx = submit(&mut w, &prompt(seed, 1000), 1);
        pump(&mut w);
        assert_eq!(tokens(&events(&mut rx)), 1);
    }
    // 1000 more rows than the idle sequence holds fit; 1000 + GROW do not.
    let _c = submit(&mut w, &prompt(1, 1500), 2000);
    w.admit();
    assert_eq!(
        probe.snapshots.load(Ordering::SeqCst),
        0,
        "a save at admission"
    );
    assert!(
        matches!(w.slots[1], Slot::Idle(_)),
        "the other conversation stays"
    );
    let rec = w.last_admit.clone().expect("admitted");
    assert_eq!((rec.seq, rec.common, rec.source), (0, 1000, "live"));
}

/// ttft4-20261119 #39: the conversation's run sat after another one, its next prompt grew past
/// the pool's end, and the other conversation was evicted. Alone, its run still could not grow
/// in place, so admission saved it to RAM and restored it. The run now slides down on the
/// device: only the eviction saves.
#[test]
fn a_run_that_cannot_grow_in_place_slides_instead_of_a_round_trip_through_ram() {
    let kv = FakeKv::new(N_VOCAB, 2, 8192, 4096).with_chunked();
    let probe = kv.probe();
    let mut w = worker_with(kv, 2048);
    // A at rows 0.., B after it; A's next prompt moves it behind B, to the pool's end.
    for (seed, n) in [(1, 1000), (2, 1000), (1, 1800)] {
        let mut rx = submit(&mut w, &prompt(seed, n), 1);
        pump(&mut w);
        assert_eq!(tokens(&events(&mut rx)), 1);
    }
    assert_eq!(probe.snapshots.load(Ordering::SeqCst), 0);
    let mut rx = submit(&mut w, &prompt(1, 2600), 1);
    w.admit();
    let rec = w.last_admit.clone().expect("admitted");
    assert_eq!(
        (rec.seq, rec.source, rec.common),
        (0, "live", 1800),
        "{rec:?}"
    );
    assert_eq!(
        rec.saved,
        vec![(1, 1000)],
        "only the other conversation is saved"
    );
    assert_eq!(probe.snapshots.load(Ordering::SeqCst), 1);
    pump(&mut w);
    assert_eq!(tokens(&events(&mut rx)), 1);
}

/// A request that goes back to the queue between placement attempts reports, on its admission,
/// the saves every attempt made for it, not only the last one's (ttft4-20261119 logged them as
/// queueing).
#[test]
fn the_admission_record_counts_every_attempt() {
    let kv = FakeKv::new(N_VOCAB, 3, 8000, 1200);
    let mut w = worker_with(kv, 2048);
    let mut a = submit(&mut w, &prompt(1, 400), 1);
    pump(&mut w);
    assert_eq!(tokens(&events(&mut a)), 1);
    // A long generation holds 700 rows; the idle sequence 401.
    let _l = submit(&mut w, &prompt(2, 400), 300);
    for _ in 0..3 {
        w.admit();
        w.step();
    }
    // 601 rows: evicting the idle sequence is not enough, so the request waits for the live one.
    let _c = submit(&mut w, &prompt(3, 600), 1);
    w.admit();
    assert_eq!(w.waiting.len(), 1, "it waits");
    pump(&mut w);
    let rec = w.last_admit.clone().expect("admitted");
    assert_eq!(rec.prompt, 600);
    assert!(rec.attempts > 1, "{rec:?}");
    assert_eq!(rec.saved.first(), Some(&(0, 400)), "{rec:?}");
    assert_eq!(rec.saved.len(), 2, "{rec:?}");
}
