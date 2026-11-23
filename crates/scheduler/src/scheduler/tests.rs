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
        ..SampleParams::default()
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

/// A save made inside a step (here a preempt) is counted apart from the admissions' `save_ms`,
/// so the TTFT save share can include it (ADR 0074 History).
#[test]
fn a_preempt_inside_a_step_records_its_save() {
    let kv = FakeKv::new(N_VOCAB, 2, 8000, 4000);
    let probe = kv.probe();
    let mut w = worker_with(kv, 2048);
    let _a = submit(&mut w, &prompt(1, 100), 50);
    let _b = submit(&mut w, &prompt(2, 100), 50);
    w.admit();
    w.step();
    assert_eq!(w.step_saves, 0);
    let saved0 = probe.snapshots.load(Ordering::SeqCst);
    // Slot 0 is the older: the newer one, on slot 1, is the victim.
    w.make_room(0, false);
    assert_eq!(w.step_saves, 1);
    assert_eq!(probe.snapshots.load(Ordering::SeqCst), saved0 + 1);
    assert_eq!(w.saved.last().map(|s| s.0), Some(1));
    assert_eq!(w.waiting.len(), 1, "the victim is requeued");
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

/// The smaller reservation still holds the RAM entry a request restores: an entry longer than
/// the prompt plus one would not fit it, and the restore would fall back to a cold prefill.
#[test]
fn a_shrunk_reservation_still_holds_the_restored_entry() {
    let kv = FakeKv::new(N_VOCAB, 2, 8000, 2151);
    let mut w = worker_with(kv, 2048);
    let mut b = submit(&mut w, &prompt(2, 1000), 1);
    pump(&mut w);
    assert_eq!(tokens(&events(&mut b)), 1);
    // A saved conversation of 1200 positions; the request shares 1000 of them.
    let saved = prompt(1, 1200);
    w.cache.insert(
        &w.backend.kv_format(),
        saved.iter().map(|&t| t as i32).collect(),
        saved
            .iter()
            .flat_map(|&t| (t as i32).to_le_bytes())
            .collect(),
    );
    let mut p = saved[..1000].to_vec();
    p.extend(prompt(7, 100));
    let _c = submit(&mut w, &p, 2000);
    w.admit();
    let rec = w.last_admit.clone().expect("admitted");
    assert_eq!((rec.source, rec.common), ("ram", 1000), "{rec:?}");
}

// ---- disk restore off the step (ADR 0087 addendum 2026-11-23) ----

use crate::prefix_disk::ReadFn;
use std::sync::atomic::AtomicBool;
use tetraeder_model::snapshot_pool::{PoolConfig, SnapshotPool};

/// A reader that waits for the test's gate, then reads the file; with `fail` it errors instead.
fn gated_reader(gate: &Arc<AtomicBool>, fail: bool) -> ReadFn {
    let gate = Arc::clone(gate);
    Arc::new(move |path, format, size, pool| {
        while !gate.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(1));
        }
        if fail {
            return Err(std::io::Error::other("injected read failure"));
        }
        crate::prefix_disk::read_file(path, format, size, pool)
    })
}

struct Spilled {
    w: Worker,
    pool: Arc<SnapshotPool>,
    gate: Arc<AtomicBool>,
    dir: PathBuf,
    /// The prompt of the entry that sits on disk.
    saved: Vec<TokenId>,
}

impl Spilled {
    /// Two slots; one entry of 1200 tokens spilled to disk behind a closed gate.
    fn new(tag: &str, fail: bool) -> Self {
        let dir = std::env::temp_dir().join(format!("tt-sched-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let pool = Arc::new(SnapshotPool::new(PoolConfig {
            spares: 4,
            thp: false,
            min_bytes: 1,
            round: 1,
        }));
        let gate = Arc::new(AtomicBool::new(false));
        let cache = PrefixCache::with_pool(6000, Arc::clone(&pool))
            .with_disk_reader(gated_reader(&gate, fail))
            .with_disk(&dir, 100_000);
        let mut w = worker_cache(FakeKv::new(N_VOCAB, 2, 8000, 4000), 2048, cache);
        let fmt = w.backend.kv_format();
        let mut saved = Vec::new();
        for seed in [1, 9] {
            let p = prompt(seed, 1200);
            let toks: Vec<i32> = p.iter().map(|&t| t as i32).collect();
            let bytes: Vec<u8> = toks.iter().flat_map(|t| t.to_le_bytes()).collect();
            w.cache.insert(&fmt, toks, bytes);
            if seed == 1 {
                saved = p;
            }
        }
        // The second entry does not fit beside the first: the first spills.
        w.cache.flush_disk();
        assert_eq!(w.cache.disk_bytes(), 4800);
        Self {
            w,
            pool,
            gate,
            dir,
            saved,
        }
    }

    /// A prompt that shares 1000 tokens with the spilled entry.
    fn follow_up(&self) -> Vec<TokenId> {
        let mut p = self.saved[..1000].to_vec();
        p.extend(prompt(7, 100));
        p
    }

    fn submit_follow_up(&mut self, max: u32) -> Rx {
        let p = self.follow_up();
        submit(&mut self.w, &p, max)
    }

    fn files(&self) -> usize {
        std::fs::read_dir(&self.dir).map_or(0, |r| r.count())
    }

    /// Steps the worker until `done`, without a hot spin; panics after ten seconds.
    fn run_until(&mut self, mut done: impl FnMut(&mut Worker) -> bool) {
        let t0 = Instant::now();
        while !done(&mut self.w) {
            assert!(t0.elapsed() < Duration::from_secs(10), "timed out");
            self.w.flush_drains();
            self.w.admit();
            self.w.sweep();
            if !self.w.step() {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }
}

impl Drop for Spilled {
    fn drop(&mut self) {
        self.gate.store(true, Ordering::SeqCst);
    }
}

/// While the read is slow, another running sequence keeps producing tokens and the restoring
/// request holds no slot; once the read is in it is admitted with the entry and finishes.
// hardware: none
#[test]
fn a_disk_restore_does_not_stall_a_running_sequence() {
    let mut t = Spilled::new("nostall", false);
    let mut other = submit(&mut t.w, &prompt(3, 40), 400);
    t.run_until(|w| w.live_count() == 1);
    let mut restoring = t.submit_follow_up(2);
    let mut seen = 0;
    t.run_until(|_| {
        seen += tokens(&events(&mut other));
        seen >= 25
    });
    assert!(t.w.waiting.front().is_some_and(Req::reading));
    assert_eq!((t.w.live_count(), t.w.cache.restoring()), (1, 1));
    assert!(events(&mut restoring).is_empty());
    // The file is not consumed until the read is in.
    assert_eq!(t.w.cache.disk_bytes(), 4800);
    t.gate.store(true, Ordering::SeqCst);
    t.run_until(|w| w.last_admit.as_ref().is_some_and(|r| r.source == "disk"));
    let rec = t.w.last_admit.clone().expect("admitted");
    assert_eq!((rec.source, rec.common), ("disk", 1000), "{rec:?}");
    assert_eq!((t.w.cache.disk_bytes(), t.w.cache.restoring()), (0, 0));
    pump(&mut t.w);
    assert_eq!(tokens(&events(&mut restoring)), 2);
    drop(t);
}

/// A request behind a restoring one is admitted at once; the restoring one keeps its place.
// hardware: none
#[test]
fn a_request_behind_a_restoring_one_is_admitted_first() {
    let mut t = Spilled::new("behind", false);
    let mut restoring = t.submit_follow_up(2);
    t.run_until(|w| w.cache.restoring() == 1);
    let mut behind = submit(&mut t.w, &prompt(4, 30), 3);
    t.run_until(|w| w.live_count() == 1);
    t.run_until(|w| tokens(&events(&mut behind)) == 3 || w.live_count() == 0);
    assert!(events(&mut restoring).is_empty());
    assert!(t.w.waiting.front().is_some_and(Req::reading));
    t.gate.store(true, Ordering::SeqCst);
    pump(&mut t.w);
    assert_eq!(tokens(&events(&mut restoring)), 2);
}

/// The generated tokens of a request restored from disk equal those of a RAM restore.
// hardware: none
#[test]
fn a_disk_restore_gives_the_tokens_of_a_ram_restore() {
    let mut t = Spilled::new("same", false);
    t.gate.store(true, Ordering::SeqCst);
    let mut rx = t.submit_follow_up(6);
    pump(&mut t.w);
    let disk = events(&mut rx);
    assert_eq!(t.w.last_admit.as_ref().map(|r| r.source), Some("disk"));

    // The same entry kept in RAM.
    let mut w = worker_cache(
        FakeKv::new(N_VOCAB, 2, 8000, 4000),
        2048,
        PrefixCache::new(1 << 20),
    );
    let toks: Vec<i32> = t.saved.iter().map(|&x| x as i32).collect();
    let bytes: Vec<u8> = toks.iter().flat_map(|x| x.to_le_bytes()).collect();
    w.cache.insert(&w.backend.kv_format(), toks, bytes);
    let mut rx = submit(&mut w, &t.follow_up(), 6);
    pump(&mut w);
    let ram = events(&mut rx);
    let rec = w.last_admit.clone().expect("admitted");
    assert_eq!((rec.source, rec.common), ("ram", 1000));
    let toks = |evs: &[GenEvent]| {
        evs.iter()
            .filter_map(|e| match e {
                GenEvent::Token(t) => Some(*t),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(toks(&disk).len(), 6);
    assert_eq!(toks(&disk), toks(&ram));
}

/// A read that fails admits the request without reuse, with the whole prompt prefilled.
// hardware: none
#[test]
fn a_failed_disk_read_falls_back_to_a_full_prefill() {
    let mut t = Spilled::new("fail", true);
    let mut rx = t.submit_follow_up(2);
    t.gate.store(true, Ordering::SeqCst);
    pump(&mut t.w);
    let evs = events(&mut rx);
    assert_eq!(tokens(&evs), 2, "{evs:?}");
    assert!(matches!(evs.last(), Some(GenEvent::Finished { .. })));
    let rec = t.w.last_admit.clone().expect("admitted");
    assert_eq!(rec.common, 0, "{rec:?}");
    // The unreadable entry is gone.
    assert_eq!((t.w.cache.disk_bytes(), t.files()), (0, 0));
}

/// A client that leaves during the read ends its request; the buffer goes back to the pool and
/// the file stays, so the next request restores from it.
// hardware: none
#[test]
fn cancelling_during_a_disk_read_returns_the_buffer_and_keeps_the_file() {
    let mut t = Spilled::new("cancel", false);
    let (tx, mut rx) = mpsc::channel(100);
    let cancel = CancellationToken::new();
    let params = SamplingParams {
        max_tokens: 2,
        temperature: Some(0.0),
        ..SamplingParams::default()
    };
    t.w.waiting
        .push_back(Req::new(t.follow_up(), params, None, cancel.clone(), tx));
    t.run_until(|w| w.cache.restoring() == 1);
    let files = t.files();
    let (recycled, freed) = (t.pool.stats().recycled, t.pool.stats().freed_bytes);
    cancel.cancel();
    t.run_until(|w| w.waiting.is_empty());
    // The reader still has the file open: the read counts until it replies.
    assert_eq!(t.w.cache.restoring(), 1);
    let evs = events(&mut rx);
    assert!(
        matches!(
            evs.last(),
            Some(GenEvent::Finished {
                reason: FinishReason::Cancelled,
                ..
            })
        ),
        "{evs:?}"
    );
    t.gate.store(true, Ordering::SeqCst);
    let t0 = Instant::now();
    // Kept by the pool, or freed when its spares are full.
    let back = |t: &Spilled| {
        let s = t.pool.stats();
        s.recycled > recycled || s.freed_bytes > freed
    };
    while !back(&t) && t0.elapsed() < Duration::from_secs(10) {
        t.w.cache.flush_disk();
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(back(&t), "the buffer went back");
    // The tier notices the reply at its next poll.
    while t.w.cache.restoring() > 0 && t0.elapsed() < Duration::from_secs(10) {
        t.w.cache.flush_disk();
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(t.w.cache.restoring(), 0);
    assert_eq!((t.w.cache.disk_bytes(), t.files()), (4800, files));
    // The entry is still usable.
    let mut again = t.submit_follow_up(2);
    pump(&mut t.w);
    assert_eq!(tokens(&events(&mut again)), 2);
    assert_eq!(t.w.last_admit.as_ref().map(|r| r.source), Some("disk"));
}

// ---- context shift and rerank prompts (ADR 0090) ----

#[test]
fn the_shift_range_follows_llama_servers_arithmetic() {
    // default: keep nothing, drop half
    assert_eq!(shift_range(63, 64, 10, false, None, None), (0, 31));
    // a BOS is always kept; n_keep counts after it
    assert_eq!(shift_range(63, 64, 10, true, None, None), (1, 31));
    assert_eq!(shift_range(63, 64, 10, true, Some(4), None), (5, 29));
    // negative n_keep keeps the prompt
    assert_eq!(shift_range(63, 64, 10, false, Some(-1), None), (10, 26));
    // n_keep is capped at n_ctx - 4
    assert_eq!(shift_range(63, 64, 10, false, Some(500), None).0, 60);
    // n_discard is the request's, never all of what is left
    assert_eq!(shift_range(63, 64, 10, false, Some(4), Some(8)), (4, 8));
    assert_eq!(shift_range(63, 64, 10, false, Some(4), Some(500)), (4, 58));
    // one position left, or none: nothing to drop
    assert_eq!(shift_range(5, 64, 10, false, Some(4), None).1, 0);
    assert_eq!(shift_range(3, 64, 10, false, Some(4), None).1, 0);
}

#[test]
fn the_rerank_template_is_filled_in_one_pass() {
    assert_eq!(
        fill_rerank_template("Q: {query}\nD: {document}\n", "a", "b"),
        "Q: a\nD: b\n"
    );
    // a placeholder inside the text is not expanded again
    assert_eq!(
        fill_rerank_template("{query}|{document}", "{document}", "{query}"),
        "{document}|{query}"
    );
    assert_eq!(fill_rerank_template("é{query}ü", "x", "y"), "éxü");
    assert_eq!(
        fill_rerank_template("no placeholders", "x", "y"),
        "no placeholders"
    );
}

#[test]
fn a_rerank_pair_without_a_template_is_the_query_and_the_document() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-qwen3-f16.gguf");
    let m = Model::load(
        &path,
        &tetraeder_ggml_sys::llama::ModelParams {
            vocab_only: true,
            ..Default::default()
        },
    )
    .unwrap();
    // Qwen3 adds no BOS, EOS or SEP: the pair is the two texts' tokens, one after the other
    let want: Vec<i32> = [
        m.tokenize("Hello", false, false).unwrap(),
        m.tokenize("world", false, false).unwrap(),
    ]
    .concat();
    assert_eq!(rerank_tokens(&m, "Hello", "world").unwrap(), want);
}

// ---- review 2 (ADR 0090 addendum) ----

/// A drafter that knows the fake backend's next token, so every draft is accepted.
struct Oracle;

impl Drafter for Oracle {
    fn draft(&mut self, history: &[i32], max: usize) -> Result<Vec<i32>, crate::spec::SpecError> {
        let mut h = history.to_vec();
        let mut out = Vec::new();
        for _ in 0..max {
            let l = fake::logits_for(N_VOCAB, &h);
            let t = (0..l.len()).fold(0, |b, i| if l[i] > l[b] { i } else { b }) as i32;
            h.push(t);
            out.push(t);
        }
        Ok(out)
    }
    fn method(&self) -> &'static str {
        "oracle"
    }
}

// hardware: none
#[test]
fn a_full_context_does_not_end_a_shifting_request_that_speculates() {
    let kv = FakeKv::new(N_VOCAB, 1, 16, 16).with_shift();
    let mut w = worker(kv);
    w.ctx_shift = true;
    let cfg = SpecConfig {
        method: crate::spec::SpecMethod::Ngram(crate::spec::NgramDrafter::default()),
        len: crate::spec::DraftLen::Fixed(4),
        max_batch: 1,
    };
    w.set_spec(&cfg, Box::new(Oracle));
    let mut rx = submit(&mut w, &prompt(1, 6), 60);
    pump(&mut w);
    // The drafts that fill the cache to its last row used to end the request with Length.
    assert_eq!(finished(&events(&mut rx)), (FinishReason::Length, 60));
}

// hardware: none
#[test]
fn a_cancelled_pinned_request_leaves_the_queue_while_its_slot_runs() {
    let kv = FakeKv::new(N_VOCAB, 2, 256, 512);
    let mut w = worker(kv);
    let _a = submit(&mut w, &prompt(1, 20), 100_000);
    let (mut r, mut rx) = req(&prompt(2, 10), 5);
    r.params.extra.slot = Some(0);
    let cancel = r.cancel.clone();
    w.waiting.push_back(r);
    for _ in 0..3 {
        w.admit();
        w.step();
    }
    assert!(matches!(w.slots[0], Slot::Live(_)));
    assert_eq!(w.waiting.len(), 1, "pinned behind the running request");
    cancel.cancel();
    w.admit();
    assert!(w.waiting.is_empty());
    assert!(matches!(
        events(&mut rx).last(),
        Some(GenEvent::Finished {
            reason: FinishReason::Cancelled,
            ..
        })
    ));
    assert!(
        matches!(w.slots[0], Slot::Live(_)),
        "the other request runs on"
    );
}

// hardware: none
#[test]
fn a_shifted_live_sequence_is_not_a_copy_source() {
    let kv = FakeKv::new(N_VOCAB, 2, 64, 128).with_shift();
    let probe = kv.probe();
    let mut w = worker(kv);
    w.ctx_shift = true;
    let _a = submit(&mut w, &prompt(1, 40), 100_000);
    let mut shifted = false;
    for _ in 0..200 {
        w.admit();
        w.step();
        shifted = matches!(&w.slots[0], Slot::Live(s) if s.req.shifted);
        if shifted {
            break;
        }
    }
    assert!(shifted, "the first request shifted");
    // A prompt that is the first 30 tokens the shifted sequence now holds.
    let held = match &w.slots[0] {
        Slot::Live(s) => s.req.toks[..30]
            .iter()
            .map(|&t| t as u32)
            .collect::<Vec<_>>(),
        _ => unreachable!(),
    };
    let _b = submit(&mut w, &held, 3);
    w.admit();
    assert!(probe.seq_cps.lock().unwrap().is_empty(), "no copy");
}

// hardware: none
#[test]
fn a_slot_job_whose_client_left_is_dropped_not_kept_waiting() {
    let kv = FakeKv::new(N_VOCAB, 1, 256, 256);
    let mut w = worker(kv);
    let _a = submit(&mut w, &prompt(1, 20), 100_000);
    for _ in 0..3 {
        w.admit();
        w.step();
    }
    assert!(matches!(w.slots[0], Slot::Live(_)));
    let count = Arc::new(AtomicUsize::new(0));
    let job = |count: &Arc<AtomicUsize>| {
        let (reply, rx) = tokio::sync::oneshot::channel();
        let held = SlotPermit::take(count).expect("a permit");
        (
            SlotJob {
                id: 0,
                action: SlotAction::Erase,
                reply,
                _held: held,
            },
            rx,
        )
    };
    let (gone, rx) = job(&count);
    let (waiting, _rx) = job(&count);
    drop(rx);
    w.pending_slots.push(gone);
    w.pending_slots.push(waiting);
    w.run_slot_jobs();
    assert_eq!(w.pending_slots.len(), 1, "only the one with a client waits");
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

// hardware: none
#[test]
fn slot_jobs_are_capped() {
    let count = Arc::new(AtomicUsize::new(0));
    let held: Vec<_> = (0..MAX_SLOT_JOBS)
        .map(|_| SlotPermit::take(&count).expect("under the cap"))
        .collect();
    assert!(SlotPermit::take(&count).is_none());
    drop(held);
    assert_eq!(count.load(Ordering::SeqCst), 0);
    assert!(SlotPermit::take(&count).is_some());
}

fn slot_dir(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("tetraeder-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&d).expect("temp dir");
    d
}

// hardware: none
#[test]
fn a_restore_refuses_a_directory_and_a_file_larger_than_a_slot() {
    let dir = slot_dir("restore-guards");
    let mut w = worker(FakeKv::new(N_VOCAB, 1, 16, 16));
    w.slot_dir = Some(dir.clone());
    std::fs::create_dir_all(dir.join("sub")).unwrap();
    let restore = |name: &str| SlotAction::Restore {
        filename: name.into(),
    };
    let err = |w: &mut Worker, name: &str| match w.slot_action(0, &restore(name)) {
        Err(EngineError::Invalid(m)) => m,
        other => panic!("{other:?}"),
    };
    assert!(err(&mut w, "sub").contains("not a regular file"));
    // The fake's rows are 4 bytes, so a slot of 16 holds far less than 2 MiB.
    std::fs::write(dir.join("big.bin"), vec![0u8; 2 << 20]).unwrap();
    assert!(err(&mut w, "big.bin").contains("more than a slot holds"));
    std::fs::remove_dir_all(&dir).unwrap();
}

// hardware: none
#[test]
fn a_save_is_written_whole_under_its_name_and_leaves_no_temp_file() {
    let dir = slot_dir("save-atomic");
    let mut w = worker(FakeKv::new(N_VOCAB, 1, 64, 64));
    w.slot_dir = Some(dir.clone());
    std::fs::write(dir.join("s.bin"), b"old").unwrap();
    let save = SlotAction::Save {
        filename: "s.bin".into(),
    };
    let Ok(SlotReply::Saved { bytes, .. }) = w.slot_action(0, &save) else {
        panic!("not saved")
    };
    assert_eq!(std::fs::metadata(dir.join("s.bin")).unwrap().len(), bytes);
    let names: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert_eq!(names.len(), 1, "{names:?}");
    // A save that cannot be renamed into place leaves nothing behind either.
    std::fs::remove_file(dir.join("s.bin")).unwrap();
    std::fs::create_dir(dir.join("s.bin")).unwrap();
    assert!(w.slot_action(0, &save).is_err());
    let left = std::fs::read_dir(&dir).unwrap().count();
    assert_eq!(left, 1, "only the directory stays");
    std::fs::remove_dir_all(&dir).unwrap();
}
