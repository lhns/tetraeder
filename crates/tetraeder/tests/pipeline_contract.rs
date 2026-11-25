//! `PipelineBackend` against fake segments (ADR 0091, step 4): how the contract of a
//! `BatchBackend` holds across segments, with no model and no transport. Rows: segment 0 embeds a
//! token as the float of its id, every segment adds the number of its layers, the head turns the
//! row into logits `x + j`.
//!
//! Hardware label: none.

use std::ops::Range;
use std::sync::{Arc, Mutex};
use tetraeder::pipeline::PipelineBackend;
use tetraeder_scheduler::{
    BackendError, BatchBackend, Feed, Outputs, SegmentBackend, SegmentInfo, SeqId, SeqInput,
    StepOut,
};

const N_EMBD: usize = 4;
const N_VOCAB: usize = 6;
const CAP: usize = 64;
/// Bytes of one KV row of one layer in a fake snapshot.
const ROW_BYTES: usize = 3;

#[derive(Default)]
struct State {
    lens: Vec<usize>,
    reserved: Vec<usize>,
    /// Calls the test can look at, as `rm 0 3`.
    log: Vec<String>,
    /// The next `step` fails with this error.
    fail_step: Option<BackendError>,
    /// `seq_rm` fails.
    fail_rm: bool,
    /// Rows this segment holds at most.
    cap: usize,
}

struct Fake {
    info: SegmentInfo,
    s: Arc<Mutex<State>>,
    out: Vec<f32>,
    spans: Vec<Range<usize>>,
}

fn fake(
    embed: bool,
    head: bool,
    layers: (u32, u32),
    cap: usize,
) -> (Box<dyn SegmentBackend>, Arc<Mutex<State>>) {
    let s = Arc::new(Mutex::new(State {
        lens: vec![0; 2],
        reserved: vec![0; 2],
        cap,
        ..State::default()
    }));
    let f = Fake {
        info: SegmentInfo {
            embed,
            head,
            n_embd: N_EMBD,
            layers,
        },
        s: Arc::clone(&s),
        out: Vec::new(),
        spans: Vec::new(),
    };
    (Box::new(f), s)
}

impl BatchBackend for Fake {
    fn n_vocab(&self) -> usize {
        N_VOCAB
    }
    fn n_seq_max(&self) -> usize {
        2
    }
    fn n_ctx_seq(&self) -> usize {
        self.s.lock().unwrap().cap
    }
    fn kv_capacity(&self) -> usize {
        self.s.lock().unwrap().cap
    }
    fn kv_free(&self) -> usize {
        let s = self.s.lock().unwrap();
        s.cap - s.reserved.iter().sum::<usize>()
    }
    fn seq_len(&self, seq: SeqId) -> usize {
        self.s.lock().unwrap().lens[seq as usize]
    }
    fn seq_reserved(&self, seq: SeqId) -> usize {
        self.s.lock().unwrap().reserved[seq as usize]
    }
    fn seq_reserve(&mut self, seq: SeqId, n: usize) -> Result<(), BackendError> {
        let mut s = self.s.lock().unwrap();
        s.log.push(format!("reserve {seq} {n}"));
        let have = s.reserved[seq as usize];
        let free = s.cap - s.reserved.iter().sum::<usize>();
        if n > have && n - have > free {
            return Err(BackendError::KvFull {
                seq,
                needed: n - have,
                free,
            });
        }
        s.reserved[seq as usize] = have.max(n);
        Ok(())
    }
    fn seq_rm(&mut self, seq: SeqId, keep: usize) -> Result<(), BackendError> {
        let mut s = self.s.lock().unwrap();
        s.log.push(format!("rm {seq} {keep}"));
        if s.fail_rm {
            return Err(BackendError::Failed("cannot cut".into()));
        }
        let l = s.lens[seq as usize].min(keep);
        s.lens[seq as usize] = l;
        s.reserved[seq as usize] = if keep == 0 {
            0
        } else {
            s.reserved[seq as usize].min(l)
        };
        Ok(())
    }
    fn seq_cp(&mut self, src: SeqId, dst: SeqId, n: usize) -> Result<(), BackendError> {
        let mut s = self.s.lock().unwrap();
        s.log.push(format!("cp {src} {dst} {n}"));
        s.lens[dst as usize] = n;
        Ok(())
    }
    fn decode(&mut self, _: &[SeqInput<'_>]) -> Result<Outputs<'_>, BackendError> {
        Err(BackendError::Failed("a segment is stepped".into()))
    }
    fn seq_snapshot(&mut self, seq: SeqId) -> Result<Vec<u8>, BackendError> {
        let n = self.seq_len(seq);
        let layers = (self.info.layers.1 - self.info.layers.0) as usize;
        Ok(vec![self.info.layers.0 as u8; n * layers * ROW_BYTES])
    }
    fn seq_restore(&mut self, seq: SeqId, bytes: &[u8], n: usize) -> Result<(), BackendError> {
        let layers = (self.info.layers.1 - self.info.layers.0) as usize;
        let mut s = self.s.lock().unwrap();
        s.log.push(format!("restore {seq} {n} {}", bytes.len()));
        assert_eq!(bytes.len(), n * layers * ROW_BYTES);
        assert!(bytes.iter().all(|&b| b == self.info.layers.0 as u8));
        s.lens[seq as usize] = n;
        Ok(())
    }
    fn on_gpu(&self) -> bool {
        false
    }
    fn bytes_streamed_total(&self) -> Option<u64> {
        None
    }
    fn kv_row_bytes(&self) -> usize {
        (self.info.layers.1 - self.info.layers.0) as usize * ROW_BYTES
    }
}

impl SegmentBackend for Fake {
    fn segment(&self) -> SegmentInfo {
        self.info
    }

    fn step(
        &mut self,
        inputs: &[SeqInput<'_>],
        feed: Feed<'_>,
    ) -> Result<StepOut<'_>, BackendError> {
        let mut s = self.s.lock().unwrap();
        s.log.push(format!("step {}", inputs.len()));
        if let Some(e) = s.fail_step.take() {
            return Err(e);
        }
        for i in inputs {
            let after = s.lens[i.seq as usize] + i.tokens.len();
            if after > s.cap {
                return Err(BackendError::KvFull {
                    seq: i.seq,
                    needed: after,
                    free: s.cap - s.lens.iter().sum::<usize>(),
                });
            }
        }
        let n: usize = inputs.iter().map(|i| i.tokens.len()).sum();
        let mut x: Vec<f32> = match (feed, self.info.embed) {
            (Feed::Tokens, true) => inputs
                .iter()
                .flat_map(|i| i.tokens.iter().map(|&t| t as f32))
                .collect(),
            (Feed::Hidden(h), false) => {
                assert_eq!(h.len(), n * N_EMBD, "hidden rows for every token");
                h.chunks(N_EMBD).map(|r| r[0]).collect()
            }
            _ => return Err(BackendError::Failed("wrong feed".into())),
        };
        let add = (self.info.layers.1 - self.info.layers.0) as f32;
        for v in &mut x {
            *v += add;
        }
        for i in inputs {
            s.lens[i.seq as usize] += i.tokens.len();
            let l = s.lens[i.seq as usize];
            s.reserved[i.seq as usize] = s.reserved[i.seq as usize].max(l);
        }
        self.spans.clear();
        self.out.clear();
        if self.info.head {
            let mut row = 0;
            let mut at = 0;
            for i in inputs {
                let k = if i.all_logits { i.tokens.len() } else { 1 };
                for r in (i.tokens.len() - k)..i.tokens.len() {
                    let v = x[at + r];
                    self.out.extend((0..N_VOCAB).map(|j| v + j as f32));
                }
                self.spans.push(row..row + k);
                row += k;
                at += i.tokens.len();
            }
            drop(s);
            Ok(StepOut::Logits(Outputs::new(
                N_VOCAB,
                &self.out,
                &self.spans,
            )))
        } else {
            let mut row = 0;
            for i in inputs {
                self.spans.push(row..row + i.tokens.len());
                row += i.tokens.len();
            }
            for v in &x {
                self.out.extend(std::iter::repeat_n(*v, N_EMBD));
            }
            drop(s);
            Ok(StepOut::Hidden(Outputs::new(
                N_EMBD,
                &self.out,
                &self.spans,
            )))
        }
    }
}

type Seg = (Box<dyn SegmentBackend>, Arc<Mutex<State>>);

/// Embed and layers 0..2, layers 2..5, layers 5..8 with the head; the middle one with `cap_mid`.
fn three(cap_mid: usize) -> (PipelineBackend, Vec<Arc<Mutex<State>>>) {
    let segs: Vec<Seg> = vec![
        fake(true, false, (0, 2), CAP),
        fake(false, false, (2, 5), cap_mid),
        fake(false, true, (5, 8), CAP),
    ];
    let states = segs.iter().map(|s| Arc::clone(&s.1)).collect();
    let p = PipelineBackend::new(segs.into_iter().map(|s| s.0).collect()).unwrap();
    (p, states)
}

fn input(seq: u32, tokens: &[i32], all: bool) -> SeqInput<'_> {
    SeqInput {
        seq,
        tokens,
        all_logits: all,
    }
}

// hardware: none
#[test]
fn the_rows_go_through_every_segment_and_come_out_as_logits() {
    let (mut p, _) = three(CAP);
    let a = [3, 4, 5];
    let b = [1];
    let out = p
        .decode(&[input(0, &a, true), input(1, &b, false)])
        .unwrap();
    assert_eq!((out.len(), out.n_vocab()), (2, N_VOCAB));
    // Three layer groups add 2 + 3 + 3 = 8 to the token id; logit j adds j.
    let rows = out.rows(0);
    assert_eq!(rows.len(), 3 * N_VOCAB);
    for (k, t) in a.iter().enumerate() {
        assert_eq!(rows[k * N_VOCAB + 2], (*t + 8 + 2) as f32);
    }
    assert_eq!(out.rows(1), &[9., 10., 11., 12., 13., 14.]);
    assert_eq!((p.seq_len(0), p.seq_len(1)), (3, 1));
    // A decode with one input and the last row only.
    let out = p.decode(&[input(0, &[7], false)]).unwrap();
    assert_eq!(out.rows(0).len(), N_VOCAB);
    assert_eq!(p.seq_len(0), 4);
}

// hardware: none
#[test]
fn a_cut_that_does_not_fit_is_refused() {
    let new = |segs: Vec<Seg>| PipelineBackend::new(segs.into_iter().map(|s| s.0).collect());
    let msg = |r: Result<PipelineBackend, String>| r.err().expect("refused");
    assert!(msg(new(vec![])).contains("at least one"));
    assert!(
        msg(new(vec![
            fake(false, false, (0, 2), CAP),
            fake(false, true, (2, 4), CAP)
        ]))
        .contains("embedding")
    );
    assert!(
        msg(new(vec![
            fake(true, false, (0, 2), CAP),
            fake(false, true, (3, 4), CAP)
        ]))
        .contains("do not meet")
    );
    assert!(
        msg(new(vec![
            fake(true, false, (0, 2), CAP),
            fake(true, false, (2, 3), CAP),
            fake(false, true, (3, 4), CAP)
        ]))
        .contains("hold the embedding")
    );
    // Embedding only, head only: no KV in them.
    assert!(
        new(vec![
            fake(true, false, (0, 0), CAP),
            fake(false, false, (0, 4), CAP),
            fake(false, true, (4, 4), CAP)
        ])
        .is_ok()
    );
}

// hardware: none
#[test]
fn a_failure_in_a_later_segment_cuts_the_earlier_ones_back() {
    let (mut p, st) = three(CAP);
    p.decode(&[input(0, &[1, 2, 3], false)]).unwrap();
    st[2].lock().unwrap().fail_step = Some(BackendError::Failed("boom".into()));
    let e = p.decode(&[input(0, &[4], false)]).err().unwrap();
    assert_eq!(e, BackendError::Failed("boom".into()));
    // Segments 0 and 1 advanced to 4 and were cut back to 3; segment 2 failed and is untouched
    // by the pipeline (its own contract).
    let lens: Vec<usize> = st.iter().map(|s| s.lock().unwrap().lens[0]).collect();
    assert_eq!(lens, [3, 3, 3]);
    assert!(st[0].lock().unwrap().log.contains(&"rm 0 3".to_string()));
    assert!(st[1].lock().unwrap().log.contains(&"rm 0 3".to_string()));
    assert_eq!(p.seq_len(0), 3);
    // The sequence still works.
    assert_eq!(p.decode(&[input(0, &[4], false)]).unwrap().len(), 1);
    assert_eq!(p.seq_len(0), 4);
}

// hardware: none
#[test]
fn a_segment_that_cannot_cut_back_marks_the_sequence_lost() {
    let (mut p, st) = three(CAP);
    p.decode(&[input(0, &[1, 2], false)]).unwrap();
    st[2].lock().unwrap().fail_step = Some(BackendError::Failed("boom".into()));
    st[1].lock().unwrap().fail_rm = true;
    assert!(p.decode(&[input(0, &[4], false)]).is_err());
    assert!(p.seq_lost(0));
    assert!(!p.seq_lost(1));
    let e = p.decode(&[input(0, &[5], false)]).err().unwrap();
    assert!(e.to_string().contains("lost"), "{e}");
    // Freeing the sequence clears the mark once the segment can cut again.
    st[1].lock().unwrap().fail_rm = false;
    p.seq_rm(0, 0).unwrap();
    assert!(!p.seq_lost(0));
    assert_eq!(p.decode(&[input(0, &[5], false)]).unwrap().len(), 1);
}

// hardware: none
#[test]
fn a_full_segment_fails_the_reservation_and_the_others_give_it_back() {
    let (mut p, st) = three(8);
    // The head segment has 64 rows, the middle one 8: the pipeline's free rows are the least.
    assert_eq!((p.kv_capacity(), p.kv_free(), p.n_ctx_seq()), (8, 8, 8));
    let e = p.seq_reserve(0, 20).unwrap_err();
    assert!(matches!(e, BackendError::KvFull { seq: 0, .. }), "{e}");
    // The first segment reserved 20 and gave it back when the second refused.
    assert_eq!(st[0].lock().unwrap().reserved[0], 0);
    assert_eq!(p.seq_reserved(0), 0);
    p.seq_reserve(0, 6).unwrap();
    assert_eq!(p.seq_reserved(0), 6);
    // A decode that outgrows the middle segment fails there with its KvFull and leaves the
    // first at the old length.
    let long: Vec<i32> = (0..10).collect();
    let e = p.decode(&[input(0, &long, false)]).err().unwrap();
    assert!(matches!(e, BackendError::KvFull { seq: 0, .. }), "{e}");
    assert_eq!(st[0].lock().unwrap().lens[0], 0);
    assert_eq!(p.seq_len(0), 0);
}

// hardware: none
#[test]
fn a_head_only_segment_holds_no_kv_and_does_not_count_for_free_rows() {
    let segs = vec![
        fake(true, false, (0, 0), 1),
        fake(false, false, (0, 4), 32),
        fake(false, true, (4, 4), 1),
    ];
    let mut p = PipelineBackend::new(segs.into_iter().map(|s| s.0).collect()).unwrap();
    assert_eq!((p.kv_capacity(), p.kv_free(), p.n_ctx_seq()), (32, 32, 32));
    assert_eq!(p.kv_row_bytes(), 4 * ROW_BYTES);
    p.decode(&[input(0, &[1], false)]).unwrap();
}

// hardware: none
#[test]
fn a_snapshot_is_the_segments_in_order_and_restores_by_row_bytes() {
    let (mut p, st) = three(CAP);
    p.decode(&[input(0, &[1, 2, 3, 4], false)]).unwrap();
    let snap = p.seq_snapshot(0).unwrap();
    // 2, 3 and 3 layers of 3 bytes for 4 positions, each filled with its first layer.
    assert_eq!(snap.len(), 4 * 8 * ROW_BYTES);
    assert_eq!(snap[0], 0);
    assert_eq!(snap[4 * 2 * ROW_BYTES], 2);
    assert_eq!(snap[4 * 5 * ROW_BYTES], 5);
    p.seq_restore(1, &snap, 4).unwrap();
    assert_eq!(p.seq_len(1), 4);
    let restores: Vec<String> = st[1]
        .lock()
        .unwrap()
        .log
        .iter()
        .filter(|l| l.starts_with("restore"))
        .cloned()
        .collect();
    assert_eq!(restores, ["restore 1 4 36"]);
    let e = p.seq_restore(1, &snap[1..], 4).unwrap_err();
    assert!(e.to_string().contains("cut the same way"), "{e}");
}

// hardware: none
#[test]
fn the_kv_format_carries_the_cut_points_when_two_segments_hold_layers() {
    let (p, _) = three(CAP);
    assert_eq!(p.kv_format(), "f16/f16@0-2,2-5,5-8");
    // Embedding-only and head-only segments hold no KV: one layer-holding segment is the
    // one-node layout, with the one-node key.
    let segs = vec![
        fake(true, false, (0, 0), 1),
        fake(false, false, (0, 4), 32),
        fake(false, true, (4, 4), 1),
    ];
    let p = PipelineBackend::new(segs.into_iter().map(|s| s.0).collect()).unwrap();
    assert_eq!(p.kv_format(), "f16/f16");
}

// hardware: none
#[test]
fn copies_and_cuts_go_to_every_segment() {
    let (mut p, st) = three(CAP);
    p.decode(&[input(0, &[1, 2, 3], false)]).unwrap();
    p.seq_cp(0, 1, 2).unwrap();
    assert_eq!(p.seq_len(1), 2);
    p.seq_rm(0, 1).unwrap();
    assert_eq!((p.seq_len(0), p.seq_len(1)), (1, 2));
    for s in &st {
        assert!(s.lock().unwrap().log.contains(&"cp 0 1 2".to_string()));
    }
}

/// What a replica's pipeline reports once it is built through `replica_pipeline` (ADR 0091, step
/// 5; step 4 left `with_metrics` and `set_predicted` without a caller, so the segment panel and
/// `TetraederSegmentDrift` stayed dark): one decode records `tetraeder.segment.duration` and the
/// residual for every segment, and the planner's step time as the baseline.
// hardware: none
#[test]
fn a_decode_records_the_segment_metrics_when_built_for_a_replica() {
    use opentelemetry::metrics::MeterProvider as _;
    use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
    use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};
    use tetraeder::pipeline::replica_pipeline;

    let exp = InMemoryMetricExporter::default();
    let mp = SdkMeterProvider::builder()
        .with_reader(PeriodicReader::builder(exp.clone()).build())
        .build();
    let metrics = tetraeder_telemetry::Metrics::new(&mp.meter("test"));
    let segs: Vec<Box<dyn SegmentBackend>> = vec![
        fake(true, false, (0, 2), CAP).0,
        fake(false, false, (2, 5), CAP).0,
        fake(false, true, (5, 8), CAP).0,
    ];
    let mut p = replica_pipeline(segs, Some(metrics), &[0.01, 0.02, 0.03]).unwrap();
    // A prompt, then one decode step of one sequence: the step the residual is measured on.
    p.decode(&[input(0, &[1, 2, 3], false)]).unwrap();
    p.decode(&[input(0, &[4], false)]).unwrap();
    mp.force_flush().unwrap();
    let (mut duration, mut residual, mut predicted) = (Vec::new(), Vec::new(), Vec::new());
    for rm in exp.get_finished_metrics().unwrap().iter().rev().take(1) {
        for sm in rm.scope_metrics() {
            for m in sm.metrics() {
                let segments = |dps: Vec<String>| {
                    let mut d = dps;
                    d.sort();
                    d.dedup();
                    d
                };
                match (m.name(), m.data()) {
                    (
                        "tetraeder.segment.duration",
                        AggregatedMetrics::F64(MetricData::Histogram(h)),
                    ) => {
                        duration = segments(
                            h.data_points()
                                .map(|dp| format!("{:?}", dp.attributes().collect::<Vec<_>>()))
                                .collect(),
                        );
                    }
                    (
                        "tetraeder.plan.residual",
                        AggregatedMetrics::F64(MetricData::Histogram(h)),
                    ) => {
                        residual = segments(
                            h.data_points()
                                .map(|dp| format!("{:?}", dp.attributes().collect::<Vec<_>>()))
                                .collect(),
                        );
                    }
                    (
                        "tetraeder.plan.predicted_step",
                        AggregatedMetrics::F64(MetricData::Gauge(g)),
                    ) => {
                        predicted = segments(
                            g.data_points()
                                .map(|dp| format!("{:?}", dp.attributes().collect::<Vec<_>>()))
                                .collect(),
                        );
                    }
                    _ => {}
                }
            }
        }
    }
    assert_eq!(duration.len(), 3, "a duration per segment: {duration:?}");
    assert_eq!(residual.len(), 3, "a residual per segment: {residual:?}");
    assert_eq!(
        predicted.len(),
        3,
        "a prediction per segment: {predicted:?}"
    );
}

/// The rows of one decode of `tokens` on sequence 0 of a fresh three-segment pipeline cut into
/// `micro` micro-batches.
fn prefill_rows(micro: u32, tokens: &[i32], all: bool) -> (Vec<f32>, usize) {
    let (p, _) = three(CAP);
    let mut p = p.with_prefill_micro_batches(micro);
    let out = p.decode(&[input(0, tokens, all)]).unwrap();
    let rows = out.rows(0).to_vec();
    (rows, p.seq_len(0))
}

// hardware: none
#[test]
fn a_pipelined_prefill_returns_the_rows_of_the_walk() {
    let tokens: Vec<i32> = (1..=11).collect();
    for all in [false, true] {
        let (walk, n) = prefill_rows(1, &tokens, all);
        assert_eq!(n, tokens.len());
        for micro in [2, 3, 4, 20] {
            let (piped, n) = prefill_rows(micro, &tokens, all);
            assert_eq!(piped, walk, "micro {micro} all_logits {all}");
            assert_eq!(n, tokens.len(), "micro {micro}: KV rows of every segment");
        }
    }
}

fn steps_of(s: &Arc<Mutex<State>>) -> usize {
    let st = s.lock().unwrap();
    st.log.iter().filter(|l| l.starts_with("step")).count()
}

// hardware: none
#[test]
fn a_pipelined_prefill_steps_every_segment_once_per_micro_batch() {
    let (p, states) = three(CAP);
    let mut p = p.with_prefill_micro_batches(3);
    let tokens: Vec<i32> = (1..=7).collect();
    p.decode(&[input(0, &tokens, false)]).unwrap();
    // `plan::micro_batch_sizes(7, 3)` is 3, 2, 2: three steps on every segment.
    for s in &states {
        assert_eq!(steps_of(s), 3);
        assert_eq!(s.lock().unwrap().lens[0], 7);
    }
    // A decode step and a step of several inputs walk the segments once.
    p.decode(&[input(0, &[8], false)]).unwrap();
    p.decode(&[input(0, &[9, 10], false), input(1, &[1, 2, 3], false)])
        .unwrap();
    for s in &states {
        assert_eq!(steps_of(s), 5);
    }
}

// hardware: none
#[test]
fn a_failure_in_a_micro_batch_of_a_pipelined_prefill_cuts_every_segment_back() {
    let (p, states) = three(CAP);
    let mut p = p.with_prefill_micro_batches(4);
    p.decode(&[input(0, &[1, 2, 3], false)]).unwrap();
    // The middle segment fails its first micro-batch; the first segment holds more by then.
    states[1].lock().unwrap().fail_step = Some(BackendError::Failed("boom".into()));
    let tokens: Vec<i32> = (1..=12).collect();
    let e = p.decode(&[input(0, &tokens, false)]).err().unwrap();
    assert_eq!(e.to_string(), "boom");
    for (i, s) in states.iter().enumerate() {
        assert_eq!(s.lock().unwrap().lens[0], 3, "segment {i} is where it was");
    }
    assert!(!p.seq_lost(0));
    // The sequence still works, pipelined.
    p.decode(&[input(0, &tokens, false)]).unwrap();
    assert_eq!(p.seq_len(0), 15);
}

// hardware: none
#[test]
fn a_pipelined_prefill_that_outgrows_a_segment_fails_with_kv_full_and_cuts_back() {
    let (p, states) = three(10);
    let mut p = p.with_prefill_micro_batches(3);
    let tokens: Vec<i32> = (1..=12).collect();
    let e = p.decode(&[input(0, &tokens, false)]).err().unwrap();
    assert!(matches!(e, BackendError::KvFull { .. }), "{e}");
    for s in &states {
        assert_eq!(s.lock().unwrap().lens[0], 0);
    }
}

// hardware: none
#[test]
fn a_pipelined_prefill_records_the_segment_duration_of_every_micro_batch() {
    use opentelemetry::metrics::MeterProvider as _;
    use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
    use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};
    use tetraeder::pipeline::replica_pipeline;

    let exp = InMemoryMetricExporter::default();
    let mp = SdkMeterProvider::builder()
        .with_reader(PeriodicReader::builder(exp.clone()).build())
        .build();
    let metrics = tetraeder_telemetry::Metrics::new(&mp.meter("test"));
    let segs: Vec<Box<dyn SegmentBackend>> = vec![
        fake(true, false, (0, 2), CAP).0,
        fake(false, false, (2, 5), CAP).0,
        fake(false, true, (5, 8), CAP).0,
    ];
    let mut p = replica_pipeline(segs, Some(metrics), &[])
        .unwrap()
        .with_prefill_micro_batches(4);
    let tokens: Vec<i32> = (1..=10).collect();
    p.decode(&[input(0, &tokens, false)]).unwrap();
    mp.force_flush().unwrap();
    let mut counts = Vec::new();
    for rm in exp.get_finished_metrics().unwrap().iter().rev().take(1) {
        for sm in rm.scope_metrics() {
            for m in sm.metrics() {
                if let (
                    "tetraeder.segment.duration",
                    AggregatedMetrics::F64(MetricData::Histogram(h)),
                ) = (m.name(), m.data())
                {
                    counts = h.data_points().map(|dp| dp.count()).collect();
                }
            }
        }
    }
    assert_eq!(counts, vec![4, 4, 4], "a sample per micro-batch and segment");
}
