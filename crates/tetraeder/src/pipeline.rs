//! `PipelineBackend`: one replica across nodes as a [`BatchBackend`] (ADR 0091, step 4).
//!
//! The head process holds one backend per segment of the replica (a [`RemoteBackend`] for each
//! segment on another node, a local builder for a head-only segment on the leader) and walks
//! them in order, a star from the leader: segment 0 takes the tokens, every later one takes the
//! hidden rows the one before returned, the last returns the logits. So one decode step is one
//! round trip per remote segment.
//!
//! KV lives per segment, each in its own layers. Sequence calls go to every segment. A step that
//! fails in segment `i` leaves segment `i` as it was (the segment's own contract) and the leader
//! cuts the earlier segments back to the lengths from before the step (`seq_rm`, ADR 0065); a
//! segment that cannot be cut back marks its sequences lost.
//!
//! Snapshots are the segments' snapshots one after the other in segment order. That is not the
//! bytes of a one-node snapshot (those hold all K layers, then all V layers) and has the same
//! size as one, so a snapshot restores only into a pipeline cut at the same layers. What keeps a
//! foreign snapshot out is [`BatchBackend::kv_format`], the key of the prefix cache and of slot
//! files: it carries the cut points. Moving KV between differently cut replicas (step 5) needs
//! the K and V halves of each segment's share separated first.
//!
//! Pipelined prefill (ADR 0077, 0091 step 2 pricing): with `prefill_micro_batches` above one, a
//! step of a single input of several tokens is cut into micro-batches by
//! `plan::micro_batch_sizes`. Segment `s` takes micro-batch `k` as soon as segment `s - 1` has
//! returned it, while segment `s - 1` already works on `k + 1`; a remote segment has up to
//! `DEPTH` (2) requests queued, so the next one is on the link while the worker computes. The head
//! still forwards every hidden row itself (the star), so each boundary is two links, in and out,
//! as `predict_cluster` prices them. KV rows per segment are appended micro-batch by micro-batch
//! in order, which is what the sequential walk does with one chunk. Steps of several inputs
//! (decoders next to a chunk) and decode steps walk the segments in turn as before.

use std::collections::{HashSet, VecDeque};
use std::ops::Range;
use std::time::{Duration, Instant};
use tetraeder_scheduler::{
    BackendError, BatchBackend, Feed, Outputs, SegmentBackend, SegmentInfo, SeqId, SeqInput,
    StepOut,
};
use tetraeder_telemetry::Metrics;

/// Steps queued on one remote segment in a pipelined prefill: the one it computes and the one on
/// the link. More would only buffer rows in the worker.
const DEPTH: usize = 2;

/// The rows a segment returned for the one input of a micro-batch of `tokens` tokens, owned so
/// the segment is free again. A segment without the head must return a row per token.
fn own_rows(
    out: StepOut<'_>,
    seg: usize,
    tokens: usize,
    is_last: bool,
) -> Result<Vec<f32>, BackendError> {
    let out = out.rows();
    let width = out.n_vocab().max(1);
    if out.len() != 1 {
        return Err(BackendError::Failed(format!(
            "segment {seg} returned {} inputs for one",
            out.len()
        )));
    }
    let rows = out.rows(0);
    if !is_last && rows.len() / width != tokens {
        return Err(BackendError::Failed(format!(
            "segment {seg} returned {} rows for {tokens} tokens of 1 inputs",
            rows.len() / width
        )));
    }
    Ok(rows.to_vec())
}

/// Decode steps of one token on one sequence are what the planner predicts; only they feed the
/// residual.
fn is_decode_step(inputs: &[SeqInput<'_>]) -> bool {
    matches!(inputs, [one] if one.tokens.len() == 1)
}

/// What one segment spent in the steps it took, as the leader saw them: `own` is the segment's
/// own compute time (a remote worker reports it), `wall` the time from sending the step to its
/// reply, so `wall - own` is the hop. In a pipelined prefill steps overlap and `wall` includes
/// the time a step waited behind the one before it. Cumulative since the pipeline was built.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SegmentTimes {
    /// Steps of more than one token (prompt chunks).
    pub prefill_steps: u64,
    pub prefill_own_s: f64,
    pub prefill_wall_s: f64,
    /// Steps of one token.
    pub decode_steps: u64,
    pub decode_own_s: f64,
    pub decode_wall_s: f64,
}

impl SegmentTimes {
    /// What came in after `earlier`.
    pub fn since(&self, earlier: &Self) -> Self {
        Self {
            prefill_steps: self.prefill_steps - earlier.prefill_steps,
            prefill_own_s: self.prefill_own_s - earlier.prefill_own_s,
            prefill_wall_s: self.prefill_wall_s - earlier.prefill_wall_s,
            decode_steps: self.decode_steps - earlier.decode_steps,
            decode_own_s: self.decode_own_s - earlier.decode_own_s,
            decode_wall_s: self.decode_wall_s - earlier.decode_wall_s,
        }
    }
}

pub struct PipelineBackend {
    segs: Vec<Box<dyn SegmentBackend>>,
    infos: Vec<SegmentInfo>,
    /// The hidden rows between two segments.
    hidden: Vec<f32>,
    logits: Vec<f32>,
    spans: Vec<Range<usize>>,
    /// Sequences whose segments disagree after a failed rollback, until `seq_rm(seq, 0)`.
    lost: HashSet<SeqId>,
    metrics: Option<Metrics>,
    /// The predicted decode step of each segment, seconds.
    predicted: Vec<Option<f64>>,
    /// Micro-batches a prefill step is cut into (ADR 0077); 1 walks the segments in turn.
    micro: u32,
    times: Vec<SegmentTimes>,
}

impl PipelineBackend {
    /// `segs` in order. Refused when they do not cover the layers once, from the embedding to
    /// the head, with one hidden width.
    pub fn new(segs: Vec<Box<dyn SegmentBackend>>) -> Result<Self, String> {
        let infos: Vec<SegmentInfo> = segs.iter().map(|s| s.segment()).collect();
        let (Some(first), Some(last)) = (infos.first(), infos.last()) else {
            return Err("a pipeline needs at least one segment".into());
        };
        if !first.embed || !last.head {
            return Err("the first segment must hold the embedding and the last the head".into());
        }
        let embeds = infos.iter().filter(|i| i.embed).count();
        let heads = infos.iter().filter(|i| i.head).count();
        if embeds != 1 || heads != 1 {
            return Err(format!(
                "{embeds} segments hold the embedding and {heads} the head, one each is needed"
            ));
        }
        for w in infos.windows(2) {
            if w[0].layers.1 != w[1].layers.0 {
                return Err(format!(
                    "segments do not meet: layers {}..{} then {}..{}",
                    w[0].layers.0, w[0].layers.1, w[1].layers.0, w[1].layers.1
                ));
            }
            // A single-segment whole model has n_embd 0 only when remote and whole; a cut
            // needs the width to check the rows between the segments.
            if w[0].n_embd != w[1].n_embd || w[0].n_embd == 0 {
                return Err("segments disagree on the hidden width".into());
            }
        }
        if infos.len() > 1 && infos.iter().any(|i| i.embed && i.head) {
            return Err("a segment that holds both the embedding and the head is the model".into());
        }
        let n = segs.len();
        Ok(Self {
            segs,
            infos,
            hidden: Vec::new(),
            logits: Vec::new(),
            spans: Vec::new(),
            lost: HashSet::new(),
            metrics: None,
            predicted: vec![None; n],
            micro: 1,
            times: vec![SegmentTimes::default(); n],
        })
    }

    /// Records `tetraeder.segment.duration` and the residual against the predictions.
    pub fn with_metrics(mut self, m: Metrics) -> Self {
        self.metrics = Some(m);
        self
    }

    /// The planner's decode step time of each segment (`ClusterPrediction::segments`), seconds;
    /// `tetraeder.plan.residual` per segment is measured against it, and the prediction is
    /// exported as `tetraeder.plan.predicted_step` with the segment.
    pub fn set_predicted(&mut self, per_segment: &[f64]) {
        for (i, p) in self.predicted.iter_mut().enumerate() {
            *p = per_segment.get(i).copied().filter(|&s| s > 0.0);
            if let (Some(m), Some(s)) = (&self.metrics, *p) {
                m.plan_predicted_segment(i as u32, s);
            }
        }
    }

    /// Cuts a prefill step of one input into `m` micro-batches kept in flight across the
    /// segments (the plan's `knobs.prefill_micro_batches`); 1, the default, walks the segments
    /// in turn.
    pub fn with_prefill_micro_batches(mut self, m: u32) -> Self {
        self.set_prefill_micro_batches(m);
        self
    }

    pub fn set_prefill_micro_batches(&mut self, m: u32) {
        self.micro = m.max(1);
    }

    pub fn segments(&self) -> &[SegmentInfo] {
        &self.infos
    }

    /// Sizes of the micro-batches of this step, when it is a prefill that is worth pipelining:
    /// one input of at least two tokens that returns its last row only (ADR 0077: a speculative
    /// verification wants every row and walks), two or more segments, and `micro` above one.
    fn micro_sizes(&self, inputs: &[SeqInput<'_>]) -> Option<Vec<usize>> {
        let [one] = inputs else { return None };
        if self.micro < 2 || self.segs.len() < 2 || one.tokens.len() < 2 || one.all_logits {
            return None;
        }
        Some(tetraeder_plan::micro_batch_sizes(
            one.tokens.len(),
            self.micro,
        ))
    }

    /// The time each segment spent in its steps (see [`SegmentTimes`]).
    pub fn segment_times(&self) -> &[SegmentTimes] {
        &self.times
    }

    /// The expert-cache counters of every segment, from its backend (a remote one asks its
    /// worker); `None` for a segment without an expert cache.
    pub fn segment_stats(&mut self) -> Vec<Option<serde_json::Value>> {
        self.segs.iter_mut().map(|s| s.expert_stats()).collect()
    }

    fn note_step(&mut self, s: usize, wall: Duration, decode: bool) {
        let own = self.segs[s].last_step_time().unwrap_or(wall);
        let t = &mut self.times[s];
        if decode {
            t.decode_steps += 1;
            t.decode_own_s += own.as_secs_f64();
            t.decode_wall_s += wall.as_secs_f64();
        } else {
            t.prefill_steps += 1;
            t.prefill_own_s += own.as_secs_f64();
            t.prefill_wall_s += wall.as_secs_f64();
        }
    }

    fn record_segment(&mut self, s: usize, wall: Duration) {
        // Only a pipelined prefill comes here.
        self.note_step(s, wall, false);
        if let Some(m) = &self.metrics {
            let own = self.segs[s].last_step_time().unwrap_or(wall);
            m.segment_step(s as u32, own.as_secs_f64());
        }
    }

    /// A pipelined prefill of one input. On an error every started step is finished (its reply
    /// read), then all segments are cut back to the lengths from before the step: a segment
    /// that failed micro-batch `k` holds the ones before it.
    fn decode_micro(
        &mut self,
        inp: &SeqInput<'_>,
        sizes: &[usize],
        before: &[(SeqId, usize)],
    ) -> Result<(), BackendError> {
        let mut offs = vec![0usize];
        for n in sizes {
            offs.push(offs[offs.len() - 1] + n);
        }
        let mut flight: Vec<VecDeque<(usize, Instant)>> = vec![VecDeque::new(); self.segs.len()];
        if let Err(e) = self.micro_loop(inp, &offs, &mut flight) {
            for (s, q) in flight.iter_mut().enumerate() {
                for _ in q.drain(..) {
                    let _ = self.segs[s].finish_step();
                }
            }
            self.cut_back(self.segs.len(), before);
            self.logits.clear();
            self.spans.clear();
            return Err(e);
        }
        let rows = self.logits.len() / self.n_vocab().max(1);
        self.spans.push(0..rows);
        Ok(())
    }

    /// Starts every step whose input is there on the segments that compute apart from us, runs
    /// one ready step on a segment that does not, else waits for the oldest started step. Rows
    /// move on as they arrive; `flight[s]` holds the started steps of segment `s` (micro-batch,
    /// start time).
    fn micro_loop(
        &mut self,
        inp: &SeqInput<'_>,
        offs: &[usize],
        flight: &mut [VecDeque<(usize, Instant)>],
    ) -> Result<(), BackendError> {
        let n_seg = self.segs.len();
        let last = n_seg - 1;
        let m = offs.len() - 1;
        let piece = |k: usize| SeqInput {
            seq: inp.seq,
            tokens: &inp.tokens[offs[k]..offs[k + 1]],
            all_logits: inp.all_logits,
        };
        let overlap: Vec<bool> = self.segs.iter().map(|s| s.can_overlap()).collect();
        // hid[s][k]: the rows segment `s` returned for micro-batch `k`, until `s + 1` took them.
        let mut hid: Vec<Vec<Option<Vec<f32>>>> = vec![vec![None; m]; n_seg];
        let mut next = vec![0usize; n_seg];
        let mut done_last = 0;
        self.logits.clear();
        self.spans.clear();
        while done_last < m {
            for s in (0..n_seg).filter(|&s| overlap[s]) {
                while next[s] < m
                    && flight[s].len() < DEPTH
                    && (s == 0 || hid[s - 1][next[s]].is_some())
                {
                    let k = next[s];
                    let rows = if s == 0 { None } else { hid[s - 1][k].take() };
                    let feed = rows.as_deref().map_or(Feed::Tokens, Feed::Hidden);
                    let p = piece(k);
                    self.segs[s].start_step(std::slice::from_ref(&p), feed)?;
                    flight[s].push_back((k, Instant::now()));
                    next[s] += 1;
                }
            }
            // One step on a segment of our own, then back to feeding the others.
            let ready = (0..n_seg)
                .rev()
                .find(|&s| !overlap[s] && next[s] < m && (s == 0 || hid[s - 1][next[s]].is_some()));
            if let Some(s) = ready {
                let k = next[s];
                let rows = if s == 0 { None } else { hid[s - 1][k].take() };
                let feed = rows.as_deref().map_or(Feed::Tokens, Feed::Hidden);
                let p = piece(k);
                let t = Instant::now();
                let res = self.segs[s].step(std::slice::from_ref(&p), feed);
                let wall = t.elapsed();
                let data = own_rows(res?, s, p.tokens.len(), s == last)?;
                self.record_segment(s, wall);
                next[s] += 1;
                if s == last {
                    self.logits = data;
                    done_last += 1;
                } else {
                    hid[s][k] = Some(data);
                }
                continue;
            }
            // Take the reply of a started step that has come in, the segment furthest along
            // first; never block on one segment while another one's reply may be waiting.
            let ready = (0..n_seg)
                .rev()
                .find(|&s| !flight[s].is_empty() && self.segs[s].reply_ready(Duration::ZERO));
            let Some(s) = ready else {
                if flight.iter().all(VecDeque::is_empty) {
                    return Err(BackendError::Failed(
                        "the pipelined prefill stalled with nothing in flight".into(),
                    ));
                }
                // Nothing has come in: look again shortly. A timed wait on a channel rounds up
                // to the timer tick on Windows (15 ms), a sleep does not.
                std::thread::sleep(Duration::from_micros(100));
                continue;
            };
            let Some((k, t)) = flight[s].pop_front() else {
                continue;
            };
            let res = self.segs[s].finish_step();
            let wall = t.elapsed();
            let data = own_rows(res?, s, offs[k + 1] - offs[k], s == last)?;
            self.record_segment(s, wall);
            if s == last {
                // The input returns its last row only: the last micro-batch's row is the result.
                self.logits = data;
                done_last += 1;
            } else {
                hid[s][k] = Some(data);
            }
        }
        Ok(())
    }

    /// Segments that hold layers, so a KV cache.
    fn kv_segs(&self) -> impl Iterator<Item = &dyn SegmentBackend> {
        self.segs
            .iter()
            .zip(&self.infos)
            .filter(|(_, i)| i.layers.0 < i.layers.1)
            .map(|(s, _)| s.as_ref())
    }

    /// Puts `seqs` back to the lengths in `lens` on the first `n` segments. A segment that
    /// cannot marks the sequence lost.
    fn cut_back(&mut self, n: usize, lens: &[(SeqId, usize)]) {
        for &(seq, len) in lens {
            for s in &mut self.segs[..n] {
                if s.seq_len(seq) > len && s.seq_rm(seq, len).is_err() {
                    self.lost.insert(seq);
                }
            }
        }
    }

    /// Undoes a call that went through the first `n` segments: the sequence is freed there.
    fn free_on(&mut self, n: usize, seq: SeqId) {
        for s in &mut self.segs[..n] {
            if s.seq_rm(seq, 0).is_err() {
                self.lost.insert(seq);
            }
        }
    }
}

impl BatchBackend for PipelineBackend {
    fn n_vocab(&self) -> usize {
        self.segs.last().map_or(0, |s| s.n_vocab())
    }

    fn n_seq_max(&self) -> usize {
        self.segs.iter().map(|s| s.n_seq_max()).min().unwrap_or(0)
    }

    fn n_ctx_seq(&self) -> usize {
        self.kv_segs()
            .map(|s| s.n_ctx_seq())
            .min()
            .unwrap_or_else(|| self.segs[0].n_ctx_seq())
    }

    fn kv_capacity(&self) -> usize {
        self.kv_segs()
            .map(|s| s.kv_capacity())
            .min()
            .unwrap_or_else(|| self.segs[0].kv_capacity())
    }

    fn kv_free(&self) -> usize {
        self.kv_segs()
            .map(|s| s.kv_free())
            .min()
            .unwrap_or_else(|| self.segs[0].kv_free())
    }

    /// The shortest of the segments (equal while they agree).
    fn seq_len(&self, seq: SeqId) -> usize {
        self.segs.iter().map(|s| s.seq_len(seq)).min().unwrap_or(0)
    }

    fn seq_reserved(&self, seq: SeqId) -> usize {
        self.segs
            .iter()
            .map(|s| s.seq_reserved(seq))
            .min()
            .unwrap_or(0)
    }

    fn seq_lost(&self, seq: SeqId) -> bool {
        self.lost.contains(&seq) || self.segs.iter().any(|s| s.seq_lost(seq))
    }

    fn seq_reserve(&mut self, seq: SeqId, n: usize) -> Result<(), BackendError> {
        for i in 0..self.segs.len() {
            if let Err(e) = self.segs[i].seq_reserve(seq, n) {
                // The earlier segments give back what this call reserved: their slack goes.
                let len = self.seq_len(seq);
                for s in &mut self.segs[..i] {
                    let _ = s.seq_rm(seq, len);
                }
                return Err(e);
            }
        }
        Ok(())
    }

    fn seq_rm(&mut self, seq: SeqId, keep: usize) -> Result<(), BackendError> {
        let mut first = None;
        let mut done = 0;
        for s in &mut self.segs {
            match s.seq_rm(seq, keep) {
                Ok(()) => done += 1,
                Err(e) => {
                    first.get_or_insert(e);
                }
            }
        }
        match first {
            None => {
                if keep == 0 {
                    self.lost.remove(&seq);
                }
                Ok(())
            }
            Some(e) => {
                if done > 0 {
                    // Some segments cut and some did not: they disagree on the length.
                    self.lost.insert(seq);
                }
                Err(e)
            }
        }
    }

    fn seq_cp(&mut self, src: SeqId, dst: SeqId, n: usize) -> Result<(), BackendError> {
        for i in 0..self.segs.len() {
            if let Err(e) = self.segs[i].seq_cp(src, dst, n) {
                self.free_on(i, dst);
                return Err(e);
            }
        }
        self.lost.remove(&dst);
        Ok(())
    }

    fn decode(&mut self, inputs: &[SeqInput<'_>]) -> Result<Outputs<'_>, BackendError> {
        self.spans.clear();
        self.logits.clear();
        if inputs.is_empty() {
            return Ok(Outputs::new(self.n_vocab(), &self.logits, &self.spans));
        }
        for i in inputs {
            if self.lost.contains(&i.seq) {
                return Err(BackendError::Failed(format!(
                    "sequence {} was lost when a segment failed to roll back",
                    i.seq
                )));
            }
        }
        let before: Vec<(SeqId, usize)> = inputs
            .iter()
            .map(|i| (i.seq, self.seq_len(i.seq)))
            .collect();
        let n_tokens: usize = inputs.iter().map(|i| i.tokens.len()).sum();
        if let Some(sizes) = self.micro_sizes(inputs) {
            self.decode_micro(&inputs[0], &sizes, &before)?;
            let width = self.n_vocab();
            return Ok(Outputs::new(width, &self.logits, &self.spans));
        }
        let decode_step = is_decode_step(inputs);
        let last = self.segs.len() - 1;
        for i in 0..=last {
            let feed = if i == 0 {
                Feed::Tokens
            } else {
                Feed::Hidden(&self.hidden)
            };
            let t = Instant::now();
            let res = self.segs[i].step(inputs, feed);
            let wall = t.elapsed();
            let out = match res {
                Ok(o) => o.rows(),
                Err(e) => {
                    self.cut_back(i, &before);
                    self.logits.clear();
                    self.spans.clear();
                    return Err(e);
                }
            };
            // The rows of every input, in input order.
            let width = out.n_vocab().max(1);
            let rows: usize = (0..out.len()).map(|k| out.rows(k).len() / width).sum();
            let expect = if i == last { None } else { Some(n_tokens) };
            if expect.is_some_and(|n| n != rows) || out.len() != inputs.len() {
                let msg = format!(
                    "segment {i} returned {rows} rows for {n_tokens} tokens of {} inputs",
                    inputs.len()
                );
                self.cut_back(i, &before);
                return Err(BackendError::Failed(msg));
            }
            let buf = if i == last {
                &mut self.logits
            } else {
                &mut self.hidden
            };
            buf.clear();
            if i == last {
                let mut start = 0;
                for k in 0..out.len() {
                    let r = out.rows(k);
                    buf.extend_from_slice(r);
                    let n = r.len() / width;
                    self.spans.push(start..start + n);
                    start += n;
                }
            } else {
                for k in 0..out.len() {
                    buf.extend_from_slice(out.rows(k));
                }
            }
            self.note_step(i, wall, decode_step);
            if let Some(m) = &self.metrics {
                let own = self.segs[i].last_step_time().unwrap_or(wall);
                m.segment_step(i as u32, own.as_secs_f64());
                if let (true, Some(p)) = (decode_step, self.predicted[i]) {
                    m.plan_residual_segment(i as u32, (own.as_secs_f64() - p).abs() / p);
                }
            }
        }
        let width = self.n_vocab();
        Ok(Outputs::new(width, &self.logits, &self.spans))
    }

    fn seq_snapshot(&mut self, seq: SeqId) -> Result<Vec<u8>, BackendError> {
        let mut out = Vec::new();
        for s in &mut self.segs {
            out.extend_from_slice(&s.seq_snapshot(seq)?);
        }
        Ok(out)
    }

    fn seq_restore(&mut self, seq: SeqId, bytes: &[u8], n: usize) -> Result<(), BackendError> {
        let rows: Vec<usize> = self.segs.iter().map(|s| s.kv_row_bytes()).collect();
        let want: usize = rows.iter().map(|r| r * n).sum();
        let has_kv = self
            .infos
            .iter()
            .zip(&rows)
            .any(|(i, &r)| i.layers.0 < i.layers.1 && r == 0);
        if has_kv || bytes.len() != want {
            return Err(BackendError::Failed(format!(
                "a pipeline snapshot of {n} positions is {want} bytes cut at these layers; \
                 got {} (a snapshot restores only into a pipeline cut the same way)",
                bytes.len()
            )));
        }
        let mut rest = bytes;
        let mut failed = None;
        for (i, (s, &r)) in self.segs.iter_mut().zip(&rows).enumerate() {
            let (mine, tail) = rest.split_at(r * n);
            rest = tail;
            if let Err(e) = s.seq_restore(seq, mine, n) {
                failed = Some((i, e));
                break;
            }
        }
        if let Some((i, e)) = failed {
            // The sequence is empty on every segment, so they agree.
            self.free_on(i + 1, seq);
            return Err(e);
        }
        self.lost.remove(&seq);
        Ok(())
    }

    fn can_cut_suffix(&self) -> bool {
        self.segs.iter().all(|s| s.can_cut_suffix())
    }

    /// The format of the segments' KV, and with two or more segments that hold layers their cut
    /// points (`f16/f16@0-2,2-4`): the snapshot bytes of one cut are not those of another.
    fn kv_format(&self) -> String {
        let base = self
            .kv_segs()
            .next()
            .map_or_else(|| self.segs[0].kv_format(), |s| s.kv_format());
        let cuts: Vec<String> = self
            .infos
            .iter()
            .filter(|i| i.layers.0 < i.layers.1)
            .map(|i| format!("{}-{}", i.layers.0, i.layers.1))
            .collect();
        if cuts.len() > 1 {
            format!("{base}@{}", cuts.join(","))
        } else {
            base
        }
    }

    fn on_gpu(&self) -> bool {
        self.segs.iter().any(|s| s.on_gpu())
    }

    fn bytes_streamed_total(&self) -> Option<u64> {
        self.segs
            .iter()
            .map(|s| s.bytes_streamed_total())
            .try_fold(0u64, |a, b| b.map(|b| a + b))
    }

    fn kv_row_bytes(&self) -> usize {
        self.segs.iter().map(|s| s.kv_row_bytes()).sum()
    }

    fn max_step_tokens(&self) -> usize {
        self.segs
            .iter()
            .map(|s| s.max_step_tokens())
            .min()
            .unwrap_or(usize::MAX)
    }
}

/// The backend of a replica that spans nodes: the segments behind one [`PipelineBackend`], with
/// the metrics wired (`tetraeder.segment.duration`, the residual per segment) and the planner's
/// decode step time of each segment as its baseline. Without this call the segment panel and
/// the `TetraederSegmentDrift` alert stay dark.
pub fn replica_pipeline(
    segs: Vec<Box<dyn SegmentBackend>>,
    metrics: Option<Metrics>,
    predicted_s: &[f64],
) -> Result<PipelineBackend, String> {
    let mut p = PipelineBackend::new(segs)?;
    if let Some(m) = metrics {
        p = p.with_metrics(m);
    }
    p.set_predicted(predicted_s);
    Ok(p)
}

/// The scheduler's step tokens for a replica that runs `plan`: its prefill micro-batches times
/// its ubatch, so a step is the one the planner priced (ADR 0077, 0091). `None` when the plan
/// names no ubatch, or `--step-tokens` is given (the operator's flag wins).
pub fn plan_step_tokens(plan: &tetraeder_plan::Plan, explicit: Option<u32>) -> Option<usize> {
    if explicit.is_some() {
        return None;
    }
    let ubatch = plan.knobs.ubatch?;
    Some(plan.knobs.prefill_micro_batches.max(1) as usize * ubatch.max(1) as usize)
}

/// The share of the layers of a segment plan (the TOML of `plan::segment_plan`) for a model of
/// `n_layer` layers. The embedding and the head come from the layer bounds, not from the rules
/// (a catch-all rule matches both in any stage): `plan::segment_shape`.
pub fn segment_spec_of(
    plan_toml: &str,
    n_layer: u32,
) -> Result<tetraeder_model::SegmentSpec, String> {
    let plan =
        tetraeder_plan::Plan::from_toml(plan_toml).map_err(|e| format!("not a plan: {e}"))?;
    let shape = tetraeder_plan::segment_shape(&plan, n_layer)?;
    let spec = tetraeder_model::SegmentSpec {
        layers: shape.layers.0..shape.layers.1,
        embed: shape.embed,
        head: shape.head,
    };
    spec.validate(n_layer).map_err(|e| e.to_string())?;
    Ok(spec)
}

/// Layers of the model in a GGUF.
pub fn n_layer_of(model: &std::path::Path) -> Result<u32, String> {
    let g = tetraeder_model::Gguf::open(model).map_err(|e| format!("{}: {e}", model.display()))?;
    let arch = g.str("general.architecture").map_err(|e| e.to_string())?;
    let n = g
        .u64(&format!("{arch}.block_count"))
        .map_err(|e| e.to_string())?;
    u32::try_from(n).map_err(|e| e.to_string())
}

/// One segment of a replica: the node that runs it and the single-node plan its worker reads.
#[derive(Debug, Clone)]
pub struct SegmentAssign {
    pub peer: tetraeder_transport::Peer,
    pub plan_toml: String,
}

/// The segments of `replica` of `plan` on the hardware graph `hw`, each with its node's address
/// from `cfg` and its sub-plan (`plan::segments` and `plan::segment_plan`).
pub fn segment_assignments(
    plan: &tetraeder_plan::Plan,
    replica: usize,
    hw: &tetraeder_profiler::HardwareGraph,
    cfg: &tetraeder_control::ClusterConfig,
) -> Result<Vec<SegmentAssign>, String> {
    let segs = tetraeder_plan::segments(plan, replica, hw).map_err(|e| e.join("; "))?;
    segs.iter()
        .map(|s| {
            let node = s
                .node
                .clone()
                .ok_or("a segment of a cluster plan sits on a node")?;
            let peer = cfg
                .peer(&tetraeder_transport::NodeId::new(node.clone()))
                .ok_or_else(|| format!("`{node}` is not a node of cluster.toml"))?;
            let sub = tetraeder_plan::segment_plan(plan, replica, s);
            let plan_toml = sub.to_toml().map_err(|e| e.to_string())?;
            Ok(SegmentAssign { peer, plan_toml })
        })
        .collect()
}

/// Tells each segment's daemon to start its worker (`Assign`, as the head, on a probe connection
/// of session 0) and connects a [`RemoteBackend`] to each worker. `tune` adjusts the config of
/// every backend (deadlines, restart policy). The backends are in segment order, ready for
/// [`PipelineBackend::new`].
pub fn start_remote_segments(
    t: &std::sync::Arc<dyn tetraeder_transport::Transport>,
    replica: u32,
    segs: &[SegmentAssign],
    model_sha256: &str,
    tune: &dyn Fn(&mut crate::remote_backend::RemoteConfig),
) -> Result<Vec<crate::remote_backend::RemoteBackend>, String> {
    use crate::remote_backend::{RemoteBackend, RemoteConfig};
    let mut out = Vec::new();
    for (i, s) in segs.iter().enumerate() {
        let segment = u32::try_from(i).map_err(|e| e.to_string())?;
        tetraeder_control::client::request(
            t.as_ref(),
            &s.peer,
            &tetraeder_control::proto::NodeRequest::Assign {
                replica,
                segment,
                plan: s.plan_toml.clone(),
                model_sha256: model_sha256.to_owned(),
            },
            &[],
        )?;
        let mut cfg =
            RemoteConfig::node(std::sync::Arc::clone(t), s.peer.clone(), replica, segment);
        tune(&mut cfg);
        out.push(RemoteBackend::start(cfg).map_err(|e| format!("segment {i}: {e}"))?);
    }
    Ok(out)
}
