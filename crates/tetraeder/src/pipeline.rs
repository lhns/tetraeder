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
//! bytes of a one-node snapshot (those hold all K layers, then all V layers), so a snapshot
//! restores only into a pipeline cut at the same layers, and the prefix cache keys it by
//! [`BatchBackend::kv_format`]; moving KV between differently cut replicas (step 5) needs the
//! K and V halves of each segment's share separated first.
//!
//! Not done here: micro-batches in flight across segments for a prefill (ADR 0077). A prefill
//! walks the segments one chunk at a time, as a decode does.

use std::collections::HashSet;
use std::ops::Range;
use std::time::{Duration, Instant};
use tetraeder_scheduler::{
    BackendError, BatchBackend, Feed, Outputs, SegmentBackend, SegmentInfo, SeqId, SeqInput,
};
use tetraeder_telemetry::Metrics;

/// Decode steps of one token on one sequence are what the planner predicts; only they feed the
/// residual.
fn is_decode_step(inputs: &[SeqInput<'_>]) -> bool {
    matches!(inputs, [one] if one.tokens.len() == 1)
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

    pub fn segments(&self) -> &[SegmentInfo] {
        &self.infos
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

    fn kv_format(&self) -> String {
        self.kv_segs()
            .next()
            .map_or_else(|| self.segs[0].kv_format(), |s| s.kv_format())
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

/// How long a step took in each segment, for tests.
pub fn step_times(p: &PipelineBackend) -> Vec<Option<Duration>> {
    p.segs.iter().map(|s| s.last_step_time()).collect()
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
