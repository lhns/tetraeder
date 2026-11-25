//! Prediction across nodes (ADR 0091, "Cost model: network links").
//!
//! A replica cut into segments is priced as each segment's single-node [`predict`] on its node's
//! graph plus the network hops between them. Hops follow the star of the pipeline backend: the
//! leader (the first segment's node) sends every segment its input and takes its output, so a
//! segment on another node costs one hop in and one hop out, and the output is the hidden rows,
//! or the logits row when the segment holds the head. A segment on the leader costs no hop.
//!
//! ```text
//! decode:  T = sum over the path of  segment 1 / tg  and  hop_time(bytes)
//! prefill: per step of micro x ubatch tokens, cut into micro chunks, the path is a flow shop:
//!          a segment or a link takes a chunk after the chunk before it, and after the chunk's
//!          own previous element (GPipe, the schedule of ADR 0077, hops serialized per link)
//! ```
//!
//! With one micro-batch a segment's time for the step is the difference of its whole-prompt
//! prefill time at the step's end and start, so the steps add up to its own prefill time; with
//! links that cost nothing the result is then the sum of the segment times. With several
//! micro-batches each chunk is a batch of its own: its time is the difference of the prefill
//! times of the segment at ubatch = chunk size, so the per-batch terms (the RAM-to-GPU offload
//! copy, the weight reads, the launches) are paid once per chunk and the chunks add up to more
//! than the segment's whole-prompt time (ADR 0092 section 5).
//! One segment predicts exactly as [`predict`] does (ADR 0091, C7).

use crate::predict::{PredictError, Prediction, RunFeatures, predict};
use crate::transfer_time;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use tetraeder_plan::micro_batch_sizes;
use tetraeder_profiler::{HardwareGraph, curve_time};

/// One-way time of `bytes` from node `from` to node `to` (`""` is the local node): the probe
/// curve of that direction (`CollectiveOp::Copy` between devices of the two nodes, comm `tls`
/// preferred) interpolated as ADR 0070's curves, else `alpha + bytes / beta` of the `Network`
/// link. Measured curves win: when one exists, alpha and beta are not read. `None` without
/// either; 0 within a node.
pub fn hop_time(hw: &HardwareGraph, from: &str, to: &str, bytes: f64) -> Option<f64> {
    if from == to {
        return Some(0.0);
    }
    if let Some(c) = hw.network_curve(from, to) {
        return curve_time(&c.points, bytes);
    }
    let l = hw.network_link(from, to)?;
    Some(transfer_time(
        l.alpha_s.value,
        l.beta_bytes_per_s.value,
        bytes,
    ))
}

/// One segment of a replica: its node, its single-node run features (ids as the node's own
/// graph names them) and what it holds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentFeatures {
    pub node: Option<String>,
    pub run: RunFeatures,
    pub embed: bool,
    pub head: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusterFeatures {
    /// In pipeline order; the first one's node is the leader.
    pub segments: Vec<SegmentFeatures>,
    /// A hidden row is `n_embd x 4` bytes, a logits row `n_vocab x 4`.
    pub n_embd: u32,
    pub n_vocab: u32,
    /// Chunks a prefill step is cut into (ADR 0077); 1 runs the segments in turn.
    pub prefill_micro_batches: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusterPrediction {
    /// Each segment on its own, as [`predict`] returns it.
    pub segments: Vec<Prediction>,
    /// Network time in one decode token.
    pub hop_s_per_token: f64,
    pub pp_tps: f64,
    pub tg_tps: f64,
    /// `prompt / pp + gen / tg`.
    pub request_s: f64,
}

/// What a hop carries.
#[derive(Clone, Copy)]
enum Rows {
    Hidden,
    Logits,
}

/// One element of the path a token takes.
enum Elem {
    Seg(usize),
    Hop {
        from: String,
        to: String,
        rows: Rows,
    },
}

fn key(n: &Option<String>) -> String {
    n.clone().unwrap_or_default()
}

/// The path of one pass through the segments, as the star through the leader.
fn path(f: &ClusterFeatures) -> Vec<Elem> {
    let leader = key(&f.segments[0].node);
    let mut out = Vec::new();
    for (i, s) in f.segments.iter().enumerate() {
        let node = key(&s.node);
        if i > 0 {
            let prev = key(&f.segments[i - 1].node);
            if prev != leader {
                out.push(Elem::Hop {
                    from: prev,
                    to: leader.clone(),
                    rows: Rows::Hidden,
                });
            }
            if node != leader {
                out.push(Elem::Hop {
                    from: leader.clone(),
                    to: node.clone(),
                    rows: Rows::Hidden,
                });
            }
        }
        out.push(Elem::Seg(i));
    }
    let last = &f.segments[f.segments.len() - 1];
    let node = key(&last.node);
    if node != leader {
        let rows = if last.head {
            Rows::Logits
        } else {
            Rows::Hidden
        };
        out.push(Elem::Hop {
            from: node,
            to: leader,
            rows,
        });
    }
    out
}

/// Seconds a segment needs to prefill prompt tokens, from its own prediction.
struct PrefillTimes<'a> {
    graphs: &'a [HardwareGraph],
    f: &'a ClusterFeatures,
    memo: BTreeMap<(usize, u32, u32), f64>,
}

impl PrefillTimes<'_> {
    /// Seconds segment `seg` needs for `tokens` prompt tokens in ubatches of `ubatch`.
    fn upto(&mut self, seg: usize, tokens: u32, ubatch: u32) -> Result<f64, PredictError> {
        if tokens == 0 {
            return Ok(0.0);
        }
        if let Some(t) = self.memo.get(&(seg, tokens, ubatch)) {
            return Ok(*t);
        }
        let mut run = self.f.segments[seg].run.clone();
        run.ubatch = ubatch;
        let p = predict(&self.graphs[seg], &run, tokens, 1)?;
        let t: f64 = p.pp_breakdown.iter().map(|e| e.1).sum();
        self.memo.insert((seg, tokens, ubatch), t);
        Ok(t)
    }

    /// Seconds the segment needs for the chunk `[at, at + m)` of a pipelined step. A chunk is a
    /// batch of its own (ADR 0077): it pays what a ubatch pays once, the offload copy of the
    /// RAM-resident weights included (ADR 0092 section 5), so the chunks are priced as ubatches
    /// of `m` tokens. Without micro-batches the chunk is the step and the segment's own ubatch
    /// applies.
    fn chunk(&mut self, seg: usize, at: u32, m: u32, micro: u32) -> Result<f64, PredictError> {
        let ub = if micro > 1 {
            m
        } else {
            self.f.segments[seg].run.ubatch
        };
        Ok(self.upto(seg, at + m, ub)? - self.upto(seg, at, ub)?)
    }
}

/// See the module doc.
pub fn predict_cluster(
    hw: &HardwareGraph,
    f: &ClusterFeatures,
    prompt: u32,
    gen_tokens: u32,
) -> Result<ClusterPrediction, PredictError> {
    if f.segments.is_empty() || prompt == 0 {
        return Err(PredictError::Invalid("need segments and a prompt".into()));
    }
    let graphs: Vec<HardwareGraph> = f
        .segments
        .iter()
        .map(|s| hw.node_graph(s.node.as_deref()))
        .collect();
    let mut segs = Vec::with_capacity(f.segments.len());
    for (g, s) in graphs.iter().zip(&f.segments) {
        segs.push(predict(g, &s.run, prompt, gen_tokens)?);
    }
    if f.segments.len() == 1 {
        let p = &segs[0];
        return Ok(ClusterPrediction {
            hop_s_per_token: 0.0,
            pp_tps: p.pp_tok_s,
            tg_tps: p.tg_tok_s,
            request_s: p.request_s,
            segments: segs,
        });
    }
    let elems = path(f);
    let hop = |from: &str, to: &str, bytes: f64| {
        if bytes <= 0.0 {
            return Ok(0.0);
        }
        hop_time(hw, from, to, bytes).ok_or_else(|| PredictError::NoLink {
            a: from.to_string(),
            b: to.to_string(),
        })
    };
    let row = |rows: Rows| match rows {
        Rows::Hidden => 4.0 * f64::from(f.n_embd),
        Rows::Logits => 4.0 * f64::from(f.n_vocab),
    };

    // Decode: one token through the path.
    let mut hops = 0.0;
    let mut t_tok = 0.0;
    for e in &elems {
        match e {
            Elem::Seg(i) => t_tok += 1.0 / segs[*i].tg_tok_s,
            Elem::Hop { from, to, rows } => {
                let t = hop(from, to, row(*rows))?;
                hops += t;
                t_tok += t;
            }
        }
    }

    // Prefill: steps of micro x ubatch tokens, each a flow shop over the path.
    let micro = f.prefill_micro_batches.max(1);
    let ubatch = f.segments[0].run.ubatch.max(1);
    let mut times = PrefillTimes {
        graphs: &graphs,
        f,
        memo: BTreeMap::new(),
    };
    let (mut t_pp, mut done) = (0.0f64, 0u32);
    while done < prompt {
        let n = (ubatch * micro).min(prompt - done);
        let chunks = micro_batch_sizes(n as usize, micro);
        let mut free: BTreeMap<String, f64> = BTreeMap::new();
        let mut at = done;
        let mut finish = 0.0f64;
        for (ci, &c) in chunks.iter().enumerate() {
            let m = c as u32;
            let mut ready = 0.0f64;
            for e in &elems {
                let (name, dur) = match e {
                    Elem::Seg(i) => (format!("seg {i}"), times.chunk(*i, at, m, micro)?),
                    Elem::Hop { from, to, rows } => {
                        // The logits row leaves once per step, with its last chunk.
                        let bytes = match rows {
                            Rows::Hidden => row(*rows) * f64::from(m),
                            Rows::Logits if ci + 1 == chunks.len() => row(*rows),
                            Rows::Logits => 0.0,
                        };
                        (format!("hop {from}>{to}"), hop(from, to, bytes)?)
                    }
                };
                let slot = free.entry(name).or_insert(0.0);
                *slot = ready.max(*slot) + dur.max(0.0);
                ready = *slot;
            }
            finish = finish.max(ready);
            at += m;
        }
        t_pp += finish;
        done += n;
    }
    Ok(ClusterPrediction {
        segments: segs,
        hop_s_per_token: hops,
        pp_tps: f64::from(prompt) / t_pp,
        tg_tps: 1.0 / t_tok,
        request_s: t_pp + f64::from(gen_tokens) * t_tok,
    })
}

#[cfg(test)]
mod tests;
