//! One judged Case S round across the segments of a pipeline (ADR 0091 answers 6 to 8, ADR 0092
//! owner answer 1): the warm request(s), then a timed request of the same prompt that
//! recomputes every token (the sequence is cleared first and the pipeline has no prefix reuse
//! or prompt cache), pp then tg, with each segment's counters.
//!
//! The protocol and the JSON are the exit harness's (`crates/model/tests/exit_regime.rs`): the
//! files are the ones `scripts/phase6_exit_s_summary.py` reads. `tetraeder cluster judged-round`
//! builds the pipeline and calls [`run_round`]; the tests drive it on any [`PipelineBackend`].

use crate::pipeline::{PipelineBackend, SegmentTimes};
use serde_json::{Value, json};
use std::ops::Range;
use std::time::Instant;
use tetraeder_scheduler::{BatchBackend, SeqId, SeqInput};

/// The one sequence the round uses.
const SEQ: SeqId = 0;

pub struct Workload<'a> {
    pub prompt: &'a [i32],
    /// Greedy tokens decoded after the prompt.
    pub gen_tokens: usize,
    /// Prompt tokens per `decode` call: the plan's micro-batches times its ubatch, so a call is
    /// the step the planner priced (the pipeline cuts it into the micro-batches).
    pub step_tokens: usize,
    /// Untimed full requests before the timed one.
    pub warm: usize,
}

/// FNV-1a of the token ids as little-endian i32, hex: the `prompt.fnv1a64` of the exit harness.
pub fn tokens_hash(tokens: &[i32]) -> String {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in tokens.iter().flat_map(|t| t.to_le_bytes()) {
        h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
    }
    format!("{h:016x}")
}

fn argmax(row: &[f32]) -> i32 {
    let mut best = 0;
    for (i, v) in row.iter().enumerate() {
        if *v > row[best] {
            best = i;
        }
    }
    best as i32
}

/// The last row of the one input's output.
fn last_row(p: &mut PipelineBackend, tokens: &[i32]) -> Result<Vec<f32>, String> {
    let nv = p.n_vocab();
    let out = p
        .decode(&[SeqInput {
            seq: SEQ,
            tokens,
            all_logits: false,
        }])
        .map_err(|e| e.to_string())?;
    let rows = out.rows(0);
    if rows.len() < nv {
        return Err(format!("{} logits for a vocabulary of {nv}", rows.len()));
    }
    Ok(rows[rows.len() - nv..].to_vec())
}

/// What the pipeline held at one moment: each segment's expert counters and times.
struct Reading {
    stats: Vec<Option<Value>>,
    times: Vec<SegmentTimes>,
}

fn read(p: &mut PipelineBackend) -> Reading {
    Reading {
        stats: p.segment_stats(),
        times: p.segment_times().to_vec(),
    }
}

struct Timed {
    prefill_s: f64,
    decode_s: f64,
    /// The sequence length after the prefill: every prompt token was computed.
    computed_prompt_tokens: usize,
    fed: Vec<i32>,
    first_top: Vec<(usize, f32)>,
}

/// One request from an empty sequence, every prompt token computed.
fn request(
    p: &mut PipelineBackend,
    w: &Workload<'_>,
    mid: &mut dyn FnMut(&mut PipelineBackend),
) -> Result<Timed, String> {
    p.seq_rm(SEQ, 0).map_err(|e| e.to_string())?;
    let t0 = Instant::now();
    let mut last = Vec::new();
    for chunk in w.prompt.chunks(w.step_tokens.max(1)) {
        last = last_row(p, chunk)?;
    }
    let prefill_s = t0.elapsed().as_secs_f64();
    let computed_prompt_tokens = p.seq_len(SEQ);
    let mut top: Vec<(usize, f32)> = last.iter().copied().enumerate().collect();
    top.sort_by(|a, b| b.1.total_cmp(&a.1));
    top.truncate(10);
    mid(p);
    let t0 = Instant::now();
    let mut fed = Vec::with_capacity(w.gen_tokens);
    let mut next = argmax(&last);
    for _ in 0..w.gen_tokens {
        fed.push(next);
        let row = last_row(p, &[next])?;
        next = argmax(&row);
    }
    Ok(Timed {
        prefill_s,
        decode_s: t0.elapsed().as_secs_f64(),
        computed_prompt_tokens,
        fed,
        first_top: top,
    })
}

/// A counter of a layer entry, 0 when absent.
fn num(v: &Value, k: &str) -> u64 {
    v.get(k).and_then(Value::as_u64).unwrap_or(0)
}

fn layers_of(v: &Value) -> &[Value] {
    v.get("layers")
        .and_then(Value::as_array)
        .map_or(&[], Vec::as_slice)
}

/// The exit harness's `layers_json` on two readings: per layer of `span`, the share of decode
/// layer steps with every routed expert in the compute tier, the RAM tier's hit rate and the
/// pair share (compute-tier hits over hits plus CPU experts, null where the layer saw none).
pub fn layers_json(a: &Value, b: &Value, span: Range<usize>) -> Value {
    let (mut full, mut ram_hit, mut share, mut pairs) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut hits, mut cpu) = (0u64, 0u64);
    for (la, lb) in layers_of(a)
        .iter()
        .zip(layers_of(b))
        .skip(span.start)
        .take(span.len())
    {
        let f = num(lb, "full_steps") - num(la, "full_steps");
        let p = num(lb, "partial_steps") - num(la, "partial_steps");
        full.push(f as f64 / (f + p).max(1) as f64);
        if la.get("has_ram") == Some(&Value::Bool(true))
            && lb.get("has_ram") == Some(&Value::Bool(true))
        {
            let h = num(lb, "ram_hits") - num(la, "ram_hits");
            let m = num(lb, "ram_misses") - num(la, "ram_misses");
            ram_hit.push(h as f64 / (h + m).max(1) as f64);
        }
        let (h, c) = (
            num(lb, "compute_hits") - num(la, "compute_hits"),
            num(lb, "cpu") - num(la, "cpu"),
        );
        hits += h;
        cpu += c;
        pairs.push(h + c);
        share.push(if h + c > 0 {
            json!(h as f64 / (h + c) as f64)
        } else {
            Value::Null
        });
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    json!({
        "vram_pair_share": if hits + cpu > 0 { json!(hits as f64 / (hits + cpu) as f64) } else { Value::Null },
        "first_layer": span.start,
        "vram_pair_share_layers": share,
        "pairs_layers": pairs,
        "full_hit_rate_mean": mean(&full),
        "ram_hit_rate_mean": mean(&ram_hit),
        "full_hit_rate": full,
        "ram_hit_rate": ram_hit,
    })
}

/// Bytes read from storage between two readings, `None` without counters.
fn bytes_between(a: &Option<Value>, b: &Option<Value>) -> Option<u64> {
    let (a, b) = (a.as_ref()?, b.as_ref()?);
    Some(num(b, "bytes_read").saturating_sub(num(a, "bytes_read")))
}

/// Whether the compute and RAM slots of a layer hold every expert (the exit harness's `covers`).
fn covers(stats: &Value) -> bool {
    let n = num(stats, "n_expert");
    n > 0 && num(stats, "vram_slots") + num(stats, "ram_slots") >= n
}

fn times_json(t: &SegmentTimes) -> Value {
    json!({
        "prefill_steps": t.prefill_steps,
        "prefill_own_s": t.prefill_own_s,
        "prefill_wall_s": t.prefill_wall_s,
        "prefill_hop_s": (t.prefill_wall_s - t.prefill_own_s).max(0.0),
        "decode_steps": t.decode_steps,
        "decode_own_s": t.decode_own_s,
        "decode_wall_s": t.decode_wall_s,
        "decode_hop_s": (t.decode_wall_s - t.decode_own_s).max(0.0),
    })
}

/// One segment as the round saw it: the node that runs it and its layers.
#[derive(Debug, Clone)]
pub struct SegmentLabel {
    pub node: String,
}

/// The files of one round: `cluster` for `cluster.json`, and one `(node, JSON)` per segment for
/// `seg-<node>.json`.
pub struct RoundFiles {
    pub cluster: Value,
    pub segments: Vec<(String, Value)>,
}

/// Runs the warm requests and the timed one on `p` and builds the files. `labels` names the node
/// of each segment, in order.
pub fn run_round(
    p: &mut PipelineBackend,
    labels: &[SegmentLabel],
    w: &Workload<'_>,
) -> Result<RoundFiles, String> {
    if labels.len() != p.segments().len() {
        return Err("one label per segment".into());
    }
    let mut warm = Vec::new();
    for _ in 0..w.warm {
        let t = request(p, w, &mut |_| {})?;
        warm.push(json!({
            "prefill_s": t.prefill_s,
            "prefill_tok_s": w.prompt.len() as f64 / t.prefill_s,
            "decode_s": t.decode_s,
        }));
    }
    let start = read(p);
    let mut after_prefill = None;
    let timed = request(p, w, &mut |p| after_prefill = Some(read(p)))?;
    let after_prefill = after_prefill.ok_or("no reading after the prefill")?;
    let end = read(p);
    if timed.computed_prompt_tokens != w.prompt.len() {
        return Err(format!(
            "the timed prefill left {} positions for a prompt of {}: tokens were not all computed",
            timed.computed_prompt_tokens,
            w.prompt.len()
        ));
    }
    let n = w.prompt.len();
    let prompt = json!({"tokens": n, "fnv1a64": tokens_hash(w.prompt)});
    let infos: Vec<_> = p.segments().to_vec();
    let mut segments = Vec::new();
    let mut seg_json = Vec::new();
    for (i, info) in infos.iter().enumerate() {
        let node = &labels[i].node;
        let span = info.layers.0 as usize..info.layers.1 as usize;
        let pre_t = after_prefill.times[i].since(&start.times[i]);
        let dec_t = end.times[i].since(&after_prefill.times[i]);
        let (s0, s1, s2) = (&start.stats[i], &after_prefill.stats[i], &end.stats[i]);
        let prefill_bytes = bytes_between(s0, s1);
        let covering = s2.as_ref().is_some_and(covers);
        let rate = |tokens: usize, s: f64| {
            if s > 0.0 {
                json!(tokens as f64 / s)
            } else {
                Value::Null
            }
        };
        let run = json!({
            "engine": "tetraeder",
            "segment": {
                "node": node,
                "layers": [info.layers.0, info.layers.1],
                "embed": info.embed,
                "head": info.head,
            },
            "prompt": prompt,
            // the segment's own compute, from its worker; the cluster rates are in cluster.json
            "prefill_tok_s": rate(n, pre_t.prefill_own_s),
            "decode_tok_s": rate(w.gen_tokens, dec_t.decode_own_s),
            "warm": {
                "requests": w.warm,
                "covering": covering,
                "timed_prefill_disk_bytes": prefill_bytes.unwrap_or(0),
                "unclean": covering && prefill_bytes.unwrap_or(0) > 0,
                "prefix_reuse": false,
                "prompt_cache": false,
                "computed_prompt_tokens": timed.computed_prompt_tokens,
            },
            "warm_requests": warm,
            "prefill_io": prefill_bytes.map_or(Value::Null, |b| json!({"bytes": b})),
            "decode_io": bytes_between(s1, s2).map_or(Value::Null, |b| json!({"bytes": b})),
            "decode_layers": match (s1, s2) {
                (Some(a), Some(b)) => layers_json(a, b, span.clone()),
                _ => Value::Null,
            },
            "prefill_layers": match (s0, s1) {
                (Some(a), Some(b)) => layers_json(a, b, span),
                _ => Value::Null,
            },
            "tiers": s2.as_ref().map(|s| json!({
                "n_expert": num(s, "n_expert"),
                "vram_slots": num(s, "vram_slots"),
                "ram_slots": num(s, "ram_slots"),
            })),
            "times": {"prefill": times_json(&pre_t), "decode": times_json(&dec_t)},
        });
        seg_json.push(json!({"node": node, "layers": [info.layers.0, info.layers.1], "prefill": times_json(&pre_t), "decode": times_json(&dec_t),
                             "prefill_io_bytes": prefill_bytes, "pair_share": run["decode_layers"]["vram_pair_share"]}));
        segments.push((node.clone(), json!({"runs": [run]})));
    }
    let cluster = json!({"runs": [{
        "engine": "tetraeder",
        "prompt": prompt,
        "prefill_tokens": n,
        "prefill_s": timed.prefill_s,
        "prefill_tok_s": n as f64 / timed.prefill_s,
        "decode_tokens": w.gen_tokens,
        "decode_s": timed.decode_s,
        "decode_tok_s": w.gen_tokens as f64 / timed.decode_s,
        "step_tokens": w.step_tokens,
        "warm_requests": warm,
        "decode_fed": timed.fed,
        "first_top": timed.first_top.iter().map(|(i, v)| json!([i, v])).collect::<Vec<_>>(),
        "segments": seg_json,
    }]});
    Ok(RoundFiles { cluster, segments })
}

/// The prompt tokens of a hidden-row dump: the frozen prompt (the first `n_prompt` ids).
pub fn prompt_of_dump(path: &std::path::Path) -> Result<Vec<i32>, String> {
    let f = tetraeder_model::hidden_file::HiddenFile::read(path)?;
    Ok(f.tokens[..f.n_prompt].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_prompt_hash_is_the_exit_harness_one() {
        // fnv1a64 of the empty list is the offset basis
        assert_eq!(tokens_hash(&[]), "cbf29ce484222325");
        assert_ne!(tokens_hash(&[1, 2, 3]), tokens_hash(&[3, 2, 1]));
    }

    fn layer(compute_hits: u64, cpu: u64, full: u64, partial: u64, rh: u64, rm: u64) -> Value {
        json!({"compute_hits": compute_hits, "cpu": cpu, "full_steps": full, "partial_steps": partial,
               "ram_hits": rh, "ram_misses": rm, "has_ram": true})
    }

    /// The deltas, over the span only.
    #[test]
    fn layers_json_takes_the_difference_over_the_span() {
        let a = json!({"layers": [layer(0, 0, 0, 0, 0, 0), layer(10, 10, 1, 1, 5, 5), layer(0, 0, 0, 0, 0, 0)]});
        let b = json!({"layers": [layer(9, 9, 9, 9, 9, 9), layer(40, 20, 4, 1, 25, 5), layer(9, 9, 9, 9, 9, 9)]});
        let j = layers_json(&a, &b, 1..2);
        assert_eq!(j["first_layer"], 1);
        // 30 hits and 10 cpu experts
        assert_eq!(j["vram_pair_share"], 0.75);
        assert_eq!(j["pairs_layers"], json!([40]));
        // 3 full steps of 4
        assert_eq!(j["full_hit_rate_mean"], 0.75);
        // 20 hits of 20 lookups
        assert_eq!(j["ram_hit_rate_mean"], 1.0);
    }

    #[test]
    fn covering_needs_the_tiers_to_hold_every_expert() {
        assert!(covers(
            &json!({"n_expert": 8, "vram_slots": 3, "ram_slots": 5})
        ));
        assert!(!covers(
            &json!({"n_expert": 8, "vram_slots": 3, "ram_slots": 4})
        ));
        assert!(!covers(&json!({})));
    }
}
