//! The segment and warm parts of the exit harness (ADR 0092, steps E2 and E3): flag parsing, the
//! hidden-row dump file, the fixed-seed hidden input, a request run through one or more layer
//! segments, and the warm protocol. `exit_regime.rs` builds its JSON from these and the CI tests
//! drive them on the tiny fixtures.
//!
//! Hardware label: none of this needs hardware; the callers say.

use super::argmax;
use serde_json::{Value, json};
use std::ops::Range;
use std::path::PathBuf;
use std::time::Instant;
use tetraeder_model::{ExpertStats, Qwen3Moe, SegmentSpec, SeqEval, StepProfile};

/// The seed of the fixed hidden input when `TETRAEDER_EXIT_HIDDEN_SEED` is not set.
pub const DEFAULT_SEED: u64 = 20261124;

/// `TETRAEDER_EXIT_HIDDEN_OUT=layer,file`: dump the hidden rows entering `layer`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HiddenOut {
    pub layer: u32,
    pub file: PathBuf,
}

/// The hidden input of a span that does not start at layer 0.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HiddenIn {
    File(PathBuf),
    Seed(u64),
}

/// The segment flags of the exit harness, parsed.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SegmentArgs {
    /// `TETRAEDER_EXIT_LAYERS=a..b`: run only these layers.
    pub layers: Option<Range<u32>>,
    /// `TETRAEDER_EXIT_HEAD=1`: the span ends at the last layer and keeps the head.
    pub head: bool,
    pub hidden_in: Option<HiddenIn>,
    pub hidden_out: Option<HiddenOut>,
}

/// `a..b` with `a < b <= n_layer`.
pub fn parse_span(s: &str, n_layer: u32) -> Result<Range<u32>, String> {
    let bad = || format!("TETRAEDER_EXIT_LAYERS {s:?}: want a..b with a < b <= {n_layer}");
    let (a, b) = s.split_once("..").ok_or_else(bad)?;
    let a: u32 = a.trim().parse().map_err(|_| bad())?;
    let b: u32 = b.trim().parse().map_err(|_| bad())?;
    if a >= b || b > n_layer {
        return Err(bad());
    }
    Ok(a..b)
}

/// `layer,file` with `0 < layer < n_layer`.
pub fn parse_hidden_out(s: &str, n_layer: u32) -> Result<HiddenOut, String> {
    let bad =
        || format!("TETRAEDER_EXIT_HIDDEN_OUT {s:?}: want layer,file with 0 < layer < {n_layer}");
    let (l, f) = s.split_once(',').ok_or_else(bad)?;
    let layer: u32 = l.trim().parse().map_err(|_| bad())?;
    if layer == 0 || layer >= n_layer || f.trim().is_empty() {
        return Err(bad());
    }
    Ok(HiddenOut {
        layer,
        file: PathBuf::from(f.trim()),
    })
}

impl SegmentArgs {
    /// Reads the flags through `get` (the process environment, or a map in a test). Errors name
    /// the flag.
    pub fn parse(get: impl Fn(&str) -> Option<String>, n_layer: u32) -> Result<Self, String> {
        let layers = get("TETRAEDER_EXIT_LAYERS")
            .map(|s| parse_span(&s, n_layer))
            .transpose()?;
        let head = get("TETRAEDER_EXIT_HEAD").is_some_and(|v| v == "1");
        let hidden_out = get("TETRAEDER_EXIT_HIDDEN_OUT")
            .map(|s| parse_hidden_out(&s, n_layer))
            .transpose()?;
        let file = get("TETRAEDER_EXIT_HIDDEN_IN").map(PathBuf::from);
        let seed = get("TETRAEDER_EXIT_HIDDEN_SEED")
            .map(|s| {
                s.trim()
                    .parse::<u64>()
                    .map_err(|_| format!("TETRAEDER_EXIT_HIDDEN_SEED {s:?}: not a number"))
            })
            .transpose()?;
        let Some(span) = layers.clone() else {
            if file.is_some() || seed.is_some() || head {
                return Err(
                    "TETRAEDER_EXIT_HIDDEN_IN, _HIDDEN_SEED and _HEAD need TETRAEDER_EXIT_LAYERS"
                        .into(),
                );
            }
            return Ok(Self {
                hidden_out,
                ..Self::default()
            });
        };
        if hidden_out.is_some() {
            return Err(
                "TETRAEDER_EXIT_HIDDEN_OUT dumps from a whole-model run: not with TETRAEDER_EXIT_LAYERS"
                    .into(),
            );
        }
        if head && span.end != n_layer {
            return Err(format!(
                "TETRAEDER_EXIT_HEAD=1 needs the span to end at layer {n_layer}"
            ));
        }
        let hidden_in = if span.start == 0 {
            // layer 0 takes tokens; a file only lends its token ids to the decode steps
            if seed.is_some() {
                return Err("TETRAEDER_EXIT_HIDDEN_SEED: a span from layer 0 takes tokens".into());
            }
            file.map(HiddenIn::File)
        } else {
            Some(match file {
                Some(f) => HiddenIn::File(f),
                None => HiddenIn::Seed(seed.unwrap_or(DEFAULT_SEED)),
            })
        };
        let me = Self {
            layers,
            head,
            hidden_in,
            hidden_out: None,
        };
        for s in me.specs(n_layer) {
            s.validate(n_layer).map_err(|e| e.to_string())?;
        }
        Ok(me)
    }

    /// Whether the run is a segment run (a span or a dump), not the whole-model path.
    pub fn is_segment_run(&self) -> bool {
        self.layers.is_some() || self.hidden_out.is_some()
    }

    /// The segments to load, in order: the span, or for a dump the two halves of the model cut at
    /// the dumped layer (the first without the head, the second with it), or none.
    pub fn specs(&self, n_layer: u32) -> Vec<SegmentSpec> {
        if let Some(span) = &self.layers {
            return vec![SegmentSpec {
                layers: span.clone(),
                embed: span.start == 0,
                head: self.head,
            }];
        }
        match &self.hidden_out {
            Some(o) => vec![
                SegmentSpec {
                    layers: 0..o.layer,
                    embed: true,
                    head: false,
                },
                SegmentSpec {
                    layers: o.layer..n_layer,
                    embed: false,
                    head: true,
                },
            ],
            None => Vec::new(),
        }
    }

    /// The segment whose output rows a dump keeps: the first of two.
    pub fn tap_at(&self) -> Option<usize> {
        self.hidden_out.as_ref().map(|_| 0)
    }
}

pub use tetraeder_model::hidden_file::HiddenFile;

/// FNV-1a over bytes, as hex (the file's identity in the result JSON).
pub fn fnv1a_bytes(b: &[u8]) -> String {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &x in b {
        h = (h ^ u64::from(x)).wrapping_mul(0x100_0000_01b3);
    }
    format!("{h:016x}")
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `n_rows * n_embd` standard-normal values from `seed` (splitmix64, Box-Muller): the fixed
/// input of a span that has no real hidden rows. The same on every platform.
pub fn seeded_rows(seed: u64, n_rows: usize, n_embd: usize) -> Vec<f32> {
    let mut s = seed;
    let mut uniform = || ((splitmix(&mut s) >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    let mut out = Vec::with_capacity(n_rows * n_embd);
    while out.len() < n_rows * n_embd {
        let (u, v) = (uniform(), uniform());
        let r = (-2.0 * u.ln()).sqrt();
        out.push((r * (std::f64::consts::TAU * v).cos()) as f32);
        if out.len() < n_rows * n_embd {
            out.push((r * (std::f64::consts::TAU * v).sin()) as f32);
        }
    }
    out
}

/// `n` fixed pseudo-random token ids, for the decode steps of a span that picks no tokens.
pub fn pseudo_tokens(n: usize, n_vocab: usize) -> Vec<i32> {
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    (0..n)
        .map(|_| (splitmix(&mut s) % n_vocab as u64) as i32)
        .collect()
}

/// What a run of the first segment is fed.
#[derive(Debug, Clone, PartialEq)]
pub struct Inputs {
    /// Hidden rows for the prompt tokens, then for the decode steps; `None` for a segment with
    /// the embedding.
    pub rows: Option<Vec<f32>>,
    /// The decode tokens; `None` means greedy (only a segment with embedding and head can).
    pub decode_tokens: Option<Vec<i32>>,
}

/// The input of the first segment `first` for `tokens` (the prompt) and `n_decode` decode steps.
pub fn resolve_inputs(
    args: &SegmentArgs,
    first: &SegmentSpec,
    n_embd: usize,
    n_vocab: usize,
    tokens: &[i32],
    n_decode: usize,
) -> Result<Inputs, String> {
    let file = match &args.hidden_in {
        Some(HiddenIn::File(p)) => {
            let f = HiddenFile::read(p)?;
            if f.n_embd != n_embd || f.n_prompt != tokens.len() || f.tokens[..f.n_prompt] != *tokens
            {
                return Err(format!(
                    "{}: rows are for another prompt or model width",
                    p.display()
                ));
            }
            if f.tokens.len() < tokens.len() + n_decode {
                return Err(format!(
                    "{}: {} decode rows, {n_decode} wanted",
                    p.display(),
                    f.tokens.len() - f.n_prompt
                ));
            }
            if !first.embed && f.layer != first.layers.start {
                return Err(format!(
                    "{}: rows enter layer {}, the span starts at {}",
                    p.display(),
                    f.layer,
                    first.layers.start
                ));
            }
            Some(f)
        }
        _ => None,
    };
    let n_rows = tokens.len() + n_decode;
    let rows = if first.embed {
        None
    } else {
        Some(match (&file, &args.hidden_in) {
            (Some(f), _) => f.rows[..n_rows * n_embd].to_vec(),
            (None, Some(HiddenIn::Seed(s))) => seeded_rows(*s, n_rows, n_embd),
            _ => return Err("a span without the embedding needs hidden input".into()),
        })
    };
    let decode_tokens = if first.embed && first.head {
        None
    } else {
        Some(match &file {
            Some(f) => f.tokens[tokens.len()..n_rows].to_vec(),
            None => pseudo_tokens(n_decode, n_vocab),
        })
    };
    Ok(Inputs {
        rows,
        decode_tokens,
    })
}

/// One request: the prompt in ubatches, then `n_decode` decode steps.
pub struct Workload<'a> {
    pub tokens: &'a [i32],
    pub ub: usize,
    pub n_decode: usize,
    /// `None`: greedy from the logits.
    pub decode_tokens: Option<&'a [i32]>,
    /// Hidden input rows for the prompt, then the decode steps.
    pub rows: Option<&'a [f32]>,
}

/// Counter snapshots of one request (first model only).
pub struct Snaps {
    pub start: ExpertStats,
    pub after_prefill: ExpertStats,
    pub end: ExpertStats,
}

/// The outcome of one request.
pub struct Request {
    pub prefill_s: f64,
    pub ubatch_s: Vec<f64>,
    pub decode_s: f64,
    /// The last output row after the prefill (logits, or a hidden row without the head).
    pub last: Vec<f32>,
    /// The last output row after the decode.
    pub last_decode: Vec<f32>,
    /// The tokens fed to the decode steps.
    pub decode_fed: Vec<i32>,
    /// The output rows of the tapped segment, for every prompt token then every decode token.
    pub tap: Vec<f32>,
    /// Only for a single segment (its counters are the whole run's).
    pub snaps: Option<Snaps>,
    pub prefill_profile: Option<StepProfile>,
    pub decode_profile: Option<StepProfile>,
}

impl Request {
    /// Bytes the prefill read from disk (0 when the counters are not available).
    pub fn prefill_disk_bytes(&self) -> u64 {
        self.snaps
            .as_ref()
            .map_or(0, |s| s.after_prefill.io.bytes_read - s.start.io.bytes_read)
    }
}

/// One step through the segments in order; each takes the previous one's hidden rows. The output
/// rows of segment `tap_at` are appended to `tap`. Returns the last segment's output.
fn step(
    models: &mut [&mut Qwen3Moe],
    tokens: &[i32],
    rows: Option<&[f32]>,
    tap_at: Option<usize>,
    tap: &mut Vec<f32>,
) -> Vec<f32> {
    let mut carried: Vec<f32> = Vec::new();
    for (i, m) in models.iter_mut().enumerate() {
        let mut out = Vec::new();
        let input = if i == 0 {
            rows
        } else {
            Some(carried.as_slice())
        };
        m.decode_hidden(
            &[SeqEval {
                seq: 0,
                tokens,
                all_logits: false,
            }],
            input,
            &mut out,
        )
        .expect("segment step");
        if tap_at == Some(i) {
            tap.extend_from_slice(&out);
        }
        carried = out;
    }
    carried
}

fn last_row(out: &[f32], width: usize) -> Vec<f32> {
    out[out.len() - width..].to_vec()
}

/// Runs one request from an empty sequence: every prompt token is computed (the harness has no
/// prefix reuse or prompt cache), then the decode steps. `tag` names the request in `log`;
/// `on_step` runs after every step (the routing recorder).
pub fn run_request(
    models: &mut [&mut Qwen3Moe],
    w: &Workload<'_>,
    tag: &str,
    tap_at: Option<usize>,
    log: &mut Vec<String>,
    on_step: &mut dyn FnMut(&mut [&mut Qwen3Moe]),
) -> Request {
    let n_embd = models[0].hparams().n_embd as usize;
    let width = models[models.len() - 1].out_width();
    let single = models.len() == 1;
    for m in models.iter_mut() {
        m.reset();
    }
    log.push(format!("{tag}:reset"));
    let start = if single {
        models[0].expert_stats()
    } else {
        None
    };
    if single {
        models[0].take_profile();
    }
    let mut tap = Vec::new();
    log.push(format!("{tag}:prefill"));
    let t0 = Instant::now();
    let mut ubatch_s = Vec::new();
    let mut last = Vec::new();
    let mut off = 0;
    for chunk in w.tokens.chunks(w.ub.max(1)) {
        let rows = w
            .rows
            .map(|r| &r[off * n_embd..(off + chunk.len()) * n_embd]);
        let t = Instant::now();
        let out = step(models, chunk, rows, tap_at, &mut tap);
        ubatch_s.push(t.elapsed().as_secs_f64());
        last = last_row(&out, width);
        off += chunk.len();
        on_step(models);
    }
    let prefill_s = t0.elapsed().as_secs_f64();
    let after_prefill = if single {
        models[0].expert_stats()
    } else {
        None
    };
    let prefill_profile = single.then(|| models[0].take_profile());

    log.push(format!("{tag}:decode"));
    let t0 = Instant::now();
    let mut next = if w.decode_tokens.is_none() {
        argmax(&last) as i32
    } else {
        0
    };
    let mut fed = Vec::new();
    let mut last_decode = last.clone();
    for i in 0..w.n_decode {
        let tok = w.decode_tokens.map_or(next, |t| t[i]);
        fed.push(tok);
        let rows = w
            .rows
            .map(|r| &r[(w.tokens.len() + i) * n_embd..(w.tokens.len() + i + 1) * n_embd]);
        let out = step(models, &[tok], rows, tap_at, &mut tap);
        last_decode = last_row(&out, width);
        if w.decode_tokens.is_none() {
            next = argmax(&last_decode) as i32;
        }
        on_step(models);
    }
    let decode_s = t0.elapsed().as_secs_f64();
    let end = if single {
        models[0].expert_stats()
    } else {
        None
    };
    let decode_profile = single.then(|| models[0].take_profile());
    let snaps = match (start, after_prefill, end) {
        (Some(start), Some(after_prefill), Some(end)) => Some(Snaps {
            start,
            after_prefill,
            end,
        }),
        _ => None,
    };
    Request {
        prefill_s,
        ubatch_s,
        decode_s,
        last,
        last_decode,
        decode_fed: fed,
        tap,
        snaps,
        prefill_profile,
        decode_profile,
    }
}

/// The hook that does nothing, for a request with no recorder.
pub fn no_hook(_: &mut [&mut Qwen3Moe]) {}

/// The warm part of the protocol (ADR 0092 section 3): `n` untimed full requests (prefill and
/// decode) of the same prompt; the routing they logged is dropped so a trace holds the timed
/// request only. The caller then runs the timed request, which clears the sequence first.
pub fn warm_requests(
    models: &mut [&mut Qwen3Moe],
    w: &Workload<'_>,
    n: usize,
    log: &mut Vec<String>,
) -> Vec<Request> {
    let mut out = Vec::new();
    for k in 0..n {
        out.push(run_request(
            models,
            w,
            &format!("warm{k}"),
            None,
            log,
            &mut no_hook,
        ));
    }
    if n > 0 {
        for m in models.iter_mut() {
            drop(m.take_routing_log());
        }
        log.push("discard_routing".into());
    }
    out
}

/// The warm requests and the timed one.
pub struct Protocol {
    pub warm: Vec<Request>,
    pub timed: Request,
}

pub fn run_protocol(
    models: &mut [&mut Qwen3Moe],
    w: &Workload<'_>,
    n_warm: usize,
    tap_at: Option<usize>,
    log: &mut Vec<String>,
    on_step: &mut dyn FnMut(&mut [&mut Qwen3Moe]),
) -> Protocol {
    let warm = warm_requests(models, w, n_warm, log);
    let timed = run_request(models, w, "timed", tap_at, log, on_step);
    Protocol { warm, timed }
}

/// Whether the compute and RAM slots of a layer hold every expert (`tier_slots` of the model).
pub fn covers(tier: Option<(usize, Option<usize>)>, n_expert: usize) -> bool {
    tier.is_some_and(|(v, r)| v + r.unwrap_or(0) >= n_expert)
}

/// A covering segment whose timed prefill read from disk is unclean (ADR 0092 section 3).
pub fn is_unclean(covering: bool, timed_prefill_bytes: u64) -> bool {
    covering && timed_prefill_bytes > 0
}

/// The `warm` block of a result: the protocol as run and the clean flag.
pub fn warm_summary(n_warm: usize, covering: bool, timed_prefill_bytes: u64) -> Value {
    json!({
        "requests": n_warm,
        "covering": covering,
        "timed_prefill_disk_bytes": timed_prefill_bytes,
        "unclean": is_unclean(covering, timed_prefill_bytes),
        // the harness drives the model directly: every prompt token is computed
        "prefix_reuse": false,
        "prompt_cache": false,
    })
}
