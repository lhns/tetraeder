//! Memory simulation of ADR 0067 step 1: the smallest pool, per layout, at which workload W
//! completes without preempting a live sequence, and the `n_kv` of every decode step.
//!
//! W: the 10 Tetraeder sessions of `bench/results/20261111d/claude-code-proxy` (232 requests;
//! prompt = input + cache-read tokens, output = output tokens) as 4 streams (sessions
//! round-robin by file name, a stream's requests back to back). A request continues the open
//! conversation of its session with the longest last prompt not longer than its own (the rule
//! of `scripts/claude_code_cache_classes.py`), else starts one. A finished request leaves its
//! sequence as an idle prefix; idle prefixes are evicted, oldest first, when rows are needed.
//!
//! Time is lockstep ticks: in a tick every stream with a request takes one step (the prefill of
//! its uncached prompt tokens when the request starts, else one decode token) and all of them
//! are one batch. A tick holding a decode token is a decode step. A request of `out` output
//! tokens takes one prefill and `out - 1` decode steps. Any failed reservation after every idle
//! prefix was evicted (admission or growth) fails the run: the stream cannot wait for rows
//! without leaving the 4-way concurrency the pool is sized for.
//!
//! Set `TETRAEDER_KV_SIM_OUT` to write `sim.json`.

use serde_json::{Value, json};
use std::collections::{BTreeMap, VecDeque};
use std::path::PathBuf;
use tetraeder_kvcache::{Chunked, GROW, KvError, KvPool, Paged, ROW_ALIGN, SeqId, StepInput};

const MAX_TOKENS: usize = 32000;
const MAX_POSITIONS: usize = 131072;
const STREAMS: usize = 4;
/// Qwen3-8B Q4_K_M weight bytes (`model_size`, bench/results/20261103/gateway/overhead-q8b-llama-bench.json).
const WT: f64 = 5_021_827_072.0;
/// K plus V bytes of one row over all layers, Qwen3-8B F16: 36 layers x 2 x 1024 x 2 bytes.
const R: f64 = 147_456.0;
/// Threshold X of ADR 0067.
const X: f64 = 0.25;

#[derive(Clone, Copy, Debug)]
enum Kind {
    Chunked,
    Paged(usize),
}

impl Kind {
    fn pool(self, units: usize) -> KvPool {
        let rows = units * ROW_ALIGN;
        match self {
            Kind::Chunked => KvPool::new(Box::new(Chunked::new(rows))),
            Kind::Paged(b) => KvPool::new(Box::new(Paged::new(rows, b))),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Request {
    prompt: usize,
    out: usize,
}

/// Per stream: (session, request) in order.
type Streams = Vec<Vec<(usize, Request)>>;

fn streams_of(sessions: Vec<Vec<Request>>) -> Streams {
    let mut streams: Streams = vec![Vec::new(); STREAMS];
    for (i, s) in sessions.into_iter().enumerate() {
        streams[i % STREAMS].extend(s.into_iter().map(|r| (i, r)));
    }
    streams
}

fn proxy_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/results/20261111d/claude-code-proxy")
}

fn workload_w() -> Streams {
    let mut files: Vec<PathBuf> = std::fs::read_dir(proxy_dir())
        .expect("proxy log directory")
        .map(|e| e.unwrap().path())
        .filter(|p| {
            let n = p.file_name().unwrap().to_string_lossy().into_owned();
            n.starts_with("proxy-tetraeder-r1-") && n.ends_with(".jsonl")
        })
        .collect();
    files.sort();
    let sessions = files
        .iter()
        .map(|f| {
            std::fs::read_to_string(f)
                .unwrap()
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| serde_json::from_str::<Value>(l).unwrap())
                .filter(|r| r["status"] == 200 && !r["input_tokens"].is_null())
                .map(|r| Request {
                    prompt: (r["input_tokens"].as_u64().unwrap()
                        + r["cache_read_tokens"].as_u64().unwrap_or(0))
                        as usize,
                    out: r["output_tokens"].as_u64().unwrap() as usize,
                })
                .collect()
        })
        .collect();
    streams_of(sessions)
}

/// The c4 bench: 16 requests of 512 prompt tokens and 128 output tokens, concurrency 4; each
/// request is its own conversation.
fn workload_c4() -> Streams {
    streams_of(
        (0..16)
            .map(|_| {
                vec![Request {
                    prompt: 512,
                    out: 128,
                }]
            })
            .collect(),
    )
}

struct Conv {
    last_prompt: usize,
    seq: Option<SeqId>,
}

#[derive(Default, Debug)]
struct Outcome {
    ok: bool,
    /// n_kv of every decode step.
    n_kv: Vec<usize>,
    /// Sequence length before the token, per decode input, in bins of ROW_ALIGN.
    len_hist: BTreeMap<usize, u64>,
    relocations: u64,
    moved_rows: u64,
    evictions: u64,
}

struct Stream {
    reqs: Vec<(usize, Request)>,
    next: usize,
    /// Decode steps left of the open request, if any.
    left: Option<usize>,
    session: usize,
    conv: usize,
    seq: SeqId,
    /// Admitted in this tick: it prefills, so it does not also decode.
    fresh: bool,
}

struct Sim {
    pool: KvPool,
    kind: Kind,
    /// Keep a finished request's sequence as an idle prefix (else free it at once).
    keep_idle: bool,
    max_tokens: usize,
    convs: Vec<Vec<Conv>>,
    /// Oldest first: (sequence, session, conversation).
    idle: VecDeque<(SeqId, usize, usize)>,
    next_seq: SeqId,
    out: Outcome,
}

impl Sim {
    /// Reserves, evicting idle prefixes oldest first until it fits.
    fn reserve(&mut self, seq: SeqId, n: usize) -> bool {
        loop {
            match self.pool.reserve(seq, n) {
                Ok(copies) => {
                    if !copies.is_empty() {
                        self.out.relocations += 1;
                        self.out.moved_rows += copies.iter().map(|c| c.n as u64).sum::<u64>();
                    }
                    return true;
                }
                Err(KvError::Full { .. }) => {
                    let Some((victim, s, c)) = self.idle.pop_front() else {
                        return false;
                    };
                    self.pool.seq_rm(victim, 0);
                    self.convs[s][c].seq = None;
                    self.out.evictions += 1;
                }
                Err(e) => panic!("{e}"),
            }
        }
    }

    /// Starts the stream's request: conversation, sequence, reservation. Returns the prefill size.
    fn admit(&mut self, st: &mut Stream) -> Option<usize> {
        let (session, r) = st.reqs[st.next];
        let pick = self.convs[session]
            .iter()
            .enumerate()
            .filter(|(_, c)| c.last_prompt <= r.prompt)
            .max_by_key(|(_, c)| c.last_prompt)
            .map(|(i, _)| i);
        let ci = pick.unwrap_or_else(|| {
            self.convs[session].push(Conv {
                last_prompt: 0,
                seq: None,
            });
            self.convs[session].len() - 1
        });
        let (seq, keep) = match self.convs[session][ci].seq {
            Some(seq) => {
                let at = self.idle.iter().position(|e| e.0 == seq).unwrap();
                self.idle.remove(at);
                let keep = self.convs[session][ci]
                    .last_prompt
                    .min(self.pool.seq_len(seq))
                    .min(r.prompt - 1);
                self.pool.seq_rm(seq, keep);
                (seq, keep)
            }
            None => {
                self.next_seq += 1;
                (self.next_seq, 0)
            }
        };
        self.convs[session][ci].seq = Some(seq);
        self.convs[session][ci].last_prompt = r.prompt;
        (st.session, st.conv, st.seq) = (session, ci, seq);
        let want = match self.kind {
            Kind::Chunked => r.prompt + self.max_tokens.min(GROW),
            Kind::Paged(_) => r.prompt,
        };
        self.reserve(seq, want).then_some(r.prompt - keep)
    }

    fn finish(&mut self, st: &Stream) {
        if !self.keep_idle {
            self.pool.seq_rm(st.seq, 0);
            self.convs[st.session][st.conv].seq = None;
            return;
        }
        // Release the reservation beyond the positions held; the prefix stays as an idle one.
        self.pool.seq_rm(st.seq, usize::MAX);
        self.idle.push_back((st.seq, st.session, st.conv));
    }
}

/// Runs `streams` on a pool of `units` x 256 rows.
fn run(streams: &Streams, kind: Kind, units: usize, max_tokens: usize, keep_idle: bool) -> Outcome {
    let n_sessions = streams
        .iter()
        .flatten()
        .map(|(s, _)| s + 1)
        .max()
        .unwrap_or(0);
    let mut sim = Sim {
        pool: kind.pool(units),
        kind,
        keep_idle,
        max_tokens,
        convs: (0..n_sessions).map(|_| Vec::new()).collect(),
        idle: VecDeque::new(),
        next_seq: 0,
        out: Outcome::default(),
    };
    let mut live: Vec<Stream> = streams
        .iter()
        .map(|reqs| Stream {
            reqs: reqs.clone(),
            next: 0,
            left: None,
            session: 0,
            conv: 0,
            seq: 0,
            fresh: false,
        })
        .collect();
    loop {
        let mut inputs: Vec<StepInput> = Vec::new();
        let mut who: Vec<usize> = Vec::new();
        let mut decodes = 0;
        for (i, st) in live.iter_mut().enumerate() {
            st.fresh = false;
            if st.left.is_none() {
                if st.next >= st.reqs.len() {
                    continue;
                }
                let r = st.reqs[st.next].1;
                assert!(r.prompt + r.out <= MAX_POSITIONS);
                let Some(n) = sim.admit(st) else {
                    return sim.out;
                };
                st.left = Some(r.out.saturating_sub(1));
                st.fresh = true;
                inputs.push(StepInput {
                    seq: st.seq,
                    n_tokens: n,
                    all_logits: false,
                });
            } else {
                let need = sim.pool.seq_len(st.seq) + 1;
                if !sim.reserve(st.seq, need) {
                    return sim.out;
                }
                let bin = sim.pool.seq_len(st.seq) / ROW_ALIGN * ROW_ALIGN;
                *sim.out.len_hist.entry(bin).or_default() += 1;
                decodes += 1;
                inputs.push(StepInput {
                    seq: st.seq,
                    n_tokens: 1,
                    all_logits: false,
                });
            }
            who.push(i);
        }
        if inputs.is_empty() {
            break;
        }
        let n_kv = sim
            .pool
            .step_n_kv(&inputs)
            .expect("reservations cover the step");
        if decodes > 0 {
            sim.out.n_kv.push(n_kv);
        }
        sim.pool.commit(&inputs);
        for i in who {
            let st = &mut live[i];
            if !st.fresh {
                *st.left.as_mut().unwrap() -= 1;
            }
            if st.left == Some(0) {
                st.left = None;
                st.next += 1;
                sim.finish(st);
            }
        }
    }
    sim.out.ok = true;
    sim.out
}

/// Smallest pool in ROW_ALIGN units at which `streams` complete, by bisection.
fn min_units(streams: &Streams, kind: Kind, max_tokens: usize, keep_idle: bool) -> usize {
    let (mut lo, mut hi) = (0, 16);
    while !run(streams, kind, hi, max_tokens, keep_idle).ok {
        lo = hi;
        hi *= 2;
        assert!(hi <= 1 << 14, "no pool completes the workload");
    }
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if run(streams, kind, mid, max_tokens, keep_idle).ok {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

#[test]
fn c4_sanity_needs_the_hand_computed_pool() {
    // Four sequences of 512 + 128 positions: 640 rows, 768 with alignment, 3072 rows.
    let w = workload_c4();
    for kind in [Kind::Chunked, Kind::Paged(256)] {
        assert!(run(&w, kind, 12, 128, true).ok, "{kind:?} at 3072 rows");
        assert!(!run(&w, kind, 11, 128, true).ok, "{kind:?} at 2816 rows");
        assert_eq!(min_units(&w, kind, 128, true), 12, "{kind:?}");
    }
}

fn y_mask(paged: &Outcome, chunked: &Outcome) -> f64 {
    assert_eq!(
        paged.n_kv.len(),
        chunked.n_kv.len(),
        "same steps under both layouts"
    );
    let extra: f64 = paged
        .n_kv
        .iter()
        .zip(&chunked.n_kv)
        .map(|(&p, &c)| (p as f64 - c as f64) * R)
        .sum();
    let step: f64 = chunked.n_kv.iter().map(|&c| WT + c as f64 * R).sum();
    extra / step
}

fn mean(v: &[usize]) -> f64 {
    v.iter().sum::<usize>() as f64 / v.len() as f64
}

#[test]
fn workload_w_pool_sizes() {
    let w = workload_w();
    assert_eq!(w.iter().map(Vec::len).sum::<usize>(), 232);

    let units_c = min_units(&w, Kind::Chunked, MAX_TOKENS, true);
    let units_p = min_units(&w, Kind::Paged(256), MAX_TOKENS, true);
    let units_p64 = min_units(&w, Kind::Paged(64), MAX_TOKENS, true);
    for (kind, units) in [
        (Kind::Chunked, units_c),
        (Kind::Paged(256), units_p),
        (Kind::Paged(64), units_p64),
    ] {
        assert!(
            run(&w, kind, units, MAX_TOKENS, true).ok,
            "{kind:?} completes at its pool"
        );
        assert!(
            !run(&w, kind, units - 1, MAX_TOKENS, true).ok,
            "{kind:?} fails one unit below"
        );
    }

    let chunked = run(&w, Kind::Chunked, units_c, MAX_TOKENS, true);
    let paged = run(&w, Kind::Paged(256), units_p, MAX_TOKENS, true);
    let paged_at_c = run(&w, Kind::Paged(256), units_c, MAX_TOKENS, true);
    assert!(paged_at_c.ok);
    let saving = 1.0 - units_p as f64 / units_c as f64;
    // Robustness: no idle prefixes at all (a request frees its sequence when it finishes).
    let saving_no_idle = 1.0
        - min_units(&w, Kind::Paged(256), MAX_TOKENS, false) as f64
            / min_units(&w, Kind::Chunked, MAX_TOKENS, false) as f64;
    let y = y_mask(&paged, &chunked);
    let y_same_pool = y_mask(&paged_at_c, &chunked);
    eprintln!(
        "pool units: chunked {units_c}, paged_256 {units_p}, paged_64 {units_p64}; saving {saving:.4} \
         (X = {X}; without idle prefixes {saving_no_idle:.4}); y_mask {y:.5} (paged at the chunked pool: {y_same_pool:.5}); {} decode steps; \
         chunked relocations {} ({} rows), evictions chunked {} paged {}",
        chunked.n_kv.len(),
        chunked.relocations,
        chunked.moved_rows,
        chunked.evictions,
        paged.evictions,
    );
    if let Some(path) = std::env::var_os("TETRAEDER_KV_SIM_OUT") {
        let doc = json!({
            "workload": {
                "source": "bench/results/20261111d/claude-code-proxy/proxy-tetraeder-r1-*.jsonl",
                "requests": 232,
                "streams": STREAMS,
                "max_tokens": MAX_TOKENS,
                "max_positions": MAX_POSITIONS,
                "row_bytes": R,
                "weight_bytes": WT,
                "unit_rows": ROW_ALIGN,
            },
            "pool_rows": {
                "chunked": units_c * ROW_ALIGN,
                "paged_256": units_p * ROW_ALIGN,
                "paged_64": units_p64 * ROW_ALIGN,
            },
            "saving": saving,
            "saving_without_idle_prefixes": saving_no_idle,
            "y_mask": y,
            "y_mask_at_chunked_pool": y_same_pool,
            "relocations": {
                "count": chunked.relocations,
                "rows": chunked.moved_rows,
            },
            "evictions": {
                "chunked": chunked.evictions,
                "paged_256": paged.evictions,
            },
            "steps": chunked.n_kv.len(),
            "mean_n_kv": {
                "chunked": mean(&chunked.n_kv),
                "paged_256": mean(&paged.n_kv),
            },
            "decode_len_hist": chunked
                .len_hist
                .iter()
                .map(|(k, v)| (k.to_string(), json!(v)))
                .collect::<serde_json::Map<_, _>>(),
        });
        std::fs::write(path, serde_json::to_string_pretty(&doc).unwrap() + "\n").unwrap();
    }
}
