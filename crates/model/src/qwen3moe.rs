//! Qwen3 MoE graph builder (ADR 0030), op for op like libllama's `qwen3moe` graph.
//!
//! A step runs as `n_layer + 1` graph segments. Segment `l` finishes layer `l - 1` (expert FFN
//! with the host-mapped expert ids, residual) and starts layer `l` (attention, router, top-k);
//! it ends at the top-k. The host reads the ids, maps experts to slots (identity for now,
//! [`map_experts`]) and passes the mapped ids to the next segment through a state tensor.
//! Activations between segments stay in backend memory, in two alternating state slots.
//!
//! With fewer expert slots than experts (ADR 0033) the expert tensors are slot tensors; between
//! segments the [`Streamer`] makes the routed experts resident and the host writes slot ids.

use crate::experts::{ExpertStats, RoutingStep, Streamer};
use crate::gguf::GgufError;
use crate::hparams::{Hparams, ModelFile, names};
use tetraeder_ggml_sys::ggml::{
    Backend, Context, GgmlError, Graph, OwnedTensor, RopeParams, Sched, Tensor, Type,
};
use tetraeder_memory::IoOptions;
use tetraeder_memory::metrics::MemoryMetrics;
use tetraeder_plan::ModelDesc;
use tetraeder_telemetry::{Phase, StepSample, Telemetry};

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error(transparent)]
    Gguf(#[from] GgufError),
    #[error(transparent)]
    Ggml(#[from] GgmlError),
    #[error("cannot read weights: {0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, ModelError>;

/// Where the weights, the KV cache and the compute run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendChoice {
    /// CPU with this many threads (`None`: the available parallelism).
    Cpu { threads: Option<usize> },
    /// The `index`-th GPU (CUDA or Vulkan, whatever the build has); the CPU is the fallback.
    Gpu { index: usize },
}

#[derive(Clone)]
pub struct LoadOptions {
    pub backend: BackendChoice,
    /// KV cache length in tokens (rounded up to a multiple of 256).
    pub n_ctx: usize,
    /// Expert slots per MoE layer on the compute backend (ADR 0033). `None`, or at least the
    /// expert count: every expert is loaded at load time and stays resident. Fewer: experts
    /// are read from the GGUF on demand.
    pub expert_slots: Option<usize>,
    /// Host-memory expert slots per layer between the file and the compute slots. Only with
    /// fewer compute slots than experts.
    pub ram_slots: Option<usize>,
    /// Direct I/O settings for expert reads.
    pub io: IoOptions,
    /// Step and expert cache metrics go here when set.
    pub telemetry: Option<Arc<Telemetry>>,
    /// Keep the router's choices (see [`Qwen3Moe::take_routing_log`]); off by default.
    pub record_routing: bool,
}

impl std::fmt::Debug for LoadOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadOptions")
            .field("backend", &self.backend)
            .field("n_ctx", &self.n_ctx)
            .field("expert_slots", &self.expert_slots)
            .field("ram_slots", &self.ram_slots)
            .field("telemetry", &self.telemetry.is_some())
            .finish_non_exhaustive()
    }
}

impl LoadOptions {
    pub fn new(backend: BackendChoice) -> Self {
        Self {
            backend,
            n_ctx: 2048,
            expert_slots: None,
            ram_slots: None,
            io: IoOptions::default(),
            telemetry: None,
            record_routing: false,
        }
    }
}

const KV_PAD: usize = 256;
const MASK_PAD: usize = 32;
/// Upper bound of nodes in one segment; the scheduler's hash tables are sized for it.
const SEGMENT_NODES: usize = 1024;
const F16_NEG_INF: u16 = 0xFC00;

fn pad(n: usize, to: usize) -> usize {
    n.div_ceil(to) * to
}

struct Layer {
    attn_norm: OwnedTensor,
    wq: OwnedTensor,
    wk: OwnedTensor,
    wv: OwnedTensor,
    wo: OwnedTensor,
    q_norm: OwnedTensor,
    k_norm: OwnedTensor,
    ffn_norm: OwnedTensor,
    gate_inp: OwnedTensor,
    /// Stacked expert weights, or slot tensors with fewer entries when streaming.
    gate_exps: OwnedTensor,
    up_exps: OwnedTensor,
    down_exps: OwnedTensor,
}

impl Layer {
    fn exps(&self) -> [Tensor<'_>; 3] {
        [
            self.gate_exps.get(),
            self.up_exps.get(),
            self.down_exps.get(),
        ]
    }
}

struct Weights {
    tok_embd: OwnedTensor,
    out_norm: OwnedTensor,
    /// `None` when the output projection is tied to the token embedding.
    output: Option<OwnedTensor>,
    layers: Vec<Layer>,
}

/// Contiguous per-layer F16 KV cache: `[n_embd_kv, n_ctx]` for K and for V.
struct Kv {
    k: Vec<OwnedTensor>,
    v: Vec<OwnedTensor>,
    n_ctx: usize,
}

/// Activations that cross segment boundaries, one slot per layer parity.
struct Slot {
    resid: OwnedTensor,
    xn: OwnedTensor,
    /// Normalized router weights `[1, n_used, n]`.
    w: OwnedTensor,
    /// Top-k expert ids as the router chose them `[n_used, n]`, read by the host.
    ids: OwnedTensor,
    /// The ids the expert matmuls use, written by the host.
    ids_map: OwnedTensor,
    /// Sum of the expert FFN passes when a layer step needed more than one (ADR 0033).
    acc: OwnedTensor,
}

/// Per-step inputs and slots; reallocated when the token count or the padded KV length changes.
struct State {
    n_tokens: usize,
    n_kv: usize,
    tokens: OwnedTensor,
    pos: OwnedTensor,
    mask: OwnedTensor,
    slots: [Slot; 2],
}

/// A Qwen3 MoE model. The experts are all resident in backend memory or streamed through slots.
pub struct Qwen3Moe {
    file: ModelFile,
    sched: Sched,
    primary: Backend,
    weights: Weights,
    kv: Kv,
    state: Option<State>,
    n_past: usize,
    segments_run: u64,
    passes_run: u64,
    streamer: Option<Streamer>,
    metrics: MemoryMetrics,
    telemetry: Option<Arc<Telemetry>>,
    routing: Option<Vec<RoutingStep>>,
}

impl std::fmt::Debug for Qwen3Moe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3Moe")
            .field("hparams", &self.file.hparams)
            .field("backend", &self.primary)
            .field("n_past", &self.n_past)
            .finish()
    }
}

fn f16_bits_tensor_bytes(bits: &[u16]) -> Vec<u8> {
    bits.iter().flat_map(|b| b.to_ne_bytes()).collect()
}

impl Qwen3Moe {
    pub fn load(path: impl AsRef<Path>, opts: &LoadOptions) -> Result<Qwen3Moe> {
        let file = ModelFile::open(path.as_ref())?;
        let h = &file.hparams;
        if h.n_head % h.n_head_kv != 0 || h.n_expert_used > h.n_expert || h.n_rot != h.head_dim {
            return Err(ModelError::Invalid("unsupported hyperparameters".into()));
        }
        let (primary, backends) = match opts.backend {
            BackendChoice::Cpu { threads } => {
                let n = threads
                    .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from));
                let cpu = Backend::cpu(n as i32)?;
                (cpu.clone(), vec![cpu])
            }
            BackendChoice::Gpu { index } => {
                let gpu = Backend::gpu(index)?;
                let n = std::thread::available_parallelism().map_or(1, usize::from);
                (gpu.clone(), vec![gpu, Backend::cpu(n as i32)?])
            }
        };
        let sched = Sched::new(&backends, SEGMENT_NODES * 4, false)?;
        let slots = match opts.expert_slots {
            Some(0) => {
                return Err(ModelError::Invalid(
                    "expert_slots must be at least 1".into(),
                ));
            }
            Some(n) if n < h.n_expert as usize => Some(n),
            _ => None,
        };
        let weights = load_weights(&file, &primary, slots)?;
        let kv = alloc_kv(h, pad(opts.n_ctx.max(1), KV_PAD), &primary)?;
        let metrics = MemoryMetrics::new(opts.telemetry.clone());
        let streamer = slots
            .map(|n| {
                let tensors: Vec<_> = weights.layers.iter().map(Layer::exps).collect();
                Streamer::new(
                    &file,
                    &tensors,
                    n,
                    opts.ram_slots.map(|r| r.max(1)),
                    opts.io,
                    metrics.clone(),
                    primary.name().to_string(),
                )
            })
            .transpose()?;
        tracing::debug!(
            backend = primary.name(),
            n_layer = h.n_layer,
            n_expert = h.n_expert,
            "qwen3moe loaded"
        );
        Ok(Qwen3Moe {
            file,
            sched,
            primary,
            weights,
            kv,
            state: None,
            n_past: 0,
            segments_run: 0,
            passes_run: 0,
            streamer,
            metrics,
            telemetry: opts.telemetry.clone(),
            routing: opts.record_routing.then(Vec::new),
        })
    }

    pub fn hparams(&self) -> &Hparams {
        &self.file.hparams
    }

    pub fn file(&self) -> &ModelFile {
        &self.file
    }

    /// The placement description (units and sizes) of this model.
    pub fn model_desc(&self) -> Result<ModelDesc> {
        Ok(self.file.model_desc()?)
    }

    pub fn n_vocab(&self) -> usize {
        self.file.hparams.n_vocab as usize
    }

    /// Tokens in the KV cache.
    pub fn n_past(&self) -> usize {
        self.n_past
    }

    pub fn n_ctx(&self) -> usize {
        self.kv.n_ctx
    }

    /// Graph segments computed so far (`n_layer + 1` per step).
    pub fn segments_run(&self) -> u64 {
        self.segments_run
    }

    /// Extra expert FFN graphs run because a layer step needed more distinct experts than
    /// there are slots (ADR 0033).
    pub fn overflow_passes(&self) -> u64 {
        self.passes_run
    }

    /// Expert cache and read counters; `None` when every expert is resident.
    pub fn expert_stats(&self) -> Option<ExpertStats> {
        self.streamer.as_ref().map(Streamer::stats)
    }

    /// The routing recorded since the last call (empty unless `record_routing` is set), one
    /// entry per layer step in execution order.
    pub fn take_routing_log(&mut self) -> Vec<RoutingStep> {
        self.routing
            .as_mut()
            .map(std::mem::take)
            .unwrap_or_default()
    }

    /// Forgets the KV cache; the next `eval` starts at position 0.
    pub fn reset(&mut self) {
        self.n_past = 0;
    }

    /// Runs `tokens` after the tokens already in the cache and returns the logits of every
    /// position, `tokens.len() * n_vocab` values, position-major.
    pub fn eval(&mut self, tokens: &[i32]) -> Result<Vec<f32>> {
        self.run(tokens, false)
    }

    /// Like [`Qwen3Moe::eval`] but only for the last position (`n_vocab` values): prefill does
    /// not need the head for the other positions.
    pub fn eval_last(&mut self, tokens: &[i32]) -> Result<Vec<f32>> {
        self.run(tokens, true)
    }

    fn run(&mut self, tokens: &[i32], last_only: bool) -> Result<Vec<f32>> {
        let t0 = Instant::now();
        let h = self.file.hparams.clone();
        let n = tokens.len();
        if n == 0 {
            return Err(ModelError::Invalid("no tokens".into()));
        }
        if self.n_past + n > self.kv.n_ctx {
            return Err(ModelError::Invalid(format!(
                "context full: {} cached + {n} new > {}",
                self.n_past, self.kv.n_ctx
            )));
        }
        if let Some(&t) = tokens.iter().find(|&&t| t < 0 || t as u32 >= h.n_vocab) {
            return Err(ModelError::Invalid(format!(
                "token {t} outside the vocabulary"
            )));
        }
        let n_kv = pad(self.n_past + n, KV_PAD);
        if self
            .state
            .as_ref()
            .is_none_or(|s| s.n_tokens != n || s.n_kv != n_kv)
        {
            self.state = None; // free the old buffers first
            self.state = Some(alloc_state(&h, n, n_kv, &self.primary)?);
        }
        let st = self.state.as_ref().expect("state allocated above");
        st.tokens.get().set_i32(tokens)?;
        let pos: Vec<i32> = (self.n_past..self.n_past + n).map(|p| p as i32).collect();
        st.pos.get().set_i32(&pos)?;
        let mut mask = vec![0u16; n_kv * pad(n, MASK_PAD)];
        for i in 0..n {
            for j in (self.n_past + i + 1)..n_kv {
                mask[i * n_kv + j] = F16_NEG_INF;
            }
        }
        st.mask.get().set_bytes(&f16_bits_tensor_bytes(&mask), 0)?;

        let mut mapped = Vec::new();
        let mut logits = Vec::new();
        // The previous layer step ran in several passes and left the FFN sum in `acc`.
        let mut from_acc = false;
        for seg in 0..=h.n_layer as usize {
            let ctx = Context::new(SEGMENT_NODES, SEGMENT_NODES)?;
            let (graph, out) = build_segment(
                &h,
                &self.weights,
                &self.kv,
                st,
                &ctx,
                seg,
                self.n_past,
                last_only,
                from_acc,
            );
            from_acc = false;
            self.sched.alloc_graph(&graph)?;
            self.sched.compute(&graph)?;
            self.segments_run += 1;
            if let Some(l) = out {
                logits = l.get_f32()?;
            }
            if seg < h.n_layer as usize {
                let slot = &st.slots[seg % 2];
                let chosen = slot.ids.get().get_i32()?;
                if let Some(log) = &mut self.routing {
                    log.push(RoutingStep {
                        layer: seg,
                        ids: chosen.clone(),
                    });
                }
                let Some(streamer) = &mut self.streamer else {
                    // All experts resident: the router's ids are the tensor rows.
                    self.metrics.layer_step(true);
                    slot.ids_map.get().set_i32(&chosen)?;
                    continue;
                };
                let layer = &self.weights.layers[seg];
                let chunks = streamer.plan(seg, &chosen)?;
                if let [chunk] = chunks.as_slice() {
                    let slots = streamer.make_resident(&self.file, seg, &layer.exps(), chunk)?;
                    map_chunk(&chosen, chunk, &slots, &mut mapped, None);
                    slot.ids_map.get().set_i32(&mapped)?;
                    continue;
                }
                // More distinct experts than slots: one FFN pass per chunk, summed (ADR 0033).
                let weights = slot.w.get().get_f32()?;
                let mut masked = Vec::new();
                for (pass, chunk) in chunks.iter().enumerate() {
                    let slots = streamer.make_resident(&self.file, seg, &layer.exps(), chunk)?;
                    masked.clone_from(&weights);
                    map_chunk(&chosen, chunk, &slots, &mut mapped, Some(&mut masked));
                    slot.ids_map.get().set_i32(&mapped)?;
                    slot.w.get().set_f32(&masked)?;
                    let ctx = Context::new(SEGMENT_NODES, SEGMENT_NODES)?;
                    let graph = ffn_pass(&h, &ctx, layer, slot, pass == 0);
                    self.sched.alloc_graph(&graph)?;
                    self.sched.compute(&graph)?;
                    self.passes_run += 1;
                }
                from_acc = true;
            }
        }
        self.n_past += n;
        if let Some(t) = &self.telemetry {
            t.record_step(StepSample {
                phase: if n > 1 { Phase::Prefill } else { Phase::Decode },
                stage: 0,
                graphs: false,
                duration: t0.elapsed(),
                tokens: n as u32,
                segments: h.n_layer + 1,
            });
        }
        Ok(logits)
    }
}

/// Writes into `mapped` the slot of each chosen expert. Experts outside `chunk` (they belong to
/// another pass) point at the first slot of the chunk and, in `masked`, get router weight 0 so
/// they add nothing.
fn map_chunk(
    chosen: &[i32],
    chunk: &[u32],
    slots: &[u32],
    mapped: &mut Vec<i32>,
    mut masked: Option<&mut Vec<f32>>,
) {
    let slot_of = |e: i32| chunk.iter().position(|&c| c as i32 == e).map(|i| slots[i]);
    mapped.clear();
    for (i, &e) in chosen.iter().enumerate() {
        match slot_of(e) {
            Some(s) => mapped.push(s as i32),
            None => {
                mapped.push(slots[0] as i32);
                if let Some(m) = masked.as_deref_mut() {
                    m[i] = 0.0;
                }
            }
        }
    }
}

/// With `slots`, the expert tensors have that many entries and are left empty (zeros) for the
/// streamer to fill; everything else is read from the file.
fn load_weights(file: &ModelFile, backend: &Backend, slots: Option<usize>) -> Result<Weights> {
    let h = &file.hparams;
    let n_tensors = 3 + h.n_layer as usize * 12;
    let ctx = Rc::new(Context::new(n_tensors + 4, 0)?);
    let mut loaded: Vec<(OwnedTensor, u64, u64)> = Vec::new();
    let mut make = |name: &str| -> Result<OwnedTensor> {
        let info = file.gguf.tensor(name)?;
        let mut ne: Vec<i64> = info.ne.iter().map(|&x| x as i64).collect();
        let streamed = slots.filter(|_| name.contains("_exps."));
        if let Some(n) = streamed {
            ne[2] = n as i64;
        }
        let nd = ne.iter().rposition(|&x| x != 1).map_or(1, |i| i + 1);
        let t = ctx.new_persistent(info.ty, &ne[..nd]);
        t.get().set_name(name);
        if streamed.is_none() {
            if t.get().nbytes() as u64 != info.nbytes {
                return Err(ModelError::Invalid(format!("size mismatch for `{name}`")));
            }
            loaded.push((t.clone(), info.offset, info.nbytes));
        }
        Ok(t)
    };
    let tok_embd = make(names::TOK_EMBD)?;
    let out_norm = make(names::OUTPUT_NORM)?;
    let output = if h.tied_output {
        None
    } else {
        Some(make(names::OUTPUT)?)
    };
    let mut layers = Vec::new();
    for il in 0..h.n_layer {
        let mut l = |w: &str| make(&names::layer(il, w));
        layers.push(Layer {
            attn_norm: l("attn_norm")?,
            wq: l("attn_q")?,
            wk: l("attn_k")?,
            wv: l("attn_v")?,
            wo: l("attn_output")?,
            q_norm: l("attn_q_norm")?,
            k_norm: l("attn_k_norm")?,
            ffn_norm: l("ffn_norm")?,
            gate_inp: l("ffn_gate_inp")?,
            gate_exps: l("ffn_gate_exps")?,
            up_exps: l("ffn_up_exps")?,
            down_exps: l("ffn_down_exps")?,
        });
    }
    ctx.alloc_tensors(backend)?;
    if slots.is_some() {
        // A slot that is never loaded must still hold finite weights: an overflow pass runs
        // the matmuls of unused slots with router weight 0, and 0 * NaN is NaN.
        ctx.clear_buffer(0);
    }
    let mut f = std::fs::File::open(&file.path)?;
    let mut buf = vec![0u8; 16 << 20];
    for (t, offset, nbytes) in loaded {
        f.seek(SeekFrom::Start(offset))?;
        let mut done = 0u64;
        while done < nbytes {
            let len = ((nbytes - done) as usize).min(buf.len());
            f.read_exact(&mut buf[..len])?;
            t.get().set_bytes(&buf[..len], done as usize)?;
            done += len as u64;
        }
    }
    Ok(Weights {
        tok_embd,
        out_norm,
        output,
        layers,
    })
}

fn alloc_kv(h: &Hparams, n_ctx: usize, backend: &Backend) -> Result<Kv> {
    let ctx = Rc::new(Context::new(2 * h.n_layer as usize + 4, 0)?);
    let n_embd_kv = (h.n_head_kv * h.head_dim) as i64;
    let mk = |kind: &str, il: u32| {
        let t = ctx.new_persistent(Type::F16, &[n_embd_kv, n_ctx as i64]);
        t.get().set_name(&format!("{kind}_cache_{il}"));
        t
    };
    let k = (0..h.n_layer).map(|il| mk("k", il)).collect();
    let v = (0..h.n_layer).map(|il| mk("v", il)).collect();
    ctx.alloc_tensors(backend)?;
    // Padding rows are masked, but 0 * NaN is NaN: never leave them uninitialized.
    ctx.clear_buffer(0);
    Ok(Kv { k, v, n_ctx })
}

fn alloc_state(h: &Hparams, n: usize, n_kv: usize, backend: &Backend) -> Result<State> {
    let ctx = Rc::new(Context::new(32, 0)?);
    let (ne, nu, nt) = (h.n_embd as i64, h.n_expert_used as i64, n as i64);
    let slot = |i: usize| {
        let t = |ty, dims: &[i64], what: &str| {
            let t = ctx.new_persistent(ty, dims);
            t.get().set_name(&format!("{what}_{i}"));
            t
        };
        Slot {
            resid: t(Type::F32, &[ne, nt], "resid"),
            xn: t(Type::F32, &[ne, nt], "xn"),
            w: t(Type::F32, &[1, nu, nt], "weights"),
            ids: t(Type::I32, &[nu, nt], "ids"),
            ids_map: t(Type::I32, &[nu, nt], "ids_mapped"),
            acc: t(Type::F32, &[ne, nt], "moe_acc"),
        }
    };
    let state = State {
        n_tokens: n,
        n_kv,
        tokens: ctx.new_persistent(Type::I32, &[nt]),
        pos: ctx.new_persistent(Type::I32, &[nt]),
        mask: ctx.new_persistent(Type::F16, &[n_kv as i64, pad(n, MASK_PAD) as i64]),
        slots: [slot(0), slot(1)],
    };
    ctx.alloc_tensors(backend)?;
    Ok(state)
}

/// Builds segment `seg`. Returns the graph and, for the last segment, the logits tensor.
#[allow(clippy::too_many_arguments)]
fn build_segment<'a>(
    h: &Hparams,
    w: &'a Weights,
    kv: &'a Kv,
    st: &'a State,
    ctx: &'a Context,
    seg: usize,
    n_past: usize,
    last_only: bool,
    from_acc: bool,
) -> (Graph<'a>, Option<Tensor<'a>>) {
    let mut g = ctx.new_graph(SEGMENT_NODES);
    let n_layer = h.n_layer as usize;

    let x = if seg == 0 {
        ctx.get_rows(w.tok_embd.get(), st.tokens.get())
    } else {
        let s = &st.slots[(seg - 1) % 2];
        let moe = if from_acc {
            s.acc.get()
        } else {
            moe_ffn(h, ctx, &w.layers[seg - 1], s)
        };
        ctx.add(moe, s.resid.get())
    };

    if seg == n_layer {
        let mut cur = ctx.mul(ctx.rms_norm(x, h.rms_eps), w.out_norm.get());
        if last_only && st.n_tokens > 1 {
            let nb1 = cur.nb()[1];
            cur = ctx.view(cur, &[h.n_embd as i64, 1], &[nb1], (st.n_tokens - 1) * nb1);
        }
        let head = w.output.as_ref().unwrap_or(&w.tok_embd);
        let logits = ctx.mul_mat(head.get(), cur).set_output();
        g.expand(logits);
        return (g, Some(logits));
    }

    let s = &st.slots[seg % 2];
    let (ffn_inp, xn, weights, ids) =
        attention_and_router(h, ctx, &mut g, w, kv, st, seg, x, n_past);
    g.expand(ctx.cpy(ffn_inp, s.resid.get()));
    g.expand(ctx.cpy(xn, s.xn.get()));
    g.expand(ctx.cpy(weights, s.w.get()));
    g.expand(ctx.cpy(ids, s.ids.get()));
    (g, None)
}

/// Layer `il`: attention with Q/K norm, RoPE and the KV cache, the residual, the FFN norm and the
/// router (softmax, top-k, normalized weights). Returns `(ffn_inp, normed, weights, ids)`.
#[allow(clippy::too_many_arguments)]
fn attention_and_router<'a>(
    h: &Hparams,
    ctx: &'a Context,
    g: &mut Graph<'a>,
    w: &'a Weights,
    kv: &'a Kv,
    st: &'a State,
    il: usize,
    x: Tensor<'a>,
    n_past: usize,
) -> (Tensor<'a>, Tensor<'a>, Tensor<'a>, Tensor<'a>) {
    let lw = &w.layers[il];
    let n = st.n_tokens as i64;
    let (hd, nh, nkv) = (h.head_dim as i64, h.n_head as i64, h.n_head_kv as i64);
    let rope = RopeParams {
        n_dims: h.n_rot as i32,
        mode: 2, // NeoX
        n_ctx_orig: h.n_ctx_train as i32,
        freq_base: h.rope_freq_base,
        freq_scale: 1.0,
        ext_factor: 0.0,
        attn_factor: 1.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
    };

    let cur = ctx.mul(ctx.rms_norm(x, h.rms_eps), lw.attn_norm.get());
    let q = ctx.reshape(ctx.mul_mat(lw.wq.get(), cur), &[hd, nh, n]);
    let k = ctx.reshape(ctx.mul_mat(lw.wk.get(), cur), &[hd, nkv, n]);
    let v = ctx.mul_mat(lw.wv.get(), cur);
    let q = ctx.mul(ctx.rms_norm(q, h.rms_eps), lw.q_norm.get());
    let q = ctx.rope_ext(q, st.pos.get(), &rope);
    let k = ctx.mul(ctx.rms_norm(k, h.rms_eps), lw.k_norm.get());
    let k = ctx.rope_ext(k, st.pos.get(), &rope);

    // Store K and V in the cache rows of these positions; the cache must be written before the
    // attention reads it, so the stores enter the graph first.
    let (kc, vc) = (kv.k[il].get(), kv.v[il].get());
    let row = kc.nb()[1];
    let n_embd_kv = hd * nkv;
    for (src, cache) in [(k, kc), (v, vc)] {
        let src = ctx.reshape(src, &[n_embd_kv, n]);
        let dst = ctx.view(cache, &[n_embd_kv, n], &[row], n_past * row);
        g.expand(ctx.cpy(src, dst));
    }

    let n_kv = st.n_kv as i64;
    let kview = ctx.view(kc, &[hd, n_kv, nkv], &[row, hd as usize * 2], 0);
    let vview = ctx.view(vc, &[hd, n_kv, nkv], &[row, hd as usize * 2], 0);
    let q = ctx.permute(q, [0, 2, 1, 3]);
    let attn = ctx.flash_attn_ext(q, kview, vview, st.mask.get(), 1.0 / (hd as f32).sqrt());
    let attn = ctx.reshape(attn, &[hd * nh, n]);
    let ffn_inp = ctx.add(ctx.mul_mat(lw.wo.get(), attn), x);

    let xn = ctx.mul(ctx.rms_norm(ffn_inp, h.rms_eps), lw.ffn_norm.get());
    let (ne, nu) = (h.n_expert as i64, h.n_expert_used as i64);
    let probs = ctx.soft_max(ctx.mul_mat(lw.gate_inp.get(), xn));
    let ids = ctx.argsort_top_k(probs, nu as i32);
    let sel = ctx.get_rows(ctx.reshape(probs, &[1, ne, n]), ids);
    let sel = ctx.reshape(sel, &[nu, n]);
    // Normalize over the selected experts; the clamp keeps the divisor representable in F16.
    let sum = ctx.clamp(ctx.sum_rows(sel), 6.103_515_6e-5, f32::INFINITY);
    let weights = ctx.reshape(ctx.div(sel, sum), &[1, nu, n]);
    (ffn_inp, xn, weights, ids)
}

/// One pass of the expert FFN of a layer step that needed several: its output is added to `acc`
/// (or replaces it in the first pass).
fn ffn_pass<'a>(
    h: &Hparams,
    ctx: &'a Context,
    lw: &'a Layer,
    s: &'a Slot,
    first: bool,
) -> Graph<'a> {
    let mut g = ctx.new_graph(SEGMENT_NODES);
    let out = moe_ffn(h, ctx, lw, s);
    let acc = s.acc.get();
    g.expand(ctx.cpy(if first { out } else { ctx.add(out, acc) }, acc));
    g
}

/// Expert FFN of layer `il` for the ids in the slot's `ids_map`: `down(silu(gate) * up)`, scaled
/// by the router weights and summed over the selected experts.
fn moe_ffn<'a>(h: &Hparams, ctx: &'a Context, lw: &'a Layer, s: &'a Slot) -> Tensor<'a> {
    let n = s.xn.get().ne()[1];
    let cur = ctx.reshape(s.xn.get(), &[h.n_embd as i64, 1, n]);
    let ids = s.ids_map.get();
    let up = ctx.mul_mat_id(lw.up_exps.get(), cur, ids);
    let gate = ctx.mul_mat_id(lw.gate_exps.get(), cur, ids);
    let act = ctx.swiglu_split(gate, up);
    let experts = ctx.mul_mat_id(lw.down_exps.get(), act, ids);
    let experts = ctx.mul(experts, s.w.get());
    let [_, nb1, nb2, _] = experts.nb();
    let n_embd = h.n_embd as i64;
    let view = |i: usize| ctx.view(experts, &[n_embd, n], &[nb2], i * nb1);
    (1..h.n_expert_used as usize).fold(view(0), |acc, i| ctx.add(acc, view(i)))
}
