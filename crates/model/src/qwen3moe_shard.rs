//! The Qwen3 MoE builder on the own executor with every expert resident (ADR 0069, 0070): one
//! rank or many, tensor, expert and layer parallel and tensor times pipeline.
//!
//! Per step key the builder makes one graph per rank. A tensor-parallel stage cuts its rank
//! graphs after each partial sum (the `wo` output and the expert sum) and lists an all-reduce
//! there; a stage boundary lists a copy of the hidden state. With one rank and one stage the
//! graph holds exactly the ops of the resident whole-step graph of `qwen3moe.rs`, in the same
//! order, as one range.
//!
//! Slices follow ADR 0070: `wq`, `wk`, `wv` by heads (rows of the matrix), `wo` by the same heads
//! (columns), expert gate and up by the rows that match the down slice, expert down by whole
//! quant blocks with the larger share rotating over layers, the output head by vocab. Expert
//! parallel (`ep = tp`) gives rank i the experts `[i * E / g, (i + 1) * E / g)` and maps the
//! router's ids to local slot indices (-1 for experts elsewhere) with `get_rows` on a per-rank
//! table. A sequence's KV rows are the same on every rank; each rank holds its KV heads.

use crate::attn::{AttnHparams, AttnStep, AttnWeights, Kv, alloc_kv, attention_out, copy_rows};
use crate::experts::MissPolicy;
use crate::gguf::TensorInfo;
use crate::hparams::{Hparams, ModelFile, names};
use crate::qwen3moe::{
    Layer, LoadOptions, MemoryUse, ModelError, Result, router, sum_experts, weighted_experts,
};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::rc::Rc;
use std::sync::Arc;
use tetraeder_executor::{ExecError, Executor, Op, Program, RankDevice, RankGraph, Ranks};
use tetraeder_ggml_sys::ggml::{
    Backend, Context, Graph, OwnedGraph, OwnedTensor, Sched, Tensor, Type,
};
use tetraeder_kvcache::StepPlan;
use tetraeder_plan::{Topology, block_slice};
use tetraeder_telemetry::Telemetry;

/// Nodes (and tensors) a rank graph reserves per layer; a layer uses about 75.
const NODES_PER_LAYER: usize = 192;

fn exec_err(e: ExecError) -> ModelError {
    ModelError::Invalid(e.to_string())
}

fn invalid<T>(msg: String) -> Result<T> {
    Err(ModelError::Invalid(msg))
}

/// The options a sharded load cannot honour (streaming, tiers, op offload, CPU experts and the
/// other per-step host work of the `Sched` paths), each named in the error.
pub(crate) fn check_options(o: &LoadOptions, h: &Hparams) -> Result<()> {
    let mut bad: Vec<&str> = Vec::new();
    if o.expert_slots.is_some_and(|n| n < h.n_expert as usize) {
        bad.push("expert_slots");
    }
    if o.expert_slot_bytes.is_some() {
        bad.push("expert_slot_bytes");
    }
    if o.ram_slots.is_some() {
        bad.push("ram_slots");
    }
    if o.ram_slot_bytes.is_some() {
        bad.push("ram_slot_bytes");
    }
    if o.op_offload {
        bad.push("op_offload");
    }
    if o.miss == MissPolicy::Cpu {
        bad.push("miss = cpu");
    }
    if !o.formats.is_empty() {
        bad.push("formats");
    }
    if o.record_routing {
        bad.push("record_routing");
    }
    if o.overlap {
        bad.push("overlap");
    }
    if o.admit_share.is_some() {
        bad.push("admit_share");
    }
    if o.routing_profile.is_some() {
        bad.push("routing_profile");
    }
    if bad.is_empty() {
        Ok(())
    } else {
        invalid(format!(
            "a sharded load keeps every expert resident on the executor; it cannot use: {}",
            bad.join(", ")
        ))
    }
}

/// A stage as the builder runs it, after `layer_limit`.
#[derive(Debug, Clone)]
struct StageRt {
    span: Range<u32>,
    ranks: Vec<usize>,
    /// Expert parallel: whole experts per rank (`tp = ep`).
    ep: bool,
    embed: bool,
    head: bool,
}

struct RankLayer {
    w: Layer,
    /// Index into the rank's KV tensors.
    kv: usize,
}

struct HeadRt {
    out_norm: OwnedTensor,
    w: OwnedTensor,
    vocab: Range<usize>,
}

struct RankRt {
    backend: Backend,
    stage: usize,
    /// Attention hyperparameters with the rank's local head counts.
    ah: AttnHparams,
    layers: Vec<RankLayer>,
    kv: Kv,
    weights_bytes: u64,
    kv_bytes: u64,
    /// A tensor of the weights context: graphs keep the context alive through it.
    anchor: OwnedTensor,
    tok_embd: Option<OwnedTensor>,
    head: Option<HeadRt>,
    /// Expert parallel: `[1, n_expert]` I32, the local index of each expert or -1.
    map: Option<OwnedTensor>,
}

/// Inputs of one rank for one step key.
struct RankState {
    tokens: Option<OwnedTensor>,
    pos: OwnedTensor,
    rows: OwnedTensor,
    mask: OwnedTensor,
    outputs: Option<OwnedTensor>,
    /// The hidden state a later stage receives from the previous one.
    hid: Option<OwnedTensor>,
}

enum OpSpec {
    Compute {
        rank: usize,
        range: usize,
    },
    AllReduce(Vec<(usize, OwnedTensor)>),
    Copy {
        src: (usize, OwnedTensor),
        dst: (usize, OwnedTensor),
    },
}

/// A step graph set kept for its key (`LoadOptions::reuse_step_graph`).
struct StepProg {
    key: (usize, usize, usize),
    // Declared first: dropped before the state they read.
    graphs: Vec<Option<OwnedGraph>>,
    cuts: Vec<Vec<usize>>,
    ops: Vec<OpSpec>,
    /// Per head rank: its logits `[vocab slice, n_out]` and where the slice sits in the vocab.
    logits: Vec<(usize, OwnedTensor, Range<usize>)>,
    outputs: Vec<usize>,
    states: Vec<Option<RankState>>,
}

pub(crate) struct Sharded {
    exec: Executor,
    h: Hparams,
    stages: Vec<StageRt>,
    ranks: Vec<Option<RankRt>>,
    /// Per layer, the ranks holding its KV heads in head order with their KV tensor index.
    layer_kv: Vec<Vec<(usize, usize)>>,
    n_rows: usize,
    cache: Option<StepProg>,
    scheds: Vec<Option<Sched>>,
    /// Step graphs built since load.
    pub(crate) built: u64,
    telemetry: Option<Arc<Telemetry>>,
}

/// Where a tensor's bytes sit in the file.
struct Src {
    shard: usize,
    offset: u64,
    ty: Type,
    ne: [u64; 4],
}

/// Creates the tensors of one rank in a context of its own and fills them after allocation.
struct Loader<'f> {
    file: &'f ModelFile,
    ctx: Rc<Context>,
    rank: usize,
    pending: Vec<(OwnedTensor, Src, [Range<u64>; 3])>,
}

impl<'f> Loader<'f> {
    fn new(file: &'f ModelFile, rank: usize, layers: usize) -> Result<Loader<'f>> {
        Ok(Loader {
            file,
            ctx: Rc::new(Context::new(12 * layers + 16, 0)?),
            rank,
            pending: Vec::new(),
        })
    }

    /// The tensor `name` cut to `cut` (`None`: the whole dimension).
    fn tensor(&mut self, name: &str, cut: [Option<Range<u64>>; 3]) -> Result<OwnedTensor> {
        let info: &TensorInfo = self.file.gguf.tensor(name)?;
        if info.ne[3] != 1 {
            return invalid(format!("`{name}` has {} dimensions", 4));
        }
        let [c0, c1, c2] = cut;
        let r = [
            c0.unwrap_or(0..info.ne[0]),
            c1.unwrap_or(0..info.ne[1]),
            c2.unwrap_or(0..info.ne[2]),
        ];
        for (i, x) in r.iter().enumerate() {
            if x.start >= x.end || x.end > info.ne[i] {
                return invalid(format!(
                    "rank {}: `{name}`: slice {}..{} of dimension {i} ({} long)",
                    self.rank, x.start, x.end, info.ne[i]
                ));
            }
        }
        let blk = info.ty.block_size() as u64;
        if r[0].start % blk != 0 || (r[0].end - r[0].start) % blk != 0 {
            return invalid(format!(
                "rank {}: `{name}`: columns {}..{} are not whole blocks of {blk}",
                self.rank, r[0].start, r[0].end
            ));
        }
        let ne: Vec<i64> = r.iter().map(|x| (x.end - x.start) as i64).collect();
        let nd = ne.iter().rposition(|&x| x != 1).map_or(1, |i| i + 1);
        let t = self.ctx.new_persistent(info.ty, &ne[..nd]);
        t.get().set_name(name);
        let src = Src {
            shard: info.shard,
            offset: info.offset,
            ty: info.ty,
            ne: info.ne,
        };
        self.pending.push((t.clone(), src, r));
        Ok(t)
    }

    /// The whole tensor.
    fn whole(&mut self, name: &str) -> Result<OwnedTensor> {
        self.tensor(name, [None, None, None])
    }

    /// Allocates the rank's weights and reads the slices from the file.
    fn load(self, backend: &Backend, shards: &mut [File]) -> Result<u64> {
        let bytes = self.ctx.alloc_weights(backend)?;
        let mut buf: Vec<u8> = Vec::new();
        let mut out: Vec<u8> = Vec::new();
        for (t, src, r) in &self.pending {
            fill(shards, src, r, t, &mut buf, &mut out)?;
        }
        Ok(bytes as u64)
    }
}

/// Reads the sub-tensor `r` (columns, rows, planes) of `src` into `t`. Rows are read in chunks of
/// at most 16 MiB; a column cut copies the wanted part of each row.
fn fill(
    shards: &mut [File],
    src: &Src,
    r: &[Range<u64>; 3],
    t: &OwnedTensor,
    buf: &mut Vec<u8>,
    out: &mut Vec<u8>,
) -> Result<()> {
    let ty = src.ty;
    let row = ty.row_size(src.ne[0] as i64);
    let c0 = ty.row_size(r[0].start as i64);
    let w0 = ty.row_size((r[0].end - r[0].start) as i64);
    let full0 = r[0] == (0..src.ne[0]);
    let plane = row as u64 * src.ne[1];
    let rows_per = ((16usize << 20) / row.max(1)).max(1) as u64;
    let f = &mut shards[src.shard];
    let t = t.get();
    let mut at = 0usize;
    for i2 in r[2].clone() {
        let mut a = r[1].start;
        while a < r[1].end {
            let b = (a + rows_per).min(r[1].end);
            let nrows = (b - a) as usize;
            f.seek(SeekFrom::Start(src.offset + i2 * plane + a * row as u64))?;
            buf.resize(nrows * row, 0);
            f.read_exact(&mut buf[..])?;
            if full0 {
                t.set_bytes(&buf[..], at)?;
                at += buf.len();
            } else {
                out.clear();
                for k in 0..nrows {
                    out.extend_from_slice(&buf[k * row + c0..k * row + c0 + w0]);
                }
                t.set_bytes(&out[..], at)?;
                at += out.len();
            }
            a = b;
        }
    }
    if at != t.nbytes() {
        return invalid(format!(
            "slice of `{}` read {at} bytes, the tensor holds {}",
            t.name(),
            t.nbytes()
        ));
    }
    Ok(())
}

/// Slice `i` of `n` equal parts of `len` elements (heads, vocab).
fn even(len: usize, n: usize, i: usize) -> Range<usize> {
    i * len / n..(i + 1) * len / n
}

fn elems(r: Range<usize>, unit: usize) -> Range<u64> {
    (r.start * unit) as u64..(r.end * unit) as u64
}

impl Sharded {
    /// Loads the rank slices of `topo`. `threads` is the thread count of CPU ranks and `kv_rows`
    /// the rows of the KV cache.
    pub(crate) fn load(
        file: &ModelFile,
        topo: &Topology,
        opts: &LoadOptions,
        threads: i32,
        kv_rows: usize,
    ) -> Result<Sharded> {
        let h = file.hparams.clone();
        if topo.n_layer != h.n_layer {
            return invalid(format!(
                "the topology has {} layers, the model {}",
                topo.n_layer, h.n_layer
            ));
        }
        if h.n_expert_used < 2 {
            return invalid("sharded loads need at least two experts per token".to_string());
        }
        let limit = opts.layer_limit.unwrap_or(h.n_layer).min(h.n_layer);
        if limit == 0 {
            return invalid("layer_limit must be at least 1".to_string());
        }
        // Stages up to the layer limit; the last one holds the head.
        let mut stages: Vec<StageRt> = Vec::new();
        let mut owner: Vec<Option<usize>> = vec![None; topo.ranks.len()];
        for (si, s) in topo.stages.iter().enumerate() {
            if s.layers.0 >= limit {
                break;
            }
            let tp = s.ranks.len();
            if tp == 0 || tp != s.tp as usize {
                return invalid(format!(
                    "stage {si}: {} ranks for tp = {}",
                    s.ranks.len(),
                    s.tp
                ));
            }
            if s.ep != 1 && s.ep != s.tp {
                return invalid(format!("stage {si}: ep = {} needs tp = ep", s.ep));
            }
            if !(h.n_head as usize).is_multiple_of(tp) || !(h.n_head_kv as usize).is_multiple_of(tp)
            {
                return invalid(format!(
                    "stage {si}: tp = {tp} does not divide {} heads and {} KV heads",
                    h.n_head, h.n_head_kv
                ));
            }
            let ep = s.ep > 1;
            if ep && !(h.n_expert as usize).is_multiple_of(tp) {
                return invalid(format!(
                    "stage {si}: ep = {tp} does not divide {} experts",
                    h.n_expert
                ));
            }
            for &r in &s.ranks {
                if r >= topo.ranks.len() {
                    return invalid(format!("stage {si}: rank {r} is not in the topology"));
                }
                if owner[r].replace(si).is_some() {
                    return invalid(format!(
                        "rank {r} is in two stages; the sharded MoE builder runs one stage per rank"
                    ));
                }
            }
            stages.push(StageRt {
                span: s.layers.0..s.layers.1.min(limit),
                ranks: s.ranks.clone(),
                ep,
                embed: si == 0,
                head: false,
            });
        }
        if let Some(last) = stages.last_mut() {
            last.head = true;
        } else {
            return invalid("the topology has no stage".to_string());
        }

        let devices: Vec<RankDevice> = topo
            .ranks
            .iter()
            .map(|r| match r.gpu_index {
                Some(i) => RankDevice::Gpu(i),
                None => RankDevice::Cpu { threads },
            })
            .collect();
        let ranks = Ranks::open(&devices).map_err(exec_err)?;
        for (si, st) in stages.iter().enumerate() {
            if st.ep && !tetraeder_ggml_sys::PATCH_0004 {
                for &r in &st.ranks {
                    if !ranks.backend(r).is_cpu() {
                        return invalid(format!(
                            "stage {si}: expert parallel on a GPU rank needs patch 0004 (CUDA mul_mat_id skips expert id -1), which this build lacks"
                        ));
                    }
                }
            }
        }

        let mut shards = file
            .gguf
            .paths
            .iter()
            .map(File::open)
            .collect::<std::io::Result<Vec<_>>>()?;
        let mut rts: Vec<Option<RankRt>> = (0..topo.ranks.len()).map(|_| None).collect();
        let mut layer_kv: Vec<Vec<(usize, usize)>> = vec![Vec::new(); limit as usize];
        for (si, st) in stages.iter().enumerate() {
            let g = st.ranks.len();
            for (slot, &r) in st.ranks.iter().enumerate() {
                let rt = Self::load_rank(
                    file,
                    opts,
                    &h,
                    (si, st),
                    (slot, g),
                    (r, ranks.backend(r)),
                    kv_rows,
                    &mut shards,
                )?;
                for il in st.span.clone() {
                    layer_kv[il as usize].push((r, (il - st.span.start) as usize));
                }
                rts[r] = Some(rt);
            }
        }
        let scheds = (0..rts.len()).map(|_| None).collect();
        let exec = Executor::new(ranks);
        if let Some(t) = &opts.telemetry {
            exec.ranks().record(t.metrics());
        }
        Ok(Sharded {
            exec,
            h,
            stages,
            ranks: rts,
            layer_kv,
            n_rows: kv_rows,
            cache: None,
            scheds,
            built: 0,
            telemetry: opts.telemetry.clone(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn load_rank(
        file: &ModelFile,
        opts: &LoadOptions,
        h: &Hparams,
        (si, st): (usize, &StageRt),
        (slot, g): (usize, usize),
        (rank, backend): (usize, &Backend),
        kv_rows: usize,
        shards: &mut [File],
    ) -> Result<RankRt> {
        let (nh, nkv, hd) = (h.n_head as usize, h.n_head_kv as usize, h.head_dim as usize);
        let n_local = st.span.len();
        let mut ld = Loader::new(file, rank, n_local)?;
        let heads = elems(even(nh, g, slot), hd);
        let kv_heads = elems(even(nkv, g, slot), hd);
        let (e0, e1) = (
            (slot * h.n_expert as usize / g) as u64,
            ((slot + 1) * h.n_expert as usize / g) as u64,
        );
        let mut layers = Vec::new();
        for il in st.span.clone() {
            let mut one = |what: &str, cut: [Option<Range<u64>>; 3]| -> Result<OwnedTensor> {
                ld.tensor(&names::layer(il, what), cut)
            };
            let attn = AttnWeights {
                attn_norm: one("attn_norm", [None, None, None])?,
                wq: one("attn_q", [None, Some(heads.clone()), None])?,
                wk: one("attn_k", [None, Some(kv_heads.clone()), None])?,
                wv: one("attn_v", [None, Some(kv_heads.clone()), None])?,
                wo: one("attn_output", [Some(heads.clone()), None, None])?,
                q_norm: one("attn_q_norm", [None, None, None])?,
                k_norm: one("attn_k_norm", [None, None, None])?,
            };
            let ffn_norm = one("ffn_norm", [None, None, None])?;
            let gate_inp = one("ffn_gate_inp", [None, None, None])?;
            let (gate_exps, up_exps, down_exps) = if st.ep {
                let c = Some(e0..e1);
                (
                    one("ffn_gate_exps", [None, None, c.clone()])?,
                    one("ffn_up_exps", [None, None, c.clone()])?,
                    one("ffn_down_exps", [None, None, c])?,
                )
            } else if g > 1 {
                let name = names::layer(il, "ffn_down_exps");
                let down = file.gguf.tensor(&name)?;
                let f = block_slice(
                    u64::from(h.n_ff_exp),
                    down.ty.block_size() as u64,
                    g as u32,
                    slot as u32,
                    il,
                )
                .map_err(|e| {
                    ModelError::Invalid(format!("rank {rank}: layer {il} expert cut: {e}"))
                })?;
                (
                    one("ffn_gate_exps", [None, Some(f.clone()), None])?,
                    one("ffn_up_exps", [None, Some(f.clone()), None])?,
                    one("ffn_down_exps", [Some(f), None, None])?,
                )
            } else {
                (
                    one("ffn_gate_exps", [None, None, None])?,
                    one("ffn_up_exps", [None, None, None])?,
                    one("ffn_down_exps", [None, None, None])?,
                )
            };
            layers.push(RankLayer {
                w: Layer {
                    attn,
                    ffn_norm,
                    gate_inp,
                    gate_exps,
                    up_exps,
                    down_exps,
                },
                kv: (il - st.span.start) as usize,
            });
        }
        let tok_embd = if st.embed || (st.head && h.tied_output) {
            Some(ld.whole(names::TOK_EMBD)?)
        } else {
            None
        };
        let head = if st.head {
            let vocab = even(h.n_vocab as usize, g, slot);
            let whole = vocab == (0..h.n_vocab as usize);
            let w = if h.tied_output && whole {
                tok_embd.clone().expect("loaded above for a tied head")
            } else {
                let name = if h.tied_output {
                    names::TOK_EMBD
                } else {
                    names::OUTPUT
                };
                ld.tensor(
                    name,
                    [None, Some(vocab.start as u64..vocab.end as u64), None],
                )?
            };
            Some(HeadRt {
                out_norm: ld.whole(names::OUTPUT_NORM)?,
                w,
                vocab,
            })
        } else {
            None
        };
        let map = if st.ep {
            let t = ld
                .ctx
                .new_persistent(Type::I32, &[1, i64::from(h.n_expert)]);
            t.get().set_name("expert_map");
            Some(t)
        } else {
            None
        };
        let anchor = layers[0].w.attn.attn_norm.clone();
        let weights_bytes = ld.load(backend, shards)?;
        if let Some(m) = &map {
            // The map is a tensor of the weights context, allocated with it.
            let table: Vec<i32> = (0..h.n_expert as u64)
                .map(|e| {
                    if (e0..e1).contains(&e) {
                        (e - e0) as i32
                    } else {
                        -1
                    }
                })
                .collect();
            m.get().set_i32(&table)?;
        }

        let mut ah = h.attn();
        ah.n_head = (nh / g) as u32;
        ah.n_head_kv = (nkv / g) as u32;
        ah.n_layer = n_local as u32;
        let (kv, kv_bytes) = alloc_kv(&ah, kv_rows, backend, opts.kv, opts.kv_rotation)?;
        Ok(RankRt {
            backend: backend.clone(),
            stage: si,
            ah,
            layers,
            kv,
            weights_bytes,
            kv_bytes: kv_bytes as u64,
            anchor,
            tok_embd,
            head,
            map,
        })
    }

    fn rank(&self, r: usize) -> &RankRt {
        self.ranks[r].as_ref().expect("a rank of a built stage")
    }

    /// The ranks that hold weights, in rank order.
    fn used(&self) -> impl Iterator<Item = (usize, &RankRt)> {
        self.ranks
            .iter()
            .enumerate()
            .filter_map(|(r, x)| x.as_ref().map(|x| (r, x)))
    }

    pub(crate) fn memory(&self) -> Vec<MemoryUse> {
        let mut out = Vec::new();
        for (_, rt) in self.used() {
            let tier = if rt.backend.is_cpu() { "ram" } else { "vram" };
            let device = rt.backend.name();
            out.push(MemoryUse {
                device: device.clone(),
                tier,
                kind: "weights",
                format: None,
                capacity: rt.weights_bytes,
                used: rt.weights_bytes,
            });
            out.push(MemoryUse {
                device,
                tier,
                kind: "kv",
                format: Some(rt.kv.fmt.to_string()),
                capacity: rt.kv_bytes,
                used: rt.kv_bytes,
            });
        }
        out
    }

    /// Bytes of one cached position across all layers (K and V) and all ranks.
    pub(crate) fn kv_row_bytes(&self) -> usize {
        self.used()
            .map(|(_, rt)| {
                rt.kv
                    .k
                    .iter()
                    .chain(&rt.kv.v)
                    .map(|t| t.get().nbytes() / self.n_rows)
                    .sum::<usize>()
            })
            .sum()
    }

    /// Copies KV rows on every rank. Each entry is `(src_row, dst_row, n_rows)`.
    pub(crate) fn copy_rows(&mut self, list: &[(usize, usize, usize)]) -> Result<()> {
        for (slot, sched) in self.ranks.iter().zip(self.scheds.iter_mut()) {
            let Some(rt) = slot else { continue };
            if sched.is_none() {
                *sched = Some(Sched::new(std::slice::from_ref(&rt.backend), 16384, false)?);
            }
            if let Some(s) = sched.as_mut() {
                copy_rows(&rt.kv, s, list)?;
            }
        }
        Ok(())
    }

    /// The one-rank layout of the K and V rows of `runs` (first row, length): all K tensors in
    /// layer order, then all V, each row holding every KV head in head order.
    pub(crate) fn snapshot(&self, runs: &[(usize, usize)]) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        for v in [false, true] {
            for layer in &self.layer_kv {
                let parts = self.kv_parts(layer, v);
                for &(start, len) in runs {
                    let chunks = parts
                        .iter()
                        .map(|(t, row)| t.get().get_bytes(start * row, len * row))
                        .collect::<std::result::Result<Vec<_>, _>>()?;
                    for i in 0..len {
                        for (c, (_, row)) in chunks.iter().zip(&parts) {
                            out.extend_from_slice(&c[i * row..(i + 1) * row]);
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// Writes a [`Sharded::snapshot`] back into the rows of `runs`.
    pub(crate) fn restore(&self, runs: &[(usize, usize)], bytes: &[u8]) -> Result<()> {
        let mut rest = bytes;
        for v in [false, true] {
            for layer in &self.layer_kv {
                let parts = self.kv_parts(layer, v);
                let total: usize = parts.iter().map(|(_, row)| row).sum();
                for &(start, len) in runs {
                    let (head, tail) = rest.split_at(len * total);
                    rest = tail;
                    let mut at = 0;
                    for (t, row) in &parts {
                        let mut buf = Vec::with_capacity(len * row);
                        for i in 0..len {
                            let o = i * total + at;
                            buf.extend_from_slice(&head[o..o + row]);
                        }
                        t.get().set_bytes(&buf, start * row)?;
                        at += row;
                    }
                }
            }
        }
        Ok(())
    }

    /// The KV tensors of one layer in head order with their row bytes.
    fn kv_parts(&self, layer: &[(usize, usize)], v: bool) -> Vec<(OwnedTensor, usize)> {
        layer
            .iter()
            .map(|&(r, local)| {
                let kv = &self.rank(r).kv;
                let t = if v { &kv.v[local] } else { &kv.k[local] };
                (t.clone(), t.get().nbytes() / self.n_rows)
            })
            .collect()
    }

    fn alloc_state(
        &self,
        rt: &RankRt,
        (n, n_kv, n_out): (usize, usize, usize),
    ) -> Result<RankState> {
        let st = &self.stages[rt.stage];
        let ctx = Rc::new(Context::new(8, 0)?);
        let nt = n as i64;
        let state = RankState {
            tokens: st.embed.then(|| ctx.new_persistent(Type::I32, &[nt])),
            pos: ctx.new_persistent(Type::I32, &[nt]),
            rows: ctx.new_persistent(Type::I32, &[nt]),
            mask: ctx.new_persistent(Type::F16, &[n_kv as i64, nt]),
            outputs: (st.head && n_out < n).then(|| ctx.new_persistent(Type::I32, &[n_out as i64])),
            hid: (!st.embed)
                .then(|| ctx.new_persistent(Type::F32, &[i64::from(self.h.n_embd), nt])),
        };
        ctx.alloc_tensors(&rt.backend)?;
        Ok(state)
    }

    /// One step: uploads the inputs, runs the program and returns the logits of the output
    /// tokens, `n_vocab` per row, into `out`.
    pub(crate) fn step(
        &mut self,
        tokens: &[i32],
        plan: &StepPlan,
        reuse: bool,
        out: &mut Vec<f32>,
    ) -> Result<()> {
        let key = (tokens.len(), plan.n_kv, plan.outputs.len());
        let prog = match self.cache.take() {
            Some(p) if reuse && p.key == key => p,
            _ => {
                let p = self.build(key)?;
                self.built += 1;
                p
            }
        };
        for st in prog.states.iter().flatten() {
            if let Some(t) = &st.tokens {
                t.get().set_i32(tokens)?;
            }
            st.pos.get().set_i32(&plan.pos)?;
            st.rows.get().set_i32(&plan.rows)?;
            st.mask.get().set_f16_bits(&plan.mask)?;
            if let Some(t) = &st.outputs {
                t.get().set_i32(&plan.outputs)?;
            }
        }
        {
            let graphs: Vec<RankGraph<'_>> = prog
                .graphs
                .iter()
                .enumerate()
                .filter_map(|(r, g)| {
                    g.as_ref().map(|g| RankGraph {
                        rank: r,
                        graph: g.get(),
                        cuts: prog.cuts[r].clone(),
                    })
                })
                .collect();
            let ops: Vec<Op<'_>> = prog
                .ops
                .iter()
                .map(|o| match o {
                    OpSpec::Compute { rank, range } => Op::Compute {
                        rank: *rank,
                        range: *range,
                    },
                    OpSpec::AllReduce(v) => Op::AllReduce {
                        tensors: v.iter().map(|(r, t)| (*r, t.get())).collect(),
                    },
                    OpSpec::Copy { src, dst } => Op::Copy {
                        src: (src.0, src.1.get()),
                        dst: (dst.0, dst.1.get()),
                    },
                })
                .collect();
            let p = Program {
                graphs,
                ops,
                outputs: prog.outputs.clone(),
            };
            self.exec.run(&p).map_err(exec_err)?;
        }
        let n_out = plan.outputs.len();
        let nv = self.h.n_vocab as usize;
        if prog.logits.len() == 1 && prog.logits[0].2 == (0..nv) {
            prog.logits[0].1.get().get_f32_into(out)?;
        } else {
            out.clear();
            out.resize(n_out * nv, 0.0);
            for (_, t, vocab) in &prog.logits {
                let part = t.get().get_f32()?;
                let w = vocab.len();
                for row in 0..n_out {
                    out[row * nv + vocab.start..row * nv + vocab.end]
                        .copy_from_slice(&part[row * w..(row + 1) * w]);
                }
            }
        }
        if let Some(t) = &self.telemetry {
            self.exec.take_counters().record(t.metrics());
        }
        if reuse {
            self.cache = Some(prog);
        }
        Ok(())
    }

    /// The rank graphs and the program for one step key.
    fn build(&self, key: (usize, usize, usize)) -> Result<StepProg> {
        let (n, n_kv, n_out) = key;
        let h = &self.h;
        let nr = self.ranks.len();
        let mut states: Vec<Option<RankState>> = (0..nr).map(|_| None).collect();
        let mut ctxs: Vec<Option<Rc<Context>>> = (0..nr).map(|_| None).collect();
        for (r, rt) in self.used() {
            states[r] = Some(self.alloc_state(rt, key)?);
            let nodes = NODES_PER_LAYER * (rt.layers.len() + 1);
            ctxs[r] = Some(Rc::new(Context::new(nodes, nodes)?));
        }
        let built = {
            let mut b = Build {
                gs: (0..nr).map(|_| None).collect(),
                cuts: vec![Vec::new(); nr],
                ops: Vec::new(),
                logits: Vec::new(),
            };
            for (r, rt) in self.used() {
                let nodes = NODES_PER_LAYER * (rt.layers.len() + 1);
                b.gs[r] = Some(cx(&ctxs, r).new_graph(nodes));
            }
            let mut xs: Vec<Option<Tensor<'_>>> = vec![None; nr];
            for (si, st) in self.stages.iter().enumerate() {
                let g = st.ranks.len();
                for &r in &st.ranks {
                    let rt = self.rank(r);
                    let sr = states[r].as_ref().expect("state of a used rank");
                    xs[r] = Some(match (&rt.tok_embd, &sr.hid) {
                        (Some(emb), None) if st.embed => cx(&ctxs, r).get_rows(
                            emb.get(),
                            sr.tokens.as_ref().expect("embed rank has tokens").get(),
                        ),
                        (_, Some(hid)) => hid.get(),
                        _ => unreachable!("a stage after the first has a hidden input"),
                    });
                }
                for il in st.span.clone() {
                    let local = (il - st.span.start) as usize;
                    let mut parts: Vec<Tensor<'_>> = Vec::with_capacity(g);
                    for &r in &st.ranks {
                        let rt = self.rank(r);
                        let sr = states[r].as_ref().expect("state of a used rank");
                        let step = AttnStep {
                            pos: sr.pos.get(),
                            rows: sr.rows.get(),
                            mask: sr.mask.get(),
                            n_tokens: n,
                            n_kv,
                        };
                        let gr = b.gs[r].as_mut().expect("graph of a used rank");
                        parts.push(attention_out(
                            &rt.ah,
                            cx(&ctxs, r),
                            gr,
                            &rt.layers[local].w.attn,
                            &rt.kv,
                            rt.layers[local].kv,
                            &step,
                            xs[r].expect("hidden state"),
                        ));
                    }
                    if g > 1 {
                        b.reduce(&st.ranks, &parts);
                    }
                    let mut inps: Vec<Tensor<'_>> = Vec::with_capacity(g);
                    let mut ms: Vec<Tensor<'_>> = Vec::with_capacity(g);
                    for (i, &r) in st.ranks.iter().enumerate() {
                        let rt = self.rank(r);
                        let c = cx(&ctxs, r);
                        let lw = &rt.layers[local].w;
                        let ffn_inp = c.add(parts[i], xs[r].expect("hidden state"));
                        let (xn, weights, ids) = router(h, c, lw, ffn_inp);
                        // The router's nodes enter the graph before the expert FFN, consecutive
                        // as in libllama's `build_moe_ffn`, so CUDA's topk-moe fusion fires.
                        b.gs[r].as_mut().expect("graph").expand(weights);
                        let ids = match &rt.map {
                            Some(map) if st.ep => {
                                let nu = i64::from(h.n_expert_used);
                                let flat = c.reshape(c.cont(ids), &[nu * n as i64, 1]);
                                c.reshape(c.get_rows(map.get(), flat), &[nu, n as i64])
                            }
                            _ => ids,
                        };
                        let experts = weighted_experts(h, c, lw, xn, ids, weights);
                        ms.push(sum_experts(h, c, experts));
                        inps.push(ffn_inp);
                    }
                    if g > 1 {
                        b.reduce(&st.ranks, &ms);
                    }
                    for (i, &r) in st.ranks.iter().enumerate() {
                        xs[r] = Some(cx(&ctxs, r).add(ms[i], inps[i]));
                    }
                }
                if st.head {
                    for &r in &st.ranks {
                        let rt = self.rank(r);
                        let sr = states[r].as_ref().expect("state of a used rank");
                        let hr = rt.head.as_ref().expect("head rank");
                        let c = cx(&ctxs, r);
                        let x = xs[r].expect("hidden state");
                        let mut cur = c.mul(c.rms_norm(x, h.rms_eps), hr.out_norm.get());
                        if n_out < n {
                            cur =
                                c.get_rows(cur, sr.outputs.as_ref().expect("outputs input").get());
                        }
                        let logits = c.mul_mat(hr.w.get(), cur).set_output();
                        let gr = b.gs[r].as_mut().expect("graph");
                        gr.expand(logits);
                        b.cut(r);
                        b.logits.push((r, logits, hr.vocab.clone()));
                    }
                } else {
                    for &r in &st.ranks {
                        let x = xs[r].expect("hidden state");
                        b.gs[r].as_mut().expect("graph").expand(x);
                        b.cut(r);
                    }
                    let next = &self.stages[si + 1];
                    for (k, &dst) in next.ranks.iter().enumerate() {
                        let src = st.ranks[k % g];
                        let hid = states[dst]
                            .as_ref()
                            .and_then(|s| s.hid.clone())
                            .expect("hidden input of the next stage");
                        b.ops.push(BOp::Copy {
                            src: (src, xs[src].expect("hidden state")),
                            dst: (dst, hid),
                        });
                    }
                }
            }
            self.keep(&ctxs, &states, b)?
        };
        let (graphs, cuts, ops, logits) = built;
        let outputs = logits.iter().map(|l| l.0).collect();
        Ok(StepProg {
            key,
            graphs,
            cuts,
            ops,
            logits,
            outputs,
            states,
        })
    }

    /// Turns the build into owned handles that outlive the borrows.
    #[allow(clippy::type_complexity)]
    fn keep(
        &self,
        ctxs: &[Option<Rc<Context>>],
        states: &[Option<RankState>],
        b: Build<'_>,
    ) -> Result<(
        Vec<Option<OwnedGraph>>,
        Vec<Vec<usize>>,
        Vec<OpSpec>,
        Vec<(usize, OwnedTensor, Range<usize>)>,
    )> {
        let Build {
            gs,
            cuts,
            ops,
            logits,
        } = b;
        let rc = |r: usize| ctxs[r].as_ref().expect("context of a used rank");
        let mut graphs: Vec<Option<OwnedGraph>> = Vec::with_capacity(gs.len());
        for (r, g) in gs.into_iter().enumerate() {
            let Some(g) = g else {
                graphs.push(None);
                continue;
            };
            let rt = self.rank(r);
            let sr = states[r].as_ref().expect("state of a used rank");
            let mut deps: Vec<&OwnedTensor> = vec![&rt.anchor, &rt.kv.k[0], &rt.kv.v[0], &sr.pos];
            deps.extend(rt.kv.rot_k.iter().chain(&rt.kv.rot_v));
            graphs.push(Some(rc(r).keep_graph(g, &deps)?));
        }
        let mut specs = Vec::with_capacity(ops.len());
        for op in ops {
            specs.push(match op {
                BOp::Compute(rank, range) => OpSpec::Compute { rank, range },
                BOp::AllReduce(v) => {
                    let mut owned = Vec::with_capacity(v.len());
                    for (r, t) in v {
                        owned.push((r, rc(r).keep(t)?));
                    }
                    OpSpec::AllReduce(owned)
                }
                BOp::Copy { src, dst } => OpSpec::Copy {
                    src: (src.0, rc(src.0).keep(src.1)?),
                    dst,
                },
            });
        }
        let mut outs = Vec::with_capacity(logits.len());
        for (r, t, vocab) in logits {
            outs.push((r, rc(r).keep(t)?, vocab));
        }
        Ok((graphs, cuts, specs, outs))
    }
}

fn cx(ctxs: &[Option<Rc<Context>>], r: usize) -> &Context {
    ctxs[r].as_ref().expect("context of a used rank")
}

enum BOp<'a> {
    Compute(usize, usize),
    AllReduce(Vec<(usize, Tensor<'a>)>),
    Copy {
        src: (usize, Tensor<'a>),
        dst: (usize, OwnedTensor),
    },
}

/// Rank graphs, their range cuts and the program ops while a step is built.
struct Build<'a> {
    gs: Vec<Option<Graph<'a>>>,
    cuts: Vec<Vec<usize>>,
    ops: Vec<BOp<'a>>,
    logits: Vec<(usize, Tensor<'a>, Range<usize>)>,
}

impl<'a> Build<'a> {
    /// Ends the current range of rank `r` at the graph's last node and lists it for compute.
    fn cut(&mut self, r: usize) {
        let n = self.gs[r].as_ref().expect("graph").n_nodes();
        self.cuts[r].push(n);
        self.ops.push(BOp::Compute(r, self.cuts[r].len() - 1));
    }

    /// Ends a range at `parts[i]` on every rank of the group and all-reduces them.
    fn reduce(&mut self, ranks: &[usize], parts: &[Tensor<'a>]) {
        for (i, &r) in ranks.iter().enumerate() {
            self.gs[r].as_mut().expect("graph").expand(parts[i]);
            self.cut(r);
        }
        self.ops.push(BOp::AllReduce(
            ranks.iter().copied().zip(parts.iter().copied()).collect(),
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn even_splits_cover_the_range() {
        for (len, n) in [(32usize, 2usize), (32, 4), (151_936, 4), (4, 2), (7, 3)] {
            let mut next = 0;
            for i in 0..n {
                let r = even(len, n, i);
                assert_eq!(r.start, next, "len {len}, n {n}, slice {i}");
                next = r.end;
            }
            assert_eq!(next, len);
        }
    }

    #[test]
    fn element_ranges_scale_by_the_unit() {
        assert_eq!(elems(2..4, 128), 256..512);
    }
}

/// The op order of every rank graph where ggml-cuda decides its fusions (as `qwen3moe.rs`
/// `graph_order`): the router is the topk-moe window and the weighted expert output is followed
/// by its views and adds, with no range boundary inside either, on every rank of every shape.
#[cfg(test)]
mod graph_order {
    use super::*;
    use crate::qwen3moe::BackendChoice;
    use std::path::PathBuf;
    use tetraeder_plan::{RankSpec, Span, StageSpec};

    const TOPK_MOE: [&str; 10] = [
        "SOFT_MAX", "RESHAPE", "ARGSORT", "VIEW", "GET_ROWS", "RESHAPE", "SUM_ROWS", "CLAMP",
        "DIV", "RESHAPE",
    ];

    fn topo(n_ranks: usize, ep: bool) -> Topology {
        let ranks = (0..n_ranks)
            .map(|i| RankSpec {
                device: format!("cpu{i}"),
                gpu_index: None,
            })
            .collect();
        Topology {
            ranks,
            stages: vec![StageSpec {
                layers: Span(0, 2),
                ranks: (0..n_ranks).collect(),
                tp: n_ranks as u32,
                ep: if ep { n_ranks as u32 } else { 1 },
                embed: true,
                head: true,
            }],
            n_layer: 2,
        }
    }

    fn check(prog: &StepProg, n_layer: usize, n_used: usize, what: &str) {
        for (r, g) in prog.graphs.iter().enumerate() {
            let Some(g) = g else { continue };
            let g = g.get();
            let names: Vec<String> = (0..g.n_nodes()).map(|i| g.node(i).op_name()).collect();
            let inside = |i: usize, len: usize| prog.cuts[r].iter().any(|&c| c > i && c < i + len);
            let (mut routers, mut reductions) = (0, 0);
            for (i, name) in names.iter().enumerate() {
                match name.as_str() {
                    "SOFT_MAX" => {
                        routers += 1;
                        assert_eq!(
                            &names[i..i + TOPK_MOE.len()],
                            &TOPK_MOE,
                            "{what}, rank {r}: router at node {i}"
                        );
                        assert!(
                            !inside(i, TOPK_MOE.len()),
                            "{what}, rank {r}: cut in router"
                        );
                    }
                    "MUL"
                        if g.node(i)
                            .src(0)
                            .is_some_and(|e| e.op_name() == "MUL_MAT_ID") =>
                    {
                        reductions += 1;
                        let span = 2 * n_used;
                        for (j, op) in names[i + 1..i + span].iter().enumerate() {
                            assert!(
                                op == "VIEW" || op == "ADD",
                                "{what}, rank {r}: node {} after the weighting is {op}",
                                i + 1 + j
                            );
                        }
                        assert!(!inside(i, span), "{what}, rank {r}: cut in the reduction");
                    }
                    _ => {}
                }
            }
            assert_eq!(
                (routers, reductions),
                (n_layer, n_layer),
                "{what}, rank {r}"
            );
        }
    }

    #[test]
    fn every_rank_graph_keeps_the_cuda_fusion_order() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-qwen3moe-f16.gguf");
        let file = ModelFile::open(&path).expect("fixture opens");
        let h = file.hparams.clone();
        let opts = LoadOptions::new(BackendChoice::Cpu { threads: Some(1) });
        for (name, t) in [
            ("single", topo(1, false)),
            ("tp-2", topo(2, false)),
            ("ep-2", topo(2, true)),
        ] {
            let sh = Sharded::load(&file, &t, &opts, 1, 256).expect("loads");
            for (n, n_out) in [(1, 1), (5, 5), (5, 1)] {
                let prog = sh.build((n, 256, n_out)).expect("builds");
                let what = format!("{name}, n {n}, outputs {n_out}");
                check(&prog, 2, h.n_expert_used as usize, &what);
            }
        }
    }
}
