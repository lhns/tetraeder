//! The dense builder on the executor (ADR 0069, 0070): per rank its slice of the weights and its
//! KV heads and layers, one graph per rank and step cut at the boundaries, and the program that
//! issues them. A one-rank topology is the whole-step graph of `build_step` on one backend.
//!
//! Layout of a step. Stage by stage; inside a stage of degree `tp` the ranks run layer by layer
//! in lock step. Attention ends in `wo`, a partial sum when `tp > 1`: the range ends there and an
//! all-reduce follows, then the residual add starts the next range; the FFN does the same with
//! `ffn_down`. The last node of a stage that is not the last is the hidden state `x`; each rank
//! copies it to the same position of the next stage (one hop tensor per rank and step). The last
//! stage's ranks cut the head by vocabulary; the logits are read per rank and joined in vocab
//! order.

use super::{DenseHparams, DenseOptions, Layer, STEP_NODES_PER_LAYER, State, alloc_state};
use crate::attn::{
    AttnStep, AttnWeights, Kv, alloc_kv_layers, attention_out, copy_rows_on, rms_norm_mul,
};
use crate::gguf::Gguf;
use crate::hparams::names;
use crate::qwen3moe::{BackendChoice, MemoryUse, ModelError, Result};
use crate::shard::{WeightLoader, block_part, part, row_part};
use std::collections::BTreeSet;
use std::ops::Range;
use std::rc::Rc;
use tetraeder_executor::{
    ExecCounters, ExecError, Executor, Op, Program, RankDevice, RankGraph, Ranks,
};
use tetraeder_ggml_sys::ggml::{Backend, Context, OwnedGraph, OwnedTensor, Tensor, Type};
use tetraeder_kvcache::StepPlan;
use tetraeder_plan::Topology;

/// Upper bound of nodes in the graph of a rank holding `layers` layers.
fn nodes_for(layers: usize) -> usize {
    STEP_NODES_PER_LAYER * (layers + 1)
}

fn exec_err(e: ExecError) -> ModelError {
    ModelError::Invalid(format!("executor: {e}"))
}

/// What one rank holds of the model.
struct RankWeights {
    /// Stage 0 only.
    tok_embd: Option<OwnedTensor>,
    /// Last stage only.
    out_norm: Option<OwnedTensor>,
    /// The rank's vocabulary slice of the head. `None`: the head is the whole token embedding
    /// (tied output on a rank that holds all of it).
    head: Option<OwnedTensor>,
    layers: Vec<Layer>,
    /// A tensor of the weights' context, so a kept graph holds it.
    anchor: OwnedTensor,
}

struct RankData {
    backend: Backend,
    /// Layers of the rank's stage that run (`layer_limit` applied).
    layers: Range<u32>,
    weights: RankWeights,
    weights_bytes: usize,
    kv: Kv,
    kv_bytes: usize,
    /// Vocabulary rows of the head slice; empty off the last stage.
    vocab: Range<usize>,
    state: Option<State>,
}

enum StepOp {
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

/// The graphs and program of one step key, kept while the key repeats.
struct Step {
    key: (usize, usize, usize),
    /// In rank order.
    graphs: Vec<OwnedGraph>,
    cuts: Vec<Vec<usize>>,
    ops: Vec<StepOp>,
    /// The head's logits `[vocab slice, n_outputs]` of each last-stage rank, in vocab order.
    logits: Vec<(usize, OwnedTensor)>,
}

pub(super) struct Ranked {
    topo: Topology,
    exec: Executor,
    ranks: Vec<RankData>,
    /// Layers that run, from the first.
    n_layers: u32,
    step: Option<Step>,
    state_key: Option<(usize, usize, usize)>,
    totals: ExecCounters,
}

fn cpu_threads(opts: &DenseOptions, cpu: Option<usize>) -> i32 {
    opts.threads.map(|t| t.max(1) as usize).or(cpu).map_or_else(
        || std::thread::available_parallelism().map_or(1, usize::from) as i32,
        |t| t as i32,
    )
}

impl Ranked {
    pub(super) fn load(
        gguf: &Gguf,
        h: &DenseHparams,
        topo: &Topology,
        opts: &DenseOptions,
        rows: usize,
    ) -> Result<Ranked> {
        let a = &h.attn;
        check_topology(topo, h, opts)?;
        let n_layers = opts.layer_limit.map_or(a.n_layer, |l| l.min(a.n_layer));
        let devices: Vec<RankDevice> = topo
            .ranks
            .iter()
            .enumerate()
            .map(|(i, r)| match opts.backend {
                BackendChoice::Cpu { threads } => RankDevice::Cpu {
                    threads: cpu_threads(opts, threads),
                },
                BackendChoice::Gpu { .. } => RankDevice::Gpu(r.gpu_index.unwrap_or(i)),
            })
            .collect();
        let exec = Executor::new(Ranks::open(&devices).map_err(exec_err)?);
        if let Some(t) = &opts.telemetry {
            exec.ranks().record(t.metrics());
        }

        let n_stages = topo.stages.len();
        let mut ranks: Vec<Option<RankData>> = (0..topo.ranks.len()).map(|_| None).collect();
        for (s, stage) in topo.stages.iter().enumerate() {
            let tp = stage.ranks.len() as u32;
            let layers = stage.layers.0..stage.layers.1.min(n_layers);
            for (i, &r) in stage.ranks.iter().enumerate() {
                let backend = exec.ranks().backend(r).clone();
                let i = i as u32;
                let (weights, weights_bytes, vocab) = load_rank(
                    gguf,
                    h,
                    (tp, i),
                    layers.clone(),
                    stage.embed,
                    s + 1 == n_stages,
                    &backend,
                )?;
                let heads_kv = part(a.n_head_kv, tp, i).len() as u32;
                let (kv, kv_bytes) = alloc_kv_layers(
                    a,
                    layers.clone(),
                    heads_kv,
                    rows,
                    &backend,
                    opts.kv,
                    opts.kv_rotation,
                )?;
                ranks[r] = Some(RankData {
                    backend,
                    layers: layers.clone(),
                    weights,
                    weights_bytes,
                    kv,
                    kv_bytes,
                    vocab,
                    state: None,
                });
            }
        }
        let ranks = ranks
            .into_iter()
            .map(|r| r.ok_or_else(|| ModelError::Invalid("a rank is in no stage".into())))
            .collect::<Result<Vec<_>>>()?;
        Ok(Ranked {
            topo: topo.clone(),
            exec,
            ranks,
            n_layers,
            step: None,
            state_key: None,
            totals: ExecCounters::default(),
        })
    }

    /// Whether K and V are rotated (every rank agrees).
    pub(super) fn kv_rotated(&self) -> (bool, bool) {
        let kv = &self.ranks[0].kv;
        (kv.rot_k.is_some(), kv.rot_v.is_some())
    }

    pub(super) fn drop_step(&mut self) {
        self.step = None;
    }

    pub(super) fn n_ranks(&self) -> usize {
        self.ranks.len()
    }

    pub(super) fn comm_path(&self) -> &'static str {
        self.exec.ranks().comm_path()
    }

    pub(super) fn counters(&self) -> &ExecCounters {
        &self.totals
    }

    pub(super) fn reset_counters(&mut self) {
        self.totals = ExecCounters::default();
    }

    pub(super) fn memory(&self) -> Vec<MemoryUse> {
        let mut out = Vec::new();
        for (r, rd) in self.ranks.iter().enumerate() {
            let tier = if rd.backend.is_cpu() { "ram" } else { "vram" };
            let device = if self.ranks.len() > 1 {
                format!("{}#{r}", rd.backend.name())
            } else {
                rd.backend.name()
            };
            let mut push = |kind, capacity: usize| {
                out.push(MemoryUse {
                    device: device.clone(),
                    tier,
                    kind,
                    format: (kind == "kv").then(|| rd.kv.fmt.to_string()),
                    capacity: capacity as u64,
                    used: capacity as u64,
                })
            };
            push("weights", rd.weights_bytes);
            push("kv", rd.kv_bytes);
            push("workspace", self.exec.buffer_size(r));
        }
        out
    }

    /// Name and tier of the rank that reports the pool's blocks.
    pub(super) fn primary(&self) -> (String, &'static str) {
        let b = &self.ranks[0].backend;
        (b.name(), if b.is_cpu() { "ram" } else { "vram" })
    }

    /// The tensors of the cache in snapshot order, all K layers then all V layers; each is the
    /// slices of one layer across the ranks of its stage, in head order.
    pub(super) fn kv_groups(&self) -> Vec<Vec<Tensor<'_>>> {
        let mut out = Vec::new();
        for v in [false, true] {
            for il in 0..self.n_layers {
                let stage = self.topo.stage_of_layer(il);
                out.push(
                    stage
                        .ranks
                        .iter()
                        .map(|&r| {
                            let rd = &self.ranks[r];
                            let at = (il - rd.layers.start) as usize;
                            if v {
                                rd.kv.v[at].get()
                            } else {
                                rd.kv.k[at].get()
                            }
                        })
                        .collect(),
                );
            }
        }
        out
    }

    /// Copies KV rows on every rank.
    pub(super) fn copy_rows(&self, copies: &[(usize, usize, usize)]) -> Result<()> {
        for rd in &self.ranks {
            copy_rows_on(&rd.kv, &rd.backend, copies)?;
        }
        Ok(())
    }

    /// Runs one step of `plan` over `tokens` and reads the logits into `out`, `n_out` rows of
    /// `n_vocab`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run(
        &mut self,
        h: &DenseHparams,
        plan: &StepPlan,
        tokens: &[i32],
        key: (usize, usize, usize),
        reuse: bool,
        built: &mut u64,
        telemetry: Option<&tetraeder_telemetry::Metrics>,
        out: &mut Vec<f32>,
    ) -> Result<()> {
        if self.state_key != Some(key) {
            // The kept graph holds the old state's context: drop it first, then the buffers.
            self.step = None;
            for rd in &mut self.ranks {
                rd.state = None;
            }
            for rd in &mut self.ranks {
                rd.state = Some(alloc_state(key, &rd.backend)?);
            }
            self.state_key = Some(key);
        }
        for rd in &self.ranks {
            let st = rd.state.as_ref().expect("state allocated above");
            st.tokens.get().set_i32(tokens)?;
            st.pos.get().set_i32(&plan.pos)?;
            st.rows.get().set_i32(&plan.rows)?;
            st.mask.get().set_f16_bits(&plan.mask)?;
            st.outputs.get().set_i32(&plan.outputs)?;
        }

        let reusable = reuse && self.step.as_ref().is_some_and(|s| s.key == key);
        let fresh;
        let step: &Step = if reusable {
            self.step.as_ref().expect("checked above")
        } else {
            self.step = None;
            fresh = self.build(h, key)?;
            *built += 1;
            if reuse {
                self.step.insert(fresh)
            } else {
                &fresh
            }
        };

        let graphs: Vec<RankGraph<'_>> = step
            .graphs
            .iter()
            .enumerate()
            .map(|(rank, g)| RankGraph {
                rank,
                graph: g.get(),
                cuts: step.cuts[rank].clone(),
            })
            .collect();
        let ops: Vec<Op<'_>> = step
            .ops
            .iter()
            .map(|op| match op {
                StepOp::Compute { rank, range } => Op::Compute {
                    rank: *rank,
                    range: *range,
                },
                StepOp::AllReduce(ts) => Op::AllReduce {
                    tensors: ts.iter().map(|(r, t)| (*r, t.get())).collect(),
                },
                StepOp::Copy { src, dst } => Op::Copy {
                    src: (src.0, src.1.get()),
                    dst: (dst.0, dst.1.get()),
                },
            })
            .collect();
        let program = Program {
            graphs,
            ops,
            outputs: step.logits.iter().map(|(r, _)| *r).collect(),
        };
        let res = self.exec.run(&program).map_err(exec_err);
        let c = self.exec.take_counters();
        if let Some(m) = telemetry {
            c.record(m);
        }
        absorb(&mut self.totals, &c);
        res?;

        let n_vocab = h.attn.n_vocab as usize;
        if let [(_, t)] = step.logits.as_slice() {
            t.get().get_f32_into(out)?;
        } else {
            let n_out = key.2;
            out.clear();
            out.resize(n_out * n_vocab, 0.0);
            let mut slice = Vec::new();
            for (r, t) in &step.logits {
                t.get().get_f32_into(&mut slice)?;
                let v = &self.ranks[*r].vocab;
                for row in 0..n_out {
                    let from = &slice[row * v.len()..(row + 1) * v.len()];
                    out[row * n_vocab + v.start..row * n_vocab + v.end].copy_from_slice(from);
                }
            }
        }
        Ok(())
    }

    /// Builds the rank graphs and the program of a step.
    fn build(&self, h: &DenseHparams, key: (usize, usize, usize)) -> Result<Step> {
        let a = &h.attn;
        let (n, n_kv, n_out) = key;
        let nr = self.ranks.len();
        let n_embd = i64::from(a.n_embd);
        let ctxs = self
            .ranks
            .iter()
            .map(|rd| {
                let nodes = nodes_for(rd.layers.len());
                Context::new(nodes + 8, nodes).map(Rc::new)
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let cx: Vec<&Context> = ctxs.iter().map(|c| &**c).collect();
        let mut gs = Vec::with_capacity(nr);
        for (rd, c) in self.ranks.iter().zip(cx.iter().copied()) {
            gs.push(c.new_graph(nodes_for(rd.layers.len())));
        }
        let mut cuts: Vec<Vec<usize>> = vec![Vec::new(); nr];
        let mut ops: Vec<StepOp> = Vec::new();
        let mut logits: Vec<(usize, OwnedTensor)> = Vec::new();
        let mut incoming: Vec<Option<Tensor<'_>>> = vec![None; nr];
        let st_of = |r: usize| self.ranks[r].state.as_ref().expect("state allocated");

        let n_stages = self.topo.stages.len();
        for (s, stage) in self.topo.stages.iter().enumerate() {
            let tp = stage.ranks.len();
            let last_stage = s + 1 == n_stages;
            let layers = self.ranks[stage.ranks[0]].layers.clone();
            let mut xs: Vec<Tensor<'_>> = Vec::with_capacity(tp);
            for &r in &stage.ranks {
                xs.push(match incoming[r].take() {
                    Some(hop) => hop,
                    None => {
                        let w = self.ranks[r].weights.tok_embd.as_ref().ok_or_else(|| {
                            ModelError::Invalid("the first stage holds no embedding".into())
                        })?;
                        cx[r].get_rows(w.get(), st_of(r).tokens.get())
                    }
                });
            }
            for il in layers.clone() {
                let select = last_stage && il + 1 == layers.end && n_out < n;
                let mut parts: Vec<Tensor<'_>> = Vec::with_capacity(tp);
                for (i, &r) in stage.ranks.iter().enumerate() {
                    let rd = &self.ranks[r];
                    let at = (il - rd.layers.start) as usize;
                    let st = st_of(r);
                    let step = AttnStep {
                        pos: st.pos.get(),
                        rows: st.rows.get(),
                        mask: st.mask.get(),
                        n_tokens: n,
                        n_kv,
                    };
                    parts.push(attention_out(
                        a,
                        cx[r],
                        &mut gs[r],
                        &rd.weights.layers[at].attn,
                        &rd.kv,
                        at,
                        &step,
                        xs[i],
                    ));
                }
                if tp > 1 {
                    boundary(&mut gs, &mut cuts, &mut ops, &ctxs, &stage.ranks, &parts)?;
                }
                let mut ffn_inps: Vec<Tensor<'_>> = Vec::with_capacity(tp);
                let mut downs: Vec<Tensor<'_>> = Vec::with_capacity(tp);
                for (i, &r) in stage.ranks.iter().enumerate() {
                    let rd = &self.ranks[r];
                    let lw = &rd.weights.layers[(il - rd.layers.start) as usize];
                    let c = cx[r];
                    let mut ffn_inp = c.add(parts[i], xs[i]);
                    if select {
                        // libllama selects the output rows after attention and before the
                        // residual add; a row-wise add commutes with the selection.
                        ffn_inp = c.get_rows(ffn_inp, st_of(r).outputs.get());
                    }
                    let cur = rms_norm_mul(c, ffn_inp, lw.ffn_norm.get(), a.rms_eps);
                    let ffn =
                        c.swiglu_split(c.mul_mat(lw.gate.get(), cur), c.mul_mat(lw.up.get(), cur));
                    ffn_inps.push(ffn_inp);
                    downs.push(c.mul_mat(lw.down.get(), ffn));
                }
                if tp > 1 {
                    boundary(&mut gs, &mut cuts, &mut ops, &ctxs, &stage.ranks, &downs)?;
                }
                for (i, &r) in stage.ranks.iter().enumerate() {
                    xs[i] = cx[r].add(downs[i], ffn_inps[i]);
                }
            }

            if last_stage {
                for (i, &r) in stage.ranks.iter().enumerate() {
                    let rd = &self.ranks[r];
                    let c = cx[r];
                    let norm = rd.weights.out_norm.as_ref().ok_or_else(|| {
                        ModelError::Invalid("the last stage holds no output norm".into())
                    })?;
                    let cur = rms_norm_mul(c, xs[i], norm.get(), a.rms_eps);
                    let head = rd
                        .weights
                        .head
                        .as_ref()
                        .or(rd.weights.tok_embd.as_ref())
                        .ok_or_else(|| ModelError::Invalid("no output head".into()))?;
                    let lg = c.mul_mat(head.get(), cur).set_output();
                    gs[r].expand(lg);
                    cuts[r].push(gs[r].n_nodes());
                    ops.push(StepOp::Compute {
                        rank: r,
                        range: cuts[r].len() - 1,
                    });
                    logits.push((r, ctxs[r].keep(lg)?));
                }
            } else {
                let next = &self.topo.stages[s + 1];
                for (i, &r) in stage.ranks.iter().enumerate() {
                    let to = next.ranks[i];
                    gs[r].expand(xs[i]);
                    cuts[r].push(gs[r].n_nodes());
                    ops.push(StepOp::Compute {
                        rank: r,
                        range: cuts[r].len() - 1,
                    });
                    // The destination is an input of the next stage's graph, same layout.
                    let hop = cx[to].new_tensor(Type::F32, &[n_embd, n as i64]);
                    hop.set_name("hop");
                    incoming[to] = Some(hop);
                    ops.push(StepOp::Copy {
                        src: (r, ctxs[r].keep(xs[i])?),
                        dst: (to, ctxs[to].keep(hop)?),
                    });
                }
            }
        }

        let mut graphs = Vec::with_capacity(nr);
        for (r, g) in gs.into_iter().enumerate() {
            let rd = &self.ranks[r];
            let st = st_of(r);
            let mut deps: Vec<&OwnedTensor> = vec![&rd.weights.anchor, &st.pos];
            deps.extend(rd.kv.k.first());
            deps.extend(rd.kv.v.first());
            deps.extend(rd.kv.rot_k.iter().chain(&rd.kv.rot_v));
            graphs.push(ctxs[r].keep_graph(g, &deps)?);
        }
        Ok(Step {
            key,
            graphs,
            cuts,
            ops,
            logits,
        })
    }
}

/// Ends the range of every rank of a group at `tensors[i]` (rank `ranks[i]`'s partial sum) and
/// adds the compute ops and the all-reduce.
fn boundary<'a>(
    gs: &mut [tetraeder_ggml_sys::ggml::Graph<'a>],
    cuts: &mut [Vec<usize>],
    ops: &mut Vec<StepOp>,
    ctxs: &[Rc<Context>],
    ranks: &[usize],
    tensors: &[Tensor<'a>],
) -> Result<()> {
    let mut owned = Vec::with_capacity(ranks.len());
    for (i, &r) in ranks.iter().enumerate() {
        gs[r].expand(tensors[i]);
        cuts[r].push(gs[r].n_nodes());
        ops.push(StepOp::Compute {
            rank: r,
            range: cuts[r].len() - 1,
        });
        owned.push((r, ctxs[r].keep(tensors[i])?));
    }
    ops.push(StepOp::AllReduce(owned));
    Ok(())
}

fn absorb(total: &mut ExecCounters, c: &ExecCounters) {
    if total.ranges.len() < c.ranges.len() {
        total.ranges.resize(c.ranges.len(), 0);
    }
    for (a, b) in total.ranges.iter_mut().zip(&c.ranges) {
        *a += b;
    }
    for (k, v) in &c.collective_bytes {
        *total.collective_bytes.entry(*k).or_default() += v;
    }
    total.allocations += c.allocations;
}

/// What the executor cannot run, with every reason.
fn check_topology(topo: &Topology, h: &DenseHparams, opts: &DenseOptions) -> Result<()> {
    let a = &h.attn;
    let mut errs: Vec<String> = Vec::new();
    if topo.n_layer != a.n_layer {
        errs.push(format!(
            "the topology has {} layers, the model {}",
            topo.n_layer, a.n_layer
        ));
    }
    if opts.layer_limit == Some(0) {
        errs.push("layer_limit must be at least 1".into());
    }
    if opts.layer_limit.is_some() && topo.stages.len() > 1 {
        errs.push("layer_limit needs a one-stage topology".into());
    }
    let mut seen = BTreeSet::new();
    for (s, st) in topo.stages.iter().enumerate() {
        let tp = st.ranks.len() as u32;
        if tp == 0 {
            errs.push(format!("stage {s} has no ranks"));
            continue;
        }
        if a.n_head % tp != 0 || a.n_head_kv % tp != 0 {
            errs.push(format!(
                "stage {s}: tp = {tp} does not divide {} heads and {} KV heads",
                a.n_head, a.n_head_kv
            ));
        }
        if let Some(next) = topo.stages.get(s + 1)
            && next.ranks.len() != st.ranks.len()
        {
            errs.push(format!(
                "stages {s} and {}: tp {} and {} differ, a hop pairs rank i with rank i",
                s + 1,
                st.ranks.len(),
                next.ranks.len()
            ));
        }
        for &r in &st.ranks {
            if r >= topo.ranks.len() {
                errs.push(format!("stage {s}: rank {r} is not in the topology"));
            } else if !seen.insert(r) {
                errs.push(format!("rank {r} is in two stages"));
            }
        }
    }
    if seen.len() != topo.ranks.len() {
        errs.push("a rank is in no stage".into());
    }
    if errs.is_empty() {
        Ok(())
    } else {
        Err(ModelError::Invalid(errs.join("; ")))
    }
}

/// Loads the weights of rank `i` of a stage of degree `tp` (ADR 0070): `wq`, `wk`, `wv` by
/// the rank's heads (rows `ne1`), `wo` by the same heads (`ne0`), `ffn_gate` and `ffn_up` by the
/// block share of `ffn_down` (`ne1`), `ffn_down` by that share (`ne0`), the head by vocabulary.
/// With `tp = 1` nothing is cut. Returns the weights, their size and the head's vocab range.
fn load_rank(
    gguf: &Gguf,
    h: &DenseHparams,
    (tp, i): (u32, u32),
    layers: Range<u32>,
    embed: bool,
    head: bool,
    backend: &Backend,
) -> Result<(RankWeights, usize, Range<usize>)> {
    let a = &h.attn;
    let hd = u64::from(a.head_dim);
    let cut = tp > 1;
    let heads = |n: u32| {
        let r = if cut { part(n, tp, i) } else { 0..n };
        u64::from(r.start) * hd..u64::from(r.end) * hd
    };
    let (q_rows, kv_rows) = (heads(a.n_head), heads(a.n_head_kv));
    let mut ld = WeightLoader::new(gguf, layers.len() * 11 + 4)?;
    let mut layer_w = Vec::with_capacity(layers.len());
    for il in layers {
        let nm = |w: &str| names::layer(il, w);
        let ff = if cut {
            let bs = gguf.tensor(&nm("ffn_down"))?.ty.block_size() as u64;
            Some(block_part(u64::from(h.n_ff), bs, tp, i, il)?)
        } else {
            None
        };
        let only = |r: &Range<u64>| cut.then(|| r.clone());
        layer_w.push(Layer {
            attn: AttnWeights {
                attn_norm: ld.tensor(&nm("attn_norm"), None, None)?,
                wq: ld.tensor(&nm("attn_q"), None, only(&q_rows))?,
                wk: ld.tensor(&nm("attn_k"), None, only(&kv_rows))?,
                wv: ld.tensor(&nm("attn_v"), None, only(&kv_rows))?,
                wo: ld.tensor(&nm("attn_output"), only(&q_rows), None)?,
                q_norm: ld.tensor(&nm("attn_q_norm"), None, None)?,
                k_norm: ld.tensor(&nm("attn_k_norm"), None, None)?,
            },
            ffn_norm: ld.tensor(&nm("ffn_norm"), None, None)?,
            gate: ld.tensor(&nm("ffn_gate"), None, ff.clone())?,
            up: ld.tensor(&nm("ffn_up"), None, ff.clone())?,
            down: ld.tensor(&nm("ffn_down"), ff, None)?,
        });
    }
    let tok_embd = if embed {
        Some(ld.tensor(names::TOK_EMBD, None, None)?)
    } else {
        None
    };
    let n_vocab = u64::from(a.n_vocab);
    let (mut out_norm, mut head_w, mut vocab) = (None, None, 0..0);
    if head {
        out_norm = Some(ld.tensor(names::OUTPUT_NORM, None, None)?);
        let rows = if cut {
            row_part(n_vocab, tp, i)
        } else {
            0..n_vocab
        };
        vocab = rows.start as usize..rows.end as usize;
        // A tied head reads the token embedding; a rank that holds all of it uses it as is.
        let whole_embd = h.tied_output && embed && rows == (0..n_vocab);
        if !whole_embd {
            let name = if h.tied_output {
                names::TOK_EMBD
            } else {
                names::OUTPUT
            };
            head_w = Some(ld.tensor(name, None, Some(rows))?);
        }
    }
    let anchor = layer_w
        .first()
        .map(|l| l.ffn_norm.clone())
        .ok_or_else(|| ModelError::Invalid("a stage with no layers".into()))?;
    let bytes = ld.finish(backend)?;
    Ok((
        RankWeights {
            tok_embd,
            out_norm,
            head: head_w,
            layers: layer_w,
            anchor,
        },
        bytes,
        vocab,
    ))
}
