//! Attention shared by the model builders (ADR 0065, 0066): norms, RoPE parameters, the KV
//! tensors with their Hadamard rotation, the causal mask and the attention block of one layer.
//! Extracted from `qwen3moe.rs` with the op sequence unchanged (phase 4, C1.5).

use crate::qwen3moe::{ModelError, Result};
use tetraeder_ggml_sys::ggml::{
    Backend, Context, Graph, OwnedTensor, RopeParams, Sched, Tensor, Type, hadamard_matrix,
};
use tetraeder_kvcache::SeqAttn;
use tetraeder_plan::{Format, KvFormat};

use crate::convert::ggml_type;
use std::ops::Range;
use std::rc::Rc;

/// KV rows are padded to this many for the attention kernels.
pub const KV_PAD: usize = 256;
/// F16 bits of minus infinity, the masked entries.
pub const F16_NEG_INF: u16 = 0xFC00;

/// Query rows a flash-attention mask buffer is padded to. At the pin (b12426), ggml-cuda scans
/// the mask of a call with at least 1024 queries in tiles of up to 64 rows
/// (`flash_attn_mask_to_KV_max`, fattn-common.cuh:669-724, launched at 1117-1133; tiles of
/// `64/ncols2` rows, fattn.cu:190 and fattn-tile.cuh:1161) and reads the last tile whole. With a
/// mask of exactly `n_tokens` rows that read runs past the buffer: an illegal memory access on a
/// 1207-token step (session 20261116pc). libllama never sends 1024 queries (ubatch 512).
pub const MASK_ROW_PAD: usize = 64;

/// Elements the last attention call over a `[n_kv, n_tokens]` mask block may read past it.
pub(crate) fn mask_tail(n_tokens: usize, n_kv: usize) -> usize {
    (n_tokens.div_ceil(MASK_ROW_PAD) * MASK_ROW_PAD - n_tokens) * n_kv
}

/// The `[n_kv, n_tokens]` mask of a step as a view of the state's padded mask buffer.
pub(crate) fn mask_view<'a>(
    ctx: &'a Context,
    buf: Tensor<'a>,
    n_kv: usize,
    n_tokens: usize,
) -> Tensor<'a> {
    let row = buf.nb()[1];
    ctx.view(buf, &[n_kv as i64, n_tokens as i64], &[row], 0)
}

/// The hyperparameters attention and the KV cache read.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttnHparams {
    pub n_layer: u32,
    pub n_embd: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub head_dim: u32,
    pub n_rot: u32,
    pub n_ctx_train: u32,
    pub n_vocab: u32,
    pub rms_eps: f32,
    pub rope_freq_base: f32,
}

/// One layer's attention weights.
pub(crate) struct AttnWeights {
    pub attn_norm: OwnedTensor,
    pub wq: OwnedTensor,
    pub wk: OwnedTensor,
    pub wv: OwnedTensor,
    pub wo: OwnedTensor,
    pub q_norm: OwnedTensor,
    pub k_norm: OwnedTensor,
}

/// Contiguous per-layer KV cache: `[n_embd_kv, n_rows]` for K and for V, in the types of `fmt`.
pub(crate) struct Kv {
    pub k: Vec<OwnedTensor>,
    pub v: Vec<OwnedTensor>,
    pub n_rows: usize,
    pub fmt: KvFormat,
    /// Hadamard matrices for K (and Q) and for V, F32 `[n, n]` in a context of their own so the
    /// KV buffer stays exactly the cache. `None`: that side is not rotated.
    pub rot_k: Option<OwnedTensor>,
    pub rot_v: Option<OwnedTensor>,
}

/// Inputs of one step's attention. `rows` are the KV rows written (set_rows indices), `pos` the
/// RoPE positions; the MoE builder passes its `pos` tensor for both until C4.
pub(crate) struct AttnStep<'a> {
    pub pos: Tensor<'a>,
    pub rows: Tensor<'a>,
    pub mask: Tensor<'a>,
    pub n_tokens: usize,
    pub n_kv: usize,
}

/// One sequence's tokens in a multi-sequence step (C4 and C5 use it).
pub struct SeqEval<'a> {
    pub seq: u32,
    pub tokens: &'a [i32],
    pub all_logits: bool,
}

/// The ggml types of a KV format; a quantized type needs the head size to be a whole number of
/// blocks, because attention views one head's row at a time.
pub(crate) fn kv_types(h: &AttnHparams, fmt: KvFormat) -> Result<(Type, Type)> {
    if !fmt.is_allowed() {
        return Err(ModelError::Invalid(format!(
            "KV format {fmt} is not one of the supported pairs"
        )));
    }
    let ty = |f: Format| {
        let t =
            ggml_type(f).ok_or_else(|| ModelError::Invalid(format!("no ggml type for {f:?}")))?;
        if h.head_dim as i64 % t.block_size() != 0 {
            return Err(ModelError::Invalid(format!(
                "KV format {fmt}: head size {} is not a multiple of the {f:?} block size {}",
                h.head_dim,
                t.block_size()
            )));
        }
        Ok(t)
    };
    Ok((ty(fmt.k)?, ty(fmt.v)?))
}

/// Side of the Hadamard matrix for K, as libllama picks it (llama-kv-cache.cpp:329-337): the
/// largest power of two dividing the head size, at least 64.
fn rot_size_k(head_dim: usize) -> usize {
    let mut n = 64;
    while head_dim.is_multiple_of(2 * n) {
        n *= 2;
    }
    n
}

/// V uses the smallest matrix (llama-kv-cache.cpp:338-345).
const ROT_SIZE_V: usize = 64;

/// The Hadamard matrices of a KV format: libllama rotates a side when its type is quantized and
/// the head size is a multiple of 64 (llama-kv-cache.cpp:310-327). `want` overrides the quantized
/// test for both sides; asking for rotation with another head size is an error.
fn alloc_rot(
    h: &AttnHparams,
    types: (Type, Type),
    want: Option<bool>,
    backend: &Backend,
) -> Result<(Option<OwnedTensor>, Option<OwnedTensor>)> {
    let hd = h.head_dim as usize;
    if want == Some(true) && !hd.is_multiple_of(64) {
        return Err(ModelError::Invalid(format!(
            "KV rotation needs a head size that is a multiple of 64, not {hd}"
        )));
    }
    let on = |t: Type| hd.is_multiple_of(64) && want.unwrap_or_else(|| t.is_quantized());
    let (k, v) = (on(types.0), on(types.1));
    if !k && !v {
        return Ok((None, None));
    }
    let ctx = Rc::new(Context::new(2, 0)?);
    let mk = |n: usize, name: &str| {
        let t = ctx.new_persistent(Type::F32, &[n as i64, n as i64]);
        t.get().set_name(name);
        (n, t)
    };
    let rk = k.then(|| mk(rot_size_k(hd), "kv_rot_k"));
    let rv = v.then(|| mk(ROT_SIZE_V, "kv_rot_v"));
    ctx.alloc_tensors(backend)?;
    let fill = |r: Option<(usize, OwnedTensor)>| -> Result<Option<OwnedTensor>> {
        r.map(|(n, t)| {
            t.get().set_f32(&hadamard_matrix(n))?;
            Ok::<_, ModelError>(t)
        })
        .transpose()
    };
    Ok((fill(rk)?, fill(rv)?))
}

/// Allocates the KV tensors of `n_rows` rows per layer. Returns the cache and its size in bytes.
pub(crate) fn alloc_kv(
    h: &AttnHparams,
    n_rows: usize,
    backend: &Backend,
    fmt: KvFormat,
    rotation: Option<bool>,
) -> Result<(Kv, usize)> {
    alloc_kv_layers(h, 0..h.n_layer, h.n_head_kv, n_rows, backend, fmt, rotation)
}

/// The KV tensors of one rank (ADR 0070): the layers `layers` and `n_head_kv` KV heads of each,
/// indexed by `il - layers.start`. `alloc_kv` is the whole model on one backend.
pub(crate) fn alloc_kv_layers(
    h: &AttnHparams,
    layers: Range<u32>,
    n_head_kv: u32,
    n_rows: usize,
    backend: &Backend,
    fmt: KvFormat,
    rotation: Option<bool>,
) -> Result<(Kv, usize)> {
    let (tk, tv) = kv_types(h, fmt)?;
    let (rot_k, rot_v) = alloc_rot(h, (tk, tv), rotation, backend)?;
    let ctx = Rc::new(Context::new(2 * layers.len() + 4, 0)?);
    let n_embd_kv = (n_head_kv * h.head_dim) as i64;
    let mk = |kind: &str, ty: Type, il: u32| {
        let t = ctx.new_persistent(ty, &[n_embd_kv, n_rows as i64]);
        t.get().set_name(&format!("{kind}_cache_{il}"));
        t
    };
    let k = layers.clone().map(|il| mk("k", tk, il)).collect();
    let v = layers.map(|il| mk("v", tv, il)).collect();
    let bytes = ctx.alloc_tensors(backend)?;
    // Padding rows are masked, but 0 * NaN is NaN: never leave them uninitialized.
    ctx.clear_buffer(0);
    Ok((
        Kv {
            k,
            v,
            n_rows,
            fmt,
            rot_k,
            rot_v,
        },
        bytes,
    ))
}

/// NeoX RoPE as Qwen3 uses it.
pub(crate) fn rope_params(h: &AttnHparams) -> RopeParams {
    RopeParams {
        n_dims: h.n_rot as i32,
        mode: 2, // NeoX
        n_ctx_orig: h.n_ctx_train as i32,
        freq_base: h.rope_freq_base,
        freq_scale: 1.0,
        ext_factor: 0.0,
        attn_factor: 1.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
    }
}

/// `rms_norm(x) * w`.
pub(crate) fn rms_norm_mul<'a>(
    ctx: &'a Context,
    x: Tensor<'a>,
    w: Tensor<'a>,
    eps: f32,
) -> Tensor<'a> {
    ctx.mul(ctx.rms_norm(x, eps), w)
}

/// Layer `il`: RMS norm, attention with Q/K norm, RoPE, KV rotation, the KV write and flash
/// attention, `wo` and the residual. Returns `ffn_inp`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention<'a>(
    h: &AttnHparams,
    ctx: &'a Context,
    g: &mut Graph<'a>,
    lw: &'a AttnWeights,
    kv: &'a Kv,
    il: usize,
    st: &AttnStep<'a>,
    x: Tensor<'a>,
) -> Tensor<'a> {
    attention_seqs(h, ctx, g, lw, kv, il, st, &[], x)
}

/// [`attention`] with `seqs` non-empty: one flash attention per input over its own rows
/// (`SeqAttn`), the mask being the inputs' blocks back to back (1-D), and the outputs joined in
/// token order. Empty `seqs`: the masked attention over rows `0..n_kv`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_seqs<'a>(
    h: &AttnHparams,
    ctx: &'a Context,
    g: &mut Graph<'a>,
    lw: &'a AttnWeights,
    kv: &'a Kv,
    il: usize,
    st: &AttnStep<'a>,
    seqs: &[SeqAttn],
    x: Tensor<'a>,
) -> Tensor<'a> {
    ctx.add(attention_seqs_out(h, ctx, g, lw, kv, il, st, seqs, x), x)
}

/// [`attention`] without the residual: the output of `wo`, which a tensor-parallel rank holds as
/// a partial sum (ADR 0070). `h.n_head` and `h.n_head_kv` are the rank's local head counts, and
/// `lw` and `kv` hold its slices.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_out<'a>(
    h: &AttnHparams,
    ctx: &'a Context,
    g: &mut Graph<'a>,
    lw: &'a AttnWeights,
    kv: &'a Kv,
    il: usize,
    st: &AttnStep<'a>,
    x: Tensor<'a>,
) -> Tensor<'a> {
    attention_seqs_out(h, ctx, g, lw, kv, il, st, &[], x)
}

/// [`attention_seqs`] without the residual.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_seqs_out<'a>(
    h: &AttnHparams,
    ctx: &'a Context,
    g: &mut Graph<'a>,
    lw: &'a AttnWeights,
    kv: &'a Kv,
    il: usize,
    st: &AttnStep<'a>,
    seqs: &[SeqAttn],
    x: Tensor<'a>,
) -> Tensor<'a> {
    let n = st.n_tokens as i64;
    let hd = h.head_dim as i64;
    let (nh, nkv) = (lw.wq.get().ne()[1] / hd, lw.wk.get().ne()[1] / hd);
    let rope = rope_params(h);

    let cur = rms_norm_mul(ctx, x, lw.attn_norm.get(), h.rms_eps);
    let q = ctx.reshape(ctx.mul_mat(lw.wq.get(), cur), &[hd, nh, n]);
    let k = ctx.reshape(ctx.mul_mat(lw.wk.get(), cur), &[hd, nkv, n]);
    let v = ctx.mul_mat(lw.wv.get(), cur);
    let q = rms_norm_mul(ctx, q, lw.q_norm.get(), h.rms_eps);
    let q = ctx.rope_ext(q, st.pos, &rope);
    let k = rms_norm_mul(ctx, k, lw.k_norm.get(), h.rms_eps);
    let k = ctx.rope_ext(k, st.pos, &rope);
    // Rotation as in libllama's `build_attn` (llama-graph.cpp:3246-3253): Q and K after RoPE with
    // the K matrix, V with the V matrix. Q.K is unchanged because the matrix is orthonormal.
    let (q, k) = match &kv.rot_k {
        Some(r) => (
            ctx.mul_mat_hadamard(q, r.get()),
            ctx.mul_mat_hadamard(k, r.get()),
        ),
        None => (q, k),
    };
    let v = match &kv.rot_v {
        Some(r) => ctx.mul_mat_hadamard(v, r.get()),
        None => v,
    };

    // Store K and V in the cache rows of these positions; the cache must be written before the
    // attention reads it, so the stores enter the graph first.
    let (kc, vc) = (kv.k[il].get(), kv.v[il].get());
    // `set_rows` with row indices, as libllama does: unlike a copy into a view at `n_past`, the
    // node is the same every step, so ggml-cuda can replay a captured CUDA graph (ADR 0036).
    let row = kc.nb()[1];
    let n_embd_kv = hd * nkv;
    for (src, cache) in [(k, kc), (v, vc)] {
        let src = ctx.reshape(src, &[n_embd_kv, n]);
        g.expand(ctx.set_rows(cache, src, st.rows));
    }

    let scale = 1.0 / (hd as f32).sqrt();
    // K and V rows `start..start+n_kv`; the head stride is one head's row in the cache type
    // (2 bytes per element for F16).
    let kv_view = |start: usize, n_kv: usize| {
        let n_kv = n_kv as i64;
        let vrow = vc.nb()[1];
        (
            ctx.view(
                kc,
                &[hd, n_kv, nkv],
                &[row, kc.ty().row_size(hd)],
                start * row,
            ),
            ctx.view(
                vc,
                &[hd, n_kv, nkv],
                &[vrow, vc.ty().row_size(hd)],
                start * vrow,
            ),
        )
    };
    let attn = if seqs.is_empty() {
        let (kview, vview) = kv_view(0, st.n_kv);
        let q = ctx.permute(q, [0, 2, 1, 3]);
        ctx.flash_attn_ext(q, kview, vview, st.mask, scale)
    } else {
        let qb = q.nb();
        let mb = st.mask.nb()[0];
        let parts: Vec<_> = seqs
            .iter()
            .map(|s| {
                let qs = ctx.view(
                    q,
                    &[hd, nh, s.n_tokens as i64],
                    &[qb[1], qb[2]],
                    s.first_token * qb[2],
                );
                let qs = ctx.permute(qs, [0, 2, 1, 3]);
                let (kview, vview) = kv_view(s.kv_start, s.n_kv);
                let m = ctx.view(
                    st.mask,
                    &[s.n_kv as i64, s.n_tokens as i64],
                    &[s.n_kv * mb],
                    s.mask_offset * mb,
                );
                ctx.flash_attn_ext(qs, kview, vview, m, scale)
            })
            .collect();
        concat_tokens(ctx, &parts)
    };
    let attn = ctx.reshape(attn, &[hd * nh, n]);
    // Undo the V rotation (llama-graph.cpp:3291-3293); the matrix is its own inverse.
    let attn = match &kv.rot_v {
        Some(r) => ctx.mul_mat_hadamard(attn, r.get()),
        None => attn,
    };
    ctx.mul_mat(lw.wo.get(), attn)
}

/// Joins attention outputs `[d, n_head, n_i]` along the tokens, pairwise so each element is
/// copied about log2(parts) times.
fn concat_tokens<'a>(ctx: &'a Context, parts: &[Tensor<'a>]) -> Tensor<'a> {
    match parts {
        [one] => *one,
        _ => {
            let (a, b) = parts.split_at(parts.len() / 2);
            ctx.concat(concat_tokens(ctx, a), concat_tokens(ctx, b), 2)
        }
    }
}

/// `[n_kv, n]` F16 bits as libllama builds it (no row padding): row i is open (0.0) on columns
/// `0..=n_past+i` and `-inf` after.
pub fn causal_mask(n_past: usize, n: usize, n_kv: usize) -> Vec<u16> {
    let mut mask = vec![0u16; n_kv * n];
    for i in 0..n {
        for j in (n_past + i + 1)..n_kv {
            mask[i * n_kv + j] = F16_NEG_INF;
        }
    }
    mask
}

/// The copies that move rows, checked against the cache size.
fn valid_copies(kv: &Kv, copies: &[(usize, usize, usize)]) -> Result<Vec<(usize, usize, usize)>> {
    let copies: Vec<_> = copies.iter().copied().filter(|c| c.2 > 0).collect();
    for &(s, d, n) in &copies {
        if s + n > kv.n_rows || d + n > kv.n_rows {
            return Err(ModelError::Invalid(format!(
                "KV row copy {s}..{} to {d}..{} outside {} rows",
                s + n,
                d + n,
                kv.n_rows
            )));
        }
        if s < d + n && d < s + n {
            return Err(ModelError::Invalid(format!(
                "KV row copy {s}..{} to {d}..{} overlaps itself",
                s + n,
                d + n
            )));
        }
    }
    Ok(copies)
}

/// `copies` cut into consecutive groups whose rows are pairwise disjoint, so each group runs as
/// one graph and the groups run in order: a slide of a run over itself (ADR 0067 history)
/// writes, in a later copy, rows an earlier one read.
fn disjoint_groups(copies: &[(usize, usize, usize)]) -> Vec<&[(usize, usize, usize)]> {
    let meet = |a: usize, na: usize, b: usize, nb: usize| a < b + nb && b < a + na;
    let clash = |&(s1, d1, n1): &(usize, usize, usize), &(s2, d2, n2): &(usize, usize, usize)| {
        meet(d1, n1, s2, n2) || meet(s1, n1, d2, n2) || meet(d1, n1, d2, n2)
    };
    let mut groups = Vec::new();
    let mut start = 0;
    for (j, c) in copies.iter().enumerate() {
        if copies[start..j].iter().any(|e| clash(e, c)) {
            groups.push(&copies[start..j]);
            start = j;
        }
    }
    if start < copies.len() {
        groups.push(&copies[start..]);
    }
    groups
}

/// `cpy` between views of the cache, every layer, K and V, in one graph of `ctx`.
fn copy_graph<'a>(
    ctx: &'a Context,
    kv: &'a Kv,
    copies: &[(usize, usize, usize)],
    nodes: usize,
) -> Graph<'a> {
    let mut g = ctx.new_graph(4 * nodes + 8);
    for t in kv.k.iter().chain(&kv.v) {
        let t = t.get();
        let (ne0, row) = (t.ne()[0], t.nb()[1]);
        for &(s, d, n) in copies {
            let n = n as i64;
            let src = ctx.view(t, &[ne0, n], &[row], s * row);
            let dst = ctx.view(t, &[ne0, n], &[row], d * row);
            g.expand(ctx.cpy(src, dst));
        }
    }
    g
}

/// Copies KV rows on the device, every layer, K and V, with `cpy` between views. Each entry is
/// `(src_row, dst_row, n_rows)`; its source and destination must not overlap. Entries apply in
/// order: those whose rows are disjoint share one graph, and an entry touching rows of an
/// earlier one waits for it in a later graph. Used by C4 and C5 for `seq_cp` and chunked moves.
pub(crate) fn copy_rows(
    kv: &Kv,
    sched: &mut Sched,
    copies: &[(usize, usize, usize)],
) -> Result<()> {
    let copies = valid_copies(kv, copies)?;
    for group in disjoint_groups(&copies) {
        let nodes = group.len() * kv.k.len() * 2;
        let ctx = Context::new(4 * nodes + 8, 4 * nodes + 8)?;
        let g = copy_graph(&ctx, kv, group, nodes);
        sched.alloc_graph(&g)?;
        sched.compute(&g)?;
    }
    Ok(())
}

/// [`copy_rows`] for a cache on one rank of the executor (no scheduler): the graph only has
/// views of the cache, so it needs no allocation. Returns when the copy is done.
pub(crate) fn copy_rows_on(
    kv: &Kv,
    backend: &Backend,
    copies: &[(usize, usize, usize)],
) -> Result<()> {
    let copies = valid_copies(kv, copies)?;
    for group in disjoint_groups(&copies) {
        let nodes = group.len() * kv.k.len() * 2;
        let ctx = Context::new(4 * nodes + 8, 4 * nodes + 8)?;
        let g = copy_graph(&ctx, kv, group, nodes);
        backend.compute_async(&g)?;
        backend.synchronize();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The loop `run` had before the extraction.
    fn old_mask(n_past: usize, n: usize, n_kv: usize) -> Vec<u16> {
        let mut mask = vec![0u16; n_kv * n];
        for i in 0..n {
            for j in (n_past + i + 1)..n_kv {
                mask[i * n_kv + j] = F16_NEG_INF;
            }
        }
        mask
    }

    #[test]
    fn causal_mask_matches_the_old_loop() {
        for n_past in [0usize, 5, 255] {
            for n in [1, 7] {
                let n_kv = (n_past + n).div_ceil(KV_PAD) * KV_PAD;
                let m = causal_mask(n_past, n, n_kv);
                assert_eq!(m, old_mask(n_past, n, n_kv), "n_past {n_past}, n {n}");
                assert_eq!(m.len(), n_kv * n);
                assert_eq!(m[n_past], 0);
                if n_past + 1 < n_kv {
                    assert_eq!(m[n_past + 1], F16_NEG_INF);
                }
            }
        }
    }

    /// hardware: cpu.
    #[test]
    fn cpu_copy_rows_moves_ranges_and_leaves_the_rest() {
        let backend = Backend::cpu(2).unwrap();
        let h = AttnHparams {
            n_layer: 2,
            n_embd: 8,
            n_head: 2,
            n_head_kv: 1,
            head_dim: 4,
            n_rot: 4,
            n_ctx_train: 64,
            n_vocab: 8,
            rms_eps: 1e-6,
            rope_freq_base: 10000.0,
        };
        let fmt = KvFormat {
            k: Format::F16,
            v: Format::F16,
        };
        let (kv, _) = alloc_kv(&h, 32, &backend, fmt, None).unwrap();
        let row = kv.k[0].get().nb()[1];
        let fill = |t: &OwnedTensor, salt: u8| {
            let b: Vec<u8> = (0..32 * row).map(|i| (i / row) as u8 ^ salt).collect();
            t.get().set_bytes(&b, 0).unwrap();
        };
        for (i, t) in kv.k.iter().chain(&kv.v).enumerate() {
            fill(t, 0x40 * (i as u8 % 4));
        }
        let before: Vec<Vec<u8>> =
            kv.k.iter()
                .chain(&kv.v)
                .map(|t| t.get().get_bytes(0, 32 * row).unwrap())
                .collect();
        let mut sched = Sched::new(std::slice::from_ref(&backend), 64, false).unwrap();
        let copies = [(0, 10, 3), (4, 20, 1), (14, 24, 5)];
        copy_rows(&kv, &mut sched, &copies).unwrap();
        for (ti, t) in kv.k.iter().chain(&kv.v).enumerate() {
            let after = t.get().get_bytes(0, 32 * row).unwrap();
            let mut want = before[ti].clone();
            for &(s, d, n) in &copies {
                want[d * row..(d + n) * row].copy_from_slice(&before[ti][s * row..(s + n) * row]);
            }
            assert_eq!(after, want, "tensor {ti}");
        }
        assert!(copy_rows(&kv, &mut sched, &[(30, 0, 5)]).is_err());

        // The same copies without a scheduler, as the sharded builders run them on one rank
        // (a GPU rank has no CPU backend to make a `Sched` of).
        for (t, b) in kv.k.iter().chain(&kv.v).zip(&before) {
            t.get().set_bytes(b, 0).unwrap();
        }
        copy_rows_on(&kv, &backend, &copies).unwrap();
        for (ti, t) in kv.k.iter().chain(&kv.v).enumerate() {
            let after = t.get().get_bytes(0, 32 * row).unwrap();
            let mut want = before[ti].clone();
            for &(s, d, n) in &copies {
                want[d * row..(d + n) * row].copy_from_slice(&before[ti][s * row..(s + n) * row]);
            }
            assert_eq!(after, want, "on tensor {ti}");
        }
        assert!(copy_rows_on(&kv, &backend, &[(30, 0, 5)]).is_err());

        // A slide of rows 8..29 down to 0 in chunks of 8: each chunk reads rows the one before it
        // wrote over, so the result is the copies applied one after another.
        let slide = [(8, 0, 8), (16, 8, 8), (24, 16, 5)];
        for on_rank in [false, true] {
            for (t, b) in kv.k.iter().chain(&kv.v).zip(&before) {
                t.get().set_bytes(b, 0).unwrap();
            }
            if on_rank {
                copy_rows_on(&kv, &backend, &slide).unwrap();
            } else {
                copy_rows(&kv, &mut sched, &slide).unwrap();
            }
            for (ti, t) in kv.k.iter().chain(&kv.v).enumerate() {
                let after = t.get().get_bytes(0, 32 * row).unwrap();
                let mut want = before[ti].clone();
                want[..21 * row].copy_from_slice(&before[ti][8 * row..29 * row]);
                assert_eq!(after, want, "slide, on rank {on_rank}, tensor {ti}");
            }
        }
        assert!(
            copy_rows(&kv, &mut sched, &[(0, 4, 8)]).is_err(),
            "a copy over itself"
        );
        assert_eq!(disjoint_groups(&slide).len(), 3);
        assert_eq!(disjoint_groups(&copies).len(), 1);
    }
}
