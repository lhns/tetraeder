//! Context, tensor handles, graphs and the ops a Qwen3 MoE forward needs.

use super::{Backend, Buffer, GgmlError, Result, Type, cstr};
use crate::raw as sys;
use std::cell::RefCell;
use std::ffi::CString;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Ids of [`Graph`]s, unique in the process.
static GRAPH_IDS: AtomicU64 = AtomicU64::new(1);

/// A ggml context with `no_alloc`: it holds tensor and graph metadata only. Tensor data lives in
/// the buffer from [`Context::alloc_tensors`] (weights, KV cache, state) or is assigned by a
/// [`super::Sched`] (graph intermediates). Tensors die with the context.
pub struct Context {
    ptr: NonNull<sys::ggml_context>,
    buffer: RefCell<Option<Buffer>>,
}

impl Context {
    /// Room for `max_tensors` tensors (leaves and op results) and one graph of `graph_nodes`.
    pub fn new(max_tensors: usize, graph_nodes: usize) -> Result<Context> {
        Self::with_graphs(max_tensors, &[graph_nodes])
    }

    /// Room for `max_tensors` tensors and one graph per entry of `graph_nodes` (a zero entry
    /// reserves nothing), for range graphs and other families of small graphs.
    pub fn with_graphs(max_tensors: usize, graph_nodes: &[usize]) -> Result<Context> {
        crate::llama::init();
        // SAFETY: pure size arithmetic.
        let mut mem = max_tensors * unsafe { sys::ggml_tensor_overhead() } + 1024;
        for &nodes in graph_nodes.iter().filter(|&&n| n > 0) {
            // SAFETY: as above.
            mem += unsafe { sys::ggml_graph_overhead_custom(nodes, false) } + 64;
        }
        let params = sys::ggml_init_params {
            mem_size: mem,
            mem_buffer: std::ptr::null_mut(),
            no_alloc: true,
        };
        // SAFETY: ggml allocates and owns the pool (mem_buffer is null).
        let ptr = unsafe { sys::ggml_init(params) };
        let ptr = NonNull::new(ptr).ok_or(GgmlError::Alloc {
            what: "ggml context",
            bytes: mem,
        })?;
        Ok(Context {
            ptr,
            buffer: RefCell::new(None),
        })
    }

    /// Allocates all tensors created so far in one buffer on `backend`'s default buffer type.
    /// The context owns the buffer. Returns its size.
    pub fn alloc_tensors(&self, backend: &Backend) -> Result<usize> {
        let buf = Buffer::alloc_ctx_tensors(self.ptr.as_ptr(), backend)?;
        let size = buf.size();
        *self.buffer.borrow_mut() = Some(buf);
        Ok(size)
    }

    /// Like [`Context::alloc_tensors`], with the buffer marked as weights: ops reading these
    /// tensors run on `backend` (see [`Buffer::set_usage_weights`]).
    pub fn alloc_weights(&self, backend: &Backend) -> Result<usize> {
        let size = self.alloc_tensors(backend)?;
        if let Some(b) = self.buffer.borrow().as_ref() {
            b.set_usage_weights();
        }
        Ok(size)
    }

    /// Like [`Context::alloc_weights`] in a CPU extra buffer type (AMX, repack) for the CPU
    /// `backend`. Each tensor must then be written whole, in one `set_bytes` at offset 0.
    pub fn alloc_weights_extra(
        &self,
        buft: super::cpu::ExtraBuft,
        backend: &Backend,
    ) -> Result<usize> {
        let buf = Buffer::alloc_ctx_tensors_extra(self.ptr.as_ptr(), buft, backend)?;
        let size = buf.size();
        buf.set_usage_weights();
        *self.buffer.borrow_mut() = Some(buf);
        Ok(size)
    }

    /// Allocates all tensors created so far in host memory that `backend` can copy from
    /// asynchronously: its device's host buffer type (pinned, `cudaMallocHost` on CUDA), else
    /// plain CPU memory (a CPU backend, or a device without one). Returns the size and whether the
    /// buffer is of the device's host type (CUDA falls back to plain memory when pinning fails).
    pub fn alloc_host_staging(&self, backend: &Backend) -> Result<(usize, bool)> {
        let buf = Buffer::alloc_host_staging(self.ptr.as_ptr(), backend)?;
        let size = buf.size();
        let pinned = buf.is_device_host(backend);
        *self.buffer.borrow_mut() = Some(buf);
        Ok((size, pinned))
    }

    /// [`Buffer::advise_hugepages`] on the tensor buffer; the default (not applied) without one.
    pub fn advise_hugepages(&self) -> super::HugepageAdvice {
        self.buffer
            .borrow()
            .as_ref()
            .map(Buffer::advise_hugepages)
            .unwrap_or_default()
    }

    /// Sets every byte of the tensor buffer.
    pub fn clear_buffer(&self, value: u8) {
        if let Some(b) = self.buffer.borrow().as_ref() {
            b.clear(value);
        }
    }

    fn t<'a>(&'a self, p: *mut sys::ggml_tensor) -> Tensor<'a> {
        Tensor {
            ptr: NonNull::new(p).expect("ggml returned a null tensor"),
            _ctx: PhantomData,
        }
    }

    /// A new tensor of 1 to 4 dimensions (`ne[0]` is the contiguous one).
    pub fn new_tensor<'a>(&'a self, ty: Type, ne: &[i64]) -> Tensor<'a> {
        assert!((1..=4).contains(&ne.len()), "1 to 4 dimensions");
        // SAFETY: live context; ne has ne.len() entries.
        self.t(unsafe {
            sys::ggml_new_tensor(self.ptr.as_ptr(), ty.raw(), ne.len() as i32, ne.as_ptr())
        })
    }

    /// A tensor that keeps this context alive, for long-lived data (weights, KV cache, state).
    pub fn new_persistent(self: &Rc<Self>, ty: Type, ne: &[i64]) -> OwnedTensor {
        let ptr = self.new_tensor(ty, ne).ptr;
        OwnedTensor {
            _ctx: Rc::clone(self),
            ptr,
        }
    }

    /// True if `p` points into this context's memory pool, where ggml keeps the metadata of its
    /// tensors and graphs.
    fn owns<T>(&self, p: *const T) -> bool {
        // SAFETY: live context; the getters read two fields.
        let (base, size) = unsafe {
            (
                sys::ggml_get_mem_buffer(self.ptr.as_ptr()) as usize,
                sys::ggml_get_mem_size(self.ptr.as_ptr()),
            )
        };
        (p as usize).wrapping_sub(base) < size
    }

    /// An [`OwnedTensor`] for a tensor of this context (an op result, say), so it outlives the
    /// borrow. Errors if the tensor belongs to another context.
    pub fn keep(self: &Rc<Self>, t: Tensor<'_>) -> Result<OwnedTensor> {
        if !self.owns(t.ptr.as_ptr()) {
            return Err(GgmlError::Invalid(
                "keep: the tensor is not in this context".into(),
            ));
        }
        Ok(OwnedTensor {
            _ctx: Rc::clone(self),
            ptr: t.ptr,
        })
    }

    /// Keeps `g` (a graph of this context) with everything it needs alive, so it can be computed
    /// again later. `deps` name the other contexts the graph reads (weights, KV cache, state):
    /// every node, source and view source must be in this context or in one of theirs.
    ///
    /// The only other memory a tensor of a context points to is a backend buffer, and the only
    /// ways to attach one are [`Context::alloc_tensors`] (owned by that context) and a
    /// [`super::Sched`] allocation (intermediates, owned by the scheduler). So holding the
    /// contexts holds every buffer the graph's leaves use.
    pub fn keep_graph(self: &Rc<Self>, g: Graph<'_>, deps: &[&OwnedTensor]) -> Result<OwnedGraph> {
        let mut ctxs = vec![Rc::clone(self)];
        for d in deps {
            if !ctxs.iter().any(|c| Rc::ptr_eq(c, &d._ctx)) {
                ctxs.push(Rc::clone(&d._ctx));
            }
        }
        // Null (no source, no view source) counts as known.
        let known = |p: *const sys::ggml_tensor| p.is_null() || ctxs.iter().any(|c| c.owns(p));
        if !self.owns(g.ptr.as_ptr()) {
            return Err(GgmlError::Invalid(
                "keep_graph: the graph is not in this context".into(),
            ));
        }
        for i in 0..g.n_nodes() {
            let node = g.node(i).ptr.as_ptr();
            // SAFETY: a node of the graph, whose context `self` is live. Its sources and view
            // sources are checked to lie in held contexts before they are read one level deeper;
            // the pointers themselves are only compared.
            let ok = unsafe {
                let t = &*node;
                known(node)
                    && known(t.view_src)
                    && t.src
                        .iter()
                        .all(|&s| known(s) && (s.is_null() || known((*s).view_src)))
            };
            if !ok {
                return Err(GgmlError::Invalid(format!(
                    "keep_graph: node {i} reads a tensor outside the graph's context and its deps"
                )));
            }
        }
        Ok(OwnedGraph {
            ctxs,
            ptr: g.ptr,
            id: g.id,
        })
    }

    /// A new graph that can hold `nodes` nodes (the context must have been sized for it).
    pub fn new_graph<'a>(&'a self, nodes: usize) -> Graph<'a> {
        // SAFETY: live context sized for the graph by `Context::new`.
        let ptr = unsafe { sys::ggml_new_graph_custom(self.ptr.as_ptr(), nodes, false) };
        Graph {
            ptr: NonNull::new(ptr).expect("ggml returned a null graph"),
            id: GRAPH_IDS.fetch_add(1, Ordering::Relaxed),
            _ctx: PhantomData,
        }
    }

    // ---- ops ----------------------------------------------------------------------------

    /// Rows of `a` selected by the I32 tensor `idx`.
    pub fn get_rows<'a>(&'a self, a: Tensor<'a>, idx: Tensor<'a>) -> Tensor<'a> {
        // SAFETY: live context and tensors (lifetimes); ggml aborts on invalid shapes.
        self.t(unsafe { sys::ggml_get_rows(self.ptr.as_ptr(), a.raw(), idx.raw()) })
    }

    /// Writes the rows of `src` (F32 `[n, n_rows]`) into `dst` at the row numbers in the I32
    /// tensor `idx`. Returns a view of `dst`; use it to order later reads after the write.
    pub fn set_rows<'a>(&'a self, dst: Tensor<'a>, src: Tensor<'a>, idx: Tensor<'a>) -> Tensor<'a> {
        // SAFETY: see `get_rows`.
        self.t(unsafe { sys::ggml_set_rows(self.ptr.as_ptr(), dst.raw(), src.raw(), idx.raw()) })
    }

    pub fn add<'a>(&'a self, a: Tensor<'a>, b: Tensor<'a>) -> Tensor<'a> {
        // SAFETY: see `get_rows`.
        self.t(unsafe { sys::ggml_add(self.ptr.as_ptr(), a.raw(), b.raw()) })
    }

    /// `a + b` written into `a`: the result is a view of `a` (no data of its own).
    pub fn add_inplace<'a>(&'a self, a: Tensor<'a>, b: Tensor<'a>) -> Tensor<'a> {
        // SAFETY: see `get_rows`.
        self.t(unsafe { sys::ggml_add_inplace(self.ptr.as_ptr(), a.raw(), b.raw()) })
    }

    pub fn mul<'a>(&'a self, a: Tensor<'a>, b: Tensor<'a>) -> Tensor<'a> {
        // SAFETY: see `get_rows`.
        self.t(unsafe { sys::ggml_mul(self.ptr.as_ptr(), a.raw(), b.raw()) })
    }

    pub fn div<'a>(&'a self, a: Tensor<'a>, b: Tensor<'a>) -> Tensor<'a> {
        // SAFETY: see `get_rows`.
        self.t(unsafe { sys::ggml_div(self.ptr.as_ptr(), a.raw(), b.raw()) })
    }

    pub fn scale<'a>(&'a self, a: Tensor<'a>, s: f32) -> Tensor<'a> {
        // SAFETY: see `get_rows`.
        self.t(unsafe { sys::ggml_scale(self.ptr.as_ptr(), a.raw(), s) })
    }

    pub fn clamp<'a>(&'a self, a: Tensor<'a>, min: f32, max: f32) -> Tensor<'a> {
        // SAFETY: see `get_rows`.
        self.t(unsafe { sys::ggml_clamp(self.ptr.as_ptr(), a.raw(), min, max) })
    }

    pub fn rms_norm<'a>(&'a self, a: Tensor<'a>, eps: f32) -> Tensor<'a> {
        // SAFETY: see `get_rows`.
        self.t(unsafe { sys::ggml_rms_norm(self.ptr.as_ptr(), a.raw(), eps) })
    }

    pub fn soft_max<'a>(&'a self, a: Tensor<'a>) -> Tensor<'a> {
        // SAFETY: see `get_rows`.
        self.t(unsafe { sys::ggml_soft_max(self.ptr.as_ptr(), a.raw()) })
    }

    pub fn sum_rows<'a>(&'a self, a: Tensor<'a>) -> Tensor<'a> {
        // SAFETY: see `get_rows`.
        self.t(unsafe { sys::ggml_sum_rows(self.ptr.as_ptr(), a.raw()) })
    }

    /// Indices of the `k` largest elements per row, I32, in descending order.
    pub fn argsort_top_k<'a>(&'a self, a: Tensor<'a>, k: i32) -> Tensor<'a> {
        // SAFETY: see `get_rows`.
        self.t(unsafe { sys::ggml_argsort_top_k(self.ptr.as_ptr(), a.raw(), k) })
    }

    /// `a` is `[k, n, ...]`, `b` is `[k, m, ...]`; the result is `[n, m, ...]`.
    pub fn mul_mat<'a>(&'a self, a: Tensor<'a>, b: Tensor<'a>) -> Tensor<'a> {
        // SAFETY: see `get_rows`.
        self.t(unsafe { sys::ggml_mul_mat(self.ptr.as_ptr(), a.raw(), b.raw()) })
    }

    /// Orthonormal Hadamard rotation of every `n`-element group of `cur`, `n` the side of the
    /// square matrix `rot` (F32, `[n, n]`, see `hadamard_matrix`). The same ops as libllama's
    /// `llama_mul_mat_hadamard` (llama-impl.h), including the hint that lets a backend use a
    /// fast transform instead of reading `rot`. `n` must divide the element count of `cur`.
    pub fn mul_mat_hadamard<'a>(&'a self, cur: Tensor<'a>, rot: Tensor<'a>) -> Tensor<'a> {
        let c = self.ptr.as_ptr();
        let n = rot.ne()[0];
        let ne = cur.ne();
        let rows = ne.iter().product::<i64>() / n;
        // SAFETY: see `get_rows`; the hint is set on the `mul_mat` node made right here.
        self.t(unsafe {
            let flat = if sys::ggml_is_contiguous(cur.raw()) {
                sys::ggml_reshape_2d(c, cur.raw(), n, rows)
            } else {
                sys::ggml_cont_2d(c, cur.raw(), n, rows)
            };
            let res = sys::ggml_mul_mat(c, rot.raw(), flat);
            sys::ggml_mul_mat_set_hint(res, sys::GGML_HINT_SRC0_IS_HADAMARD as _);
            sys::ggml_reshape_4d(c, res, ne[0], ne[1], ne[2], ne[3])
        })
    }

    /// Per-token expert matmul: `experts` is the stacked `[k, n, n_expert]` weight, `b` is
    /// `[k, n_used or 1, n_tokens]`, `ids` is I32 `[n_used, n_tokens]`.
    pub fn mul_mat_id<'a>(
        &'a self,
        experts: Tensor<'a>,
        b: Tensor<'a>,
        ids: Tensor<'a>,
    ) -> Tensor<'a> {
        // SAFETY: see `get_rows`.
        self.t(unsafe {
            sys::ggml_mul_mat_id(self.ptr.as_ptr(), experts.raw(), b.raw(), ids.raw())
        })
    }

    /// `silu(gate) * up`.
    pub fn swiglu_split<'a>(&'a self, gate: Tensor<'a>, up: Tensor<'a>) -> Tensor<'a> {
        // SAFETY: see `get_rows`.
        self.t(unsafe { sys::ggml_swiglu_split(self.ptr.as_ptr(), gate.raw(), up.raw()) })
    }

    pub fn rope_ext<'a>(&'a self, a: Tensor<'a>, pos: Tensor<'a>, p: &RopeParams) -> Tensor<'a> {
        // SAFETY: see `get_rows`; the frequency-factor tensor is optional (null).
        self.t(unsafe {
            sys::ggml_rope_ext(
                self.ptr.as_ptr(),
                a.raw(),
                pos.raw(),
                std::ptr::null_mut(),
                p.n_dims,
                p.mode,
                p.n_ctx_orig,
                p.freq_base,
                p.freq_scale,
                p.ext_factor,
                p.attn_factor,
                p.beta_fast,
                p.beta_slow,
            )
        })
    }

    /// Flash attention. `q` is `[d, n_tokens, n_head]`, `k` and `v` are `[d, n_kv, n_head_kv]`
    /// (F16 or F32), `mask` is F16 `[n_kv, n_tokens]` (or wider). The result is
    /// `[d, n_head, n_tokens]`. CUDA's mask scan (from 1024 queries) reads whole query tiles of up
    /// to 64 rows, so the mask's buffer must extend to `pad(n_tokens, 64)` zeroed rows.
    pub fn flash_attn_ext<'a>(
        &'a self,
        q: Tensor<'a>,
        k: Tensor<'a>,
        v: Tensor<'a>,
        mask: Tensor<'a>,
        scale: f32,
    ) -> Tensor<'a> {
        // SAFETY: see `get_rows`.
        let t = self.t(unsafe {
            sys::ggml_flash_attn_ext(
                self.ptr.as_ptr(),
                q.raw(),
                k.raw(),
                v.raw(),
                mask.raw(),
                scale,
                0.0,
                0.0,
            )
        });
        // libllama accumulates attention in F32.
        // SAFETY: t is a flash-attention node.
        unsafe { sys::ggml_prec_set_acc(t.raw(), sys::GGML_PREC_F32 as _) };
        t
    }

    pub fn cast<'a>(&'a self, a: Tensor<'a>, ty: Type) -> Tensor<'a> {
        // SAFETY: see `get_rows`.
        self.t(unsafe { sys::ggml_cast(self.ptr.as_ptr(), a.raw(), ty.raw()) })
    }

    /// Copies `a` into `dst` (converting the type); the result is a view of `dst`.
    pub fn cpy<'a>(&'a self, a: Tensor<'a>, dst: Tensor<'a>) -> Tensor<'a> {
        // SAFETY: see `get_rows`.
        self.t(unsafe { sys::ggml_cpy(self.ptr.as_ptr(), a.raw(), dst.raw()) })
    }

    /// `a` and `b` joined along dimension `dim` (other dimensions equal); the result is contiguous.
    pub fn concat<'a>(&'a self, a: Tensor<'a>, b: Tensor<'a>, dim: i32) -> Tensor<'a> {
        // SAFETY: see `get_rows`.
        self.t(unsafe { sys::ggml_concat(self.ptr.as_ptr(), a.raw(), b.raw(), dim) })
    }

    pub fn cont<'a>(&'a self, a: Tensor<'a>) -> Tensor<'a> {
        // SAFETY: see `get_rows`.
        self.t(unsafe { sys::ggml_cont(self.ptr.as_ptr(), a.raw()) })
    }

    /// Same data, new shape (`a` must be contiguous and the element counts equal).
    pub fn reshape<'a>(&'a self, a: Tensor<'a>, ne: &[i64]) -> Tensor<'a> {
        let c = self.ptr.as_ptr();
        // SAFETY: see `get_rows`.
        self.t(unsafe {
            match *ne {
                [a0] => sys::ggml_reshape_1d(c, a.raw(), a0),
                [a0, a1] => sys::ggml_reshape_2d(c, a.raw(), a0, a1),
                [a0, a1, a2] => sys::ggml_reshape_3d(c, a.raw(), a0, a1, a2),
                [a0, a1, a2, a3] => sys::ggml_reshape_4d(c, a.raw(), a0, a1, a2, a3),
                _ => panic!("reshape takes 1 to 4 dimensions"),
            }
        })
    }

    /// A view with shape `ne`, byte strides `nb` for dimensions 1.. (one fewer than `ne`) and a
    /// byte `offset` into `a`.
    pub fn view<'a>(
        &'a self,
        a: Tensor<'a>,
        ne: &[i64],
        nb: &[usize],
        offset: usize,
    ) -> Tensor<'a> {
        assert_eq!(
            ne.len(),
            nb.len() + 1,
            "one stride per dimension after the first"
        );
        let c = self.ptr.as_ptr();
        // SAFETY: see `get_rows`.
        self.t(unsafe {
            match (ne, nb) {
                (&[a0], &[]) => sys::ggml_view_1d(c, a.raw(), a0, offset),
                (&[a0, a1], &[b1]) => sys::ggml_view_2d(c, a.raw(), a0, a1, b1, offset),
                (&[a0, a1, a2], &[b1, b2]) => {
                    sys::ggml_view_3d(c, a.raw(), a0, a1, a2, b1, b2, offset)
                }
                (&[a0, a1, a2, a3], &[b1, b2, b3]) => {
                    sys::ggml_view_4d(c, a.raw(), a0, a1, a2, a3, b1, b2, b3, offset)
                }
                _ => panic!("view takes 1 to 4 dimensions"),
            }
        })
    }

    /// Reorders dimensions: dimension `i` of `a` becomes dimension `axes[i]`.
    pub fn permute<'a>(&'a self, a: Tensor<'a>, axes: [i32; 4]) -> Tensor<'a> {
        // SAFETY: see `get_rows`.
        self.t(unsafe {
            sys::ggml_permute(
                self.ptr.as_ptr(),
                a.raw(),
                axes[0],
                axes[1],
                axes[2],
                axes[3],
            )
        })
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // The buffer first: its tensors' metadata is still alive while it is freed.
        self.buffer.get_mut().take();
        // SAFETY: owned; tensor handles borrow the context, so none is live.
        unsafe { sys::ggml_free(self.ptr.as_ptr()) }
    }
}

/// Rotary embedding parameters (`ggml_rope_ext`).
#[derive(Debug, Clone, Copy)]
pub struct RopeParams {
    pub n_dims: i32,
    /// ggml rope mode: 0 normal, 2 NeoX (Qwen3).
    pub mode: i32,
    pub n_ctx_orig: i32,
    pub freq_base: f32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

/// A tensor of a [`Context`]. Copyable; it is only valid while the context lives.
#[derive(Clone, Copy)]
pub struct Tensor<'a> {
    ptr: NonNull<sys::ggml_tensor>,
    _ctx: PhantomData<&'a ()>,
}

impl std::fmt::Debug for Tensor<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor({:?} {:?})", self.ty(), self.ne())
    }
}

impl<'a> Tensor<'a> {
    pub(super) fn raw(self) -> *mut sys::ggml_tensor {
        self.ptr.as_ptr()
    }

    /// Elements per dimension.
    pub fn ne(self) -> [i64; 4] {
        // SAFETY: the handle borrows its live context.
        unsafe { (*self.ptr.as_ptr()).ne }
    }

    /// Bytes per step in each dimension.
    pub fn nb(self) -> [usize; 4] {
        // SAFETY: as above.
        unsafe { (*self.ptr.as_ptr()).nb }
    }

    #[allow(clippy::unnecessary_cast)] // C enum: i32 on MSVC, u32 elsewhere
    pub fn ty(self) -> Type {
        // SAFETY: as above.
        Type(unsafe { (*self.ptr.as_ptr()).type_ } as u32)
    }

    pub fn nbytes(self) -> usize {
        // SAFETY: as above.
        unsafe { sys::ggml_nbytes(self.ptr.as_ptr()) }
    }

    pub fn set_name(self, name: &str) -> Self {
        let c = CString::new(name.replace('\0', "")).expect("NUL removed");
        // SAFETY: as above; ggml copies the name.
        unsafe { sys::ggml_set_name(self.ptr.as_ptr(), c.as_ptr()) };
        self
    }

    pub fn name(self) -> String {
        // SAFETY: as above; the name is stored inline in the tensor.
        cstr(unsafe { (*self.ptr.as_ptr()).name.as_ptr() })
    }

    /// ggml's op name, for example `SOFT_MAX` or `MUL_MAT_ID`; `NONE` for a leaf.
    pub fn op_name(self) -> String {
        // SAFETY: as above; ggml returns a static NUL-terminated name for every op.
        cstr(unsafe { sys::ggml_op_name((*self.ptr.as_ptr()).op) })
    }

    /// Same tensor (identity, not contents).
    pub fn is(self, other: Tensor<'_>) -> bool {
        self.ptr == other.ptr
    }

    /// Source `i` of the op, if set.
    pub fn src(self, i: usize) -> Option<Tensor<'a>> {
        // SAFETY: as above. A source belongs to this context or to one the graph's builder
        // borrows for at least `'a` (weights, KV cache, state).
        let p = unsafe { (*self.ptr.as_ptr()).src }.get(i).copied()?;
        NonNull::new(p).map(|ptr| Tensor {
            ptr,
            _ctx: PhantomData,
        })
    }

    /// Marks the tensor as a graph output, so the scheduler keeps its data after compute.
    pub fn set_output(self) -> Self {
        // SAFETY: as above.
        unsafe { sys::ggml_set_output(self.ptr.as_ptr()) };
        self
    }

    /// Marks the tensor as a graph input (data written before compute).
    pub fn set_input(self) -> Self {
        // SAFETY: as above.
        unsafe { sys::ggml_set_input(self.ptr.as_ptr()) };
        self
    }

    fn check_range(self, offset: usize, len: usize) -> Result<()> {
        // SAFETY: as above.
        let has_buffer = unsafe { !(*self.ptr.as_ptr()).buffer.is_null() };
        if !has_buffer {
            return Err(GgmlError::Invalid(format!(
                "tensor `{}` has no backend buffer yet",
                self.name()
            )));
        }
        if offset + len > self.nbytes() {
            return Err(GgmlError::Invalid(format!(
                "range {offset}+{len} exceeds tensor `{}` ({} bytes)",
                self.name(),
                self.nbytes()
            )));
        }
        Ok(())
    }

    /// Writes raw bytes at `offset` (host to backend memory).
    pub fn set_bytes(self, data: &[u8], offset: usize) -> Result<()> {
        self.check_range(offset, data.len())?;
        // SAFETY: the range is inside the tensor and the tensor has a live buffer.
        unsafe {
            sys::ggml_backend_tensor_set(
                self.ptr.as_ptr(),
                data.as_ptr().cast(),
                offset,
                data.len(),
            )
        };
        Ok(())
    }

    /// Address of the tensor's data when its buffer is host memory (a CPU buffer), else `None`.
    /// For a background reader that fills slots with plain copies (ADR 0038): the caller keeps
    /// the tensor and its buffer alive, in range and unshared while the copy runs.
    pub fn host_ptr(self) -> Option<*mut u8> {
        // SAFETY: the handle borrows its live context; the buffer is checked for null first.
        unsafe {
            let t = self.ptr.as_ptr();
            let host = !(*t).buffer.is_null() && sys::ggml_backend_buffer_is_host((*t).buffer);
            (host && !(*t).data.is_null()).then(|| (*t).data.cast())
        }
    }

    /// The address of `len` bytes at `offset` of a tensor in host memory, checked.
    fn host_range(self, offset: usize, len: usize) -> Result<*const u8> {
        self.check_range(offset, len)?;
        let base = self.host_ptr().ok_or_else(|| {
            GgmlError::Invalid(format!("tensor `{}` is not in host memory", self.name()))
        })?;
        // SAFETY: `offset + len` is inside the tensor's data (checked above).
        Ok(unsafe { base.add(offset) }.cast_const())
    }

    /// Copies `len` bytes at `offset` of this host tensor to `dst` at `dst_offset` (any backend
    /// buffer), blocking until done: one `ggml_backend_tensor_set` straight from this tensor's
    /// memory, no intermediate copy.
    pub fn copy_to(
        self,
        offset: usize,
        dst: Tensor<'_>,
        dst_offset: usize,
        len: usize,
    ) -> Result<()> {
        let src = self.host_range(offset, len)?;
        dst.check_range(dst_offset, len)?;
        // SAFETY: both ranges are inside live tensors (checked); `src` is host memory.
        unsafe { sys::ggml_backend_tensor_set(dst.raw(), src.cast(), dst_offset, len) };
        Ok(())
    }

    /// Enqueues the copy of `len` bytes at `offset` of this host tensor to `dst` at
    /// `dst_offset` on `backend`'s stream (`ggml_backend_tensor_set_async`); without an async
    /// path (the CPU backend) it copies before returning. The copy reads this tensor's memory
    /// until `backend` is synchronized, so the caller must not write the range nor free the
    /// tensor before [`Backend::synchronize`]; doing so changes `dst`'s contents, not memory
    /// safety (pinned and device buffers are freed with an implicit device synchronization,
    /// and a pageable source is staged before the call returns). `dst` must be in a buffer
    /// `backend` can write, else an error (ggml would abort).
    pub fn copy_to_async(
        self,
        offset: usize,
        backend: &Backend,
        dst: Tensor<'_>,
        dst_offset: usize,
        len: usize,
    ) -> Result<()> {
        let src = self.host_range(offset, len)?;
        dst.check_range(dst_offset, len)?;
        if !backend.can_write(dst) {
            return Err(GgmlError::Invalid(format!(
                "backend `{}` cannot write tensor `{}`",
                backend.name(),
                dst.name()
            )));
        }
        // SAFETY: both ranges are inside live tensors (checked) and `backend` supports `dst`'s
        // buffer type, which ggml asserts. The source contract is the caller's (see above).
        unsafe {
            sys::ggml_backend_tensor_set_async(
                backend.raw(),
                dst.raw(),
                src.cast(),
                dst_offset,
                len,
            )
        };
        Ok(())
    }

    /// `ggml_backend_tensor_copy_async`: copies this tensor (in a buffer `src_backend` computes
    /// on) to `dst` (a buffer of `dst_backend`). With both backends on CUDA it is a peer copy on
    /// the source stream and the destination stream waits for it with an event (no host wait).
    /// Without an async path (a CPU backend) ggml synchronizes both backends and copies before
    /// returning. Both tensors need a buffer and the same type, shape and strides.
    pub fn copy_tensor_async(
        self,
        src_backend: &Backend,
        dst_backend: &Backend,
        dst: Tensor<'_>,
    ) -> Result<()> {
        self.check_range(0, 0)?;
        dst.check_range(0, 0)?;
        if self.ty() != dst.ty() || self.ne() != dst.ne() || self.nb() != dst.nb() {
            return Err(GgmlError::Invalid(format!(
                "copy of `{}` to `{}`: different layouts ({:?} {:?} and {:?} {:?})",
                self.name(),
                dst.name(),
                self.ty(),
                self.ne(),
                dst.ty(),
                dst.ne(),
            )));
        }
        // SAFETY: both tensors are live with buffers and the same layout (checked; ggml asserts
        // it); the backends are live. ggml either copies asynchronously between streams ordered
        // by an event or synchronizes both backends first.
        unsafe {
            sys::ggml_backend_tensor_copy_async(
                src_backend.raw(),
                dst_backend.raw(),
                self.raw(),
                dst.raw(),
            )
        };
        Ok(())
    }

    /// Reads raw bytes at `offset` (backend memory to host).
    pub fn get_bytes(self, offset: usize, len: usize) -> Result<Vec<u8>> {
        self.check_range(offset, len)?;
        let mut out = vec![0u8; len];
        // SAFETY: the range is inside the tensor; `out` has `len` bytes.
        unsafe {
            sys::ggml_backend_tensor_get(self.ptr.as_ptr(), out.as_mut_ptr().cast(), offset, len)
        };
        Ok(out)
    }

    /// Appends `len` bytes at `offset` (backend memory to host) to `out`, blocking: one copy
    /// straight into `out`'s spare capacity, no zero-filled temporary.
    pub fn get_bytes_into(self, offset: usize, len: usize, out: &mut Vec<u8>) -> Result<()> {
        self.check_range(offset, len)?;
        out.reserve(len);
        let at = out.len();
        // SAFETY: the range is inside the tensor (checked) and `reserve` left at least `len`
        // bytes of capacity after `at`. `ggml_backend_tensor_get` writes all `len` bytes before
        // it returns (or aborts the process), so they are initialized when the length grows.
        unsafe {
            sys::ggml_backend_tensor_get(
                self.ptr.as_ptr(),
                out.as_mut_ptr().add(at).cast(),
                offset,
                len,
            );
            out.set_len(at + len);
        }
        Ok(())
    }

    /// Enqueues the copy of `len` bytes at `offset` of this tensor into the host tensor `dst`
    /// at `dst_offset` on `backend`'s stream (`ggml_backend_tensor_get_async`); without an async
    /// path (the CPU backend) it copies before returning. The copy writes `dst` until `backend`
    /// is synchronized, so the caller must not read or write that range of `dst` before
    /// [`Backend::synchronize`]; doing so sees or changes the bytes being copied, not memory
    /// safety (pinned buffers are freed with an implicit device synchronization). This tensor
    /// must be in a buffer `backend` can read ([`Backend::can_read`]), else an error (ggml
    /// would abort).
    pub fn read_to_async(
        self,
        offset: usize,
        backend: &Backend,
        dst: Tensor<'_>,
        dst_offset: usize,
        len: usize,
    ) -> Result<()> {
        self.check_range(offset, len)?;
        let out = dst.host_range(dst_offset, len)?;
        if !backend.can_read(self) {
            return Err(GgmlError::Invalid(format!(
                "backend `{}` cannot read tensor `{}`",
                backend.name(),
                self.name()
            )));
        }
        // SAFETY: both ranges are inside live tensors (checked), `dst` is host memory, and
        // `backend` reads this tensor's buffer type, which ggml asserts. The destination
        // contract is the caller's (see above).
        unsafe {
            sys::ggml_backend_tensor_get_async(
                backend.raw(),
                self.ptr.as_ptr(),
                out.cast_mut().cast(),
                offset,
                len,
            )
        };
        Ok(())
    }

    pub fn set_f32(self, data: &[f32]) -> Result<()> {
        self.set_bytes(pod_bytes(data), 0)
    }

    pub fn set_i32(self, data: &[i32]) -> Result<()> {
        self.set_bytes(pod_bytes(data), 0)
    }

    /// Writes F16 data given as raw IEEE half bits.
    pub fn set_f16_bits(self, data: &[u16]) -> Result<()> {
        self.set_bytes(pod_bytes(data), 0)
    }

    /// Enqueues the write of `data` at `offset` on `backend`'s stream
    /// (`ggml_backend_tensor_set_async`) instead of blocking on a copy of its own as
    /// [`Tensor::set_bytes`] does; without an async path (the CPU backend) it copies before
    /// returning. CUDA stages a pageable source (memory not registered with CUDA, as every Rust
    /// allocation here) before the call returns, so `data` may change afterwards; the write is ordered before the work issued later on that stream. `self` must
    /// be in a buffer `backend` can write, else an error (ggml would abort).
    pub fn set_bytes_async(self, backend: &Backend, data: &[u8], offset: usize) -> Result<()> {
        self.check_range(offset, data.len())?;
        if !backend.can_write(self) {
            return Err(GgmlError::Invalid(format!(
                "backend `{}` cannot write tensor `{}`",
                backend.name(),
                self.name()
            )));
        }
        // SAFETY: the range is inside the live tensor (checked) and `backend` supports its
        // buffer type, which ggml asserts. `data` is read before the call returns or staged.
        unsafe {
            sys::ggml_backend_tensor_set_async(
                backend.raw(),
                self.ptr.as_ptr(),
                data.as_ptr().cast(),
                offset,
                data.len(),
            )
        };
        Ok(())
    }

    /// [`Tensor::set_i32`] on `backend`'s stream ([`Tensor::set_bytes_async`]).
    pub fn set_i32_async(self, backend: &Backend, data: &[i32]) -> Result<()> {
        self.set_bytes_async(backend, pod_bytes(data), 0)
    }

    /// [`Tensor::set_f16_bits`] on `backend`'s stream ([`Tensor::set_bytes_async`]).
    pub fn set_f16_bits_async(self, backend: &Backend, data: &[u16]) -> Result<()> {
        self.set_bytes_async(backend, pod_bytes(data), 0)
    }

    /// The whole tensor as `f32`, read straight into the result (decode reads 151K logits per
    /// token; a byte copy and conversion cost as much as the transfer).
    pub fn get_f32(self) -> Result<Vec<f32>> {
        let mut out = vec![0f32; self.nbytes() / 4];
        self.read_into(&mut out)?;
        Ok(out)
    }

    /// [`Tensor::get_f32`] into `out`, reusing its allocation.
    pub fn get_f32_into(self, out: &mut Vec<f32>) -> Result<()> {
        out.resize(self.nbytes() / 4, 0.0);
        self.read_into(out)
    }

    pub fn get_i32(self) -> Result<Vec<i32>> {
        let mut out = vec![0i32; self.nbytes() / 4];
        self.read_into(&mut out)?;
        Ok(out)
    }

    /// Reads the first `size_of_val(out)` bytes of the tensor into `out`.
    fn read_into<T: Pod>(self, out: &mut [T]) -> Result<()> {
        let len = std::mem::size_of_val(out);
        self.check_range(0, len)?;
        // SAFETY: the range is inside the tensor; `out` has `len` writable bytes, and every bit
        // pattern is a valid `T` (`Pod`).
        unsafe { sys::ggml_backend_tensor_get(self.ptr.as_ptr(), out.as_mut_ptr().cast(), 0, len) };
        Ok(())
    }
}

/// Plain numbers without padding, for which any bytes are a valid value.
trait Pod: Copy {}
impl Pod for f32 {}
impl Pod for i32 {}
impl Pod for u16 {}

/// The bytes of a slice of plain numbers, without a copy.
fn pod_bytes<T: Pod>(data: &[T]) -> &[u8] {
    // SAFETY: `T` has no padding, so all `size_of_val(data)` bytes are initialized; `u8` has
    // alignment 1 and the borrow keeps `data` alive.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), std::mem::size_of_val(data)) }
}

/// A tensor that owns a share of its context. [`OwnedTensor::get`] lends a [`Tensor`] bound to
/// the borrow, so builders can store weights in plain structs.
#[derive(Clone)]
pub struct OwnedTensor {
    _ctx: Rc<Context>,
    ptr: NonNull<sys::ggml_tensor>,
}

impl OwnedTensor {
    pub fn get(&self) -> Tensor<'_> {
        Tensor {
            ptr: self.ptr,
            _ctx: PhantomData,
        }
    }
}

/// A graph that holds the contexts of everything it reads ([`Context::keep_graph`]).
pub struct OwnedGraph {
    /// The graph's own context first, then the contexts of its deps. Never read; held so the
    /// tensors and buffers the graph points to stay alive.
    #[allow(dead_code)]
    ctxs: Vec<Rc<Context>>,
    ptr: NonNull<sys::ggml_cgraph>,
    id: u64,
}

impl OwnedGraph {
    /// The graph, with the same id, so a [`super::Sched`] that allocated it still accepts it.
    pub fn get(&self) -> Graph<'_> {
        Graph {
            ptr: self.ptr,
            id: self.id,
            _ctx: PhantomData,
        }
    }
}

/// A compute graph over the tensors of one [`Context`].
pub struct Graph<'a> {
    pub(super) ptr: NonNull<sys::ggml_cgraph>,
    pub(super) id: u64,
    _ctx: PhantomData<&'a Context>,
}

impl<'a> Graph<'a> {
    /// Nodes `[i0, i1)` of `self` as a graph object in `ctx` (`ggml_graph_view_new`, patch 0005,
    /// ADR 0075): it shares this graph's node array, use counts and visited set, so a backend
    /// fuses in it as in the whole graph. Allocation stays with the graph `self` was allocated
    /// as: computing the range needs that allocation alive. `ctx` must have room for one graph.
    /// The range has its own id (an allocator or scheduler does not know it).
    pub fn range(&self, ctx: &'a Context, i0: usize, i1: usize) -> Graph<'a> {
        assert!(
            i0 < i1 && i1 <= self.n_nodes(),
            "node range {i0}..{i1} out of {}",
            self.n_nodes()
        );
        // SAFETY: live context and graph; 0 <= i0 < i1 <= n_nodes (checked above), which fit
        // in i32 because n_nodes does. The view points into this graph's arrays, which live as
        // long as its context ('a).
        let ptr = unsafe {
            sys::ggml_graph_view_new(
                ctx.ptr.as_ptr(),
                self.ptr.as_ptr(),
                i0 as i32,
                i1 as i32,
                false,
            )
        };
        Graph {
            ptr: NonNull::new(ptr).expect("ggml returned a null graph view"),
            id: GRAPH_IDS.fetch_add(1, Ordering::Relaxed),
            _ctx: PhantomData,
        }
    }

    /// With `true`, ggml-cuda runs this graph directly on every call and never captures or
    /// replays a CUDA graph for it (`ggml_graph_set_no_capture`, patch 0006, ADR 0081). Other
    /// backends ignore it. A view from [`Graph::range`] starts unflagged.
    pub fn set_no_capture(&mut self, no_capture: bool) {
        // SAFETY: live graph; the call writes one bool in the graph object.
        unsafe { sys::ggml_graph_set_no_capture(self.ptr.as_ptr(), no_capture) }
    }

    /// Adds `t` and everything it depends on, in dependency order.
    pub fn expand(&mut self, t: Tensor<'a>) {
        // SAFETY: the graph and the tensor belong to live contexts.
        unsafe { sys::ggml_build_forward_expand(self.ptr.as_ptr(), t.raw()) }
    }

    /// Identity of the graph: unique per graph built in this process, kept by
    /// [`Context::keep_graph`] and [`OwnedGraph::get`].
    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn n_nodes(&self) -> usize {
        // SAFETY: live graph.
        unsafe { sys::ggml_graph_n_nodes(self.ptr.as_ptr()) as usize }
    }

    /// Node `i` in execution order, for checks of the graph's op order.
    pub fn node(&self, i: usize) -> Tensor<'a> {
        assert!(i < self.n_nodes(), "node {i} out of range");
        // SAFETY: live graph and an index below its node count.
        let p = unsafe { sys::ggml_graph_node(self.ptr.as_ptr(), i as i32) };
        Tensor {
            ptr: NonNull::new(p).expect("graph nodes are non-null"),
            _ctx: PhantomData,
        }
    }
}

/// Consecutive node ranges of one graph as views of it (`ggml_graph_view_new`, ADR 0075), kept
/// apart from the source graph so a caller can cache them (ADR 0069). A range shares the source
/// graph's nodes and use counts, so a backend fuses in it as in the whole graph. The ranges point
/// into the source graph and use its allocation, so they are handed out only against that
/// graph: [`RangeGraphs::get`] takes it and the result borrows it.
pub struct RangeGraphs {
    // Owns the memory of the range graphs; never read.
    _ctx: Context,
    graphs: Vec<(NonNull<sys::ggml_cgraph>, u64)>,
    source: u64,
}

impl RangeGraphs {
    /// Ranges `[0, cuts[0])`, `[cuts[0], cuts[1])` and so on. `cuts` is strictly increasing and
    /// ends at `src.n_nodes()`. With `uid` each range gets a fresh ggml graph uid (patch 0005),
    /// which tells a backend that its nodes and their sources do not change while it is computed
    /// again: ggml-cuda then skips its per-call node comparison before a CUDA graph replay, as for
    /// the scheduler's splits (ADR 0069 History 2026-11-19). The caller keeps that promise by
    /// cutting new ranges whenever `src` is built or allocated again.
    pub fn new(src: &Graph<'_>, cuts: &[usize], uid: bool) -> Result<RangeGraphs> {
        let n = src.n_nodes();
        let mut prev = 0;
        for &c in cuts {
            if c <= prev {
                return Err(GgmlError::Invalid(format!(
                    "range cuts {cuts:?} are not strictly increasing from 0"
                )));
            }
            prev = c;
        }
        if prev != n {
            return Err(GgmlError::Invalid(format!(
                "range cuts {cuts:?} do not end at the graph's {n} nodes"
            )));
        }
        // A view is one small graph object: the room of a one-node graph covers it.
        let ctx = Context::with_graphs(0, &vec![1; cuts.len()])?;
        let mut graphs = Vec::with_capacity(cuts.len());
        let mut start = 0;
        for &end in cuts {
            // SAFETY: live context with room for one more graph object, live source graph and
            // 0 <= start < end <= n_nodes (checked above). The view points into the source
            // graph's arrays; it is only handed out while that graph is borrowed (`get`).
            let ptr = unsafe {
                sys::ggml_graph_view_new(
                    ctx.ptr.as_ptr(),
                    src.ptr.as_ptr(),
                    start as i32,
                    end as i32,
                    uid,
                )
            };
            let ptr = NonNull::new(ptr).expect("ggml returned a null graph view");
            graphs.push((ptr, GRAPH_IDS.fetch_add(1, Ordering::Relaxed)));
            start = end;
        }
        Ok(RangeGraphs {
            _ctx: ctx,
            graphs,
            source: src.id,
        })
    }

    /// Sets or clears [`Graph::set_no_capture`] on every range (patch 0006, ADR 0081).
    pub fn set_no_capture(&mut self, no_capture: bool) {
        for &(ptr, _) in &self.graphs {
            // SAFETY: the graph objects live in `self._ctx`, which `self` owns.
            unsafe { sys::ggml_graph_set_no_capture(ptr.as_ptr(), no_capture) }
        }
    }

    pub fn len(&self) -> usize {
        self.graphs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.graphs.is_empty()
    }

    /// Range `i`, for the graph these ranges were cut from. The result borrows both, so the
    /// tensors it computes stay alive for as long as it is used.
    pub fn get<'s>(&'s self, src: &'s Graph<'_>, i: usize) -> Result<Graph<'s>> {
        if src.id != self.source {
            return Err(GgmlError::Invalid(
                "range graphs were cut from another graph".into(),
            ));
        }
        let &(ptr, id) = self.graphs.get(i).ok_or_else(|| {
            GgmlError::Invalid(format!("range {i} of {} range graphs", self.graphs.len()))
        })?;
        Ok(Graph {
            ptr,
            id,
            _ctx: PhantomData,
        })
    }
}
