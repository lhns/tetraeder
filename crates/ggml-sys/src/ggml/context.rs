//! Context, tensor handles, graphs and the ops a Qwen3 MoE forward needs.

use super::{Backend, Buffer, GgmlError, Result, Type, cstr};
use crate::raw as sys;
use std::cell::RefCell;
use std::ffi::CString;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

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
        crate::llama::init();
        // SAFETY: pure size arithmetic.
        let mut mem = max_tensors * unsafe { sys::ggml_tensor_overhead() } + 1024;
        if graph_nodes > 0 {
            // SAFETY: as above.
            mem += unsafe { sys::ggml_graph_overhead_custom(graph_nodes, false) };
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

    /// A new graph that can hold `nodes` nodes (the context must have been sized for it).
    pub fn new_graph<'a>(&'a self, nodes: usize) -> Graph<'a> {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        // SAFETY: live context sized for the graph by `Context::new`.
        let ptr = unsafe { sys::ggml_new_graph_custom(self.ptr.as_ptr(), nodes, false) };
        Graph {
            ptr: NonNull::new(ptr).expect("ggml returned a null graph"),
            id: NEXT.fetch_add(1, Ordering::Relaxed),
            _ctx: PhantomData,
        }
    }

    // ---- ops ----------------------------------------------------------------------------

    /// Rows of `a` selected by the I32 tensor `idx`.
    pub fn get_rows<'a>(&'a self, a: Tensor<'a>, idx: Tensor<'a>) -> Tensor<'a> {
        // SAFETY: live context and tensors (lifetimes); ggml aborts on invalid shapes.
        self.t(unsafe { sys::ggml_get_rows(self.ptr.as_ptr(), a.raw(), idx.raw()) })
    }

    pub fn add<'a>(&'a self, a: Tensor<'a>, b: Tensor<'a>) -> Tensor<'a> {
        // SAFETY: see `get_rows`.
        self.t(unsafe { sys::ggml_add(self.ptr.as_ptr(), a.raw(), b.raw()) })
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
    /// (F16 or F32), `mask` is F16 `[n_kv, pad(n_tokens, 32)]`. The result is
    /// `[d, n_head, n_tokens]`.
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

    pub fn nelements(self) -> usize {
        // SAFETY: as above.
        unsafe { sys::ggml_nelements(self.ptr.as_ptr()) as usize }
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

    pub fn set_f32(self, data: &[f32]) -> Result<()> {
        self.set_bytes(
            &data
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<_>>(),
            0,
        )
    }

    pub fn set_i32(self, data: &[i32]) -> Result<()> {
        self.set_bytes(
            &data
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<_>>(),
            0,
        )
    }

    /// Writes F16 data given as raw IEEE half bits.
    pub fn set_f16_bits(self, data: &[u16]) -> Result<()> {
        self.set_bytes(
            &data
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<_>>(),
            0,
        )
    }

    pub fn get_f32(self) -> Result<Vec<f32>> {
        let b = self.get_bytes(0, self.nbytes())?;
        Ok(b.chunks_exact(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    pub fn get_i32(self) -> Result<Vec<i32>> {
        let b = self.get_bytes(0, self.nbytes())?;
        Ok(b.chunks_exact(4)
            .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
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

/// A compute graph over the tensors of one [`Context`].
pub struct Graph<'a> {
    pub(super) ptr: NonNull<sys::ggml_cgraph>,
    pub(super) id: u64,
    _ctx: PhantomData<&'a Context>,
}

impl<'a> Graph<'a> {
    /// Adds `t` and everything it depends on, in dependency order.
    pub fn expand(&mut self, t: Tensor<'a>) {
        // SAFETY: the graph and the tensor belong to live contexts.
        unsafe { sys::ggml_build_forward_expand(self.ptr.as_ptr(), t.raw()) }
    }

    pub fn n_nodes(&self) -> usize {
        // SAFETY: live graph.
        unsafe { sys::ggml_graph_n_nodes(self.ptr.as_ptr()) as usize }
    }
}
