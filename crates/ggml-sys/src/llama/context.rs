//! Thread pool, context parameters and the inference context.

use super::batch::Batch;
use super::model::Model;
use super::{LlamaError, Result, init};
use crate::raw as sys;
use std::sync::Arc;

/// A CPU threadpool that contexts attach to (llama-bench does the same).
pub struct ThreadPool {
    ptr: sys::ggml_threadpool_t,
}

// SAFETY: the handle is opaque, this wrapper exposes no operations on it, and ggml's pool is
// internally synchronized; it is only passed to llama_attach_threadpool and freed on drop.
unsafe impl Send for ThreadPool {}
// SAFETY: see `Send`.
unsafe impl Sync for ThreadPool {}

impl ThreadPool {
    pub fn new(n_threads: i32) -> Result<Self> {
        init();
        if n_threads < 1 {
            return Err(LlamaError::Invalid("n_threads must be at least 1".into()));
        }
        // SAFETY: plain value return, then `params` is a valid in-out struct.
        let ptr = unsafe {
            let mut params = sys::ggml_threadpool_params_default(n_threads);
            sys::ggml_threadpool_new(&mut params)
        };
        if ptr.is_null() {
            return Err(LlamaError::Invalid("threadpool creation failed".into()));
        }
        Ok(Self { ptr })
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        // SAFETY: owned; contexts keep an Arc so none uses it any more.
        unsafe { sys::ggml_threadpool_free(self.ptr) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlashAttn {
    Auto,
    On,
    Off,
}

/// Context parameters. `Default` is libllama's defaults; K/V cache types stay default.
#[derive(Debug, Clone)]
pub struct ContextParams {
    /// 0 = from the model (can be huge: size the KV cache deliberately).
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub flash_attn: FlashAttn,
    pub offload_kqv: bool,
    pub swa_full: bool,
    pub no_perf: bool,
}

impl Default for ContextParams {
    fn default() -> Self {
        init();
        // SAFETY: plain value return.
        let d = unsafe { sys::llama_context_default_params() };
        let fa = d.flash_attn_type as i64;
        Self {
            n_ctx: d.n_ctx,
            n_batch: d.n_batch,
            n_ubatch: d.n_ubatch,
            n_threads: d.n_threads,
            n_threads_batch: d.n_threads_batch,
            flash_attn: if fa == sys::LLAMA_FLASH_ATTN_TYPE_ENABLED as i64 {
                FlashAttn::On
            } else if fa == sys::LLAMA_FLASH_ATTN_TYPE_DISABLED as i64 {
                FlashAttn::Off
            } else {
                FlashAttn::Auto
            },
            offload_kqv: d.offload_kqv,
            swa_full: d.swa_full,
            no_perf: d.no_perf,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PerfData {
    pub load_ms: f64,
    pub p_eval_ms: f64,
    pub eval_ms: f64,
    pub n_p_eval: i32,
    pub n_eval: i32,
    pub n_reused: i32,
}

/// An inference context (KV cache and compute buffers) over a shared model.
pub struct Context {
    pub(super) ptr: *mut sys::llama_context,
    model: Arc<Model>,
    pool: Option<Arc<ThreadPool>>,
    /// Which tokens of the last decoded batch have logits. libllama aborts (debug builds) or
    /// logs (release) on an invalid output index, so the wrapper checks indices itself.
    outputs: Vec<bool>,
}

// SAFETY: a llama_context has no thread affinity; it may move between threads but is not
// thread-safe, hence no `Sync` (all mutation needs `&mut self`).
unsafe impl Send for Context {}

impl Context {
    pub fn new(model: &Arc<Model>, params: &ContextParams) -> Result<Self> {
        // SAFETY: plain value return.
        let mut p = unsafe { sys::llama_context_default_params() };
        p.n_ctx = params.n_ctx;
        p.n_batch = params.n_batch;
        p.n_ubatch = params.n_ubatch;
        p.n_threads = params.n_threads;
        p.n_threads_batch = params.n_threads_batch;
        p.flash_attn_type = (match params.flash_attn {
            FlashAttn::Auto => sys::LLAMA_FLASH_ATTN_TYPE_AUTO,
            FlashAttn::On => sys::LLAMA_FLASH_ATTN_TYPE_ENABLED,
            FlashAttn::Off => sys::LLAMA_FLASH_ATTN_TYPE_DISABLED,
        }) as _;
        p.offload_kqv = params.offload_kqv;
        p.swa_full = params.swa_full;
        p.no_perf = params.no_perf;
        // SAFETY: the model is live; the context keeps the Arc.
        let ptr = unsafe { sys::llama_init_from_model(model.ptr, p) };
        if ptr.is_null() {
            return Err(LlamaError::ContextInit);
        }
        Ok(Self {
            ptr,
            model: Arc::clone(model),
            pool: None,
            outputs: Vec::new(),
        })
    }

    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }

    /// Context size actually in use (may differ from the request).
    pub fn n_ctx(&self) -> u32 {
        // SAFETY: live context.
        unsafe { sys::llama_n_ctx(self.ptr) }
    }

    /// Largest batch `decode` accepts, as libllama settled it (may differ from the request).
    pub fn n_batch(&self) -> u32 {
        // SAFETY: live context.
        unsafe { sys::llama_n_batch(self.ptr) }
    }

    /// Runs the batch. Codes: 1 no KV slot, 2 aborted, -1 invalid batch, others fatal.
    pub fn decode(&mut self, batch: &Batch) -> Result<()> {
        // SAFETY: the batch arrays are owned by `batch` and hold `n_tokens` valid entries.
        let rc = unsafe { sys::llama_decode(self.ptr, batch.raw()) };
        self.outputs.clear();
        if rc == 0 {
            self.outputs.extend(batch.logits_flags());
        }
        match rc {
            0 => Ok(()),
            1 => Err(LlamaError::NoKvSlot),
            2 => Err(LlamaError::Aborted),
            -1 => Err(LlamaError::InvalidBatch),
            n => Err(LlamaError::DecodeFatal(n)),
        }
    }

    /// Waits for pending (asynchronous) backend work.
    pub fn synchronize(&mut self) {
        // SAFETY: live context.
        unsafe { sys::llama_synchronize(self.ptr) }
    }

    /// Logits of output `i` of the last decode (`-1` = the last output), `n_vocab` floats.
    /// Valid until the next decode.
    pub fn logits_ith(&mut self, i: i32) -> Result<&[f32]> {
        self.check_output(i)?;
        let n = usize::try_from(self.model.n_vocab()).unwrap_or(0);
        // SAFETY: live context; the index was validated against the last batch; a non-null
        // result points at `n_vocab` floats owned by the context, valid until the next decode,
        // which needs `&mut self` and so ends this borrow.
        unsafe {
            let p = sys::llama_get_logits_ith(self.ptr, i);
            if p.is_null() {
                return Err(LlamaError::NoLogits(i));
            }
            Ok(std::slice::from_raw_parts(p, n))
        }
    }

    /// Accepts `-n..=-1` (n-th output from the end) and batch indices that requested logits.
    pub(super) fn check_output(&self, i: i32) -> Result<()> {
        let n_out = self.outputs.iter().filter(|&&b| b).count();
        let ok = match usize::try_from(i) {
            Ok(i) => self.outputs.get(i).copied().unwrap_or(false),
            Err(_) => i.unsigned_abs() as usize <= n_out,
        };
        if ok {
            Ok(())
        } else {
            Err(LlamaError::NoLogits(i))
        }
    }

    /// Empties the KV cache (metadata only, like llama-bench).
    pub fn memory_clear(&mut self) {
        // SAFETY: live context; its memory handle is valid while the context is.
        unsafe { sys::llama_memory_clear(sys::llama_get_memory(self.ptr), false) }
    }

    /// Highest position stored for sequence `seq`, or -1 when empty.
    pub fn seq_pos_max(&self, seq: i32) -> i32 {
        // SAFETY: as `memory_clear`.
        unsafe { sys::llama_memory_seq_pos_max(sys::llama_get_memory(self.ptr), seq) }
    }

    pub fn set_threads(&mut self, n_threads: i32, n_threads_batch: i32) {
        // SAFETY: live context.
        unsafe { sys::llama_set_n_threads(self.ptr, n_threads, n_threads_batch) }
    }

    /// Uses `pool` for CPU compute. The context keeps the pool alive.
    pub fn attach_threadpool(&mut self, pool: Arc<ThreadPool>) {
        // SAFETY: both handles are live; the Arc stored below keeps the pool alive for as long
        // as the context, and the context is freed before its fields drop.
        unsafe { sys::llama_attach_threadpool(self.ptr, pool.ptr, std::ptr::null_mut()) };
        self.pool = Some(pool);
    }

    pub fn perf(&self) -> PerfData {
        // SAFETY: live context.
        let d = unsafe { sys::llama_perf_context(self.ptr) };
        PerfData {
            load_ms: d.t_load_ms,
            p_eval_ms: d.t_p_eval_ms,
            eval_ms: d.t_eval_ms,
            n_p_eval: d.n_p_eval,
            n_eval: d.n_eval,
            n_reused: d.n_reused,
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: owned and freed once; `model` and `pool` drop after this runs.
        unsafe { sys::llama_free(self.ptr) }
    }
}
