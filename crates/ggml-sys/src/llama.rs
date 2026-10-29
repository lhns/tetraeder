//! Safe wrapper over libllama (b11976, vendored; ADR 0026).
//!
//! Every handle frees itself on drop. Parameter structs start from libllama's defaults.
//! Buffers that libllama keeps pointers to (override patterns, tensor split) are owned by
//! the [`Model`]. `llama_backend_init` runs once and the backend is never freed in-process.
//!
//! Ownership: a [`Context`] holds an `Arc<Model>` (and an `Arc<ThreadPool>` once attached), so a
//! context can never outlive what it uses and an engine can own all three without
//! self-references.

use crate::raw as sys;
use std::ffi::{CStr, CString, c_char, c_void};
use std::path::Path;
use std::str::FromStr;
use std::sync::{Arc, OnceLock};

/// A vocabulary token id.
pub type Token = i32;

#[derive(Debug, thiserror::Error)]
pub enum LlamaError {
    #[error("cannot load model `{path}` (libllama gave no reason; see the `llama` log target)")]
    ModelLoad { path: String },
    #[error("cannot create context (see the `llama` log target; usually out of memory)")]
    ContextInit,
    #[error("invalid argument: {0}")]
    Invalid(String),
    #[error("unknown buffer type `{name}`; known: {}", known.join(", "))]
    UnknownBufferType { name: String, known: Vec<String> },
    #[error("decode failed: no KV slot for the batch (context too small or batch too large)")]
    NoKvSlot,
    #[error("decode aborted")]
    Aborted,
    #[error("decode failed: invalid batch")]
    InvalidBatch,
    #[error("decode failed with fatal error {0}")]
    DecodeFatal(i32),
    #[error("no logits for output index {0} (the token was not marked as an output)")]
    NoLogits(i32),
    #[error("tokenization failed: {0}")]
    Tokenize(String),
    #[error("token {0} has no text piece")]
    Piece(Token),
    #[error("batch is full ({0} tokens)")]
    BatchFull(usize),
    #[error("quantization failed (code {0}); see the `llama` log target")]
    Quantize(u32),
    #[error("cannot read GGUF file `{0}`")]
    Gguf(String),
}

pub type Result<T> = std::result::Result<T, LlamaError>;

// ---------------------------------------------------------------------------------------------
// Backend init and logging
// ---------------------------------------------------------------------------------------------

unsafe extern "C" fn log_callback(
    level: sys::ggml_log_level,
    text: *const c_char,
    _user: *mut c_void,
) {
    if text.is_null() {
        return;
    }
    // SAFETY: libllama passes a NUL-terminated string that is valid for the call.
    let s = unsafe { CStr::from_ptr(text) }.to_string_lossy();
    let s = s.trim_end_matches(['\n', '\r']);
    if s.is_empty() {
        return;
    }
    if level == sys::GGML_LOG_LEVEL_ERROR {
        tracing::error!(target: "llama", "{s}");
    } else if level == sys::GGML_LOG_LEVEL_WARN {
        tracing::warn!(target: "llama", "{s}");
    } else if level == sys::GGML_LOG_LEVEL_INFO {
        tracing::debug!(target: "llama", "{s}");
    } else {
        tracing::trace!(target: "llama", "{s}");
    }
}

/// Initializes libllama once and routes its logs to `tracing` (target `llama`: error, warn,
/// info as debug, rest as trace). Called by every constructor; safe to call repeatedly.
pub fn init() {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        // SAFETY: called once; the callback is a plain function that needs no user data.
        unsafe {
            sys::llama_log_set(Some(log_callback), std::ptr::null_mut());
            sys::llama_backend_init();
        }
    });
}

// ---------------------------------------------------------------------------------------------
// Devices and buffer types
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    Gpu,
    IGpu,
    Accel,
    Other,
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub description: String,
    pub kind: DeviceKind,
    pub memory_free: u64,
    pub memory_total: u64,
}

fn cstr_lossy(p: *const c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    // SAFETY: ggml returns NUL-terminated static or object-owned strings.
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

/// All ggml backend devices (CPU, GPUs, accelerators).
pub fn devices() -> Vec<DeviceInfo> {
    init();
    // SAFETY: the registry is initialized; device handles stay valid for the process.
    unsafe {
        (0..sys::ggml_backend_dev_count())
            .map(|i| {
                let d = sys::ggml_backend_dev_get(i);
                let (mut free, mut total) = (0usize, 0usize);
                sys::ggml_backend_dev_memory(d, &mut free, &mut total);
                let t = sys::ggml_backend_dev_type(d) as i64;
                let kind = if t == sys::GGML_BACKEND_DEVICE_TYPE_CPU as i64 {
                    DeviceKind::Cpu
                } else if t == sys::GGML_BACKEND_DEVICE_TYPE_GPU as i64 {
                    DeviceKind::Gpu
                } else if t == sys::GGML_BACKEND_DEVICE_TYPE_IGPU as i64 {
                    DeviceKind::IGpu
                } else if t == sys::GGML_BACKEND_DEVICE_TYPE_ACCEL as i64 {
                    DeviceKind::Accel
                } else {
                    DeviceKind::Other
                };
                DeviceInfo {
                    name: cstr_lossy(sys::ggml_backend_dev_name(d)),
                    description: cstr_lossy(sys::ggml_backend_dev_description(d)),
                    kind,
                    memory_free: free as u64,
                    memory_total: total as u64,
                }
            })
            .collect()
    }
}

/// A ggml buffer type (where tensors live), for example `CPU` or `CUDA0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferType(sys::ggml_backend_buffer_type_t);

// SAFETY: buffer types are immutable objects owned by the backend registry for the whole
// process; this wrapper only reads their name and hands the pointer back to libllama.
unsafe impl Send for BufferType {}
// SAFETY: see `Send`.
unsafe impl Sync for BufferType {}

impl BufferType {
    pub fn name(&self) -> String {
        // SAFETY: the handle is non-null and registry-owned.
        cstr_lossy(unsafe { sys::ggml_backend_buft_name(self.0) })
    }
}

fn all_buffer_types() -> Vec<BufferType> {
    init();
    let mut out = Vec::new();
    // SAFETY: registry handles are valid; the host buffer type may be null.
    unsafe {
        for i in 0..sys::ggml_backend_dev_count() {
            let d = sys::ggml_backend_dev_get(i);
            for b in [
                sys::ggml_backend_dev_buffer_type(d),
                sys::ggml_backend_dev_host_buffer_type(d),
            ] {
                if !b.is_null() && !out.contains(&BufferType(b)) {
                    out.push(BufferType(b));
                }
            }
        }
    }
    out
}

/// Names accepted by [`buffer_type`].
pub fn buffer_type_names() -> Vec<String> {
    all_buffer_types().iter().map(BufferType::name).collect()
}

/// Looks a buffer type up by name, case-insensitively, over every device's buffer type and
/// host buffer type (the set `llama-cli --override-tensor` accepts, plus host buffers).
pub fn buffer_type(name: &str) -> Option<BufferType> {
    all_buffer_types()
        .into_iter()
        .find(|b| b.name().eq_ignore_ascii_case(name))
}

/// Like [`buffer_type`], with an error that lists the known names.
pub fn resolve_buffer_type(name: &str) -> Result<BufferType> {
    buffer_type(name).ok_or_else(|| LlamaError::UnknownBufferType {
        name: name.to_owned(),
        known: buffer_type_names(),
    })
}

// ---------------------------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitMode {
    None,
    Layer,
    Row,
    Tensor,
}

impl SplitMode {
    fn raw(self) -> sys::llama_split_mode {
        (match self {
            Self::None => sys::LLAMA_SPLIT_MODE_NONE,
            Self::Layer => sys::LLAMA_SPLIT_MODE_LAYER,
            Self::Row => sys::LLAMA_SPLIT_MODE_ROW,
            Self::Tensor => sys::LLAMA_SPLIT_MODE_TENSOR,
        }) as _
    }

    fn from_raw(v: sys::llama_split_mode) -> Self {
        match v as i64 {
            x if x == sys::LLAMA_SPLIT_MODE_NONE as i64 => Self::None,
            x if x == sys::LLAMA_SPLIT_MODE_ROW as i64 => Self::Row,
            x if x == sys::LLAMA_SPLIT_MODE_TENSOR as i64 => Self::Tensor,
            _ => Self::Layer,
        }
    }
}

impl FromStr for SplitMode {
    type Err = LlamaError;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "none" => Ok(Self::None),
            "layer" => Ok(Self::Layer),
            "row" => Ok(Self::Row),
            "tensor" => Ok(Self::Tensor),
            _ => Err(LlamaError::Invalid(format!(
                "unknown split mode `{s}` (none, layer, row, tensor)"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadMode {
    Auto,
    None,
    Mmap,
    Mlock,
    MmapMlock,
    DirectIo,
}

impl LoadMode {
    fn raw(self) -> sys::llama_load_mode {
        (match self {
            Self::Auto => sys::LLAMA_LOAD_MODE_AUTO,
            Self::None => sys::LLAMA_LOAD_MODE_NONE,
            Self::Mmap => sys::LLAMA_LOAD_MODE_MMAP,
            Self::Mlock => sys::LLAMA_LOAD_MODE_MLOCK,
            Self::MmapMlock => sys::LLAMA_LOAD_MODE_MMAP_MLOCK,
            Self::DirectIo => sys::LLAMA_LOAD_MODE_DIRECT_IO,
        }) as _
    }

    fn from_raw(v: sys::llama_load_mode) -> Self {
        match v as i64 {
            x if x == sys::LLAMA_LOAD_MODE_NONE as i64 => Self::None,
            x if x == sys::LLAMA_LOAD_MODE_MMAP as i64 => Self::Mmap,
            x if x == sys::LLAMA_LOAD_MODE_MLOCK as i64 => Self::Mlock,
            x if x == sys::LLAMA_LOAD_MODE_MMAP_MLOCK as i64 => Self::MmapMlock,
            x if x == sys::LLAMA_LOAD_MODE_DIRECT_IO as i64 => Self::DirectIo,
            _ => Self::Auto,
        }
    }
}

impl FromStr for LoadMode {
    type Err = LlamaError;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "none" => Ok(Self::None),
            "mmap" => Ok(Self::Mmap),
            "mlock" => Ok(Self::Mlock),
            "mmap-mlock" => Ok(Self::MmapMlock),
            "direct-io" => Ok(Self::DirectIo),
            _ => Err(LlamaError::Invalid(format!(
                "unknown load mode `{s}` (auto, none, mmap, mlock, mmap-mlock, direct-io)"
            ))),
        }
    }
}

/// Model load parameters. `Default` is libllama's defaults.
#[derive(Debug, Clone)]
pub struct ModelParams {
    /// Layers stored on GPU; negative means all.
    pub n_gpu_layers: i32,
    pub split_mode: SplitMode,
    pub main_gpu: i32,
    /// Share of the model per device; at most `llama_max_devices()` entries, empty = default.
    pub tensor_split: Vec<f32>,
    /// `(regex, buffer type)`: tensors whose name matches the regex (unanchored search) are
    /// placed in that buffer type. First match wins.
    pub buffer_overrides: Vec<(String, BufferType)>,
    pub load_mode: LoadMode,
    pub vocab_only: bool,
}

impl Default for ModelParams {
    fn default() -> Self {
        init();
        // SAFETY: plain value return.
        let d = unsafe { sys::llama_model_default_params() };
        Self {
            n_gpu_layers: d.n_gpu_layers,
            split_mode: SplitMode::from_raw(d.split_mode),
            main_gpu: d.main_gpu,
            tensor_split: Vec::new(),
            buffer_overrides: Vec::new(),
            load_mode: LoadMode::from_raw(d.load_mode),
            vocab_only: false,
        }
    }
}

/// Memory libllama keeps pointers to for the life of the model.
struct Keepalive {
    _patterns: Vec<CString>,
    _overrides: Vec<sys::llama_model_tensor_buft_override>,
    _split: Vec<f32>,
}

/// A loaded model. Immutable after load.
pub struct Model {
    ptr: *mut sys::llama_model,
    vocab: *const sys::llama_vocab,
    _keep: Keepalive,
}

// SAFETY: a loaded llama_model is read-only; libllama (and its server) share one model across
// threads and contexts. The keepalive buffers are only read by libllama.
unsafe impl Send for Model {}
// SAFETY: see `Send`.
unsafe impl Sync for Model {}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Model({})", self.desc())
    }
}

fn path_cstring(path: &Path) -> Result<CString> {
    let s = path
        .to_str()
        .ok_or_else(|| LlamaError::Invalid(format!("path is not UTF-8: {}", path.display())))?;
    CString::new(s).map_err(|_| LlamaError::Invalid("path contains a NUL byte".into()))
}

impl Model {
    pub fn load(path: impl AsRef<Path>, params: &ModelParams) -> Result<Model> {
        init();
        let path = path.as_ref();
        let cpath = path_cstring(path)?;

        // SAFETY: plain value returns.
        let (max_dev, max_ovr) = unsafe {
            (
                sys::llama_max_devices(),
                sys::llama_max_tensor_buft_overrides(),
            )
        };
        if params.tensor_split.len() > max_dev {
            return Err(LlamaError::Invalid(format!(
                "tensor_split has {} entries but this build supports {max_dev} devices",
                params.tensor_split.len()
            )));
        }
        // The terminating entry takes one slot.
        if params.buffer_overrides.len() + 1 > max_ovr {
            return Err(LlamaError::Invalid(format!(
                "{} buffer overrides given, at most {} are supported",
                params.buffer_overrides.len(),
                max_ovr.saturating_sub(1)
            )));
        }

        let mut patterns = Vec::new();
        let mut overrides = Vec::new();
        for (pat, buft) in &params.buffer_overrides {
            let c = CString::new(pat.as_str())
                .map_err(|_| LlamaError::Invalid("override pattern contains a NUL byte".into()))?;
            // The CString's heap buffer does not move when the Vec grows or the Model moves.
            overrides.push(sys::llama_model_tensor_buft_override {
                pattern: c.as_ptr(),
                buft: buft.0,
            });
            patterns.push(c);
        }
        let mut split = Vec::new();
        if !params.tensor_split.is_empty() {
            split = params.tensor_split.clone();
            split.resize(max_dev, 0.0);
        }

        // SAFETY: plain value return.
        let mut p = unsafe { sys::llama_model_default_params() };
        p.n_gpu_layers = params.n_gpu_layers;
        p.split_mode = params.split_mode.raw();
        p.load_mode = params.load_mode.raw();
        p.main_gpu = params.main_gpu;
        p.vocab_only = params.vocab_only;
        if !split.is_empty() {
            p.tensor_split = split.as_ptr();
        }
        if !overrides.is_empty() {
            overrides.push(sys::llama_model_tensor_buft_override {
                pattern: std::ptr::null(),
                buft: std::ptr::null_mut(),
            });
            p.tensor_buft_overrides = overrides.as_ptr();
        }

        // SAFETY: `cpath` is NUL-terminated; the pointers in `p` point into `patterns`,
        // `overrides` and `split`, which are moved into the returned Model.
        let ptr = unsafe { sys::llama_model_load_from_file(cpath.as_ptr(), p) };
        if ptr.is_null() {
            return Err(LlamaError::ModelLoad {
                path: path.display().to_string(),
            });
        }
        // SAFETY: `ptr` is a live model.
        let vocab = unsafe { sys::llama_model_get_vocab(ptr) };
        Ok(Model {
            ptr,
            vocab,
            _keep: Keepalive {
                _patterns: patterns,
                _overrides: overrides,
                _split: split,
            },
        })
    }

    /// Short description, for example `qwen3 0.6B Q8_0`.
    pub fn desc(&self) -> String {
        let mut buf = vec![0u8; 256];
        // SAFETY: the buffer length is passed; libllama NUL-terminates within it.
        let n = unsafe {
            sys::llama_model_desc(self.ptr, buf.as_mut_ptr().cast::<c_char>(), buf.len())
        };
        buf.truncate(usize::try_from(n).unwrap_or(0).min(255));
        String::from_utf8_lossy(&buf).into_owned()
    }

    /// Total size of all tensors in bytes.
    pub fn size_bytes(&self) -> u64 {
        // SAFETY: live model.
        unsafe { sys::llama_model_size(self.ptr) }
    }

    pub fn n_params(&self) -> u64 {
        // SAFETY: live model.
        unsafe { sys::llama_model_n_params(self.ptr) }
    }

    pub fn n_layer(&self) -> i32 {
        // SAFETY: live model.
        unsafe { sys::llama_model_n_layer(self.ptr) }
    }

    pub fn n_ctx_train(&self) -> i32 {
        // SAFETY: live model.
        unsafe { sys::llama_model_n_ctx_train(self.ptr) }
    }

    pub fn n_vocab(&self) -> i32 {
        // SAFETY: the vocab pointer belongs to the live model.
        unsafe { sys::llama_vocab_n_tokens(self.vocab) }
    }

    /// Whether the tokenizer prepends BOS when `add_special` is set.
    pub fn add_bos(&self) -> bool {
        // SAFETY: as `n_vocab`.
        unsafe { sys::llama_vocab_get_add_bos(self.vocab) }
    }

    pub fn bos(&self) -> Token {
        // SAFETY: as `n_vocab`.
        unsafe { sys::llama_vocab_bos(self.vocab) }
    }

    pub fn is_eog(&self, token: Token) -> bool {
        // SAFETY: as `n_vocab`.
        unsafe { sys::llama_vocab_is_eog(self.vocab, token) }
    }

    /// Text to tokens. `add_special` adds BOS/EOS as the model asks; `parse_special` turns
    /// special-token text such as `<|im_start|>` into their ids.
    pub fn tokenize(
        &self,
        text: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<Token>> {
        let len = i32::try_from(text.len())
            .map_err(|_| LlamaError::Tokenize("text longer than i32::MAX bytes".into()))?;
        let mut out: Vec<Token> = vec![0; text.len() / 2 + 8];
        for _ in 0..2 {
            let cap = i32::try_from(out.len())
                .map_err(|_| LlamaError::Tokenize("too many tokens".into()))?;
            // SAFETY: `text` is valid for `len` bytes (no NUL needed); `out` for `cap` tokens.
            let n = unsafe {
                sys::llama_tokenize(
                    self.vocab,
                    text.as_ptr().cast::<c_char>(),
                    len,
                    out.as_mut_ptr(),
                    cap,
                    add_special,
                    parse_special,
                )
            };
            if n >= 0 {
                out.truncate(n as usize);
                return Ok(out);
            }
            if n == i32::MIN {
                return Err(LlamaError::Tokenize("token count overflow".into()));
            }
            out.resize(n.unsigned_abs() as usize, 0);
        }
        Err(LlamaError::Tokenize(
            "size protocol did not converge".into(),
        ))
    }

    /// Token to raw text bytes (may be a partial UTF-8 sequence). `special` renders special
    /// tokens as text.
    pub fn token_to_piece(&self, token: Token, special: bool) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; 32];
        for _ in 0..2 {
            // SAFETY: `buf` is valid for its length; no NUL terminator is written.
            let n = unsafe {
                sys::llama_token_to_piece(
                    self.vocab,
                    token,
                    buf.as_mut_ptr().cast::<c_char>(),
                    buf.len() as i32,
                    0,
                    special,
                )
            };
            if n >= 0 {
                buf.truncate(n as usize);
                return Ok(buf);
            }
            buf.resize(n.unsigned_abs() as usize, 0);
        }
        Err(LlamaError::Piece(token))
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        // SAFETY: the pointer is owned and freed exactly once; contexts hold an Arc<Model>.
        unsafe { sys::llama_model_free(self.ptr) }
    }
}

// ---------------------------------------------------------------------------------------------
// Thread pool, context, batch
// ---------------------------------------------------------------------------------------------

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
    ptr: *mut sys::llama_context,
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
        // SAFETY: live context; the index was validated against the last batch; a non-null result points at `n_vocab` floats owned by the
        // context, valid until the next decode, which needs `&mut self` and so ends this borrow.
        unsafe {
            let p = sys::llama_get_logits_ith(self.ptr, i);
            if p.is_null() {
                return Err(LlamaError::NoLogits(i));
            }
            Ok(std::slice::from_raw_parts(p, n))
        }
    }

    /// Accepts `-n..=-1` (n-th output from the end) and batch indices that requested logits.
    fn check_output(&self, i: i32) -> Result<()> {
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

/// A token batch for one sequence set, backed by `llama_batch_init`.
pub struct Batch {
    raw: sys::llama_batch,
    cap: usize,
    len: usize,
}

// SAFETY: the batch exclusively owns its heap arrays; there is no shared state.
unsafe impl Send for Batch {}

impl Batch {
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        // SAFETY: allocates arrays for `cap` tokens and one sequence id per token.
        let raw = unsafe { sys::llama_batch_init(cap as i32, 0, 1) };
        Self { raw, cap, len: 0 }
    }

    pub fn capacity(&self) -> usize {
        self.cap
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Appends a token at position `pos` of sequence `seq`; `logits` requests its output.
    pub fn push(&mut self, token: Token, pos: i32, seq: i32, logits: bool) -> Result<()> {
        if self.len == self.cap {
            return Err(LlamaError::BatchFull(self.cap));
        }
        let i = self.len;
        // SAFETY: `i < cap`, and every array has `cap` entries; each `seq_id[i]` points at a
        // one-element array (n_seq_max = 1).
        unsafe {
            *self.raw.token.add(i) = token;
            *self.raw.pos.add(i) = pos;
            *self.raw.n_seq_id.add(i) = 1;
            **self.raw.seq_id.add(i) = seq;
            *self.raw.logits.add(i) = i8::from(logits);
        }
        self.len += 1;
        Ok(())
    }

    fn logits_flags(&self) -> impl Iterator<Item = bool> + '_ {
        // SAFETY: the first `len` entries of `logits` were written by `push`.
        (0..self.len).map(|i| unsafe { *self.raw.logits.add(i) } != 0)
    }

    fn raw(&self) -> sys::llama_batch {
        let mut b = self.raw;
        b.n_tokens = self.len as i32;
        b
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        // SAFETY: allocated by llama_batch_init and freed once.
        unsafe { sys::llama_batch_free(self.raw) }
    }
}

// ---------------------------------------------------------------------------------------------
// Sampler
// ---------------------------------------------------------------------------------------------

/// A sampler chain. Build with [`Sampler::chain`] and the adders, or use [`Sampler::greedy`].
pub struct Sampler {
    ptr: *mut sys::llama_sampler,
}

// SAFETY: a sampler chain owns its state and has no thread affinity; it is not Sync.
unsafe impl Send for Sampler {}

impl Sampler {
    /// An empty chain.
    pub fn chain() -> Self {
        init();
        // SAFETY: default params are plain values; the chain is owned by the returned value.
        let ptr = unsafe {
            let mut p = sys::llama_sampler_chain_default_params();
            p.no_perf = true;
            sys::llama_sampler_chain_init(p)
        };
        Self { ptr }
    }

    fn add(self, s: *mut sys::llama_sampler) -> Self {
        // SAFETY: the chain takes ownership of `s`.
        unsafe { sys::llama_sampler_chain_add(self.ptr, s) };
        self
    }

    pub fn greedy() -> Self {
        // SAFETY: constructor returning an owned sampler.
        Self::chain().add(unsafe { sys::llama_sampler_init_greedy() })
    }

    pub fn top_k(self, k: i32) -> Self {
        // SAFETY: as `greedy`.
        self.add(unsafe { sys::llama_sampler_init_top_k(k) })
    }

    pub fn top_p(self, p: f32, min_keep: usize) -> Self {
        // SAFETY: as `greedy`.
        self.add(unsafe { sys::llama_sampler_init_top_p(p, min_keep) })
    }

    pub fn temperature(self, t: f32) -> Self {
        // SAFETY: as `greedy`.
        self.add(unsafe { sys::llama_sampler_init_temp(t) })
    }

    /// Final stage of a random chain: draws from the distribution with a seeded RNG.
    pub fn dist(self, seed: u32) -> Self {
        // SAFETY: as `greedy`.
        self.add(unsafe { sys::llama_sampler_init_dist(seed) })
    }

    /// Samples from output `idx` of the last decode (`-1` = last) and accepts the token into
    /// the chain's state.
    pub fn sample(&mut self, ctx: &mut Context, idx: i32) -> Result<Token> {
        ctx.check_output(idx)?;
        // SAFETY: both handles are live; `idx` was validated against the last batch's outputs.
        Ok(unsafe { sys::llama_sampler_sample(self.ptr, ctx.ptr, idx) })
    }

    /// Feeds a token into the chain's state (for example repetition history). `sample`
    /// already does this for the token it returns.
    pub fn accept(&mut self, token: Token) {
        // SAFETY: live sampler.
        unsafe { sys::llama_sampler_accept(self.ptr, token) }
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        // SAFETY: owned chain; its members are freed with it.
        unsafe { sys::llama_sampler_free(self.ptr) }
    }
}

// ---------------------------------------------------------------------------------------------
// Quantization and GGUF inspection
// ---------------------------------------------------------------------------------------------

/// Target file type of a quantization run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ftype {
    F16,
    Q4_0,
    Q4_1,
    Q8_0,
    Q4KM,
    Q5KM,
    Q6K,
}

impl Ftype {
    fn raw(self) -> sys::llama_ftype {
        (match self {
            Self::F16 => sys::LLAMA_FTYPE_MOSTLY_F16,
            Self::Q4_0 => sys::LLAMA_FTYPE_MOSTLY_Q4_0,
            Self::Q4_1 => sys::LLAMA_FTYPE_MOSTLY_Q4_1,
            Self::Q8_0 => sys::LLAMA_FTYPE_MOSTLY_Q8_0,
            Self::Q4KM => sys::LLAMA_FTYPE_MOSTLY_Q4_K_M,
            Self::Q5KM => sys::LLAMA_FTYPE_MOSTLY_Q5_K_M,
            Self::Q6K => sys::LLAMA_FTYPE_MOSTLY_Q6_K,
        }) as _
    }
}

/// A ggml tensor storage type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Bf16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
}

impl GgmlType {
    fn raw(self) -> sys::ggml_type {
        (match self {
            Self::F32 => sys::GGML_TYPE_F32,
            Self::F16 => sys::GGML_TYPE_F16,
            Self::Bf16 => sys::GGML_TYPE_BF16,
            Self::Q4_0 => sys::GGML_TYPE_Q4_0,
            Self::Q4_1 => sys::GGML_TYPE_Q4_1,
            Self::Q5_0 => sys::GGML_TYPE_Q5_0,
            Self::Q5_1 => sys::GGML_TYPE_Q5_1,
            Self::Q8_0 => sys::GGML_TYPE_Q8_0,
            Self::Q2K => sys::GGML_TYPE_Q2_K,
            Self::Q3K => sys::GGML_TYPE_Q3_K,
            Self::Q4K => sys::GGML_TYPE_Q4_K,
            Self::Q5K => sys::GGML_TYPE_Q5_K,
            Self::Q6K => sys::GGML_TYPE_Q6_K,
        }) as _
    }
}

#[derive(Debug, Clone)]
pub struct QuantizeParams {
    pub ftype: Ftype,
    /// <= 0: all hardware threads.
    pub nthread: i32,
    /// Allow quantizing tensors that are already quantized (lossy).
    pub allow_requantize: bool,
    /// `(regex, type)`: a tensor whose name matches the regex (`std::regex_search`, ECMAScript,
    /// unanchored, so `output\.weight` also matches `foo.output.weight`) gets that type instead
    /// of the one the ftype picks. First matching entry wins. Only applied when the ftype is a
    /// quantized type. Escape dots.
    pub tensor_types: Vec<(String, GgmlType)>,
}

impl QuantizeParams {
    pub fn new(ftype: Ftype) -> Self {
        Self {
            ftype,
            nthread: 0,
            allow_requantize: false,
            tensor_types: Vec::new(),
        }
    }
}

/// Quantizes `src` into `dst`. An invalid override regex surfaces as `Quantize(1)` with the
/// reason in the log (libllama catches the exception).
pub fn quantize(
    src: impl AsRef<Path>,
    dst: impl AsRef<Path>,
    params: &QuantizeParams,
) -> Result<()> {
    init();
    let (src, dst) = (path_cstring(src.as_ref())?, path_cstring(dst.as_ref())?);
    let mut patterns = Vec::new();
    let mut overrides = Vec::new();
    for (pat, ty) in &params.tensor_types {
        let c = CString::new(pat.as_str())
            .map_err(|_| LlamaError::Invalid("type override pattern contains a NUL byte".into()))?;
        overrides.push(sys::llama_model_tensor_override {
            pattern: c.as_ptr(),
            type_: ty.raw(),
        });
        patterns.push(c);
    }
    // SAFETY: plain value return.
    let mut p = unsafe { sys::llama_model_quantize_default_params() };
    p.ftype = params.ftype.raw();
    p.nthread = params.nthread;
    p.allow_requantize = params.allow_requantize;
    if !overrides.is_empty() {
        overrides.push(sys::llama_model_tensor_override {
            pattern: std::ptr::null(),
            type_: sys::GGML_TYPE_F32 as _,
        });
        p.tt_overrides = overrides.as_ptr();
    }
    // SAFETY: paths are NUL-terminated; `p` points into `overrides`/`patterns`, alive here.
    let rc = unsafe { sys::llama_model_quantize(src.as_ptr(), dst.as_ptr(), &p) };
    drop(patterns);
    if rc == 0 {
        Ok(())
    } else {
        Err(LlamaError::Quantize(rc))
    }
}

/// `(tensor name, ggml type name)` for every tensor of a GGUF file, read without loading data.
pub fn gguf_tensor_types(path: impl AsRef<Path>) -> Result<Vec<(String, String)>> {
    init();
    let path = path.as_ref();
    let cpath = path_cstring(path)?;
    let params = sys::gguf_init_params {
        no_alloc: true,
        ctx: std::ptr::null_mut(),
    };
    // SAFETY: `cpath` is NUL-terminated; the context is freed below on every path.
    let g = unsafe { sys::gguf_init_from_file(cpath.as_ptr(), params) };
    if g.is_null() {
        return Err(LlamaError::Gguf(path.display().to_string()));
    }
    // SAFETY: `g` is live; ids are below the tensor count; returned strings live as long as `g`.
    let out = unsafe {
        (0..sys::gguf_get_n_tensors(g))
            .map(|i| {
                (
                    cstr_lossy(sys::gguf_get_tensor_name(g, i)),
                    cstr_lossy(sys::ggml_type_name(sys::gguf_get_tensor_type(g, i))),
                )
            })
            .collect()
    };
    // SAFETY: `g` came from gguf_init_from_file and is freed once.
    unsafe { sys::gguf_free(g) };
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enum_parsing() {
        assert_eq!("Layer".parse::<SplitMode>().unwrap(), SplitMode::Layer);
        assert_eq!("tensor".parse::<SplitMode>().unwrap(), SplitMode::Tensor);
        assert!("bogus".parse::<SplitMode>().is_err());
        assert_eq!(
            "mmap-mlock".parse::<LoadMode>().unwrap(),
            LoadMode::MmapMlock
        );
        assert_eq!("direct-io".parse::<LoadMode>().unwrap(), LoadMode::DirectIo);
        assert!("mmap_mlock".parse::<LoadMode>().is_err());
    }

    #[test]
    fn enum_raw_roundtrip() {
        for m in [
            SplitMode::None,
            SplitMode::Layer,
            SplitMode::Row,
            SplitMode::Tensor,
        ] {
            assert_eq!(SplitMode::from_raw(m.raw()), m);
        }
        for m in [
            LoadMode::Auto,
            LoadMode::None,
            LoadMode::Mmap,
            LoadMode::Mlock,
            LoadMode::MmapMlock,
            LoadMode::DirectIo,
        ] {
            assert_eq!(LoadMode::from_raw(m.raw()), m);
        }
    }

    #[test]
    fn cpu_device_and_buffer_type_exist() {
        let devs = devices();
        assert!(devs.iter().any(|d| d.kind == DeviceKind::Cpu));
        let cpu = buffer_type("cpu").expect("CPU buffer type");
        assert_eq!(cpu.name(), "CPU");
        assert!(buffer_type("no-such-buffer").is_none());
        assert!(matches!(
            resolve_buffer_type("nope"),
            Err(LlamaError::UnknownBufferType { known, .. }) if known.iter().any(|k| k == "CPU")
        ));
    }

    #[test]
    fn default_params_follow_libllama() {
        let m = ModelParams::default();
        assert!(m.tensor_split.is_empty() && m.buffer_overrides.is_empty() && !m.vocab_only);
        let c = ContextParams::default();
        assert!(c.n_batch > 0 && c.n_ubatch > 0);
    }

    #[test]
    fn batch_capacity_is_checked() {
        let mut b = Batch::new(2);
        assert!(b.is_empty());
        b.push(1, 0, 0, false).unwrap();
        b.push(2, 1, 0, true).unwrap();
        assert!(matches!(
            b.push(3, 2, 0, true),
            Err(LlamaError::BatchFull(2))
        ));
        assert_eq!(b.len(), 2);
        b.clear();
        assert!(b.is_empty());
    }

    #[test]
    fn missing_model_is_an_error() {
        let e = Model::load("definitely/not/here.gguf", &ModelParams::default()).unwrap_err();
        assert!(matches!(e, LlamaError::ModelLoad { .. }));
    }
}
