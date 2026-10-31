//! Model parameters, loading and the vocabulary calls.

use super::devices::BufferType;
use super::{LlamaError, Result, Token, init, path_cstring};
use crate::raw as sys;
use std::ffi::{CString, c_char};
use std::path::Path;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitMode {
    None,
    Layer,
    Row,
    Tensor,
}

impl SplitMode {
    pub(super) fn raw(self) -> sys::llama_split_mode {
        (match self {
            Self::None => sys::LLAMA_SPLIT_MODE_NONE,
            Self::Layer => sys::LLAMA_SPLIT_MODE_LAYER,
            Self::Row => sys::LLAMA_SPLIT_MODE_ROW,
            Self::Tensor => sys::LLAMA_SPLIT_MODE_TENSOR,
        }) as _
    }

    pub(super) fn from_raw(v: sys::llama_split_mode) -> Self {
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
    pub(super) fn raw(self) -> sys::llama_load_mode {
        (match self {
            Self::Auto => sys::LLAMA_LOAD_MODE_AUTO,
            Self::None => sys::LLAMA_LOAD_MODE_NONE,
            Self::Mmap => sys::LLAMA_LOAD_MODE_MMAP,
            Self::Mlock => sys::LLAMA_LOAD_MODE_MLOCK,
            Self::MmapMlock => sys::LLAMA_LOAD_MODE_MMAP_MLOCK,
            Self::DirectIo => sys::LLAMA_LOAD_MODE_DIRECT_IO,
        }) as _
    }

    pub(super) fn from_raw(v: sys::llama_load_mode) -> Self {
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
    pub(super) ptr: *mut sys::llama_model,
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
