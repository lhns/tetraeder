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
use std::sync::OnceLock;

mod batch;
mod context;
mod devices;
mod model;
mod quantize;
mod sampler;

pub use batch::Batch;
pub use context::{Context, ContextParams, FlashAttn, PerfData, ThreadPool};
pub use devices::{
    BufferType, DeviceInfo, DeviceKind, buffer_type, buffer_type_names, devices,
    resolve_buffer_type,
};
pub use model::{LoadMode, Model, ModelParams, SplitMode};
pub use quantize::{Ftype, GgmlType, QuantizeParams, gguf_tensor_types, quantize};
pub use sampler::Sampler;

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

fn cstr_lossy(p: *const c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    // SAFETY: ggml returns NUL-terminated static or object-owned strings.
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

fn path_cstring(path: &Path) -> Result<CString> {
    let s = path
        .to_str()
        .ok_or_else(|| LlamaError::Invalid(format!("path is not UTF-8: {}", path.display())))?;
    CString::new(s).map_err(|_| LlamaError::Invalid("path contains a NUL byte".into()))
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
