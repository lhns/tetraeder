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
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

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
    #[error("ggml backends not loaded: {0}")]
    BackendLoad(String),
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

static CUDA_WARMUPS: AtomicU64 = AtomicU64::new(0);
static CUDA_RESETS: AtomicU64 = AtomicU64::new(0);
static CUDA_REUSED: AtomicU64 = AtomicU64::new(0);
static CUDA_UPDATE_FAILED: AtomicU64 = AtomicU64::new(0);

/// How often ggml-cuda logged its CUDA graph events since process start. It logs them only when
/// the process starts with `GGML_CUDA_GRAPH_DEBUG` set; all zero otherwise.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CudaGraphLogCounts {
    /// Captures: a graph finished its warmup and was captured.
    pub warmups: u64,
    /// The warmup counter restarted because the graph's properties changed.
    pub resets: u64,
    /// A graph with the uid of the last one was launched without comparing properties.
    pub reused: u64,
    /// Updating the captured graph failed.
    pub update_failed: u64,
}

pub fn cuda_graph_log_counts() -> CudaGraphLogCounts {
    let get = |c: &AtomicU64| c.load(Ordering::Relaxed);
    CudaGraphLogCounts {
        warmups: get(&CUDA_WARMUPS),
        resets: get(&CUDA_RESETS),
        reused: get(&CUDA_REUSED),
        update_failed: get(&CUDA_UPDATE_FAILED),
    }
}

/// Counts the CUDA graph messages of ggml-cuda (`llama_log_set` routes ggml's logger here too).
fn count_cuda_graph_event(s: &str) {
    let counter = if s.contains("CUDA graph warmup complete") {
        &CUDA_WARMUPS
    } else if s.contains("CUDA graph warmup reset") {
        &CUDA_RESETS
    } else if s.contains("CUDA Graph id") && s.contains("reused") {
        &CUDA_REUSED
    } else if s.contains("CUDA graph update failed") {
        &CUDA_UPDATE_FAILED
    } else {
        return;
    };
    counter.fetch_add(1, Ordering::Relaxed);
}

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
    count_cuda_graph_event(s);
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

/// Where the ggml backends came from (ADR 0052).
#[derive(Debug)]
pub(crate) struct Loaded {
    /// DL build: the directory the modules were loaded from.
    pub module_dir: Option<PathBuf>,
    /// `TETRAEDER_GGML_CPU_VARIANT`, when set.
    pub forced: Option<String>,
}

/// Initializes libllama once and routes its logs to `tracing` (target `llama`: error, warn,
/// info as debug, rest as trace). Called by every constructor; safe to call repeatedly.
/// A module load error is kept for the constructors (`init_checked`); `init` ignores it.
pub fn init() {
    let _ = init_checked();
}

/// Like [`init`], but returns the backend load error (DL build: a CPU or GPU module that is
/// missing or does not load).
pub(crate) fn init_checked() -> std::result::Result<&'static Loaded, String> {
    static INIT: OnceLock<std::result::Result<Loaded, String>> = OnceLock::new();
    INIT.get_or_init(|| {
        // SAFETY: called once; the callback is a plain function that needs no user data.
        unsafe { sys::llama_log_set(Some(log_callback), std::ptr::null_mut()) };
        let loaded = load_backends();
        // SAFETY: called once. With backends registered it skips its own load.
        unsafe { sys::llama_backend_init() };
        loaded
    })
    .as_ref()
    .map_err(Clone::clone)
}

#[cfg(not(ggml_dl))]
fn load_backends() -> std::result::Result<Loaded, String> {
    Ok(Loaded {
        module_dir: None,
        forced: None,
    })
}

/// Loads the ggml modules: the best CPU variant and the GPU modules of this build, or with
/// `TETRAEDER_GGML_CPU_VARIANT=<tag>` that one CPU module and the GPU modules.
#[cfg(ggml_dl)]
fn load_backends() -> std::result::Result<Loaded, String> {
    use crate::ggml::cpu;
    let env_dir = std::env::var_os("TETRAEDER_GGML_BACKEND_DIR").map(PathBuf::from);
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf));
    let dir = cpu::resolve_module_dir(
        env_dir.as_deref(),
        exe_dir.as_deref(),
        Path::new(env!("TETRAEDER_GGML_MODULE_DIR")),
        cpu::dir_has_cpu_module,
    );
    let forced = std::env::var("TETRAEDER_GGML_CPU_VARIANT")
        .ok()
        .filter(|v| !v.is_empty());
    let load_file = |file: &Path| -> bool {
        let Ok(c) = path_cstring(file) else {
            return false;
        };
        // SAFETY: valid C string; null means the module is missing or not usable here.
        !unsafe { sys::ggml_backend_load(c.as_ptr()) }.is_null()
    };
    if let Some(tag) = &forced {
        let file = dir.join(cpu::cpu_module_file_name(tag));
        if !load_file(&file) {
            return Err(format!(
                "cannot load CPU module {} (TETRAEDER_GGML_CPU_VARIANT={tag}): the file is missing, \
                 or this CPU cannot run it",
                file.display()
            ));
        }
        for gpu in ["ggml-cuda", "ggml-vulkan"] {
            let file = dir.join(cpu::module_file_name(gpu));
            if file.is_file() {
                load_file(&file);
            }
        }
    } else {
        let c = path_cstring(&dir).map_err(|e| e.to_string())?;
        // SAFETY: valid C string.
        unsafe { sys::ggml_backend_load_all_from_path(c.as_ptr()) };
    }
    // SAFETY: registry query.
    if unsafe { sys::ggml_backend_dev_by_type(sys::GGML_BACKEND_DEVICE_TYPE_CPU) }.is_null() {
        return Err(format!(
            "no CPU backend after loading modules from {}: no usable {} there (set \
             TETRAEDER_GGML_BACKEND_DIR to the directory with the ggml modules)",
            dir.display(),
            cpu::cpu_module_file_name("<variant>")
        ));
    }
    for (feature, reg, module) in [
        (cfg!(feature = "cuda"), c"CUDA", "ggml-cuda"),
        (cfg!(feature = "vulkan"), c"Vulkan", "ggml-vulkan"),
    ] {
        // SAFETY: registry query with a valid C string.
        if feature && unsafe { sys::ggml_backend_reg_by_name(reg.as_ptr()) }.is_null() {
            return Err(format!(
                "{} module not loaded: {} in {} is missing or failed to load (driver or runtime \
                 library not found?)",
                reg.to_string_lossy(),
                cpu::module_file_name(module),
                dir.display()
            ));
        }
    }
    tracing::info!(
        dir = %dir.display(),
        forced = forced.as_deref().unwrap_or("none"),
        backends = %registry_names().join(", "),
        "ggml modules loaded"
    );
    Ok(Loaded {
        module_dir: Some(dir),
        forced,
    })
}

/// Names of the registered ggml backends.
#[cfg(ggml_dl)]
fn registry_names() -> Vec<String> {
    // SAFETY: registry queries; names are static strings.
    unsafe {
        (0..sys::ggml_backend_reg_count())
            .map(|i| cstr_lossy(sys::ggml_backend_reg_name(sys::ggml_backend_reg_get(i))))
            .collect()
    }
}

/// libllama's system info line: the CPU features ggml was compiled with (`AVX2 = 1 | AVX512 = 0 | ...`)
/// and the backends it found. Tells whether `TETRAEDER_GGML_NATIVE` took effect.
pub fn system_info() -> String {
    init();
    // SAFETY: returns a static NUL-terminated string.
    cstr_lossy(unsafe { sys::llama_print_system_info() })
        .trim()
        .to_string()
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
