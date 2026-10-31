//! Quantization and GGUF inspection.

use super::{LlamaError, Result, cstr_lossy, init, path_cstring};
use crate::raw as sys;
use std::ffi::CString;
use std::path::Path;

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
