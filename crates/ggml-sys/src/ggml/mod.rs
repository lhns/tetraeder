//! Safe surface over raw ggml for own graph builders (ADR 0030).
//!
//! Raw pointers stay in this module. Tensor handles carry the lifetime of the [`Context`] that
//! created them (weights from a longer-lived context coerce to the shorter lifetime of a graph
//! context). ggml aborts the process on shape mismatches, as in C: builders must pass valid shapes.
//!
//! Ownership: a [`Context`] owns the backend buffer of its tensors; a [`Buffer`] and a
//! [`Sched`] keep their [`Backend`]s alive.

use std::ffi::{CStr, c_char};

mod backend;
mod context;
mod sched;

pub use backend::{Backend, Buffer};
pub use context::{Context, Graph, OwnedTensor, RopeParams, Tensor};
pub use sched::Sched;

use crate::raw as sys;

#[derive(Debug, thiserror::Error)]
pub enum GgmlError {
    #[error("backend `{0}` is not available")]
    NoBackend(String),
    #[error("cannot allocate {what} ({bytes} bytes)")]
    Alloc { what: &'static str, bytes: usize },
    #[error("graph compute failed with status {0}")]
    Compute(i32),
    #[error("invalid argument: {0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, GgmlError>;

/// A ggml tensor storage type. The id is ggml's `enum ggml_type`, which is also the type id
/// stored in GGUF files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Type(u32);

// bindgen types C enums as i32 on MSVC and u32 elsewhere, so the casts are needed on Windows only.
#[allow(clippy::unnecessary_cast)]
impl Type {
    pub const F32: Type = Type(sys::GGML_TYPE_F32 as u32);
    pub const F16: Type = Type(sys::GGML_TYPE_F16 as u32);
    pub const I32: Type = Type(sys::GGML_TYPE_I32 as u32);
    pub const BF16: Type = Type(sys::GGML_TYPE_BF16 as u32);

    /// The type with this ggml id, if ggml knows it as a storage type.
    pub fn from_id(id: u32) -> Option<Type> {
        crate::llama::init();
        if id >= sys::GGML_TYPE_COUNT as u32 {
            return None;
        }
        // SAFETY: the id is below GGML_TYPE_COUNT; ggml returns a static name or null.
        let name = unsafe { sys::ggml_type_name(id as _) };
        // Removed types stay in the enum with a zero block size.
        // SAFETY: as above.
        let blck = unsafe { sys::ggml_blck_size(id as _) };
        (!name.is_null() && blck > 0).then_some(Type(id))
    }

    pub fn id(self) -> u32 {
        self.0
    }

    /// ggml's lower-case name, for example `f16` or `q4_K`.
    pub fn name(self) -> String {
        // SAFETY: valid ids (constructor invariant) have a static NUL-terminated name.
        cstr(unsafe { sys::ggml_type_name(self.0 as _) })
    }

    /// Elements per quantization block.
    pub fn block_size(self) -> i64 {
        // SAFETY: valid id.
        unsafe { sys::ggml_blck_size(self.0 as _) }
    }

    /// Bytes of one row of `ne` elements (`ne` must be a multiple of the block size).
    pub fn row_size(self, ne: i64) -> usize {
        // SAFETY: valid id; pure arithmetic.
        unsafe { sys::ggml_row_size(self.0 as _, ne) }
    }

    pub(crate) fn raw(self) -> sys::ggml_type {
        self.0 as _
    }
}

pub(crate) fn cstr(p: *const c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    // SAFETY: ggml returns NUL-terminated static or object-owned strings.
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::unnecessary_cast)] // see `impl Type`
    fn types() {
        assert_eq!(Type::F16.name(), "f16");
        assert_eq!(Type::F32.row_size(10), 40);
        assert_eq!(Type::F16.row_size(10), 20);
        assert_eq!(Type::from_id(Type::I32.id()), Some(Type::I32));
        assert_eq!(Type::from_id(100_000), None);
        let q8 = Type::from_id(sys::GGML_TYPE_Q8_0 as u32).unwrap();
        assert_eq!((q8.block_size(), q8.row_size(64)), (32, 34 * 2));
    }
}
