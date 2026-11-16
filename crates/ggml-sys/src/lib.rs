//! Bindings to the pinned, patched ggml and libllama (built from `vendor/llama.cpp`).
//!
//! `raw` holds the generated bindings, `llama` a safe wrapper over libllama, `ggml` a safe
//! surface over raw ggml for own graph builders. This crate and
//! `memory` are the only places where `unsafe` is allowed.

pub mod ggml;

/// True when the patch queue of this build includes patch 0004 (CUDA `mul_mat_id` skips expert id
/// -1, ADR 0070). Expert parallel on a GPU rank needs it.
pub const PATCH_0004: bool = cfg!(tetraeder_patch_0004);
pub use ggml::cpu;
pub mod llama;

#[allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    clippy::all,
    clippy::pedantic,
    unsafe_op_in_unsafe_fn,
    // bindgen's bitfield getters transmute u32 to i32 (rustc 1.95 under -D warnings)
    unnecessary_transmutes,
    dead_code
)]
pub mod raw {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub use raw::*;
