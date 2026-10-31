//! Bindings to the pinned, patched ggml and libllama (built from `vendor/llama.cpp`).
//!
//! `raw` holds the generated bindings, `llama` a safe wrapper over libllama, `ggml` a safe
//! surface over raw ggml for own graph builders. This crate and
//! `memory` are the only places where `unsafe` is allowed.

pub mod ggml;
pub mod llama;

#[allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    clippy::all,
    clippy::pedantic,
    unsafe_op_in_unsafe_fn,
    dead_code
)]
pub mod raw {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub use raw::*;
