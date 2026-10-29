//! Raw bindings to the pinned, patched ggml and libllama (built from `vendor/llama.cpp`).
//!
//! This crate and `memory` are the only places where `unsafe` is allowed.
#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case)]
#![allow(clippy::all, clippy::pedantic, unsafe_op_in_unsafe_fn, dead_code)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
