//! Shared test helpers. Models are found via `TETRAEDER_MODELS_DIR`, else the nearest
//! `.hf_cache` above this crate. Tests skip (pass) when the model file is missing.
#![allow(dead_code)]

use std::path::PathBuf;
use tetraeder_engine::{ContextConfig, Engine, Placement};

pub const MODEL: &str = "Qwen__Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf";

pub fn model_path() -> Option<PathBuf> {
    let dir = match std::env::var_os("TETRAEDER_MODELS_DIR") {
        Some(d) => PathBuf::from(d),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .map(|a| a.join(".hf_cache"))
            .find(|p| p.is_dir())?,
    };
    let p = dir.join(MODEL);
    p.is_file().then_some(p)
}

/// Returns the model path or prints a skip reason and returns from the calling test.
#[macro_export]
macro_rules! require_model {
    () => {
        match $crate::common::model_path() {
            Some(p) => p,
            None => {
                eprintln!(
                    "skip: {} not found (run `cargo xtask fetch-model`)",
                    $crate::common::MODEL
                );
                return;
            }
        }
    };
}

pub fn cpu_ctx() -> ContextConfig {
    ContextConfig {
        n_ctx: 512,
        threads: Some(2),
        ..ContextConfig::default()
    }
}

pub fn cpu_engine(
    path: &std::path::Path,
    telemetry: Option<std::sync::Arc<tetraeder_telemetry::Telemetry>>,
) -> Engine {
    Engine::load(path, &Placement::cpu(), &cpu_ctx(), telemetry).unwrap()
}
