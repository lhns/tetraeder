//! Shared test helpers. Models are found via `TETRAEDER_MODELS_DIR`, else the nearest
//! `.hf_cache` above this crate. Tests skip (pass) when the model file is missing, unless
//! `TETRAEDER_REQUIRE_MODELS` is set (CI): then they fail.
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

/// Skip a test for a missing resource: prints `skip: <test>: <reason>` (CI lists these lines), or
/// panics when `TETRAEDER_REQUIRE_MODELS` is set.
pub fn skip_or_fail(reason: &str) {
    let test = std::thread::current().name().unwrap_or("?").to_string();
    assert!(
        std::env::var_os("TETRAEDER_REQUIRE_MODELS").is_none(),
        "{test}: {reason} (TETRAEDER_REQUIRE_MODELS is set)"
    );
    eprintln!("skip: {test}: {reason}");
}

/// Returns the model path or skips (see [`skip_or_fail`]) and returns from the calling test.
#[macro_export]
macro_rules! require_model {
    () => {
        match $crate::common::model_path() {
            Some(p) => p,
            None => {
                $crate::common::skip_or_fail(&format!(
                    "{} not found (run `cargo xtask fetch-model`)",
                    $crate::common::MODEL
                ));
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
