//! Loading the backends for the scheduler and reading model facts from the GGUF.

use crate::backend::{Backend, LlamaBackend, MOE_CHUNK, MoeBackend};
use std::path::Path;
use std::sync::Arc;
use tetraeder_engine::{ContextConfig, Engine, Placement};
use tetraeder_ggml_sys::llama::{Model, ModelParams};
use tetraeder_model::{Gguf, LoadOptions, Qwen3Moe};
use tetraeder_telemetry::Telemetry;

/// A backend and the libllama model that tokenizes for it.
pub struct Loaded {
    pub backend: Box<dyn Backend>,
    pub tokenizer: Arc<Model>,
}

/// libllama for any architecture; the tokenizer is the loaded model.
pub fn load_llama(
    path: &Path,
    placement: &Placement,
    cfg: &ContextConfig,
    telemetry: Option<Arc<Telemetry>>,
) -> Result<Loaded, String> {
    let engine = Engine::load(path, placement, cfg, telemetry).map_err(|e| e.to_string())?;
    let tokenizer = Arc::clone(engine.model());
    Ok(Loaded {
        backend: Box::new(LlamaBackend::new(engine, placement.n_gpu_layers != 0)),
        tokenizer,
    })
}

/// The Qwen3 MoE builder; libllama loads only the vocabulary, for tokenizing.
pub fn load_moe(path: &Path, opts: &LoadOptions) -> Result<Loaded, String> {
    load_moe_chunked(path, opts, MOE_CHUNK)
}

/// [`load_moe`] with `chunk` tokens per backend call.
pub fn load_moe_chunked(path: &Path, opts: &LoadOptions, chunk: usize) -> Result<Loaded, String> {
    let tokenizer = Arc::new(
        Model::load(
            path,
            &ModelParams {
                vocab_only: true,
                ..ModelParams::default()
            },
        )
        .map_err(|e| e.to_string())?,
    );
    let on_gpu = matches!(opts.backend, tetraeder_model::BackendChoice::Gpu { .. });
    let model = Qwen3Moe::load(path, opts).map_err(|e| e.to_string())?;
    Ok(Loaded {
        backend: Box::new(MoeBackend::with_chunk(model, on_gpu, chunk)),
        tokenizer,
    })
}

/// `general.architecture` of a GGUF (the first shard of a split one).
pub fn architecture(path: &Path) -> Result<String, String> {
    let g = Gguf::open(path).map_err(|e| e.to_string())?;
    g.str("general.architecture")
        .map(str::to_owned)
        .map_err(|e| e.to_string())
}

/// `tokenizer.chat_template` of a GGUF, if it has one.
pub fn chat_template(path: &Path) -> Option<String> {
    let g = Gguf::open(path).ok()?;
    g.str("tokenizer.chat_template").ok().map(str::to_owned)
}

/// The file name without extension and without a `-00001-of-00003` shard suffix.
pub fn model_id(path: &Path) -> String {
    let stem = path
        .file_stem()
        .map_or_else(|| "model".into(), |s| s.to_string_lossy().into_owned());
    strip_shard_suffix(&stem).to_owned()
}

fn strip_shard_suffix(stem: &str) -> &str {
    let digits = |s: &str| s.len() == 5 && s.bytes().all(|b| b.is_ascii_digit());
    if let Some((rest, total)) = stem.rsplit_once("-of-")
        && digits(total)
        && let Some((name, idx)) = rest.rsplit_once('-')
        && digits(idx)
    {
        return name;
    }
    stem
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_suffix_is_stripped() {
        let id = |p: &str| model_id(Path::new(p));
        assert_eq!(
            id("/m/Qwen3-30B-A3B-Q4_K_M-00001-of-00003.gguf"),
            "Qwen3-30B-A3B-Q4_K_M"
        );
        assert_eq!(id("/m/tiny-qwen3moe-f16.gguf"), "tiny-qwen3moe-f16");
        assert_eq!(id("x.gguf"), "x");
    }
}
