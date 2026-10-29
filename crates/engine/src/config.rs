//! Typed engine configuration: placement (where weights live), context sizing, sampling.

use crate::EngineError;
use tetraeder_ggml_sys::llama::{self, FlashAttn, LoadMode, ModelParams, SplitMode};

/// `<regex>=<buffer type>`: tensors whose name matches the regex (unanchored search) are
/// stored in that buffer type, for example `blk\.0\..*=CPU`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferOverride {
    pub pattern: String,
    pub buffer: String,
}

impl BufferOverride {
    /// Splits at the last `=`, so patterns may contain `=`.
    pub fn parse(s: &str) -> Result<Self, EngineError> {
        let (pattern, buffer) = s.rsplit_once('=').ok_or_else(|| {
            EngineError::Placement(format!("override `{s}`: expected <regex>=<buffer type>"))
        })?;
        if pattern.is_empty() || buffer.is_empty() {
            return Err(EngineError::Placement(format!(
                "override `{s}`: pattern and buffer type must not be empty"
            )));
        }
        Ok(Self {
            pattern: pattern.to_owned(),
            buffer: buffer.to_owned(),
        })
    }
}

impl std::str::FromStr for BufferOverride {
    type Err = EngineError;
    fn from_str(s: &str) -> Result<Self, EngineError> {
        Self::parse(s)
    }
}

/// Where the weights go. `Default` is libllama's defaults.
#[derive(Debug, Clone)]
pub struct Placement {
    /// Layers on GPU; negative means all, 0 keeps everything on the CPU.
    pub n_gpu_layers: i32,
    pub split_mode: SplitMode,
    pub main_gpu: i32,
    pub tensor_split: Vec<f32>,
    pub overrides: Vec<BufferOverride>,
    pub load_mode: LoadMode,
}

impl Default for Placement {
    fn default() -> Self {
        let d = ModelParams::default();
        Self {
            n_gpu_layers: d.n_gpu_layers,
            split_mode: d.split_mode,
            main_gpu: d.main_gpu,
            tensor_split: d.tensor_split,
            overrides: Vec::new(),
            load_mode: d.load_mode,
        }
    }
}

impl Placement {
    /// CPU only: no layers offloaded.
    pub fn cpu() -> Self {
        Self {
            n_gpu_layers: 0,
            ..Self::default()
        }
    }

    pub(crate) fn to_model_params(&self) -> Result<ModelParams, EngineError> {
        let mut buffer_overrides = Vec::with_capacity(self.overrides.len());
        for o in &self.overrides {
            let buft = llama::buffer_type(&o.buffer).ok_or_else(|| {
                EngineError::Placement(format!(
                    "unknown buffer type `{}` in override `{}`; known: {}",
                    o.buffer,
                    o.pattern,
                    llama::buffer_type_names().join(", ")
                ))
            })?;
            buffer_overrides.push((o.pattern.clone(), buft));
        }
        Ok(ModelParams {
            n_gpu_layers: self.n_gpu_layers,
            split_mode: self.split_mode,
            main_gpu: self.main_gpu,
            tensor_split: self.tensor_split.clone(),
            buffer_overrides,
            load_mode: self.load_mode,
            vocab_only: false,
        })
    }
}

/// Context sizing and threads. `Default` is libllama's defaults with the model's context.
#[derive(Debug, Clone)]
pub struct ContextConfig {
    /// 0 = the model's training context, which can make the KV cache very large: set it.
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    /// `None`: all available hardware threads.
    pub threads: Option<i32>,
    /// `None`: same as `threads`.
    pub threads_batch: Option<i32>,
    pub flash_attn: FlashAttn,
}

impl Default for ContextConfig {
    fn default() -> Self {
        let d = llama::ContextParams::default();
        Self {
            n_ctx: 0,
            n_batch: d.n_batch,
            n_ubatch: d.n_ubatch,
            threads: None,
            threads_batch: None,
            flash_attn: d.flash_attn,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sampling {
    Greedy,
    Random {
        seed: u32,
        temperature: f32,
        /// <= 0 disables top-k.
        top_k: i32,
        /// >= 1.0 disables top-p.
        top_p: f32,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn override_splits_at_last_equals() {
        let o = BufferOverride::parse(r"blk\.0\..*=CPU").unwrap();
        assert_eq!(o.pattern, r"blk\.0\..*");
        assert_eq!(o.buffer, "CPU");
        let o: BufferOverride = "a=b=CUDA0".parse().unwrap();
        assert_eq!((o.pattern.as_str(), o.buffer.as_str()), ("a=b", "CUDA0"));
    }

    #[test]
    fn bad_overrides_are_errors() {
        for s in ["", "nobuffer", "=CPU", "x="] {
            assert!(BufferOverride::parse(s).is_err(), "{s:?}");
        }
    }

    #[test]
    fn unknown_buffer_type_is_a_placement_error_listing_names() {
        let p = Placement {
            overrides: vec![BufferOverride::parse("x=NOPE").unwrap()],
            ..Placement::cpu()
        };
        let e = p.to_model_params().unwrap_err().to_string();
        assert!(e.contains("NOPE") && e.contains("CPU"), "{e}");
    }

    #[test]
    fn known_buffer_type_resolves_case_insensitively() {
        let p = Placement {
            overrides: vec![BufferOverride::parse("x=cpu").unwrap()],
            ..Placement::cpu()
        };
        assert_eq!(p.to_model_params().unwrap().buffer_overrides.len(), 1);
    }

    #[test]
    fn mode_strings_parse() {
        assert_eq!("row".parse::<SplitMode>().unwrap(), SplitMode::Row);
        assert!("x".parse::<LoadMode>().is_err());
    }
}
