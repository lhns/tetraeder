//! Typed engine configuration: placement (where weights live), context sizing, sampling.

use crate::EngineError;
use tetraeder_ggml_sys::ggml::Type;
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

/// Tensors that `--cpu-moe` and `--n-cpu-moe` keep on the CPU: the expert FFN weights, by the
/// regex of llama.cpp's `LLM_FFN_EXPS_REGEX` (common/common.h).
pub const FFN_EXPS_REGEX: &str = r"\.ffn_(up|down|gate|gate_up)_(ch|)exps";

/// The overrides llama-server adds for `--cpu-moe` (all layers) and `--n-cpu-moe N` (the first
/// `N`), in its order: the first flag's pattern first. A layer beyond the model's matches nothing.
pub fn cpu_moe_overrides(all: bool, first_layers: Option<usize>) -> Vec<BufferOverride> {
    let cpu = |pattern: String| BufferOverride {
        pattern,
        buffer: "CPU".into(),
    };
    let mut out = Vec::new();
    if all {
        out.push(cpu(FFN_EXPS_REGEX.into()));
    }
    for i in 0..first_layers.unwrap_or(0) {
        out.push(cpu(format!(r"blk\.{i}{FFN_EXPS_REGEX}")));
    }
    out
}

/// The KV cache types `-ctk` and `-ctv` accept in llama-server (`kv_cache_types`, common/arg.cpp),
/// in its order.
pub const KV_CACHE_TYPES: [&str; 10] = [
    "f32", "f16", "bf16", "f8_e4m3", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1",
];

/// A KV cache type by llama-server's names (exact, lower case).
pub fn parse_kv_cache_type(name: &str) -> Result<Type, String> {
    KV_CACHE_TYPES
        .contains(&name)
        .then(|| Type::from_name(name))
        .flatten()
        .ok_or_else(|| {
            format!(
                "unsupported cache type `{name}`; allowed: {}",
                KV_CACHE_TYPES.join(", ")
            )
        })
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
    /// CPU extra buffer types (weight repacking, AMX), libllama's default true.
    pub extra_bufts: bool,
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
            extra_bufts: d.extra_bufts,
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
            extra_bufts: self.extra_bufts,
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
    /// libllama offloads large host-side ops to the device (default on, like libllama).
    pub op_offload: bool,
    /// Sequences the context holds at once (default 1).
    pub n_seq_max: u32,
    /// One KV pool shared by all sequences, `n_ctx` positions in total (default true). With
    /// false, libllama gives each sequence `n_ctx / n_seq_max`.
    pub kv_unified: bool,
    /// Keep the KV cache and the attention on the GPU (libllama's `offload_kqv`, default true);
    /// false is llama-server's `--no-kv-offload`.
    pub offload_kqv: bool,
    /// K and V cache types (default F16). Quantized ones need flash attention (V) and a head
    /// size that is a multiple of the block size; libllama checks that when it builds the context.
    pub type_k: Type,
    pub type_v: Type,
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
            op_offload: d.op_offload,
            n_seq_max: 1,
            kv_unified: true,
            offload_kqv: d.offload_kqv,
            type_k: d.type_k,
            type_v: d.type_v,
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

    // hardware: none
    #[test]
    fn cpu_moe_overrides_are_llama_servers_patterns() {
        assert!(cpu_moe_overrides(false, None).is_empty());
        let all = cpu_moe_overrides(true, None);
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].pattern, FFN_EXPS_REGEX);
        assert_eq!(all[0].buffer, "CPU");
        let two = cpu_moe_overrides(false, Some(2));
        assert_eq!(
            two[0].pattern,
            r"blk\.0\.ffn_(up|down|gate|gate_up)_(ch|)exps"
        );
        assert_eq!(
            two[1].pattern,
            r"blk\.1\.ffn_(up|down|gate|gate_up)_(ch|)exps"
        );
        assert_eq!(cpu_moe_overrides(true, Some(3)).len(), 4);
    }

    // hardware: none
    #[test]
    fn kv_cache_types_are_llama_servers_names() {
        for n in KV_CACHE_TYPES {
            assert_eq!(parse_kv_cache_type(n).unwrap().name(), n, "{n}");
        }
        // ggml knows more types and matches case-insensitively; llama-server does not
        for bad in ["q4_K", "Q8_0", "q2_k", "", "f64"] {
            let e = parse_kv_cache_type(bad).unwrap_err();
            assert!(e.contains("allowed: f32, f16, bf16"), "{bad}: {e}");
        }
    }

    #[test]
    fn mode_strings_parse() {
        assert_eq!("row".parse::<SplitMode>().unwrap(), SplitMode::Row);
        assert!("x".parse::<LoadMode>().is_err());
    }
}
