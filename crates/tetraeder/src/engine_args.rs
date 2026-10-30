//! Placement and context flags shared by `run` and `bench` (ADR 0026).

use clap::Args;
use std::path::PathBuf;
use tetraeder_engine::{BufferOverride, ContextConfig, FlashAttn, LoadMode, Placement, SplitMode};

#[derive(Args, Debug, Clone)]
pub struct ModelArgs {
    /// GGUF model file.
    #[arg(short = 'm', long)]
    pub model: PathBuf,
    /// Layers on GPU; -1 = all, 0 = CPU only.
    #[arg(long = "ngl", alias = "n-gpu-layers", default_value_t = -1, allow_hyphen_values = true)]
    pub ngl: i32,
    /// none, layer, row or tensor.
    #[arg(long, default_value = "layer")]
    pub split_mode: SplitMode,
    /// GPU for the whole model with split mode none.
    #[arg(long, default_value_t = 0)]
    pub main_gpu: i32,
    /// Share of the model per GPU, comma separated (e.g. 3,1).
    #[arg(long, value_delimiter = ',')]
    pub tensor_split: Vec<f32>,
    /// `<regex>=<buffer type>`, e.g. `exps=CPU` (repeatable).
    #[arg(long = "override-tensor", alias = "ot")]
    pub override_tensor: Vec<BufferOverride>,
    /// auto, none, mmap, mlock, mmap-mlock or direct-io.
    #[arg(long, default_value = "auto")]
    pub load_mode: LoadMode,
    /// CPU threads; default all hardware threads.
    #[arg(short = 't', long)]
    pub threads: Option<i32>,
    /// Logical batch (tokens per decode call).
    #[arg(short = 'b', long, default_value_t = 2048)]
    pub batch: u32,
    /// Physical batch.
    #[arg(long, default_value_t = 512)]
    pub ubatch: u32,
    /// auto, on or off.
    #[arg(long, default_value = "auto", value_parser = parse_flash_attn)]
    pub flash_attn: FlashAttn,
}

fn parse_flash_attn(s: &str) -> Result<FlashAttn, String> {
    match s {
        "auto" => Ok(FlashAttn::Auto),
        "on" => Ok(FlashAttn::On),
        "off" => Ok(FlashAttn::Off),
        _ => Err(format!("`{s}`: expected auto, on or off")),
    }
}

impl ModelArgs {
    pub fn placement(&self) -> Placement {
        Placement {
            n_gpu_layers: self.ngl,
            split_mode: self.split_mode,
            main_gpu: self.main_gpu,
            tensor_split: self.tensor_split.clone(),
            overrides: self.override_tensor.clone(),
            load_mode: self.load_mode,
        }
    }

    pub fn context(&self, n_ctx: u32) -> ContextConfig {
        ContextConfig {
            n_ctx,
            n_batch: self.batch,
            n_ubatch: self.ubatch,
            threads: self.threads,
            threads_batch: None,
            flash_attn: self.flash_attn,
        }
    }
}
