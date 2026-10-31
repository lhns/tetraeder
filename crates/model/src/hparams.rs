//! Qwen3 MoE hyperparameters, tensor index and the placement description from a GGUF header.

use crate::gguf::{Gguf, GgufError, TensorInfo};
use std::path::Path;
use tetraeder_plan::{Format, ModelDesc, UnitSize};

pub const ARCH: &str = "qwen3moe";

/// Hyperparameters of a `qwen3moe` model (keys as llama.cpp writes them).
#[derive(Debug, Clone, PartialEq)]
pub struct Hparams {
    pub name: String,
    pub n_layer: u32,
    pub n_ctx_train: u32,
    pub n_embd: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    /// Per-head width of Q, K and V.
    pub head_dim: u32,
    pub n_rot: u32,
    pub n_expert: u32,
    pub n_expert_used: u32,
    pub n_ff_exp: u32,
    pub n_vocab: u32,
    pub rms_eps: f32,
    pub rope_freq_base: f32,
    /// True when the model has no `output.weight` and reuses the token embedding.
    pub tied_output: bool,
}

/// Tensor names of the architecture.
pub mod names {
    pub const TOK_EMBD: &str = "token_embd.weight";
    pub const OUTPUT_NORM: &str = "output_norm.weight";
    pub const OUTPUT: &str = "output.weight";

    pub fn layer(il: u32, what: &str) -> String {
        format!("blk.{il}.{what}.weight")
    }
}

/// The three stacked expert tensors of a layer, in the order the expert FFN uses them.
pub const EXPERT_TENSORS: [&str; 3] = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];

impl Hparams {
    pub fn from_gguf(g: &Gguf) -> Result<Hparams, GgufError> {
        let arch = g.str("general.architecture")?;
        if arch != ARCH {
            return Err(GgufError::Corrupt(format!(
                "architecture `{arch}`, expected `{ARCH}`"
            )));
        }
        let k = |s: &str| format!("{ARCH}.{s}");
        let n_head = g.u64(&k("attention.head_count"))? as u32;
        let n_embd = g.u64(&k("embedding_length"))? as u32;
        let head_dim =
            g.u64_or(&k("attention.key_length"), (n_embd / n_head.max(1)) as u64)? as u32;
        let value_len = g.u64_or(&k("attention.value_length"), head_dim as u64)? as u32;
        if value_len != head_dim {
            return Err(GgufError::Corrupt("key and value lengths differ".into()));
        }
        if g.get(&k("rope.scaling.type"))
            .is_some_and(|v| !matches!(v, crate::gguf::Value::Str(s) if s == "none"))
        {
            return Err(GgufError::Corrupt("rope scaling is not supported".into()));
        }
        let tok = g.tensor(names::TOK_EMBD)?;
        Ok(Hparams {
            name: g.str("general.name").unwrap_or(ARCH).to_string(),
            n_layer: g.u64(&k("block_count"))? as u32,
            n_ctx_train: g.u64(&k("context_length"))? as u32,
            n_embd,
            n_head,
            n_head_kv: g.u64(&k("attention.head_count_kv"))? as u32,
            head_dim,
            n_rot: g.u64_or(&k("rope.dimension_count"), head_dim as u64)? as u32,
            n_expert: g.u64(&k("expert_count"))? as u32,
            n_expert_used: g.u64(&k("expert_used_count"))? as u32,
            n_ff_exp: g.u64(&k("expert_feed_forward_length"))? as u32,
            n_vocab: tok.ne[1] as u32,
            rms_eps: g.f32(&k("attention.layer_norm_rms_epsilon"))?,
            rope_freq_base: g.f32_or(&k("rope.freq_base"), 10000.0)?,
            tied_output: !g.has_tensor(names::OUTPUT),
        })
    }
}

/// Where one expert's weights live in the file: the gate, up and down slabs of one layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertSlabs {
    /// `(file offset, bytes)` of gate, up and down, in [`EXPERT_TENSORS`] order.
    pub parts: [(u64, u64); 3],
}

impl ExpertSlabs {
    pub fn bytes(&self) -> u64 {
        self.parts.iter().map(|p| p.1).sum()
    }
}

/// A parsed GGUF with the architecture's hyperparameters.
#[derive(Debug)]
pub struct ModelFile {
    pub path: std::path::PathBuf,
    pub gguf: Gguf,
    pub hparams: Hparams,
}

impl ModelFile {
    pub fn open(path: impl AsRef<Path>) -> Result<ModelFile, GgufError> {
        let gguf = Gguf::open(path.as_ref())?;
        let hparams = Hparams::from_gguf(&gguf)?;
        Ok(ModelFile {
            path: path.as_ref().to_path_buf(),
            gguf,
            hparams,
        })
    }

    pub fn layer_tensor(&self, il: u32, what: &str) -> Result<&TensorInfo, GgufError> {
        self.gguf.tensor(&names::layer(il, what))
    }

    /// Byte ranges of expert `e` of layer `il` (what ADR 0031's I/O reads per slot).
    pub fn expert_slabs(&self, il: u32, e: u32) -> Result<ExpertSlabs, GgufError> {
        let mut parts = [(0, 0); 3];
        for (p, name) in parts.iter_mut().zip(EXPERT_TENSORS) {
            let t = self.layer_tensor(il, name)?;
            *p = t.slab(e as u64).ok_or_else(|| {
                GgufError::Corrupt(format!("expert {e} out of range in `{name}`"))
            })?;
        }
        Ok(ExpertSlabs { parts })
    }

    /// Unit sizes for the placement plan (`plan::ModelDesc`). All layers are alike; the sizes come
    /// from layer 0. The source format of a unit is the type of its largest tensor.
    pub fn model_desc(&self) -> Result<ModelDesc, GgufError> {
        let h = &self.hparams;
        let size = |ts: &[&TensorInfo], matmul: bool| -> Result<UnitSize, GgufError> {
            let params: u64 = ts.iter().map(|t| t.n_elements()).sum();
            // Matmul weights cost 2 FLOPs per parameter per token; embedding lookups and norms
            // are negligible.
            let flops: u64 = ts
                .iter()
                .filter(|t| matmul && t.ne[1] > 1)
                .map(|t| t.n_elements())
                .sum();
            Ok(UnitSize {
                params,
                flops_per_token: 2.0 * flops as f64,
                source_format: main_format(ts)?,
            })
        };
        let get = |w: &str| self.layer_tensor(0, w);
        let tok = self.gguf.tensor(names::TOK_EMBD)?;
        let out_norm = self.gguf.tensor(names::OUTPUT_NORM)?;

        let attn = size(
            &[
                get("attn_norm")?,
                get("attn_q")?,
                get("attn_k")?,
                get("attn_v")?,
                get("attn_output")?,
                get("attn_q_norm")?,
                get("attn_k_norm")?,
            ],
            true,
        )?;
        let mlp = size(&[get("ffn_norm")?, get("ffn_gate_inp")?], true)?;
        // One expert: its slab of each stacked tensor. Active FLOPs per token per expert.
        let n_exp = h.n_expert as u64;
        let stacked = [
            get("ffn_gate_exps")?,
            get("ffn_up_exps")?,
            get("ffn_down_exps")?,
        ];
        let expert = UnitSize {
            params: stacked.iter().map(|t| t.n_elements()).sum::<u64>() / n_exp,
            flops_per_token: 2.0
                * (stacked.iter().map(|t| t.n_elements()).sum::<u64>() / n_exp) as f64,
            source_format: main_format(&stacked)?,
        };
        let head = if h.tied_output {
            size(&[out_norm], false)?
        } else {
            size(&[out_norm, self.gguf.tensor(names::OUTPUT)?], true)?
        };
        Ok(ModelDesc {
            name: h.name.clone(),
            layers: h.n_layer,
            heads: h.n_head,
            kv_heads: h.n_head_kv,
            experts: h.n_expert,
            experts_used: h.n_expert_used,
            embed: size(&[tok], false)?,
            attn,
            mlp,
            expert: Some(expert),
            head,
            // K and V, F16.
            kv_bytes_per_token_per_layer: 2 * 2 * (h.n_head_kv * h.head_dim) as u64,
            // F32 hidden state.
            activation_bytes_per_token: 4 * h.n_embd as u64,
        })
    }
}

fn plan_format(t: &TensorInfo) -> Result<Format, GgufError> {
    let name = t.ty.name();
    Format::ALL
        .into_iter()
        .find(|f| format!("{f:?}").eq_ignore_ascii_case(&name))
        .ok_or_else(|| GgufError::Corrupt(format!("tensor `{}` has unplanned type {name}", t.name)))
}

/// The format of the unit's largest tensor (norms and routers stay F32 and are tiny).
fn main_format(ts: &[&TensorInfo]) -> Result<Format, GgufError> {
    let t = ts
        .iter()
        .max_by_key(|t| t.nbytes)
        .expect("a unit has tensors");
    plan_format(t)
}
