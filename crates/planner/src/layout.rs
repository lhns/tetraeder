//! Per-tensor bytes of a model: what memory checks and lowering need (ADR 0040).
//!
//! [`build_layout`] is pure over a tensor list so it is testable without a GGUF; the
//! `tetraeder plan layout` command reads the GGUF and calls it.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use tetraeder_plan::{Format, KvFormat, ModelDesc, UnitSize};

/// One tensor of the file. `ty` is the ggml type name in upper case (`Q4_K`, `F32`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorRec {
    pub name: String,
    pub ty: String,
    pub dims: Vec<u64>,
    pub bytes: u64,
}

/// Model shape from the GGUF metadata. `n_vocab` comes from `token_embd.weight`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelMeta {
    pub model: String,
    pub arch: String,
    pub n_layer: u32,
    pub n_embd: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub head_dim: u32,
    /// 0 for a dense model.
    pub n_expert: u32,
    pub n_expert_used: u32,
}

/// Bytes of one layer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerBytes {
    /// Attention weights and norms (and recurrent-state tensors).
    pub attn_bytes: u64,
    /// Dense FFN; for an MoE layer the router, shared experts and norms.
    pub ffn_bytes: u64,
    /// All stacked routed experts of the layer; 0 for a dense layer.
    pub expert_bytes: u64,
}

impl LayerBytes {
    /// Everything but the routed experts: what stays on the GPU in an MoE placement.
    pub fn dense_bytes(&self) -> u64 {
        self.attn_bytes + self.ffn_bytes
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypeSummary {
    pub count: u64,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelLayout {
    pub model: String,
    pub arch: String,
    pub n_layer: u32,
    pub n_embd: u32,
    pub n_vocab: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub head_dim: u32,
    pub n_expert: u32,
    pub n_expert_used: u32,
    pub layers: Vec<LayerBytes>,
    /// `token_embd.weight`; llama.cpp keeps it on the CPU.
    pub embed_bytes: u64,
    /// Output head with the output norm and other global tensors. A tied output (no
    /// `output.weight`) adds nothing beyond the norm.
    pub output_bytes: u64,
    /// Tensor type name to count and bytes.
    pub tensor_types: BTreeMap<String, TypeSummary>,
}

impl ModelLayout {
    pub fn is_moe(&self) -> bool {
        self.n_expert > 0
    }

    pub fn total_bytes(&self) -> u64 {
        self.embed_bytes
            + self.output_bytes
            + self
                .layers
                .iter()
                .map(|l| l.attn_bytes + l.ffn_bytes + l.expert_bytes)
                .sum::<u64>()
    }

    /// The format of the tensor type with the most bytes; F16 (never an upconvert) when that
    /// type is not a plan format.
    pub fn main_format(&self) -> Format {
        self.tensor_types
            .iter()
            .max_by_key(|(_, s)| s.bytes)
            .and_then(|(name, _)| serde_json::from_str(&format!("\"{name}\"")).ok())
            .unwrap_or(Format::F16)
    }

    /// Unit-level view for the plan validator: every unit of a kind is alike, so layer sizes
    /// are averaged (rounded up) and bytes come out in [`Self::main_format`].
    pub fn to_model_desc(&self) -> ModelDesc {
        let f = self.main_format();
        let n = u64::from(self.n_layer.max(1));
        let avg = |sum: u64, div: u64| sum.div_ceil(div.max(1));
        let size = |bytes: u64| {
            let (elems, block) = f.block();
            UnitSize {
                params: bytes.div_ceil(block) * elems,
                flops_per_token: 2.0 * (bytes as f64 * elems as f64 / block as f64),
                source_format: f,
                roles: Vec::new(),
            }
        };
        let attn = avg(self.layers.iter().map(|l| l.attn_bytes).sum(), n);
        let mlp = avg(self.layers.iter().map(|l| l.ffn_bytes).sum(), n);
        let experts: u64 = self.layers.iter().map(|l| l.expert_bytes).sum();
        ModelDesc {
            name: self.model.clone(),
            layers: self.n_layer,
            heads: self.n_head,
            kv_heads: self.n_head_kv,
            experts: self.n_expert,
            experts_used: self.n_expert_used,
            embed: size(self.embed_bytes),
            attn: size(attn),
            mlp: size(mlp),
            expert: self
                .is_moe()
                .then(|| size(avg(experts, n * u64::from(self.n_expert)))),
            head: size(self.output_bytes),
            kv_bytes_per_token_per_layer: self.kv_bytes_per_token_per_layer(),
            activation_bytes_per_token: u64::from(self.n_embd) * 4,
        }
    }

    /// K and V in f16 for one token of one layer.
    pub fn kv_bytes_per_token_per_layer(&self) -> u64 {
        self.kv_bytes_per_token_per_layer_as(KvFormat::default())
    }

    /// K and V in `kv` for one token of one layer.
    pub fn kv_bytes_per_token_per_layer_as(&self, kv: KvFormat) -> u64 {
        kv.bytes(u64::from(self.n_head_kv) * u64::from(self.head_dim))
    }
}

/// Sort the tensors into embed, per-layer attention, FFN and experts, and output.
///
/// `blk.N.<rest>`: `rest` containing `_exps` is a routed expert tensor; starting with `attn_` or
/// `ssm_` is attention; anything else (FFN, router, shared experts, FFN norm) is FFN.
/// `token_embd*` is the embedding; every other tensor outside a block is the output side.
pub fn build_layout(meta: &ModelMeta, tensors: &[TensorRec]) -> Result<ModelLayout, String> {
    let mut layers = vec![LayerBytes::default(); meta.n_layer as usize];
    let (mut embed, mut output) = (0u64, 0u64);
    let mut n_vocab = 0u64;
    let mut types: BTreeMap<String, TypeSummary> = BTreeMap::new();
    for t in tensors {
        let s = types.entry(t.ty.clone()).or_default();
        s.count += 1;
        s.bytes += t.bytes;
        if let Some(rest) = t.name.strip_prefix("blk.") {
            let (n, rest) = rest
                .split_once('.')
                .ok_or_else(|| format!("bad tensor name {:?}", t.name))?;
            let n: usize = n
                .parse()
                .map_err(|_| format!("bad layer in tensor name {:?}", t.name))?;
            let l = layers
                .get_mut(n)
                .ok_or_else(|| format!("tensor {:?}: layer {n} >= {}", t.name, meta.n_layer))?;
            if rest.contains("_exps") {
                l.expert_bytes += t.bytes;
            } else if rest.starts_with("attn_") || rest.starts_with("ssm_") {
                l.attn_bytes += t.bytes;
            } else {
                l.ffn_bytes += t.bytes;
            }
        } else if t.name.starts_with("token_embd") {
            embed += t.bytes;
            if t.name == "token_embd.weight" {
                n_vocab = t.dims.get(1).copied().unwrap_or(0);
            }
        } else {
            output += t.bytes;
        }
    }
    if n_vocab == 0 {
        return Err("no token_embd.weight with a vocabulary dimension".into());
    }
    if meta.n_expert == 0 && layers.iter().any(|l| l.expert_bytes > 0) {
        return Err("expert tensors in a model with expert_count 0".into());
    }
    Ok(ModelLayout {
        model: meta.model.clone(),
        arch: meta.arch.clone(),
        n_layer: meta.n_layer,
        n_embd: meta.n_embd,
        n_vocab: u32::try_from(n_vocab).map_err(|_| "vocabulary too large".to_string())?,
        n_head: meta.n_head,
        n_head_kv: meta.n_head_kv,
        head_dim: meta.head_dim,
        n_expert: meta.n_expert,
        n_expert_used: meta.n_expert_used,
        layers,
        embed_bytes: embed,
        output_bytes: output,
        tensor_types: types,
    })
}

/// Synthetic layouts shaped like the models of the exit test, shared by the tests (also the
/// integration tests of this crate, hence public).
#[doc(hidden)]
pub mod fixtures {
    use super::*;

    fn rec(name: &str, ty: &str, dims: &[u64]) -> TensorRec {
        let f: Format = serde_json::from_str(&format!("\"{ty}\"")).unwrap();
        TensorRec {
            name: name.into(),
            ty: ty.into(),
            dims: dims.to_vec(),
            bytes: f.bytes(dims.iter().product()),
        }
    }

    /// Qwen3-32B Q4_K shape: 64 layers, 8 kv heads of 128.
    pub fn dense_tensors() -> (ModelMeta, Vec<TensorRec>) {
        let (e, v, ff, q, kv) = (5120u64, 151_936u64, 25_600u64, 8192u64, 1024u64);
        let meta = ModelMeta {
            model: "qwen3-32b".into(),
            arch: "qwen3".into(),
            n_layer: 64,
            n_embd: e as u32,
            n_head: 64,
            n_head_kv: 8,
            head_dim: 128,
            n_expert: 0,
            n_expert_used: 0,
        };
        let mut t = vec![rec("token_embd.weight", "Q4_K", &[e, v])];
        for n in 0..64 {
            let b = |s: &str| format!("blk.{n}.{s}.weight");
            t.push(rec(&b("attn_norm"), "F32", &[e]));
            t.push(rec(&b("attn_q"), "Q4_K", &[e, q]));
            t.push(rec(&b("attn_k"), "Q4_K", &[e, kv]));
            t.push(rec(&b("attn_v"), "Q4_K", &[e, kv]));
            t.push(rec(&b("attn_output"), "Q4_K", &[q, e]));
            t.push(rec(&b("ffn_norm"), "F32", &[e]));
            t.push(rec(&b("ffn_gate"), "Q4_K", &[e, ff]));
            t.push(rec(&b("ffn_up"), "Q4_K", &[e, ff]));
            t.push(rec(&b("ffn_down"), "Q4_K", &[ff, e]));
        }
        t.push(rec("output_norm.weight", "F32", &[e]));
        t.push(rec("output.weight", "Q6_K", &[e, v]));
        (meta, t)
    }

    /// Qwen3-30B-A3B Q4_K shape: 48 layers, 128 experts of 768, top 8, 4 kv heads of 128.
    pub fn moe_tensors() -> (ModelMeta, Vec<TensorRec>) {
        let (e, v, ex, q, kv, ne) = (2048u64, 151_936u64, 768u64, 4096u64, 512u64, 128u64);
        let meta = ModelMeta {
            model: "qwen3-30b-a3b".into(),
            arch: "qwen3moe".into(),
            n_layer: 48,
            n_embd: e as u32,
            n_head: 32,
            n_head_kv: 4,
            head_dim: 128,
            n_expert: ne as u32,
            n_expert_used: 8,
        };
        let mut t = vec![rec("token_embd.weight", "Q4_K", &[e, v])];
        for n in 0..48 {
            let b = |s: &str| format!("blk.{n}.{s}.weight");
            t.push(rec(&b("attn_norm"), "F32", &[e]));
            t.push(rec(&b("attn_q"), "Q4_K", &[e, q]));
            t.push(rec(&b("attn_k"), "Q4_K", &[e, kv]));
            t.push(rec(&b("attn_v"), "Q4_K", &[e, kv]));
            t.push(rec(&b("attn_output"), "Q4_K", &[q, e]));
            t.push(rec(&b("ffn_norm"), "F32", &[e]));
            t.push(rec(&b("ffn_gate_inp"), "F32", &[e, ne]));
            t.push(rec(&b("ffn_gate_exps"), "Q4_K", &[e, ex, ne]));
            t.push(rec(&b("ffn_up_exps"), "Q4_K", &[e, ex, ne]));
            t.push(rec(&b("ffn_down_exps"), "Q4_K", &[ex, e, ne]));
        }
        t.push(rec("output_norm.weight", "F32", &[e]));
        t.push(rec("output.weight", "Q6_K", &[e, v]));
        (meta, t)
    }

    pub fn dense() -> ModelLayout {
        let (m, t) = dense_tensors();
        build_layout(&m, &t).unwrap()
    }

    pub fn moe() -> ModelLayout {
        let (m, t) = moe_tensors();
        build_layout(&m, &t).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::*;
    use super::*;

    #[test]
    fn dense_layout_sums_match_tensors() {
        let (meta, tensors) = dense_tensors();
        let l = build_layout(&meta, &tensors).unwrap();
        assert_eq!((l.n_vocab, l.n_layer, l.is_moe()), (151_936, 64, false));
        assert!(l.layers.iter().all(|x| x.expert_bytes == 0));
        assert_eq!(
            l.total_bytes(),
            tensors.iter().map(|t| t.bytes).sum::<u64>()
        );
        assert_eq!(l.main_format(), Format::Q4_K);
        // attention: q, k, v, o in Q4_K plus an F32 norm
        let attn = Format::Q4_K.bytes(5120 * (8192 + 1024 + 1024 + 8192)) + 5120 * 4;
        assert_eq!(l.layers[7].attn_bytes, attn);
        let out = Format::Q6_K.bytes(5120 * 151_936) + 5120 * 4;
        assert_eq!(l.output_bytes, out);
        assert_eq!(l.tensor_types["Q6_K"].count, 1);
    }

    #[test]
    fn moe_layout_splits_experts_from_router() {
        let l = moe();
        assert!(l.is_moe() && l.n_expert_used == 8);
        let experts = 3 * Format::Q4_K.bytes(2048 * 768 * 128);
        assert_eq!(l.layers[0].expert_bytes, experts);
        // router (F32) and the FFN norm
        assert_eq!(l.layers[0].ffn_bytes, (2048 * 128 + 2048) * 4);
        let total = l.total_bytes() as f64 / 1e9;
        assert!((16.0..20.0).contains(&total), "{total}");
    }

    #[test]
    fn tied_output_adds_only_the_norm() {
        let (meta, mut t) = dense_tensors();
        t.retain(|x| x.name != "output.weight");
        let l = build_layout(&meta, &t).unwrap();
        assert_eq!(l.output_bytes, 5120 * 4);
    }

    #[test]
    fn bad_input_is_an_error() {
        let (meta, mut t) = dense_tensors();
        t.push(TensorRec {
            name: "blk.64.attn_q.weight".into(),
            ty: "Q4_K".into(),
            dims: vec![1],
            bytes: 1,
        });
        assert!(build_layout(&meta, &t).is_err());
        let (meta, mut t) = dense_tensors();
        t.remove(0);
        assert!(build_layout(&meta, &t).is_err());
    }

    #[test]
    fn json_roundtrip_and_model_desc() {
        let l = moe();
        let back: ModelLayout = serde_json::from_str(&serde_json::to_string(&l).unwrap()).unwrap();
        assert_eq!(back, l);
        let d = l.to_model_desc();
        assert_eq!((d.layers, d.experts, d.experts_used), (48, 128, 8));
        assert_eq!(d.kv_bytes_per_token_per_layer, 2 * 4 * 128 * 2);
        assert!(d.expert.is_some());
        assert!(!dense().to_model_desc().is_moe());
    }
}
