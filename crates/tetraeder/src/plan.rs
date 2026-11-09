//! `tetraeder plan layout`: per-tensor bytes of a GGUF as JSON for the planner (ADR 0040).

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use std::path::{Path, PathBuf};
use tetraeder_model::Gguf;
use tetraeder_planner::{ModelMeta, TensorRec, build_layout};

#[derive(Args)]
pub struct PlanArgs {
    #[command(subcommand)]
    command: PlanCommand,
}

#[derive(Subcommand)]
enum PlanCommand {
    /// Write the model layout JSON of a GGUF (a split model is opened from its first shard).
    Layout {
        gguf: PathBuf,
        /// Output file; stdout when absent.
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

pub fn run(a: &PlanArgs) -> Result<()> {
    match &a.command {
        PlanCommand::Layout { gguf, out } => {
            let json = serde_json::to_string_pretty(&layout(gguf)?)?;
            match out {
                Some(p) => {
                    std::fs::write(p, json).with_context(|| format!("write {}", p.display()))
                }
                None => {
                    println!("{json}");
                    Ok(())
                }
            }
        }
    }
}

pub(crate) fn layout(path: &Path) -> Result<tetraeder_planner::ModelLayout> {
    let g = Gguf::open(path).with_context(|| format!("open {}", path.display()))?;
    let arch = g.str("general.architecture")?.to_string();
    let key = |k: &str| format!("{arch}.{k}");
    let n_embd = g.u64(&key("embedding_length"))? as u32;
    let n_head = g.u64(&key("attention.head_count"))? as u32;
    // Per-layer arrays (hybrid models) do not read as one number; fall back to MHA.
    let n_head_kv = g
        .u64(&key("attention.head_count_kv"))
        .map_or(n_head, |v| v as u32);
    let meta = ModelMeta {
        model: g
            .str("general.name")
            .map(str::to_string)
            .unwrap_or_else(|_| {
                path.file_stem()
                    .map_or_else(String::new, |s| s.to_string_lossy().into_owned())
            }),
        n_layer: g.u64(&key("block_count"))? as u32,
        n_embd,
        n_head,
        n_head_kv,
        head_dim: g.u64_or(
            &key("attention.key_length"),
            u64::from(n_embd / n_head.max(1)),
        )? as u32,
        n_expert: g.u64_or(&key("expert_count"), 0)? as u32,
        n_expert_used: g.u64_or(&key("expert_used_count"), 0)? as u32,
        arch,
    };
    let tensors: Vec<TensorRec> = g
        .tensors
        .iter()
        .map(|t| TensorRec {
            name: t.name.clone(),
            ty: t.ty.name().to_uppercase(),
            dims: t.ne.iter().copied().filter(|&d| d > 1).collect(),
            bytes: t.nbytes,
        })
        .collect();
    build_layout(&meta, &tensors).map_err(anyhow::Error::msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// hardware: none. The tiny MoE fixture (`cargo xtask tiny-moe`): 2 layers, 4 heads, 2 KV heads.
    #[test]
    fn layout_of_the_tiny_moe() {
        let p = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-qwen3moe-f16.gguf");
        let l = layout(&p).unwrap();
        assert_eq!((l.n_layer, l.n_head, l.n_head_kv), (2, 4, 2));
        assert_eq!((l.n_expert, l.n_expert_used), (8, 2));
        assert_eq!(l.layers.len(), 2);
        assert!(
            l.layers
                .iter()
                .all(|x| x.expert_bytes > 0 && x.attn_bytes > 0)
        );
        assert!(l.embed_bytes > 0);
    }
}
