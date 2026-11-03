//! `tetraeder plan layout`: per-tensor bytes of a GGUF as JSON for the planner (ADR 0040).

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use std::path::{Path, PathBuf};
use tetraeder_model::Gguf;
use tetraeder_model::gguf::Value;
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

fn layout(path: &Path) -> Result<tetraeder_planner::ModelLayout> {
    let g = Gguf::open(path).with_context(|| format!("open {}", path.display()))?;
    let arch = g.str("general.architecture")?.to_string();
    let key = |k: &str| format!("{arch}.{k}");
    let n_embd = g.u64(&key("embedding_length"))? as u32;
    let n_head = g.u64(&key("attention.head_count"))? as u32;
    // Per-layer arrays (hybrid models) are skipped by the reader; fall back to MHA.
    let n_head_kv = match g.get(&key("attention.head_count_kv")) {
        Some(Value::U64(v)) => *v as u32,
        _ => n_head,
    };
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
