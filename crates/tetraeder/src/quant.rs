//! `tetraeder quant export`: converts a GGUF to other tensor formats through the conversion
//! cache and writes it as one GGUF or as shards split by tensor groups (ADR 0041, 0042).

use anyhow::{Context, bail};
use clap::{Args, Subcommand};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tetraeder_model::Gguf;
use tetraeder_model::convert::{self, ConvertOptions, format_of};
use tetraeder_model::gguf_write::{write_gguf, write_split};
use tetraeder_plan::Format;

#[derive(Subcommand)]
pub enum QuantCommand {
    /// Convert tensors to target formats and write a GGUF, optionally split by tensor groups.
    Export(ExportArgs),
}

#[derive(Args)]
pub struct ExportArgs {
    /// Source GGUF (the first shard of a split model).
    #[arg(long)]
    model: PathBuf,
    /// JSON file `{"tensor name": "FORMAT"}`.
    #[arg(long)]
    formats: Option<PathBuf>,
    /// Take each tensor's type from this GGUF.
    #[arg(long)]
    formats_from: Option<PathBuf>,
    /// Every 2-D or higher `.weight` tensor except `ffn_gate_inp` gets this format; tensors
    /// already below it keep theirs. `--formats-from` and `--formats` override it per tensor.
    #[arg(long, value_parser = parse_format)]
    all: Option<Format>,
    /// JSON file `[{"id": "name", "tensors": ["tensor name", ...]}, ...]`: shard 1 holds the
    /// metadata and the tensors in no group, shard i + 1 the tensors of group i.
    #[arg(long)]
    groups: Option<PathBuf>,
    /// Output file, or the prefix of `<prefix>-0000k-of-0000N.gguf` with `--groups`.
    #[arg(long)]
    out: PathBuf,
    /// Conversion threads; default all hardware threads.
    #[arg(long)]
    threads: Option<usize>,
    /// Conversion cache directory; default `$TETRAEDER_CACHE_DIR`, else `.tetraeder-cache` next
    /// to the model.
    #[arg(long)]
    cache: Option<PathBuf>,
}

fn parse_format(s: &str) -> Result<Format, String> {
    serde_json::from_value(serde_json::Value::String(s.to_uppercase()))
        .map_err(|_| format!("unknown format `{s}`"))
}

pub fn run(cmd: &QuantCommand) -> anyhow::Result<()> {
    match cmd {
        QuantCommand::Export(a) => export(a),
    }
}

fn read_json(path: &Path) -> anyhow::Result<serde_json::Value> {
    let s = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&s).with_context(|| format!("parsing {}", path.display()))
}

fn formats(a: &ExportArgs, g: &Gguf) -> anyhow::Result<BTreeMap<String, Format>> {
    let mut map = BTreeMap::new();
    if let Some(all) = a.all {
        for t in &g.tensors {
            let two_d = t.ne[1] * t.ne[2] * t.ne[3] > 1;
            if !two_d || !t.name.ends_with(".weight") || t.name.contains("ffn_gate_inp") {
                continue;
            }
            // Never upward: a tensor already below the target keeps its format.
            if format_of(t.ty).is_some_and(|src| src.at_least(all)) {
                map.insert(t.name.clone(), all);
            }
        }
    }
    if let Some(p) = &a.formats_from {
        let other = Gguf::open(p).with_context(|| format!("opening {}", p.display()))?;
        for t in &other.tensors {
            if t.ne[1] * t.ne[2] * t.ne[3] == 1 || t.name.contains("ffn_gate_inp") {
                continue;
            }
            let f = format_of(t.ty).with_context(|| {
                format!("`{}` has type {} in {}", t.name, t.ty.name(), p.display())
            })?;
            map.insert(t.name.clone(), f);
        }
    }
    if let Some(p) = &a.formats {
        let v = read_json(p)?;
        let obj = v.as_object().context("--formats must be a JSON object")?;
        for (name, f) in obj {
            let f: Format =
                serde_json::from_value(f.clone()).with_context(|| format!("format of `{name}`"))?;
            map.insert(name.clone(), f);
        }
    }
    Ok(map)
}

fn groups(p: &Path) -> anyhow::Result<Vec<Vec<String>>> {
    let v = read_json(p)?;
    let list = v.as_array().context("--groups must be a JSON array")?;
    list.iter()
        .map(|g| {
            let tensors = g
                .get("tensors")
                .and_then(|t| t.as_array())
                .context("each group needs a `tensors` array")?;
            tensors
                .iter()
                .map(|n| Ok(n.as_str().context("tensor names are strings")?.to_string()))
                .collect()
        })
        .collect()
}

fn export(a: &ExportArgs) -> anyhow::Result<()> {
    let mut g = Gguf::open(&a.model).with_context(|| format!("opening {}", a.model.display()))?;
    let bytes_in: u64 = g.tensors.iter().map(|t| t.nbytes).sum();
    let map = formats(a, &g)?;
    if map.is_empty() && a.groups.is_none() {
        bail!("nothing to do: give --formats, --formats-from or --all (or --groups)");
    }
    let threads = a
        .threads
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from));
    let t0 = Instant::now();
    let stats = convert::convert(
        &mut g,
        &map,
        &ConvertOptions {
            cache: a.cache.clone(),
            threads,
            telemetry: None,
        },
    )?;
    let convert_seconds = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let files = match &a.groups {
        Some(p) => write_split(&g, &a.out, &groups(p)?)?,
        None => write_gguf(&g, &a.out)?,
    };
    let write_seconds = t1.elapsed().as_secs_f64();
    let bytes_out: u64 = files
        .iter()
        .map(|f| std::fs::metadata(f).map(|m| m.len()))
        .sum::<std::io::Result<u64>>()?;
    println!(
        "{}",
        serde_json::json!({
            "model": a.model,
            "out": files,
            "convert_seconds": convert_seconds,
            "write_seconds": write_seconds,
            "bytes_in": bytes_in,
            "bytes_out": bytes_out,
            "threads": threads,
            "tensors_converted": stats.converted,
            "cache_hits": stats.cache_hits,
        })
    );
    Ok(())
}
