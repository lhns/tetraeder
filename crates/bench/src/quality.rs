//! Quality next to speed: a gate whose candidate is a Tetraeder build must also produce the reference output.
//!
//! The check is ADR 0026's parity test (greedy text of `llama-simple` is the reference, see `cargo xtask golden`),
//! run on the box with the bench models and the backend of each gate pair, for a few short prompts. `parity`
//! writes `parity.json`; `run` attaches the matching check to each such gate pair as `quality`. A gate with a
//! failed or missing check reports `error`: a fast engine that gets different outputs has not passed.

use crate::engine::{self, resolve_bin, run_capture};
use crate::env;
use crate::run::{PairResult, Status};
use crate::spec::{self, Device, EngineKind, Role, Spec};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Duration;

pub const PARITY_SCHEMA: u32 = 1;
pub const PARITY_FILE: &str = "parity.json";
pub const CHECK: &str = "greedy-golden";
/// Short on purpose: the check must stay cheap (model load dominates). Same prompts as `cargo xtask golden`.
pub const PROMPTS: [&str; 2] = ["The capital of France is", "def fibonacci(n):"];
pub const N_PREDICT: u32 = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QualityStatus {
    Ok,
    /// Outputs differ from the reference.
    Failed,
    /// The check could not run (missing binary or model, crash).
    Error,
    /// No parity result for this model and device.
    Missing,
}

/// The quality verdict attached to a gate pair.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Quality {
    pub check: String,
    pub status: QualityStatus,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParityCheck {
    /// Model key of the spec.
    pub model: String,
    pub device: Device,
    #[serde(flatten)]
    pub quality: Quality,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParityFile {
    pub schema: u32,
    pub spec: String,
    pub checks: Vec<ParityCheck>,
}

impl ParityFile {
    pub fn load(path: &Path) -> Result<ParityFile> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
    }

    pub fn all_ok(&self) -> bool {
        self.checks
            .iter()
            .all(|c| c.quality.status == QualityStatus::Ok)
    }
}

/// The (model, device) targets of a spec: those of gate pairs whose candidate is a Tetraeder build.
pub fn targets(spec: &Spec) -> Vec<(String, Device)> {
    let mut out: Vec<(String, Device)> = Vec::new();
    for p in spec.pair.iter().filter(|p| needs_quality(spec, p)) {
        if let Some(v) = spec.variant(&p.b) {
            let t = (v.model.clone(), v.device);
            if !out.contains(&t) {
                out.push(t);
            }
        }
    }
    out
}

/// Gate pairs with a Tetraeder candidate need a quality check; a llama.cpp against llama.cpp gate has nothing to check.
pub fn needs_quality(spec: &Spec, p: &spec::Pair) -> bool {
    p.role == Role::Gate
        && spec
            .variant(&p.b)
            .and_then(|v| spec.engine(&v.engine))
            .is_some_and(|e| e.kind == EngineKind::Tetraeder)
}

/// Why the parity output differs from the reference, or `None` if equal. `llama-simple` prints the prompt and then the
/// generated pieces, each run ends with one newline; `tetraeder run` prints the generated pieces only.
pub fn compare_outputs(prompt: &str, reference: &str, ours: &str) -> Option<String> {
    let norm = |s: &str| {
        let s = s.replace("\r\n", "\n");
        s.strip_suffix('\n').map(str::to_owned).unwrap_or(s)
    };
    let (reference, ours) = (norm(reference), norm(ours));
    let Some(ref_gen) = reference.strip_prefix(prompt) else {
        return Some(format!(
            "reference does not start with the prompt {prompt:?}: {reference:?}"
        ));
    };
    if ref_gen == ours {
        return None;
    }
    let at = ref_gen
        .chars()
        .zip(ours.chars())
        .take_while(|(a, b)| a == b)
        .count();
    Some(format!(
        "prompt {prompt:?}: outputs differ at char {at}: reference {ref_gen:?} vs tetraeder {ours:?}"
    ))
}

/// The quality of a gate pair from the parity file. Missing file or entry is `Missing`.
pub fn lookup(file: Option<&ParityFile>, model: &str, device: Device) -> Quality {
    let found = file.and_then(|f| {
        f.checks
            .iter()
            .find(|c| c.model == model && c.device == device)
    });
    match found {
        Some(c) => c.quality.clone(),
        None => Quality {
            check: CHECK.into(),
            status: QualityStatus::Missing,
            detail: if file.is_some() {
                format!("no parity result for model `{model}` on {device:?}")
            } else {
                format!("no {PARITY_FILE} (run `tetraeder-bench parity`)")
            },
        },
    }
}

/// Attach the quality of a gate pair and turn a failed or missing one into an error, even if the speed gate
/// passed or failed. Pairs that were skipped or errored already are left alone apart from the record.
pub fn attach(res: &mut PairResult, q: Quality) {
    if q.status != QualityStatus::Ok && matches!(res.status, Status::Ok | Status::Regression) {
        let note = format!(
            "quality {}: {}",
            format!("{:?}", q.status).to_lowercase(),
            q.detail
        );
        res.reason = if res.reason.is_empty() {
            note
        } else {
            format!("{}; {note}", res.reason)
        };
        res.status = Status::Error;
    }
    res.quality = Some(q);
}

#[derive(Debug, Clone, Default)]
pub struct ParityOptions {
    /// Reference binary; default `$LLAMA_SIMPLE`, else /opt/llama.cpp/bin/llama-simple.
    pub llama_simple: Option<PathBuf>,
    /// Overrides the spec's Tetraeder engine binary.
    pub tetraeder: Option<PathBuf>,
    pub models_dir: PathBuf,
    pub quiet: bool,
}

fn check_one(
    spec: &Spec,
    model_key: &str,
    device: Device,
    opts: &ParityOptions,
) -> Result<(), (QualityStatus, String)> {
    let err = |e: String| (QualityStatus::Error, e);
    let model = spec
        .model
        .get(model_key)
        .ok_or_else(|| err(format!("unknown model `{model_key}`")))?
        .path(&opts.models_dir)
        .map_err(|e| err(format!("{e:#}")))?;
    if !model.is_file() {
        return Err(err(format!("model file {} not found", model.display())));
    }
    let simple_cfg = opts
        .llama_simple
        .as_ref()
        .map(|p| p.display().to_string())
        .or_else(|| std::env::var("LLAMA_SIMPLE").ok().filter(|s| !s.is_empty()))
        .unwrap_or_else(|| "/opt/llama.cpp/bin/llama-simple".into());
    let simple = resolve_bin(&simple_cfg)
        .ok_or_else(|| err(format!("reference binary `{simple_cfg}` not found")))?;
    let engine = spec
        .engine
        .iter()
        .find(|e| e.kind == EngineKind::Tetraeder)
        .ok_or_else(|| err("spec has no tetraeder engine".into()))?;
    let tet_cfg = opts.tetraeder.as_ref().map_or_else(
        || {
            engine
                .bin_env
                .as_ref()
                .and_then(|k| std::env::var(k).ok())
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| engine.bin.clone())
        },
        |p| p.display().to_string(),
    );
    let tet =
        resolve_bin(&tet_cfg).ok_or_else(|| err(format!("engine binary `{tet_cfg}` not found")))?;
    let ngl = if device == Device::Cuda { "99" } else { "0" };
    let envs = engine::device_env(device);
    let timeout = Duration::from_secs(spec.spec.timeout_s);
    let s = |x: &str| x.to_string();
    let n = N_PREDICT.to_string();
    for prompt in PROMPTS {
        let ref_args = vec![
            s("-m"),
            model.display().to_string(),
            s("-n"),
            n.clone(),
            s("-ngl"),
            s(ngl),
            s(prompt),
        ];
        let our_args = vec![
            s("run"),
            s("--model"),
            model.display().to_string(),
            s("--prompt"),
            s(prompt),
            s("-n"),
            n.clone(),
            s("--ctx"),
            s("512"),
            s("--ngl"),
            s(ngl),
        ];
        let reference = run_capture(&simple, &ref_args, &envs, timeout)
            .map_err(|e| err(format!("reference: {e:#}")))?;
        let ours = run_capture(&tet, &our_args, &envs, timeout)
            .map_err(|e| err(format!("tetraeder: {e:#}")))?;
        if let Some(d) = compare_outputs(prompt, &reference.stdout, &ours.stdout) {
            return Err((QualityStatus::Failed, d));
        }
    }
    Ok(())
}

/// Run the parity check for every target of the spec that this host can run.
pub fn run_parity(spec: &Spec, opts: &ParityOptions) -> ParityFile {
    let has_gpu = !env::capture().gpus.is_empty();
    let mut checks = Vec::new();
    for (model, device) in targets(spec) {
        if device == Device::Cuda && !has_gpu {
            continue; // the gate pair is skipped on this host as well
        }
        if !opts.quiet {
            eprintln!("parity {model} on {device:?}");
        }
        let detail_ok = format!(
            "{} prompts x {N_PREDICT} greedy tokens equal",
            PROMPTS.len()
        );
        let (status, detail) = match check_one(spec, &model, device, opts) {
            Ok(()) => (QualityStatus::Ok, detail_ok),
            Err((s, d)) => (s, d),
        };
        checks.push(ParityCheck {
            model,
            device,
            quality: Quality {
                check: CHECK.into(),
                status,
                detail,
            },
        });
    }
    ParityFile {
        schema: PARITY_SCHEMA,
        spec: spec.spec.name.clone(),
        checks,
    }
}

pub fn write_parity(file: &ParityFile, out_dir: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(out_dir).with_context(|| format!("create {}", out_dir.display()))?;
    let path = out_dir.join(PARITY_FILE);
    std::fs::write(&path, serde_json::to_string_pretty(file)? + "\n")
        .with_context(|| format!("write {}", path.display()))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_outputs_pass_and_differences_are_located() {
        let p = "The capital of France is";
        assert_eq!(
            compare_outputs(p, "The capital of France is Paris.\n", " Paris.\n"),
            None
        );
        assert_eq!(
            compare_outputs(p, "The capital of France is Paris.\r\n", " Paris.\n"),
            None
        );
        let d = compare_outputs(p, "The capital of France is Paris.\n", " Parma.\n").unwrap();
        assert!(d.contains("differ at char 4"), "{d}");
        assert!(compare_outputs(p, "garbage\n", " x\n").is_some());
    }

    #[test]
    fn lookup_reports_missing_file_and_entry() {
        assert_eq!(
            lookup(None, "dense", Device::Cuda).status,
            QualityStatus::Missing
        );
        let f = ParityFile {
            schema: 1,
            spec: "s".into(),
            checks: vec![ParityCheck {
                model: "dense".into(),
                device: Device::Cuda,
                quality: Quality {
                    check: CHECK.into(),
                    status: QualityStatus::Ok,
                    detail: "d".into(),
                },
            }],
        };
        assert_eq!(
            lookup(Some(&f), "dense", Device::Cuda).status,
            QualityStatus::Ok
        );
        assert_eq!(
            lookup(Some(&f), "dense", Device::Cpu).status,
            QualityStatus::Missing
        );
        assert_eq!(
            lookup(Some(&f), "moe", Device::Cuda).status,
            QualityStatus::Missing
        );
    }
}
