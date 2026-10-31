//! The pinned benchmark spec (TOML): engines, models, workload, variants, pairs, noise bands.

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    pub spec: Meta,
    /// Engine pins: name -> commit (or "pending" while the engine is not wired up).
    #[serde(default)]
    pub pins: BTreeMap<String, String>,
    pub workload: Workload,
    #[serde(default)]
    pub model: BTreeMap<String, Model>,
    #[serde(default)]
    pub engine: Vec<Engine>,
    #[serde(default)]
    pub variant: Vec<Variant>,
    #[serde(default)]
    pub pair: Vec<Pair>,
    #[serde(default)]
    pub noise_band: Vec<NoiseBand>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Meta {
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// Interleaved A,B pairs per spec pair.
    pub repeats: usize,
    /// Discarded invocations per variant before the measured repeats.
    #[serde(default)]
    pub warmup: usize,
    #[serde(default)]
    pub page_cache: PageCache,
    /// Per engine invocation; a hung engine is killed and the pair reports an error.
    #[serde(default = "default_timeout")]
    pub timeout_s: u64,
    #[serde(default = "default_resamples")]
    pub bootstrap_resamples: usize,
    #[serde(default = "default_confidence")]
    pub confidence: f64,
    #[serde(default = "default_seed")]
    pub seed: u64,
}

fn default_timeout() -> u64 {
    1800
}
fn default_resamples() -> usize {
    10_000
}
fn default_confidence() -> f64 {
    0.95
}
fn default_seed() -> u64 {
    20240601
}

/// `warm`: model files stay in the page cache (warmup loads them). `cold`: drop caches before every
/// sample (Linux root only; recorded as not applied elsewhere).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PageCache {
    #[default]
    Warm,
    Cold,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Workload {
    pub batch: u32,
    pub ubatch: u32,
    pub prompt_len: u32,
    pub gen_len: u32,
    /// CPU threads (`-t`) for every variant; omitted = `available_parallelism()` (cgroup quota aware).
    /// A variant's own `threads` wins. Both engines always get the same explicit count.
    pub threads: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    /// `org/name:file.gguf`, the `taxameter --models` format. Lives in `<models_dir>/<org>__<name>/<file>`.
    pub hf: String,
    #[serde(default)]
    pub note: String,
}

impl Model {
    pub fn org_name_file(&self) -> Result<(&str, &str)> {
        let bad = || format!("model `{}`: expected org/name:file.gguf", self.hf);
        let (repo, file) = self.hf.split_once(':').with_context(bad)?;
        ensure!(repo.contains('/') && !file.is_empty(), "{}", bad());
        Ok((repo, file))
    }

    pub fn path(&self, models_dir: &Path) -> Result<PathBuf> {
        let (repo, file) = self.org_name_file()?;
        Ok(models_dir.join(repo.replace('/', "__")).join(file))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum EngineKind {
    LlamaBench,
    Tetraeder,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Engine {
    pub id: String,
    pub kind: EngineKind,
    /// Environment variable that overrides `bin`.
    #[serde(default)]
    pub bin_env: Option<String>,
    pub bin: String,
    /// False while the engine cannot be run; its pairs are skipped with a clear status.
    #[serde(default = "yes")]
    pub runnable: bool,
    #[serde(default)]
    pub note: String,
}

fn yes() -> bool {
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Device {
    Cpu,
    Cuda,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Variant {
    pub id: String,
    pub engine: String,
    pub model: String,
    pub device: Device,
    /// CPU threads for this variant; overrides the workload value. Do not put `-t` in `args`.
    #[serde(default)]
    pub threads: Option<u32>,
    /// Extra engine flags, e.g. `["-ngl", "99", "-ot", "exps=CPU"]`. `-p`, `-n` and `-r` here replace the
    /// workload defaults (`-r` defaults to 1: the harness repeats).
    #[serde(default)]
    pub args: Vec<String>,
}

/// True if `args` sets any of the flag spellings.
pub fn has_flag(args: &[String], names: &[&str]) -> bool {
    args.iter().any(|a| names.contains(&a.as_str()))
}

/// The thread count the harness passes to every engine for `v`: the variant's, else the workload's, else
/// `default` (the harness host's `available_parallelism()`).
pub fn threads_for(w: &Workload, v: &Variant, default: u32) -> u32 {
    v.threads.or(w.threads).unwrap_or(default).max(1)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// b is the candidate; a median ratio below `1 - band` fails the run.
    Gate,
    /// Reported only.
    Info,
    /// a and b are the same variant (A/A): measures the noise of the box.
    Noise,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Pair {
    pub id: String,
    /// Baseline variant.
    pub a: String,
    /// Candidate variant (equal to `a` for a noise pair).
    pub b: String,
    pub role: Role,
}

/// Allowed drop of the paired ratio, per device and GPU class. Provisional until measured on the nightly box.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NoiseBand {
    pub class: String,
    pub device: Device,
    /// Case-insensitive substring of the GPU name; empty = default for the device.
    #[serde(default)]
    pub gpu_match: String,
    /// Fraction, e.g. 0.05 = a ratio below 0.95 is a regression.
    pub band: f64,
}

impl Spec {
    pub fn load(path: &Path) -> Result<Spec> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        Self::parse(&text).with_context(|| format!("spec {}", path.display()))
    }

    pub fn parse(text: &str) -> Result<Spec> {
        let spec: Spec = toml::from_str(text)?;
        spec.validate()?;
        Ok(spec)
    }

    pub fn variant(&self, id: &str) -> Option<&Variant> {
        self.variant.iter().find(|v| v.id == id)
    }

    pub fn engine(&self, id: &str) -> Option<&Engine> {
        self.engine.iter().find(|e| e.id == id)
    }

    /// Noise band for a device; for CUDA the first entry whose `gpu_match` is in the GPU name, else the device default.
    pub fn band_for(&self, device: Device, gpu_name: Option<&str>) -> Option<(&str, f64)> {
        let name = gpu_name.unwrap_or("").to_lowercase();
        let specific = self.noise_band.iter().find(|n| {
            n.device == device
                && !n.gpu_match.is_empty()
                && name.contains(&n.gpu_match.to_lowercase())
        });
        let default = self
            .noise_band
            .iter()
            .find(|n| n.device == device && n.gpu_match.is_empty());
        specific.or(default).map(|n| (n.class.as_str(), n.band))
    }

    fn validate(&self) -> Result<()> {
        ensure!(self.spec.repeats >= 1, "repeats must be at least 1");
        ensure!(
            (0.5..1.0).contains(&self.spec.confidence),
            "confidence must be in [0.5, 1)"
        );
        for (what, ids) in [
            (
                "engine",
                self.engine.iter().map(|e| &e.id).collect::<Vec<_>>(),
            ),
            ("variant", self.variant.iter().map(|v| &v.id).collect()),
            ("pair", self.pair.iter().map(|p| &p.id).collect()),
        ] {
            let mut seen = std::collections::BTreeSet::new();
            for id in ids {
                ensure!(seen.insert(id), "duplicate {what} id `{id}`");
            }
        }
        for v in &self.variant {
            ensure!(
                self.engine(&v.engine).is_some(),
                "variant `{}`: unknown engine `{}`",
                v.id,
                v.engine
            );
            let m = self
                .model
                .get(&v.model)
                .with_context(|| format!("variant `{}`: unknown model `{}`", v.id, v.model))?;
            m.org_name_file()?;
            ensure!(
                !has_flag(&v.args, &["-t", "--threads"]),
                "variant `{}`: use `threads = N` instead of -t in args",
                v.id
            );
        }
        for p in &self.pair {
            let (Some(a), Some(b)) = (self.variant(&p.a), self.variant(&p.b)) else {
                bail!("pair `{}`: unknown variant", p.id)
            };
            ensure!(
                a.device == b.device && a.model == b.model,
                "pair `{}`: variants must share device and model",
                p.id
            );
            ensure!(
                (p.role == Role::Noise) == (p.a == p.b),
                "pair `{}`: a noise pair must have a == b, and only a noise pair",
                p.id
            );
        }
        for n in &self.noise_band {
            ensure!(
                (0.0..1.0).contains(&n.band),
                "noise band `{}` out of range",
                n.class
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIN: &str = r#"
[spec]
name = "t"
repeats = 3
[workload]
batch = 512
ubatch = 128
prompt_len = 16
gen_len = 8
[model.m]
hf = "org/name:file.gguf"
[[engine]]
id = "e"
kind = "llama-bench"
bin = "x"
[[variant]]
id = "v"
engine = "e"
model = "m"
device = "cpu"
[[pair]]
id = "p"
a = "v"
b = "v"
role = "noise"
[[noise_band]]
class = "cpu"
device = "cpu"
band = 0.1
[[noise_band]]
class = "4090"
device = "cuda"
gpu_match = "4090"
band = 0.03
[[noise_band]]
class = "cuda"
device = "cuda"
band = 0.08
"#;

    #[test]
    fn parses_and_resolves() {
        let s = Spec::parse(MIN).unwrap();
        assert_eq!(s.spec.warmup, 0);
        assert_eq!(s.spec.page_cache, PageCache::Warm);
        let p = s.model["m"].path(Path::new(".hf_cache")).unwrap();
        assert_eq!(
            p,
            Path::new(".hf_cache").join("org__name").join("file.gguf")
        );
    }

    #[test]
    fn band_lookup_by_gpu_class() {
        let s = Spec::parse(MIN).unwrap();
        assert_eq!(s.band_for(Device::Cpu, None), Some(("cpu", 0.1)));
        assert_eq!(
            s.band_for(Device::Cuda, Some("NVIDIA GeForce RTX 4090")),
            Some(("4090", 0.03))
        );
        assert_eq!(
            s.band_for(Device::Cuda, Some("NVIDIA A100")),
            Some(("cuda", 0.08))
        );
    }

    #[test]
    fn rejects_bad_references() {
        assert!(Spec::parse(&MIN.replace("model = \"m\"", "model = \"zz\"")).is_err());
        assert!(
            Spec::parse(&MIN.replace("hf = \"org/name:file.gguf\"", "hf = \"nocolon\"")).is_err()
        );
        assert!(Spec::parse(&MIN.replace("role = \"noise\"", "role = \"gate\"")).is_err());
        assert!(Spec::parse(&MIN.replace("repeats = 3", "repeats = 0")).is_err());
    }

    #[test]
    fn shipped_specs_are_valid() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/spec");
        let mut n = 0;
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.extension().is_some_and(|x| x == "toml") {
                Spec::load(&p).unwrap();
                n += 1;
            }
        }
        assert!(n >= 2);
    }
}
