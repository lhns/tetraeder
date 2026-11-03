//! The bench spec that measures a fixture's candidates, anchors and baselines (ADR 0040).
//!
//! Three groups, all on the same box in one session so the harness pairs them:
//! - anchors, a fixed set that calibrates the cost model; never shared with candidate runs
//!   even when a configuration coincides,
//! - candidates, exactly the [`Enumeration`],
//! - baselines: llama.cpp's own fit and hand-tuned `-ot` / `-ncmoe` configs.

use crate::candidate::{
    Budget, Candidate, Enumeration, Placement, Workload, build_candidate, thread_values,
};
use crate::layout::ModelLayout;
use crate::lower::{dense_ngl, to_llama_bench_args, to_variant_args};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::Write;

/// Request size of the CPU-only anchors: a 32B dense model at the fixture prompt on a few
/// threads would take tens of minutes.
pub const CPU_ANCHOR_PROMPT: u32 = 256;
pub const CPU_ANCHOR_GEN: u32 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Group {
    Anchor,
    Candidate,
    Baseline,
}

/// One variant of the spec with what the scorer needs to turn rows into request time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CampaignVariant {
    pub id: String,
    pub group: Group,
    /// `cpu` or `cuda` (the bench spec's device).
    pub device: String,
    pub threads: u32,
    /// Engine flags beyond `-t`, as written to the spec.
    pub args: Vec<String>,
    /// The `-p` and `-n` this variant runs with (its own args override the workload's).
    pub prompt: u32,
    pub r#gen: u32,
    /// Candidate id for `cand-*` variants.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env: BTreeMap<String, String>,
    /// Where the weights go, for the scorer's features; absent for CPU-only anchors (all on the
    /// CPU) and `base-fit` (llama.cpp decides). `base-ot-hand` on a MoE moves the experts of the
    /// first layers (`-ncmoe`), recorded as the same count of layers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placement: Option<Placement>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Campaign {
    pub name: String,
    pub model_key: String,
    pub hf: String,
    pub workload: Workload,
    pub threads: u32,
    pub variants: Vec<CampaignVariant>,
    /// (id, a, b), all `info`.
    pub pairs: Vec<(String, String, String)>,
}

fn variant(
    id: &str,
    group: Group,
    device: &str,
    threads: u32,
    args: &[&str],
    w: &Workload,
) -> CampaignVariant {
    CampaignVariant {
        id: id.to_string(),
        group,
        device: device.to_string(),
        threads,
        args: args.iter().map(|s| s.to_string()).collect(),
        prompt: w.prompt,
        r#gen: w.r#gen,
        candidate: None,
        env: BTreeMap::new(),
        placement: None,
    }
}

/// Build the campaign of one fixture.
///
/// Anchors: CPU only (`-ngl 0`, device `cpu`, pp256 tg32) at threads {quota, quota / 2,
/// max(2, quota / 4)}; the smallest GPU placement (MoE: all experts on the CPU) with op offload
/// on and off at ubatch 512; the largest GPU placement at ubatch 512, also with CUDA graphs
/// off (`anchor-gpu-eager`). Baselines: `base-fit` (`-fitt 1024`), `base-ot-all` (MoE), and
/// `base-ot-hand`: MoE `-ncmoe n` with the smallest `n` that fits by the same estimate, dense
/// `-ngl` of the largest layer count that fits.
pub fn build(
    model_key: &str,
    hf: &str,
    layout: &ModelLayout,
    budget: &Budget,
    workload: &Workload,
    en: &Enumeration,
) -> Campaign {
    let q = budget.threads.max(1);
    let mut vs = Vec::new();

    let cpu_w = Workload {
        prompt: CPU_ANCHOR_PROMPT,
        r#gen: CPU_ANCHOR_GEN,
    };
    let mut cpu_threads = vec![q, thread_values(q)[1], (q / 4).max(2)];
    cpu_threads.dedup();
    cpu_threads.sort_unstable_by(|a, b| b.cmp(a));
    cpu_threads.dedup();
    for t in cpu_threads {
        let args = [
            "-ngl", "0", "-fa", "1", "-b", "2048", "-ub", "512", "-p", "256", "-n", "32",
        ];
        vs.push(variant(
            &format!("anchor-cpu-t{t}"),
            Group::Anchor,
            "cpu",
            t,
            &args,
            &cpu_w,
        ));
    }

    let lower = |pl: Placement, op: bool| {
        let c = build_candidate("anchor", layout, pl, (q, 512, op), workload);
        to_variant_args(&c, layout)
    };
    let gpu_variant = |id: &str, pl: Placement, op: bool| {
        let mut v = variant(id, Group::Anchor, "cuda", q, &[], workload);
        v.args = lower(pl, op);
        v.placement = Some(pl);
        v
    };
    if let (Some(&large), Some(&small)) = (en.placements.first(), en.placements.last()) {
        vs.push(gpu_variant("anchor-gpu-small-opo-on", small, true));
        vs.push(gpu_variant("anchor-gpu-small-opo-off", small, false));
        vs.push(gpu_variant("anchor-gpu-large", large, true));
        let mut eager = gpu_variant("anchor-gpu-eager", large, true);
        eager
            .env
            .insert("GGML_CUDA_DISABLE_GRAPHS".into(), "1".into());
        vs.push(eager);
    }

    for c in &en.candidates {
        let mut v = variant(
            &format!("cand-{}", c.id),
            Group::Candidate,
            "cuda",
            c.threads(),
            &[],
            workload,
        );
        v.args = to_variant_args(c, layout);
        v.candidate = Some(c.id.clone());
        v.placement = Some(c.placement);
        vs.push(v);
    }

    vs.push(variant(
        "base-fit",
        Group::Baseline,
        "cuda",
        q,
        &["-fitt", "1024", "-fa", "1"],
        workload,
    ));
    if layout.is_moe() {
        let mut v = variant(
            "base-ot-all",
            Group::Baseline,
            "cuda",
            q,
            &["-ngl", "99", "-ot", "exps=CPU", "-fa", "1"],
            workload,
        );
        v.placement = Some(Placement::Moe {
            gpu_expert_layers: 0,
        });
        vs.push(v);
    }
    if let Some(&large) = en.placements.first() {
        let mut v = variant("base-ot-hand", Group::Baseline, "cuda", q, &[], workload);
        let mut args: Vec<String> = match large {
            Placement::Dense { gpu_layers } => {
                vec![
                    "-ngl".into(),
                    dense_ngl(gpu_layers, layout.n_layer).to_string(),
                ]
            }
            Placement::Moe { gpu_expert_layers } => {
                let mut a = vec!["-ngl".to_string(), "99".into()];
                let n = layout.n_layer.saturating_sub(gpu_expert_layers);
                if n > 0 {
                    a.extend(["-ncmoe".to_string(), n.to_string()]);
                }
                a
            }
        };
        args.extend(["-fa".to_string(), "1".to_string()]);
        v.args = args;
        v.placement = Some(large);
        vs.push(v);
    }

    let pairs = pair_up(&vs);
    Campaign {
        name: format!("planner-{model_key}"),
        model_key: model_key.to_string(),
        hf: hf.to_string(),
        workload: *workload,
        threads: q,
        variants: vs,
        pairs,
    }
}

/// Disjoint consecutive pairs inside each (group, device); an odd last variant pairs with the
/// first of its set; a set of one has no pair.
fn pair_up(vs: &[CampaignVariant]) -> Vec<(String, String, String)> {
    let mut sets: Vec<((Group, &str), Vec<&str>)> = Vec::new();
    for v in vs {
        let key = (v.group, v.device.as_str());
        match sets.iter_mut().find(|(k, _)| *k == key) {
            Some((_, ids)) => ids.push(&v.id),
            None => sets.push((key, vec![&v.id])),
        }
    }
    let mut out = Vec::new();
    for (_, ids) in sets {
        if ids.len() < 2 {
            continue;
        }
        let mut chunks: Vec<(&str, &str)> = ids
            .chunks(2)
            .filter(|c| c.len() == 2)
            .map(|c| (c[0], c[1]))
            .collect();
        if ids.len() % 2 == 1 {
            chunks.push((ids[ids.len() - 1], ids[0]));
        }
        for (a, b) in chunks {
            out.push((format!("info-{a}-vs-{b}"), a.to_string(), b.to_string()));
        }
    }
    out
}

/// A TOML string: literal when possible (regexes keep their backslashes).
fn s(x: &str) -> String {
    if x.contains('\'') || x.contains('\n') {
        format!("{x:?}")
    } else {
        format!("'{x}'")
    }
}

fn list(xs: &[String]) -> String {
    let v: Vec<String> = xs.iter().map(|x| s(x)).collect();
    format!("[{}]", v.join(", "))
}

impl Campaign {
    /// The bench spec in `tetraeder-bench` format.
    pub fn to_toml(&self) -> String {
        let mut o = String::new();
        let w = &self.workload;
        let _ = write!(
            o,
            "# Generated by `tetraeder-plan candidates` (ADR 0040). Do not edit.\n\
             [spec]\nname = {name}\ndescription = {desc}\nrepeats = 2\nwarmup = 0\n\
             page_cache = \"warm\"\ntimeout_s = 900\n\n\
             [workload]\nbatch = 2048\nubatch = 512\nprompt_len = {p}\ngen_len = {g}\nthreads = {t}\n\n\
             [model.{key}]\nhf = {hf}\n\n\
             [[engine]]\nid = \"llama-bench\"\nkind = \"llama-bench\"\nbin_env = \"LLAMA_BENCH\"\n\
             bin = \"/opt/llama.cpp/bin/llama-bench\"\n",
            name = s(&self.name),
            desc = s(&format!("Planner measurement set for {}", self.model_key)),
            p = w.prompt,
            g = w.r#gen,
            t = self.threads,
            key = self.model_key,
            hf = s(&self.hf),
        );
        for v in &self.variants {
            let _ = write!(
                o,
                "\n[[variant]]\nid = {}\nengine = \"llama-bench\"\nmodel = {}\ndevice = {}\nthreads = {}\nargs = {}\n",
                s(&v.id),
                s(&self.model_key),
                s(&v.device),
                v.threads,
                list(&v.args)
            );
            if !v.env.is_empty() {
                let kv: Vec<String> = v
                    .env
                    .iter()
                    .map(|(k, x)| format!("{k} = {}", s(x)))
                    .collect();
                let _ = writeln!(o, "env = {{ {} }}", kv.join(", "));
            }
        }
        for (id, a, b) in &self.pairs {
            let _ = write!(
                o,
                "\n[[pair]]\nid = {}\na = {}\nb = {}\nrole = \"info\"\n",
                s(id),
                s(a),
                s(b)
            );
        }
        o.push_str(
            "\n# Info pairs ignore the bands, but every device needs one.\n\
             [[noise_band]]\nclass = \"cpu\"\ndevice = \"cpu\"\nband = 0.1\n\n\
             [[noise_band]]\nclass = \"cuda-default\"\ndevice = \"cuda\"\nband = 0.08\n",
        );
        o
    }
}

/// One candidate with everything the scorer reads.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateEntry {
    pub candidate: Candidate,
    /// The full llama-bench argument list (with `-t`), apart from model and test sizes.
    pub lowered_args: Vec<String>,
    /// The spec variant that measures it (its `prompt` and `gen` are what it runs with).
    pub variant: CampaignVariant,
}

/// `candidates.json`: written next to the results; the measured set of a fixture.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidatesFile {
    pub model: String,
    pub budget: Budget,
    pub workload: Workload,
    pub candidates: Vec<CandidateEntry>,
    pub anchors: Vec<CampaignVariant>,
    pub baselines: Vec<CampaignVariant>,
    pub dropped: Vec<(String, String)>,
}

impl CandidatesFile {
    pub fn new(c: &Campaign, layout: &ModelLayout, budget: &Budget, en: &Enumeration) -> Self {
        let by_group = |g: Group| -> Vec<CampaignVariant> {
            c.variants
                .iter()
                .filter(|v| v.group == g)
                .cloned()
                .collect()
        };
        let candidates = en
            .candidates
            .iter()
            .map(|cand| CandidateEntry {
                candidate: cand.clone(),
                lowered_args: to_llama_bench_args(cand, layout),
                variant: c
                    .variants
                    .iter()
                    .find(|v| v.candidate.as_deref() == Some(&cand.id))
                    .cloned()
                    .expect("every candidate has a variant"),
            })
            .collect();
        Self {
            model: layout.model.clone(),
            budget: *budget,
            workload: c.workload,
            candidates,
            anchors: by_group(Group::Anchor),
            baselines: by_group(Group::Baseline),
            dropped: en.dropped.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate::enumerate;
    use crate::layout::fixtures;
    use tetraeder_bench::spec::{Device, Role, Spec};

    const BUDGET: Budget = Budget {
        gpu_bytes: 14_000_000_000,
        ram_bytes: 60_000_000_000,
        threads: 15,
    };
    const WORK: Workload = Workload {
        prompt: 2048,
        r#gen: 128,
    };

    fn make(l: &ModelLayout) -> (Campaign, Enumeration) {
        let en = enumerate(&BUDGET, l, &WORK);
        let c = build("m", "org/name:file.gguf", l, &BUDGET, &WORK, &en);
        (c, en)
    }

    #[test]
    fn spec_parses_with_bench_types() {
        for l in [fixtures::dense(), fixtures::moe()] {
            let (c, en) = make(&l);
            let spec = Spec::parse(&c.to_toml()).unwrap_or_else(|e| panic!("{e:#}"));
            assert_eq!(spec.variant.len(), c.variants.len());
            assert_eq!(spec.spec.repeats, 2);
            assert_eq!(spec.spec.timeout_s, 900);
            assert!(spec.pair.iter().all(|p| p.role == Role::Info));
            assert_eq!(
                spec.variant
                    .iter()
                    .filter(|v| v.id.starts_with("cand-"))
                    .count(),
                en.candidates.len()
            );
            for v in &spec.variant {
                assert!(v.threads.is_some());
            }
            // regexes survive TOML quoting
            if l.is_moe() {
                let v = spec.variant("cand-c00").unwrap();
                assert!(v.args.iter().any(|a| a.starts_with(r"blk\.(")));
            }
        }
    }

    #[test]
    fn anchors_and_baselines_moe() {
        let (c, _) = make(&fixtures::moe());
        let get = |id: &str| {
            c.variants
                .iter()
                .find(|v| v.id == id)
                .unwrap_or_else(|| panic!("{id}"))
        };
        let cpu: Vec<_> = c.variants.iter().filter(|v| v.device == "cpu").collect();
        assert_eq!(
            cpu.iter().map(|v| v.threads).collect::<Vec<_>>(),
            [15, 7, 3]
        );
        for v in &cpu {
            assert!(v.args.windows(2).any(|w| w == ["-ngl", "0"]));
            assert_eq!((v.prompt, v.r#gen), (256, 32));
            assert!(v.args.windows(2).any(|w| w == ["-p", "256"]));
        }
        assert_eq!(
            get("anchor-gpu-small-opo-on").args[..4],
            ["-ngl", "99", "-ot", "exps=CPU"]
        );
        assert!(
            get("anchor-gpu-small-opo-off")
                .args
                .contains(&"-nopo".to_string())
        );
        assert_eq!(get("anchor-gpu-eager").env["GGML_CUDA_DISABLE_GRAPHS"], "1");
        assert_eq!(get("anchor-gpu-eager").args, get("anchor-gpu-large").args);
        assert_eq!(get("base-fit").args, ["-fitt", "1024", "-fa", "1"]);
        assert_eq!(
            get("base-ot-all").args,
            ["-ngl", "99", "-ot", "exps=CPU", "-fa", "1"]
        );
        let hand = get("base-ot-hand");
        assert_eq!(hand.args[..3], ["-ngl", "99", "-ncmoe"]);
        let n: u32 = hand.args[3].parse().unwrap();
        assert_eq!(
            48 - n,
            c.variants
                .iter()
                .find(|v| v.id == "cand-c00")
                .map(|_| 0)
                .unwrap_or(0)
                + {
                    // largest placement
                    let (_, en) = make(&fixtures::moe());
                    en.placements[0].gpu_share()
                }
        );
        assert_eq!(get("cand-c00").candidate.as_deref(), Some("c00"));
    }

    #[test]
    fn dense_baseline_has_no_ot() {
        let (c, _) = make(&fixtures::dense());
        assert!(c.variants.iter().all(|v| v.id != "base-ot-all"));
        let hand = c.variants.iter().find(|v| v.id == "base-ot-hand").unwrap();
        assert_eq!(hand.args[0], "-ngl");
        assert_eq!(hand.args.len(), 4);
    }

    #[test]
    fn pairs_stay_inside_group_and_device() {
        let (c, _) = make(&fixtures::dense());
        let spec = Spec::parse(&c.to_toml()).unwrap();
        for p in &spec.pair {
            let (a, b) = (spec.variant(&p.a).unwrap(), spec.variant(&p.b).unwrap());
            assert_eq!(a.device, b.device);
            assert_eq!(a.id.split('-').next(), b.id.split('-').next());
        }
        // every variant of a set with two or more members is in a pair
        let paired: std::collections::BTreeSet<_> = spec
            .pair
            .iter()
            .flat_map(|p| [p.a.clone(), p.b.clone()])
            .collect();
        for v in &spec.variant {
            let same = spec
                .variant
                .iter()
                .filter(|w| {
                    w.device == v.device && w.id.split('-').next() == v.id.split('-').next()
                })
                .count();
            assert_eq!(paired.contains(&v.id), same >= 2, "{}", v.id);
        }
        assert!(spec.variant.iter().any(|v| v.device == Device::Cpu));
    }

    #[test]
    fn candidates_file_has_run_sizes() {
        let l = fixtures::dense();
        let (c, en) = make(&l);
        let f = CandidatesFile::new(&c, &l, &BUDGET, &en);
        assert_eq!(f.candidates.len(), en.candidates.len());
        assert!(f.anchors.iter().all(|a| a.prompt > 0 && a.r#gen > 0));
        assert!(f.anchors.iter().any(|a| a.prompt == 256));
        assert!(
            f.candidates
                .iter()
                .all(|e| e.variant.prompt == 2048 && e.variant.r#gen == 128)
        );
        assert!(f.candidates[0].lowered_args.contains(&"-t".to_string()));
        let back: CandidatesFile =
            serde_json::from_str(&serde_json::to_string(&f).unwrap()).unwrap();
        assert_eq!(back, f);
    }
}
