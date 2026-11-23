//! `kv = auto`: the planner picks the KV cache format (ADR 0086).
//!
//! Formats are tried in fidelity order [`AUTO_ORDER`]; the first one with a feasible plan at the
//! requested context wins, and planning within it is unchanged. BF16 and Q4_0 are never picked.
//! The KV pool is `ctx` rows shared by all sequences (ADR 0067), so parallel slots do not
//! multiply it.

use crate::builder::{BUILDER_UBATCHES, enumerate_builder_kv, ram_slot_options};
use crate::candidate::{
    Budget, Enumeration, Placement, UBATCHES, Workload, enumerate_cpu_kv, enumerate_kv, footprint,
};
use crate::layout::ModelLayout;
use crate::lower::ServeEngine;
use crate::serve::{ServeInput, has_gpu, sizing};
use std::fmt;
use std::str::FromStr;
use tetraeder_plan::KvFormat;

/// Formats `auto` may pick, most faithful first.
pub const AUTO_ORDER: [KvFormat; 3] = [KvFormat::F16, KvFormat::Q8_0, KvFormat::Q8_0_Q4_0];

/// Block size of the quantized formats: the head size must be a multiple of it.
const QUANT_BLOCK: u32 = 32;

/// `--kv`: one format, or `auto`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvChoice {
    /// What plans without the field mean today.
    #[default]
    Default,
    Auto,
    Fixed(KvFormat),
}

impl KvChoice {
    /// The format of a plan that does not choose: `Default` and `Auto` start at F16.
    pub fn fixed(self) -> Option<KvFormat> {
        match self {
            Self::Default => Some(KvFormat::default()),
            Self::Fixed(f) => Some(f),
            Self::Auto => None,
        }
    }
}

impl FromStr for KvChoice {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        if s.trim().eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        let kv: KvFormat = s.parse()?;
        if !kv.is_allowed() {
            return Err(format!(
                "KV format {kv} has no flash-attention kernel (allowed: {})",
                KvFormat::ALLOWED.map(|f| f.to_string()).join(", ")
            ));
        }
        Ok(Self::Fixed(kv))
    }
}

impl fmt::Display for KvChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Default => write!(f, "{}", KvFormat::default()),
            Self::Auto => f.write_str("auto"),
            Self::Fixed(kv) => write!(f, "{kv}"),
        }
    }
}

/// The format `auto` chose and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvPick {
    pub format: KvFormat,
    /// One clause per format tried, in order, ending with the one that fits, e.g.
    /// `f16/f16 needs 19.3 GB KV, 14.1 GB free; q8_0/q8_0 fits (10.3 GB KV)`.
    pub reason: String,
    /// The formats that did not fit, with their clause.
    pub skipped: Vec<(KvFormat, String)>,
}

impl KvPick {
    /// The line `explain` output and candidate notes carry.
    pub fn explain(&self) -> String {
        format!("kv {}: {}", self.format, self.reason)
    }
}

fn gb(b: u64) -> String {
    format!("{:.1} GB", b as f64 / 1e9)
}

/// KV bytes of `kv` at the lightest placement of a side, and the bytes free for it beyond the
/// other estimates of that placement.
fn need_free(
    budget: &Budget,
    layout: &ModelLayout,
    w: &Workload,
    (pl, ubatch): (Placement, u32),
    kv: KvFormat,
) -> (u64, u64) {
    let f = footprint(layout, pl, ubatch, w, kv);
    if pl == Placement::Cpu {
        (
            f.kv_host,
            budget.ram_bytes.saturating_sub(f.host - f.kv_host),
        )
    } else {
        (f.kv_gpu, budget.gpu_bytes.saturating_sub(f.gpu - f.kv_gpu))
    }
}

/// The placement with the fewest bytes next to the KV on a side: what a format must fit in at
/// least. `engine` is `None` for the CPU.
pub(crate) fn lightest(layout: &ModelLayout, engine: Option<ServeEngine>) -> (Placement, u32) {
    match engine {
        None => (Placement::Cpu, UBATCHES[0]),
        Some(ServeEngine::Builder) => {
            let k = layout.n_expert_used.max(1);
            let pl = Placement::Builder {
                vram_slots: k,
                ram_slots: ram_slot_options(layout, k)[0],
            };
            (pl, BUILDER_UBATCHES[0])
        }
        Some(ServeEngine::Llama) if layout.is_moe() => (
            Placement::Moe {
                gpu_expert_layers: 0,
            },
            UBATCHES[0],
        ),
        Some(ServeEngine::Llama) => (Placement::Dense { gpu_layers: 1 }, UBATCHES[0]),
    }
}

/// Walks [`AUTO_ORDER`]: `try_fmt` returns the clause of a format that fits (`Ok`) or does not
/// (`Err`). Quantized formats are skipped when the head size is not a multiple of 32.
pub fn pick_kv(
    layout: &ModelLayout,
    mut try_fmt: impl FnMut(KvFormat) -> Result<String, String>,
) -> Result<KvPick, String> {
    let mut skipped: Vec<(KvFormat, String)> = Vec::new();
    for kv in AUTO_ORDER {
        let quant = kv != KvFormat::F16;
        let verdict = if quant && !layout.head_dim.is_multiple_of(QUANT_BLOCK) {
            Err(format!(
                "{kv} needs a head size that is a multiple of {QUANT_BLOCK}, the model has {}",
                layout.head_dim
            ))
        } else {
            try_fmt(kv)
        };
        match verdict {
            Ok(clause) => {
                let mut parts: Vec<&str> = skipped.iter().map(|(_, c)| c.as_str()).collect();
                parts.push(&clause);
                return Ok(KvPick {
                    format: kv,
                    reason: parts.join("; "),
                    skipped,
                });
            }
            Err(clause) => skipped.push((kv, clause)),
        }
    }
    let all: Vec<&str> = skipped.iter().map(|(_, c)| c.as_str()).collect();
    Err(format!("no KV format fits: {}", all.join("; ")))
}

/// The clauses of one placement: does `kv` fit, with the numbers.
fn clause(
    budget: &Budget,
    layout: &ModelLayout,
    w: &Workload,
    at: (Placement, u32),
    kv: KvFormat,
    fits: bool,
) -> Result<String, String> {
    let (need, free) = need_free(budget, layout, w, at, kv);
    if fits {
        Ok(format!("{kv} fits ({} KV)", gb(need)))
    } else {
        Err(format!("{kv} needs {} KV, {} free", gb(need), gb(free)))
    }
}

/// The enumerator of the input's engine for `kv`: the GPU placements, or `None` without a GPU.
fn gpu_enumeration(inp: &ServeInput, w: &Workload, kv: KvFormat) -> Option<Enumeration> {
    has_gpu(&inp.hw).then(|| match inp.engine {
        ServeEngine::Llama => enumerate_kv(&inp.budget, &inp.layout, w, kv),
        ServeEngine::Builder => enumerate_builder_kv(&inp.budget, &inp.layout, w, kv),
    })
}

/// The candidates of `inp` with the KV cache chosen by `choice`, and the pick when it was `auto`.
///
/// `Default` and `Fixed` enumerate as [`crate::serve::candidates`] does, in that format (GPU
/// placements, the CPU when none fits). `Auto` runs the format ladder on the GPU placements
/// first and falls back to the CPU only after every format failed there, so a box with a GPU
/// is not left on the CPU when a smaller KV cache would fit it. The pick's line is added to the
/// notes of every candidate.
pub fn enumerate_choice(
    inp: &ServeInput,
    choice: KvChoice,
) -> Result<(Enumeration, Option<KvPick>), String> {
    let w = sizing(inp);
    let (b, l) = (&inp.budget, &inp.layout);
    if let Some(kv) = choice.fixed() {
        let en = gpu_enumeration(inp, &w, kv)
            .filter(|e| !e.candidates.is_empty())
            .unwrap_or_else(|| enumerate_cpu_kv(b, l, &w, kv));
        return Ok((en, None));
    }
    let gpu_at = lightest(l, Some(inp.engine));
    let cpu_at = lightest(l, None);
    let mut found: Option<Enumeration> = None;
    let on_gpu = has_gpu(&inp.hw);
    let mut gpu_failed = String::new();
    if on_gpu {
        let pick = pick_kv(l, |kv| {
            let en = gpu_enumeration(inp, &w, kv).expect("has a GPU");
            let fits = !en.candidates.is_empty();
            let c = clause(b, l, &w, gpu_at, kv, fits);
            if fits {
                found = Some(en);
            }
            c
        });
        match pick {
            Ok(p) => return Ok(finish(found, p)),
            Err(e) => gpu_failed = e,
        }
    }
    let pick = pick_kv(l, |kv| {
        let en = enumerate_cpu_kv(b, l, &w, kv);
        let fits = !en.candidates.is_empty();
        let c = clause(b, l, &w, cpu_at, kv, fits);
        if fits {
            found = Some(en);
        }
        c
    });
    match pick {
        Ok(mut p) => {
            if on_gpu {
                p.reason = format!(
                    "on the CPU after the GPU failed ({gpu_failed}); {}",
                    p.reason
                );
            }
            Ok(finish(found, p))
        }
        Err(e) if on_gpu => Err(format!("{gpu_failed}; on the CPU: {e}")),
        Err(e) => Err(e),
    }
}

fn finish(found: Option<Enumeration>, pick: KvPick) -> (Enumeration, Option<KvPick>) {
    let mut en = found.expect("the winning format left its enumeration");
    let line = pick.explain();
    for c in &mut en.candidates {
        c.notes.push(line.clone());
    }
    (en, Some(pick))
}

/// The format for one fixed placement (the serve builders place themselves): `Default` and
/// `Fixed` return their format, `Auto` the first of [`AUTO_ORDER`] whose footprint fits the
/// budget. Errors name the numbers of every format tried.
pub fn pick_for_placement(
    budget: &Budget,
    layout: &ModelLayout,
    w: &Workload,
    at: (Placement, u32),
    choice: KvChoice,
) -> Result<(KvFormat, Option<KvPick>), String> {
    if let Some(kv) = choice.fixed() {
        return Ok((kv, None));
    }
    let pick = pick_kv(layout, |kv| {
        let f = footprint(layout, at.0, at.1, w, kv);
        let fits = f.gpu <= budget.gpu_bytes && f.host <= budget.ram_bytes;
        clause(budget, layout, w, at, kv, fits)
    })?;
    Ok((pick.format, Some(pick)))
}
