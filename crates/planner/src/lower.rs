//! A candidate as llama-bench arguments.

use crate::builder::slot_row_bytes;
use crate::candidate::{CPU, Candidate, GPU, Placement};
use crate::layout::ModelLayout;
use std::collections::BTreeMap;
use tetraeder_plan::{Knobs, KvFormat, Plan, Span, UnitKind};

/// llama-bench `-ngl` for a dense placement. llama.cpp offloads the last `n - 1` layers and the
/// output head for `-ngl n` (the input layer never), so `g` layers plus the head are `g + 1`;
/// `L + 1` is everything.
pub fn dense_ngl(gpu_layers: u32, n_layer: u32) -> u32 {
    if gpu_layers == 0 {
        0
    } else {
        (gpu_layers + 1).min(n_layer + 1)
    }
}

/// Regex alternatives matching exactly the decimal numbers `lo..=hi` (no leading zeros), for
/// example `24..=47` gives `2[4-9]|3[0-9]|4[0-7]`. Full decades next to each other share a
/// class.
pub fn range_regex(lo: u32, hi: u32) -> String {
    assert!(lo <= hi, "empty range {lo}..={hi}");
    let mut parts = Vec::new();
    let mut start = lo;
    let mut width = digits(lo);
    while start <= hi {
        // numbers of this width: up to 10^width - 1
        let top = 10u32.saturating_pow(width) - 1;
        let end = hi.min(top);
        parts.extend(same_width(&to_digits(start, width), &to_digits(end, width)));
        start = end + 1;
        width += 1;
    }
    parts.join("|")
}

fn digits(n: u32) -> u32 {
    n.checked_ilog10().unwrap_or(0) + 1
}

fn to_digits(n: u32, width: u32) -> Vec<u8> {
    format!("{n:0w$}", w = width as usize)
        .bytes()
        .map(|b| b - b'0')
        .collect()
}

fn class(a: u8, b: u8) -> String {
    if a == b {
        a.to_string()
    } else {
        format!("[{a}-{b}]")
    }
}

/// Patterns for the digit strings between `lo` and `hi` (same length, leading zeros allowed).
fn same_width(lo: &[u8], hi: &[u8]) -> Vec<String> {
    if lo.len() == 1 {
        return vec![class(lo[0], hi[0])];
    }
    let rest = lo.len() - 1;
    if lo[0] == hi[0] {
        return same_width(&lo[1..], &hi[1..])
            .into_iter()
            .map(|s| format!("{}{s}", lo[0]))
            .collect();
    }
    let zeros = vec![0u8; rest];
    let nines = vec![9u8; rest];
    let full_lo = lo[1..] == zeros[..];
    let full_hi = hi[1..] == nines[..];
    let mut out = Vec::new();
    if !full_lo {
        for s in same_width(&lo[1..], &nines) {
            out.push(format!("{}{s}", lo[0]));
        }
    }
    let mid_lo = if full_lo { lo[0] } else { lo[0] + 1 };
    let mid_hi = if full_hi { hi[0] } else { hi[0] - 1 };
    if mid_lo <= mid_hi {
        out.push(format!("{}{}", class(mid_lo, mid_hi), "[0-9]".repeat(rest)));
    }
    if !full_hi {
        for s in same_width(&zeros, &hi[1..]) {
            out.push(format!("{}{s}", hi[0]));
        }
    }
    out
}

/// The `-ot` pattern that puts the stacked experts of layers `from..n_layer` on the CPU.
pub fn cpu_experts_pattern(from: u32, n_layer: u32) -> String {
    if from == 0 {
        return "exps=CPU".to_string();
    }
    format!(
        r"blk\.({})\.ffn_(up|down|gate)_exps=CPU",
        range_regex(from, n_layer - 1)
    )
}

/// Which serving engine runs a plan: libllama or the own MoE builder (ADR 0055).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServeEngine {
    Llama,
    Builder,
}

/// Physical batch the builder's serve path chunks prefill at: `tetraeder_scheduler::MOE_CHUNK`.
/// The planner does not depend on the scheduler, so the value is repeated here.
pub const SERVE_BUILDER_UBATCH: u32 = 512;

/// Knobs the serving engines honour. A knob that is off is fixed to what the engine runs, so a
/// prediction describes what happens (ADR 0055).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Applies {
    /// libllama takes the op offload switch from the plan (today it always offloads).
    pub llama_op_offload: bool,
    /// The builder's serve path takes its prefill chunk from the plan (today a fixed 512).
    pub builder_ubatch: bool,
}

impl Applies {
    pub fn today() -> Self {
        Self {
            llama_op_offload: false,
            builder_ubatch: false,
        }
    }

    pub fn all() -> Self {
        Self {
            llama_op_offload: true,
            builder_ubatch: true,
        }
    }

    /// The plan with the knobs this engine cannot honour set to what it runs, and a message per
    /// changed knob. Idempotent; never touches the placement. Op offload only matters (and is
    /// only changed) in a plan that has a GPU in its stage.
    pub fn project(&self, plan: &Plan, engine: ServeEngine) -> (Plan, Vec<String>) {
        let mut p = plan.clone();
        let mut notes = Vec::new();
        let has_gpu = plan
            .replicas
            .iter()
            .flat_map(|r| &r.stages)
            .any(|s| s.devices.iter().any(|d| d == GPU));
        match engine {
            ServeEngine::Llama
                if !self.llama_op_offload && has_gpu && p.knobs.op_offload != Some(true) =>
            {
                notes.push("op_offload: libllama always offloads, set to true".into());
                p.knobs.op_offload = Some(true);
            }
            ServeEngine::Builder
                if !self.builder_ubatch && p.knobs.ubatch != Some(SERVE_BUILDER_UBATCH) =>
            {
                notes.push(format!(
                    "ubatch: the builder chunks prefill at {SERVE_BUILDER_UBATCH}, set to {SERVE_BUILDER_UBATCH}"
                ));
                p.knobs.ubatch = Some(SERVE_BUILDER_UBATCH);
            }
            _ => {}
        }
        (p, notes)
    }
}

/// A plan as the load options of a serving engine, free of engine types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Lowered {
    /// libllama: `-ngl`, `-ot` patterns (`<regex>=CPU`), threads, batches.
    Llama {
        n_gpu_layers: i32,
        cpu_patterns: Vec<String>,
        threads: u32,
        batch: u32,
        ubatch: u32,
        op_offload: bool,
    },
    /// The MoE builder: GPU index (`None`: CPU), expert slots per layer in VRAM (`None`: all
    /// resident) and in the RAM tier.
    Builder {
        gpu: Option<u32>,
        expert_slots: Option<u32>,
        ram_slots: Option<u32>,
        threads: u32,
        ubatch: u32,
        op_offload: bool,
    },
}

/// Inverse of [`build_plan`] for the four executable shapes (`Dense`, `Moe`, `Builder`, `Cpu`).
/// Anything else is an error that names what is not executable. The knobs are the plan's own.
pub fn recognize(plan: &Plan, layout: &ModelLayout) -> Result<(Placement, Knobs), String> {
    let l = layout.n_layer;
    if plan.replicas.len() != 1 {
        return Err(format!("{} replicas", plan.replicas.len()));
    }
    let r = &plan.replicas[0];
    if r.stages.len() != 1 {
        return Err(format!("{} stages", r.stages.len()));
    }
    let st = &r.stages[0];
    if st.layers != Span(0, l) {
        return Err(format!(
            "stage layers {}..{} instead of 0..{l}",
            st.layers.0, st.layers.1
        ));
    }
    if st.tp.unwrap_or(1) != 1 || st.ep.unwrap_or(1) != 1 {
        return Err("tensor or expert parallelism".into());
    }
    if !st.phase_budgets.is_empty() {
        return Err("phase budgets".into());
    }
    if plan.knobs.prefill_micro_batches != 1 {
        return Err("prefill micro batches".into());
    }
    let main = layout.main_format();
    // Device of each unit: true is the GPU.
    let mut attn: Vec<Option<bool>> = vec![None; l as usize];
    let mut mlp = attn.clone();
    let mut expert = attn.clone();
    let (mut embed, mut head): (Option<bool>, Option<bool>) = (None, None);
    for a in &st.assign {
        let gpu = match a.home.as_deref() {
            Some([d]) if d == GPU => true,
            Some([d]) if d == CPU => false,
            h => return Err(format!("home {h:?}")),
        };
        if a.compute
            .as_ref()
            .is_some_and(|c| Some(c) != a.home.as_ref())
        {
            return Err("compute device differs from home".into());
        }
        if a.format != Some(main) {
            return Err(format!("format {:?}", a.format));
        }
        if !a.role_formats.is_empty() {
            return Err("role formats".into());
        }
        if let Some(kv) = a.kv_format
            && kv != KvFormat::default()
        {
            return Err(format!("kv format {kv}"));
        }
        if a.units.experts.is_some() {
            return Err("expert subsets".into());
        }
        let span = a.units.layers.unwrap_or(Span(0, l));
        if span.1 > l || span.0 > span.1 {
            return Err(format!("layers {}..{}", span.0, span.1));
        }
        for kind in &a.units.kinds {
            match kind {
                UnitKind::Attn => set_span(&mut attn, span, gpu)?,
                UnitKind::Mlp => set_span(&mut mlp, span, gpu)?,
                UnitKind::Expert => set_span(&mut expert, span, gpu)?,
                UnitKind::Embed | UnitKind::Head => {
                    if a.units.layers.is_some() {
                        return Err("embed or head with a layer span".into());
                    }
                    let slot = if *kind == UnitKind::Embed {
                        &mut embed
                    } else {
                        &mut head
                    };
                    if slot.replace(gpu).is_some() {
                        return Err("embed or head assigned twice".into());
                    }
                }
            }
        }
    }
    let (Some(false), Some(head)) = (embed, head) else {
        return Err("the embedding is not on the CPU or the head is missing".into());
    };
    if attn != mlp || attn.iter().any(Option::is_none) {
        return Err("units missing or attn and mlp on different devices".into());
    }
    let dense: Vec<bool> = attn.iter().map(|x| x.unwrap_or(false)).collect();
    let experts: Vec<bool> = if layout.is_moe() {
        if expert.iter().any(Option::is_none) {
            return Err("experts not assigned".into());
        }
        expert.iter().map(|x| x.unwrap_or(false)).collect()
    } else if expert.iter().any(Option::is_some) {
        return Err("experts of a dense model".into());
    } else {
        vec![]
    };
    let all = |v: &[bool], x: bool| v.iter().all(|&y| y == x);
    let placement = if !st.expert_cache.is_empty() {
        // Builder: dense and head on the GPU, experts at home on the CPU, the two slot tiers.
        if !(layout.is_moe() && all(&dense, true) && head && all(&experts, false)) {
            return Err("an expert cache outside the builder shape".into());
        }
        let slots = |d: &str| {
            st.expert_cache
                .iter()
                .find(|c| c.device == d)
                .map(|c| c.slots)
        };
        match (st.expert_cache.len(), slots(GPU), slots(CPU)) {
            (2, Some(vram_slots), Some(ram_slots)) => Placement::Builder {
                vram_slots,
                ram_slots,
            },
            _ => return Err("expert cache tiers other than gpu0 and cpu0".into()),
        }
    } else if !st.devices.iter().any(|d| d == GPU) {
        if !(all(&dense, false) && !head && all(&experts, false)) {
            return Err("a GPU unit in a stage without a GPU".into());
        }
        Placement::Cpu
    } else if layout.is_moe() {
        if !(all(&dense, true) && head) {
            return Err("dense weights of an MoE model off the GPU".into());
        }
        let k = experts.iter().take_while(|&&g| g).count();
        if experts[k..].iter().any(|&g| g) {
            return Err("GPU experts are not a leading layer span".into());
        }
        Placement::Moe {
            gpu_expert_layers: k as u32,
        }
    } else {
        let g = dense.iter().rev().take_while(|&&x| x).count();
        if dense[..l as usize - g].iter().any(|&x| x) {
            return Err("GPU layers are not a trailing layer span".into());
        }
        if head != (g > 0) {
            return Err("the output head is not with the GPU layers".into());
        }
        Placement::Dense {
            gpu_layers: g as u32,
        }
    };
    Ok((placement, plan.knobs))
}

fn set_span(v: &mut [Option<bool>], span: Span, gpu: bool) -> Result<(), String> {
    for i in span.0..span.1 {
        if v[i as usize].replace(gpu).is_some() {
            return Err(format!("layer {i} assigned twice"));
        }
    }
    Ok(())
}

pub(crate) fn llama_lowered(
    placement: Placement,
    knobs: &Knobs,
    layout: &ModelLayout,
) -> Option<Lowered> {
    let (n_gpu_layers, cpu_patterns) = match placement {
        Placement::Builder { .. } => return None,
        Placement::Dense { gpu_layers } => (dense_ngl(gpu_layers, layout.n_layer) as i32, vec![]),
        Placement::Moe { gpu_expert_layers } => (
            99,
            if gpu_expert_layers < layout.n_layer {
                vec![cpu_experts_pattern(gpu_expert_layers, layout.n_layer)]
            } else {
                vec![]
            },
        ),
        Placement::Cpu => (0, vec![]),
    };
    let ubatch = knobs.ubatch.unwrap_or(512);
    Some(Lowered::Llama {
        n_gpu_layers,
        cpu_patterns,
        threads: knobs.threads.unwrap_or(1),
        batch: ubatch.max(2048),
        ubatch,
        op_offload: knobs.op_offload.unwrap_or(true),
    })
}

/// A plan as load options of `engine`, through [`recognize`].
pub fn lower(plan: &Plan, layout: &ModelLayout, engine: ServeEngine) -> Result<Lowered, String> {
    let (placement, knobs) = recognize(plan, layout)?;
    match engine {
        ServeEngine::Llama => llama_lowered(placement, &knobs, layout)
            .ok_or_else(|| "a builder placement needs the builder engine".to_string()),
        ServeEngine::Builder => {
            if !layout.is_moe() {
                return Err("the builder needs a MoE model".into());
            }
            let (gpu, expert_slots, ram_slots) = match placement {
                Placement::Builder {
                    vram_slots,
                    ram_slots,
                } => (
                    Some(0),
                    (vram_slots < layout.n_expert).then_some(vram_slots),
                    (ram_slots > 0).then_some(ram_slots),
                ),
                Placement::Cpu => (None, None, None),
                p => return Err(format!("{} needs libllama", p.describe())),
            };
            Ok(Lowered::Builder {
                gpu,
                expert_slots,
                ram_slots,
                threads: knobs.threads.unwrap_or(1),
                ubatch: knobs.ubatch.unwrap_or(512),
                op_offload: knobs.op_offload.unwrap_or(true),
            })
        }
    }
}

/// Arguments of a candidate without `-t`, `-p`, `-n`, `-r` and `-o`, which the bench harness
/// owns: `-ngl`, `-ot`, `-b`, `-ub`, `-nopo`, `-fa`. Rendered from [`Lowered::Llama`], so
/// llama-bench and `serve` take their options from one function.
///
/// MoE candidates use `-ngl 99` and move the experts of the later layers with one `-ot`
/// (llama-bench separates several patterns with `;`; one range needs only one). `-b` is
/// `max(2048, ub)`. Builder candidates have none.
pub fn to_variant_args(c: &Candidate, layout: &ModelLayout) -> Vec<String> {
    llama_lowered(c.placement, &c.plan.knobs, layout)
        .map(|l| llama_args(&l))
        .unwrap_or_default()
}

/// The llama-bench arguments of a [`Lowered::Llama`]; empty for the builder.
pub fn llama_args(l: &Lowered) -> Vec<String> {
    let Lowered::Llama {
        n_gpu_layers,
        cpu_patterns,
        batch,
        ubatch,
        op_offload,
        ..
    } = l
    else {
        return Vec::new();
    };
    let mut a = vec!["-ngl".to_string(), n_gpu_layers.to_string()];
    if !cpu_patterns.is_empty() {
        a.extend(["-ot".to_string(), cpu_patterns.join(";")]);
    }
    a.extend(["-b".to_string(), batch.to_string()]);
    a.extend(["-ub".to_string(), ubatch.to_string()]);
    if !op_offload {
        a.extend(["-nopo".to_string(), "1".to_string()]);
    }
    a.extend(["-fa".to_string(), "1".to_string()]);
    a
}

/// The environment of the `exit_regime` test binary that runs a builder candidate
/// (crates/model/tests/exit_regime.rs). Slot byte budgets buy exactly the planned slot counts
/// per layer (ADR 0037: `floor(bytes / slot row bytes)`), the request is the candidate's
/// prompt and generation length, and the ubatch list holds the one ubatch.
///
/// # Panics
/// If the candidate is not a builder candidate.
pub fn to_exit_regime_env(c: &Candidate, layout: &ModelLayout) -> BTreeMap<String, String> {
    let Placement::Builder {
        vram_slots,
        ram_slots,
    } = c.placement
    else {
        panic!("{} is not a builder candidate", c.id);
    };
    let row = slot_row_bytes(layout);
    let kv = |k: &str, v: String| (format!("TETRAEDER_EXIT_{k}"), v);
    BTreeMap::from([
        kv("GPU", "1".into()),
        kv("VRAM_SLOT_BYTES", (u64::from(vram_slots) * row).to_string()),
        kv("RAM_SLOT_BYTES", (u64::from(ram_slots) * row).to_string()),
        kv("THREADS", c.threads().to_string()),
        kv("UBATCH", c.ubatch().to_string()),
        kv("OP_OFFLOAD", u8::from(c.op_offload()).to_string()),
        kv("PREFILL", c.prompt.to_string()),
        kv("DECODE", c.r#gen.to_string()),
    ])
}

/// [`to_variant_args`] plus `-t`: a complete llama-bench command line apart from the model and
/// the test sizes.
pub fn to_llama_bench_args(c: &Candidate, layout: &ModelLayout) -> Vec<String> {
    let mut a = to_variant_args(c, layout);
    a.extend(["-t".to_string(), c.threads().to_string()]);
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidate::{Workload, build_candidate, build_plan};
    use crate::layout::fixtures;
    use tetraeder_plan::Format;

    /// Tiny matcher for the regex subset `range_regex` emits.
    fn matches(pattern: &str, n: u32) -> bool {
        let s = n.to_string();
        pattern.split('|').any(|alt| {
            let mut cls = Vec::new();
            let b = alt.as_bytes();
            let mut i = 0;
            while i < b.len() {
                if b[i] == b'[' {
                    cls.push((b[i + 1], b[i + 3]));
                    i += 5;
                } else {
                    cls.push((b[i], b[i]));
                    i += 1;
                }
            }
            cls.len() == s.len()
                && s.bytes()
                    .zip(&cls)
                    .all(|(d, &(lo, hi))| (lo..=hi).contains(&d))
        })
    }

    #[test]
    fn range_regex_known_forms() {
        assert_eq!(range_regex(24, 47), "2[4-9]|3[0-9]|4[0-7]");
        assert_eq!(range_regex(40, 63), "[4-5][0-9]|6[0-3]");
        assert_eq!(range_regex(7, 7), "7");
        assert_eq!(range_regex(3, 9), "[3-9]");
        assert_eq!(range_regex(8, 12), "[8-9]|1[0-2]");
        assert_eq!(range_regex(0, 47), "[0-9]|[1-3][0-9]|4[0-7]");
        assert_eq!(range_regex(98, 101), "9[8-9]|10[0-1]");
    }

    #[test]
    fn range_regex_is_exact() {
        for lo in 0..130 {
            for hi in lo..130 {
                let p = range_regex(lo, hi);
                for n in 0..1000 {
                    assert_eq!(
                        matches(&p, n),
                        (lo..=hi).contains(&n),
                        "{lo}..={hi} {p} {n}"
                    );
                }
            }
        }
    }

    #[test]
    fn moe_args() {
        let l = fixtures::moe();
        let w = Workload {
            prompt: 2048,
            r#gen: 128,
        };
        let c = build_candidate(
            "c00",
            &l,
            Placement::Moe {
                gpu_expert_layers: 24,
            },
            (15, 512, false),
            &w,
        );
        let a = to_llama_bench_args(&c, &l);
        let want = [
            "-ngl",
            "99",
            "-ot",
            r"blk\.(2[4-9]|3[0-9]|4[0-7])\.ffn_(up|down|gate)_exps=CPU",
            "-b",
            "2048",
            "-ub",
            "512",
            "-nopo",
            "1",
            "-fa",
            "1",
            "-t",
            "15",
        ];
        assert_eq!(a, want);
        let c = build_candidate(
            "c01",
            &l,
            Placement::Moe {
                gpu_expert_layers: 0,
            },
            (7, 2048, true),
            &w,
        );
        assert_eq!(
            to_variant_args(&c, &l),
            [
                "-ngl", "99", "-ot", "exps=CPU", "-b", "2048", "-ub", "2048", "-fa", "1"
            ]
        );
        let c = build_candidate(
            "c02",
            &l,
            Placement::Moe {
                gpu_expert_layers: 48,
            },
            (7, 512, true),
            &w,
        );
        assert!(!to_variant_args(&c, &l).contains(&"-ot".to_string()));
    }

    #[test]
    fn dense_args() {
        let l = fixtures::dense();
        let w = Workload {
            prompt: 2048,
            r#gen: 128,
        };
        let c = build_candidate(
            "c00",
            &l,
            Placement::Dense { gpu_layers: 40 },
            (15, 2048, true),
            &w,
        );
        assert_eq!(
            to_llama_bench_args(&c, &l),
            [
                "-ngl", "41", "-b", "2048", "-ub", "2048", "-fa", "1", "-t", "15"
            ]
        );
        assert_eq!(dense_ngl(64, 64), 65);
        assert_eq!(dense_ngl(0, 64), 0);
    }

    // hardware: none
    #[test]
    fn builder_env() {
        use crate::builder::{build_builder_candidate, slot_row_bytes};
        let l = fixtures::moe();
        let w = Workload {
            prompt: 2048,
            r#gen: 128,
        };
        let c = build_builder_candidate("c00", &l, (16, 60), (7, 2048, false), &w);
        let row = slot_row_bytes(&l);
        let env = to_exit_regime_env(&c, &l);
        let get = |k: &str| env[&format!("TETRAEDER_EXIT_{k}")].as_str();
        assert_eq!(get("VRAM_SLOT_BYTES"), (16 * row).to_string());
        assert_eq!(get("RAM_SLOT_BYTES"), (60 * row).to_string());
        assert_eq!(
            [get("GPU"), get("THREADS"), get("UBATCH"), get("OP_OFFLOAD")],
            ["1", "7", "2048", "0"]
        );
        assert_eq!((get("PREFILL"), get("DECODE")), ("2048", "128"));
        assert_eq!(env.len(), 8);
        assert!(to_variant_args(&c, &l).is_empty());
        let on = build_builder_candidate("c01", &l, (8, 30), (15, 256, true), &w);
        assert_eq!(
            to_exit_regime_env(&on, &l)["TETRAEDER_EXIT_OP_OFFLOAD"],
            "1"
        );
    }

    // hardware: none
    #[test]
    #[should_panic(expected = "not a builder candidate")]
    fn env_of_a_libllama_candidate_panics() {
        let l = fixtures::moe();
        let w = Workload {
            prompt: 2048,
            r#gen: 128,
        };
        let c = build_candidate(
            "c00",
            &l,
            Placement::Moe {
                gpu_expert_layers: 4,
            },
            (15, 512, true),
            &w,
        );
        to_exit_regime_env(&c, &l);
    }
    // hardware: none
    fn knobs(threads: u32, ubatch: u32, op: bool) -> Knobs {
        Knobs {
            threads: Some(threads),
            ubatch: Some(ubatch),
            op_offload: Some(op),
            ..Default::default()
        }
    }

    const W: Workload = Workload {
        prompt: 2048,
        r#gen: 128,
    };

    fn plan_of(l: &ModelLayout, p: Placement, k: Knobs) -> Plan {
        build_plan(l, p, k, k.ubatch.unwrap(), &W, KvFormat::default())
    }

    fn moe_k(k: u32) -> Placement {
        Placement::Moe {
            gpu_expert_layers: k,
        }
    }

    // hardware: none
    #[test]
    fn recognize_round_trips_every_placement() {
        let k = knobs(7, 2048, false);
        let dense = fixtures::dense();
        let moe = fixtures::moe();
        for p in [
            Placement::Dense { gpu_layers: 0 },
            Placement::Dense { gpu_layers: 1 },
            Placement::Dense { gpu_layers: 40 },
            Placement::Dense { gpu_layers: 64 },
            Placement::Cpu,
        ] {
            let plan = plan_of(&dense, p, k);
            assert_eq!(recognize(&plan, &dense), Ok((p, k)), "{p:?}");
        }
        for p in [
            moe_k(0),
            moe_k(24),
            moe_k(48),
            Placement::Builder {
                vram_slots: 8,
                ram_slots: 60,
            },
            Placement::Builder {
                vram_slots: 128,
                ram_slots: 0,
            },
            Placement::Cpu,
        ] {
            let plan = plan_of(&moe, p, k);
            assert_eq!(recognize(&plan, &moe), Ok((p, k)), "{p:?}");
        }
    }

    // hardware: none
    #[test]
    fn recognize_refuses_what_does_not_run() {
        let l = fixtures::moe();
        let good = plan_of(&l, moe_k(24), knobs(7, 512, true));
        let err = |p: &Plan| recognize(p, &l).unwrap_err();

        let mut two = good.clone();
        two.replicas.push(two.replicas[0].clone());
        assert!(err(&two).contains("2 replicas"), "{}", err(&two));

        let mut kv = good.clone();
        for a in &mut kv.replicas[0].stages[0].assign {
            if a.units.kinds.contains(&UnitKind::Attn) {
                a.kv_format = Some(KvFormat::Q8_0);
            }
        }
        assert!(err(&kv).contains("kv format q8_0"), "{}", err(&kv));

        let mut role = good.clone();
        role.replicas[0].stages[0].assign[1]
            .role_formats
            .insert("attn.v".into(), Format::Q8_0);
        assert!(err(&role).contains("role formats"), "{}", err(&role));

        // the GPU experts moved to the tail of the layers
        let mut moved = good.clone();
        let st = &mut moved.replicas[0].stages[0];
        st.assign[2].units.layers = Some(Span(24, 48));
        st.assign[3].units.layers = Some(Span(0, 24));
        assert!(
            err(&moved).contains("leading layer span"),
            "{}",
            err(&moved)
        );

        let mut short = good.clone();
        short.replicas[0].stages[0].layers = Span(0, 24);
        assert!(err(&short).contains("stage layers"), "{}", err(&short));

        let mut fmt = good;
        fmt.replicas[0].stages[0].assign[1].format = Some(Format::F16);
        assert!(err(&fmt).contains("format"), "{}", err(&fmt));
    }

    // hardware: none
    #[test]
    fn lowering_matches_the_bench_arguments() {
        let l = fixtures::moe();
        let p = plan_of(&l, moe_k(24), knobs(15, 512, false));
        let want = Lowered::Llama {
            n_gpu_layers: 99,
            cpu_patterns: vec![r"blk\.(2[4-9]|3[0-9]|4[0-7])\.ffn_(up|down|gate)_exps=CPU".into()],
            threads: 15,
            batch: 2048,
            ubatch: 512,
            op_offload: false,
        };
        assert_eq!(lower(&p, &l, ServeEngine::Llama), Ok(want));
        // a builder placement needs the builder engine and the other way round
        let b = plan_of(
            &l,
            Placement::Builder {
                vram_slots: 16,
                ram_slots: 60,
            },
            knobs(7, 512, true),
        );
        assert!(lower(&b, &l, ServeEngine::Llama).is_err());
        assert!(lower(&p, &l, ServeEngine::Builder).is_err());
        assert_eq!(
            lower(&b, &l, ServeEngine::Builder),
            Ok(Lowered::Builder {
                gpu: Some(0),
                expert_slots: Some(16),
                ram_slots: Some(60),
                threads: 7,
                ubatch: 512,
                op_offload: true,
            })
        );
        let dense = fixtures::dense();
        let d = plan_of(
            &dense,
            Placement::Dense { gpu_layers: 40 },
            knobs(15, 2048, true),
        );
        assert!(matches!(
            lower(&d, &dense, ServeEngine::Llama),
            Ok(Lowered::Llama {
                n_gpu_layers: 41,
                ..
            })
        ));
        assert!(lower(&d, &dense, ServeEngine::Builder).is_err());
        let c = plan_of(&dense, Placement::Cpu, knobs(8, 512, false));
        assert!(matches!(
            lower(&c, &dense, ServeEngine::Llama),
            Ok(Lowered::Llama { n_gpu_layers: 0, ref cpu_patterns, .. }) if cpu_patterns.is_empty()
        ));
        assert!(
            lower(&c, &dense, ServeEngine::Builder).is_err(),
            "dense has no builder"
        );
    }

    // hardware: none
    #[test]
    fn builder_lowering_of_all_slots_and_cpu() {
        let l = fixtures::moe();
        let all = plan_of(
            &l,
            Placement::Builder {
                vram_slots: 128,
                ram_slots: 0,
            },
            knobs(7, 512, true),
        );
        assert_eq!(
            lower(&all, &l, ServeEngine::Builder),
            Ok(Lowered::Builder {
                gpu: Some(0),
                expert_slots: None,
                ram_slots: None,
                threads: 7,
                ubatch: 512,
                op_offload: true,
            })
        );
        let cpu = plan_of(&l, Placement::Cpu, knobs(2, 512, false));
        assert_eq!(
            lower(&cpu, &l, ServeEngine::Builder),
            Ok(Lowered::Builder {
                gpu: None,
                expert_slots: None,
                ram_slots: None,
                threads: 2,
                ubatch: 512,
                op_offload: false,
            })
        );
    }

    // hardware: none
    #[test]
    fn project_fixes_what_the_engine_cannot_honour() {
        let l = fixtures::moe();
        let off = plan_of(&l, moe_k(24), knobs(7, 2048, false));
        let (p, notes) = Applies::today().project(&off, ServeEngine::Llama);
        assert_eq!((p.knobs.op_offload, notes.len()), (Some(true), 1));
        assert_eq!(p.knobs.ubatch, Some(2048));
        assert_eq!(Applies::today().project(&p, ServeEngine::Llama).0, p);
        assert!(
            Applies::all()
                .project(&off, ServeEngine::Llama)
                .1
                .is_empty()
        );
        let b = plan_of(
            &l,
            Placement::Builder {
                vram_slots: 16,
                ram_slots: 60,
            },
            knobs(7, 2048, false),
        );
        let (p, notes) = Applies::today().project(&b, ServeEngine::Builder);
        assert_eq!(
            (p.knobs.ubatch, notes.len()),
            (Some(SERVE_BUILDER_UBATCH), 1)
        );
        assert_eq!(p.knobs.op_offload, Some(false));
        // a CPU plan has no offload to switch
        let c = plan_of(&l, Placement::Cpu, knobs(2, 512, false));
        assert!(
            Applies::today()
                .project(&c, ServeEngine::Llama)
                .1
                .is_empty()
        );
    }
}
