//! A plan file as an executor topology (brief E6): the plan, the model's layout and the hardware
//! graph go through the validator (Phase 5) and `Topology::from_plan`, so `bench --engine dense
//! --plan` and `serve --engine dense --plan` run exactly the shapes the planner emits.

use crate::replan::control::plan_label;
use crate::replan::source::HwSource;
use anyhow::{Context as _, Result, anyhow};
use std::collections::BTreeMap;
use std::path::Path;
use tetraeder_plan::{Context, Phase, Plan, Topology, validate};

/// What a plan file runs as.
pub struct ExecPlan {
    pub topology: Topology,
    /// Eight hex digits of the plan's TOML (`serve --plan` labels plans the same way).
    pub plan_id: String,
}

/// Reads and validates `plan` for the model at `model`. The hardware graph is `hw` (TOML or
/// JSON) or discovered on this box. Rank i of the topology is the GPU the hardware graph names
/// for it (its NVML index, which is ggml's device order unless `CUDA_DEVICE_ORDER` says
/// otherwise).
pub fn load_plan(model: &Path, plan: &Path, hw: Option<&Path>) -> Result<ExecPlan> {
    let layout = crate::plan::layout(model)?;
    let desc = layout.to_model_desc();
    let text = std::fs::read_to_string(plan)
        .with_context(|| format!("cannot read plan {}", plan.display()))?;
    let plan_doc =
        Plan::from_toml(&text).with_context(|| format!("cannot parse plan {}", plan.display()))?;
    let source = hw.map_or(HwSource::Discover, |p| HwSource::File(p.to_path_buf()));
    let (graph, notes) = source.read(false).map_err(anyhow::Error::msg)?;
    for n in notes {
        eprintln!("hardware: {n}");
    }
    let reserve = BTreeMap::new();
    let ctx = Context {
        hw: &graph,
        model: &desc,
        phase: Phase(5),
        reserve: &reserve,
        hints: None,
    };
    let expanded = validate(&plan_doc, &ctx).map_err(|issues| {
        anyhow!(
            "plan {} is not valid for this box: {}",
            plan.display(),
            issues
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("; ")
        )
    })?;
    let topology = Topology::from_plan(&plan_doc, &expanded, &graph).map_err(|reasons| {
        anyhow!(
            "plan {} cannot run on the executor: {}",
            plan.display(),
            reasons.join("; ")
        )
    })?;
    Ok(ExecPlan {
        topology,
        plan_id: plan_label(&plan_doc),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tetraeder_plan::Hints;
    use tetraeder_planner::enumerate_multi;
    use tetraeder_profiler::HardwareGraph;

    fn root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    // hardware: none
    /// Every shape the planner enumerates for two simulated RTX 3090s runs through
    /// `load_plan` as a topology of the same ranks (`layer-2` everywhere, `tp-2` where the
    /// planner proposes it: Linux).
    #[test]
    fn planner_candidates_load_as_topologies() {
        let model = root().join("tests/fixtures/tiny-qwen3-f16.gguf");
        let hw_path = root().join("crates/costmodel/fixtures/two_3090_hoststaged.toml");
        let hw = HardwareGraph::from_toml(&std::fs::read_to_string(&hw_path).unwrap()).unwrap();
        let layout = crate::plan::layout(&model).unwrap();
        let found = enumerate_multi(&hw, &layout, &Hints::default(), Phase(5));
        let ids: Vec<&str> = found.candidates.iter().map(|c| c.id.as_str()).collect();
        assert!(
            ids.contains(&"layer-2"),
            "{ids:?}, pruned {:?}",
            found.pruned
        );
        for c in &found.candidates {
            let file = std::env::temp_dir().join(format!(
                "tetraeder-plan-{}-{}.toml",
                std::process::id(),
                c.id
            ));
            std::fs::write(&file, c.plan.to_toml().unwrap()).unwrap();
            let p = load_plan(&model, &file, Some(&hw_path)).unwrap();
            std::fs::remove_file(&file).ok();
            assert_eq!(p.topology.ranks.len(), c.gpus.len(), "{}", c.id);
            assert_eq!(p.plan_id.len(), 8);
            match c.id.as_str() {
                "layer-2" => assert_eq!(p.topology.stages.len(), 2),
                "tp-2" => assert_eq!(p.topology.stages[0].ranks.len(), 2),
                _ => {}
            }
            eprintln!("{}: {:?}", c.id, p.topology.stages);
        }
    }
}
