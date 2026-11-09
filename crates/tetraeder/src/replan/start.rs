//! Startup of `serve --plan`: hardware, plan, first load, replan thread.

use super::control::{self, ReplanHandle, plan_label};
use super::engine::{Loader, Lowerer, SwitchableEngine};
use super::glue::{builder_options, llama_options};
use super::source::{HwSource, PlannerSource, Shared, band_of};
use crate::serve::{EngineKind, SchedParts, ServeArgs};
use anyhow::Context as _;
use std::sync::Arc;
use std::time::Duration;
use tetraeder_plan::{Hints, Plan, Stability};
use tetraeder_planner::{
    Applies, Lowered, Placement, ServeEngine, ServeInput, Workload, budget_of, fill_knobs, lower,
    pick, predict_plan, recognize,
};
use tetraeder_scheduler::{Scheduler, load_llama, load_moe};
use tetraeder_telemetry::Telemetry;

/// What `serve` needs once the plan is chosen and loaded.
pub struct Planned {
    pub engine: Arc<SwitchableEngine>,
    /// `None` when the graph cannot predict: the plan runs, replanning is off.
    pub replan: Option<ReplanHandle>,
    pub dwell_s: f64,
}

/// The engine that runs a placement: the builder for its own placement, libllama for the rest,
/// and for the CPU what `--engine` says.
fn engine_for(p: Placement, flag: ServeEngine) -> ServeEngine {
    match p {
        Placement::Builder { .. } => ServeEngine::Builder,
        Placement::Dense { .. } | Placement::Moe { .. } => ServeEngine::Llama,
        Placement::Cpu => flag,
    }
}

fn parse_plan_file(path: &str) -> anyhow::Result<Plan> {
    let text = std::fs::read_to_string(path).with_context(|| format!("cannot read plan {path}"))?;
    Plan::from_toml(&text).with_context(|| format!("cannot parse plan {path}"))
}

pub fn start(
    a: &ServeArgs,
    spec: &str,
    telemetry: &Arc<Telemetry>,
    handle: tokio::runtime::Handle,
) -> anyhow::Result<Planned> {
    let path = a.model.clone().context("-m is required")?;
    let layout = crate::plan::layout(&path)?;
    let flag = match a.engine {
        EngineKind::Llama => ServeEngine::Llama,
        EngineKind::Moe => ServeEngine::Builder,
        EngineKind::Auto => {
            let arch = tetraeder_scheduler::architecture(&path)
                .map_err(anyhow::Error::msg)
                .context("cannot read the model architecture")?;
            if arch == "qwen3moe" {
                ServeEngine::Builder
            } else {
                ServeEngine::Llama
            }
        }
    };

    let hw_source = a.hw.clone().map_or(HwSource::Discover, HwSource::File);
    let (hw, notes) = hw_source.read(true).map_err(anyhow::Error::msg)?;
    for n in notes {
        eprintln!("hardware: {n}");
    }
    let quota = std::thread::available_parallelism().map_or(1, |n| n.get() as u32);
    let budget = budget_of(&hw, quota, a.plan_vram, a.plan_ram);
    let band = band_of(&hw);
    let applies = Applies::today();
    let inp = ServeInput {
        hw,
        layout: layout.clone(),
        budget,
        ctx: a.ctx,
        workload: a.plan_workload,
        engine: flag,
        applies,
    };

    let (plan, prediction) = if spec == "auto" {
        let picked = pick(&inp).map_err(anyhow::Error::msg)?;
        for n in &picked.notes {
            eprintln!("plan: {n}");
        }
        (picked.plan, picked.prediction)
    } else {
        let mut plan = parse_plan_file(spec)?;
        for n in fill_knobs(&mut plan, &budget) {
            eprintln!("plan: {n}");
        }
        let (placement, _) = recognize(&plan, &layout)
            .map_err(|e| anyhow::anyhow!("plan {spec} is not executable: {e}"))?;
        let (projected, changed) = applies.project(&plan, engine_for(placement, flag));
        for n in changed {
            eprintln!("plan: {n}");
        }
        let pred = predict_plan(&inp.hw, &layout, &projected, &inp.workload);
        if let Err(e) = &pred {
            eprintln!("plan: no prediction ({e}); replanning is off");
        }
        (projected, pred.ok())
    };
    let (placement, _) = recognize(&plan, &layout).map_err(anyhow::Error::msg)?;
    let engine_kind = engine_for(placement, flag);
    let first = lower(&plan, &layout, engine_kind).map_err(anyhow::Error::msg)?;
    let label = plan_label(&plan);

    let parts = SchedParts::new(a)?;
    let base = a.place.clone();
    let (ctx, miss, prefetch) = (a.ctx, a.miss, a.prefetch);
    let tel = Arc::clone(telemetry);
    let model_path = path.clone();
    let loader: Loader = Box::new(move |l: &Lowered| {
        let cfg = parts.config(&model_path, tel.metrics().clone());
        let p = model_path.clone();
        let t = Arc::clone(&tel);
        match l {
            Lowered::Llama { .. } => {
                let (pl, cc) = llama_options(l, &base, ctx)?;
                Scheduler::start(cfg, move || load_llama(&p, &pl, &cc, Some(t)))
            }
            Lowered::Builder { .. } => {
                let o = builder_options(l, ctx, miss, prefetch, Arc::clone(&t))?;
                Scheduler::start(cfg, move || load_moe(&p, &o))
            }
        }
    });
    let lay = layout.clone();
    let lowerer: Lowerer = Box::new(move |p: &Plan| {
        let (placement, _) = recognize(p, &lay)?;
        lower(p, &lay, engine_for(placement, flag))
    });
    let engine = Arc::new(
        SwitchableEngine::new(
            handle,
            loader,
            lowerer,
            &first,
            Some(telemetry.metrics().clone()),
        )
        .map_err(anyhow::Error::msg)?,
    );
    eprintln!(
        "plan {label}: {} threads={} ubatch={} ({})",
        placement.describe(),
        plan.knobs.threads.unwrap_or(0),
        plan.knobs.ubatch.unwrap_or(0),
        if engine_kind == ServeEngine::Builder {
            "builder"
        } else {
            "libllama"
        }
    );

    let dwell_s = a.replan_dwell as f64;
    let Some(prediction) = prediction else {
        return Ok(Planned {
            engine,
            replan: None,
            dwell_s,
        });
    };
    let stability = match &a.hints {
        Some(f) => {
            let text = std::fs::read_to_string(f)
                .with_context(|| format!("cannot read hints {}", f.display()))?;
            Hints::from_toml(&text)
                .with_context(|| format!("cannot parse hints {}", f.display()))?
                .stability
                .unwrap_or_default()
        }
        None => Stability::default(),
    };
    let shared = Shared::new(inp);
    let source = PlannerSource::new(
        Arc::clone(&shared),
        hw_source,
        (quota, a.plan_vram, a.plan_ram),
        engine.last_load_s(),
    );
    let replan = control::spawn(control::Params {
        tick: Duration::from_secs(a.replan_tick.max(1)),
        dwell_s,
        drain_timeout_s: a.drain_timeout as f64,
        stability,
        plan,
        placement: Some(placement),
        prediction: Some(prediction),
        band,
        engine: Arc::clone(&engine),
        source,
        shared,
        ring: telemetry.ring(),
        metrics: Some(telemetry.metrics().clone()),
    })
    .context("cannot start the replan thread")?;
    Ok(Planned {
        engine,
        replan: Some(replan),
        dwell_s,
    })
}

/// Parses `P:N` (prompt and generated tokens) for `--plan-workload`.
pub fn parse_workload(s: &str) -> Result<Workload, String> {
    let (p, n) = s
        .split_once(':')
        .ok_or_else(|| format!("`{s}`: expected P:N, for example 2048:128"))?;
    let num = |x: &str| {
        x.parse::<u32>()
            .ok()
            .filter(|&v| v > 0)
            .ok_or_else(|| format!("`{s}`: P and N must be positive integers"))
    };
    Ok(Workload {
        prompt: num(p)?,
        r#gen: num(n)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // hardware: none
    #[test]
    fn workload_flag_parses() {
        assert_eq!(
            parse_workload("2048:128").unwrap(),
            Workload {
                prompt: 2048,
                r#gen: 128
            }
        );
        for bad in ["2048", "a:b", "0:5", "5:0", "1:2:3", ""] {
            assert!(parse_workload(bad).is_err(), "{bad}");
        }
    }

    // hardware: none
    #[test]
    fn the_cpu_placement_follows_the_engine_flag() {
        assert_eq!(
            engine_for(Placement::Cpu, ServeEngine::Builder),
            ServeEngine::Builder
        );
        assert_eq!(
            engine_for(Placement::Cpu, ServeEngine::Llama),
            ServeEngine::Llama
        );
        assert_eq!(
            engine_for(
                Placement::Moe {
                    gpu_expert_layers: 1
                },
                ServeEngine::Builder
            ),
            ServeEngine::Llama
        );
        assert_eq!(
            engine_for(
                Placement::Builder {
                    vram_slots: 1,
                    ram_slots: 0
                },
                ServeEngine::Llama
            ),
            ServeEngine::Builder
        );
    }
}
