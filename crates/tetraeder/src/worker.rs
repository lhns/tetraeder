//! `tetraeder worker`: one `BatchBackend` served over stdin and stdout (ADR 0069, "Worker
//! process, one per box"). `serve --worker` starts it as a child behind `RemoteBackend`; the
//! frames are described in `remote_backend`. Stdout carries frames only, so nothing else may
//! print there; diagnostics go to stderr.

use crate::serve::{ServeArgs, loader_with};
use anyhow::Context;
use clap::Args;
use std::path::PathBuf;
use std::sync::Arc;
use tetraeder::worker_loop::{Fault, Flow, parse_fault, serve_loop};
use tetraeder_model::SegmentSpec;
use tetraeder_scheduler::BatchBackend;
use tetraeder_telemetry::{Telemetry, TelemetryConfig};

// The fake backend of the scheduler's tests, for `--fake`.
#[allow(clippy::all, clippy::pedantic)]
#[path = "../../scheduler/tests/fake/mod.rs"]
mod fake;

#[derive(Args)]
pub struct WorkerArgs {
    /// The same model, engine and placement arguments as `serve`. `--fake` serves the scheduler
    /// tests' `FakeKv` (`--ctx` rows, `--parallel` sequences) instead of a model.
    #[command(flatten)]
    pub serve: ServeArgs,
    /// Test faults, `KIND:TOKEN` with KIND crash, hang, sticky, garbage or truncate: a `decode`
    /// input of two or more tokens holding TOKEN triggers it.
    #[arg(long, hide = true, value_parser = parse_fault)]
    pub fake_fault: Vec<(Fault, i32)>,
    /// Vocabulary size of the fake backend.
    #[arg(long, hide = true, default_value_t = 1000)]
    pub fake_vocab: usize,
    /// Every n-th decode of the fake backend fails and rolls back.
    #[arg(long, hide = true)]
    pub fake_fail_every: Option<u32>,
}

/// The file of the segment plan a node's daemon writes for an `Assign` (ADR 0091, step 4): the
/// single-node plan `plan::segment_plan` cut, read here for its layer span.
pub const SEGMENT_PLAN_ENV: &str = "TETRAEDER_SEGMENT_PLAN";

/// The share of the layers `--plan FILE` or [`SEGMENT_PLAN_ENV`] gives this worker; `None` when
/// neither is set (the whole model). The embedding and the head are not read from the rules
/// (a catch-all rule matches both in any stage); `plan::segment_shape` decides from the layer
/// bounds.
pub fn segment_spec(a: &ServeArgs) -> anyhow::Result<Option<SegmentSpec>> {
    let path = match a.plan.as_deref() {
        Some("auto") => {
            anyhow::bail!("worker: --plan auto is the replanner's; give a segment plan file")
        }
        Some(p) => PathBuf::from(p),
        None => match std::env::var_os(SEGMENT_PLAN_ENV) {
            Some(p) => PathBuf::from(p),
            None => return Ok(None),
        },
    };
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("cannot read the segment plan {}", path.display()))?;
    let model = a.model.as_ref().context("-m is required")?;
    let n_layer = tetraeder::pipeline::n_layer_of(model).map_err(anyhow::Error::msg)?;
    let spec = tetraeder::pipeline::segment_spec_of(&text, n_layer)
        .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    Ok(Some(spec))
}

/// The expert slots per layer a segment plan gives: the `expert_cache` of its stages, `gpu0` for
/// the compute tier and `cpu0` for the RAM tier (the shape `planner::lower` reads as a builder
/// placement). `None` for a plan without an expert cache. The slot counts of a cluster plan are
/// the plan's own: no flag or environment variable repeats them.
pub fn plan_slots(plan_toml: &str) -> anyhow::Result<Option<(usize, Option<usize>)>> {
    let plan = tetraeder_plan::Plan::from_toml(plan_toml)
        .map_err(|e| anyhow::anyhow!("not a plan: {e}"))?;
    let caches: Vec<&tetraeder_plan::SlotCache> = plan
        .replicas
        .iter()
        .flat_map(|r| &r.stages)
        .flat_map(|s| &s.expert_cache)
        .collect();
    let slots = |d: &str| {
        let mut it = caches.iter().filter(|c| c.device == d).map(|c| c.slots);
        let first = it.next();
        anyhow::ensure!(
            it.all(|s| Some(s) == first),
            "the stages of a segment disagree on the slots of {d}"
        );
        Ok(first.map(|s| s as usize))
    };
    let (gpu, cpu) = (slots("gpu0")?, slots("cpu0")?);
    Ok(gpu.map(|g| (g, cpu)))
}

/// The arguments of the child for the command line of `serve --worker`: `serve` becomes
/// `worker`, `--worker` is dropped.
pub fn child_args(argv: impl Iterator<Item = String>) -> anyhow::Result<Vec<std::ffi::OsString>> {
    let mut argv = argv.skip(1);
    anyhow::ensure!(
        argv.next().as_deref() == Some("serve"),
        "--worker needs the command line `tetraeder serve ...`"
    );
    Ok(std::iter::once("worker".to_owned())
        .chain(argv.filter(|a| a != "--worker"))
        .map(Into::into)
        .collect())
}

pub fn run(mut a: WorkerArgs) -> anyhow::Result<()> {
    let segment = segment_spec(&a.serve)?;
    // A segment plan with an expert cache sets the slots, unless the flags already did.
    if segment.is_some()
        && a.serve.expert_slots.is_none()
        && a.serve.ram_slots.is_none()
        && let Some(f) = a
            .serve
            .plan
            .clone()
            .map(PathBuf::from)
            .or_else(|| std::env::var_os(SEGMENT_PLAN_ENV).map(PathBuf::from))
        && let Some((vram, ram)) = plan_slots(&std::fs::read_to_string(&f)?)?
    {
        a.serve.expert_slots = Some(vram);
        a.serve.ram_slots = ram;
    }
    let telemetry = Arc::new(Telemetry::new(&TelemetryConfig::from_env())?);
    telemetry.install_global()?;
    let mut backend: Box<dyn BatchBackend> = if a.serve.fake {
        let ctx = a.serve.ctx as usize;
        Box::new(
            fake::FakeKv::new(a.fake_vocab, a.serve.parallel as usize, ctx, ctx)
                .with_fail_every(a.fake_fail_every),
        )
    } else {
        let load = loader_with(&a.serve, Arc::clone(&telemetry), segment)?;
        load().map_err(anyhow::Error::msg)?.backend
    };
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let res = serve_loop(
        backend.as_mut(),
        &mut stdin.lock(),
        &mut stdout.lock(),
        &a.fake_fault,
    );
    drop(backend);
    telemetry.shutdown();
    // A test fault ends the process the way a dying worker does.
    if let Ok(Flow::Exit(code)) = res {
        std::process::exit(code);
    }
    res.map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::plan_slots;

    fn plan(caches: &str) -> String {
        format!(
            "model = \"m\"\n[[replicas]]\n[[replicas.stages]]\nlayers = [0, 4]\ndevices = [\"gpu0\"]\ntp = 1\nep = 1\n{caches}"
        )
    }

    /// hardware: none. A segment plan's expert cache is the worker's slots: gpu0 the compute
    /// tier, cpu0 the RAM tier.
    #[test]
    fn the_slots_are_the_plans_expert_cache() {
        let p = plan(
            "[[replicas.stages.expert_cache]]\ndevice = \"gpu0\"\nslots = 32\n\
             [[replicas.stages.expert_cache]]\ndevice = \"cpu0\"\nslots = 96\n",
        );
        assert_eq!(plan_slots(&p).unwrap(), Some((32, Some(96))));
        let gpu_only = plan("[[replicas.stages.expert_cache]]\ndevice = \"gpu0\"\nslots = 8\n");
        assert_eq!(plan_slots(&gpu_only).unwrap(), Some((8, None)));
        assert_eq!(plan_slots(&plan("")).unwrap(), None);
    }
}
