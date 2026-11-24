//! `tetraeder worker`: one `BatchBackend` served over stdin and stdout (ADR 0069, "Worker
//! process, one per box"). `serve --worker` starts it as a child behind `RemoteBackend`; the
//! frames are described in `remote_backend`. Stdout carries frames only, so nothing else may
//! print there; diagnostics go to stderr.

use crate::serve::{ServeArgs, loader_with};
use anyhow::Context;
use clap::Args;
use std::path::PathBuf;
use std::sync::Arc;
use tetraeder_model::SegmentSpec;
use tetraeder::worker_loop::{Fault, Flow, parse_fault, serve_loop};
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
    let plan = tetraeder_plan::Plan::from_toml(&text)
        .with_context(|| format!("{} is not a plan", path.display()))?;
    let model = a.model.as_ref().context("-m is required")?;
    let gguf = tetraeder_model::Gguf::open(model)
        .with_context(|| format!("cannot open {}", model.display()))?;
    let arch = gguf.str("general.architecture")?.to_string();
    let n_layer = u32::try_from(gguf.u64(&format!("{arch}.block_count"))?)?;
    let shape = tetraeder_plan::segment_shape(&plan, n_layer).map_err(anyhow::Error::msg)?;
    let spec = SegmentSpec {
        layers: shape.layers.0..shape.layers.1,
        embed: shape.embed,
        head: shape.head,
    };
    spec.validate(n_layer)?;
    Ok(Some(spec))
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

pub fn run(a: &WorkerArgs) -> anyhow::Result<()> {
    let segment = segment_spec(&a.serve)?;
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
