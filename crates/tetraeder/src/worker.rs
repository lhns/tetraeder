//! `tetraeder worker`: one `BatchBackend` served over stdin and stdout (ADR 0069, "Worker
//! process, one per box"). `serve --worker` starts it as a child behind `RemoteBackend`; the
//! frames are described in `remote_backend`. Stdout carries frames only, so nothing else may
//! print there; diagnostics go to stderr.

use crate::serve::{ServeArgs, loader};
use clap::Args;
use std::sync::Arc;
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
    anyhow::ensure!(
        a.serve.plan.is_none(),
        "worker: --plan needs the multi-device executor, which is not wired in yet"
    );
    let telemetry = Arc::new(Telemetry::new(&TelemetryConfig::from_env())?);
    telemetry.install_global()?;
    let mut backend: Box<dyn BatchBackend> = if a.serve.fake {
        let ctx = a.serve.ctx as usize;
        Box::new(
            fake::FakeKv::new(a.fake_vocab, a.serve.parallel as usize, ctx, ctx)
                .with_fail_every(a.fake_fail_every),
        )
    } else {
        let load = loader(&a.serve, Arc::clone(&telemetry))?;
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
