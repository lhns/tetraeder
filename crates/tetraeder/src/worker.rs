//! `tetraeder worker`: one `BatchBackend` served over stdin and stdout (ADR 0069, "Worker
//! process, one per box"). `serve --worker` starts it as a child behind `RemoteBackend`; the
//! frames are described in `remote_backend`. Stdout carries frames only, so nothing else may
//! print there; diagnostics go to stderr.

use crate::serve::{ServeArgs, loader};
use clap::Args;
use std::io::Write;
use std::sync::Arc;
use std::time::Duration;
use tetraeder::remote_backend::wire::{Info, Reply, Request, SeqState, State, WireError};
use tetraeder::remote_backend::{encode_raw, parse_header, read_frame, write_frame};
use tetraeder_scheduler::{BackendError, BatchBackend, SeqInput};
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    /// Exit without a reply.
    Crash,
    /// Never reply.
    Hang,
    /// Reply `Failed` with the sticky mark.
    Sticky,
    /// Reply with a frame whose header is not JSON.
    Garbage,
    /// Send half a frame and exit.
    Truncate,
}

fn parse_fault(s: &str) -> Result<(Fault, i32), String> {
    let (kind, token) = s
        .split_once(':')
        .ok_or_else(|| format!("`{s}`: expected KIND:TOKEN"))?;
    let kind = match kind {
        "crash" => Fault::Crash,
        "hang" => Fault::Hang,
        "sticky" => Fault::Sticky,
        "garbage" => Fault::Garbage,
        "truncate" => Fault::Truncate,
        _ => {
            return Err(format!(
                "`{kind}`: expected crash, hang, sticky, garbage or truncate"
            ));
        }
    };
    let token = token
        .parse()
        .map_err(|_| format!("`{token}`: not a token id"))?;
    Ok((kind, token))
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

/// Errors that mark the worker as unusable for another call (a CUDA error leaves the context
/// broken: the next call fails too).
fn is_sticky(msg: &str) -> bool {
    msg.to_ascii_lowercase().contains("cuda error")
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
    res
}

/// Answers requests until the input ends (the parent is gone or closed the pipe).
fn serve_loop(
    backend: &mut dyn BatchBackend,
    input: &mut impl std::io::Read,
    output: &mut impl Write,
    faults: &[(Fault, i32)],
) -> anyhow::Result<()> {
    loop {
        let Some(frame) = read_frame(input).map_err(anyhow::Error::msg)? else {
            return Ok(());
        };
        let req: Request = parse_header(&frame).map_err(anyhow::Error::msg)?;
        let mut payload = Vec::new();
        let mut reply = handle(backend, &req, &frame.payload, faults, output, &mut payload)?;
        reply.state = state(backend);
        write_frame(output, &reply, &payload).map_err(anyhow::Error::msg)?;
    }
}

fn state(b: &dyn BatchBackend) -> State {
    State {
        kv_free: b.kv_free(),
        seqs: (0..b.n_seq_max() as u32)
            .map(|s| SeqState {
                len: b.seq_len(s),
                reserved: b.seq_reserved(s),
            })
            .collect(),
        bytes_streamed: b.bytes_streamed_total(),
    }
}

fn info(b: &dyn BatchBackend) -> Info {
    Info {
        n_vocab: b.n_vocab(),
        n_seq_max: b.n_seq_max(),
        n_ctx_seq: b.n_ctx_seq(),
        kv_capacity: b.kv_capacity(),
        kv_format: b.kv_format(),
        on_gpu: b.on_gpu(),
        kv_row_bytes: b.kv_row_bytes(),
        max_step_tokens: u64::try_from(b.max_step_tokens()).unwrap_or(u64::MAX),
    }
}

fn error(e: BackendError) -> WireError {
    match e {
        BackendError::KvFull { seq, needed, free } => WireError::KvFull { seq, needed, free },
        BackendError::Failed(msg) => WireError::Failed {
            sticky: is_sticky(&msg),
            msg,
        },
    }
}

/// Runs one request; the state of the reply is filled in by the caller. Faults act here.
fn handle(
    b: &mut dyn BatchBackend,
    req: &Request,
    bytes: &[u8],
    faults: &[(Fault, i32)],
    output: &mut impl Write,
    payload: &mut Vec<u8>,
) -> anyhow::Result<Reply> {
    let mut reply = Reply {
        err: None,
        spans: Vec::new(),
        info: None,
        state: State::default(),
    };
    let res: Result<(), BackendError> = match req {
        Request::Info => {
            reply.info = Some(info(b));
            Ok(())
        }
        Request::SeqReserve { seq, n } => b.seq_reserve(*seq, *n),
        Request::SeqRm { seq, keep } => b.seq_rm(*seq, *keep),
        Request::SeqCp { src, dst, n } => b.seq_cp(*src, *dst, *n),
        Request::SeqSnapshot { seq } => b.seq_snapshot(*seq).map(|snap| *payload = snap),
        Request::SeqRestore { seq, n } => b.seq_restore(*seq, bytes, *n),
        Request::Decode { inputs } => {
            if let Some(fault) = fault_for(inputs, faults) {
                match fault {
                    Fault::Crash => std::process::exit(3),
                    Fault::Hang => loop {
                        std::thread::sleep(Duration::from_secs(3600));
                    },
                    Fault::Garbage => {
                        let frame = encode_raw(b"not json", &[]).map_err(anyhow::Error::msg)?;
                        output.write_all(&frame)?;
                        output.flush()?;
                        std::process::exit(0);
                    }
                    Fault::Truncate => {
                        output.write_all(&[200, 0, 0, 0, 1, 2, 3])?;
                        output.flush()?;
                        std::process::exit(0);
                    }
                    Fault::Sticky => {
                        reply.err = Some(WireError::Failed {
                            msg: "CUDA error: injected".into(),
                            sticky: true,
                        });
                        return Ok(reply);
                    }
                }
            }
            let ins: Vec<SeqInput<'_>> = inputs
                .iter()
                .map(|i| SeqInput {
                    seq: i.seq,
                    tokens: &i.tokens,
                    all_logits: i.all_logits,
                })
                .collect();
            b.decode(&ins).map(|out| {
                let nv = out.n_vocab().max(1);
                let mut row = 0;
                for i in 0..out.len() {
                    let rows = out.rows(i);
                    let end = row + rows.len() / nv;
                    reply.spans.push((row, end));
                    row = end;
                    payload.reserve(rows.len() * 4);
                    for f in rows {
                        payload.extend_from_slice(&f.to_le_bytes());
                    }
                }
            })
        }
    };
    reply.err = res.err().map(error);
    Ok(reply)
}

fn fault_for(
    inputs: &[tetraeder::remote_backend::wire::WireInput],
    faults: &[(Fault, i32)],
) -> Option<Fault> {
    faults.iter().find_map(|&(kind, token)| {
        inputs
            .iter()
            .any(|i| i.tokens.len() >= 2 && i.tokens.contains(&token))
            .then_some(kind)
    })
}
