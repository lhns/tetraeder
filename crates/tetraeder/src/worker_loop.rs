//! The worker's request loop (ADR 0069), shared by `tetraeder worker` (stdin and stdout) and by
//! tests that serve a backend over an in-process transport (ADR 0091). Frames: `remote_backend`.
//!
//! Test faults end the loop with `Flow::Exit(code)` instead of exiting the process; the binary
//! turns that into `process::exit`, a test into a closed connection.

use crate::remote_backend::wire::{Info, Reply, Request, SeqState, State, WireError, WireInput};
use crate::remote_backend::{encode_raw, parse_header, read_frame, write_frame};
use std::io::{Read, Write};
use std::time::Duration;
use tetraeder_scheduler::{BackendError, BatchBackend, SeqInput};

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

/// Parses a `--fake-fault KIND:TOKEN` value.
pub fn parse_fault(s: &str) -> Result<(Fault, i32), String> {
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

/// How a served connection ended.
#[derive(Debug, PartialEq, Eq)]
pub enum Flow {
    /// The input ended: the parent is gone or closed the pipe.
    Eof,
    /// A test fault: end the process (or connection) with this status.
    Exit(i32),
}

enum Handled {
    Reply(Reply),
    Exit(i32),
}

/// Errors that mark the worker as unusable for another call (a CUDA error leaves the context
/// broken: the next call fails too).
fn is_sticky(msg: &str) -> bool {
    msg.to_ascii_lowercase().contains("cuda error")
}

/// Answers requests until the input ends (the parent is gone or closed the pipe).
pub fn serve_loop(
    backend: &mut dyn BatchBackend,
    input: &mut impl Read,
    output: &mut impl Write,
    faults: &[(Fault, i32)],
) -> anyhow::Result<Flow> {
    loop {
        let Some(frame) = read_frame(input).map_err(anyhow::Error::msg)? else {
            return Ok(Flow::Eof);
        };
        let req: Request = parse_header(&frame).map_err(anyhow::Error::msg)?;
        let mut payload = Vec::new();
        let mut reply = match handle(backend, &req, &frame.payload, faults, output, &mut payload)? {
            Handled::Reply(r) => r,
            Handled::Exit(code) => return Ok(Flow::Exit(code)),
        };
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
) -> anyhow::Result<Handled> {
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
        Request::Decode { inputs, .. } => {
            if let Some(fault) = fault_for(inputs, faults) {
                match fault {
                    Fault::Crash => return Ok(Handled::Exit(3)),
                    Fault::Hang => loop {
                        std::thread::sleep(Duration::from_secs(3600));
                    },
                    Fault::Garbage => {
                        let frame = encode_raw(b"not json", &[]).map_err(anyhow::Error::msg)?;
                        output.write_all(&frame)?;
                        output.flush()?;
                        return Ok(Handled::Exit(0));
                    }
                    Fault::Truncate => {
                        output.write_all(&[200, 0, 0, 0, 1, 2, 3])?;
                        output.flush()?;
                        return Ok(Handled::Exit(0));
                    }
                    Fault::Sticky => {
                        reply.err = Some(WireError::Failed {
                            msg: "CUDA error: injected".into(),
                            sticky: true,
                        });
                        return Ok(Handled::Reply(reply));
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
    Ok(Handled::Reply(reply))
}

fn fault_for(inputs: &[WireInput], faults: &[(Fault, i32)]) -> Option<Fault> {
    faults.iter().find_map(|&(kind, token)| {
        inputs
            .iter()
            .any(|i| i.tokens.len() >= 2 && i.tokens.contains(&token))
            .then_some(kind)
    })
}
