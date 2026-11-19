//! Executor options read from the environment, and the per-step trace (ADR 0069 History
//! 2026-11-19, the layer-N decode diagnosis).

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

/// How an [`crate::Executor`] issues. [`ExecOptions::from_env`] reads them once at construction;
/// `ExecOptions::default()` has every option off.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExecOptions {
    /// On by default, `TETRAEDER_RANGE_UID=0` turns it off: ranges get a ggml graph uid, so ggml-cuda skips its per-call node
    /// comparison before a CUDA graph replay (patch 0005).
    pub range_uid: bool,
    /// On by default, `TETRAEDER_ASYNC_INPUTS=0` turns it off: builders upload step inputs on each rank's stream instead of
    /// one blocking copy per tensor ([`crate::Executor::async_inputs`]).
    pub async_inputs: bool,
    /// `TETRAEDER_EXEC_TRACE=<file>`: one JSON line per step appended to the file.
    pub trace: Option<PathBuf>,
    /// `TETRAEDER_EXEC_TRACE_SYNC=1` (with a trace): synchronize the ranks of every op after
    /// issuing it, so each record holds that op's device time; changes the overlap, not results.
    pub trace_sync: bool,
}

fn flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v == "1")
}

/// A flag that is on unless the variable is `0`.
fn flag_default_on(name: &str) -> bool {
    std::env::var(name).map_or(true, |v| v != "0")
}

impl ExecOptions {
    pub fn from_env() -> ExecOptions {
        ExecOptions {
            range_uid: flag_default_on("TETRAEDER_RANGE_UID"),
            async_inputs: flag_default_on("TETRAEDER_ASYNC_INPUTS"),
            trace: std::env::var_os("TETRAEDER_EXEC_TRACE")
                .filter(|p| !p.is_empty())
                .map(PathBuf::from),
            trace_sync: flag("TETRAEDER_EXEC_TRACE_SYNC"),
        }
    }
}

/// One record: what, on which rank (and peer rank of a copy), host nanoseconds of the call.
struct Record {
    kind: &'static str,
    rank: usize,
    peer: Option<usize>,
    ns: u64,
}

/// Records of the current step, written as one line by [`Trace::flush`]:
/// `{"step":N,"sync":bool,"ev":[[kind,rank,peer,ns],...]}` with peer -1 except for copies.
pub(crate) struct Trace {
    out: BufWriter<File>,
    pub(crate) sync: bool,
    step: u64,
    records: Vec<Record>,
}

impl Trace {
    pub(crate) fn open(opts: &ExecOptions) -> Option<Trace> {
        let path = opts.trace.as_ref()?;
        match OpenOptions::new().create(true).append(true).open(path) {
            Ok(f) => {
                tracing::info!(path = %path.display(), sync = opts.trace_sync, "executor trace");
                Some(Trace {
                    out: BufWriter::new(f),
                    sync: opts.trace_sync,
                    step: 0,
                    records: Vec::new(),
                })
            }
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "executor trace not opened");
                None
            }
        }
    }

    pub(crate) fn note(&mut self, kind: &'static str, rank: usize, peer: Option<usize>, ns: u64) {
        self.records.push(Record {
            kind,
            rank,
            peer,
            ns,
        });
    }

    pub(crate) fn flush(&mut self) {
        if self.records.is_empty() {
            return;
        }
        let mut line = format!("{{\"step\":{},\"sync\":{},\"ev\":[", self.step, self.sync);
        for (i, r) in self.records.iter().enumerate() {
            if i > 0 {
                line.push(',');
            }
            let peer = r.peer.map_or(-1, |p| p as i64);
            line.push_str(&format!("[\"{}\",{},{peer},{}]", r.kind, r.rank, r.ns));
        }
        line.push_str("]}\n");
        if let Err(e) = self
            .out
            .write_all(line.as_bytes())
            .and_then(|()| self.out.flush())
        {
            tracing::warn!(error = %e, "executor trace write failed");
        }
        self.records.clear();
        self.step += 1;
    }
}

impl Drop for Trace {
    fn drop(&mut self) {
        self.flush();
    }
}
