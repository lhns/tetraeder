//! `tetraeder worker --fake` (a `FakeKv`) behind `RemoteBackend` (ADR 0069, "Worker process, one
//! per box" and "Failure handling"): the scheduler's batching properties P1 (every request ends
//! in exactly one terminal event) and P4 (batching never mixes sequences) through the pipe, and
//! the supervision cases: a child that dies mid-step, hangs, answers garbage or half a frame, or
//! reports a sticky error, and the restart cap.
//!
//! Faults are `--fake-fault KIND:TOKEN` and fire on a `decode` input of two or more tokens that
//! holds TOKEN, so sampled tokens (one-token inputs) never trigger them.
//!
//! Hardware label: cpu (the tokenizer comes from the tiny fixture).

#[allow(clippy::all, clippy::pedantic)]
#[path = "../../scheduler/tests/fake/mod.rs"]
mod fake;

mod mode {
    use super::{CTX, FLOOR};
    use std::ffi::OsString;
    use std::time::Duration;
    use tetraeder::remote_backend::RemoteConfig;

    pub fn worker_config(vocab: usize, parallel: usize, extra: &[&str]) -> RemoteConfig {
        let mut args: Vec<OsString> = vec![
            "worker".into(),
            "--fake".into(),
            "--fake-vocab".into(),
            vocab.to_string().into(),
            "--parallel".into(),
            parallel.to_string().into(),
            "--ctx".into(),
            CTX.to_string().into(),
        ];
        args.extend(extra.iter().map(OsString::from));
        let mut c = RemoteConfig::new(env!("CARGO_BIN_EXE_tetraeder"), args);
        c.deadline_floor = FLOOR;
        c.backoff_base = Duration::from_millis(10);
        c.startup_timeout = Duration::from_secs(60);
        c
    }
}

include!("worker_cases/cases.rs");

use tetraeder::remote_backend::RemoteConfig;

// hardware: cpu
#[test]
fn a_worker_that_cannot_start_is_a_startup_error() {
    let err = |c| RemoteBackend::start(c).err().expect("startup error");
    let e = err(RemoteConfig::new("tetraeder-no-such-program", Vec::new()));
    assert!(e.contains("cannot start"), "{e}");
    let mut c = mode::worker_config(VOCAB, 4, &["--no-such-flag"]);
    c.startup_timeout = Duration::from_secs(30);
    let e = err(c);
    assert!(e.contains("did not start"), "{e}");
}

// hardware: cpu
#[test]
fn the_restart_cap_ends_the_process() {
    if std::env::var_os("TETRAEDER_CAP_CHILD").is_some() {
        // The default handler exits with status 1.
        let mut b =
            RemoteBackend::start(mode::worker_config(VOCAB, 4, &["--fake-fault", "crash:60"]))
                .unwrap();
        crash_until_the_cap(&mut b);
        panic!("the process should have exited");
    }
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "the_restart_cap_ends_the_process", "--nocapture"])
        .env("TETRAEDER_CAP_CHILD", "1")
        .status()
        .unwrap();
    assert_eq!(status.code(), Some(1));
}
