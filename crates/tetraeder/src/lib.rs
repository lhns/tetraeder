//! The parts of the `tetraeder` binary that integration tests drive: the backend that talks to a
//! worker process (ADR 0069).

pub mod pipeline;
pub mod remote_backend;
pub mod worker_loop;
