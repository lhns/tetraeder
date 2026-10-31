# 0032. Memory crate: reserve as accounting, phase budget semantics, I/O dependencies

Status: Accepted (Phase 2)

## Context

ADR 0031 fixes the arenas, the slot cache and direct I/O. Building them left a few choices open: what a phase budget may borrow, whether the reserve is a buffer, and which crates carry the platform I/O.

## Decision

- **Phase budgets** (plan IR `phase_budgets`, Phase 2+): per stage and device `prefill` and `decode` bytes for slots and workspace. Each phase must satisfy `slots <= slot arena` and `slots + workspace <= slot arena + workspace arena` of the stage on that device, so borrowing happens only between those two arenas and KV is never touched. Weights plus KV plus a phase's bytes must fit capacity minus reserve. Without an entry both phases use the arenas as sized. The expanded plan always carries budgets for every (stage, device) with a slot or workspace buffer. The phase switch repartitions slots inside the existing buffers; it never reallocates.
- **Reserve** is a measured number (others + backend init + allocator overhead), reported for the validator's reserve map. It is not a live buffer, which would count twice against capacity. Metrics report it as kind `reserve`.
- **Dependencies** of `tetraeder-memory`: `io-uring` (Linux, io_uring engine; the thread pool needs no crate), `libc` (Linux, the `O_DIRECT` value differs by architecture), `windows-sys` (Windows, overlapped reads; already in the lock file). The ggml arenas and the telemetry hooks are default features, so the pure parts cross-check without the ggml C build.
- **Test generator**: the plan crate's deterministic plan generator moves from its test file to a `testing` feature, so other crates test against the same plans.

## Consequences

- The io_uring engine is only compiled and cross-checked off Linux; it runs where the kernel and the container allow it (Docker's default seccomp blocks it and the thread pool takes over). Treat it as unverified until a Linux run reports `backend=io_uring`.
- Plans that differ only in phase budgets show up in `diff` and as a `SetBudgets` migration step.
