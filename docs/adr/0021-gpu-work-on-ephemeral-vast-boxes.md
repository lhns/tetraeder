# 0021. All GPU work runs on ephemeral Vast boxes

Status: Accepted

## Context

There is no own GPU host, and the dev VM has no GPU. The design doc assumed a permanent nightly GPU host with locked clocks. A scheduled rental is cheaper (about one hour a night at 0.35 USD) and needs no hardware to maintain.

## Decision

All GPU work (benchmarks, parity, spikes) runs on ephemeral Vast.ai boxes, driven only by `scripts/vastctl.py`. Nightly GPU CI is a scheduled rent, run, fetch, destroy job (`vastctl nightly`). Boxes start from a prebuilt GHCR image (`ghcr.io/lhns/tetraeder-box`) with the toolchain and ggml already built, so a box only builds Rust incrementally. Budget caps: 0.45 USD/h, one instance.

## Consequences

- Clocks cannot be locked and the host changes nightly, so benchmarks are paired same-box runs with a noise band per GPU class; absolute tok/s is never compared across nights.
- Windows GPU runtime is untested (Vast is Linux only): Windows gets CPU dev and tests plus CUDA compile-only jobs in CI.
- HIP, RDMA and Metal cannot be tested on Vast; M1 stays CUDA and CPU (ADR 0019).
- Fault injection is limited to what a container allows; the simulator covers the rest.
