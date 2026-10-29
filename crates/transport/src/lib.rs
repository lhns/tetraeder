//! One trait for collectives and point-to-point transfers: NCCL, RCCL, QUIC, RDMA, shared memory.
//!
//! Phase 0 content: the transport spike (ADR 0023). `bench` measures round-trip latency and throughput over
//! plain TCP, TCP with TLS and QUIC; no trait yet, that comes with Phase 6 once the spike has picked a transport.

pub mod bench;
pub mod proto;
pub mod tls;
