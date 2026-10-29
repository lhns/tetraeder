//! OpenTelemetry metrics, traces and logs, plus GPU and host metrics (ADR 0012).
//!
//! Nothing leaves the machine unless an OTLP endpoint is configured. Without one, no
//! exporter exists; the typed metrics still feed the in-process ring buffer that
//! `tetraeder top` and the detector read.

mod config;
mod host;
mod metrics;
mod ring;
mod setup;
pub mod top;

pub use config::TelemetryConfig;
#[cfg(feature = "nvml")]
pub use host::NvmlHost;
pub use host::{DeviceKind, DeviceSample, HostMetrics, SysinfoHost, default_sources};
pub use metrics::{CacheResult, Direction, Metrics, Phase, StepSample, attr};
pub use ring::{StepRing, StepStats};
pub use setup::{Telemetry, TelemetryError};
