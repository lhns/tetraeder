use std::time::Duration;

/// Telemetry settings. The default exports nothing.
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    pub service_name: String,
    pub service_version: String,
    /// OTLP/HTTP base URL, for example `http://localhost:4318`. `None`: no exporter.
    pub otlp_endpoint: Option<String>,
    pub export_interval: Duration,
    /// Number of recent step samples kept in process.
    pub ring_capacity: usize,
    /// Extra resource attributes, for example `host.name`.
    pub resource_attributes: Vec<(String, String)>,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            service_name: "tetraeder".into(),
            service_version: env!("CARGO_PKG_VERSION").into(),
            otlp_endpoint: None,
            export_interval: Duration::from_secs(10),
            ring_capacity: 4096,
            resource_attributes: Vec::new(),
        }
    }
}

impl TelemetryConfig {
    /// Defaults plus the standard `OTEL_EXPORTER_OTLP_ENDPOINT` variable (empty counts as unset).
    pub fn from_env() -> Self {
        Self {
            otlp_endpoint: std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
                .ok()
                .filter(|v| !v.trim().is_empty()),
            ..Self::default()
        }
    }
}
