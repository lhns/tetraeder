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
    /// Defaults plus the standard `OTEL_EXPORTER_OTLP_ENDPOINT` (empty counts as unset) and
    /// `OTEL_METRIC_EXPORT_INTERVAL` (milliseconds; box runs use 1000, ADR 0059) variables.
    /// `OTEL_RESOURCE_ATTRIBUTES` is read by the SDK's resource detector.
    pub fn from_env() -> Self {
        Self::from_vars(|k| std::env::var(k).ok())
    }

    fn from_vars(var: impl Fn(&str) -> Option<String>) -> Self {
        let d = Self::default();
        Self {
            otlp_endpoint: var("OTEL_EXPORTER_OTLP_ENDPOINT").filter(|v| !v.trim().is_empty()),
            export_interval: var("OTEL_METRIC_EXPORT_INTERVAL")
                .and_then(|v| v.trim().parse::<u64>().ok())
                .filter(|&ms| ms > 0)
                .map_or(d.export_interval, Duration::from_millis),
            ..d
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_sets_endpoint_and_export_interval() {
        let c = TelemetryConfig::from_vars(|k| match k {
            "OTEL_EXPORTER_OTLP_ENDPOINT" => Some("http://127.0.0.1:4318".into()),
            "OTEL_METRIC_EXPORT_INTERVAL" => Some("1000".into()),
            _ => None,
        });
        assert_eq!(c.otlp_endpoint.as_deref(), Some("http://127.0.0.1:4318"));
        assert_eq!(c.export_interval, Duration::from_secs(1));
        let d = TelemetryConfig::from_vars(|k| match k {
            "OTEL_EXPORTER_OTLP_ENDPOINT" => Some(" ".into()),
            "OTEL_METRIC_EXPORT_INTERVAL" => Some("soon".into()),
            _ => None,
        });
        assert!(d.otlp_endpoint.is_none());
        assert_eq!(
            d.export_interval,
            TelemetryConfig::default().export_interval
        );
    }
}
