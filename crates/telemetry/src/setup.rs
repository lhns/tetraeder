use crate::config::TelemetryConfig;
use crate::host::{DeviceKind, DeviceSample, HostMetrics};
use crate::metrics::{Metrics, StepSample};
use crate::ring::StepRing;
use opentelemetry::KeyValue;
use opentelemetry::global;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_appender_tracing::layer::OpenTelemetryTracingBridge;
use opentelemetry_otlp::{LogExporter, MetricExporter, Protocol, SpanExporter, WithExportConfig};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::logs::SdkLoggerProvider;
use opentelemetry_sdk::metrics::SdkMeterProvider;
use opentelemetry_sdk::trace::SdkTracerProvider;
use std::sync::Arc;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

#[derive(Debug, thiserror::Error)]
pub enum TelemetryError {
    #[error("OTLP exporter setup failed: {0}")]
    Exporter(String),
    #[error("tracing subscriber already installed")]
    Subscriber,
}

fn exporter_err(e: impl std::fmt::Display) -> TelemetryError {
    TelemetryError::Exporter(e.to_string())
}

/// Owns the OpenTelemetry providers, the typed metrics and the step ring.
///
/// Without an OTLP endpoint there are no exporters and no tracer or logger provider.
pub struct Telemetry {
    metrics: Metrics,
    ring: Arc<StepRing>,
    meter_provider: SdkMeterProvider,
    tracer_provider: Option<SdkTracerProvider>,
    logger_provider: Option<SdkLoggerProvider>,
}

fn resource(cfg: &TelemetryConfig) -> Resource {
    Resource::builder()
        .with_service_name(cfg.service_name.clone())
        .with_attributes(
            [KeyValue::new(
                "service.version",
                cfg.service_version.clone(),
            )]
            .into_iter()
            .chain(
                cfg.resource_attributes
                    .iter()
                    .map(|(k, v)| KeyValue::new(k.clone(), v.clone())),
            ),
        )
        .build()
}

impl Telemetry {
    /// Builds providers from `cfg`; touches no global state.
    pub fn new(cfg: &TelemetryConfig) -> Result<Self, TelemetryError> {
        let res = resource(cfg);
        let Some(base) = cfg.otlp_endpoint.as_deref() else {
            let mp = SdkMeterProvider::builder().with_resource(res).build();
            return Ok(Self::from_parts(cfg, mp, None, None));
        };
        let url = |signal: &str| format!("{}/v1/{signal}", base.trim_end_matches('/'));

        let metric_exp = MetricExporter::builder()
            .with_http()
            .with_protocol(Protocol::HttpBinary)
            .with_endpoint(url("metrics"))
            .build()
            .map_err(exporter_err)?;
        let mp = SdkMeterProvider::builder()
            .with_resource(res.clone())
            .with_reader(
                opentelemetry_sdk::metrics::PeriodicReader::builder(metric_exp)
                    .with_interval(cfg.export_interval)
                    .build(),
            )
            .build();

        let span_exp = SpanExporter::builder()
            .with_http()
            .with_protocol(Protocol::HttpBinary)
            .with_endpoint(url("traces"))
            .build()
            .map_err(exporter_err)?;
        let tp = SdkTracerProvider::builder()
            .with_resource(res.clone())
            .with_batch_exporter(span_exp)
            .build();

        let log_exp = LogExporter::builder()
            .with_http()
            .with_protocol(Protocol::HttpBinary)
            .with_endpoint(url("logs"))
            .build()
            .map_err(exporter_err)?;
        let lp = SdkLoggerProvider::builder()
            .with_resource(res)
            .with_batch_exporter(log_exp)
            .build();

        Ok(Self::from_parts(cfg, mp, Some(tp), Some(lp)))
    }

    /// Wraps caller-built providers (tests, benchmarks, custom exporters).
    pub fn from_parts(
        cfg: &TelemetryConfig,
        meter_provider: SdkMeterProvider,
        tracer_provider: Option<SdkTracerProvider>,
        logger_provider: Option<SdkLoggerProvider>,
    ) -> Self {
        use opentelemetry::metrics::MeterProvider as _;
        let meter = meter_provider.meter("tetraeder");
        Self {
            metrics: Metrics::new(&meter),
            ring: Arc::new(StepRing::new(cfg.ring_capacity)),
            meter_provider,
            tracer_provider,
            logger_provider,
        }
    }

    pub fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    pub fn ring(&self) -> Arc<StepRing> {
        Arc::clone(&self.ring)
    }

    /// True when any exporter is configured.
    pub fn exporting(&self) -> bool {
        self.tracer_provider.is_some()
    }

    /// Record one step into the ring and the step histogram.
    pub fn record_step(&self, s: StepSample) {
        self.ring.push(s);
        self.metrics.step_duration(&s);
    }

    /// Sample a host source, publish the readings as gauges, and return them.
    pub fn sample_host(&self, src: &mut dyn HostMetrics) -> Vec<DeviceSample> {
        let samples = src.sample();
        for d in &samples {
            self.metrics.device_busy(&d.id, d.busy);
            let tier = match d.kind {
                DeviceKind::Cpu => "ram",
                DeviceKind::Gpu => "vram",
            };
            self.metrics.memory_used(&d.id, tier, "total", d.mem_used);
            self.metrics
                .memory_capacity(&d.id, tier, "total", d.mem_total);
            if let Some(w) = d.power_w {
                self.metrics.device_power(&d.id, w);
            }
            if let Some(t) = d.temp_c {
                self.metrics.device_temperature(&d.id, t);
            }
        }
        samples
    }

    /// Set the global meter and tracer providers and install the `tracing` subscriber:
    /// stderr output filtered by `RUST_LOG`, plus span and log bridges when exporting.
    pub fn install_global(&self) -> Result<(), TelemetryError> {
        global::set_meter_provider(self.meter_provider.clone());
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        let fmt = tracing_subscriber::fmt::layer().with_writer(std::io::stderr);
        let reg = tracing_subscriber::registry().with(filter).with(fmt);
        match (&self.tracer_provider, &self.logger_provider) {
            (Some(tp), Some(lp)) => {
                global::set_tracer_provider(tp.clone());
                let spans = tracing_opentelemetry::layer().with_tracer(tp.tracer("tetraeder"));
                // Keep the exporters' own HTTP stack out of the log bridge to avoid feedback.
                let logs = OpenTelemetryTracingBridge::new(lp).with_filter(
                    tracing_subscriber::filter::filter_fn(|m| {
                        let t = m.target();
                        !(t.starts_with("hyper")
                            || t.starts_with("reqwest")
                            || t.starts_with("opentelemetry"))
                    }),
                );
                reg.with(spans).with(logs).try_init()
            }
            _ => reg.try_init(),
        }
        .map_err(|_| TelemetryError::Subscriber)
    }

    /// Flush and stop exporters.
    pub fn shutdown(&self) {
        let _ = self.meter_provider.shutdown();
        if let Some(tp) = &self.tracer_provider {
            let _ = tp.shutdown();
        }
        if let Some(lp) = &self.logger_provider {
            let _ = lp.shutdown();
        }
    }
}

impl Drop for Telemetry {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{CacheResult, Direction, Phase};
    use opentelemetry_sdk::metrics::InMemoryMetricExporter;
    use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
    use std::time::Duration;

    fn step(ms: u64) -> StepSample {
        StepSample {
            phase: Phase::Decode,
            stage: 0,
            graphs: true,
            duration: Duration::from_millis(ms),
            tokens: 1,
            segments: 1,
        }
    }

    #[test]
    fn default_config_has_no_exporter() {
        let cfg = TelemetryConfig::default();
        assert!(cfg.otlp_endpoint.is_none());
        let t = Telemetry::new(&cfg).unwrap();
        assert!(!t.exporting());
        t.record_step(step(5));
        assert_eq!(t.ring().stats().count, 1);
    }

    #[test]
    fn catalog_metrics_reach_the_reader_with_semconv_names() {
        let exp = InMemoryMetricExporter::default();
        let mp = SdkMeterProvider::builder()
            .with_reader(opentelemetry_sdk::metrics::PeriodicReader::builder(exp.clone()).build())
            .build();
        let cfg = TelemetryConfig::default();
        let t = Telemetry::from_parts(&cfg, mp.clone(), None, None);
        let m = t.metrics();
        m.time_to_first_token(Duration::from_millis(80), "qwen3", "chat");
        m.time_per_output_token(Duration::from_millis(20), "qwen3", "chat");
        m.request_duration(Duration::from_secs(1), "qwen3", "chat", None);
        m.tokens(7, "qwen3", Direction::Output, CacheResult::Miss);
        t.record_step(step(12));
        m.device_busy("gpu0", 0.5);
        m.memory_used("gpu0", "vram", "weights", 1);
        m.memory_capacity("gpu0", "vram", "weights", 2);
        m.kv_cache("gpu0", "vram", "q8_0/q4_0", 3, 4, 5, 6);
        m.expert_cache_requests(3, 4, "vram", CacheResult::Hit);
        m.expert_load_duration(Duration::from_millis(2), "vram", "nvme");
        m.tier_read_bytes(4096, "nvme0", "nvme");
        m.tier_transfer(
            4096,
            Some(Duration::from_millis(1)),
            crate::Tier::Disk,
            crate::Tier::Ram,
            crate::Purpose::ExpertMiss,
        );
        m.expert_layer_hit(true);
        m.expert_prefetch(2, 4, "issued");
        m.expert_prefetch_wasted_bytes(10);
        m.expert_admitted(4, 10);
        m.expert_prefetch_accuracy(4, 1, 0.5);
        m.io_inflight("urgent", 3);
        m.read_hidden(0.7);
        m.layer_part("read_wait", Duration::from_millis(1));
        m.spec_step("ngram", 4, 3);
        m.plan_residual(0, 0.08);
        m.plan_regret("p1", 0.04);
        m.plan_predicted_step(0, 0.012);
        m.bottleneck_share("pcie", 0.6);
        m.bottleneck_fault("pcie_degraded", "pcie");
        m.replan("drift", "switched");
        m.replan_duration(Duration::from_secs(3));
        m.cpu_backend("avx2", &["AVX2".to_string()], "dl", &[]);
        m.cpu_threads(23, Some(23.04));
        m.cpu_poll(0, true);
        m.cpu_pool_pauses(3);
        mp.force_flush().unwrap();

        let mut names = Vec::new();
        for rm in exp.get_finished_metrics().unwrap() {
            for sm in rm.scope_metrics() {
                for metric in sm.metrics() {
                    names.push(metric.name().to_owned());
                    if metric.name() == "tetraeder.step.duration" {
                        let AggregatedMetrics::F64(MetricData::Histogram(h)) = metric.data() else {
                            panic!("step.duration is not an f64 histogram");
                        };
                        let dp = h.data_points().next().unwrap();
                        let keys: Vec<_> = dp.attributes().map(|kv| kv.key.to_string()).collect();
                        for k in ["tetraeder.phase", "tetraeder.stage", "tetraeder.graphs"] {
                            assert!(keys.iter().any(|x| x == k), "missing {k}");
                        }
                    }
                }
            }
        }
        for n in [
            "gen_ai.server.time_to_first_token",
            "gen_ai.server.time_per_output_token",
            "gen_ai.server.request.duration",
            "tetraeder.tokens",
            "tetraeder.step.duration",
            "tetraeder.device.busy",
            "tetraeder.memory.used",
            "tetraeder.memory.capacity",
            "tetraeder.kv.tokens",
            "tetraeder.kv.capacity",
            "tetraeder.expert.cache.requests",
            "tetraeder.expert.load.duration",
            "tetraeder.tier.read.bytes",
            "tetraeder.tier.transfer.bytes",
            "tetraeder.tier.transfer.duration",
            "tetraeder.expert.layer_hit",
            "tetraeder.expert.prefetch",
            "tetraeder.expert.prefetch.wasted",
            "tetraeder.expert.admitted",
            "tetraeder.expert.admitted.bytes",
            "tetraeder.expert.prefetch.accuracy",
            "tetraeder.io.inflight",
            "tetraeder.moe.read.hidden",
            "tetraeder.moe.layer.duration",
            "tetraeder.spec.acceptance",
            "tetraeder.spec.drafted",
            "tetraeder.spec.accepted",
            "tetraeder.plan.residual",
            "tetraeder.plan.regret",
            "tetraeder.plan.predicted_step",
            "tetraeder.bottleneck.share",
            "tetraeder.bottleneck.fault",
            "tetraeder.replan.total",
            "tetraeder.replan.duration",
            "tetraeder.cpu.variant",
            "tetraeder.cpu.isa.missing",
            "tetraeder.cpu.threads",
            "tetraeder.cpu.quota",
            "tetraeder.cpu.poll",
            "tetraeder.cpu.pool.pauses",
        ] {
            assert!(names.iter().any(|x| x == n), "missing metric {n}");
        }
    }

    /// Overhead of recording one step. Run with `--nocapture` to see the numbers.
    /// "off" = no exporter (ring + histogram into a reader-less provider);
    /// "on" = a periodic reader with an in-memory exporter, the same aggregation path as OTLP
    /// minus the network (export happens on a background thread every 10 s).
    #[test]
    fn step_recording_overhead() {
        const N: u32 = 200_000;
        let bench = |t: &Telemetry| {
            for _ in 0..1000 {
                t.record_step(step(5));
            }
            let start = std::time::Instant::now();
            for i in 0..N {
                t.record_step(step(u64::from(i % 50)));
            }
            start.elapsed().as_nanos() as f64 / f64::from(N)
        };
        let cfg = TelemetryConfig::default();
        let off = bench(&Telemetry::new(&cfg).unwrap());
        let mp = SdkMeterProvider::builder()
            .with_reader(
                opentelemetry_sdk::metrics::PeriodicReader::builder(
                    InMemoryMetricExporter::default(),
                )
                .build(),
            )
            .build();
        let on = bench(&Telemetry::from_parts(&cfg, mp, None, None));
        println!("record_step: exporter off {off:.0} ns/step, on {on:.0} ns/step");
        // A 10 ms decode step must not notice this; very generous bound to stay unflaky.
        assert!(off < 20_000.0 && on < 20_000.0);
    }

    /// What `Qwen3Moe::finish_step` records after each step, against the real OTLP exporter
    /// whose collector accepts connections and never answers, exporting every 20 ms: the export
    /// runs on the reader's thread, so no call on the step path waits for it.
    #[test]
    fn step_path_does_not_wait_for_a_hung_collector() {
        use std::net::TcpListener;
        use std::time::Instant;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let accepted = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let seen = Arc::clone(&accepted);
        std::thread::spawn(move || {
            // Hold every connection open without reading or replying.
            let mut held = Vec::new();
            for c in listener.incoming() {
                held.push(c);
                seen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        });
        let cfg = TelemetryConfig {
            otlp_endpoint: Some(format!("http://{addr}")),
            export_interval: Duration::from_millis(20),
            ..TelemetryConfig::default()
        };
        let t = Telemetry::new(&cfg).unwrap();
        let start = Instant::now();
        let mut worst = Duration::ZERO;
        let mut steps = 0u32;
        while start.elapsed() < Duration::from_millis(600) {
            let s = Instant::now();
            t.record_step(step(5));
            let m = t.metrics();
            m.memory_used("CUDA0", "vram", "weights", 1);
            m.memory_capacity("CUDA0", "vram", "weights", 2);
            m.kv_cache("CUDA0", "vram", "f16/f16", 3, 4, 5, 6);
            m.cpu_pool_pauses(1);
            worst = worst.max(s.elapsed());
            steps += 1;
            std::thread::sleep(Duration::from_millis(2));
        }
        println!("{steps} steps, worst step-path telemetry {worst:?}");
        // Exports start every 20 ms and hang for seconds; a blocking path would show that here.
        // The bound leaves room for a loaded CI runner.
        assert!(worst < Duration::from_millis(250), "{worst:?}");
        // The exporter did reach the collector, so an export was hanging meanwhile.
        assert!(accepted.load(std::sync::atomic::Ordering::Relaxed) > 0);
        // Shutdown waits for the hung export; leak it rather than slow the suite.
        std::mem::forget(t);
    }
}
