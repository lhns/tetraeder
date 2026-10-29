//! Spike (ADR 0022): does opentelemetry-rust attach exemplars to histograms recorded
//! inside a sampled span? As of opentelemetry_sdk 0.33 it does not: the data model and the
//! OTLP mapping exist, but the aggregators always produce empty exemplar lists. If this
//! test starts failing after an upgrade, exemplars landed: revisit the ADR.

use opentelemetry::metrics::MeterProvider as _;
use opentelemetry::trace::{TraceContextExt, Tracer, TracerProvider as _};
use opentelemetry::{Context, KeyValue};
use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};
use opentelemetry_sdk::trace::SdkTracerProvider;

#[test]
fn histogram_in_sampled_span_has_no_exemplars() {
    let exp = InMemoryMetricExporter::default();
    let mp = SdkMeterProvider::builder()
        .with_reader(PeriodicReader::builder(exp.clone()).build())
        .build();
    let tp = SdkTracerProvider::builder().build(); // default sampler: always on
    let tracer = tp.tracer("spike");
    let hist = mp.meter("spike").f64_histogram("spike.duration").build();

    let span = tracer.start("slow request");
    let cx = Context::current_with_span(span);
    assert!(cx.span().span_context().is_sampled());
    {
        let _g = cx.clone().attach();
        for _ in 0..100 {
            hist.record(1.5, &[KeyValue::new("k", "v")]);
        }
    }
    mp.force_flush().unwrap();

    let mut points = 0;
    let mut exemplars = 0;
    for rm in exp.get_finished_metrics().unwrap() {
        for sm in rm.scope_metrics() {
            for m in sm.metrics() {
                if let AggregatedMetrics::F64(MetricData::Histogram(h)) = m.data() {
                    for dp in h.data_points() {
                        points += 1;
                        exemplars += dp.exemplars().count();
                    }
                }
            }
        }
    }
    assert_eq!(points, 1);
    assert_eq!(exemplars, 0, "SDK now produces exemplars; update ADR 0022");
}
