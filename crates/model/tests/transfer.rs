//! `tetraeder.tier.transfer.bytes` and `.duration` (ADR 0063) from the expert paths: a miss read
//! is disk to RAM (or disk to VRAM without a RAM tier), a prefetch read is disk to RAM with
//! purpose `prefetch`, an admission upload is RAM to VRAM with purpose `admission`. Each series
//! adds up to the model's own byte counters, and `tetraeder.tier.read.bytes` is unchanged.
//!
//! Hardware label: cpu.

mod common;

use common::argmax;
use common::synth::{MoeSpec, quantized_moe};
use opentelemetry::KeyValue;
use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};
use std::collections::BTreeMap;
use std::sync::Arc;
use tetraeder_ggml_sys::llama::Ftype;
use tetraeder_model::{BackendChoice, ExpertStats, LoadOptions, MissPolicy, Qwen3Moe};
use tetraeder_telemetry::{Telemetry, TelemetryConfig};

const CPU: BackendChoice = BackendChoice::Cpu { threads: Some(2) };
const N_PROMPT: usize = 24;
const N_DECODE: usize = 6;
const SLOTS: usize = 8;

/// (source, destination, purpose).
type Key = (String, String, String);

#[derive(Default)]
struct Transfers {
    bytes: BTreeMap<Key, u64>,
    timed: BTreeMap<Key, u64>,
    /// `tetraeder.tier.read.bytes` of tier `nvme`.
    read: u64,
}

impl Transfers {
    fn bytes(&self, src: &str, dst: &str, purpose: &str) -> u64 {
        let k = (src.to_owned(), dst.to_owned(), purpose.to_owned());
        self.bytes.get(&k).copied().unwrap_or(0)
    }

    fn timed(&self, src: &str, dst: &str, purpose: &str) -> u64 {
        let k = (src.to_owned(), dst.to_owned(), purpose.to_owned());
        self.timed.get(&k).copied().unwrap_or(0)
    }
}

fn attr<'a>(mut kv: impl Iterator<Item = &'a KeyValue>, k: &str) -> String {
    kv.find(|kv| kv.key.as_str() == k)
        .map(|kv| kv.value.to_string())
        .unwrap_or_default()
}

fn key<'a>(kv: impl Iterator<Item = &'a KeyValue> + Clone) -> Key {
    (
        attr(kv.clone(), "tetraeder.transfer.source"),
        attr(kv.clone(), "tetraeder.transfer.destination"),
        attr(kv, "tetraeder.transfer.purpose"),
    )
}

/// Loads with telemetry, prefills and decodes, and returns the transfer series with the stats.
fn run(path: &std::path::Path, base: LoadOptions) -> (Transfers, ExpertStats) {
    let exp = InMemoryMetricExporter::default();
    let mp = SdkMeterProvider::builder()
        .with_reader(PeriodicReader::builder(exp.clone()).build())
        .build();
    let t = Arc::new(Telemetry::from_parts(
        &TelemetryConfig::default(),
        mp.clone(),
        None,
        None,
    ));
    let o = LoadOptions {
        telemetry: Some(t),
        ..base
    };
    let mut m = Qwen3Moe::load(path, &o).unwrap();
    let prompt: Vec<i32> = (0..N_PROMPT as i32).map(|i| (i * 37 + 5) % 256).collect();
    let l = m.eval(&prompt).unwrap();
    let mut next = argmax(&l[(N_PROMPT - 1) * m.n_vocab()..]) as i32;
    for _ in 0..N_DECODE {
        let l = m.eval(&[next]).unwrap();
        next = argmax(&l) as i32;
    }
    mp.force_flush().unwrap();
    let mut out = Transfers::default();
    for rm in exp.get_finished_metrics().unwrap() {
        for sm in rm.scope_metrics() {
            for metric in sm.metrics() {
                match (metric.name(), metric.data()) {
                    (
                        "tetraeder.tier.transfer.bytes",
                        AggregatedMetrics::U64(MetricData::Sum(s)),
                    ) => {
                        for dp in s.data_points() {
                            *out.bytes.entry(key(dp.attributes())).or_default() += dp.value();
                        }
                    }
                    (
                        "tetraeder.tier.transfer.duration",
                        AggregatedMetrics::F64(MetricData::Histogram(h)),
                    ) => {
                        for dp in h.data_points() {
                            *out.timed.entry(key(dp.attributes())).or_default() += dp.count();
                        }
                    }
                    ("tetraeder.tier.read.bytes", AggregatedMetrics::U64(MetricData::Sum(s))) => {
                        for dp in s.data_points() {
                            if attr(dp.attributes(), "tetraeder.memory.tier") == "nvme" {
                                out.read += dp.value();
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    (out, m.expert_stats().unwrap())
}

fn fixture() -> std::path::PathBuf {
    quantized_moe(&MoeSpec::REALISH, Ftype::Q4KM, "q4km-transfer")
}

fn opts(ram: Option<usize>, prefetch: bool, admit: Option<f64>) -> LoadOptions {
    LoadOptions {
        expert_slots: Some(SLOTS),
        ram_slots: ram,
        miss: MissPolicy::Cpu,
        prefetch,
        admit_share: admit,
        ..LoadOptions::new(CPU)
    }
}

/// hardware: cpu. With a RAM tier and no prefetch, every file read is a miss read into RAM, and
/// the RAM hot set is uploaded to the compute slots.
#[test]
fn cpu_miss_reads_are_disk_to_ram() {
    let (t, s) = run(&fixture(), opts(Some(12), false, None));
    assert!(s.io.bytes_read > 0);
    assert_eq!(t.bytes("disk", "ram", "expert_miss"), s.io.bytes_read);
    assert!(t.timed("disk", "ram", "expert_miss") > 0);
    assert_eq!(t.bytes("disk", "ram", "prefetch"), 0);
    assert!(t.bytes("ram", "vram", "expert_miss") > 0);
    assert_eq!(t.bytes("ram", "vram", "admission"), 0);
    assert_eq!(t.read, s.io.bytes_read, "tier.read.bytes is unchanged");
}

/// hardware: cpu. Without a RAM tier the file is read straight into the compute slots.
#[test]
fn cpu_miss_reads_without_ram_are_disk_to_vram() {
    let o = LoadOptions {
        miss: MissPolicy::Upload,
        ..opts(None, false, None)
    };
    let (t, s) = run(&fixture(), o);
    assert!(s.io.bytes_read > 0);
    assert_eq!(t.bytes("disk", "vram", "expert_miss"), s.io.bytes_read);
    assert!(t.timed("disk", "vram", "expert_miss") > 0);
    assert_eq!(t.bytes("disk", "ram", "expert_miss"), 0);
    assert_eq!(t.read, s.io.bytes_read);
}

/// hardware: cpu. Prefetch reads are disk to RAM with purpose `prefetch`; they add up to the
/// model's prefetch bytes and the miss reads to the rest.
#[test]
fn cpu_prefetch_reads_are_disk_to_ram_prefetch() {
    let (t, s) = run(&fixture(), opts(Some(12), true, None));
    assert!(s.io.prefetch_bytes > 0, "something was prefetched");
    assert_eq!(t.bytes("disk", "ram", "prefetch"), s.io.prefetch_bytes);
    assert!(t.timed("disk", "ram", "prefetch") > 0);
    assert_eq!(
        t.bytes("disk", "ram", "expert_miss"),
        s.io.bytes_read - s.io.prefetch_bytes
    );
    assert_eq!(t.read, s.io.bytes_read);
}

/// hardware: cpu. A decode admission uploads one expert from its RAM slot to the compute slot:
/// RAM to VRAM with purpose `admission`, as many bytes as the model counts.
#[test]
fn cpu_admissions_are_ram_to_vram_admission() {
    let (t, s) = run(&fixture(), opts(Some(12), false, Some(1.0)));
    assert!(s.io.admitted_bytes > 0, "decode admitted experts");
    assert_eq!(t.bytes("ram", "vram", "admission"), s.io.admitted_bytes);
    assert!(t.timed("ram", "vram", "admission") >= s.io.admitted);
}
