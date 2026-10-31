//! Expert slots (ADR 0033): the builder with fewer slots than experts, with and without a RAM
//! tier, against libllama, against an independent LRU simulation, and on read failures.
//!
//! The fixture has 8 experts per layer and routes the top 2, so 1 slot cannot hold even one
//! token's experts: that case runs the overflow passes of ADR 0033.
//!
//! Hardware label: all tests are `cpu_*`.

mod common;

use common::*;
use std::sync::Arc;
use tetraeder_model::{BackendChoice, LoadOptions, ModelError, Qwen3Moe};

const CPU: BackendChoice = BackendChoice::Cpu { threads: Some(2) };
const EXPERTS: usize = 8;

fn opts(slots: usize, ram: Option<usize>) -> LoadOptions {
    LoadOptions {
        expert_slots: Some(slots),
        ram_slots: ram,
        record_routing: true,
        ..LoadOptions::new(CPU)
    }
}

/// hardware: cpu. Slot counts 8 (all), 4, 2 and 1 per layer, each with no RAM tier and with
/// RAM tiers of 1, 3 and 8 slots: prefill logits and the greedy sequence match libllama.
#[test]
fn cpu_slot_counts_match_libllama() {
    let r = libllama_reference(&PROMPT, N_GREEDY);
    for slots in [8, 4, 2, 1] {
        for ram in [None, Some(1), Some(3), Some(8)] {
            let mut m = Qwen3Moe::load(fixture(), &opts(slots, ram)).unwrap();
            let worst = check_against(&mut m, &r, TOL_CPU);
            eprintln!("slots {slots} ram {ram:?}: max |delta| {worst:e}");
            let layers = m.hparams().n_layer as u64;
            assert_eq!(m.segments_run(), (layers + 1) * (2 + N_GREEDY as u64));
            if slots == EXPERTS {
                assert!(m.expert_stats().is_none(), "all experts resident: no cache");
                assert_eq!(m.overflow_passes(), 0);
            } else {
                let stats = m.expert_stats().expect("streaming");
                assert!(stats.layers.iter().all(|l| l.compute.misses > 0));
                assert_eq!(stats.layers[0].ram.is_some(), ram.is_some());
                assert!(stats.io.experts_read > 0);
            }
            if slots == 1 {
                assert!(m.overflow_passes() > 0, "top-2 does not fit one slot");
            }
        }
    }
}

/// hardware: cpu. The default (every expert resident) and an explicit slot count at or above the
/// expert count are the same load path and give identical logits.
#[test]
fn cpu_default_slots_reproduce_the_resident_model() {
    let eval = |o: LoadOptions| {
        Qwen3Moe::load(fixture(), &o)
            .unwrap()
            .eval(&PROMPT)
            .unwrap()
    };
    let base = eval(LoadOptions::new(CPU));
    assert_eq!(base, eval(opts(EXPERTS, Some(2))));
    assert_eq!(base, eval(opts(64, None)));
    assert!(Qwen3Moe::load(fixture(), &opts(0, None)).is_err());
}

/// Least recently used over a list, written without `LayerSlots`.
struct Lru {
    cap: usize,
    /// Least recent first.
    items: Vec<i32>,
}

impl Lru {
    fn new(cap: usize) -> Self {
        Self {
            cap,
            items: Vec::new(),
        }
    }
    fn has(&self, e: i32) -> bool {
        self.items.contains(&e)
    }
    /// True on a hit.
    fn access(&mut self, e: i32) -> bool {
        if let Some(p) = self.items.iter().position(|&x| x == e) {
            self.items.remove(p);
            self.items.push(e);
            true
        } else {
            if self.items.len() == self.cap {
                self.items.remove(0);
            }
            self.items.push(e);
            false
        }
    }
}

#[derive(Default, Debug, PartialEq, Eq, Clone, Copy)]
struct Counts {
    hits: u64,
    misses: u64,
}

/// What ADR 0033 says one layer step does, as plain code: distinct experts in order of first
/// appearance, those already resident first, chunks of the slot count; per chunk the compute
/// lookups, then the RAM tier for the chunk's misses in chunks of its own slot count (its hits
/// first), then one disk batch for the RAM misses.
struct Sim {
    slots: usize,
    ram_slots: Option<usize>,
    compute: Vec<Lru>,
    ram: Vec<Lru>,
    c: Vec<Counts>,
    r: Vec<Counts>,
    full: Vec<u64>,
    partial: Vec<u64>,
    experts_read: u64,
}

impl Sim {
    fn new(layers: usize, slots: usize, ram_slots: Option<usize>) -> Self {
        Self {
            slots,
            ram_slots,
            compute: (0..layers).map(|_| Lru::new(slots)).collect(),
            ram: (0..layers)
                .map(|_| Lru::new(ram_slots.unwrap_or(1)))
                .collect(),
            c: vec![Counts::default(); layers],
            r: vec![Counts::default(); layers],
            full: vec![0; layers],
            partial: vec![0; layers],
            experts_read: 0,
        }
    }

    fn step(&mut self, il: usize, chosen: &[i32]) {
        let mut distinct: Vec<i32> = Vec::new();
        for &e in chosen {
            if !distinct.contains(&e) {
                distinct.push(e);
            }
        }
        let (hits, misses): (Vec<i32>, Vec<i32>) =
            distinct.iter().partition(|&&e| self.compute[il].has(e));
        if misses.is_empty() {
            self.full[il] += 1;
        } else {
            self.partial[il] += 1;
        }
        let order: Vec<i32> = hits.into_iter().chain(misses).collect();
        for chunk in order.chunks(self.slots) {
            let mut loads = Vec::new();
            for &e in chunk {
                if self.compute[il].access(e) {
                    self.c[il].hits += 1;
                } else {
                    self.c[il].misses += 1;
                    loads.push(e);
                }
            }
            match self.ram_slots {
                None => self.experts_read += loads.len() as u64,
                Some(n) => {
                    for group in loads.chunks(n) {
                        let (rh, rm): (Vec<i32>, Vec<i32>) =
                            group.iter().partition(|&&e| self.ram[il].has(e));
                        for e in rh.into_iter().chain(rm) {
                            if self.ram[il].access(e) {
                                self.r[il].hits += 1;
                            } else {
                                self.r[il].misses += 1;
                                self.experts_read += 1;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// hardware: cpu. Hit and miss counts of both tiers, full and partial layer steps and the number
/// of experts read from the file equal the simulation fed with the routing the model recorded.
#[test]
fn cpu_cache_counts_match_an_lru_simulation() {
    for (slots, ram) in [
        (4, None),
        (2, None),
        (1, None),
        (4, Some(3)),
        (2, Some(2)),
        (3, Some(5)),
        (1, Some(2)),
        (2, Some(8)),
    ] {
        let mut m = Qwen3Moe::load(fixture(), &opts(slots, ram)).unwrap();
        let layers = m.hparams().n_layer as usize;
        let mut next = argmax(&m.eval_last(&PROMPT).unwrap()) as i32;
        for _ in 0..N_GREEDY {
            next = argmax(&m.eval(&[next]).unwrap()) as i32;
        }
        let log = m.take_routing_log();
        assert_eq!(log.len(), layers * (1 + N_GREEDY));
        let mut sim = Sim::new(layers, slots, ram);
        for step in &log {
            sim.step(step.layer, &step.ids);
        }
        let stats = m.expert_stats().unwrap();
        for il in 0..layers {
            let l = &stats.layers[il];
            let got = Counts {
                hits: l.compute.hits,
                misses: l.compute.misses,
            };
            assert_eq!(
                got, sim.c[il],
                "slots {slots} ram {ram:?} layer {il}: compute"
            );
            assert_eq!(l.compute.bypass, 0);
            if ram.is_some() {
                let r = l.ram.unwrap();
                let got = Counts {
                    hits: r.hits,
                    misses: r.misses,
                };
                assert_eq!(got, sim.r[il], "slots {slots} ram {ram:?} layer {il}: ram");
            }
            assert_eq!(
                (l.full_steps, l.partial_steps),
                (sim.full[il], sim.partial[il])
            );
        }
        assert_eq!(
            stats.io.experts_read, sim.experts_read,
            "slots {slots} ram {ram:?}"
        );
        assert!(stats.io.bytes_read > 0);
        // One read per layer step at most per chunk.
        assert!(stats.io.batches <= stats.io.experts_read);
    }
}

/// hardware: cpu. A read that fails in the middle of a step is an error, not a step with the
/// wrong weights, and the slots it had assigned are forgotten: after the file is back the same
/// model reproduces the reference.
#[test]
fn cpu_read_failure_is_an_error_and_leaves_no_stale_slot() {
    let r = libllama_reference(&PROMPT, N_GREEDY);
    let path = std::env::temp_dir().join(format!("tetraeder-slots-{}.gguf", std::process::id()));
    let good = std::fs::read(fixture()).unwrap();
    std::fs::write(&path, &good).unwrap();
    for ram in [None, Some(2)] {
        let mut m = Qwen3Moe::load(&path, &opts(2, ram)).unwrap();
        // Cut the file after loading: the expert reads now fail.
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(0)
            .unwrap();
        let err = m.eval(&PROMPT).unwrap_err();
        assert!(matches!(err, ModelError::Io(_)), "{err:?}");
        assert_eq!(m.n_past(), 0, "a failed step leaves the cache untouched");
        // Put the bytes back; stale slot assignments would now serve zeros as experts.
        std::fs::write(&path, &good).unwrap();
        check_against(&mut m, &r, TOL_CPU);
    }
    std::fs::remove_file(&path).ok();
}

/// hardware: cpu. The step histogram carries segments per step and the cache metrics agree
/// with the counters of the model.
#[test]
fn cpu_telemetry_reports_steps_and_cache() {
    use opentelemetry::KeyValue;
    use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
    use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};
    use tetraeder_telemetry::{Telemetry, TelemetryConfig};

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
        telemetry: Some(t.clone()),
        ..opts(2, Some(3))
    };
    let mut m = Qwen3Moe::load(fixture(), &o).unwrap();
    m.eval(&PROMPT).unwrap();
    m.eval(&[7]).unwrap();
    mp.force_flush().unwrap();
    let layers = m.hparams().n_layer as u64;
    let stats = m.expert_stats().unwrap();

    fn attr<'a>(mut kv: impl Iterator<Item = &'a KeyValue>, k: &str) -> String {
        kv.find(|kv| kv.key.as_str() == k)
            .map(|kv| kv.value.to_string())
            .unwrap_or_default()
    }
    let mut requests = std::collections::BTreeMap::<(String, String), u64>::new();
    let (mut layer_hits, mut read_bytes) = (0u64, 0u64);
    let (mut steps, mut segments) = (0u64, 0.0f64);
    for rm in exp.get_finished_metrics().unwrap() {
        for sm in rm.scope_metrics() {
            for metric in sm.metrics() {
                match (metric.name(), metric.data()) {
                    (
                        "tetraeder.expert.cache.requests",
                        AggregatedMetrics::U64(MetricData::Sum(s)),
                    ) => {
                        for dp in s.data_points() {
                            let tier = attr(dp.attributes(), "tetraeder.memory.tier");
                            let res = attr(dp.attributes(), "tetraeder.result");
                            *requests.entry((tier, res)).or_default() += dp.value();
                        }
                    }
                    ("tetraeder.expert.layer_hit", AggregatedMetrics::U64(MetricData::Sum(s))) => {
                        layer_hits += s.data_points().map(|d| d.value()).sum::<u64>();
                    }
                    ("tetraeder.tier.read.bytes", AggregatedMetrics::U64(MetricData::Sum(s))) => {
                        for dp in s.data_points() {
                            if attr(dp.attributes(), "tetraeder.memory.tier") == "nvme" {
                                read_bytes += dp.value();
                            }
                        }
                    }
                    (
                        "tetraeder.step.duration",
                        AggregatedMetrics::F64(MetricData::Histogram(h)),
                    ) => {
                        steps += h.data_points().map(|d| d.count()).sum::<u64>();
                    }
                    (
                        "tetraeder.step.segments",
                        AggregatedMetrics::F64(MetricData::Histogram(h)),
                    ) => {
                        segments += h.data_points().map(|d| d.sum()).sum::<f64>();
                    }
                    _ => {}
                }
            }
        }
    }
    assert_eq!(steps, 2);
    assert_eq!(segments as u64, 2 * (layers + 1));
    assert_eq!(layer_hits, 2 * layers);
    assert_eq!(read_bytes, stats.io.bytes_read);
    let sum = |tier: &str, res: &str| {
        requests
            .get(&(tier.into(), res.into()))
            .copied()
            .unwrap_or(0)
    };
    let (h, mi) = stats.layers.iter().fold((0, 0), |a, l| {
        (a.0 + l.compute.hits, a.1 + l.compute.misses)
    });
    assert_eq!((sum("vram", "hit"), sum("vram", "miss")), (h, mi));
    let (h, mi) = stats.layers.iter().fold((0, 0), |a, l| {
        let r = l.ram.unwrap();
        (a.0 + r.hits, a.1 + r.misses)
    });
    assert_eq!((sum("ram", "hit"), sum("ram", "miss")), (h, mi));
}
