use opentelemetry::KeyValue;
use opentelemetry::metrics::{Counter, Gauge, Histogram, Meter};
use std::time::Duration;

/// Attribute names. `gen_ai.*` and `error.type` follow the GenAI semantic conventions.
pub mod attr {
    pub const MODEL: &str = "gen_ai.request.model";
    pub const OPERATION: &str = "gen_ai.operation.name";
    pub const ERROR_TYPE: &str = "error.type";
    pub const DIRECTION: &str = "tetraeder.token.direction";
    pub const CACHE: &str = "tetraeder.token.cache";
    pub const PHASE: &str = "tetraeder.phase";
    pub const STAGE: &str = "tetraeder.stage";
    pub const GRAPHS: &str = "tetraeder.graphs";
    pub const DEVICE: &str = "tetraeder.device";
    pub const TIER: &str = "tetraeder.memory.tier";
    pub const KIND: &str = "tetraeder.memory.kind";
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Prefill,
    Decode,
}

impl Phase {
    pub fn as_str(self) -> &'static str {
        match self {
            Phase::Prefill => "prefill",
            Phase::Decode => "decode",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Input,
    Output,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheResult {
    Hit,
    Miss,
}

/// One engine step: a prefill chunk or a decode step on one stage.
#[derive(Debug, Clone, Copy)]
pub struct StepSample {
    pub phase: Phase,
    pub stage: u16,
    pub graphs: bool,
    pub duration: Duration,
    pub tokens: u32,
}

/// Seconds, from 1 ms to 60 s.
const LATENCY_BOUNDS: [f64; 14] = [
    0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 60.0,
];

/// Typed instruments for the Phase 0/1 subset of the metric catalog (docs/plan.md).
#[derive(Debug, Clone)]
pub struct Metrics {
    ttft: Histogram<f64>,
    tpot: Histogram<f64>,
    request_duration: Histogram<f64>,
    tokens: Counter<u64>,
    step_duration: Histogram<f64>,
    device_busy: Gauge<f64>,
    device_power: Gauge<f64>,
    device_temp: Gauge<f64>,
    mem_used: Gauge<u64>,
    mem_capacity: Gauge<u64>,
}

impl Metrics {
    pub fn new(meter: &Meter) -> Self {
        let hist = |name: &'static str, desc: &'static str| {
            meter
                .f64_histogram(name)
                .with_unit("s")
                .with_description(desc)
                .with_boundaries(LATENCY_BOUNDS.to_vec())
                .build()
        };
        Self {
            ttft: hist(
                "gen_ai.server.time_to_first_token",
                "Time to generate the first token",
            ),
            tpot: hist(
                "gen_ai.server.time_per_output_token",
                "Time per output token after the first",
            ),
            request_duration: hist("gen_ai.server.request.duration", "Server request duration"),
            tokens: meter
                .u64_counter("tetraeder.tokens")
                .with_unit("{token}")
                .build(),
            step_duration: hist("tetraeder.step.duration", "Duration of one engine step"),
            device_busy: meter
                .f64_gauge("tetraeder.device.busy")
                .with_unit("1")
                .with_description("Busy fraction, 0 to 1")
                .build(),
            device_power: meter
                .f64_gauge("tetraeder.device.power")
                .with_unit("W")
                .build(),
            device_temp: meter
                .f64_gauge("tetraeder.device.temperature")
                .with_unit("Cel")
                .build(),
            mem_used: meter
                .u64_gauge("tetraeder.memory.used")
                .with_unit("By")
                .build(),
            mem_capacity: meter
                .u64_gauge("tetraeder.memory.capacity")
                .with_unit("By")
                .build(),
        }
    }

    pub fn time_to_first_token(&self, d: Duration, model: &str, operation: &str) {
        self.ttft.record(
            d.as_secs_f64(),
            &[
                KeyValue::new(attr::MODEL, model.to_owned()),
                KeyValue::new(attr::OPERATION, operation.to_owned()),
            ],
        );
    }

    pub fn time_per_output_token(&self, d: Duration, model: &str) {
        self.tpot.record(
            d.as_secs_f64(),
            &[KeyValue::new(attr::MODEL, model.to_owned())],
        );
    }

    /// `error_type` is `None` on success.
    pub fn request_duration(&self, d: Duration, model: &str, error_type: Option<&str>) {
        let mut a = vec![KeyValue::new(attr::MODEL, model.to_owned())];
        if let Some(e) = error_type {
            a.push(KeyValue::new(attr::ERROR_TYPE, e.to_owned()));
        }
        self.request_duration.record(d.as_secs_f64(), &a);
    }

    pub fn tokens(&self, n: u64, model: &str, dir: Direction, cache: CacheResult) {
        self.tokens.add(
            n,
            &[
                KeyValue::new(attr::MODEL, model.to_owned()),
                KeyValue::new(
                    attr::DIRECTION,
                    match dir {
                        Direction::Input => "input",
                        Direction::Output => "output",
                    },
                ),
                KeyValue::new(
                    attr::CACHE,
                    match cache {
                        CacheResult::Hit => "hit",
                        CacheResult::Miss => "miss",
                    },
                ),
            ],
        );
    }

    pub fn step_duration(&self, s: &StepSample) {
        self.step_duration.record(
            s.duration.as_secs_f64(),
            &[
                KeyValue::new(attr::PHASE, s.phase.as_str()),
                KeyValue::new(attr::STAGE, i64::from(s.stage)),
                KeyValue::new(attr::GRAPHS, if s.graphs { "on" } else { "off" }),
            ],
        );
    }

    pub fn device_busy(&self, device: &str, fraction: f64) {
        self.device_busy
            .record(fraction, &[KeyValue::new(attr::DEVICE, device.to_owned())]);
    }

    pub fn device_power(&self, device: &str, watts: f64) {
        self.device_power
            .record(watts, &[KeyValue::new(attr::DEVICE, device.to_owned())]);
    }

    pub fn device_temperature(&self, device: &str, celsius: f64) {
        self.device_temp
            .record(celsius, &[KeyValue::new(attr::DEVICE, device.to_owned())]);
    }

    /// `tier` is for example `vram` or `ram`; `kind` is `weights`, `kv`, `workspace`,
    /// `reserve`, or `total` when the split is unknown.
    pub fn memory_used(&self, device: &str, tier: &str, kind: &str, bytes: u64) {
        self.mem_used.record(bytes, &mem_attrs(device, tier, kind));
    }

    pub fn memory_capacity(&self, device: &str, tier: &str, kind: &str, bytes: u64) {
        self.mem_capacity
            .record(bytes, &mem_attrs(device, tier, kind));
    }
}

fn mem_attrs(device: &str, tier: &str, kind: &str) -> [KeyValue; 3] {
    [
        KeyValue::new(attr::DEVICE, device.to_owned()),
        KeyValue::new(attr::TIER, tier.to_owned()),
        KeyValue::new(attr::KIND, kind.to_owned()),
    ]
}
