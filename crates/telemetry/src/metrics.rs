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
    pub const LAYER: &str = "tetraeder.layer";
    pub const RESULT: &str = "tetraeder.result";
    pub const SOURCE: &str = "tetraeder.source";
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
    /// Graph segments the step ran: 1 for a whole-graph step, `n_layer + 1` for the builder's
    /// per-MoE-layer segments (ADR 0030).
    pub segments: u32,
}

/// Seconds, from 1 ms to 60 s.
const LATENCY_BOUNDS: [f64; 14] = [
    0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 60.0,
];

/// Typed instruments for the Phase 0 to 2 subset of the metric catalog (docs/plan.md).
#[derive(Debug, Clone)]
pub struct Metrics {
    ttft: Histogram<f64>,
    tpot: Histogram<f64>,
    request_duration: Histogram<f64>,
    tokens: Counter<u64>,
    step_duration: Histogram<f64>,
    step_segments: Histogram<f64>,
    device_busy: Gauge<f64>,
    device_power: Gauge<f64>,
    device_temp: Gauge<f64>,
    mem_used: Gauge<u64>,
    mem_capacity: Gauge<u64>,
    expert_requests: Counter<u64>,
    expert_load: Histogram<f64>,
    tier_read: Counter<u64>,
    layer_hit: Counter<u64>,
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
            step_segments: meter
                .f64_histogram("tetraeder.step.segments")
                .with_unit("{segment}")
                .with_description("Graph segments per engine step")
                .with_boundaries(vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0])
                .build(),
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
            expert_requests: meter
                .u64_counter("tetraeder.expert.cache.requests")
                .with_unit("{request}")
                .with_description("Expert slot lookups by layer, tier and result")
                .build(),
            expert_load: hist(
                "tetraeder.expert.load.duration",
                "Time to bring one expert into a slot",
            ),
            tier_read: meter
                .u64_counter("tetraeder.tier.read.bytes")
                .with_unit("By")
                .with_description("Bytes read from a memory tier")
                .build(),
            layer_hit: meter
                .u64_counter("tetraeder.expert.layer_hit")
                .with_unit("{layer}")
                .with_description("MoE layer steps with all top-k experts resident (full) or not")
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
                KeyValue::new(attr::CACHE, cache_result(cache)),
            ],
        );
    }

    pub fn step_duration(&self, s: &StepSample) {
        let attrs = [
            KeyValue::new(attr::PHASE, s.phase.as_str()),
            KeyValue::new(attr::STAGE, i64::from(s.stage)),
            KeyValue::new(attr::GRAPHS, if s.graphs { "on" } else { "off" }),
        ];
        self.step_duration.record(s.duration.as_secs_f64(), &attrs);
        self.step_segments.record(f64::from(s.segments), &attrs);
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

    /// `tier` is for example `vram` or `ram`; `kind` is `weights`, `kv`, `workspace`, `slots`,
    /// `reserve`, or `total` when the split is unknown.
    pub fn memory_used(&self, device: &str, tier: &str, kind: &str, bytes: u64) {
        self.mem_used.record(bytes, &mem_attrs(device, tier, kind));
    }

    /// Expert slot lookups in the cache of `tier` for a MoE `layer`.
    pub fn expert_cache_requests(&self, n: u64, layer: u32, tier: &str, result: CacheResult) {
        self.expert_requests.add(
            n,
            &[
                KeyValue::new(attr::LAYER, i64::from(layer)),
                KeyValue::new(attr::TIER, tier.to_owned()),
                KeyValue::new(attr::RESULT, cache_result(result)),
            ],
        );
    }

    /// `source` is where the bytes came from, for example `nvme` or `ram`.
    pub fn expert_load_duration(&self, d: Duration, tier: &str, source: &str) {
        self.expert_load.record(
            d.as_secs_f64(),
            &[
                KeyValue::new(attr::TIER, tier.to_owned()),
                KeyValue::new(attr::SOURCE, source.to_owned()),
            ],
        );
    }

    pub fn tier_read_bytes(&self, bytes: u64, device: &str, tier: &str) {
        self.tier_read.add(
            bytes,
            &[
                KeyValue::new(attr::DEVICE, device.to_owned()),
                KeyValue::new(attr::TIER, tier.to_owned()),
            ],
        );
    }

    /// A MoE layer step: `full` when every top-k expert was resident, else partial.
    pub fn expert_layer_hit(&self, full: bool) {
        self.layer_hit.add(
            1,
            &[KeyValue::new(
                attr::RESULT,
                if full { "full" } else { "partial" },
            )],
        );
    }

    pub fn memory_capacity(&self, device: &str, tier: &str, kind: &str, bytes: u64) {
        self.mem_capacity
            .record(bytes, &mem_attrs(device, tier, kind));
    }
}

fn cache_result(r: CacheResult) -> &'static str {
    match r {
        CacheResult::Hit => "hit",
        CacheResult::Miss => "miss",
    }
}

fn mem_attrs(device: &str, tier: &str, kind: &str) -> [KeyValue; 3] {
    [
        KeyValue::new(attr::DEVICE, device.to_owned()),
        KeyValue::new(attr::TIER, tier.to_owned()),
        KeyValue::new(attr::KIND, kind.to_owned()),
    ]
}
