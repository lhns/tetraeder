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
    pub const DEPTH: &str = "tetraeder.prefetch.depth";
    pub const RESULT: &str = "tetraeder.result";
    pub const SOURCE: &str = "tetraeder.source";
    pub const FORMAT: &str = "tetraeder.format";
    pub const IO_CLASS: &str = "tetraeder.io.class";
    pub const PART: &str = "tetraeder.part";
    pub const METHOD: &str = "tetraeder.spec.method";
    pub const PLAN: &str = "tetraeder.plan";
    pub const RESOURCE: &str = "tetraeder.resource";
    pub const FAULT_CLASS: &str = "tetraeder.fault.class";
    pub const REPLAN_TRIGGER: &str = "tetraeder.replan.trigger";
    pub const REPLAN_OUTCOME: &str = "tetraeder.replan.outcome";
    pub const LOAD_OUTCOME: &str = "tetraeder.load.outcome";
    pub const CPU_TIER: &str = "tetraeder.cpu.tier";
    pub const CPU_ISA: &str = "tetraeder.cpu.isa";
    pub const CPU_DISPATCH: &str = "tetraeder.cpu.dispatch";
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
    /// Expert caches only: not resident, computed on the CPU instead of being loaded (ADR 0035).
    Cpu,
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
    /// per-MoE-layer segments (ADR 0030), plus the standalone expert group graphs of the step (ADR 0035).
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
    kv_tokens: Gauge<u64>,
    kv_capacity: Gauge<u64>,
    expert_requests: Counter<u64>,
    expert_load: Histogram<f64>,
    tier_read: Counter<u64>,
    layer_hit: Counter<u64>,
    prefetch: Counter<u64>,
    prefetch_wasted_bytes: Counter<u64>,
    prefetch_accuracy: Histogram<f64>,
    io_inflight: Histogram<f64>,
    read_hidden: Histogram<f64>,
    layer_part: Histogram<f64>,
    convert_duration: Histogram<f64>,
    convert_bytes: Counter<u64>,
    convert_cache: Counter<u64>,
    batch_size: Histogram<f64>,
    decode_stage: Histogram<f64>,
    prefix_hits: Counter<u64>,
    prefix_evictions: Counter<u64>,
    prefix_bytes: Gauge<u64>,
    prefix_restore: Histogram<f64>,
    spec_acceptance: Histogram<f64>,
    spec_drafted: Counter<u64>,
    spec_accepted: Counter<u64>,
    plan_residual: Histogram<f64>,
    plan_regret: Gauge<f64>,
    plan_predicted_step: Gauge<f64>,
    bottleneck_share: Gauge<f64>,
    bottleneck_fault: Counter<u64>,
    replan_total: Counter<u64>,
    replan_duration: Histogram<f64>,
    engine_load: Histogram<f64>,
    replan_admitting: Gauge<u64>,
    cpu_variant: Gauge<u64>,
    cpu_isa_missing: Gauge<u64>,
    cpu_threads: Gauge<u64>,
    cpu_quota: Gauge<f64>,
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
            kv_tokens: meter
                .u64_gauge("tetraeder.kv.tokens")
                .with_description("Positions held in the KV cache")
                .build(),
            kv_capacity: meter
                .u64_gauge("tetraeder.kv.capacity")
                .with_description("Positions the KV cache holds")
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
            prefetch: meter
                .u64_counter("tetraeder.expert.prefetch")
                .with_unit("{expert}")
                .with_description("Experts read ahead into the RAM tier by layer and result")
                .build(),
            prefetch_wasted_bytes: meter
                .u64_counter("tetraeder.expert.prefetch.wasted")
                .with_unit("By")
                .with_description("Bytes of prefetched experts evicted before they were routed")
                .build(),
            prefetch_accuracy: meter
                .f64_histogram("tetraeder.expert.prefetch.accuracy")
                .with_unit("1")
                .with_description("Share of predicted experts the router chose, per layer step")
                .with_boundaries(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
                .build(),
            io_inflight: meter
                .f64_histogram("tetraeder.io.inflight")
                .with_unit("{chunk}")
                .with_description("Expert read chunks queued or running in the I/O service, at each submit")
                .with_boundaries(vec![0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 64.0])
                .build(),
            read_hidden: meter
                .f64_histogram("tetraeder.moe.read.hidden")
                .with_unit("1")
                .with_description(
                    "Share of the expert read span of a decode layer step the main thread did not wait for",
                )
                .with_boundaries(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
                .build(),
            layer_part: hist(
                "tetraeder.moe.layer.duration",
                "Time of one part of a decode layer step with expert reads in flight",
            ),
            convert_duration: meter
                .f64_histogram("tetraeder.convert.duration")
                .with_unit("s")
                .with_description("Time to convert one tensor to its target format")
                .with_boundaries(vec![
                    0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
                ])
                .build(),
            convert_bytes: meter
                .u64_counter("tetraeder.convert.bytes")
                .with_unit("By")
                .with_description("Bytes written to the conversion cache")
                .build(),
            convert_cache: meter
                .u64_counter("tetraeder.convert.cache")
                .with_unit("{tensor}")
                .with_description(
                    "Converted tensors found in the conversion cache (hit) or converted (miss)",
                )
                .build(),
            decode_stage: meter
                .f64_histogram("tetraeder.scheduler.decode.stage")
                .with_unit("s")
                .with_description(
                    "Mean host time per decoded token of one request, by stage (eval, sample, emit); only with TETRAEDER_DECODE_PROFILE=1",
                )
                .with_boundaries(vec![
                    1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1,
                ])
                .build(),
            batch_size: meter
                .f64_histogram("tetraeder.scheduler.batch.size")
                .with_unit("{request}")
                .with_description("Requests decoded together per scheduler step")
                .with_boundaries(vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
                .build(),
            prefix_hits: meter
                .u64_counter("tetraeder.kv.prefix.hits")
                .with_unit("{request}")
                .with_description(
                    "Requests that reused a cached prompt prefix, by source (live sequence or host RAM)",
                )
                .build(),
            prefix_evictions: meter
                .u64_counter("tetraeder.kv.prefix.evictions")
                .with_unit("{entry}")
                .with_description("Saved sequences dropped from the host RAM prefix cache")
                .build(),
            prefix_bytes: meter
                .u64_gauge("tetraeder.kv.prefix.bytes")
                .with_unit("By")
                .with_description("Host bytes held by the prefix cache")
                .build(),
            prefix_restore: meter
                .f64_histogram("tetraeder.kv.prefix.restore.duration")
                .with_unit("s")
                .with_description("Time to restore a saved sequence into the KV cache")
                .with_boundaries(LATENCY_BOUNDS.to_vec())
                .build(),
            spec_acceptance: meter
                .f64_histogram("tetraeder.spec.acceptance")
                .with_unit("1")
                .with_description("Share of drafted tokens the target accepted, per verify step")
                .with_boundaries(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
                .build(),
            spec_drafted: meter
                .u64_counter("tetraeder.spec.drafted")
                .with_unit("{token}")
                .with_description("Tokens drafted for verification")
                .build(),
            spec_accepted: meter
                .u64_counter("tetraeder.spec.accepted")
                .with_unit("{token}")
                .with_description("Drafted tokens the target accepted")
                .build(),
            plan_residual: meter
                .f64_histogram("tetraeder.plan.residual")
                .with_unit("1")
                .with_description(
                    "Relative error |measured - predicted| / predicted of the step time",
                )
                .with_boundaries(vec![0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
                .build(),
            plan_regret: meter
                .f64_gauge("tetraeder.plan.regret")
                .with_unit("1")
                .with_description("Step time of the plan over the best known plan, minus 1")
                .build(),
            plan_predicted_step: meter
                .f64_gauge("tetraeder.plan.predicted_step")
                .with_unit("s")
                .with_description("Step time the planner predicted for the running plan")
                .build(),
            bottleneck_share: meter
                .f64_gauge("tetraeder.bottleneck.share")
                .with_unit("1")
                .with_description("Share of the step time bound by a resource, 0 to 1")
                .build(),
            bottleneck_fault: meter
                .u64_counter("tetraeder.bottleneck.fault")
                .with_unit("{fault}")
                .with_description("Detector windows that reported a fault, by class and resource")
                .build(),
            replan_total: meter
                .u64_counter("tetraeder.replan.total")
                .with_unit("{replan}")
                .with_description("Replanning decisions by trigger and outcome")
                .build(),
            replan_duration: meter
                .f64_histogram("tetraeder.replan.duration")
                .with_unit("s")
                .with_description("Drain, restart and health check of one plan switch")
                .with_boundaries(vec![
                    0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
                ])
                .build(),
            engine_load: meter
                .f64_histogram("tetraeder.engine.load.duration")
                .with_unit("s")
                .with_description("Time to load the model into an engine, by outcome")
                .with_boundaries(vec![
                    0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
                ])
                .build(),
            replan_admitting: meter
                .u64_gauge("tetraeder.replan.admitting")
                .with_description("1 while the server admits new requests, 0 during a plan switch")
                .build(),
            cpu_variant: meter
                .u64_gauge("tetraeder.cpu.variant")
                .with_unit("1")
                .with_description("CPU backend variant in use: value 1, labelled by tier, ISA and dispatch")
                .build(),
            cpu_isa_missing: meter
                .u64_gauge("tetraeder.cpu.isa.missing")
                .with_unit("{feature}")
                .with_description("Host CPU features the loaded CPU backend lacks (a lower variant than the host could run)")
                .build(),
            cpu_threads: meter
                .u64_gauge("tetraeder.cpu.threads")
                .with_unit("{thread}")
                .with_description("CPU compute threads the engine was configured with")
                .build(),
            cpu_quota: meter
                .f64_gauge("tetraeder.cpu.quota")
                .with_unit("{cpu}")
                .with_description("CPU limit of the engine's cgroup in cores, as detected at start")
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

    /// Prefetch of a MoE `layer`: `result` is `issued`, `used` (routed while resident) or
    /// `wasted` (evicted unused).
    pub fn expert_prefetch(&self, n: u64, layer: u32, result: &'static str) {
        self.prefetch.add(
            n,
            &[
                KeyValue::new(attr::LAYER, i64::from(layer)),
                KeyValue::new(attr::RESULT, result),
            ],
        );
    }

    pub fn expert_prefetch_wasted_bytes(&self, bytes: u64) {
        self.prefetch_wasted_bytes.add(bytes, &[]);
    }

    /// One layer step's prediction made `depth` layers ahead: `accuracy` in 0 to 1.
    pub fn expert_prefetch_accuracy(&self, layer: u32, depth: u32, accuracy: f64) {
        self.prefetch_accuracy.record(
            accuracy,
            &[
                KeyValue::new(attr::LAYER, i64::from(layer)),
                KeyValue::new(attr::DEPTH, i64::from(depth)),
            ],
        );
    }

    /// Chunks in flight in the I/O service after a submit of class `urgent` or `prefetch`
    /// (ADR 0054).
    pub fn io_inflight(&self, class: &'static str, chunks: u64) {
        self.io_inflight
            .record(chunks as f64, &[KeyValue::new(attr::IO_CLASS, class)]);
    }

    /// One decode layer step with experts in flight: `1 - read_wait / read_span`.
    pub fn read_hidden(&self, share: f64) {
        self.read_hidden.record(share, &[]);
    }

    /// Time of `part` (`cpu_ready`, `read_wait`, `cpu_landed`, `copy`) of one decode layer step.
    pub fn layer_part(&self, part: &'static str, d: Duration) {
        self.layer_part
            .record(d.as_secs_f64(), &[KeyValue::new(attr::PART, part)]);
    }

    /// One tensor converted to `format` (ADR 0047).
    pub fn convert_duration(&self, d: Duration, format: &str) {
        self.convert_duration.record(
            d.as_secs_f64(),
            &[KeyValue::new(attr::FORMAT, format.to_owned())],
        );
    }

    pub fn convert_bytes(&self, bytes: u64) {
        self.convert_bytes.add(bytes, &[]);
    }

    /// A converted tensor looked up in the cache: `hit` or `miss`.
    pub fn convert_cache(&self, hit: bool) {
        self.convert_cache.add(
            1,
            &[KeyValue::new(
                attr::RESULT,
                if hit { "hit" } else { "miss" },
            )],
        );
    }

    /// Requests in one scheduler step.
    pub fn scheduler_batch(&self, size: u32) {
        self.batch_size.record(f64::from(size), &[]);
    }

    /// Mean seconds per decoded token of one request in `stage` (decode profile).
    pub fn decode_stage(&self, stage: &'static str, seconds: f64) {
        self.decode_stage
            .record(seconds, &[KeyValue::new(attr::STAGE, stage)]);
    }

    /// A request that reused a cached prompt prefix; `source` is `live` or `ram`.
    pub fn kv_prefix_hit(&self, source: &'static str) {
        self.prefix_hits
            .add(1, &[KeyValue::new(attr::SOURCE, source)]);
    }

    /// Saved sequences evicted from the prefix cache.
    pub fn kv_prefix_evicted(&self, entries: u64) {
        self.prefix_evictions.add(entries, &[]);
    }

    /// Host bytes the prefix cache holds now.
    pub fn kv_prefix_bytes(&self, bytes: u64) {
        self.prefix_bytes.record(bytes, &[]);
    }

    /// One saved sequence restored into the KV cache.
    pub fn kv_prefix_restore(&self, d: Duration) {
        self.prefix_restore.record(d.as_secs_f64(), &[]);
    }

    /// One verify step that carried `drafted` > 0 draft tokens of which `accepted` matched.
    /// `method` is the drafter: `ngram` or `draft`.
    pub fn spec_step(&self, method: &'static str, drafted: u64, accepted: u64) {
        let a = [KeyValue::new(attr::METHOD, method)];
        if drafted > 0 {
            self.spec_acceptance
                .record(accepted as f64 / drafted as f64, &a);
        }
        self.spec_drafted.add(drafted, &a);
        self.spec_accepted.add(accepted, &a);
    }

    /// Relative step time error of a stage: `|measured - predicted| / predicted`.
    pub fn plan_residual(&self, stage: u16, residual: f64) {
        self.plan_residual
            .record(residual, &[KeyValue::new(attr::STAGE, i64::from(stage))]);
    }

    /// Regret of the running `plan` (a plan id) against the best known one, as a fraction.
    pub fn plan_regret(&self, plan: &str, regret: f64) {
        self.plan_regret
            .record(regret, &[KeyValue::new(attr::PLAN, plan.to_owned())]);
    }

    pub fn plan_predicted_step(&self, stage: u16, seconds: f64) {
        self.plan_predicted_step
            .record(seconds, &[KeyValue::new(attr::STAGE, i64::from(stage))]);
    }

    /// Share of the step time bound by `resource` (ADR 0041), 0 to 1.
    pub fn bottleneck_share(&self, resource: &str, share: f64) {
        self.bottleneck_share
            .record(share, &[KeyValue::new(attr::RESOURCE, resource.to_owned())]);
    }

    /// One detector window reported a fault of `class` (ADR 0041) hitting `resource`.
    pub fn bottleneck_fault(&self, class: &str, resource: &str) {
        self.bottleneck_fault.add(
            1,
            &[
                KeyValue::new(attr::FAULT_CLASS, class.to_owned()),
                KeyValue::new(attr::RESOURCE, resource.to_owned()),
            ],
        );
    }

    /// `trigger`: `drift`, `slo_burn`, `load_shift`, `device_change`. `outcome`: `switched`,
    /// `knobs_applied`, `rejected`, `rolled_back` (ADR 0042).
    pub fn replan(&self, trigger: &str, outcome: &str) {
        self.replan_total.add(
            1,
            &[
                KeyValue::new(attr::REPLAN_TRIGGER, trigger.to_owned()),
                KeyValue::new(attr::REPLAN_OUTCOME, outcome.to_owned()),
            ],
        );
    }

    pub fn replan_duration(&self, d: Duration) {
        self.replan_duration.record(d.as_secs_f64(), &[]);
    }

    /// One engine load; `outcome` is `ok` or `failed` (ADR 0055).
    pub fn engine_load(&self, d: Duration, outcome: &str) {
        self.engine_load.record(
            d.as_secs_f64(),
            &[KeyValue::new(attr::LOAD_OUTCOME, outcome.to_owned())],
        );
    }

    /// Whether new requests are admitted (ADR 0055).
    pub fn replan_admitting(&self, on: bool) {
        self.replan_admitting.record(u64::from(on), &[]);
    }

    /// Once at engine start (ADR 0052): the CPU backend variant that is loaded. `isa` and
    /// `missing` are ggml feature names; `dispatch` is `dl` or `static`.
    pub fn cpu_backend(&self, tier: &str, isa: &[String], dispatch: &str, missing: &[String]) {
        self.cpu_variant.record(
            1,
            &[
                KeyValue::new(attr::CPU_TIER, tier.to_owned()),
                KeyValue::new(attr::CPU_ISA, isa.join(",")),
                KeyValue::new(attr::CPU_DISPATCH, dispatch.to_owned()),
            ],
        );
        self.cpu_isa_missing.record(
            missing.len() as u64,
            &[KeyValue::new(attr::CPU_ISA, missing.join(","))],
        );
    }

    /// Once at engine start (ADR 0059): the configured compute threads and, when the cgroup has
    /// one, its CPU limit in cores (`host::cgroup_cpu_quota`), so a run above its quota shows.
    pub fn cpu_threads(&self, threads: u64, quota: Option<f64>) {
        self.cpu_threads.record(threads, &[]);
        if let Some(q) = quota {
            self.cpu_quota.record(q, &[]);
        }
    }

    /// The KV cache (ADR 0050 b): `memory.used` and `.capacity` of kind `kv` with the cache
    /// `format` (for example `q8_0/q4_0`), and the positions held and held at most.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_cache(
        &self,
        device: &str,
        tier: &str,
        format: &str,
        used: u64,
        capacity: u64,
        tokens: u64,
        capacity_tokens: u64,
    ) {
        let mut attrs = mem_attrs(device, tier, "kv").to_vec();
        attrs.push(KeyValue::new(attr::FORMAT, format.to_owned()));
        self.mem_used.record(used, &attrs);
        self.mem_capacity.record(capacity, &attrs);
        let dev = [KeyValue::new(attr::DEVICE, device.to_owned())];
        self.kv_tokens.record(tokens, &dev);
        self.kv_capacity.record(capacity_tokens, &dev);
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
        CacheResult::Cpu => "cpu",
    }
}

fn mem_attrs(device: &str, tier: &str, kind: &str) -> [KeyValue; 3] {
    [
        KeyValue::new(attr::DEVICE, device.to_owned()),
        KeyValue::new(attr::TIER, tier.to_owned()),
        KeyValue::new(attr::KIND, kind.to_owned()),
    ]
}
