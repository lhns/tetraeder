//! Cost accounting: what a finished request used, as metrics and as a [`RequestCost`].

use super::pump::Run;
use super::record_error;
use crate::chat::Usage;
use crate::engine::EngineStats;
use opentelemetry::KeyValue;
use std::time::{Duration, Instant};
use tetraeder_telemetry::{CacheResult, Direction};

/// What one finished request cost, for accounting outside the gateway.
#[derive(Debug, Clone, PartialEq)]
pub struct RequestCost {
    pub request_id: String,
    pub caller: String,
    pub model: String,
    pub operation: &'static str,
    pub prompt_tokens: u32,
    pub cached_tokens: u32,
    pub completion_tokens: u32,
    pub gpu_seconds: Option<f64>,
    pub bytes_streamed: Option<u64>,
    pub duration: Duration,
    /// `error.type` when the request did not complete (`cancelled`, `internal`, ...).
    pub error_type: Option<&'static str>,
}

/// Receives the cost of every request that reached the engine.
pub trait CostSink: Send + Sync {
    fn record(&self, cost: &RequestCost);
}

/// Attribute names used only by the gateway.
pub mod attr {
    pub const CALLER: &str = "tetraeder.caller";
    pub const RESOURCE: &str = "tetraeder.cost.resource";
    pub const PRIORITY: &str = "tetraeder.priority";
}

impl Run {
    pub(super) fn finish(
        &self,
        usage: Usage,
        stats: &EngineStats,
        first: Option<Instant>,
        last: Instant,
        error_type: Option<&'static str>,
    ) {
        let duration = self.started.elapsed();
        self.span
            .record("gen_ai.usage.input_tokens", usage.prompt_tokens);
        self.span
            .record("gen_ai.usage.output_tokens", usage.completion_tokens);
        if let Some(e) = error_type {
            record_error(&self.span, e);
        }
        if let Some(m) = &self.gw.metrics {
            let g = &m.genai;
            g.request_duration(duration, &self.model, error_type);
            if let Some(f) = first
                && usage.completion_tokens > 1
            {
                g.time_per_output_token((last - f) / (usage.completion_tokens - 1), &self.model);
            }
            let uncached = usage.prompt_tokens.saturating_sub(usage.cached_tokens);
            g.tokens(
                u64::from(uncached),
                &self.model,
                Direction::Input,
                CacheResult::Miss,
            );
            if usage.cached_tokens > 0 {
                g.tokens(
                    u64::from(usage.cached_tokens),
                    &self.model,
                    Direction::Input,
                    CacheResult::Hit,
                );
            }
            g.tokens(
                u64::from(usage.completion_tokens),
                &self.model,
                Direction::Output,
                CacheResult::Miss,
            );
            let attrs = |res: &'static str| {
                [
                    KeyValue::new(tetraeder_telemetry::attr::MODEL, self.model.clone()),
                    KeyValue::new(attr::CALLER, self.gw.caller_label(&self.caller)),
                    KeyValue::new(attr::RESOURCE, res),
                ]
            };
            if let Some(s) = stats.gpu_seconds {
                m.request_cost.record(s, &attrs("gpu_seconds"));
            }
            if let Some(b) = stats.bytes_streamed {
                m.request_cost.record(b as f64, &attrs("bytes_streamed"));
            }
        }
        if let Some(sink) = &self.gw.cost_sink {
            sink.record(&RequestCost {
                request_id: self.id.clone(),
                caller: self.caller.clone(),
                model: self.model.clone(),
                operation: self.op.as_str(),
                prompt_tokens: usage.prompt_tokens,
                cached_tokens: usage.cached_tokens,
                completion_tokens: usage.completion_tokens,
                gpu_seconds: stats.gpu_seconds,
                bytes_streamed: stats.bytes_streamed,
                duration,
                error_type,
            });
        }
    }
}
