//! The numbers of llama-server's `/metrics`, counted by the gateway (ADR 0089).
//!
//! Only what the gateway measures is exported: tokens and time of finished requests, the busy
//! and the waiting requests. llama-server's decode-call and speculation counters have no
//! counterpart here.

use std::fmt::Write;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Default, Clone, Copy)]
struct Window {
    prompt_tokens: u64,
    prompt_us: u64,
    predicted_tokens: u64,
    predicted_us: u64,
}

pub(super) struct Counters {
    started_unix: u64,
    prompt_tokens: AtomicU64,
    prompt_cached: AtomicU64,
    prompt_us: AtomicU64,
    predicted_tokens: AtomicU64,
    predicted_us: AtomicU64,
    tokens_max: AtomicU64,
    /// The counters at the last scrape: the throughput gauges average over the time between
    /// two scrapes, as llama-server's do.
    last_scrape: Mutex<Window>,
}

fn us(ms: f64) -> u64 {
    (ms.max(0.0) * 1e3) as u64
}

impl Counters {
    pub(super) fn new() -> Self {
        Self {
            started_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
            prompt_tokens: AtomicU64::new(0),
            prompt_cached: AtomicU64::new(0),
            prompt_us: AtomicU64::new(0),
            predicted_tokens: AtomicU64::new(0),
            predicted_us: AtomicU64::new(0),
            tokens_max: AtomicU64::new(0),
            last_scrape: Mutex::default(),
        }
    }

    pub(super) fn started_unix(&self) -> u64 {
        self.started_unix
    }

    /// A request that ended normally: `prompt` tokens were processed, `cached` reused,
    /// `predicted` sampled.
    pub(super) fn finished(
        &self,
        (prompt, cached, prompt_ms): (u32, u32, f64),
        (predicted, predicted_ms): (u32, f64),
    ) {
        self.prompt_tokens.fetch_add(u64::from(prompt), Relaxed);
        self.prompt_cached.fetch_add(u64::from(cached), Relaxed);
        self.prompt_us.fetch_add(us(prompt_ms), Relaxed);
        self.predicted_tokens
            .fetch_add(u64::from(predicted), Relaxed);
        self.predicted_us.fetch_add(us(predicted_ms), Relaxed);
        self.tokens_max.fetch_max(
            u64::from(prompt) + u64::from(cached) + u64::from(predicted),
            Relaxed,
        );
    }

    /// Prometheus text exposition, llama-server's names and help texts.
    pub(super) fn render(&self, processing: usize, deferred: usize) -> String {
        let now = Window {
            prompt_tokens: self.prompt_tokens.load(Relaxed),
            prompt_us: self.prompt_us.load(Relaxed),
            predicted_tokens: self.predicted_tokens.load(Relaxed),
            predicted_us: self.predicted_us.load(Relaxed),
        };
        let before = std::mem::replace(
            &mut *self.last_scrape.lock().unwrap_or_else(|e| e.into_inner()),
            now,
        );
        let per_second = |n: u64, t_us: u64| {
            if t_us == 0 {
                0.0
            } else {
                n as f64 / (t_us as f64 / 1e6)
            }
        };
        let counters = [
            (
                "prompt_tokens_total",
                "Number of prompt tokens processed, excluding cached tokens",
                now.prompt_tokens as f64,
            ),
            (
                "prompt_tokens_cached_total",
                "Number of prompt tokens reused from the cache",
                self.prompt_cached.load(Relaxed) as f64,
            ),
            (
                "prompt_seconds_total",
                "Total time spent processing prompts",
                now.prompt_us as f64 / 1e6,
            ),
            (
                "tokens_predicted_total",
                "Number of generation tokens processed",
                now.predicted_tokens as f64,
            ),
            (
                "tokens_predicted_seconds_total",
                "Total time spent generating tokens",
                now.predicted_us as f64 / 1e6,
            ),
            (
                "n_tokens_max",
                "Largest observed sequence length (prompt + generation)",
                self.tokens_max.load(Relaxed) as f64,
            ),
        ];
        let gauges = [
            (
                "prompt_tokens_seconds",
                "Average prompt throughput in tokens/s",
                per_second(
                    now.prompt_tokens - before.prompt_tokens.min(now.prompt_tokens),
                    now.prompt_us - before.prompt_us.min(now.prompt_us),
                ),
            ),
            (
                "predicted_tokens_seconds",
                "Average generation throughput in tokens/s",
                per_second(
                    now.predicted_tokens - before.predicted_tokens.min(now.predicted_tokens),
                    now.predicted_us - before.predicted_us.min(now.predicted_us),
                ),
            ),
            (
                "requests_processing",
                "Number of requests processing",
                processing as f64,
            ),
            (
                "requests_deferred",
                "Number of requests deferred",
                deferred as f64,
            ),
        ];
        let mut out = String::new();
        for (kind, items) in [("counter", &counters[..]), ("gauge", &gauges[..])] {
            for (name, help, value) in items {
                let _ = write!(
                    out,
                    "# HELP llamacpp:{name} {help}\n# TYPE llamacpp:{name} {kind}\nllamacpp:{name} {value}\n"
                );
            }
        }
        out
    }
}
