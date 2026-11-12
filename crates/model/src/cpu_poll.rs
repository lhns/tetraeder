//! Polling level of the CPU backend's threadpool, chosen from the cgroup CPU quota (ADR 0061).
//!
//! Idle ggml workers spin between graphs (level 50 by default). Under a CFS quota that spinning
//! spends the quota the next graph then waits for: on a 23-core quota with 23 threads the
//! builder was throttled in 80 to 98% of periods, and level 0 decoded 1.30x faster.

/// ggml's default level (`ggml_threadpool_params_init`).
pub const GGML_DEFAULT_POLL: u32 = 50;

/// The polling level a load used and why.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CpuPoll {
    /// The level the threadpool runs with, 0 to 100.
    pub level: u32,
    /// Chosen from the detected quota (`LoadOptions::cpu_poll` was `None`).
    pub auto: bool,
    /// The cgroup's CPU limit in cores, `None` when unlimited or not on Linux.
    pub quota: Option<f64>,
    /// The host's online CPUs, `None` when unknown.
    pub online: Option<usize>,
}

/// Resolves the level: an explicit one wins; otherwise 0 when a quota binds (below the host's
/// online CPUs, or no more whole cores than `threads`), else ggml's default.
pub fn resolve(
    explicit: Option<u32>,
    quota: Option<f64>,
    online: Option<usize>,
    threads: usize,
) -> CpuPoll {
    let binds = |q: f64| online.is_some_and(|o| q < o as f64) || threads as f64 >= q.floor();
    let level = match explicit {
        Some(p) => p.min(100),
        None if quota.is_some_and(binds) => 0,
        None => GGML_DEFAULT_POLL,
    };
    CpuPoll {
        level,
        auto: explicit.is_none(),
        quota,
        online,
    }
}

/// [`resolve`] with this machine's quota and online CPUs.
pub fn detect(explicit: Option<u32>, threads: usize) -> CpuPoll {
    resolve(
        explicit,
        tetraeder_telemetry::cgroup_cpu_quota(),
        tetraeder_telemetry::online_cpus(),
        threads,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_decision_table() {
        // (quota, online, threads, level)
        let rows = [
            // No quota: ggml's default whatever the threads.
            (None, Some(128), 128, 50),
            (None, None, 8, 50),
            // Quota below the host's CPUs: 0 even with fewer threads (20261112e: 19 threads on
            // a 23-core quota were still throttled in 82% of periods).
            (Some(23.0), Some(128), 23, 0),
            (Some(23.04), Some(128), 19, 0),
            (Some(23.0), Some(128), 4, 0),
            // A quota at or above the host's CPUs binds only with as many threads as cores.
            (Some(128.0), Some(128), 128, 0),
            (Some(128.0), Some(128), 64, 50),
            (Some(200.0), Some(128), 128, 50),
            // Online CPUs unknown: the thread clause alone.
            (Some(23.04), None, 23, 0),
            (Some(23.04), None, 24, 0),
            (Some(23.04), None, 22, 50),
        ];
        for (quota, online, threads, want) in rows {
            let p = resolve(None, quota, online, threads);
            assert_eq!(
                p.level, want,
                "quota {quota:?} online {online:?} threads {threads}"
            );
            assert!(p.auto);
            assert_eq!((p.quota, p.online), (quota, online));
        }
    }

    #[test]
    fn explicit_level_wins() {
        for (quota, online, threads) in [(Some(23.0), Some(128), 23), (None, Some(8), 8)] {
            for explicit in [0, 30, 50, 100] {
                let p = resolve(Some(explicit), quota, online, threads);
                assert_eq!(p.level, explicit);
                assert!(!p.auto);
            }
        }
        assert_eq!(resolve(Some(250), None, None, 1).level, 100);
    }
}
