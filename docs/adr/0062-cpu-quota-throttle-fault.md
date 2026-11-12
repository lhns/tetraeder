# 0062. Detector fault class `cpu_quota_throttle`: the cgroup CPU quota throttling the engine's own threads

Status: Proposed (2026-11-12). Extends ADR 0041 (fault classes) and ADR 0013 (environment first).

## Context

On the Vast boxes the container's CPU quota throttled the builder in 80 to 98% of CFS periods at `threads` = quota with ggml's default polling (`bench/results/20261112e`, ADR 0045 last paragraph). Idle workers spinning between the about 95 segment graphs per token spent the quota the compute then waited for. With `cpu_poll` 0 the throttled share fell to 0.05 and decode ran 1.30x faster. The run looked like plain CPU-bound compute: the `limits` list said "limited by cpu" and no fault fired. `cpu_cotenant` does not cover it either: nothing else uses the cores, the engine's own threads do.

## Decision

- **New class `cpu_quota_throttle`** (`FaultClass::CpuQuotaThrottle`), resource `cpu_quota`. A separate resource id keeps it apart from `cpu_cotenant` (resource `cpu`) when both hold, which `faults_never_repeat_a_cause` in the battery requires. It claims `cpu`, so the residual check does not report the same slowdown as `model_mismatch`.
- **Inputs, two new `EnvSignals` fields, both optional (`None` = not measured):**
  - `cpu_throttled_share`: `nr_throttled / nr_periods` of the container's cgroup over the run, pooled over the decode windows (cgroup v2 `cpu.stat`; v1 `cpu,cpuacct/cpu.stat`). `None` when the cgroup has no CFS quota (`nr_periods` stays 0), off Linux, or without a cgroup.
  - `cpu_stall_share`: PSI `cpu.pressure` "some" of the container, median over windows. Evidence only; a kernel without PSI loses nothing.
- **Rule.** Fires when `cpu_throttled_share >= 0.3` and `cpu` is busy at least `MATERIAL_SHARE` (10%) of the step and slower than calibrated beyond the band (or no calibrated rate). 0.3 is between the healthy runs (under 0.05 in the box and sim runs) and the faulty ones (0.80 to 0.98); the host alert `TetraederCpuThrottled` warns earlier at 0.2 on the quota alone.
- **Cost.** With a calibrated CPU rate, the excess CPU seconds as a share of the step (as for `cpu_cotenant`). Without: `cpu busy share x throttled share x 0.3`, which gives about 0.23 for the 20261112e case (CPU 0.8 busy, 0.9 throttled), the measured 1.30x.
- **How it differs from `cpu_bound`.** There is no such class: "limited by cpu" is a `limits` verdict, a property of the plan on a healthy box. The quota fault says the CPU is slower than it should be because the cgroup withholds it. Both can be reported together; the fault tells the operator what to change, the limit does not. From `cpu_cotenant`: that needs other processes using at least 15% of the quota (`cpu_other_cores`), this needs throttled periods. They are independent rules; a co-tenant inside the same cgroup can cause both, then both are reported.
- **Remediation hint (advisor, report only).** Environment: raise the quota or use a box with more cores. Setting (`threads` action): lower ggml polling (`cpu_poll` 0) and keep threads at or below the whole-core quota.
- **Telemetry.** No new metric name: `tetraeder.bottleneck.fault` with `tetraeder.fault.class` = `cpu_quota_throttle` and `tetraeder.resource` = `cpu_quota`; the existing overview panel groups by class. Alerts: `TetraederCpuQuotaThrottleFault` in `deploy/alerts/host.yml` (carries the fix in its description); `TetraederEnvironmentFault` in `slo.yml` excludes the class so one fault pages once.
- **Sampler.** `scripts/detector_sample.py` records `nr_periods`, `nr_throttled` and cpu PSI of the container per sample and writes `cpu_throttled_share` and `cpu_stall_share` into the observation (`null` without a quota or PSI).
- **Rule 4 tests.** The simulator injects the class (CPU rate / 1.3, 80% to 98% of periods throttled, PSI on odd seeds) and reports a healthy 0 to 5% otherwise, so the existing battery (every fixture times every class, singles top-1 at least 9/10 and the only finding, pairs both in the top 3, no-fault at most 1 false positive in 10) covers it on every hardware fixture. Rule tests: needs throttled periods and a slow CPU, below threshold and `None` stay silent, not a co-tenant or model mismatch, advice names `cpu_poll`.

## Consequences

- Rule 9: the cgroup and PSI reads exist on Linux only; elsewhere the fields are `None` and the rule cannot fire. The simulated tests need no hardware.
- The 0.3 threshold and the 0.3 default loss are tuned on the simulator and one box day (20261112e); a box run with `cpu_poll` 0 versus the default is the held-out check.
- The nightly CPU-runner testbed has no CFS quota and so covers none of this; a `systemd-run -p CPUQuota=` fault in `scripts/fault_testbed.sh` is a follow-up.
