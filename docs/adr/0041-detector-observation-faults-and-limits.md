# 0041. Bottleneck detector: one observation type, faults separate from the limiting resource, advisor report only

Status: Accepted (2026-11-15: detector passed on a box (`20261110`) and in the simulated battery). Earlier: Proposed (Phase 3a)

Measurements live in the session READMEs (`bench/results/<session>/README.md`), not here.

## Context

ADR 0013 orders the detector (environment first, regime, ranked shares with confidence or "balanced", model check last). The Phase 3a exit test counts correct top-1 on single-fault runs, false positives on no-fault runs and both faults in the top 3 on two-fault runs, in the simulator first (rule 4) and then on a box. A no-fault run is never free of a limit: a MoE with experts on the CPU is always bound by something. If that limit counted as a finding, every control run would fail; if nothing were reported, the report would be useless.

## Decision

- **One input type.** `analysis::Observation` is produced by the simulator's fault injection and by the box sampler alike: repeated windows (tokens, wall time, busy seconds per resource, step time split into compute, expert load wait, transfer wait, host), environment signals (throttle reasons per GPU, PCIe gen and width current and max, CPU used by other processes in cores next to the CPU quota, swap-in rate, VRAM held by other processes, disk read rate of other processes, an `io.max` limit if set) and, when the plan was calibrated, the expected per-resource demand and rates from the cost model (ADR 0040).
- **Two lists in the report.**
  - `faults`: things that are wrong and actionable, ranked by the step time they cost. Environment faults (GPU throttle, CPU co-tenant, disk contention or throttle, VRAM held by others, PCIe link below its maximum, swapping) fire only above a threshold and only when the resource they hit carries a material share of the step. A model mismatch fires when a resource runs slower than its calibrated rate beyond the anchor error band and no environment fault explains it.
  - `limits`: every resource's share of step time with a confidence interval over windows, in the regime's form (serialized at batch 1: share of the summed demands; under load: utilization against the busiest). The verdict is `balanced` when the top share's interval overlaps the second's, else `limited by <resource>`.
- **Exit-test semantics.** Top-1 and top-3 are read from `faults`. A false positive is any entry in `faults` on a no-fault run. `limits` is always reported and never counted as a finding.
- **Fault classes:** `gpu_throttle`, `cpu_cotenant`, `disk_slow` (contention, cgroup throttle or slower than calibrated), `vram_pressure`, `pcie_degraded`, `link_slow`, `swapping`, `model_mismatch`.
- **Advisor (report only in 3a).** Each fault and the top limit map to candidate actions (more VRAM slots, threads, ubatch, op offload, experts placement, or the environment fix); each is scored with the cost model as predicted gain over a horizon minus switch cost, tagged with its policy class (auto, ask, report) from the hints, and printed. Nothing is applied.
- **Tests.** Simulated: every hardware fixture times every fault class, 10 seeded runs each with window noise, plus 10 no-fault runs and two-fault pairs per fixture. Box: the own MoE builder (expert reads from disk, CPU experts) under no fault, a CPU co-tenant (thread-capped busy loops), disk contention (cgroup `io.max` when writable, otherwise a concurrent direct-I/O reader), and VRAM pressure (a process holding VRAM, so fewer slots fit); observations go to `bench/results/<date>/detector-*.json` and a CI test scores them.
- **Interfaces.** `tetraeder doctor` (one-shot report from an observation file or a live sample) and `tetraeder advise`; the report also goes out as OpenTelemetry events and the `tetraeder.bottleneck.share` gauge.

## Consequences

- The thresholds are fixed numbers in code and tuned on simulated fixtures only; the box runs are a held-out check.
- Faults the container cannot produce (power limit, clocks, PCIe width) are tested in the simulator only until real hardware allows them.

## Addendum 2026-11-09: nightly fault testbed on CPU runners

`.github/workflows/nightly.yml` runs `scripts/fault_testbed.sh` on `ubuntu-latest` (schedule plus `workflow_dispatch`, never a PR gate). It runs the real engine (`exit_regime`, CPU backend, the synthetic MoE from `tests/common/synth.rs`, a RAM tier of a quarter of the experts so misses read from disk) in decode windows, with `detector_sample.py` sampling next to it, and feeds the real observations to `tetraeder detector-score`. Same schedule shape and manifest as `detector_campaign.sh`: 3 calibration runs, shuffled rounds of no fault and each fault, then fault pairs.

Covered, 3 of the 8 fault classes:

| Injected fault | Manifest name | Detector class |
|---|---|---|
| `systemd-run --scope -p IOReadBandwidthMax=<disk> <0.3 x calibrated rate>` (cgroup `io.max`; direct readers if the runner does not enforce it) | `disk` | `DiskSlow` |
| thread-capped busy loops next to the engine | `cpu` | `CpuCotenant` |
| `-p MemoryMax=<0.4 x calibrated peak RSS>` with swap on | `mem` | `Swapping` |

The scorer reads `mem` as `Swapping`. Runner noise is large (shared host, no clock control), so the job prints the score and uploads the results but fails only on harness errors, not on the exit bars. Box-only: VRAM pressure, PCIe degradation, GPU throttle, and the slow link and model-mismatch classes.

First testbed run (2026-11-09, `20261109-runner`): `mem` was not detected although the signal was real (the stall lands inside compute, not host time), and a strong CPU co-tenant hid the other fault in a pair. Both are addressed in the next addendum.

## Addendum 2026-11-09: Swapping rule and pair schedule after the first testbed run

Observations of the first run are in `bench/results/20261109-runner/detector-testbed/` (`score-before.json`); `crates/analysis/tests/testbed.rs` scores them.

- **Diagnosis, `mem`** (`20261109-runner`). The sampler already had the evidence: swap-in in every `mem` run and none in any other single run. The stall lands where the swapped pages are touched, in compute and CPU time, not in host time, so the old `host` share test (10%) never passed.
- **Swapping rule.** Paging is swap-in at 5 MB/s or more (was 20), or major faults at that rate with a PSI memory stall of at least 1% (without it, major faults also count first reads of a mapped file). It is a fault when there is a stall: PSI memory "some" of the engine's cgroup at 10% or more, or `cpu` or `host` material and slower than calibrated (or uncalibrated). Cost: the larger of the PSI share and the excess of `cpu` and `host` over their calibrated rates (half their share when uncalibrated). The fault's resource is `ram`; it claims `cpu` and `host`, so the residual check does not report the same slowdown as a model mismatch. The rate threshold is low because the rate falls with the pace it slows and under an `io.max` that also caps swap-in; the stall requirement keeps stray pages out (`20261109-runner`).
- **Sampler.** `detector_sample.py` records major faults of the engine's process tree and the PSI `memory.pressure` total of the engine's cgroup (the scope on the testbed, the container on a box, `/proc/pressure/memory` as fallback); observations carry `major_faults_per_s` and `memory_stall_share` (absent without PSI). It also drops the sample taken while the tree exits (its counters read 0, which once made a window's CPU time negative).
- **Simulator.** Swapping now slows `cpu` and `host` 3x (was `host` 8x) at a swap-in of 180 to 220 MB/s, sets major faults to match, and reports a tenth of the stall as PSI (close to what real runs showed), on even seeds only, so runs with and without PSI both occur. The battery (3 fixtures, every fault, pair and no-fault, 10 seeds) passes; bars (9/10, at most 1) kept.
- **Diagnosis, pairs: a schedule problem, not ranking** (`20261109-runner`). With 2 busy loops the injected disk cap had no cost in the pairs, so no disk evidence exceeded the band and the detector found at most one fault: a ranking change cannot help. The testbed now uses 1 busy loop in a pair (half the threads alone) and scales a pair's disk cap by the partner's single-run pace over the calibration pace.
- **Second run** (nightly on `track-af-detector-fix`, `20261109-runner`, `detector-testbed-2/`). The new schedule works. The first version of the rule, with PSI deciding when present, missed `mem`: memory PSI stayed low at a large slowdown and swap-in fell under the disk+mem cap. Hence PSI as a floor rather than a gate and the 5 MB/s rate above. With the final rule, rescored offline, the second run finds every single fault and pair with no false positives. On the testbed the calibration has a rate only for the disk, so `cpu` is always uncalibrated and material (its threads spin): the Swapping stall check always passes there and the swap-in rate alone separates `mem` from everything else. CpuCotenant has the same property. A CPU work-unit counter would let the calibration carry a CPU rate.

## Addendum 2026-11-12: ninth fault class

`cpu_quota_throttle` (the cgroup CPU quota throttling the engine's own threads) joins the list; see ADR 0062.
