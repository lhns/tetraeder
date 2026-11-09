# 0041. Bottleneck detector: one observation type, faults separate from the limiting resource, advisor report only

Status: Proposed (Phase 3a)

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

First testbed run (2026-11-09, 4-core runner, 33 runs): single top-1 12/18 (cpu 6/6, disk 6/6, mem 0/6), no-fault false positives 0/6, pair top-3 0/6. Two findings, both left open:

- `mem` is not detected although the signal is real (swap-in 190 MB/s, decode 5x slower). The detector assumes paging shows up as host time; here the stall is inside compute (0.48 s/token compute, 0.002 host), so the `host` share test fails. The Swapping rule needs a different share test (swap-in rate with the slowdown on cpu or wall time).
- A strong CPU co-tenant (2 busy loops on 4 threads, 9x slower) hides the other fault in a pair: the engine becomes CPU-bound, the disk share falls below material (0.017 s/token). The pair schedule needs a lighter co-tenant, or the pair bar needs the second fault to be material first.

## Addendum 2026-11-09: Swapping rule and pair schedule after the first testbed run

Observations of the first run are in `bench/results/20261109-runner/detector-testbed/` (`score-before.json`); `crates/analysis/tests/testbed.rs` scores them.

- **Diagnosis, `mem`.** The sampler already had the evidence: swap-in 183 to 206 MB/s in all six `mem` runs, 0 in every other single run. The stall lands where the swapped pages are touched: compute 0.008 to 0.50 s/token, CPU busy 0.16 to 0.40 s/token, host only 0.0009 to 0.002 (0.3% of the step), so the old `host` share test (10%) never passed.
- **Swapping rule.** Paging is swap-in at 20 MB/s or more, or major faults at that rate when PSI is measured (without PSI, major faults also count first reads of a mapped file). The cost: the PSI memory "some" share of the engine's cgroup when present (the fault fires only at 10% or more); without PSI, the excess of `cpu` and `host` over their calibrated rates, or half their share when uncalibrated, and only if one of them is material and slower than calibrated (or uncalibrated). The fault's resource is `ram`; it claims `cpu` and `host`, so the residual check does not report the same slowdown as a model mismatch.
- **Sampler.** `detector_sample.py` records major faults of the engine's process tree and the PSI `memory.pressure` total of the engine's cgroup (the scope on the testbed, the container on a box, `/proc/pressure/memory` as fallback); observations carry `major_faults_per_s` and `memory_stall_share` (absent without PSI).
- **Simulator.** Swapping now slows `cpu` and `host` 3x (was `host` 8x) at a swap-in of 180 to 220 MB/s, sets major faults to match, and reports the stall as PSI on even seeds only, so both rule paths run. Battery (3 fixtures, every fault, pair and no-fault, 10 seeds): every single and pair 10/10, no-fault 0/10 with a fault on every fixture; bars (9/10, at most 1) kept.
- **Diagnosis, pairs: a schedule problem, not ranking.** In the pairs the injected disk cap had no cost: with 2 busy loops the engine decoded at 1.34 s/token with the cap and 1.36 without, reading 64 MB/token at 48 MB/s against a 201 MB/s cap, and the measured read rate was 4 GB/s (prefetch hidden behind compute). No disk evidence exceeded the band, and the detector found at most one fault, so the top 3 had room: a ranking change cannot help. The testbed now uses 1 busy loop in a pair (half the threads alone) and scales a pair's disk cap by the partner's single-run pace over the calibration pace (cpu+disk 23 MB/s, disk+mem 44 MB/s on the first run's numbers).
- **Effect on the first run's observations** (same files, new rule): single top-1 18/18 (was 12/18), no-fault false positives 0/6, pairs 2/6 (the cpu+mem pairs; the disk pairs carry no disk evidence).
