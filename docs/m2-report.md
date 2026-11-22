# M2: report for the owner (draft, 2026-11-22)

M2 ends Phase 5 (`docs/plan.md`, phase table). This report lists what passed, under which caveats, what the owner is asked to decide, and what was planned for Phases 4 and 5 but not built. It is ready for the owner's decisions below.

## What passed

- **Phase 4 (own KV and scheduler), passed 2026-11-17 (ADR 0065, Accepted).** Throughput at concurrency 4 and 1 against tuned llama-server for both counting models in `20261117k`; TTFT on the Claude Code replay in `20261117l`, and the ADR 0074 bar (median and p90 at most 1.15x llama-server, saves at most 10% of TTFT) in both rounds of `ttft7-20261119`.
- **Phase 5 (own executor, multi-GPU), passed 2026-11-21 (ADR 0068).** `phase5_pass` true over D2, M2, D4 and M4 (`bench/results/phase5-verdict/README.md`): the planner picked a member of the measured winner set in every case, every candidate within 15% of its prediction, two clean rounds per configuration, A/A inside the band.

| Case | Model, GPUs | Pick | Gate against llama.cpp |
|---|---|---|---|
| D2 | Qwen3-32B, 2 | tp-2 | 1.011 |
| M2 | Qwen3-30B-A3B, 2 | layer-2 | 1.008 |
| D4 | Qwen3-32B, 4 | tp-4 | 1.022 |
| M4 | Qwen3-30B-A3B, 4 | layer-4 | 0.987 |

## Admission probe on final main (D2a)

The admission path changed after the Phase 4 verdict commit, so the Phase 4 throughput pairs ran again on final main.

- `d2a-20261121`: Qwen3-8B passes (worst pair c4 1.091, c1 1.022). Qwen3-30B-A3B is null: all four pairs, in two attempts, were unclean on the board-limit throttle share. The box was a different, slower machine under the same host id as `20261117k` (EPYC 7302P, 32 CPUs, against EPYC 7R13, 96 CPUs). llama-server ran about 14% slower there and the own engines did not, so the higher ratios on that box are not evidence (Diagnosis section of its README).
- `d2a30b-20261122` (node `bench-d2a-30b`): null. On an EPYC 7B12 box (no 7R13 was on offer) the 30B probe c4 ratio was 0.910, under the 0.925 stop line, so no pair ran. The original probe was 0.929 (0.004 over the line) and its pairs 0.943 and 0.948 (0.018 to 0.023 over). The own engine lost 7 to 8% against the original machine, llama-server 4%. Both engines ran 23 threads.
- So the judged 30B half of the D2a probe has no verdict. Between the Phase 4 commit (6b07795) and final main the single-GPU MoE path changed only in the restore path and the scheduler's admission changes (ADR 0074); the no-capture flag (ADR 0081) and the decode toggles act only in the multi-GPU executor. The 8B is confirmed.
- `ab30b-20261122` (owner-approved, node `bench-ab30b`): same box (Ryzen 9 5950X, RTX 3090), the own 30B engine at 6b07795 against final main, three interleaved rounds, all clean. Median B/A 0.992 at c4 and 0.992 at c1 (every round 0.988 to 0.999), so by the rule fixed before the run there is no regression evidence; but all six ratios are under 1, so final main is consistently about 0.8% behind the Phase 4 build, cause unknown, and no A/A ran on that box. A was built from 6b07795 at its pin (fc504f91 plus its four patches); B's engine code equals the D2a sessions'. The 267 tok/s on the 7B12 machine was a machine effect (301 to 305 here). The A/B judges nothing against llama-server, so the D2a null stays a null.
- Arithmetic across machines, not a verdict: the original pairs times 0.992 give about 0.935 and 0.940, 0.010 to 0.015 over the 0.925 line; the original probe times 0.992 gives about 0.922, under the stop line.

## Caveats

1. **One host, one evening.** Both Phase 5 verdict sessions (`v2c-20261121`, `v4a-20261121`) ran on host 20210 on 2026-11-21. No second host confirms the picks.
2. **D2 and M2 ran on two GPUs of a 4-GPU box** (GPUs 2 and 3, `P5_DEVICES`), not on a 2-GPU box. Both engines used the same pair, and a run touching another GPU was unclean. Six 2-GPU hosts failed pulls, SSH or the power gate on 2026-11-20 and 21.
3. **Cost model 5 (ADR 0082) is fitted and confirmed only on host 20210.** Its clock ratio k (30B 1.062) comes from the same host's calibration.
4. **M4's pick is 1.3% behind llama.cpp** (0.987). It passes inside the band of 0.08 (the spec band `cuda-default`, since no measured noise file for these GPU classes has n of 3 or more), and would pass at any band of 0.013 or more. A/A runs agreed within 0.2%.
5. **Rule changes after failed sessions.** Eleven changes to the verdict rules, host choice and what is judged were made after ADR 0068 was written, each before the next session and none after a verdict (`docs/notes/phase5-rule-changes.md`).
6. **D2a scripts.** The D2a script has no TTFT stage (TTFT was judged in its own sessions). Until `d2a30b-20261122` its rerun rule replaced at most one unclean pair, so a session with both pairs unclean ended without a rerun; the fix (reps 3 and 4) has tests but has not run on a box.
7. **The 8B margin on `d2a-20261121`** (1.09 to 1.12, against 1.02 in `20261117k`) comes from the same slower baseline as the 30B. The pass stands on two clean pairs; the margin is not a gain.

## Decisions for the owner

1. **Ratify two calls made by delegation to the advisor** (`docs/notes/phase5-rule-changes.md`, 2026-11-21):
   - the anchor `layer-g@L/2` on 3 or more GPUs, which measures the low decode clock for cost model 5 (node `planner-owner-layer-anchor`);
   - D2 and M2 on two GPUs of a 4-GPU box, instead of loosening the power gate (node `bench-owner-two-of-four`).
2. **Review the rule-change list** (`docs/notes/phase5-rule-changes.md`) as a whole.
3. **Accept or reopen ADRs** still Proposed:
   - 0068 (Phase 5 verdict rules);
   - 0069 (own executor, rank programs);
   - 0081 (prefill steps are not captured, patch 0006; its CUDA backend test has run on a GPU in `s3f2-20261120`);
   - 0082 (decode at measured clocks, cost model 5).
4. **The 30B half of the D2a probe.** `docs/m2-plan.md` step 5 asked for a D2a probe on final main before reporting M2; the 30B half has no verdict after two sessions. The owner chose the same-box A/B as the next measurement (2026-11-22; result above). Decide: accept the A/B in place of the 30B half of that step (a deviation from step 5, recorded here), or require a judged D2a session for the 30B.
   Changing the probe rule or the board-limit share rule after these sessions would be a rule change after a miss, so it is not proposed here.
5. **Confirm M2 reached**, after decision 4.

## Planned for Phases 4 and 5 but not built

| Item | Plan | Tree node | State |
|---|---|---|---|
| MTP drafts (where the GGUF has heads) | feature 27, Phase 4 row | `serving-mtp-drafts` | parked |
| Self-speculative decoding with a low-bit copy | feature 52, Phase 4 row | `serving-self-spec` | parked |
| Speculative rerun with a tokenizer-sharing draft pair | m1-open item 1 | `bench-spec-rerun` | parked |
| KV tiers in RAM and on NVMe | feature 3, Phase 4 half | `memory-kv-tiers` | parked |

Speculative decoding with n-gram and draft models is built and runs inside continuous batching (`serving-speculative`, `serving-spec-batching`); on the 235B with streamed experts it is slower until the op-offload fix is rerun. None of these items is part of an exit test.

## After M2

- `cluster-transport-wan`: transport latency between two boxes (ADR 0023's measurement); the owner's transport pick then unblocks the Phase 6 nodes.
- Start a fresh session from `docs/roadmap.md`.
