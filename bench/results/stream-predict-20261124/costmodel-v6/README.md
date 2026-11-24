# Cost model 6 in sample on stream-predict-20261124

Status: the ADR 0092 done line holds in sample: arm D2 decode 2.370 tok/s predicted against 2.636 measured (e -0.101, within 0.15). Not a verdict on Phase 6; the confirmation is `bench-235b-segment-confirm` on a fresh session with predictions frozen first. Not measured (no new box run; priced from committed files of `stream-predict-20261124`).

Source: CI run 45436274673's successor on branch `planner-costmodel-stream-terms` (test `costmodel_v6_copy`, `--nocapture`, lines `IN_SAMPLE`), MODEL_VERSION 6.

| Arm | Routing | pp pred | pp meas | e_pp | tg pred | tg meas | e_tg |
|---|---|---|---|---|---|---|---|
| D2 (64 GB balloon, 16 VRAM + 52 RAM slots, streamed) | trace | 74.855 | 35.371 | +1.116 | 2.3703 | 2.6360 | -0.101 |
| R (all experts in RAM, 32 VRAM + 96 RAM slots) | measured (pair share 0.701) | 153.02 | 190.97 | -0.199 | 10.565 | 9.869 | +0.071 |

Cost model 5 on the same arms (from the session README): D2 pp +1.32, tg -0.95 (trace -0.71); R pp +0.35, tg -0.45 (trace -0.33).

Reading:
- Decode is within 0.15 in both arms once routing comes from a trace or the measured share and the copy rate no longer scales with threads.
- Prefill is not: D2 +1.12 rests on the cold first request (137.8 GB read and a 7.7 s hot-set refresh inside the timed prefill), which ADR 0092 excludes by the warm protocol rather than pricing, so this number is expected to stay off on this session's data. R -0.20 is in the regime the exit judges; its prefill terms (offload copy rate, reads after one warm pass, now removed by `memory-ram-tier-complement`) are what the confirmation session must show within 0.15.
- Arm R is report only (ADR 0092): its measured pair share is the run's own, so it is not an out-of-sample test.
