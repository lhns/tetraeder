# Phase 5: verdict-rule changes since ADR 0068 was written

For the owner, before M2 is reported (`docs/m2-plan.md`, Phase 5 item 5). ADR 0068 was written on 2026-11-15, before any Phase 5 code. This lists every later change to what counts, the session that prompted it, and who decided. Details are in ADR 0068's History and the linked READMEs.

## Rule changes

| Date | Change | Prompted by | Decided by |
|---|---|---|---|
| 2026-11-16 | Throttle judged by single samples before and after each run; SW power cap (0x4) after a run is benign. | E8 build (ADR 0065 addendum 2026-11-16) | coordinator, before any verdict session |
| 2026-11-17 | A 1 Hz sampler judges each run's window. The disqualifying bits stay the same. Each run's 0x200/0x400 and 0x4 shares must stay within 0.10 of the session median per GPU. | `20261117e`, `20261117g`: single samples failed hosts whose windows were clean on drivers 615 and later | coordinator, before the next session |
| 2026-11-18 | Shares are compared only between rounds of one configuration. More than 0.15 apart makes the round furthest from the median unclean. | `s2d-20261118-step0`: the cross-configuration rule marked every round unclean because the shapes load the GPUs differently by design. That session stays null. | coordinator, before the next session |
| 2026-11-18 | Power-limit gate at SSH: every GPU at its default power limit. | `s2f-20261118`: both GPUs were capped at about 250 W, and layer-2 beat the single GPU at batch 1 | coordinator, before the next session |
| 2026-11-18 | The 15% rule judges the analytic prediction made before the candidates run. It does not judge lines fitted from runs of each candidate (ADR 0076 not adopted). Reopen criteria fixed. | `s2g-20261118` misses | owner |
| 2026-11-19 | 15% stays on every candidate. Calibration gains recorded routing frequencies and, on 4-GPU boxes, tp-4 anchors at L/4 and L/2 (ADR 0078). | `s3b-20261119` ep and tp2-pp2 misses | owner |
| 2026-11-19 | Phase 5 verdict sessions skip host 20210 (`docs/box-driver.md`). Diagnostics may use it. | `s3b-20261119`, `s3c-20261119`: GPU 0 shows thermal 0x20 on the 32B in every session, so D4 never gets two clean rounds | coordinator (host choice, not a rule) |
| 2026-11-20 | gpu-parity gate: adds the ggml-sys capture-flag test (`gpu_no_capture_flag_skips_the_cuda_graph`, ADR 0081) and skips the KV save timing tests (`save_through_pinned_staging`, ADR 0074, judged in their own sessions). | `s3f-20261120`: the 8B save timing test missed its bound on host memory speed and ended the session at the gate | coordinator, before the next session |
| 2026-11-21 | Anchor `layer-g@L/2` on 3 or more GPUs (tg only, held out of the fit) to measure the low decode clock (ADR 0082, cost model 5). | s3c 30B layer-4 tg +0.167: GPUs busy under about a third of a decode step stay at 1695 MHz while single boosts to 1905 | owner, by delegation to the advisor |
| 2026-11-21 | Host 20210 re-admitted for verdict sessions, 4-GPU boxes up to 1.22 USD/h all-in (host choice and cap, not a judging rule). | 0 thermal 0x20 samples on all GPUs in `s3f2-20261120` and `s3g-20261121`; no other usable 4-GPU host on offer (hosts failed pulls, SSH or power) | owner |
| 2026-11-21 | D2 and M2 may run on two GPUs of a 4-GPU box (`P5_DEVICES`); both engines use the same pair; the power gate, throttle samples and clock logs judge only that pair, and a run that touches another GPU is unclean. | six 2-GPU hosts failed pulls, SSH or the power gate on 2026-11-20 and 21 | owner (by delegation to the advisor) |

## Changes to what is judged, not to the rules

- Decode toggles TETRAEDER_RANGE_UID and TETRAEDER_ASYNC_INPUTS are on by default (ADR 0069 History). They met their keep rule in `s3c-20261119-decode-decode`, so the judged executor is the one with both on.
- Cost model versions 2 to 4 (ADR 0070 History, ADR 0077, ADR 0080) change the predictions, not the 15% line. Version 4 adds a floor guard on all-reduce curves whose first point is not rising (ADR 0080). The guard changes how calibration data is read, so it is listed here.

## D4 host (owner, 2026-11-20)

One-off cap raise: one V4a box on a host other than 20210, at up to 1.70 USD/h all-in, run alone (AGENTS.md, Money). The rules are unchanged.

Fallback sizing for D4 on host 20210 (from the sessions' `verdict.json` throttle blocks): 32B runs with a disqualifying 0x20 sample on GPU 0 were 0 of 7 (`s3-20261119`), 13 of 25 (`s3b-20261119`) and 4 of 17 (`s3c-20261119`); 30B runs 1 of 7, 0 of 13, 0 of 13. s3c missed a D4 verdict by one clean round of one baseline (`-sm tensor`) at ROUNDS=2 with up to two reruns. At a clean share of 0.76 (s3c), four rounds give every one of about seven 32B configurations two clean rounds with probability about 0.72, five rounds about 0.91; at 0.48 (s3b) even six rounds give about 0.38. So D4 on 20210 is a coin on the night, not a matter of reruns alone.
