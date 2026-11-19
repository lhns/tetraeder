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

## Changes to what is judged, not to the rules

- Decode toggles TETRAEDER_RANGE_UID and TETRAEDER_ASYNC_INPUTS are on by default (ADR 0069 History). They met their keep rule in `s3c-20261119-decode-decode`, so the judged executor is the one with both on.
- Cost model versions 2 to 4 (ADR 0070 History, ADR 0077, ADR 0080) change the predictions, not the 15% line. Version 4 adds a floor guard on all-reduce curves whose first point is not rising (ADR 0080). The guard changes how calibration data is read, so it is listed here.

## Open, owner's call

- D4 needs two clean rounds on a 4-GPU host other than 20210, and so far all usable 4-GPU supply is that one host. The options:
  - wait for a second host;
  - one-off cap raise for a host above 1.20 USD/h;
  - run D4 alone on 20210 with more reruns. Reruns would not change the rule: one 0x20 sample still voids a run (`docs/m2-plan.md`, Phase 5 items 1 and 3).
