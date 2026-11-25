# Phase 6 rule changes

Changes to judged rules, host choice or what is judged after ADR 0091 and ADR 0092 were written, in the order made. Each says whether it was made before or after a verdict.

| Date | Change | Made | Reason | Status |
|---|---|---|---|---|
| 2026-11-24 | Anchors on real hidden rows; a fit band limit refuses the freeze (ADR 0091 answer 7) | after a no-verdict session (`segment-confirm-20261124`) | fixed-seed anchors routed differently (pair share 0.80 against 0.39, band 1.23 against 0.07) | owner decided |
| 2026-11-24 | One NUMA node host, early A/A abort, no prefill noise band (ADR 0091 answer 8) | after the same no-verdict session | prefill spread 8.6x on a two-socket host | owner decided |
| 2026-11-25 | Fit band limit 0.10 (ADR 0092 rerun rules) | before the rerun | a refused freeze is cheap, a judged miss sends the ADR back | owner decided |
| 2026-11-25 | Unclean rounds are rerun on the same frozen prediction up to 8 rounds, all reported, A/A last stays last (ADR 0092 rerun rules) | after a pass (`segment-confirm-20261125b`) | its first judged attempt had 1 clean round of 3; the driver ran a second attempt; the all-round medians match the clean ones | ratified by the owner 2026-11-25 |
| 2026-11-25 | Case S link rule: end-probe re-pricing within 0.08 instead of a raw-point band (ADR 0091 answer 10) | before any Case S verdict | raw small-message latency drifted 12 to 15% on the rehearsal WAN link without changing what the prediction needs; looser for small messages | advisor, delegated by the owner |
