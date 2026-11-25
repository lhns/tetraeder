# Phase 6 rule changes

Changes to judged rules, host choice or what is judged after ADR 0091 and ADR 0092 were written, in the order made. Each says whether it was made before or after a verdict.

| Date | Change | Made | Reason | Status |
|---|---|---|---|---|
| 2026-11-24 | Anchors on real hidden rows; a fit band limit refuses the freeze (ADR 0091 answer 7) | after a no-verdict session (`segment-confirm-20261124`) | fixed-seed anchors routed differently (pair share 0.80 against 0.39, band 1.23 against 0.07) | owner decided |
| 2026-11-24 | One NUMA node host, early A/A abort, no prefill noise band (ADR 0091 answer 8) | after the same no-verdict session | prefill spread 8.6x on a two-socket host | owner decided |
| 2026-11-25 | Fit band limit 0.10 (ADR 0092 rerun rules) | before the rerun | a refused freeze is cheap, a judged miss sends the ADR back | owner decided |
| 2026-11-25 | Unclean rounds are rerun on the same frozen prediction up to 8 rounds, all reported, A/A last stays last (ADR 0092 rerun rules) | after a pass (`segment-confirm-20261125b`) | its first judged attempt had 1 clean round of 3; the driver ran a second attempt; the all-round medians match the clean ones | ratified by the owner 2026-11-25 |
| 2026-11-25 | Case S link rule: the link is probed at start, before each judged round and at end; the frozen judged plan is priced again with the end probe's link curves (everything else frozen) and the session is clean only if predicted pp and tg each move by at most 0.08; a pass must hold against the frozen and the end-priced prediction, else no verdict; raw probe points are reported, not gated. Also a report-only prediction priced from a second prompt's routing trace, recorded before the judged runs, reported beside the judged errors | before any Case S verdict | the rehearsal's 57 to 68 ms WAN round trip moves the 16 KiB point by 12 to 15% between probes, so the raw-point band of ADR 0091 would make a WAN session unclean for noise the prediction does not feel; looser for small messages than the raw-point band, tied to what the prediction uses | advisor, delegated by the owner (2026-11-25) |
