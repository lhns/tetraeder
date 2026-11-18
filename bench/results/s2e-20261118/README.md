# Session s2e-20261118

- Status: Phase 5 S2 rehearsal, stopped at `calibrate` (no predictions, rounds or verdict).
- Code and box: 840d9b3 plus e6e5775, box 61255543, host 608696, 2x RTX 3090, driver 580.173.02.
- Raw files: phase5/.

## Findings

- Anchors (first real ones; pp / tg tok/s): 30B single L12 19103 / 599, L24 10002 / 349; tp-2 L12 17085 / 614, L24 9160 / 348. 32B single L16 4850 / 128, L32 2374 / 68; tp-2 L16 5811 / 209, L32 2883 / 113.
- Calibrate failed on the anchor rows' content-hash `plan_id` (fixed, e6e5775), then on "prior lacks cpu compute" (fixed: the fit needs CPU compute only when an anchor uses the CPU, `track-calib-gpu-only`).
