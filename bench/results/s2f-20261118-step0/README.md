# Session s2f-20261118-step0

- Status: clean. ADR 0069 step-0 check passes on both models: `step0_pass` true, two clean pairs each, threshold 0.92 (provisional band 0.08).
- Code and box: cf3cffb, box 61273508, host 522460 (Florida), 2x RTX 3090, driver 595.91.07, no P2P, NCCL. Same box as s2f-20261118, run right after it.
- Raw files: phase5/. Gates ok (gpu_parity, comm_probe, baseline_modes).

## Findings

Own tp-2 against llama.cpp `-sm tensor` (pp / tg tok/s, ratio own / baseline):
- 30B-A3B round 1: 4342 / 181.3 vs 4000 / 180.6, ratio 1.085 / 1.004; round 2: 4378 / 181.6 vs 4030 / 180.3, 1.087 / 1.007.
- 32B round 1: 1335 / 53.67 vs 1310 / 53.85, 1.019 / 0.997; round 2: 1333 / 53.81 vs 1313 / 53.78, 1.015 / 1.000.
- All four pairs clean (no disqualifying throttle bits); minimum ratios 1.085 / 1.004 (30B) and 1.015 / 0.997 (32B).
