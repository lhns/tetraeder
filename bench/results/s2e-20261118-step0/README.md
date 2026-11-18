# Session s2e-20261118-step0

- Status: ADR 0069 step-0 check. Qwen3-30B-A3B: two clean pairs, pass. Qwen3-32B: both pairs unclean (one 0x20 SW thermal slowdown sample per round, a disqualifying bit under ADR 0068), so `step0_pass` is null.
- Code and box: 840d9b3, box 61255543, host 608696 (New York), 2x RTX 3090, driver 580.173.02, no P2P, NCCL.
- Raw files: phase5/.

## Findings

- Own tp-2 against llama.cpp `-sm tensor`, pp / tg ratio: 30B round 1 1.086 / 1.015, round 2 1.077 / 1.010 (clean). 32B round 1 1.02 / 1.00, round 2 1.02 / 1.00 (unclean, raw rates 1438 vs 1415 and 1443 vs 1410 pp, 58.2 vs 57.9 and 58.1 vs 58.1 tg). Threshold 0.92 (provisional band 0.08).
- All GPU gates passed; `gpu2_moe_shapes_match_single` worst ratios tp-2 0.83x, ep-2 0.96x, layer-2 0x of the bound (anchor 3.99e-2 of the largest logit).
- The 32B download reset three times at about 94.6% on this host; refetched from Hugging Face with 6 connections.
