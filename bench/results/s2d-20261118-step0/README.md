# Session s2d-20261118-step0

- Status: null: ADR 0069 step-0 check, every round unclean under ADR 0068's 2026-11-17 cross-configuration throttle rule (since replaced) and a co-tenant on host 34742, so `step0_pass` is null; its ratios back only a provisional proceed (ADR 0069), not a pass. GPU parity gates passed.
- Code and box: 723519f (llama.cpp baseline fc504f9); box 61185926, host 34742, 2x RTX 3090, no P2P, NCCL, driver 615.71.09.
- Raw files: this directory.

## From ADR 0068 (addendum "throttle shares are compared per configuration across rounds", moved verbatim)

- The 2026-11-17 rule (each run's 0x200/0x400 and 0x4 shares within 0.10 of the session median per GPU) marked every S2d step-0 round unclean, including runs that agreed with their pair (`bench/results/s2d-20261118-step0/`: 30B round 1, 0.47 against 0.50). In that session's samples the bits follow load level, not slowdown: 0x200/0x400 on 30 to 80% of samples at 100 to 300 W with SM clocks of 1800 to 1965 MHz, 0x4 at idle and above 300 W. Phase 5 candidates load the GPUs differently by design, so a comparison across configurations measures the shape, not cleanliness.
- Rule from the next session on: disqualifying bits in a run's window (outside 0x1, 0x2, 0x4, 0x100, 0x200, 0x400) make it unclean, as before. 0x4, 0x200 and 0x400 are recorded per run and GPU and never compared across configurations. A configuration whose share of 0x200/0x400 or of 0x4 on a GPU differs between its own rounds by more than 0.15 is unclean in the round furthest from its median (S2d: at most 0.07 between rounds of one configuration). The A/A run stays the session check. With one round, only the disqualifying-bit rule applies.
- `s2d-20261118-step0` stays as judged (null); the co-tenant guard is unchanged (host 34742 tripped it in two sessions: fine for gates, poor for judged runs).

## From ADR 0069 (addendum "step-0 provisional proceed", moved verbatim)

- Own `tp-2` against llama.cpp `-sm tensor`, two interleaved rounds: Qwen3-32B pp 0.968 and 1.018, tg 0.991 and 1.001; Qwen3-30B-A3B pp 1.086 and 1.086, tg 1.030 and 1.013. All eight at or above `1 - b` (0.92, provisional band).
- Every round was marked unclean by the harness (the cross-configuration throttle share rule, replaced in ADR 0068's 2026-11-18 addendum, and a co-tenant on the host), so `step0_pass` is null. The step-0 text above judges two interleaved pairs against `1 - b` and does not invoke ADR 0068's clean rule; the decision is a provisional proceed: no meta-device redesign. The pairs rerun on the rehearsal box; V2's baseline gate judges the pick strictly.
- All GPU parity gates passed on this box, including `gpu2_moe_shapes_match_single` against the anchored bound (tp-2 0.83x, ep-2 0.96x, layer-2 0x of the bound; anchor 3.99e-2 of the largest logit).
