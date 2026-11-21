# Phase 5 verdict (merged)

- Status: **phase5_pass true** (ADR 0068): all four cases pass.
- Merged with `scripts/phase5_summary.py --merge` from `v4a-20261121` (D4, M4; host 20210, 4x RTX 3090) and `v2c-20261121` (D2, M2; GPUs 2 and 3 of the same box, `P5_DEVICES`, ADR 0068 History 2026-11-21).
- D4: pick tp-4, winner set {tp-4, tp2-pp2}, baseline gate 1.022 (llama.cpp `tensor`). M4: pick layer-4, winner set {layer-4}, gate 0.987 (llama.cpp `layer`). D2: pick tp-2, winner set {tp-2}, gate 1.011. M2: pick layer-2, winner set {layer-2}, gate 1.008. Every candidate within 15% of its prediction on pp and tg; three clean rounds per configuration in both sessions; A/A within 0.2%.
- Cost model 5 (ADR 0082) applied: 30B k = 1.062 (busy 1800, low 1695 MHz), held-out layer anchor -0.024.
- Rule changes made after failed sessions, for the owner: `docs/notes/phase5-rule-changes.md`. Still open before M2 is reported: the D2a probe on final main (docs/m2-plan.md item 5).
