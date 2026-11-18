# Session 20261116d2a2

- Status: null (30B pairs unclean on throttle under the single before-sample rule then in force), c4 miss: Phase 4 D2a rerun, not a pass (ADR 0065). Confirms the MoE output-row fix on GPU (ADR 0071).
- Code and box: commit unknown (env.json `git_commit` empty); box 60919833, host 159250, RTX 3090, driver 615.71.09.
- Raw files: this directory.

## From docs/m2-plan.md, section "D2a rerun result and the c4 gap" (moved verbatim)

- Parity closed on GPU for both models; c4 misses (8B probe 0.830, 30B pairs 0.845 to 0.853), c1 level. Details in the ADR 0065 addendum of the same date.
- The dense "slowdown between boxes" was the first box, not the code: on one 3090, dense against libllama per step is 0.99 to 1.03 at both 030d1f3 and HEAD (`dense-regress/table.md`).
