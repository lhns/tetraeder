# Session 20261117k

- Status: clean; Phase 4 D2a throughput verdict of record: both models pass under ADR 0065 (line 0.925, sampler throttle rule).
- Code and box: main 6b07795 (results commit 8b6db93); box 61097370, host 49107, RTX 3090, driver 615.78.08.
- Raw files: this directory.

## From docs/m2-plan.md, section "D2a verdict of record" (moved verbatim)

- Both models pass on main 6b07795: worst pairs 8B c4 1.021, c1 0.995; 30B c4 0.943, c1 0.989 (line 0.925). The way there: huge-page advice on the save path stalled admissions (`20261117d`, `h`), so the pool prepares plain spares off the admission path; the dense backend now reports its row size; the throttle rule judges BoardLimit and Reliability by a 1 Hz sampler under load (ADR 0065); `20261117e` and `g` stay null.
