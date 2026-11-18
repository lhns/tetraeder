# Session 20261117k

- Status: clean; Phase 4 D2a throughput verdict of record: both models pass under ADR 0065 (line 0.925, sampler throttle rule).
- Code and box: main 6b07795 (results commit 8b6db93); box 61097370, host 49107, RTX 3090, driver 615.78.08.
- Raw files: this directory.

## From docs/m2-plan.md, section "D2a verdict of record" (moved verbatim)

- Both models pass on main 6b07795: worst pairs 8B c4 1.021, c1 0.995; 30B c4 0.943, c1 0.989 (line 0.925). The way there: huge-page advice on the save path stalled admissions (`20261117d`, `h`), so the pool prepares plain spares off the admission path; the dense backend now reports its row size; the throttle rule judges BoardLimit and Reliability by a 1 Hz sampler under load (ADR 0065); `20261117e` and `g` stay null.

## From ADR 0065 (addendum "D2a passes on main", moved verbatim)

- Code: origin/main 6b07795 (the driver's checkout; `env.json` has an empty `git_commit` because `phase4_box_d2a.sh` ignored `COMMIT`, fixed since). It holds the step budget, the pin b12426, the mask padding, the snapshot pool with huge-page advice off, and the session sampler.
- Probes: 8B c4 1.000, c1 1.019; 30B c4 0.929, c1 1.008 (line 0.925). Two clean pairs each, no third: 8B c4 1.022 and 1.021, c1 0.995 and 0.997; 30B c4 0.943 and 0.948, c1 0.989 and 0.990. A/A within 0.003 on both. Per-model pass true for both against 1 minus the band (0.925, band 0.0753).
- Throttle by the sampler rule on all four pairs: 0 disqualifying samples; 0x200/0x400 shares 0.167 against 0.111 in 8B pair 1 and 0 elsewhere. The idle gate read 0x4 on 8 of 10 samples at idle.
- No compaction during the session (`compact_stall` and `allocstall` +0).
- This is the throughput verdict of record for both models. `phase4_pass` stays null until the TTFT pairs (D2b, session `20261117l`) report.
