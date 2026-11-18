# Session 20261117g

- Status: null, unclean: partial attempt 1 of Phase 4 D2a, stopped at TTFT with the before throttle sample outside 0x3 under the single-sample rule then in force (ADR 0065). Its numbers do not count.
- Code and box: d234c1a (results commit cbee1a7); box 61068300, RTX 3090, driver 615.78.08.
- Raw files: this directory.

## From ADR 0065 (addendum "BoardLimit and Reliability bits judged under load", moved verbatim)

- On the drivers that report them the single "before" sample is not the measured run: in `20261117g` it read 0x600 at 165 W draw (idle is about 28 W), the "after" sample at full load (346 W) read 0x4 without them, and in one pair the judged side read 0x0 while baseline and control read 0x600 and 0x400.
