# Session 20261115p1

- Status: clean for polling and C4: poll 50 loses on this host class, so ADR 0061's default stays; FP8 KV baseline unusable at b12274.
- Code and box: commit on main at the time of box 60734395 (see git log of the results commit); host 79986, RTX 4090, EPYC 7702, quota 30.6, 24 threads.
- Raw files: this directory.

## From docs/m2-plan.md, section "Point-1 box result" (moved verbatim)

- Polling: poll 50 against the auto default (0 on this quota box) 0.85 and 0.95, throttled 0.92 to 0.95 of decode periods against 0.05 to 0.08. With 1.16 to 1.22x on the Zen 4 host (26 threads, quota 30.7), the effect depends on the host class, not on headroom alone. ADR 0061's default is not changed (B3 closed: no rule change).
- C4 on the 235B: main-overlap 2.83 and 2.71 tok/s against 3.02 and 2.74 on `20261113b` (same profile, other host): no sign of a regression.
- FP8 KV baseline: llama-server at b12274 aborts (`cache_k_l0 (view)` in CUDA0 cannot run SET_ROWS with f8_e4m3 K and V): the FP8 cache is not usable on CUDA at the baseline pin. Item 9 is parked until a baseline pin where CUDA writes FP8 KV; the builder's `kvq8-slots` ran (2.75 tok/s, 0.97 of main-overlap) but has nothing to pair with.
- Cost 5.2 USD, 4.3 of it model traffic at 0.029 USD per GB: drivers must rank offers by box plus traffic (`docs/box-driver.md` rule 2).
