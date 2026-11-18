# Session 20261110c

- Status: clean paired run (b1b-levers): ADR 0054's decode bar met, its hidden share bar missed; the ADR 0054 overlap path ahead of KTransformers on AVX2, but CPU not matched (I/O workers beyond the 26 threads), closed by `20261110e`.
- Code and box: results commits 35f424d and 023f2cc; box 60057461, RTX 3090, AMD EPYC 7742 (AVX2), quota 30.72, 26 threads, 91 GB balloon, driver 595.71.05.
- Raw files: this directory.

## From ADR 0045 (addendum "an AVX2 box needs the ADR 0054 overlap", moved verbatim)

## Addendum (2026-11-10, box 60057461): an AVX2 box needs the ADR 0054 overlap

Session `b1b-levers`: AMD EPYC 7742 (AVX2 only, so no iqk VNNI kernels on either engine), 26 threads, 91 GB. KTransformers ran its `avx2` variant (spread 6.3%). Decode builder/KT: `main-prefetch` 0.92, `main-repeat` 0.89, so the b1-exit path is behind on this CPU class; with ADR 0054's I/O service and overlap 1.27 and 1.21 (`main-iosvc` 1.21). Prefill 2.1x to 2.6x. KT decodes with 171k major faults per run and reads 161 to 171 MB per token; the builder reads 550 to 565 MB.

- Not pooled with b1-exit: different CPU class.
- The verdict holds on both classes only with overlap, which `serve` now enables by default (ADR 0054 addendum). The builder's I/O workers used CPU beyond the 26 threads, which KTransformers did not; a quota-tight run (quota about threads plus one, as b1-exit) of `main-overlap` against KTransformers closes that question.

## From ADR 0045 (addendum "owner decision on slow disks; correction", 2026-11-11, the correction paragraph, moved verbatim)

Correction to the box 60190564 addendum: the extra decode bytes are not mainly prefetch. On `20261110c` (same box and options) the 64-token runs read 549 MB per token and the 256-token trace run 262, so tokens 65 to 256 read about 167 MB per token, the same as KTransformers; without prefetch the first 64 tokens still read 443 (`20261103`). The builder's excess is warm-up of the RAM tier in the first tokens after the 32K prefill; KTransformers' page cache shows none. Cause under diagnosis; the replay sentence above stands as data, not as the lever.

## From ADR 0050 (addendum 2026-11-14 "RAM-tier policies across layers in the replay", findings on this session's trace moved verbatim)

- The 256-token trace run of `20261110c` (prefetch, no overlap) read 262 MB per token. Its replay gives 262.3 for `lru` at VRAM 10, RAM 77 with prefetch, and a decode RAM hit rate of 0.9637 against 0.9637 measured. The same configuration at 64 tokens (`main-prefetch`, same session) read 549 MB per token, so tokens 64 to 255 read about 167 MB per token. Most of the excess is in the first 32 tokens after the prefill, not in the steady state.
- On the same trace, per-layer `belady` without prefetch reads 39.9 MB per token against 186.8 for `lru` without prefetch. Per-layer slots do not bound the result; the eviction choice and the tier's contents when decode starts do.

## From ADR 0054 Implementation notes, "Defaults in serve" (moved verbatim)

Defaults in serve: `tetraeder serve` takes `--io-service on|off` and `--overlap on|off` (also with `--plan`). Unset, the service is on when the builder has a RAM tier (`--ram-slots`, or `ram_slots` in the plan) and overlap is on when the service is on and the miss policy is `cpu` (a plan with a RAM tier always misses on the CPU). Without a RAM tier both are off. `--overlap on` without the service, or without a RAM tier, is a CLI error. The box pairs on the 235B (2026-11-10) gave overlap 1.38 to 1.40x the prefetch path and the service alone 1.32x. The old `Reader` path stays for `--io-service off`. The metrics above are exported through the load options' telemetry, so the panel and alert cover a served run. `--prefetch on|off` follows the same rule (on with a RAM tier, as in every measured 235B run) and `--prefetch-depth` defaults to 2, the depth of those runs.

## From ADR 0054 (addendum "paired result, defaults on", moved verbatim)

Session `b1b-levers`: RTX 3090, AMD EPYC 7742 (AVX2, no AVX-512), quota 30.7 cores, 26 threads, RAM ballooned to 91 GB, Qwen3-235B-A22B at 32K. Decode against `main-prefetch` on this box: `main-iosvc` 1.32x, `main-overlap` 1.38x and 1.40x (1.37x against the adjacent `main-repeat`); prefill 1.11x to 1.15x. Builder noise 2.5%, A/A pair 1.000.

- Decode bar (at least 15% above `main-prefetch`): met. Hidden share bar (0.5): missed, 0.42. Read wait is still 24.8 ms of a 42.8 ms read span per token, all urgent misses.
- Confound: the builder's CPU time was 106 to 116% of 26 threads, so the I/O workers used 2 to 4 of the 4.7 spare cores of quota. The gain is measured with spare cores, not in the starved case of section "Context". A box with a quota near the thread count (b1-exit had 26.9) must confirm it.
- Decision: the service and overlap become the defaults of `serve` where they apply (RAM tier; overlap also needs the `cpu` miss policy). The `Reader` path stays until the starved-case run, then is deleted.

## From ADR 0054 (addendum "where the read wait goes", moved verbatim; also uses `20261110e` and `20261110f`)

From the committed overlap runs (`bench/results/20261110c`, `e`, `f`; only host 43fae09577cc of `c` has a disk curve):

- The wait is not all urgent misses: `urgent_wait_time` counts every wait in `wait_landed`, late prefetches included. Per token 6.7 experts (78 MB) are misses, but 22.7 are pending at their layer's plan: about 16 are prefetches still in flight, 39% of the 41.2 issued.
- The disk is not the bound: decode reads 2.86 GB/s; the same host's curve gives 7.0 to 7.3 GB/s from depth 4 on, also with 25 busy threads.
- A miss lands 3.9 ms after its submit on average (one alone needs 1.6 ms of device time: 11.6 MB at 7.2 GB/s), and `cpu_ready` hides about 0.9 ms per pending step, so steps with a miss stay near or under 0.5 even at device rate. A prefetch lands 6.3 ms after its submit, against a lead of about one layer step (1.8 ms) for a depth-1 top-up and two for depth 2. Both are means of per-unit latencies, which overlap.
- Candidate causes: a layer step submits its misses after the next layers' prefetch, and with 16 workers every chunk of a layer step is in service at once, so deadline order never applies and the device shares its rate among them; and the workers' CPU under the quota while the ggml threads poll.

## From ADR 0058 Context (moved verbatim; session inferred from ADR 0045 addendum 2026-11-11 "owner decision on slow disks; correction")

- **Demand over time:** disk bytes per token per decode window, for every engine. The builder read 3.5x KTransformers' bytes per token in the first 64 tokens after a 32K prefill and the same bytes afterwards; one window hid that.
