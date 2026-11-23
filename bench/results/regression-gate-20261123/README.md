# Session regression-gate-20261123

- Status: mixed. By the rule fixed before the run, 3 of 4 metrics read no regression evidence (Qwen3-30B-A3B c4 0.982 and c1 0.991, Qwen3-8B c1 1.006) and Qwen3-8B c4 is inconclusive (median 0.970 over 2 clean rounds, 0.848 and 1.091, one round lost to a co-tenant flag); TTFT B/A is 1.019 at the median and 1.000 at p90, inside the 1.05 line, so no regression. Three more rounds run afterwards on the same box (not the verdict) put all four medians at 0.985 to 1.000.
- Purpose: node `bench-regression-gate`. Since the M2 numbers (A = `f68d2a0`, this morning's build) main gained the llama-server parity work (sampler on libllama's chain with a plain-greedy fast path, request fields, endpoints, group D), the prefix disk tier, KV format auto, recurrent-reuse handling and two review rounds. Question: did the own engines regress in throughput or TTFT? Self A/B on one box; nothing is judged against llama-server.
- A: `f68d2a0c71035a42eabcba1799e689b2f6e779c9`, built on the box from a `git archive` tarball (the push paths of `taxameter.toml`) in `/root/tetraeder-A` with its own target dir. B: main `41dcedd2` plus this branch's commit `a616ebd6` (only `scripts/`, `tests/test_ab_moe_box.py` and this README differ from main; the three later commits changed scripts and tests only). `vendor/PIN` and `patches/` are identical in A and B (`git diff f68d2a0 origin/main -- vendor/PIN patches` is empty), so both build the same ggml at fc504f91. Both release builds with CUDA (arch 80), built in parallel (A 4m34s, B 5m01s, `build-A.log`, `build-B.log`).
- Code: `scripts/gate_box.sh` ran `scripts/ab_moe_box.sh` (gate and TTFT; summary `scripts/ab_moe_summary.py`) and then `scripts/gate_extra_box.sh` (llama-server CUDA build, parity on CUDA, Qwen3.6 default flags); then `scripts/gate_kv_box.sh` (report-only kv and disk-tier runs).

## Criteria (fixed before the run)

Gate, as `bench/results/ab30b-20261122/README.md`, per model and concurrency (four metrics):

- Workload per run, one fresh server process per run: Qwen3-30B-A3B Q4_K_M `--engine moe`, Qwen3-8B Q4_K_M `--engine dense` (the Phase 4 dense flags of `scripts/phase4_box_d2a.sh`), both `--ngl -1 --parallel 4 --ctx 32768 -t <CPU quota> --accept-any-model` and otherwise default flags; one warm-up request, then c4 `--prompt-tokens 512 --max-tokens 128 --concurrency 4 --requests 16`, then c1 the same with `--concurrency 1 --requests 8`. Seeds per round (c4 round*10+4, c1 round*10+1), the same on A and B.
- Order per model: three rounds A B, B A, A B (one run per side and round). Gating fields: c4 `throughput_tps`, c1 `decode_tps_median`; round ratio = B / A.
- A run is clean when errors are 0 in both tests, no co-tenant flag, and the 1 Hz sampler shows no window sample with a bit outside 0x1, 0x2, 0x4, 0x100, 0x200, 0x400. Unclean rounds are listed with the reason and excluded from the median.
- Reading per metric, by the median B/A over the clean rounds: at least 0.98 is no regression evidence; below 0.97 with every clean round below 1 is a regression; anything else, and always with fewer than 2 clean rounds, is inconclusive.
- Outcome of the node: successful if all four metrics read no regression evidence, mixed if any is inconclusive, bad if any is a regression (a defect node then names the model and concurrency).

TTFT, one replay per build on Qwen3-8B (`--engine dense --ngl -1 --parallel 4 --ctx 40960 --cache-ram 8192 -t <quota> --accept-any-model`, as `scripts/ttft_diag.sh`), the request log of ttft4 to ttft7 (sha256 f79b4a6d...5f7b35, checked on the box before the push and again by the script; 105 requests, 95 streamed with status 200), `scripts/gateway_replay.py --speed 0 --concurrency 1`, A then B on the same box: median and p90 TTFT over the streamed requests with status 200 on both sides; B/A of each at most 1.05 means no regression, else reported; any request error in either replay is reported too.

Other jobs (reported, not judged): parity cases against llama-server and tetraeder built with CUDA, counts compared with the CPU CI run; Qwen3.6-35B-A3B UD-Q4_K_XL `serve --engine auto` with default flags, expected zero errors and 128 tokens per request; `--kv auto` against `--kv f16`; the prefix disk tier.

Not judged: any ratio against llama-server, absolute tok/s across boxes (shown for orientation only). The box is an A100, not the RTX 3090 of `ab30b`; only the paired ratios of this box count.

## Box

- Box 62242538, host 224078 (Slovenia), 1x NVIDIA A100-SXM4-40GB (compute 8.0), driver 570.133.20, power limit 400 W = default (power gate ok), AMD EPYC 7713 (64 cores, 256 host CPUs, cgroup quota 30, threads 30 on both sides and both engines), 1 TB host RAM, 150 GB disk, 0.694 USD/h (price includes disk; traffic 0.004 USD/GB). Gates at SSH: `cuda: ok`, power gate, idle gate 10/10, co-tenant sample 8.33 busy host cores of 256 over 20 s (limit max(4, 15% of 128 physical cores) = 19.2). Baseline `/opt/llama.cpp-baseline` reports b12426 fc504f91 (not used by the gate).
- Clocks at start 210 MHz idle of 1410 MHz max, memory 1215 MHz (`ab/env.json`, `ab/nvidia-smi-clock-performance.txt`). Throttle reasons before every run 0x0 (24 snapshots), after 0x0 (17) or 0x4 (power cap, 7) (`ab/gpu-*.txt`); the 1 Hz sampler (`ab/gpu-samples.csv`) shows no window sample with a disqualifying bit, board-limit share 0.00 on every run.
- Life: created 22:16:40, destroyed 00:24:29 (box age 2h08m, local clock; the box clock runs one hour behind), credit 29.54 to 28.09 USD (1.45 USD with traffic of about 55 GB). A first box (62240933, A40, host 682584, 0.588 USD/h) was destroyed after 8 minutes at the co-tenant gate (about 60 busy host cores of 256 over 20 s, measured twice; limit 19.2), 0.27 USD, nothing run on it. Destroyed after verified fetches (`results-62242538-20261123-233900.tgz`, `-234717`, `-000722`, `-002417`).
- The host is noisy: the co-tenant guard flagged Qwen3-8B rounds 1 and 4 (23.5 and 24.2 busy cores outside the container, limit 19.2), and the absolute tok/s of one side moves by up to 15% between rounds on both A and B (for example 8B c4 A 271.6 to 334.4).

## Gate results (rule as fixed: rounds 1 to 3)

Source: `ab/judged3-summary-q30b.json`, `ab/judged3-summary-q8b.json` (from `ab/ab-<model>-<A|B>-r<n>-<c4|c1>.json`, `scripts/ab_moe_summary.py`). tok/s: c4 `throughput_tps`, c1 `decode_tps_median`.

| model | metric | rounds B/A (clean rounds) | median | reading |
|---|---|---|---|---|
| Qwen3-30B-A3B moe | c4 | 0.998, 0.938, 0.982 | 0.982 | no regression evidence |
| Qwen3-30B-A3B moe | c1 | 0.991, 0.953, 1.008 | 0.991 | no regression evidence |
| Qwen3-8B dense | c4 | round 1 unclean; 0.848, 1.091 | 0.970 | inconclusive (a round at or above 1 and a median between the lines) |
| Qwen3-8B dense | c1 | round 1 unclean; 0.969, 1.043 | 1.006 | no regression evidence |

Per-round tok/s (A, B): 30B c4 276.8/276.2, 274.1/257.2, 264.4/259.6; 30B c1 158.1/156.7, 149.6/142.5, 152.4/153.6; 8B c4 (r1 unclean) 334.4/329.8, 332.7/282.2, 301.5/328.9; 8B c1 137.2/136.4, 136.7/132.5, 130.4/136.0. Order A B, B A, A B. Round 1 of the 8B was unclean only by the co-tenant flag on A (23.5 busy cores); its ratios (0.986 and 0.994) are listed in the summary and excluded from the median.

Outcome by the rule: mixed (one metric inconclusive, no regression anywhere). The 8B c4 ratios swing 0.848 to 1.091 with A moving as much as B (332.7 then 301.5 on A), so the spread is the host's, not a direction.

### Rounds 4 to 6 (after the verdict, same box and builds, not the verdict)

Added after seeing round 1 to 3, to see whether the 8B c4 inconclusive was noise (`ab/ab-summary-q30b.json`, `ab/ab-summary-q8b.json`, 6 rounds, orders B A, A B, B A). B/A c4 / c1 per round, 30B: 1.024 / 1.013, 0.946 / 0.969, 1.071 / 1.010; 8B: round 4 unclean (co-tenant 24.2 cores) 1.125 / 1.029, 1.136 / 1.000, 0.885 / 0.957. Medians over all clean rounds: 30B c4 0.990 (6 rounds), c1 1.000 (6); 8B c4 0.988 (rounds 2, 3, 5, 6; 0.848, 1.091, 1.136, 0.885), c1 0.985 (same rounds). Every metric would read no regression evidence on these. The 8B c4 round ratios still scatter 0.848 to 1.136 in both directions, so the 3-round reading above stays the recorded verdict.

## TTFT (A then B, one replay each)

Source: `ab/ab-ttft-A.json`, `ab/ab-ttft-B.json`, `ab/ab-ttft.json`. Over the 95 streamed requests with status 200 on both sides: A median 0.284 s, p90 0.688 s; B median 0.289 s, p90 0.688 s. B/A median 1.019, p90 1.000, both at most 1.05: no regression. Both replays 105 requests, 0 errors, status mismatch 10 against the log (as ttft7). A took 23 min, B 27 min. One replay per side, so the 1.9% median gap has no noise estimate (ttft7's two default rounds agreed within 0.003 s on another host).

## Config diff of A and B

- Same command line, same threads, same model files, same `vendor/PIN` and patch queue (above). The server start lines of A and B are the same (`server-*-A-r1.log` against `-B-`: libllama load warning, readback staging 16 MiB pinned, snapshot pool spares 2, CPU threadpool polling auto, quota 30.72).
- Defaults that differ in the code (`git diff f68d2a0 HEAD -- crates/tetraeder/src/serve.rs engine_args.rs`): B adds the llama-server sampler options (temperature 0.8, top-k 40, top-p 0.95, min-p 0.05, typical 1.0, repeat-last-n 64, and the others of the parity work), `--kv` (default f16), `--cache-disk` and others. The bench sends `temperature 0`, which takes B's plain-greedy fast path; the KV default stays f16 and the disk tier is off by default, so the gate runs the same data path as A except for the new request handling.

## Conformance on CUDA (job 2)

Source: `extra/parity.log` (`-rA`), `extra/parity-tmp/` (server logs), `extra/parity-devices.txt`, `extra/llama-server-version.txt`, `extra/build-llama-server.log`.

- Servers: llama-server built from the patched `vendor/llama.cpp` (fc504f91 plus the patch queue), `GGML_CUDA=ON`, arch 80, static (`version 1.0.0-dev, build 1, commit fc504f9`, 2m32s), and `target/release/tetraeder` (B, CUDA) on the tiny fixtures with `tests/conformance/parity/test_parity.py` (145 cases) with `TETRAEDER_REQUIRE_PARITY=1`. `test_parity_lib.py` was not run on the box: it reads `docs/`, which taxameter does not push; it is CPU-only and CI covers it.
- Result: 135 passed, 6 xfailed, 4 failed in 44 s. The CPU CI run of main `41dcedd2` (all of `tests`, parity included) has 795 passed, 2 skipped, 6 xfailed and no failure; the six xfails are the same `gap-*` cases (`serving-parity-remaining`).
- The 4 failures are all libllama (`auto`) cases and differ from CPU CI, so they are CUDA findings, not session failures:
  - `repeat-penalty[auto]` and `seeded-sampler-names[auto]`: the generated text of the random-weight fixture diverges after a few tokens (temperature 0.8 samples, so one changed float flips a draw).
  - `logprobs[auto]` and `logprobs-shape[auto]`: first-token logprob differs by 0.018 and 0.004 (tolerance 0.001), top-2 by 0.013 and 0.013. The two cases send the same prompt and tetraeder returns -1.8332 in one and -1.8114 in the other (llama-server -1.8150 in both), so the tetraeder value moves between runs, which CPU CI does not show.
- Both servers ran on the GPU as far as the logs show: llama-server logs `layer 0 is assigned to device CUDA0`; the own engines (`dense`, `moe`) log no device line on the tiny fixtures, so for them the GPU use rests on the default `--ngl -1` and on the A/B runs (95% GPU utilization) with the same binary. Not proven per case.

## Qwen3.6-35B-A3B default flags (job 3)

Source: `extra/bench-q36-default-r<n>-<c4|c1>.json`, `extra/q36-summary.json`, `extra/server-q36-default-r<n>.log`. `tetraeder serve -m Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --engine auto --ngl -1 --parallel 4 --ctx 32768 -t 30 --flash-attn on` (default prefix reuse and RAM cache, no workaround flags), two fresh servers, warm-up, then the strata-q36 workload (c4 16 requests, c1 8 requests, 512 prompt and 128 generated tokens).

- Errors 0 in all four tests (two rounds, c4 and c1); no request with fewer than 128 tokens; the client counts error frames and empty streams. The recurrent-reuse failure of `strata-q36-20261123` (`b0-default`: 7 of 16 and 4 of 8 empty streams) is gone.
- Orientation only (A100, not the A40 of strata-q36): r1 c4 237.7 tok/s, c1 decode 132.2 tok/s; r2 c4 215.1, c1 125.9; c4 TTFT median 0.57 and 0.64 s.

## `--kv auto` and the disk tier (job 4, report only)

Source: `kv/` (`kv-probe.txt`, `kv-*.json`, `server-kv-*.log`, `disk-*.json`, `server-disk-*.cache.log`). Qwen3-30B-A3B `--engine moe`, B build.

- Auto picks f16 up to ctx 229376 (no pick line) and q8_0 at 262144: `f16/f16 needs 25.8 GB KV, 23.5 GB free; q8_0/q8_0 fits (13.7 GB KV)`. `--kv f16` at 262144 does not start (cudaMalloc of 24576 MiB failed, out of memory), so there is no paired f16 run at that context. `--kv auto` at 262144 runs clean: c4 240.4 tok/s, c1 decode 131.4 tok/s, 0 errors.
- Paired f16 against q8_0 at ctx 32768, three interleaved rounds (f16 q8, q8 f16, f16 q8; seeds as the gate): c4 throughput f16 262.2, 261.9, 273.3 and q8_0 241.2, 254.7, 239.8 tok/s (q8_0/f16 0.920, 0.973, 0.877, median 0.920); c1 decode f16 153.2, 150.9, 156.6 and q8_0 131.1, 132.8, 130.8 (0.856, 0.880, 0.835, median 0.856). So the quantized KV costs about 8% at c4 and 14% at c1 on this box, and the auto run at 262144 matches the q8_0 numbers at 32768 (240.4 and 131.4); auto leaves f16 only where f16 does not fit. Not judged; ADR 0086 keeps f16 the default.
- Disk tier, two passes (RAM only against RAM plus `--cache-disk`, Qwen3-8B, first 50 requests of the replay log, ctx 40960): `--cache-ram 2048` TTFT median/p90 0.278/0.786 s against 0.274/0.841 s, `--cache-ram 4096` 0.259/0.837 against 0.277/0.813. In neither pass was the tier used: all 50 admits are `source=live` (47) or `copy` (3), no `restore_ms`, so the differences are noise (the wall times differ by 25% in both directions: 491 against 424 s, 389 against 493 s). The replay is one linear session on live slots and never evicts a sequence; measuring restore against recompute needs more sessions than slots. Not measured.

## Deviations

- First box destroyed at the co-tenant gate (above); the second box is an A100 SXM4 (offer 31632904), not an RTX 3090 (the q36 job needs more than 24 GB and no 48 GB offer under the cap passed the gates; A40 at 0.588 USD/h was the first choice).
- The summary stage on the box wrote all-inconclusive summaries (a file-name mismatch in `ab_moe_summary.py` for the per-model names, fixed in a later commit); `ab/ab-summary-*.json` and `ab/judged3-summary-*.json` were regenerated locally from the fetched run files and sampler CSV with the fixed script.
- Rounds 4 to 6 ran after the first verdict (rerun of `ab_moe_box.sh` with `ROUNDS=6 SKIP_BUILD=1 TTFT=0`); reported as a supplement, the rule's verdict stays at three rounds.
- The parity stage ran `test_parity.py` only (see above) and with the venv `pytest`, not `uv run` (pyproject is not pushed).
- The first disk-tier pass used `--cache-ram 2048`, the second 4096; neither evicted (above).
- `docs/features.md` has no row: its table holds paired ratios against other engines and this is a self A/B (B over A of the same engine), so the page has no place for it (`scripts/check_features.py` unaffected).
