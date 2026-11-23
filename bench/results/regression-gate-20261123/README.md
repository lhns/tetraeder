# Session regression-gate-20261123

- Status: pending (criteria written before the box; results follow).
- Purpose: node `bench-regression-gate`. Since the M2 numbers (A = `f68d2a0`, this morning's build) main gained the llama-server parity work (sampler on libllama's chain with a plain-greedy fast path, request fields, endpoints, group D), the prefix disk tier, KV format auto, recurrent-reuse handling and two review rounds. Question: did the own engines regress in throughput or TTFT? Self A/B on one box; nothing is judged against llama-server.
- A: `f68d2a0`, built on the box from a `git archive` tarball in its own checkout and target dir. B: current main `41dcedd2` plus this branch's script commits (only `scripts/`, `tests/test_ab_moe_box.py` and this README differ from main; recorded in `env.json`). `vendor/PIN` and `patches/` are identical in A and B (`git diff f68d2a0 origin/main -- vendor/PIN patches` is empty), so both build the same ggml. Both release builds with CUDA.
- Code: `scripts/gate_box.sh` runs `scripts/ab_moe_box.sh` (gate and TTFT; summary `scripts/ab_moe_summary.py`) and then `scripts/gate_extra_box.sh` (llama-server CUDA build, parity on CUDA, Qwen3.6 default flags).

## Criteria (fixed before the run)

Gate, as `bench/results/ab30b-20261122/README.md`, per model and concurrency (four metrics):

- Workload per run, one fresh server process per run: Qwen3-30B-A3B Q4_K_M `--engine moe`, Qwen3-8B Q4_K_M `--engine dense` (the Phase 4 dense flags of `scripts/phase4_box_d2a.sh`), both `--ngl -1 --parallel 4 --ctx 32768 -t <CPU quota> --accept-any-model` and otherwise default flags; one warm-up request, then c4 `--prompt-tokens 512 --max-tokens 128 --concurrency 4 --requests 16`, then c1 the same with `--concurrency 1 --requests 8`. Seeds per round (c4 round*10+4, c1 round*10+1), the same on A and B.
- Order per model: three rounds A B, B A, A B (one run per side and round). Gating fields: c4 `throughput_tps`, c1 `decode_tps_median`; round ratio = B / A.
- A run is clean when errors are 0 in both tests, no co-tenant flag, and the 1 Hz sampler shows no window sample with a bit outside 0x1, 0x2, 0x4, 0x100, 0x200, 0x400. Unclean rounds are listed with the reason and excluded from the median.
- Reading per metric, by the median B/A over the clean rounds: at least 0.98 is no regression evidence; below 0.97 with every clean round below 1 is a regression; anything else, and always with fewer than 2 clean rounds, is inconclusive.
- Outcome of the node: successful if all four metrics read no regression evidence, mixed if any is inconclusive, bad if any is a regression (a defect node then names the model and concurrency).

TTFT, one replay per build on Qwen3-8B (`--engine dense --ngl -1 --parallel 4 --ctx 40960 --cache-ram 8192 -t <quota> --accept-any-model`, as `scripts/ttft_diag.sh`), the request log of ttft4 to ttft7 (sha256 f79b4a6d...5f7b35 checked on the box, 105 requests, prompts up to 36000 tokens, `scripts/gateway_replay.py --speed 0 --concurrency 1`), A then B on the same box:

- Median and p90 TTFT over the streamed requests with status 200 on both sides; B/A of each at most 1.05 means no regression, else reported. Any request error in either replay is reported too.

Other jobs (reported, not judged): parity cases (`tests/conformance/parity/test_parity.py`) against llama-server and tetraeder built with CUDA, counts compared with the CPU CI run of the same commit; Qwen3.6-35B-A3B UD-Q4_K_XL `serve --engine auto` with default flags, c4 and c1 of the `strata-q36-20261123` workload, expected zero errors and 128 tokens per request.

Not judged: any ratio against llama-server, absolute tok/s across boxes (shown for orientation only). The box is not the RTX 3090 of `ab30b`; only the paired ratios of this box count.
