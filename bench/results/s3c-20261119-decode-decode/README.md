# Session s3c-20261119-decode (decode toggles, `scripts/layer_decode_box.sh`)

- Status: clean for what it measures (every run rc 0, parity rc 0). The directory is named `s3c-20261119-decode-decode` because the script appends `-decode` to `SESSION=s3c-20261119-decode`.
- Rule: ADR 0069 History 2026-11-19: a toggle becomes the default when its on/off tg128 ratio is at least 1.005 on the 30B in both rounds and the parity step passes with both on.
- Code 27c6e00. Same box as `bench/results/s3c-20261119/README.md` (61430359, host 20210, 4x RTX 3090, driver 595.91.07, 350 W default limit).
- Ordering deviation: `layer_decode_box.sh` needs the release build, the test binaries and the single and layer-4 plans of a `phase5_box.sh` run, and the brief puts it first. A bootstrap script (not committed, uploaded to the box as `/root/boot_decode.sh`) did the same release build, `cargo test --no-run` of the executor and model packages with `tetraeder-ggml-sys/cuda`, `tetraeder plan hw`, `plan layout` and `tetraeder-plan candidates --multi --plans-dir` into `bench/results/s3c-20261119-decode/phase5/` (the script's default `P5_DIR`; plans only, see its `DONE`). `phase5_box.sh` then ran in its own session directory.
- `BUDGET_MIN=30` (default 18). First launch: all 28 timed runs rc 0, but the parity step timed out (10 min cargo timeout, rc 124, `parity-toggles-timeout1.log`) because the script does not set `TETRAEDER_CUDA_ARCHS` (phase5_box.sh sets it to 86), so ggml-sys rebuilt without the arch. Rerun of the same script with `TETRAEDER_CUDA_ARCHS=86` skipped the finished runs and ran parity: rc 0 in 1.8 min. Not a bypass of a result; no timed run was repeated.
- Per-run numbers are tg128, `-r 5`, 30 threads (CPU quota 30). Stddev of the 5 samples is under 0.3 tok/s except llama.cpp `-sm layer` (1.8 to 2.1).

## 30B tg128 (tok/s), round 1 then round 2 (round 2 in reversed order)

| Kind | r1 | r2 |
|---|---|---|
| llama.cpp -sm none | 211.48 | 209.96 |
| llama.cpp -sm layer | 192.59 | 192.60 |
| our single | 197.99 | 196.70 |
| our single, uid+async | 207.52 | 207.58 |
| our layer-4 | 182.32 | 182.38 |
| our layer-4, uid | 184.58 | 184.66 |
| our layer-4, async | 184.17 | 184.66 |
| our layer-4, uid+async | 186.95 | 187.09 |

## On/off ratios (keep rule: at least 1.005 on the 30B in both rounds)

| Toggle | 30B r1 | 30B r2 | 32B r1 (one round) |
|---|---|---|---|
| TETRAEDER_RANGE_UID (layer4-uid / layer4) | 1.0124 | 1.0125 | not run |
| TETRAEDER_ASYNC_INPUTS (layer4-async / layer4) | 1.0102 | 1.0125 | not run |
| both (layer4-uid-async / layer4) | 1.0254 | 1.0258 | 1.0046 |
| both on one GPU (single-uid-async / single) | 1.0482 | 1.0553 | 1.0062 |

Both toggles meet the rule alone in both 30B rounds (about twice the 0.5% bar, A/A drift 0.2%). The gap to `-sm layer` shrinks from 0.947 to 0.971 (our layer-4 / llama.cpp layer on the 30B, both rounds); on the 32B layer-4 is 0.981 against `-sm layer` without the toggles, 0.985 with. Our single GPU 30B is at 0.936 of `-sm none` without the toggles and 0.981 with.

32B (one round, tok/s): llama.cpp none 39.24, layer 39.46; our single 38.37, single uid+async 38.61; layer-4 38.70, uid+async 38.88.

## Parity step (both toggles on, REQUIRE_GPU, 4 GPUs)

`parity-toggles.log`, rc 0, no skip lines: `gpu_cuda_graphs_replay_kept_ranges` ok (GGML_CUDA_GRAPH_DEBUG=1; log shows `uid false: warmups 0` against `uid true: warmups 5`, so the uid took effect), `gpu_one_rank_is_bit_equal_to_the_scheduler` ok, `gpu4_shapes_match_single` ok, `gpu4_moe_shapes_match_single` ok.

## Exec trace summary (`scripts/exec_trace_summary.py`, per step, 768 steps, toggles off unless named)

| Run | upload sum | compute (host issue) sum | copy sum | sync (rank 3 wait) | logits | step |
|---|---|---|---|---|---|---|
| 30B layer-4 | 86.5 us | 301.2 us | 45.4 us | 4947 us | 91.8 us | 5475 us |
| 30B layer-4 with sync mode (device time per rank) | 87.4 | 5431.8 (rank 0 1303, 1 1268, 2 1273, 3 1588) | 55.0 | 0.7 | 92.2 | 5671 |
| 30B single | 25.8 | 291.9 | 0 | 4651 | 92.8 | 5064 |
| 30B layer-4, uid+async | 39.1 | 40.6 | 43.2 | 5125 | 92.0 | 5344 |
| 32B layer-4 | 131.9 | 196.0 | 44.9 | 25361 | 92.0 | 25832 |

Reading: the toggles cut the host issue time of the 30B layer-4 ranges from 301 to 41 us and uploads from 87 to 39 us per token, and the step by 131 us of 5475, and measured tg moves by 2.6% (182.3 to 187.0 tok/s, 5.48 to 5.35 ms per token). The host issue (the traced step minus the GPU wait) is 0.5 ms of a 5.4 ms step with the toggles off; the rest is device time (sync mode: about 1.3 to 1.6 ms of compute per rank, 5.4 ms over the four). Peer copies cost 14 to 19 us each and do not block the host (no copy call takes a stage's time), so per-rank issue threads are not indicated by this trace.

Files: `<model>-<kind>-r<k>.json|log`, `*.trace.jsonl` and `.trace.summary.txt`, `gpu-samples.csv`, `env.txt`, `layer_decode.log` (box log incl. the bootstrap).
