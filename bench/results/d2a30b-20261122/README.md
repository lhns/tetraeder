# Session d2a30b-20261122

- Status: pending (criteria written before measuring).
- Purpose: the 30B-only rerun of the D2a admission probe on a machine of the Phase 4 class, after `d2a-20261121` was null (every 30B pair unclean on the board-limit share, host 49107 as a slower machine, see its "Diagnosis"). Node `bench-d2a-30b`.
- Code: `scripts/phase4_box_d2a.sh` with `MODELS=q30b PROBE_MIN=0.925`, branch `box-d2a30b-20261122`.

## Criteria (written before measuring)

Copied from `bench/results/d2a-20261121/README.md` for the 30B only (ADR 0065 "Verdict", throughput; passing session `20261117k`):

- Qwen3-30B-A3B Q4_K_M judged on `--engine moe`; libllama control after each pair, reported, never judged.
- Baseline: llama-server of `/opt/llama.cpp-baseline` (b12426, fc504f91; must report it at SSH), `-np 4 -c 32768 -fa on -ctk f16 -ctv f16`, KV mode by the pre-pass (`-kvu` against `-no-kvu`, higher `throughput_tps`).
- Workload: c4 `--prompt-tokens 512 --max-tokens 128 --concurrency 4 --requests 16`, c1 the same with `--concurrency 1 --requests 8`; two interleaved pairs, (baseline, Tetraeder) then (Tetraeder, baseline), seeds as in the script; one probe round first with `PROBE_MIN=0.925`.
- Gating fields: c4 `throughput_tps` and c1 `decode_tps_median`; pair ratio r = Tetraeder / baseline.
- Pass: both judged pairs clean (errors 0, co-tenant guard, throttle by the 1 Hz sampler rule including the board-limit share difference of at most 0.10, equal completion-token totals, baseline runs within the band) and min over pairs of r >= 1 - b. Band b from `bench/noise/rtx-3090.json` (n=11, 0.0753), line 0.925. No threshold changes.
- Probe stop rule: probe c4 ratio below 0.925 stops the model (pass null).
- Rerun rule ("an unclean pair is rerun, never dropped"): an unclean pair is rerun once. One unclean pair: rep 3. Both unclean: both rerun once, rep 3 in the order of rep 1 and rep 4 in the order of rep 2 (the script fix below). The verdict uses the two clean pairs among reps 1 to 4; unclean pairs stay listed in `verdict.json` (`dropped_pairs`) and here.
- If no two clean pairs result, `q30b` is recorded null and no further box is rented: the owner decides the board-limit rule for this workload.
- TTFT: not in this script, not judged (D2b).
- Originals (`20261117k`), ratios shown next to ours: 30B pairs c4 0.943 and 0.948, c1 0.989 and 0.990; probe c4 0.929, c1 1.008.
- Machine: RTX 3090, a host of the original class (EPYC 7R13, 96 host CPUs, threads 23) preferred; the CPU model is recorded after SSH, and a 7302P-class CPU is a gate failure (destroy, next host). Cap 0.90 USD/h all-in, skip list per `docs/box-driver.md`, gates: `cuda: ok`, power at default, idle, baseline reports fc504f91.

## Deviations

- Script fix before the session (bug fix to match ADR 0065): `phase4_summary.py rerun_reps` returned nothing when both of reps 1 and 2 were unclean, and the scripts ran at most one rep 3. Now both unclean pairs are rerun once (reps 3 and 4). `phase4_box.sh` and `phase4_box_d2a.sh` loop over the list; tests in `tests/test_phase4_summary.py` and `tests/test_phase4_box.py`. No threshold changed.
