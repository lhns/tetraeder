# Session d2a30b-20261122

- Status: null. The 30B probe c4 ratio was 0.910, below the 0.925 stop line (original 0.929), so the model was stopped before any pair ran; q30b is recorded null and no further box is rented: the owner decides the board-limit rule for this workload.
- Run: box 61881715, host 351328 (Japan), RTX 3090, driver 615.71.09, power 350 W at default, CPU AMD EPYC 7B12 (64 cores, 128 host CPUs, cgroup quota 61), threads 23 (the original's, `THREADS=23`), baseline fc504f9 (reported at SSH), code branch `box-d2a30b-20261122` at 3a39035 (`COMMIT`).
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

## Results

Gates at SSH all passed: `cuda: ok`, power gate (350 of 350 W), idle gate 10/10 samples, co-tenant sample (host 0.7 busy cores, container 0.5), baseline fc504f9. The session ran build, gpu-parity (ok), gpu-shapes, pre-pass (nokvu 290 tok/s against kvu 285, nokvu chosen as in the original) and the probe; then the stop rule ended it (rc 3, `PHASE4_BOX_DONE rc=3`). The `record-log`, `env`, `pairs`, `ttft` and `summary` stages did not run, so there is no `env.json` and no `verdict.json`; `probe.json` has the verdict.

| field | this session | original (`20261117k` probe) |
|---|---|---|
| probe c4 ratio | 0.910 | 0.929 |
| probe c1 ratio | 0.963 | 1.008 |
| pairs | none (stopped) | 0.943 and 0.948 (c4), 0.989 and 0.990 (c1) |

Absolute c4 tok/s, diagnostic only (never a verdict, AGENTS.md), probe, one round:

| engine | this session | `20261117k` | `d2a-20261121` |
|---|---|---|---|
| 30B llama-server | 293 | 306 | 263 to 267 |
| 30B tetraeder-moe | 267 | 288 to 291 | 285 to 288 |

- Probe errors 0 in all four runs. The board-limit bit (0x400) appears in 10 of the 694 samples of the session, mostly at the end of the own engine's probe run, with 0x600 before the pre-pass and no 0x20 in the probe windows; the probe has no cleanliness rule, only the ratio line.
- Reading: on this machine (EPYC 7B12, Zen 2) llama-server was 4% slower than on the original (EPYC 7R13) and the own engine 7% to 8% slower (267 against 288 to 291), so the c4 ratio fell to 0.910, about 2 points under the original's 0.929 probe. One probe round has the noise of a single run; the band is 0.0753. The result does not show a regression: the original's pairs (0.943, 0.948) were also below the line of 1.00 and above 0.925, and a probe of 0.910 gives no pairs to judge. It also does not confirm the verdict on this machine.
- The reruns of the script fix (reps 3 and 4) were not exercised: no pair ran.

## Deviations


- Script fix before the session (bug fix to match ADR 0065): `phase4_summary.py rerun_reps` returned nothing when both of reps 1 and 2 were unclean, and the scripts ran at most one rep 3. Now both unclean pairs are rerun once (reps 3 and 4). `phase4_box.sh` and `phase4_box_d2a.sh` loop over the list; tests in `tests/test_phase4_summary.py` and `tests/test_phase4_box.py`. No threshold changed.
- Offer 59497037 vanished between scan and `create --offer` (three searches); the box was created with `create --auto --min-cpu-cores 48` and the skip list in `--exclude-hosts` (offer id rotated, host 351328, still the 64-core class). It stayed in `loading` for 11 min 30 s (past the 10 min stall rule by 1.5 min) and then came up, so it was kept rather than re-created.
- The CPU is an EPYC 7B12 (Zen 2, 128 host CPUs), not the original EPYC 7R13 (Zen 3, 96 host CPUs); no 7R13 or 96-CPU offer was on the market (at most 3 other offers with at least 24 cores, one 32-core at host 49107 offered a likely repeat of the 7302P).
- `THREADS=23` passed to match the original; the quota here is 61.
- Models: both models provisioned (default), only q30b used.
- Box life 31 min, credit 31.67 to 31.39 USD (0.28 USD), box destroyed after a verified fetch (`results-61881715-20261122-015454.tgz`).
