# Session d2a-20261121

- Status: mixed. Qwen3-8B passes; Qwen3-30B-A3B is null (every pair unclean on the host's board-limit throttle share, in two attempts); the higher ratios come from a slower baseline on a different machine, not a faster engine (see Diagnosis). `phase4_pass` stays null by design (no TTFT in D2a).
- Run: box 61874205, host 49107 (the host of `20261117k`), RTX 3090, driver 615.78.08, power 350 W at default, CPU quota 30 (original 23), baseline fc504f9, code origin/main 7976b5a plus the criteria commit bc1966e (`env.json` git_commit).
- Purpose: confirm the Phase 4 throughput verdict (`20261117k`, main 6b07795) still holds on final main; the admission path changed after the Phase 4 verdict commit (docs/m2-plan.md step 5 before reporting M2). Node `bench-d2a-probe`.
- Code: origin/main 7976b5a (docs-only difference from the checkout this session was started on); script `scripts/phase4_box_d2a.sh`.

## Criteria (written before measuring)

Copied from ADR 0065 ("Verdict", throughput) and the passing session `20261117k`:

- Same models and engines: Qwen3-8B Q4_K_M judged on `--engine dense`, Qwen3-30B-A3B Q4_K_M judged on `--engine moe`; libllama control after each pair, reported, never judged.
- Same baseline: llama-server of `/opt/llama.cpp-baseline` (b12426, fc504f91; must report it at SSH), `-np 4 -c 32768 -fa on -ctk f16 -ctv f16`, KV mode chosen by the pre-pass (`-kvu` against `-no-kvu`, higher `throughput_tps`).
- Same workload and rounds: c4 `--prompt-tokens 512 --max-tokens 128 --concurrency 4 --requests 16`, c1 the same with `--concurrency 1 --requests 8`; two interleaved pairs per model, (baseline, Tetraeder) then (Tetraeder, baseline), seeds as in the script; probe first with `PROBE_MIN=0.925`, one probe round per model.
- Gating fields: c4 `throughput_tps` and c1 `decode_tps_median`; pair ratio r = Tetraeder / baseline.
- Pass, per model and per field: both pairs clean (errors 0, co-tenant guard, throttle by the 1 Hz sampler rule, equal completion-token totals, baseline runs within the band) and min over pairs of r >= 1 - b. Band b from `bench/noise/rtx-3090.json` (n=11, 0.0753), so the line is 0.925. An unclean pair is rerun (rep 3 if the budget allows), never dropped.
- Probe stop rule: a model whose probe c4 ratio is below 0.925 is stopped (pass null) as in `20261117k`.
- The probe passes if every D2a pair passes ADR 0065's rule as the original did. Each ratio is reported next to the original's (`20261117k`): 8B c4 1.022 and 1.021, c1 0.995 and 0.997; 30B c4 0.943 and 0.948, c1 0.989 and 0.990; probes 8B c4 1.000, c1 1.019; 30B c4 0.929, c1 1.008.
- TTFT: ADR 0065's TTFT half is not in `scripts/phase4_box_d2a.sh` (its `main` has no record-log or ttft stage; that is D2b). It is not added and not judged here. For reference, ADR 0074's bar would be median and p90 at most 1.15x llama-server with saves at most 10% of TTFT.

## Box

RTX 3090 (as `20261117k`), one box, cap 0.90 USD/h all-in, `--min-inet 1000`, skip list per docs/box-driver.md, gates: `cuda: ok`, power at default, idle, baseline reports fc504f91. Both models provisioned (default models of `bench/spec/phase0.toml`).

## Results

Rule as the original: line 1 - b = 0.925 (band 0.0753, rtx-3090 n=11). Original = `20261117k`. Ratios are Tetraeder / llama-server; baseline variant nokvu (8B) and kvu (30B; the original picked nokvu for both).

Probes (one round per model):

| model | c4 | original | c1 | original |
|---|---|---|---|---|
| 8B dense | 1.072 | 1.000 | 1.057 | 1.019 |
| 30B moe | 1.018 | 0.929 | 0.999 | 1.008 |

Pairs. Board-limit share = share of window samples with 0x200/0x400, baseline side vs Tetraeder side; a pair is clean only if they differ by at most 0.10 and no disqualifying bit appears.

| model | pair | c4 | original c4 | c1 | original c1 | board-limit share (baseline vs Tetraeder) | clean |
|---|---|---|---|---|---|---|---|
| 8B | r1 | 1.123 | 1.022 | 1.031 | 0.995 | 0.17 vs 0.17 | yes |
| 8B | r2 | 1.124 | 1.021 | 1.036 | 0.997 | 0.26 vs 0.11 | no, replaced by r3 |
| 8B | r3 | 1.091 | | 1.022 | | 0.06 vs 0.11 | yes |
| 30B, attempt 1 | r1 | 1.075 | 0.943 | 0.990 | 0.989 | 0.31 vs 0.19, one 0x20 sample on the baseline side | no |
| 30B, attempt 1 | r2 | 1.086 | 0.948 | 0.984 | 0.990 | 0.59 vs 0.12, 0x404 after the baseline run | no |
| 30B, attempt 2 | r1 | 1.081 | 0.943 | 0.979 | 0.989 | 0.71 vs 0.29, 0x404 after the baseline run | no |
| 30B, attempt 2 | r2 | 1.084 | 0.948 | 0.997 | 0.990 | 0.35 vs 0.12 | no |

(8B r2 is unclean on the board-limit share, so r3 replaced it; its files stay in `phase4/`. The 8B spread, r1 1.123, r2 1.124, r3 1.091, probe 1.072, is wide for one host; the original's c4 was 1.021 and 1.022. The baseline's c4 on this box was 275 tok/s.)

- 8B: `pass` true, pairs r1 and r3 clean; worst c4 1.091, worst c1 1.022. A/A (baseline repeat) c4 -0.013, c1 0.009.
- 30B: `pass` null. Both attempts ran the same two pairs on the same box; every one of the four pairs fails only the board-limit rule (the baseline side spends more of the window under 0x200/0x400 than the own engine in all four pairs, 0.31 to 0.71 against 0.12 to 0.29). No error, no disqualifying bit except one 0x20 sample (attempt 1, r1, baseline side). These ratios (c4 1.075 to 1.086, c1 0.979 to 0.997) are not evidence: the asymmetry the rule excludes favours the own engine in every pair, and the rule fixed before the session gives no verdict: "an unclean pair is rerun, never dropped". One rerun was allowed by the coordinator and recorded as null, host board-limit.
- Attempt 1 files (all q30b bench, gpu, server and window files, verdict.json, the box log) are in `phase4/unclean-attempt1/`; the rerun used the same `SESSION`, `PROBE_MIN` and `COMMIT`.
- The script reruns one pair (rep 3) only when exactly one of two pairs is unclean (`phase4_summary.py rerun_reps`); with both unclean it ends, so the 30B had no rep 3 in either attempt.
- TTFT: not in this script, not judged (see criteria).

## Diagnosis (coordinator, 2026-11-22, from the committed files)

- Not the same machine as `20261117k`: host id 49107 both times, but the CPU was AMD EPYC 7R13 with 96 host CPUs (threads 23) then and AMD EPYC 7302P with 32 host CPUs (threads 30) now.
- Absolute c4 tok/s, diagnostic only (never a verdict, AGENTS.md): llama-server fell, the own engines did not.

| run | 20261117k | d2a-20261121 |
|---|---|---|
| 30B llama-server | 306 | 263 to 267 |
| 30B tetraeder-moe | 288 to 291 | 285 to 288 |
| 30B tetraeder-llama (libllama control) | 301 to 302 | 281 to 283 |
| 8B llama-server | 306 | 272 to 276 |
| 8B tetraeder-dense | 313 | 297 to 310 |
| 8B tetraeder-llama (libllama control) | 304 | 279 to 282 |

- The original 30B pairs had board-limit share 0.0 on both sides; here the baseline side is 0.31 to 0.71, with lower GPU busy share (63 to 85% against 88 to 91%).
- So the 30B rise from about 0.945 to about 1.08 is the baseline slowing on this machine, not a gain of the own engine. The 8B pass stands on its two clean pairs, but its margin over the original (1.09 to 1.12 against 1.02) has the same cause.
- On this machine the own MoE engine's absolute c4 matched the Phase 4 commit's on the original machine. That alone does not rule out a regression: on `d2a30b-20261122` (EPYC 7B12) the own engine lost 7 to 8% and llama-server 4%.
- Next: one 30B-only session on a machine of the original class (EPYC 7R13 or similar, 96 host CPUs), criteria committed first. If it is null again, the owner decides the board-limit rule for this workload; no third box.

## Deviations

- Offer ids rotated between `offers` and `create` (`--offer` failed three times); boxes were created with `create --auto` with the skip list in `--exclude-hosts` and `--min-cpu-cores 20`, so the box was chosen by score among eligible offers.
- Boxes: 61871432 (host 392144) destroyed at the power gate (350 or 360 W against 420 W); 61871755 (host 73118) stalled in loading past 10 min; 61873014 (host 73118, one re-create per the stall rule) had power 350 W against 420 W; 61874205 (host 49107) ran the session. Total credit used 1.50 USD for the four boxes (33.19 to 31.69).
- The 30B rerun launched the script with the default MODELS (both), not `MODELS=q30b`, so `verdict.json` and `probe.json` stay complete; the build, gpu-parity and gpu-shapes ran again (4.5 min in all), probes, pre-pass and the 8B pairs resumed as done.
- CPU quota 30 and 32 cores (original 23); noted, not matched.
- Model download on host 49107 ran 18 to 25 MB/s (about 13 min for 19.5 GB).
