# Session d2a-20261121

- Status: pending (criteria written before measuring).
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

(to be filled)
