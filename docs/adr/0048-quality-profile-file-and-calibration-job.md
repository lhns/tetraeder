# 0048. Quality profile file and calibration job

Status: Accepted (Phase 2; read by the planner in Phase 3a)

## Context

ADR 0024 makes quality a measured number: per model, the KL divergence each (tensor group, format) choice costs. The planner (Phase 3a) reads it to pick formats, and its exit test compares the planner's mix with llama.cpp's Q4\_K\_M at the same bytes. The file is a contract between the calibration job and the planner, so its format must stay stable. libllama exposes no tensors, so a group cannot be swapped in place for dense models; rewriting a whole GGUF per sample costs about a minute on an 8B and more than a box session per model.

## Decision

- **KL** is llama-perplexity's: text cut into chunks of `ctx` tokens, the second half of each chunk scored, mean over scored tokens of KL(p\_ref || p) in nats over the full vocabulary. All numbers in a profile come from `llama-perplexity --kl-divergence` against one reference logits file.
- **Groups** are a `UnitSel` (ADR 0027) plus optional roles (ADR 0047); an empty role list means every quantizable role of the selected units. v1 granularity: `embed`, `head`, and per role class (`attn.q+k`, `attn.v`, `attn.o`, `mlp|expert.gate+up`, `mlp|expert.down`) per span of `L` layers, `L` chosen per model so the job fits one box session and recorded in the file.
- **Measurement:** every tensor outside the group stays at the **base** format, the group goes to the entry's format, the reference is the most precise source that runs on the box (BF16, else Q8\_0). The job converts the model once per format with the engine's converter (ADR 0047) and writes it as split GGUF, one shard per group (the first shard carries the metadata). A sample is a directory of symbolic links, one shard per group from the chosen format, so assembling a mix copies nothing and llama.cpp loads it as a split model.
- **File:** JSON, `<model stem>.quality.json` next to the model; measured profiles are committed under `bench/quality/`. Rust type `tetraeder_plan::QualityProfile`. Schema 1:

  ```
  schema: 1
  model: { name, arch, source, source_format, fingerprint, layers, experts }
  ggml: pin tag                      calibration: { text, sha256, ctx, chunks, scored_tokens }
  imatrix: null | { sha256, text }   reference: { format }
  base: { format, kl, bytes }        layer_span: L
  groups: [ { id, units: UnitSel, roles: [role], params,
              entries: [ { format, bytes, kl, ppl } ] } ]
  checks: [ { name, formats: { group id: format }, estimate, measured } ]
  cost: { seconds, gpu, host, threads }
  ```

  `bytes` is the group's (or the whole model's, for `base`) size in that format; every group also has an entry for the base format (its `kl` is `base.kl`). `fingerprint` is the SHA-256 over each source shard's header bytes and length (no mtime, so copies of a file share a profile). Unknown fields are ignored; a breaking change raises `schema`.
- **Estimate** for a plan: `base.kl + sum over groups and formats of w * (kl(group, format) - base.kl)`, where `w` is the fraction of the group's params the plan puts in that format. Roles outside every group and formats more precise than the base count as the base. A format without an entry has no estimate (the planner must not choose it). `checks` holds the additivity test of ADR 0024: measured KL of whole mixes against this estimate.
- **Exit-test harness:** `scripts/quality/compare_mix.py` takes a mix (tensor name to format, or a plan through `tetraeder quant formats`), builds it and the llama.cpp preset from the same source, and reports bytes, KL and perplexity of both on held-out text (not the calibration text) as result JSON. Phase 3a plugs in the planner's mix at the preset's byte budget.

## Consequences

- One calibration run per (model, workload, ggml pin). v1 measures k-quants, Q8\_0 and IQ4\_XS without an importance matrix; the `imatrix` field and IQ formats below 4 bits come with the imatrix input.
- The group granularity is a measured trade-off (profiling cost against planner gain, docs/plan.md open decision); finer groups are a new profile, same schema.
- KL on calibration text can miss agent-work losses; the tool-call check of ADR 0024 stays separate.
- Measured 2026-11-03 (bench/quality/, bench/results/20261103/quality-*.json): against a BF16 reference the summed estimate is within 14% of the measured KL on Qwen3-8B (random mixes) and on Qwen3-30B-A3B (uniform Q6\_K to Q3\_K); Q2\_K is underestimated (30B: 0.94 against 1.25). Against a Q8\_0 reference the 30B samples showed a floor of about 0.002 and the sum overestimated light mixes about 6 times, so profiles use the most precise reference (`qwen3-30b-bf16.quality.json` supersedes `qwen3-30b.quality.json`). A greedy knapsack over the profile beats Q4\_K\_M, Q3\_K\_M and Q5\_K\_M at equal bytes on held-out text on both models.

## Update: format scale and knapsack checks (2026-11-09)

- **`format_scale`** (optional, `{format: factor >= 1}`): a factor on that format's KL delta in the estimate, for formats the sum underestimates. The job fits it from the `uniform-<format>` check, `(measured - base) / (estimate - base)`, when the sum is more than 15% under (`UNDER_BAND` in `calibrate.py`). Old files lack it and mean 1. Applied by `QualityProfile::estimate`, `estimate_mix_scaled` and `ProfileQuality::loss`; the plain sum (`estimate_mix`), the recorded `checks[].estimate` and the knapsack order stay the ADR sum, so `compare_mix.py` rows still match. Qwen3-30B-A3B: Q2_K 1.3412, giving 0% error on uniform Q2_K (from 25%), at most 13.7% over the five checks (raw 25.4%), 9.2% after the power-law correction (ADR 0040). One measured point per format, so the scale is in-sample; pairs at other mixes are not measured.
- **Knapsack checks**: after the uniform and random checks the job adds `knapsack-<pct>` mixes, the reference greedy at budgets 20%, 35% and 50% of the way from the smallest mix to the base (about the Q3\_K\_M, Q4\_K\_M and Q5\_K\_M sizes on the 30B). They validate the estimate where the planner operates (ADR 0040: the correction fitted on uniform and random mixes did not transfer). Not with `--drop-shards`. They run on the next box quality session; the committed profiles do not have them yet.
- The Q8\_0-reference profile `qwen3-30b.quality.json` is deleted: the planner exit test and the profile load test use `qwen3-30b-bf16.quality.json`; its greedy mixes at the three preset budgets equal the measured `compare_mix.py` rows, and the stated 30B band is now 10% (was 31%).

## Addendum (2026-11-10): ik_llama.cpp quant types

Qwen3-8B, wikitext-2, ik_llama.cpp at its pin with its own perplexity tool and BF16 reference: the profile mix at the same or fewer bytes has mean KL 0.0448 against IQ4_KS 0.0614 and 0.0384 against IQ4_K 0.0449, top-1 agreement higher in both. `qlib.py` sizes ik types (IQ4_KS with its per-row f32 scale). Results `bench/results/20261110/quality-qwen3-8b-ik.json`.
