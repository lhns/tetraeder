# 0042. Quality profile file and calibration job

Status: Accepted (Phase 2; read by the planner in Phase 3a)

## Context

ADR 0024 makes quality a measured number: per model, the KL divergence each (tensor group, format) choice costs. The planner (Phase 3a) reads it to pick formats, and its exit test compares the planner's mix with llama.cpp's Q4\_K\_M at the same bytes. The file is a contract between the calibration job and the planner, so its format must stay stable. libllama exposes no tensors, so a group cannot be swapped in place for dense models; rewriting a whole GGUF per sample costs about a minute on an 8B and more than a box session per model.

## Decision

- **KL** is llama-perplexity's: text cut into chunks of `ctx` tokens, the second half of each chunk scored, mean over scored tokens of KL(p\_ref || p) in nats over the full vocabulary. All numbers in a profile come from `llama-perplexity --kl-divergence` against one reference logits file.
- **Groups** are a `UnitSel` (ADR 0027) plus optional roles (ADR 0041); an empty role list means every quantizable role of the selected units. v1 granularity: `embed`, `head`, and per role class (`attn.q+k`, `attn.v`, `attn.o`, `mlp|expert.gate+up`, `mlp|expert.down`) per span of `L` layers, `L` chosen per model so the job fits one box session and recorded in the file.
- **Measurement:** every tensor outside the group stays at the **base** format, the group goes to the entry's format, the reference is the most precise source that runs on the box (BF16, else Q8\_0). The job converts the model once per format with the engine's converter (ADR 0041) and writes it as split GGUF, one shard per group (the first shard carries the metadata). A sample is a directory of symbolic links, one shard per group from the chosen format, so assembling a mix copies nothing and llama.cpp loads it as a split model.
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
