# 0024. Quantization is a planner choice per unit; quality is measured per model

Status: Proposed (design; implemented in Phases 2 and 3a)

## Context

Feature 5 ("quantization chosen to fit") picked one format for the whole model. But formats can differ per tensor (GGUF already mixes them), activations cross stage and device boundaries in f16 or f32 whatever the weight format, and sensitivity differs a lot between tensors: embeddings, output head, router and shared experts hurt more when squeezed than rarely used experts. Choosing the format together with placement lets cold MoE experts sit on NVMe at low precision while hot ones stay in VRAM at high precision, which cuts bytes from the slowest tier. The planner can only trade quality if quality is a number it can compute for a candidate plan.

## Decision

- The format of every unit (tensor, layer slice, expert) is a field of the placement plan (ADR 0006), chosen by the planner together with device and tier.
- Quality is measured, not guessed: a **quality profile** per model holds the KL divergence from the full-precision model's logits for each (tensor group, format) pair on a calibration set. A plan's quality loss is estimated as the sum over its units. Additivity is an assumption with a test: the summed estimate is compared with the measured divergence of the whole chosen mix.
- Without a profile, the planner uses llama.cpp's fixed preset rules (more bits for embeddings, output, some attention and FFN-down tensors, first and last layers) as the prior.
- Weights are converted at load time from the most precise source available, only downward, and the result is cached on disk.

## Consequences

- Per model, a calibration job (a Vast batch) produces the profile and the importance matrix the 1 to 2 bit formats need.
- Op cost curves are keyed by format as well as op, shape and device (ADR 0007).
- Converted weights need a disk cache keyed by source hash, format, importance-matrix hash and ggml version.
- Profiles can differ per workload (code versus chat), like routing profiles.
- Exit tests compare against llama.cpp presets at the same byte budget, not against absolute divergence numbers.
