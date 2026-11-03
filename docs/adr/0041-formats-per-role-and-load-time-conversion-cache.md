# 0041. Formats per tensor role in the plan; load-time conversion into a per-tensor disk cache

Status: Accepted (Phase 2)

## Context

ADR 0024 makes the weight format a planner choice per unit and ADR 0027 gives each `assign` rule one `format`. But a unit holds several weight tensors (attention: q, k, v, output; an expert: gate, up, down), and they differ in sensitivity: llama.cpp's Q4\_K\_M keeps `attn_v` and `ffn_down` at Q6\_K in some layers. On Qwen3-30B-A3B the experts are about 97% of the bytes, so a plan that can only set a whole expert cannot use the lever Q4\_K\_M itself uses, and the Phase 3a exit test compares against Q4\_K\_M at equal bytes. The Qwen3 MoE builder keeps each layer's experts of one role in one stacked tensor (or slot tensor) with one ggml type, read by `mul_mat_id` and streamed by byte offset.

## Decision

- **Roles.** Every unit kind has named weight roles: `attn.q`, `attn.k`, `attn.v`, `attn.o`; `mlp.gate`, `mlp.up`, `mlp.down` (dense FFN); `expert.gate`, `expert.up`, `expert.down`; `embed` and `head` have one tensor and no role. Norms, the MoE router and other 1-D tensors keep their source format, as in llama-quantize. `UnitSize` gains `roles` (name, params, source format); empty means the unit is one role.
- **IR.** `Assign.format` stays the unit default; `Assign.role_formats` (map from role to format) overrides it per role. Expanded bytes are the sum over roles. The validator checks role names against the selected kinds and the downward-only rule per role.
- **What the builder executes** (capability row, phases 2 to 4): formats may differ per layer and role; all experts of one layer and role share one format. A plan that gives experts of one layer and role different formats is rejected until slots carry a type per slot.
- **Conversion at load.** `LoadOptions::formats` maps GGUF tensor names to target formats (resolved from a plan by the model crate). A tensor whose target differs from its source type is dequantized to f32 with ggml's type traits and quantized with `ggml_quantize_chunk`, rows split over the CPU threads. Only downward (target bits per weight at most the source's); formats that need an importance matrix are rejected until the imatrix input lands. Without an imatrix the result is byte-identical to llama-quantize's for the same tensor and type; a test per format checks it.
- **Disk cache, one file per converted tensor**, named by the SHA-256 of: source fingerprint, tensor name, target format, importance-matrix hash (`none` for now) and the ggml pin. The source fingerprint hashes each shard's header bytes, length and modification time; a copied or re-downloaded file converts again, a stale hit cannot happen without a forged header and mtime. Files are written to a temporary name and renamed. Directory: `LoadOptions::convert_cache`, else `$TETRAEDER_CACHE_DIR`, else `.tetraeder-cache` next to the model.
- **Cache files are shards.** The model's tensor index points converted tensors at their cache file (an extra shard), so resident loading, expert slabs, slot streaming and direct I/O read them like any GGUF shard and need no other change.
- **Export.** `tetraeder quant export` writes a converted model as GGUF, single or split by tensor groups, for llama.cpp tools (calibration, ADR 0042) and the conversion benchmark against llama-quantize.
- **Telemetry** (Rule 6): `tetraeder.convert.duration` (seconds per tensor, by format), `tetraeder.convert.bytes` (bytes written), `tetraeder.convert.cache` (hit or miss), a dashboard panel and an alert.
- **Dependency:** `sha2` for the cache key (no hash crate in the tree; speed is irrelevant for headers).

## Consequences

- A plan's first load with new formats pays the conversion once per tensor; changing one unit's format converts only its tensors. The planner counts it as a switch cost.
- Per-expert formats (hot experts high, cold experts low, docs/plan.md) need slot tensors per format; the IR already expresses them, the validator rejects them for now.
- The importance matrix hash is part of the key from the start; IQ formats below 4 bits come with the imatrix input.
- Hugging Face safetensors sources still wait for per-architecture name maps.
