# 0085. Strata's features: what we take, how it is judged

Status: Proposed (2026-11-23, owner asked for a plan to add Strata's features; planned with the advisor). Matrix: [docs/notes/strata-parity.md](../notes/strata-parity.md).

## Context

- Strata v0.2.20 is four engines with per-shape kernels for four hybrid model families; none runs a Qwen3 model and Tetraeder's own builders run only Qwen3, so no paired number exists (`bench-strata-pairing`, parked).
- Its closest model, Qwen3.6-35B-A3B (`qwen35moe`), runs on Tetraeder today through libllama; the pin has its graph and the CUDA ops it needs.
- Strata claims 1.44x llama.cpp decode on it (HIP, old llama.cpp). Tetraeder is at 0.94 to 1.09 of llama-server on Qwen3.
- Several Strata features exist here already (low-bit and rotated KV, n-gram drafts, expert hot set and overlap, CUDA graphs).

## Decision

1. Target: Strata tag `v0.2.20`; moving it means rereading the matrix.
2. Three classes (matrix): features that apply to Qwen3 now, built in this repo's order and judged by a paired run on a model we run; features that need a shared model; owner scope, each with its own ADR.
3. Measure before building a builder: `bench-strata-q36` (Strata q36, llama-server at the pin, `tetraeder serve --engine auto`, one GGUF, one box, c1 and c4, interleaved). Strata's binary gets a `docs/sources.md` row.
4. The own `qwen35moe` builder gets its own ADR after that measurement, whose design question is recurrent state in the KV pool, `seq_rm` and the prefix cache; it unblocks MTP drafts and the direct pairing.
5. Lossy shortcuts (route tail skip, KVA) are out (rule 2). Per-shape kernels enter only through the patch queue with a measured gap (ADR 0011).
6. Strata's numbers never change a default; only a paired run does (rule 3).

## Consequences

- Feature count does not buy Strata's speed: that comes from per-model kernels. The first number that says how far behind we are is `bench-strata-q36`.
- The class 1 nodes run beside the llama-server parity build (they touch `crates/plan`, `crates/kvcache` and `crates/model`, not the gateway).
