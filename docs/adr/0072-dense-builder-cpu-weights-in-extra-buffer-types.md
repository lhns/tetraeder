# 0072. Dense builder: CPU weights in libllama's extra buffer types

Status: Proposed (2026-11-16). Supersedes ADR 0039 for the own dense builder (`crates/model/src/qwen3.rs`) only; the MoE builder stays on 0039.

## Context

- On CPU runners with AMX (Intel Xeon 6975P-C), the dense builder's prompt processing took 4.97 ms per token against libllama's 1.84 on the same file (loop-bench, Qwen3-0.6B Q8_0), and the Phase 4 CPU rehearsal measured dense at 0.54 of llama-server at concurrency 4 with near-equal decode. (No session directory yet.)
- libllama stores matmul weights in the CPU extra buffer types where they run (`use_extra_bufts`, default on): AMX for Q8_0 and F16 on AMX Xeons, repack for Q4_K and similar on AVX2. On x86, repack has no Q8_0 path, so for Q8_0 only AMX matters. The dense builder kept every weight in plain memory, per ADR 0039 (no CPU repack of expert weights), whose argument was the MoE builder's op offload and streaming.
- Off AMX (EPYC 9V45 and 9V74 runners), dense already matches the libllama control: concurrency-4 ratio 0.909 and 0.906 against llama-server, the control 0.920 and 0.925 (rehearsal run 43778461586). (No session directory yet.)

## Decision

On a CPU backend, the dense builder puts each matmul weight in the first CPU extra buffer type that runs its matmul, using libllama's `weight_buft_supported` probe, and writes it whole in one call (`DenseOptions::extra_bufts`, default true, as libllama). The tied `tok_embd` stays in plain memory. GPU placements are unchanged, so ADR 0039's op-offload argument does not apply to them. Parity tests against libllama run both sides with extra buffer types off; `cpu_extra_bufts_match_plain_weights` bounds the difference to plain weights within 5% of the largest logit (AMX keeps F16 activations in F32, so rounding differs).

## Consequences

- Logits on AMX runners differ from the plain-weight path by kernel rounding (1.8e-2 with F16 KV, 1.4e-1 with Q8_0 KV on the random fixture, CI AMX runner). (No session directory yet.)
- Pending: the speed on an AMX runner (`step_shapes` engines `dense` against `dense_norepack`, and `llama` against `llama_norepack`); the 0.54 is expected to close but is not yet measured. (No session directory yet.)
- Also built and kept off (`DenseOptions::per_seq_attn`): attention per sequence over its own chunked run, see ADR 0065 ("Attention over sequences of unequal length", and its History entry 2026-11-16, "attention per sequence, prototyped and kept off").
