# 0083. Feature parity with llama-server: what it means and how it is checked

Status: Accepted (owner, 2026-11-23). Proposed (2026-11-22, owner asked for feature parity with llama.cpp and delegated the scope to the advisor). After M2. Gap matrix: [docs/notes/llamacpp-parity.md](../notes/llamacpp-parity.md). Tree nodes `serving-parity-*`.

## Context

- Speed against llama.cpp was judged in Phases 2, 4 and 5 (ADR 0045, 0065, 0068). What a user can ask of the server was never judged: the gateway has 6 routes against about 45 in llama-server, its engine contract carries 5 sampling fields, and unknown request fields are dropped silently (inventory in the gap matrix).
- The host sampler differs from llama-server where both accept a field: temperature is applied before the top-p mass (llama-server applies it last), there is no min-p (llama-server defaults it to 0.05), and unseeded requests take a per-process counter seed.
- With `--engine auto` every architecture other than `qwen3moe` goes to libllama, so model coverage is already llama.cpp's on that path; the own builders cover `qwen3` and `qwen3moe` only.
- "Parity" without a fixed target and a check never ends.

## Decision

1. Target: llama-server at the pin (`vendor/PIN`, b12426, fc504f91), not upstream head. Moving the pin moves the target.
2. Meaning: a request that works against llama-server works against `tetraeder serve` on the same GGUF, on CUDA and CPU, with the same response shape; at temperature 0 the same tokens.
3. Engines: a capability served through the libllama engine counts. Everything above the backend (request parsing, sampling, grammar, endpoints, response shape) must work on all three engines (libllama, dense, moe).
4. Out of scope: speed (judged by Phases 2, 4 and 5), the llama.cpp command-line tools other than the server, training and fine-tuning. Plan features 14 (other GPU backends), 29 (several models), 30 (LoRA) and 34 (multimodal, embeddings) get their own ADRs before any work; the owner sees the matrix first.
5. Check: a conformance harness sends one request set to the baseline llama-server and to `tetraeder serve` on the tiny fixture in CI and compares response shape and, at temperature 0, tokens. Every matrix row gets a failing case before its fix.
6. Done: every matrix row passes its conformance case or is scoped out in the matrix with a reason.
7. Sampling: where possible the pinned llama.cpp sampler chain is called on the logits (with Tetraeder's token mask applied first) instead of a port, so parity is by construction; a transform that cannot run without a llama context is ported and tested against the pinned one.
8. Unknown request fields stay accepted (clients send extras) but each field of the matrix is either honoured or rejected with 400, never dropped silently.

## Consequences

- Order: sampling and request fields, then endpoints, then capabilities libllama has but the gateway does not expose; the four plan features above after their ADRs. No box until a row needs a GPU.
- The sampler change alters outputs at temperature above 0 for existing clients (temperature last, min-p default); greedy outputs do not change.
- `docs/features.md` (speed) and the gap matrix (works or not) are separate pages that link each other.
