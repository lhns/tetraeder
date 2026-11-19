# 0079. One-input steps attend over their own KV run

Status: Accepted (2026-11-19). Narrows ADR 0065's "attention per sequence, prototyped and kept off" for the own dense builder (ADR 0066); steps with several inputs keep the masked attention.

## Context

- The chunked layout (ADR 0067) gives each sequence one contiguous run; the masked attention of a step reads rows `0..n_kv`, so a sequence whose run sits above another's attends over the other's rows too.
- libllama at the pin, with `-kvu` and flash attention, attends over the sequence's own cells only (`fused_kv_rows`, `ggml_flash_attn_ext_rows`).
- On the Claude Code replay this cost the sequences above another 1.35x to 1.45x decode and up to 0.4 s per prefill (`ttft5-20261119`, README, "Long-context tail").

## Decision

- on the scheduler path, a step with one input attends over that input's chunked run only (`DenseOptions::per_seq_attn_single`, default on; `TETRAEDER_DENSE_PER_SEQ_ATTN=0` restores the masked path). Before, the masked attention read rows `0..n_kv`, so a sequence whose run sits above another's attended over the other's rows too. libllama at the pin, with `-kvu` and flash attention, attends over the sequence's own cells (`fused_kv_rows`, `ggml_flash_attn_ext_rows`). In ttft5-20261119 that cost 1.35x to 1.45x decode and up to 0.4 s per prefill for the sequences above another (README, "Long-context tail"). One op either way, so the 2026-11-16 threshold of ADR 0065, which judged steps with several inputs, does not apply; those keep the masked attention. Logits bit-equal to the sequence alone and to the masked path at row 0 (CPU tests in `qwen3.rs`). The executor path is unchanged. Also: the pool reuses its mask buffer across steps, and `TETRAEDER_STEP_TRACE=1` logs where a step's time goes.


## Consequences

- One attention op per step either way; the 2026-11-16 threshold of ADR 0065 judged steps with several inputs and is not affected.
- The same run-offset cost remains in the MoE builder (masked path) and for steps with several inputs; open.
