# Session 20261110

- Status: several boxes. Detector box exit test passes (ADR 0041); Phase 2 met on the means against KTransformers and the llama.cpp cache, thin (b1-exit, ADR 0045); KV quality run unclean for ADR 0050 (b) (32K rows measured nothing); proxy run with Tetraeder tool errors above llama-server (diagnosis for ADR 0057); ik_llama.cpp quality comparison clean (ADR 0048).
- Code and box: commits of 2026-11-10 (results commits 158e850, 521c516 and others, see git log of this directory); box 60042169 (b1-exit: RTX 3090, Xeon Gold 6330N, quota 26.9, 26 threads, driver 595.91.07), box 59998776 (RTX A6000), boxes 59977867 and 59978288.
- Raw files: this directory.

## From ADR 0041 Status (moved verbatim)

Status: Accepted (2026-11-15: detector passed on a box (single 30/30, no false positives in 10, pairs 9/9) and in the simulated battery). Earlier: Proposed (Phase 3a)

## From ADR 0045 (addendum "verdict, met on the means", moved verbatim)

## Addendum (2026-11-10, box 60042169): verdict, met on the means

Session `b1-exit`: RTX 3090, Xeon Gold 6330N (Ice Lake, AVX-512 VNNI/VBMI), 91.4 GB cgroup RAM, quota 26.9 cores, 26 threads for every engine, ggml native, `MOE_CPU_SHARE=0.9` (B3 tuner). Qwen3-235B-A22B, 32K prefill, 64 decode, same prompt hash for builder and llama.cpp. Paired ratios on this box only:

| | Prefill tok/s | Decode tok/s |
|---|---|---|
| Builder (prefetch, repeat) | 69.5, 70.3 | 3.56, 3.31 |
| KTransformers (avx512_vbmi, repeat) | 36.6, 39.9 | 3.40, 2.70 |
| llama.cpp expert cache, share 0.9, mmap | 10.96 | 2.10 |
| llama.cpp expert cache, `--load-mode dio` | OOM-killed at load | |

- Prefill: 1.83x KTransformers, 6.4x the llama.cpp cache. Met.
- Decode: 1.13x KTransformers on the means (pairs 1.05 and 1.23) and 1.57x the llama.cpp cache. KTransformers spreads 23% between its two runs, the builder 7%, so the margin over KTransformers is at worst a tie within noise. Met on the means; more pairs in `b1b-levers` will firm it up.
- dio needs the CPU-share weights in anonymous memory, which cannot fit 142 GB into 91 GB, so mmap is llama.cpp's best here. Its mmap run thrashes (515k major faults in the timed run, prefill about 50 min).
- Host sampler, decode: the builder reads 570 MB per token at queue depth 34 with the CPU 96 to 99% busy; KTransformers reads 146 to 250 MB per token at depth 13 to 21 with 2 to 3 thousand major faults per token, CPU 55 to 70%.
- The 2026-11-03 box (KT 1.34x ahead) had another CPU and AVX2 builds of our ggml; since then boxes build ggml native with the iqk kernels (ADR 0052, 0053). Absolute numbers do not compare across boxes; this verdict is for this box class.
- Not run (deadline guard): `ktransformers-avx2`, the llama.cpp cache repeat. Results: `bench/results/20261110/exit-*.json`, `disk-curve-*.json`.

## From ADR 0048 (addendum "ik_llama.cpp quant types", moved verbatim)

## Addendum (2026-11-10): ik_llama.cpp quant types

Qwen3-8B, wikitext-2, ik_llama.cpp at its pin with its own perplexity tool and BF16 reference: the profile mix at the same or fewer bytes has mean KL 0.0448 against IQ4_KS 0.0614 and 0.0384 against IQ4_K 0.0449, top-1 agreement higher in both. `qlib.py` sizes ik types (IQ4_KS with its per-row f32 scale). Results `bench/results/20261110/quality-qwen3-8b-ik.json`.

## From ADR 0050 (note 2026-11-10 "what the first KL run measured, and the harness fix", moved verbatim)

**First run** (`bench/results/20261110/kv-quality.json`, box 59998776, RTX A6000): Q8_0/Q8_0 8K mean KL 0.0107, top-1 0.958 (fail), 32K 0.00087, 0.999; Q8_0/Q4_0 8K 0.0169, 0.933 (fail), 32K 0.000037, 1.000. The 32K rows measured nothing: the frozen exit prompt is about 25,848 tokens (Qwen3 vocab), the harness cycled it to 32K, so the scored positions 31744..32767 were a verbatim repeat of text 25,848 positions earlier (every 8-gram of the block occurs in its context; 0.3% at 8K). The model copies there and any KV format scores KL near 0. The calibration set was skipped (`scripts/kv_quality.sh` never set `TETRAEDER_KV_QUALITY_TEXT`). The 8K rows are real text and stand; whether they are the format's cost or a CUDA issue is open, because quantized-KV parity against libllama has only run on the CPU.
**Fix.** The ADR's "exit prompt at 32K" cannot be met literally, so: each depth's sequence ends with the first 8192 prompt tokens and the rest of the prompt, cycled to end with its last token, fills the front (`depth_seq`). 8K is unchanged; both depths score the same 1024 tokens, so they differ only in depth. The repeated 8-gram share of the scored block (2.4% at 32K, cross-references inside the docs) is in the JSON and must stay at most 0.1. The script fetches wikitext-2 and fails without the calibration set. Informative rows next to the gating ones: the builder's F16 cache again and rotated (noise floor), both formats with the rotation off, and libllama with F16 and both formats against the same reference, plus median, p99 and max KL per row. A bound that still fails goes back to this ADR; the bounds are unchanged.

## From ADR 0057 Context (moved verbatim)

The second Claude Code proxy run (bench/results/20261110/claude-code-proxy, box 59998776, RTX A6000, Qwen3-Coder-30B-A3B Q4_K_M, 10 tasks x 2 reps) gave Tetraeder 12/20 tasks and an 18.5% tool-error rate (100 of 541 calls), llama-server 13/20 and 5.6% (26 of 466). Pairing each `is_error` tool result in the request logs with the `tool_use` it answers:

| Class (proxy) | Count | What it is |
|---|---|---|
| input_validation | 52 | Edit `replace_all` sent as the string `"False"` |
| input_validation | 4 | Edit `replace_all` as `"True"` |
| input_validation | 2 | ScheduleWakeup `stop` as `"True"` |
| input_validation | 2 | ReportFindings `findings` (array) as the string of a Python `repr` list (`[{'category': ...}]`) |
| other | 29 | Bash exit code 1 (pytest failures, scripts) |
| other | 4 | Bash refused by Claude Code's rm check (multi-line `python -c`) |
| other | 7 | Read on a directory or missing file, Edit string not found or a no-op edit |
| unknown_tool | 1 | `Explore` (not in the request's tools) |

The request log holds one more `other` result (40) than the proxy counted (39). The log confirms the proxy's `is_error` patterns, which ADR 0051 left unchecked.

All 60 validation errors are one cause: Qwen3-Coder writes XML parameter values as Python literals (`True`, `False`, `None`, single-quoted strings), which the parser in `toolcall.rs` could not read as JSON, so it kept the text as a string. The "other" errors are the model's work on the task, the same kind llama-server has (26). The five Tetraeder sessions with `is_error` (llama-server: one) all ended at Claude Code's 40-turn limit, and each had 4 to 13 validation errors: the model retried the same Edit with the same `"False"`.

## From ADR 0057 Decision (moved verbatim)

Calls stay counted `parsed/ok`: the call is well formed and the value has the schema's type. Counting them `repaired` would put about 11% of calls into the `ToolCallFailures` alert, which is meant for template or model regressions.

## From ADR 0057 Consequences (moved verbatim)

Note: an estimate made from this run before the rerun; measured in `20261110b`.

- The 58 boolean calls of the run become schema-valid; the 2 ReportFindings calls get the right type (their item fields are not checked here). With all else equal the tool-error rate drops from 18.5% to about 7.4% (40 of 541) at best, near llama-server's 5.6%; converted Edits can still fail for task reasons. The turn-limit sessions lose their retry loops. The next proxy run measures it.
