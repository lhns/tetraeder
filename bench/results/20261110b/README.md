# Session 20261110b

- Status: clean proxy rerun under ADR 0057 (no validation errors left); its KV quality rows are a diagnosis (F16 floor above ADR 0050 (b)'s absolute bounds).
- Code and box: results commits 626f940 and 6a8f3ba; box 60050186, A40, driver 595.99.02, container e8af6510d680.
- Raw files: this directory.

## From ADR 0050 (note 2026-11-10b "the KL floor, and gating on the cost above it", moved verbatim)

**Second run** (`bench/results/20261110b/kv-quality.json`, box 60050186, A40). Mean KL against the builder F16 reference, 8K / 32K / calibration:

| row | mean KL | top-1 |
|---|---|---|
| builder F16, rotation off (the reference again) | 0 / 0 / 0 | 1 / 1 / 1 |
| builder F16, rotation on | 0.0093 / 0.0089 / 0.0068 | 0.960 / 0.955 / 0.963 |
| libllama F16 | 0.0079 / 0.0072 / 0.0077 | 0.958 / 0.960 / 0.968 |
| builder Q8_0/Q8_0, rotation on (gating) | 0.0107 / 0.0095 / 0.0075 | 0.958 / 0.963 / 0.968 |
| libllama Q8_0/Q8_0 | 0.0103 / 0.0094 / 0.0076 | 0.956 / 0.958 / 0.968 |
| builder Q8_0/Q4_0, rotation on (gating) | 0.0169 / 0.0145 / 0.0145 | 0.933 / 0.946 / 0.953 |

**Cause, measured.** Not a rotation bug: a wrong inverse, a one-sided rotation or a wrong scale changes attention scores outright (KL far above 1), and an overflow gives NaN; the rotated builder matches libllama bit for bit on the CPU (`kv_rotation.rs`), on CUDA both take the FWHT kernel (its dispatch conditions hold for these shapes, `fwht.cu`), and builder and libllama Q8_0/Q8_0 land at the same KL. Not run-to-run noise: the reference against itself is exactly 0. And the cost does not scale with the size of the perturbation: F16 rounding of rotated values, another engine's kernels and Q8_0 rounding (about 16 times coarser) all cost 0.007 to 0.011, where smooth error would scale with its square. So the bounds of 0.003 and 0.99 lie below what F16 itself scores.

**Cause, hypothesis (not measured).** Discontinuities saturate the cost: every quantized-weight matmul on CUDA quantizes its activations to Q8_1 (MMQ, `mmq.cu`), so an ulp change flips 8-bit rounding decisions, and the router picks 8 of 128 experts in each of 48 layers, so flipped choices reach every later position through the cache. Unexplained: the Q8_0-weights anchor (0.002979, `llama-perplexity --kl-divergence` against BF16 logits, ADR 0048) is a much larger perturbation that scores below this floor on the same calibration text. The same-engine rows below test this: under saturation libllama Q8_0 against libllama F16 lands near the floor too; if it lands near 0, the builder is the outlier and this diagnosis is wrong.
**Decision.** Gate on the cost above the floor, paired by position. The floor row is the builder's F16 cache with the rotation forced on: same engine, same batching, same rotation as the gating rows, so the KV format is the only difference. Per set, excess = mean over positions of KL(gating row) minus KL(floor row), both against the reference, with its standard error; excess top-1 loss = floor top-1 minus gating top-1. Bounds, provisional until a run has the new floor rows: Q8_0/Q8_0 excess KL at most 0.003 and excess top-1 loss at most 0.01; Q8_0/Q4_0 at most 0.015 and 0.03 (the ADR's numbers, read as cost above the floor). A row whose bound lies within two standard errors of its excess is flagged `marginal`. The absolute bounds stay in the JSON (`abs_pass`), not gated. On 20261110b the excess would be 0.0014 / 0.0006 / 0.0007 (Q8_0/Q8_0) and 0.0076 / 0.0056 / 0.0077 (Q8_0/Q4_0); those numbers are not why the bounds are what they are.
**Consequence.** One more builder run (a minute or two on the A40, the small ubatch is slower); peak host memory is the reference, one baseline and the current row (about 11 GB). The floor varies by about 0.0015 between perturbations (builder rotated against libllama F16 above), half the Q8_0/Q8_0 bound. The standard error treats positions as independent, but positions in one sequence share the perturbation through the cache, so it is a lower bound and `marginal` is optimistic. If the rerun shows marginal rows, the next step is more scored positions or more floor samples, not looser bounds.

## From ADR 0057 (addendum "rerun 20261110b", moved verbatim)

Tasks 16/20 on both engines; tool errors 44 of 369 calls (11.9%) against llama-server's 27 of 550 (4.9%). No `input_validation` errors are left. Pairing each `is_error` result in the request logs with its `tool_use`:

| Proxy class | Count | What it is |
|---|---|---|
| other | 37 | Bash exit code non-zero: 31 of 80 pytest runs, 6 of 47 python runs |
| other | 5 | Edit string not found (2), Edit no-op (1), Read on a missing file (1) or a directory (1) |
| unknown_tool | 2 | `Explore` (a subagent type of `Agent`, not a tool), parsed correctly |

The 42 "other" errors are task outcomes; llama-server's 27 cannot be split from the committed data (no request log). The proxy now classes them `bash_exit` and `file`, and the summary adds `call_error_rate` (input_validation plus unknown_tool per tool call), the part the gateway answers for.

**Bare `<function=` opener.** The one leaked marker (tetraeder-r2-rename-compute-total, first response) was a call without `<tool_call>`: `<function=Bash>...</function>\n</tool_call>`. The parser sent it as text, Claude Code ended the session after one request and the task failed. llama.cpp's grammar triggers on `<function=NAME>` as well. Decision: in the Qwen XML format, `<function=NAME>` starts a call when NAME is one of the request's tools and `<parameter=` or `</function>` follows; it ends at `</function>` (an orphan `</tool_call>` after it is consumed) or at `</tool_call>`. Otherwise only the opener is text and scanning goes on, so prose that mentions `<function=` costs no later call. Outcome `repaired`/`no_opener`, counted in `ToolCallFailures` like `empty_call` (1 of 369 calls here). Test: `qwen_xml_call_without_opener_from_run_fixture`, from `toolcall-fixtures-r2.json`.

**Unknown names.** `Explore` twice against none on llama-server meets this ADR's condition for revisiting a lazy grammar under `auto`. Still not done: it needs a decode benchmark on a box (Rule 3).
