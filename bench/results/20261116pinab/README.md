# Session 20261116pinab

- Status: diagnosis, not a verdict: pin A/B of llama-server b11976 against b12426 on one RTX 3090; part of the c4 gap is the engine pin (led to ADR 0073).
- Code and box: results commit 28914ad; box 60924954, host 79869, RTX 3090, driver 580.126.09.
- Raw files: this directory.

## From docs/m2-plan.md, section "Pin A/B: the c4 gap has two parts" (moved verbatim)

- The pin: llama-server built from b11976 reaches 0.930 (8B) and 0.834 (30B) of b12426 at c4, level at c1; libllama's 4-sequence decode step is 83.4 ms per 8 steps at b11976 against 71.8 at b12426. **Pin adoption moves onto the critical path ahead of the verdict** (reverses the deferral above; `track-pin-adopt`), then one GPU parity session on the new pin.
- Our loop, against llama-server at the same pin: 8B 0.906, 30B 1.038 (b11976's server has a 0.96 s TTFT on the 30B). The 30B step log shows 36 to 54 ms of host time before every admitting step (the prefix-cache save of the reused idle sequence) and one 400-token prompt admitted per step under the 512 budget (mixed steps 143 to 149 ms against 97 ms for a prompt alone), where llama-server admits a wave in one step. Fixes on `track-c4-loop`: no save at admission while KV room exists, step budget 2048.

## From ADR 0073 Context (moved verbatim)

- On one RTX 3090 (`bench/results/20261116pinab/pin-ab/summary.md`) llama-server built at b11976 reaches 0.930 (Qwen3-8B) and 0.834 (Qwen3-30B-A3B) of llama-server b12426 at concurrency 4, and libllama's 4-sequence decode step takes 83.4 ms per 8 steps at b11976 against 71.8 at b12426; one-sequence steps are level. Part of the c4 gap is the pin, and the verdict compares against b12426.
