# 0086. The planner picks the KV format (`--kv auto`)

Status: Accepted (owner, 2026-11-23, ADR 0085 class 1). Tree node `memory-kv-auto`.

## Context

- The KV formats exist (ADR 0050 b: F16, BF16, Q8_0, Q8_0/Q4_0, Q4_0) but every planner path sized F16 and `serve` never set `ModelOpts.kv`, so a context that fits only in Q8_0 was refused.
- Strata picks its KV format from free memory and slots (`ink_cuda_backend.cu:148-160`, matrix `docs/notes/strata-parity.md`).
- The pool is `--ctx` rows shared by all sequences (ADR 0067), so parallel slots do not multiply the KV bytes.

## Decision

1. `kv = auto` is a new choice beside the fixed formats. The default stays F16 (rule 3: no default change without a paired run).
2. Auto tries F16, Q8_0, Q8_0/Q4_0 in that order and takes the first with a feasible plan at the requested context, planning within it unchanged. BF16 and Q4_0 are never picked. Quantized formats are skipped when the head size is not a multiple of 32.
3. With a GPU the ladder runs on the GPU placements first; the CPU is tried only after all three formats failed there, so a CPU plan in F16 never hides a GPU plan in Q8_0. If none fits, the error lists each format with its numbers.
4. The pick and its reason (`f16/f16 needs 19.3 GB KV, 14.1 GB free; q8_0/q8_0 fits (10.3 GB KV)`) go in the notes of every candidate and on stderr of `serve`. The plan carries the format in the `kv_format` of its attention rules (F16 stays implicit).
5. `tetraeder serve --kv FORMAT|auto` sets the own builders' format. libllama and `--plan` keep F16 (the engine has no format option there, and the replanner sizes F16); `--kv auto` then says so, a fixed other format is refused. With `auto` the builders are sized at the placement their flags give (`--expert-slots`, `--ram-slots`, the whole model on the GPU), against the discovered GPU capacity.
6. Elastic KV (growing the pool at run time) is out of scope.

## Consequences

- Tests (rules 4 and 5): on every simulated hardware fixture and both layouts, auto matches an independent footprint oracle at contexts at and around each format's limit (F16, only Q8_0, only Q8_0/Q4_0, none); a property test checks that the chosen plan never exceeds VRAM or RAM with the KV bytes of the chosen format counted.
- Metric (rule 6): `tetraeder.memory.used` and `.capacity` of kind `kv` already carry the format (`tetraeder.format`) and the overview panel "KV pool" groups capacity by it, so the chosen format is visible without a new panel.
- Open: the replanner (`serve --plan auto`) and `recognize` still handle F16 only; the planner's `candidates` command feeds llama-bench and `exit_regime` campaigns whose lowering has no KV option, so it has no `--kv`. Judged by a paired run on a model we run (not measured).
