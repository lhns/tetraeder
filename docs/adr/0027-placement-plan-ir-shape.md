# 0027. Placement plan IR: assignment rules over a model description, one expanded form for checks, diff and migration

Status: Accepted (Phase 2)

## Context

ADR 0006 asks for one plan IR, a validator, a per-phase capability table and migration DAGs; ADR 0024 makes the weight format a per-unit plan field; ADR 0025 makes limits and device rules an input next to the plan. A 60-layer model with 128 experts per layer has about 7700 units, so a plan that lists every unit is not hand-editable, while the checks need every unit. The profiler gives every device exactly one tier and has no node concept yet.

## Decision

- **Model description** (`ModelDesc`, in `plan`, no ggml): layer count, heads, KV heads, routed experts and top-k, and per unit kind (embedding, attention, MLP, expert, head) the parameter count, FLOPs per token and source format. Units are `embed`, `layer.N.attn`, `layer.N.mlp` (dense FFN, or router plus shared experts), `layer.N.expert.E`, `head`. The model crate fills it from GGUF later.
- **Plan** (TOML and JSON): data-parallel `replicas`, each a list of pipeline `stages`. A stage has a half-open layer span (spans tile the model in order; embedding in the first stage, head in the last), a compute device group, tensor and expert parallel degrees, `assign` rules, KV and workspace bytes per device, and expert slot caches. An `assign` rule selects units (kinds, layer span, expert span) and gives one home device per shard, optionally a compute device per shard, and a format. Knobs: prefill micro-batches, CUDA graphs, batch limit.
- **Location is a device id; tier is the device's tier** in the hardware graph (VRAM, RAM, NVMe, remote). The plan stores no tier, so it cannot disagree with the hardware. `Tier::Remote` is the only cross-node marker until the hardware graph models nodes.
- **One expanded form.** `expand` turns a complete plan into a map from `(replica, unit, shard, shard count)` to `(stage, home, compute, format, bytes)` plus byte buffers (KV, workspace, slot caches) per device. The validator, diff and migration DAG all work on it. A degree change shows up as removed and added shards because the key holds the shard count.
- **Validator** (complete plans): units covered exactly once per replica; shard counts match degrees (attention and MLP have `tp` shards on distinct compute devices, experts one); `tp` divides heads, KV heads and the group size, `ep` divides experts and the group size, and with `ep > 1` each MoE layer's experts are spread evenly over `ep` compute devices; compute devices are in the group and can compute; per device, weights plus buffers stay within capacity minus the measured reserve (an input map from device id to bytes until the profiler measures it); formats only convert downward (ADR 0024); hard hints (pin, forbid, format floors) hold; fields the executing phase cannot run are rejected. All issues are collected, not just the first.
- **Partial plans:** `validate_partial` checks only what is set. Open fields are a stage with no devices, an absent degree, an absent home or format in a rule, absent KV budgets and units no rule selects. A missing compute device always means "computes where its home is".
- **Phase capability table** in code, one test per row of docs/plan.md "What each phase can execute". Phases are integers; 3 stands for both 3a and 3b, which execute the same fields.

  | Field | 1 | 2 to 4 | 5 | 6 on |
  | --- | --- | --- | --- | --- |
  | Layer split, homes on CPU or GPU | yes | yes | yes | yes |
  | Format other than the source | no | yes | yes | yes |
  | Expert homes on disk, slot caches | no | yes | yes | yes |
  | Compute device other than home | no | experts only | yes | yes |
  | `tp`, `ep` above 1 | no | no | yes | yes |
  | Several replicas, remote devices | no | no | no | yes |

- **Formats:** the ggml block size and block bytes per format are copied from `ggml-common.h` at the pin (b11976) into `plan`. "At least as precise" means at least as many bits per weight; formats with equal bits per weight count as equally precise.
- **Hints** (ADR 0025 and the Hints table in docs/plan.md) are a separate serde type. The validator enforces the hard ones (pin, forbid, format floors); limits, trade-off, preferences, costs, workload, stability and advisor policy go to the planner.
- **Migration DAG:** drain, then per shard load the new placement before freeing the old one (shards independent), buffer and knob changes, then prepare and commit. Its property test applies steps in random topological orders to the source state and requires the target's expanded form at the end.

## Consequences

- One generator of valid plans serves the property tests (memory never exceeded, every unit computed exactly once, every migration DAG ends in its target plan) on every hardware fixture.
- The simulator converts a single-replica plan without tensor or expert parallel into its pipeline stub; richer plans need a richer simulator.
- Not yet in the IR: per-layer size overrides (dense first layers in DeepSeek-style models), stage edges with an explicit transport, separate prefill and decode plans, speculative decoding knobs, nodes in the hardware graph, and memory-aware ordering of migration steps (Phase 7, live reshard). Each is a field added when its phase needs it.
