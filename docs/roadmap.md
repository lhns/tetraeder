# Roadmap

Everything between the GENERATED markers is rebuilt from `roadmap/nodes.yaml`; edit that file, not the regions. Text outside the markers is kept.

## Legend

<!-- BEGIN GENERATED: legend -->
```mermaid
flowchart TB
    %% legend
    subgraph lg_g0["Done: colour is the rating"]
        direction LR
        lg_s0["✔ very successful"]
        lg_s1["✔ successful, or fixed"]
        lg_s2["✔ mixed"]
        lg_s3["✔ bad (regression)"]
        lg_s4["✔ done, not rated"]
    end
    subgraph lg_g1["Closed"]
        direction LR
        lg_s5["✖ dead end"]
        lg_s6["↷ superseded"]
    end
    subgraph lg_g2["Open: colour is the state"]
        direction LR
        lg_s7["▶ in progress"]
        lg_s8["⏳ built, waiting"]
        lg_s9["○ next"]
        lg_s10["⛔ blocked"]
        lg_s11["⏸ parked"]
        lg_s12>"⚠ open, never rated"]
    end
    subgraph lg_shape["Shape: kind"]
        direction LR
        lg_k0["○ feature"]
        lg_k1[/"○ experiment"/]
        lg_k2("○ session")
        lg_k3["○ refactor"]
        lg_k4["○ tooling"]
        lg_k5>"⚠ defect"]
        lg_k6{{"○ owner-decision<br/>(open, not drawn)"}}
        lg_k7{{"✔ owner-decision<br/>(decided)"}}
    end
    subgraph lg_edge["Arrow (in the trees every arrow points down, forward in time)"]
        direction LR
        lg_e0a["prerequisite"]
        lg_e0b["depends on it"]
        lg_e1a["old design"]
        lg_e1b["replaced by"]
        lg_e2a["rated node"]
        lg_e2b["qualified by a later finding"]
    end
    lg_g0 ~~~ lg_g1 ~~~ lg_g2 ~~~ lg_shape ~~~ lg_edge
    lg_e0a --> lg_e0b
    lg_e1a -.-> lg_e1b
    lg_e2a -.-> lg_e2b
    linkStyle 6 stroke:#e69500,stroke-width:3px,stroke-dasharray:2 3
    classDef vsucc fill:#2e9e4f,stroke:#1b6b34,color:#fff
    classDef succ fill:#bfe8c8,stroke:#2e9e4f,color:#111
    classDef mixed fill:#ffd37a,stroke:#c98a00,color:#111
    classDef bad fill:#f28b82,stroke:#b3261e,color:#111
    classDef dead fill:#e6e6e6,stroke:#777,stroke-dasharray:5 5,color:#555
    classDef inprog fill:#a8cdf5,stroke:#1f6fc5,color:#111
    classDef next fill:#fff,stroke:#333,color:#111
    classDef blocked fill:#fff,stroke:#b3261e,stroke-dasharray:2 2,color:#111
    classDef parked fill:#f3f3f3,stroke:#999,color:#555
    classDef unrated fill:#fff,stroke:#333,stroke-dasharray:1 3,color:#111
    classDef gate fill:#d9c2f0,stroke:#7b3fb5,color:#111
    classDef decided fill:#ece0f7,stroke:#7b3fb5,stroke-dasharray:2 2,color:#111
    classDef finding fill:#ffe0b2,stroke:#d84315,stroke-width:2px,color:#111
    classDef waiting fill:#e3f2fd,stroke:#1f6fc5,stroke-dasharray:5 5,color:#111
    class lg_s0 vsucc
    class lg_s1 succ
    class lg_s2 mixed
    class lg_s3 bad
    class lg_s4 unrated
    class lg_s5,lg_s6 dead
    class lg_s7 inprog
    class lg_s8 waiting
    class lg_s9,lg_k0,lg_k1,lg_k2,lg_k3,lg_k4,lg_e0a,lg_e0b,lg_e1a,lg_e1b,lg_e2a,lg_e2b next
    class lg_s10 blocked
    class lg_s11 parked
    class lg_s12,lg_k5 finding
    class lg_k6 gate
    class lg_k7 decided
```

Owner calls are the purple shape. A decided owner call is drawn in the overview only, never in the track trees; an open one is listed under *Waiting on the owner*. The prefix on an overview box is its track. A prerequisite implied by a longer chain is not drawn; the node tables keep every edge.
<!-- END GENERATED: legend -->

## Ratings

<!-- BEGIN GENERATED: ratings -->
No node carries a rating yet.
<!-- END GENERATED: ratings -->

## Overview

<!-- BEGIN GENERATED: overview -->
```mermaid
flowchart TB
    subgraph bd_0["Nov 2026"]
        multi_prefill_no_capture["multi · ✔ Prefill runs<br/>without graph capture"]
    end
    classDef vsucc fill:#2e9e4f,stroke:#1b6b34,color:#fff
    classDef succ fill:#bfe8c8,stroke:#2e9e4f,color:#111
    classDef mixed fill:#ffd37a,stroke:#c98a00,color:#111
    classDef bad fill:#f28b82,stroke:#b3261e,color:#111
    classDef dead fill:#e6e6e6,stroke:#777,stroke-dasharray:5 5,color:#555
    classDef inprog fill:#a8cdf5,stroke:#1f6fc5,color:#111
    classDef next fill:#fff,stroke:#333,color:#111
    classDef blocked fill:#fff,stroke:#b3261e,stroke-dasharray:2 2,color:#111
    classDef parked fill:#f3f3f3,stroke:#999,color:#555
    classDef unrated fill:#fff,stroke:#333,stroke-dasharray:1 3,color:#111
    classDef gate fill:#d9c2f0,stroke:#7b3fb5,color:#111
    classDef decided fill:#ece0f7,stroke:#7b3fb5,stroke-dasharray:2 2,color:#111
    classDef finding fill:#ffe0b2,stroke:#d84315,stroke-width:2px,color:#111
    classDef waiting fill:#e3f2fd,stroke:#1f6fc5,stroke-dasharray:5 5,color:#111
    class multi_prefill_no_capture unrated
```
<!-- END GENERATED: overview -->

## In plain words

<!-- BEGIN GENERATED: plain-words -->
| title | status | in plain words |
|---|---|---|
| Price decode at measured clocks | ○ next | A GPU kept partly idle by a layer split runs at a lower clock than one GPU alone, so the 30B four-way layer split decodes slower than predicted wherever the single GPU boosts; the cost model has to price decode at the clock each layout actually gets. |
| Prefill runs without graph capture | ✔ done · unrated | Every judged prefill had timed the call in which ggml records a CUDA graph, which serialized the pipeline stages; prefill steps now run directly and decode keeps graph replay. |
| Accept the prefill capture fix | ○ next | ADR 0081 is Proposed; its GPU test and the s3f2 confirmation passed, so it waits for the owner's acceptance. |
| Two-GPU verdict session | ▶ in progress | The judged Phase 5 session on two GPUs (cases D2 and M2): every candidate within 15% of its prediction and the pick in the winner set, two clean rounds each. |
| Admission probe on final main | ○ next | The admission path changed after the Phase 4 verdict commit, so one D2a run on final main confirms the Phase 4 result still holds before M2 is reported. |
| Four-GPU verdict, first box | ⛔ blocked | The judged session on four GPUs (cases D4 and M4) on a host other than 20210, run alone; the owner allowed one box up to 1.70 USD/h for it. |
| Four-GPU verdict, second box | ⛔ blocked | The second four-GPU session, so every configuration has two clean rounds on a clean host. |
| Report milestone M2 | ⛔ blocked | M2 ends Phase 5: the owner judges the verdicts together with the list of rule changes made after failed sessions (docs/notes/phase5-rule-changes.md). |
<!-- END GENERATED: plain-words -->

## Ready now

<!-- BEGIN GENERATED: ready-now -->
**local**

- `planner-cost-model-v5`: Price decode at measured clocks

**box**

- `bench-d2a-probe`: Admission probe on final main
<!-- END GENERATED: ready-now -->

## Running

<!-- BEGIN GENERATED: running -->
**box**

- `bench-v2-verdict`: Two-GPU verdict session
<!-- END GENERATED: running -->

## Built, waiting

<!-- BEGIN GENERATED: built-waiting -->
_None._
<!-- END GENERATED: built-waiting -->

## Waiting on the owner

<!-- BEGIN GENERATED: waiting-on-owner -->
- `multi-accept-adr-0081`: Accept the prefill capture fix
- `bench-m2-report`: Report milestone M2
<!-- END GENERATED: waiting-on-owner -->

## Queue

<!-- BEGIN GENERATED: queue -->
1. `bench-v2-verdict`: Two-GPU verdict session
2. `bench-d2a-probe`: Admission probe on final main
3. `bench-v4a-verdict`: Four-GPU verdict, first box
4. `bench-v4b-verdict`: Four-GPU verdict, second box
<!-- END GENERATED: queue -->

## Timeline

<!-- BEGIN GENERATED: timeline -->
[The timeline](roadmap/timeline.md) shows every dated node as a bar from its start to its end.

Generated on 2026-11-20.
<!-- END GENERATED: timeline -->

## Done, by month

<!-- BEGIN GENERATED: done-by-month -->
<details><summary><b>Nov 2026</b>: 1 done</summary>

- `multi-prefill-no-capture`: Prefill runs without graph capture (✔ done · unrated). Every judged prefill had timed the call in which ggml records a CUDA graph, which serialized the pipeline stages; prefill steps now run directly and decode keeps graph replay.

</details>
<!-- END GENERATED: done-by-month -->

## Tracks

<!-- BEGIN GENERATED: tracks -->
Each track has its own page with its tree, node table and per-node details.

- [engine - engine and builders](roadmap/tracks/engine.md): 0 nodes
- [memory - memory and streaming](roadmap/tracks/memory.md): 0 nodes
- [planner - planner and cost model](roadmap/tracks/planner.md): 1 nodes (1 next)
- [multi - multi-GPU executor](roadmap/tracks/multi.md): 2 nodes (1 done, 1 next)
- [serving - serving](roadmap/tracks/serving.md): 0 nodes
- [bench - benchmarks and verdicts](roadmap/tracks/bench.md): 5 nodes (1 in progress, 1 next, 3 blocked)
- [ops - operations](roadmap/tracks/ops.md): 0 nodes
<!-- END GENERATED: tracks -->
