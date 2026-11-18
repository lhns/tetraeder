# Session 20261110d

- Status: clean; Phase 3b proxy bar met on all three clauses (ADR 0046 2026-11-08 bar), one run per engine; ADR 0050 (b) KL acceptance passes under the gate on the cost above the floor.
- Code and box: commit on main at the time of box 60067423 (see git log of the results commit); A40 48 GB, Claude Code 2.1.322.
- Raw files: this directory.

## From ADR 0046 (addendum "Phase 3b proxy bar met", moved verbatim)

## Addendum (2026-11-10, box 60067423): Phase 3b proxy bar met

A40 48 GB, Qwen3-Coder-30B-A3B Q4_K_M, Claude Code 2.1.322, 10 tasks x 2 reps per engine, ABBA blocks (`bench/results/20261110d/claude-code-proxy/`). Since the 20261110b run: ADR 0057 (Python-literal arguments), the bare `<function=` repair, `CLAUDE_CODE_ATTRIBUTION_HEADER=0` for both engines (the per-session billing header is gone from the prompt start).

| | Tetraeder | llama-server |
|---|---|---|
| Tasks solved | 15/20 | 14/20 |
| Prompt-cache share | 95.8% | 94.2% |
| Non-200 responses | 0 | 0 |
| Tool-error rate (bash_exit, file, other, unknown_tool) | 10.0% (41, 11, 1, 1) | 6.8% (25, 0, 3, 0) |
| Call error rate (validation and unknown tool per use) | 0.2% | 0 |
| Server decode, median | 0.992 of llama-server | |

All three clauses are met. The tool errors are task outcomes (failing tests, missing files), not parse errors; the one unknown tool is the model's misspelt `ScheduledWakeup`. The 20261110b run had missed the cache clause by 0.34 points; the cause was session length against a fixed uncached first request, and one billing-header change (ADR 0057 addendum), not the cache. One run per engine, so the margins carry run-to-run noise; the clauses are met in this run and the previous one met tasks and 400s too.
