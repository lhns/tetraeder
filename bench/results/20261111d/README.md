# Session 20261111d

- Status: clean; proxy rerun on current main: cache and non-200 clauses met, tasks clause one short in this run, met over all runs under the owner's 2026-11-11 tasks rule (ADR 0046).
- Code and box: main e426f8e; box 60205005, RTX A6000 48 GB, Xeon W-2275, 26 threads.
- Raw files: this directory.

## From docs/m2-plan.md, section "D2 split into two sessions" (moved verbatim)

Time budget: record-log (5 longest Claude Code tasks, `20261111d`) took 12.8 min and replays 3.8 M prompt tokens, so each reuse-off replay of the 8B is about 13 to 20 min on a 3090 and the TTFT stage about 1 h; with build, gpu-parity, gpu-shapes, prepass, probe, both models' pairs and downloads one session would be about 2 h 45 min, over the 2 h 15 min a 3 h box leaves. So: **D2a** throughput (gates, prepass, probe, pairs for the 8B dense and the 30B MoE), **D2b** TTFT on the 8B (record-log, the reuse on/off pairs, the informative llama-server run). Both on RTX 3090 (baseline built for arch 86; the noise file has 11 sessions), `--min-inet 1000`, RAM checked at SSH, ranked by box plus about 30 GB (D2a) or 10 GB (D2b) of traffic; at SSH `/opt/llama.cpp-baseline` must report fc504f91 or the box is destroyed. Before D2a: the harness gaps (`track-d2-gaps`: rehearsal against the b12426 server, a moe stand-in, batched real-model parity, per-model stop rule). Note for reading the results: the baseline has CUDA MoE kernel work (#32067) the b11976 engine lacks, and gpu-shapes compares against the pinned libllama, not the baseline; if the 30B probe is low while the 8B passes, the deferred pin is the first suspect.

## From ADR 0046 (addendum "rerun on current main", moved verbatim)

## Addendum (2026-11-11, box 60205005): rerun on current main

RTX A6000 48 GB, Xeon W-2275, 26 threads, same harness and tasks (`bench/results/20261111d/`); main at e426f8e, so the request log now carries `tool_outcomes`. Session shortened to 130 min (BUDGET_MIN 105) to stay inside the 3 h box cap.

| | Tetraeder | llama-server |
|---|---|---|
| Tasks solved | 15/20 | 16/20 |
| Prompt-cache share | 95.07% | 94.52% |
| Non-200 responses | 0 | 1 |
| Tool-error rate | 5.1% (19 bash_exit, 5 file, 1 unknown_tool of 489) | 10.8% (58, 11, 6 other of 695) |
| Call error rate | 0.2% | 0 |
| Server decode, median | 1.021 of llama-server | |

Tool-call outcomes: parsed/ok 486, parsed/unknown_tool 1, repaired/no_opener 2, repaired/empty_call 1, failed/bad_xml 2; two leaked markers. Tetraeder solved 0/2 of `fix-rounding` and `feature-late-fee`. Resident decode gap closed (`decode_gap.sh`: 189.8 vs 190.5 tok/s at depth 0, 123.5 vs 123.3 at 20K, 52.0 vs 51.5 at 100K); kv-quality gating rows pass under the excess-over-floor gate. Verdict: the cache and non-200 clauses are met; the tasks clause misses by one task in this run after passing by one in 20261110d. Pooled over both runs on the same harness: 30/40 each. A one-task difference in 20 is within the run-to-run spread, so the tasks clause is a tie, not a clear pass; whether a tie meets "at least as many" over pooled runs is the owner's call.

## From ADR 0046 (addendum "owner decision on the tasks clause", 2026-11-11, the pooled numbers over all proxy runs `20261103`, `20261110`, `20261110b`, `20261110d`, `20261111d`, moved verbatim)

The plan's clause reads "tasks solved at least as many as llama-server" for one run. All proxy runs, Tetraeder against llama-server: 15/15 (`20261103`), 12/13 (`20261110`), 16/16 (`20261110b`), 15/14 (`20261110d`), 15/16 (`20261111d`); per-run differences -1 to +1, pooled 73/100 against 74/100, on the current harness (last two) 30/40 each. No task fails for Tetraeder alone: `fix-rounding` 2/10 against 1/10, `feature-late-fee` 5/10 each. At 20 tasks per run the clause as written is a coin flip.
Decision (owner, 2026-11-11, post hoc; the original text stays in the plan): the tasks clause is met when Tetraeder is no worse than llama-server beyond the run-to-run spread, judged over all runs, with no task failing for Tetraeder alone. Met: pooled 73 against 74 and every run within one task. The cache and non-200 clauses are met on both current-harness runs (cache 95.8 vs 94.2 and 95.07 vs 94.52; one of four earlier runs missed cache by 0.34). Tool-error rate is not a clause and flips between runs (10.0 vs 6.8, then 5.1 vs 10.8).
