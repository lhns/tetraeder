# Session 20261111d

- Status: clean; proxy rerun on current main: cache and non-200 clauses met, tasks clause one short in this run, met over all runs under the owner's 2026-11-11 tasks rule (ADR 0046).
- Code and box: main e426f8e; box 60205005, RTX A6000 48 GB, Xeon W-2275, 26 threads.
- Raw files: this directory.

## From docs/m2-plan.md, section "D2 split into two sessions" (moved verbatim)

Time budget: record-log (5 longest Claude Code tasks, `20261111d`) took 12.8 min and replays 3.8 M prompt tokens, so each reuse-off replay of the 8B is about 13 to 20 min on a 3090 and the TTFT stage about 1 h; with build, gpu-parity, gpu-shapes, prepass, probe, both models' pairs and downloads one session would be about 2 h 45 min, over the 2 h 15 min a 3 h box leaves. So: **D2a** throughput (gates, prepass, probe, pairs for the 8B dense and the 30B MoE), **D2b** TTFT on the 8B (record-log, the reuse on/off pairs, the informative llama-server run). Both on RTX 3090 (baseline built for arch 86; the noise file has 11 sessions), `--min-inet 1000`, RAM checked at SSH, ranked by box plus about 30 GB (D2a) or 10 GB (D2b) of traffic; at SSH `/opt/llama.cpp-baseline` must report fc504f91 or the box is destroyed. Before D2a: the harness gaps (`track-d2-gaps`: rehearsal against the b12426 server, a moe stand-in, batched real-model parity, per-model stop rule). Note for reading the results: the baseline has CUDA MoE kernel work (#32067) the b11976 engine lacks, and gpu-shapes compares against the pinned libllama, not the baseline; if the 30B probe is low while the 8B passes, the deferred pin is the first suspect.
