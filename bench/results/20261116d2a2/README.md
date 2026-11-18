# Session 20261116d2a2

- Status: null (30B pairs unclean on throttle under the single before-sample rule then in force), c4 miss: Phase 4 D2a rerun, not a pass (ADR 0065). Confirms the MoE output-row fix on GPU (ADR 0071).
- Code and box: commit unknown (env.json `git_commit` empty); box 60919833, host 159250, RTX 3090, driver 615.71.09.
- Raw files: this directory.

## From docs/m2-plan.md, section "D2a rerun result and the c4 gap" (moved verbatim)

- Parity closed on GPU for both models; c4 misses (8B probe 0.830, 30B pairs 0.845 to 0.853), c1 level. Details in the ADR 0065 addendum of the same date.
- The dense "slowdown between boxes" was the first box, not the code: on one 3090, dense against libllama per step is 0.99 to 1.03 at both 030d1f3 and HEAD (`dense-regress/table.md`).

## From ADR 0065 (addendum "D2a rerun misses at c4", moved verbatim)

- Gates pass for both models: gpu-parity (the 30B alone control is exactly 0 on every row incl. (0,19)), gpu-shapes (dense within 3% of libllama per step).
- Qwen3-8B (own dense): probe c4 0.830, stopped by the 0.85 rule; c1 1.003.
- Qwen3-30B-A3B (own MoE): pairs c4 0.853 and 0.845, c1 1.026 and 1.029, A/A 0.011. The pairs are flagged unclean by the throttle check (0x600 before, 0x4 after on driver 615), so `pass` is null; the c4 ratio is below the 0.925 line either way, so this is a c4 miss, not an open result. The throttle rule is not loosened for this session.
- The libllama control through our scheduler reaches the same c4 (260.9 against 257.2 tok/s for the own MoE, 301.4 for llama-server): the builders are not the cause. Two candidates of similar size remain: our loop (about 11.9 ms per 4-row step against 10.7 ms for the pinned libllama alone, TTFT spread 0.42 to 0.74 s against 7 ms) and the pin (pinned libllama 10.7 ms per 4-row step against about 9.4 ms for llama-server b12426; 1-row steps are level). Under test: `track-c4-loop` (loop, CPU) and `track-pin-ab` (b11976 against b12426 llama-server and step shapes on one GPU).
- Before the next verdict box: decode the 0x200 and 0x400 throttle bits of driver 615 and settle whether SW power cap after a loaded run counts; `env.json` records an empty `git_commit`.

## From ADR 0065 (addendum "SW power cap after a run is benign", moved verbatim; the count spans this session and `20261116pinab`)

- Why: the after sample is taken right after the c4 run ends, at the same point for every engine, so a power cap still active from the load is the same for both sides of a pair; the RTX 3090 noise band (`bench/noise/rtx-3090.json`, 11 sessions from llama-bench, not filtered by throttle) already prices power capping. 0x4 after a run appeared in 13 of 22 records in `20261116d2a2` and `20261116pinab`, on drivers 580 and 615, for baseline and Tetraeder runs alike.
