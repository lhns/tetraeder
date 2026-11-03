# M1: open items (2026-11-03)

State at the end of the 2026-11-03 parallel-tracks session. Everything below is on main; details in the ADRs and `bench/results/20261103/`. Boxes: none up. Box work was paused by the user mid-session; ask before renting.

## Where each phase stands

| Phase | Exit test | Status |
|---|---|---|
| 2 MoE builder, model larger than RAM | at least as fast as the best baseline, logit parity | Prefill leads every baseline (1.4x to 3x). Decode beats llama.cpp `-ot` (1.23x to 1.60x), Colibri, ik_llama (cannot run). **Behind KTransformers 1.34x on a 26-core box** (3.77 vs 5.04 tok/s; tie on a 15-thread box). Parity: greedy equal; logit offset explained as GPU kernel fusion (ADR 0045 addendum). |
| 3a Formats and quality | quality per byte at least llama.cpp presets | Pass: profile mixes beat Q3/Q4/Q5_K_M on Qwen3-8B and 30B-A3B (BF16 reference). Conversion 11 to 15% faster than llama-quantize, byte-identical (ADR 0047, 0048). |
| 3a Planner | regret, rank, beat llama.cpp fit | Pass on the MoE fixture (regret 0, Spearman 0.997, pick 1.44x llama.cpp `-fitt`). Dense fixture partial (11 of 24). |
| 3a Detector | top-1 faults on real hardware | Passes in the simulator only; box campaign not run. |
| 3b Gateway | SDK conformance, Claude Code use, vs llama-server | Conformance passes on real models. Claude Code proxy (user chose a few-hour proxy over a week): 15/20 tasks on both engines; decode 0.88x on an A6000; 400s and prompt cache fixed afterwards (ADR 0049), not re-measured. vLLM not run (no GGUF support worth pairing). |
| Speculative decoding (ADR 0043) | faster than plain decode | Exact, but slower in the exit regime; op-offload fix (ADR 0044) in, rerun pending. |

## Open, by priority

1. **Phase 2 decode vs KTransformers.** One 91 GB-class box session, at least 11 cores: `scripts/exit_session.sh` with `VRAM_SLOT_BYTES=10966007808` and `OT_FIT='blk\.[0-6]\.ffn_.*_exps=CUDA0;exps=CPU'`. It measures the implemented, never-measured levers (prefetch depth 2 actually on, reader 1 MiB at queue depth 16, reads before the layer's own misses; expected 6 to 6.7 tok/s), then re-pairs KTransformers for a noise band. Next lever if short: RAM tier kept by routing frequency (measure offline from recorded routing first). ADR 0038 addendum has the diagnosis.
2. **Speculative rerun:** `STAGES=main-builder bash scripts/spec_run.sh` (VRAM fix in; the draft now fits). Same box session as 1 if time allows.
3. **Claude Code proxy rerun** on a 48 GB box at most 0.50 USD/h (user exception, up to 6 h): `scripts/claude_code_proxy.sh` with `REPS=2 BUDGET_MIN=240 STREAMING=1`. Bar: tasks solved at least llama-server's, prompt-cache reuse at least its 94%, zero 400s.
4. **Resident decode gap on the A6000** (0.88x llama-server, Qwen3-Coder-30B-A3B, 128K): profile build, alloc, compute and readback per token and CUDA graph reuse at depth 0 and 20K; suspects are graph rebuilds at KV padding boundaries and host overhead per token.
5. **Planner dense fixture and detector campaign:** `taxameter run scripts/c_session.sh --env FIXTURES=dense --env FIXTURE_MIN=70` (Qwen3-30B-A3B and Qwen3-32B only), then score as `planner-dense`; the detector campaign follows in the same script. Then write the CI test that scores `detector/obs-*.json` against `manifest.jsonl`.
6. **Builder slot-count candidates:** `scripts/planner_builder_campaign.sh` on Qwen3-30B-A3B, commit as `planner-moe-builder`.
7. **ik_llama quant types** (IQ4_KS, IQ4_K on the 8B): add `llama-quantize` and `llama-perplexity` to the ik install in `docker/box/Dockerfile`, then a 15-minute run (command in Track B's report, `scripts/quality/box_quality.sh`). ik_llama inference on the 235B: record "cannot run within 128 GB".
8. **Smaller:**
   - Quality: Q2_K underestimated about 25% on the 30B (cap it or measure pairs).
   - Quality: calibration `checks` should include knapsack-like mixes.
   - imatrix and IQ formats below 4 bits.
   - Per-expert formats need typed slots.
   - Replanner hook and plan metrics not wired into the engine.
   - `DRAIN_S`/`RELOAD_S` are guesses.
   - The small-RAM fusion-off drift check.
   - Old Q8_0-reference profile `bench/quality/qwen3-30b.quality.json` can go once the planner uses the BF16 one.

## Housekeeping for the user

- Delete by hand:
  - worktrees `../tetraeder-a-cleanup`, `../tetraeder-c-search`, `../tetraeder-c-cli` (merged; git refuses worktrees with submodules);
  - the stray paths `C:\c\Users\pierr\AppData\Local\Temp\claude\...\scratchpad\smoke\` and the empty `C:\ccbench_work`.
- taxameter 0.3.2 fixed the "Permission denied (publickey)" boxes (onstart installs the key); it held on every box since.
- The shared local cargo lock starved most builds; tracks used CI (`track-*` branches) as the compile loop. With several tracks, plan for that.
