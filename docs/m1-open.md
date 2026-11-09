# M1: open items (2026-11-03, baselines updated 2026-11-08)

State at the end of the 2026-11-03 parallel-tracks session. Everything below is on main; details in the ADRs and `bench/results/20261103/`. Boxes: none up. Box work was paused by the user mid-session; ask before renting.

## Where each phase stands

| Phase | Exit test | Status |
|---|---|---|
| 2 MoE builder, model larger than RAM | at least as fast as the best baseline, logit parity | Prefill leads every baseline (1.4x to 3x). Decode beats llama.cpp `-ot` (1.23x to 1.60x), Colibri, ik_llama (cannot run). **llama.cpp's expert cache (`--moe-cache-mib`, since #30966 also `--moe-cache-cpu-share`) not measured**, so the llama.cpp result is open (ADR 0045 addendum). **Behind KTransformers 1.34x on a 26-core box** (3.77 vs 5.04 tok/s; tie on a 15-thread box). Parity: greedy equal; logit offset explained as GPU kernel fusion (ADR 0045 addendum). |
| 3a Formats and quality | quality per byte at least llama.cpp presets | Pass: profile mixes beat Q3/Q4/Q5_K_M on Qwen3-8B and 30B-A3B (BF16 reference). Conversion 11 to 15% faster than llama-quantize, byte-identical (ADR 0047, 0048). |
| 3a Planner | regret, rank, beat llama.cpp fit | Pass on the MoE fixture (regret 0, Spearman 0.997, pick 1.44x llama.cpp `-fitt`). Dense fixture partial (11 of 24). |
| 3a Detector | top-1 faults on real hardware | Passes in the simulator only; box campaign not run. |
| 3b Gateway | SDK conformance, Claude Code use, vs llama-server | Conformance passes on real models. Claude Code proxy (user chose a few-hour proxy over a week): 15/20 tasks on both engines; decode 0.88x on an A6000; 400s and prompt cache fixed afterwards (ADR 0049), not re-measured. vLLM not run (no GGUF support worth pairing). |
| Speculative decoding (ADR 0043, Phase 4 pulled forward; not an M1 exit test) | faster than plain decode | Exact, but slower in the exit regime; op-offload fix (ADR 0044) in, rerun pending. |

## Open, by priority

1. **Phase 2 decode vs KTransformers.** One 91 GB-class box session, at least 11 cores: `scripts/exit_session.sh` with `VRAM_SLOT_BYTES=10966007808` and `OT_FIT='blk\.[0-6]\.ffn_.*_exps=CUDA0;exps=CPU'`. It measures the implemented, never-measured levers (prefetch depth 2 actually on, reader 1 MiB at queue depth 16, reads before the layer's own misses; expected 6 to 6.7 tok/s), then re-pairs KTransformers for a noise band. Add a llama.cpp expert-cache stage at the same expert bytes, through `llama-server` (`-cmoe --moe-cache-mib <slot bytes> --moe-cache-cpu-share S`), built from a commit after #30966 (a separate baseline copy in the box image; the engine pin stays). Pick S with `scripts/moe-cache-tune.py` on the 30B, then sweep only S in {0, 0.5, 0.9} on the 235B. Check first that llama-server streams in this regime (`--load-mode dio`) rather than thrashing mmap, and record the setting. Next lever if short: fast tiers kept by routing frequency (plan.md levers table; score an offline replay of recorded routing against the LRU first). ADR 0038 addendum has the diagnosis. **Gap diagnosis stages** (all in `scripts/exit_session.sh`, written 2026-11-09, read with the table in the ADR 0045 addendum "Diagnosing the KTransformers decode gap"): `main-native` and `main-native-repeat` (builder with ggml built for the native ISA, paired with `main` / `main-repeat`), `ktransformers-avx2` (`KT_KERNEL_CPU_VARIANT=avx2`), `kernels-cpu` (Q4_K kernels on CPU only: mainline, ik, baseline). Every builder and KT result now has a `host` block (disk bytes, queue depth, major faults, CPU and GPU busy, RSS, page cache; prefill and decode apart) and the CPU flags in `env`. Pre-box: the `expert-bench` workflow (expert matmul, native OFF against ON).
2. **Speculative rerun** (not an M1 exit test; useful, not blocking): `STAGES=main-builder bash scripts/spec_run.sh` (VRAM fix in; the draft now fits). Same box session as 1 if time allows.
3. **Claude Code proxy rerun** on a 48 GB box at most 0.50 USD/h (user exception, up to 6 h): `scripts/claude_code_proxy.sh` with `REPS=2 BUDGET_MIN=240 STREAMING=1`. Bar: tasks solved at least llama-server's, prompt-cache reuse at least llama-server's in the same run (93.6% in the first; ADR 0051 estimates 94.3% for ADR 0049 on that traffic), zero 400s. Brief G2 (`docs/gateway-briefs.md`) adds per-engine tool-call failure counts first; keep the request logs local and commit only the extracted fixtures. Rerun steps: fetch `*.reqlog.jsonl` with the results but do not commit them; run `scripts/toolcall_fixtures.py <reqlog> <out.json>` on each and commit the fixtures; read `cache_classes.txt` against ADR 0051's table; compare `tool_error_rate`, `leaked_markers_total` and `tool_names_unknown_total` per engine in `summary.json`.
4. **Resident decode gap on the A6000** (0.88x llama-server, Qwen3-Coder-30B-A3B, 128K): profile build, alloc, compute and readback per token and CUDA graph reuse at depth 0 and 20K; suspects are graph rebuilds at KV padding boundaries and host overhead per token.
5. **Planner dense fixture and detector campaign:** `taxameter run scripts/c_session.sh --env FIXTURES=dense --env FIXTURE_MIN=70` (Qwen3-30B-A3B and Qwen3-32B only), then score as `planner-dense`; the detector campaign follows in the same script. The scorer and CI test exist (`crates/analysis/src/score.rs`, `crates/analysis/tests/campaign.rs`): `cargo test -p tetraeder-analysis --test campaign` asserts the exit bars (single top-1 at least 90%, at most 1 false positive in the no-fault runs, pairs both in the top 3 at least 90%) on every committed `bench/results/*/detector/` with a `manifest.jsonl`, and skips when none is committed. To score a fresh campaign before committing: `tetraeder detector-score bench/results/<date>/detector` (writes `score.json`). Commit the campaign directory (obs, manifest, calib, harness, score.json) and CI gives the verdict.
6. **Builder slot-count candidates:** `scripts/planner_builder_campaign.sh` on Qwen3-30B-A3B, commit as `planner-moe-builder`.
7. **ik_llama quant types** (IQ4_KS, IQ4_K on the 8B): add `llama-quantize` and `llama-perplexity` to the ik install in `docker/box/Dockerfile`, then a 15-minute run (command in Track B's report, `scripts/quality/box_quality.sh`). ik_llama inference on the 235B: record "cannot run within 128 GB".
8. **Levers from Strata and Unsloth** (`docs/plan.md`, "Levers from Strata and llama.cpp's expert cache"), local work first. The first three are designed in ADR 0050 with briefs in `docs/levers-briefs.md` (the builder's CPU and GPU expert work and its own miss reads are serial today; low-bit KV is worth about 3% of decode here, its main value is draft-model room):
   - Low-bit KV in the builder (ADR 0050 b): about 3% of decode; more expert slots, room for the draft model.
   - Miss reads under the CPU hit group (ADR 0050 a): serial today, about 6% of decode.
   - Fast tiers by routing frequency (ADR 0050 c): recorder and replay first; built only if the replay beats LRU by 10% of disk bytes.
   - Together about 25 to 35 ms of the 66 ms per token the KTransformers gap needs; the rest has to come from the implemented, unmeasured ADR 0038 levers (item 1).
   - Message-boundary and near-tail checkpoints for the prefix cache: skipped (ADR 0051). Full-attention KV is already reused to the longest common prefix at any token; they pay only for recurrent or sliding-window state.
   - Tool-call repair in the gateway (ADR 0051, briefs G1 and G2 in `docs/gateway-briefs.md`): the first run shows one failure (an empty `<tool_call>` leaked as text, 1 of 483 responses). Build outcome counts and that one repair; further classes only from the rerun's numbers.
   - Recheck the llama-server and planner baselines with the expert cache on. Pair against Strata once both engines run one model.
9. **Smaller:**
   - Quality: Q2_K underestimated about 25% on the 30B (cap it or measure pairs).
   - Quality: calibration `checks` should include knapsack-like mixes.
   - imatrix and IQ formats below 4 bits.
   - Per-expert formats need typed slots.
   - Replanning in `serve --plan` (ADR 0055): part A is in; part B (op offload switch in libllama, builder prefill chunk from the plan, `Scheduler::shutdown` join) waits for `track-r` and `track-s`, and the box benchmark (switch time and refused requests, paired against restarting llama-server) waits for the hold.
   - `DRAIN_S`/`RELOAD_S` are guesses.
   - The small-RAM fusion-off drift check.
   - Old Q8_0-reference profile `bench/quality/qwen3-30b.quality.json` can go once the planner uses the BF16 one.

## Housekeeping for the user

- Delete by hand:
  - worktrees `../tetraeder-a-cleanup`, `../tetraeder-c-search`, `../tetraeder-c-cli` (merged; git refuses worktrees with submodules);
  - the stray paths `C:\c\Users\pierr\AppData\Local\Temp\claude\...\scratchpad\smoke\` and the empty `C:\ccbench_work`.
- taxameter 0.3.2 fixed the "Permission denied (publickey)" boxes (onstart installs the key); it held on every box since.
- The shared local cargo lock starved most builds; tracks used CI (`track-*` branches) as the compile loop. With several tracks, plan for that.
