# M1: status and open items (2026-11-10)

Session profiles for later box runs: [`docs/sessions.md`](sessions.md) (`scripts/session.sh`).

State on 2026-11-10 after five box sessions (`bench/results/20261110*/`). Everything below is on main. Boxes: none up.

## Where each phase stands

| Phase | Exit test | Status |
|---|---|---|
| 2 MoE builder, model larger than RAM | at least as fast as the best baseline, logit parity | **Met on AVX2 at `serve` defaults; AVX-512 overlap unmeasured (see M1 status).** AVX-512 box (b1-exit, `20261110/`): prefill 1.83x KTransformers and 6.4x the llama.cpp expert cache (dio OOMs at 91 GB, mmap is its best), decode 1.13x KT on the means and 1.57x the cache. AVX2 closing run (`20261110f`, 30 threads): overlap decodes 4.6x the llama.cpp cache at share 0 and 1.66x at share 0.9, prefill 2.8x. AVX2 box with CPU matched (b1c, `20261110e/`, 30 threads = floor(quota)): ADR 0054 overlap (default in `serve`) decodes 1.14x and 1.18x KT, prefill 2.3x. `-ot` beaten earlier (1.23x to 1.60x); ik_llama cannot run the 235B in 128 GB. Parity: greedy equal; logit offset explained as GPU kernel fusion. |
| 3a Formats and quality | quality per byte at least llama.cpp presets | Pass: profile mixes beat Q3/Q4/Q5_K_M on Qwen3-8B and 30B-A3B (BF16 reference). Conversion 11 to 15% faster than llama-quantize, byte-identical (ADR 0047, 0048). |
| 3a Planner | regret, rank, beat llama.cpp fit | Pass: MoE fixture (regret 0, Spearman 0.997, pick 1.44x llama.cpp `-fitt`) and dense fixture (2026-11-10, b3-planner: 24 candidates, regret 0, Spearman 0.997). |
| 3a Detector | top-1 faults on real hardware | Pass on a box (2026-11-10, b3-detector): single 30/30, false positives 0/10, pairs 9/9. |
| 3b Gateway | SDK conformance, Claude Code use, vs llama-server | **Met (2026-11-10, box 60067423, A40, `bench/results/20261110d/`, ADR 0046 addendum):** tasks 15/20 vs 14/20, prompt cache 95.8% vs 94.2%, 0 non-200s; server decode 0.992. Conformance passes. Low-bit KV passes kv-quality under the excess-over-floor gate (ADR 0050 note 2026-11-10b), all six gating rows; q8/q4 top-1 loss 0.027 at 8K is near its 0.03 bound. |
| Speculative decoding (ADR 0043, Phase 4 pulled forward; not an M1 exit test) | faster than plain decode | Exact, but slower in the exit regime; op-offload fix (ADR 0044) in, rerun pending. |

## M1 status (2026-11-10)

Every M1 exit test in `docs/plan.md` (Phases 0 to 3b) has a passing committed paired same-box run; the table above names them. The closing run (`20261110f`, box 60083507) is in. Deviations that remain:

- **Phase 2, AVX-512:** no RTX 3090 offer on Vast had an AVX-512 CPU at the time, so the closing run ran on AVX2 (EPYC 7742, the b1c host). On AVX-512 the prefetch path ties KTransformers within noise (b1-exit, 1.13x on the means). A later attempt (`20261110h`, box 60101282, Xeon Gold 6138, 38 threads) gave no KTransformers number: its server hung at CUDA graph capture for 40 min on a disk-starved host (about 400 to 860 MB/s in use). It did measure overlap/prefetch on AVX-512: decode 1.27, prefill 1.03, greedy tokens equal, overlap GPU parity tests pass. Overlap against KTransformers on AVX-512 stays unmeasured; a retry needs a host with at least 3 GB/s disk under load.
- **Phase 2, llama.cpp expert cache:** the exit text says "as its tuner picks them". The tuner returned no CPU share (all 30B experts fit in 24 GB), so the 235B ran shares 0 and 0.9 at the builder's slot size (10458 MiB), mmap (dio OOM-killed at 91 GB; major faults 0.26M and 0.41M per run, no thrash). Overlap decodes 4.6x share 0 and 1.66x share 0.9 (`20261110f`, three overlap runs 1.4% apart, greedy tokens equal). Not run again: `-ot` on the new baseline build (beaten 1.23x to 1.60x on the old one).
- The overlap GPU parity tests (`gpu_overlap_logits_equal_serial`, `cpu_experts` gpu tests) pass on the box (`20261110f/gpu-tests-150632.log`).
- Phase 3b is not affected by the new `serve` defaults: the proxy sessions serve the model resident (no RAM tier), where the I/O service, overlap and prefetch stay off, as in the 20261110d verdict run.

## Open after M1, by priority

1. **Speculative rerun** (not an M1 exit test): `STAGES=main-builder bash scripts/spec_run.sh` (VRAM fix in; the draft now fits).
2. **Hidden share under overlap** (ADR 0054): 0.38 to 0.42 against the 0.5 bar; read wait is all urgent misses. Then delete the `Reader` path.
3. **Noise bands:** run `tetraeder-bench noise update` for the A/A pairs in `20261110c`, `20261110d`, `20261110e` and the three overlap runs in `20261110f`.
4. (done) `rc=2` after `EXIT_RUN_DONE rc=0` came from the closing `sccache --stop-server`; `exit_run.sh` now exits 0.
5. **Proxy harness:** save the gateway's tool-call outcome counts (`repaired/no_opener`) at the end of a run; today they are live metrics only.

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
   - imatrix and IQ formats below 4 bits.
   - Per-expert formats need typed slots.
   - Replanning in `serve --plan` (ADR 0055): part A is in; part B (op offload switch in libllama, builder prefill chunk from the plan, `Scheduler::shutdown` join) waits for `track-r` and `track-s`, and the box benchmark (switch time and refused requests, paired against restarting llama-server) waits for the hold.
   - `DRAIN_S`/`RELOAD_S` are guesses.
   - The small-RAM fusion-off drift check.

## Housekeeping for the user

- Delete by hand:
  - worktrees `../tetraeder-a-cleanup`, `../tetraeder-c-search`, `../tetraeder-c-cli` (merged; git refuses worktrees with submodules);
  - the stray paths `C:\c\Users\pierr\AppData\Local\Temp\claude\...\scratchpad\smoke\` and the empty `C:\ccbench_work`.
- taxameter 0.3.2 fixed the "Permission denied (publickey)" boxes (onstart installs the key); it held on every box since.
- The shared local cargo lock starved most builds; tracks used CI (`track-*` branches) as the compile loop. With several tracks, plan for that. `scripts/dev-cargo.sh` (slot lock, per-worktree target, sccache) replaces the single lock; see [testing.md](testing.md#local-builds-for-parallel-agents). The test and benchmark setup and its roadmap are in [testing.md](testing.md).
