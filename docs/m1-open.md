# M1: status and open items (2026-11-12: M1 not reached. Phase 2 against the llama.cpp expert cache is not met at the 23 to 26 core profile (point 1); the AVX-512 margin over KTransformers is open (point 3); points 2 and 4 decided by the owner)

Session profiles for later box runs: [`docs/sessions.md`](sessions.md) (`scripts/session.sh`).

State on 2026-11-10 after five box sessions (`bench/results/20261110*/`). Everything below is on main. Boxes: none up.

## Where each phase stands

| Phase | Exit test | Status |
|---|---|---|
| 2 MoE builder, model larger than RAM | at least as fast as the best baseline, logit parity | **Not met against the llama.cpp cache at 23 to 26 cores (point 1 below, `20261112a`); met against KTransformers on AVX2, AVX-512 margin within its spread (point 3).** Earlier verdict, superseded: AVX-512 box (b1-exit, `20261110/`): prefill 1.83x KTransformers and 6.4x the llama.cpp expert cache (dio OOMs at 91 GB, mmap is its best), decode 1.13x KT on the means and 1.57x the cache. AVX2 closing run (`20261110f`, 30 threads): overlap decodes 4.6x the llama.cpp cache at share 0 and 1.66x at share 0.9, prefill 2.8x. AVX2 box with CPU matched (b1c, `20261110e/`, 30 threads = floor(quota)): ADR 0054 overlap (default in `serve`) decodes 1.14x and 1.18x KT, prefill 2.3x. `-ot` beaten earlier (1.23x to 1.60x); ik_llama cannot run the 235B in 128 GB. Parity: greedy equal; logit offset explained as GPU kernel fusion. |
| 3a Formats and quality | quality per byte at least llama.cpp presets | Pass: profile mixes beat Q3/Q4/Q5_K_M on Qwen3-8B and 30B-A3B (BF16 reference). Conversion 11 to 15% faster than llama-quantize, byte-identical (ADR 0047, 0048). ik_llama.cpp types on the 8B (2026-11-10, `20261110/quality-qwen3-8b-ik.json`, ik's own perplexity tool and BF16 reference): the mix at the same or fewer bytes has KL 0.0448 vs IQ4_KS 0.0614 and 0.0384 vs IQ4_K 0.0449, top-1 higher in both. |
| 3a Planner | regret, rank, beat llama.cpp fit | Pass: MoE fixture (regret 0, Spearman 0.997, pick 1.44x llama.cpp `-fitt`) and dense fixture (2026-11-10, b3-planner: 24 candidates, regret 0, Spearman 0.997). |
| 3a Detector | top-1 faults on real hardware | Pass on a box (2026-11-10, b3-detector): single 30/30, false positives 0/10, pairs 9/9. |
| 3b Gateway | SDK conformance, Claude Code use, vs llama-server | **Met (2026-11-10, box 60067423, A40, `bench/results/20261110d/`, ADR 0046 addendum):** tasks 15/20 vs 14/20, prompt cache 95.8% vs 94.2%, 0 non-200s; server decode 0.992. Conformance passes. Low-bit KV passes kv-quality under the excess-over-floor gate (ADR 0050 note 2026-11-10b), all six gating rows; q8/q4 top-1 loss 0.027 at 8K is near its 0.03 bound. |
| Speculative decoding (ADR 0043, Phase 4 pulled forward; not an M1 exit test) | faster than plain decode | Exact, but slower in the exit regime; op-offload fix (ADR 0044) in, rerun pending. |

## M1 status (2026-11-11)

Points before M1 is declared reached: (1) not met: in the only repeated, interleaved pairing (`20261112a`, 24 threads on a 23.04 quota, EPYC 7642) the llama.cpp cache at its best share (0.95) decodes 1.35x to 1.54x faster than the builder (builder spread 13%, cache 1%; builder CPU-bound, 406 and 465 ms compute per token). Per box against the best share run there: 4.6x at share 0 and 1.66x at 0.9 (`20261110f`, 30 threads); 2.3x at 0.5 (`20261111c`, 46 threads); 1.07x to 1.17x at 0.5 (`20261111g`, 23 threads); 1.34x at 0.95 on one pass (`20261111h`, 23 threads, host 130210). The builder's CPU compute per token varies 1.9x across sessions with nothing recorded explaining it. The adaptive VRAM tier (ADR 0059) raised the VRAM pair share to 0.37 but lost 0.52x to 0.61x to its synchronous uploads, with a ceiling of 1.07x even without them (`20261112c`, quota 30.7, outside the profile; its cache run thrashed mmap and gives no ratio). `20261112e` (quota 23, EPYC 7642): `cpu_poll` 0 cut cgroup throttling from 0.80 to 0.98 of periods to 0.05 and decoded 1.30x the default (3.03 tok/s, one-sided pair; the cache's best on that CPU class was 2.72 on `20261112a`); fewer threads and huge pages did not help. `20261112f` tried that pairing but is invalid (the RAM balloon died, the cache ran resident); it showed the host's memory bandwidth moving 2x between stages (42 to 28 GB/s), which fits the builder's spread. Next: the pairing again with the balloon guarded, both engines' throttling and memory bandwidth recorded per stage. Earlier leads: cgroup throttling (27% of each second throttled at threads = quota on `20261112c`), memory-bound CPU expert work (no huge pages in use), threads below the quota and `cpu_poll` 0, paired on an EPYC 7642-class box at 23 to 26 cores with the llama.cpp cache in mmap and dio; (2) Phase 3b tasks: decided by the owner (ADR 0046 addendum 2026-11-11): met as no worse than llama-server beyond run-to-run spread, pooled 73/100 against 74/100 over five runs, every run within one task; (3) on AVX-512 the margin over KTransformers (1.04 to 1.12) is within its 7.8% run spread; (4) slow disks: decided by the owner (ADR 0045 addendum 2026-11-11): they bound the scope, not the result; Phase 2 is met where read waits take at most 18% of the builder's decode step, and at 0.80 (0.5 to 0.6 GB/s disks) KTransformers decodes 1.6x faster (known limitation, post-prefill warm-up, follow-up after M1).

Every other M1 exit test in `docs/plan.md` (Phases 0 to 3b) has a passing committed paired same-box run; Phase 2 does not until point 1 is met. The closing run (`20261110f`, box 60083507) is in. Deviations that remain:

- **Phase 2, AVX-512 (`20261111a`, box 60151711, A10, Xeon Gold 6342 with AVX-512 VNNI, 30 threads both engines):** overlap decodes 4.29 and 4.27 tok/s, KTransformers (its avx512_vbmi kernels) 3.82 and 4.12. Adjacent pairs 1.12 and 1.04, cross pairs 1.04 and 1.12, means 1.08; overlap pair 0.4% apart, KT pair 7.8%. Prefill 1.11 and 1.21. Greedy tokens equal across all builder runs; overlap and `cpu_experts` GPU parity tests pass. A margin of 4% in the worst pair is within KT's own spread, so on AVX-512 the builder is at least as fast, not clearly faster. Earlier attempt `20261110h` (slow disk) gave no KT number.
- **Phase 2, llama.cpp expert cache:** the exit text says "as its tuner picks them". The tuner returned no CPU share (all 30B experts fit in 24 GB), so the 235B ran shares 0 and 0.9 at the builder's slot size (10458 MiB), mmap (dio OOM-killed at 91 GB; major faults 0.26M and 0.41M per run, no thrash). Overlap decodes 4.6x share 0 and 1.66x share 0.9 (`20261110f`, three overlap runs 1.4% apart, greedy tokens equal). Not run again: `-ot` on the new baseline build (beaten 1.23x to 1.60x on the old one).
- The overlap GPU parity tests (`gpu_overlap_logits_equal_serial`, `cpu_experts` gpu tests) pass on the box (`20261110f/gpu-tests-150632.log`).
- Phase 3b is not affected by the new `serve` defaults: the proxy sessions serve the model resident (no RAM tier), where the I/O service, overlap and prefetch stay off, as in the 20261110d verdict run.

## Open after M1, by priority

1. **Speculative rerun** (not an M1 exit test): `STAGES=main-builder bash scripts/spec_run.sh` (VRAM fix in; the draft now fits).
2. **Read wait under overlap** (ADR 0054 addenda 2026-11-11): the 0.5 hidden-share bar is misleading (`urgent_first` raised it to 0.52 while decode fell 12%). Next: judge levers on exposed wait and tok/s (`-uf-w4`, `-w8`, `-poll0` not yet run), then delete the `Reader` path.
3. **Noise bands (done 2026-11-11):** `tetraeder-bench noise update` picked up the A/A pairs of `20261110c` and `20261110e` (rtx-3090: n=11 sessions, 6 hosts, p95 d 0.060, band 0.075) and `20261110d` (a40: n=1, band 0.020, below the 3-session minimum, provisional stays); the overlap runs of `20261110f` are builder runs without a `noise/` record, so they carry no A/A pair.
4. (done) `rc=2` after `EXIT_RUN_DONE rc=0` came from the closing `sccache --stop-server`; `exit_run.sh` now exits 0.
5. (done) **Proxy harness:** each request-log line carries `tool_outcomes` ({"outcome/class": n}); the proxy summary sums them per engine as `toolcall_outcomes`.

6. **Builder slot-count candidates:** `scripts/planner_builder_campaign.sh` on Qwen3-30B-A3B, commit as `planner-moe-builder`.
7. (done) **ik_llama quant types:** the profile mix beats IQ4_KS and IQ4_K at matched bytes (Phase 3a row). ik_llama inference on the 235B: record "cannot run within 128 GB".
8. **Levers from Strata and Unsloth** (`docs/plan.md`, "Levers from Strata and llama.cpp's expert cache"), local work first. The first three are designed in ADR 0050 with briefs in `docs/levers-briefs.md` (the builder's CPU and GPU expert work and its own miss reads are serial today; low-bit KV is worth about 3% of decode here, its main value is draft-model room):
   - Low-bit KV in the builder (ADR 0050 b): about 3% of decode; more expert slots, room for the draft model.
   - Miss reads under the CPU hit group (ADR 0050 a): serial today, about 6% of decode.
   - Fast tiers by routing frequency (ADR 0050 c): recorder and replay first; built only if the replay beats LRU by 10% of disk bytes.
   - Together about 25 to 35 ms of the 66 ms per token the KTransformers gap needs; the rest has to come from the implemented, unmeasured ADR 0038 levers (item 1).
   - Message-boundary and near-tail checkpoints for the prefix cache: skipped (ADR 0051). Full-attention KV is already reused to the longest common prefix at any token; they pay only for recurrent or sliding-window state.
   - Tool-call repair in the gateway (ADR 0051, briefs G1 and G2 in `docs/gateway-briefs.md`): the first run shows one failure (an empty `<tool_call>` leaked as text, 1 of 483 responses). Build outcome counts and that one repair; further classes only from the rerun's numbers.
   - Recheck the llama-server and planner baselines with the expert cache on. Pair against Strata once both engines run one model.
9. **llama.cpp FP8 KV cache** (llama.cpp #28392, merged 2026-11-10, after the baseline pin): at 32K on the 235B the KV cache is about 6 GB in f16, about 3 GB in FP8, so about 3 GB more VRAM for the expert cache. Move `LLAMA_BASELINE_COMMIT` past it and rerun `llamacpp-cache` with an FP8 KV cache and the freed VRAM in `--moe-cache-mib`, paired with the builder. KTransformers already runs `fp8_e5m2` KV. Ampere has no FP8 math, so measure whether the conversion costs decode. Also a candidate format for the builder's low-bit KV (ADR 0050 b).
10. **Smaller:**
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
