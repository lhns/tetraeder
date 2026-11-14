# Tetraeder after M1: plan to M2 (2026-11-14, reviewed twice with Fable)

## Context
M1 is reached and the owner confirmed it (main `8e6382c`; `docs/m1-open.md`; ADR 0045 addendum 2026-11-14). The 2026-11-08 resume plan in this file is finished. Main CI is green, credit is $19.66, and no boxes are running.

The next milestone is M2, which ends after Phase 5. Its critical path is Phase 4: own KV and scheduler. Its exit test:
- at least llama.cpp server throughput at concurrency 4;
- prefix hits cut time to first token on a Claude Code replay.

Today the gateway reaches 0.41 to 0.61 of llama-server's throughput with 4 parallel slots (`-np 4`, ADR 0046). The scheduler and both engines run one sequence at a time.

The owner chose to keep the own dense builder and the KV-layout experiment inside Phase 4. This plan includes every item still open in `docs/m1-open.md`, each either scheduled or parked with a reason.

## Facts from the read-only checks
- **Replay (`bench/results/20261114/replay-235b-ram.json`, the real 235B trace).**
  - At 77 RAM slots with prefetch, the global policies cut disk bytes per token compared with per-layer LRU: gfreq:0.5 reads 164 MB, glru 172 MB, LRU 266 MB. Synchronous misses fall from 47 to 27 MB.
  - That replay ran at only 10 VRAM slots per layer. The point-3 setup is now about 26 slots, so this does not yet show what the global tier gains at the configuration we use (Track B, B0).
- **`serve` does not fill free VRAM by itself (stated otherwise on 2026-11-14, corrected).**
  - `vram_slot_tiers` (`crates/planner/src/builder.rs`) offers only 8, 16, 32, 64 and 128 slots on the 235B, at about 1.1 GB each.
  - On a 48 GB card the planner picks at most 16 slots. The measured winner was 26.
- **Poll 50 evidence conflicts.**
  - ADR 0061's tq4 run (19 threads on a 23 quota) stayed throttled in 82% of periods and decoded 0.80x.
  - In `20261114` and `20261114b`, poll 50 at 26 threads on a 30.7 quota was throttled 92 to 95% but still won 1.16 to 1.22x, on one host. The default stays unchanged until a paired run settles it.
- **20261114b per-window tok/s are sampler artifacts** (about 15 tokens over whole seconds). Use only the builder's window JSON.
- **The 55 MB routing trace exists only in a driver worktree:** `.claude/worktrees/agent-a0e200fe4acde8ff3/bench/results/20261114/routing-235b-041805.trace.zst`.

## Step 0: safeguards (coordinator)
1. Copy the trace to `bench/traces/` and git-ignore it, before any worktree cleanup.
2. Commit a point-3 session profile in `docs/sessions.md` / `scripts/session.sh` that fixes `EMULATE_RAM_GB=91`, `SESSION_MIN` and `.vast/config.json` `disk_gb 230`. The first driver launched at the wrong budget and only caught it by luck.

## Track C: Phase 4, the critical path (starts first)

### C0. Verdict definition (in the C1 ADR, before any code)
The M1 dispute was over what counted, so the C1 ADR records the following before any code.

**What is judged**
- The judged path is the own dense builder (C5) on Qwen3-8B.
- libllama runs alongside as a control.
- For the 30B MoE, the own MoE builder counts only after C4. Otherwise the MoE result is recorded as libllama-resident and does not count.

**Baseline:** llama-server tuned with `-np 4`, continuous batching, flash attention and the same KV type, from the post-#30966 baseline build.

**Pass criterion:** decode throughput at concurrency 4 and concurrency 1, in two clean interleaved pairs per model, with the worst pair at or above 1.0 beyond the GPU-class noise band (`tetraeder-bench noise`).

**TTFT:** median and p90 on one Claude Code request log, with and without prefix reuse, through the same gateway.

### C1. ADR: multi-sequence backend, continuous batching, attention across sequences (Opus design agent, `model: opus`)
- `Backend` (`crates/scheduler/src/backend.rs`) gets sequence ids: one batched decode step over N sequences, and `seq_rm` / `seq_cp` per sequence.
- The scheduler loop (`crates/scheduler/src/scheduler.rs`) admits and retires sequences every step, with chunked prefill interleaved with decode under a token budget.
- The design must answer how ggml attention batches over sequences of unequal length:
  - whether per-sequence contiguous KV can reach llama-server's batched throughput (masking over one shared buffer, as llama.cpp's unified KV does, against separate buffers);
  - the trait must hold either C6 layout.
- Property tests: every request finishes or fails cleanly, and no KV row is shared without a copy.

### C1.5. Extract attention, norm and KV from `qwen3moe.rs` (Sonnet, after C1)
- Move them into a shared module, so C4, C5 and C6 don't all edit the 2,848-line file.
- Done when the MoE logits are bit-equal before and after, and the parity suite is green.

### C2. Continuous batching on the libllama backend (Sonnet)
- Use libllama's multiple sequences (`crates/engine`; today it calls only `seq_rm(0, ...)`).
- This is the control path and the gateway's batching plumbing. CI smoke: `scripts/gateway_bench.py --concurrency 4` on the tiny fixture.

### C3. TTFT comparison script, then prefix reuse on sequences (Sonnet)
- First the missing comparison on `scripts/gateway_replay.py`. The current snapshot cache (`crates/scheduler/src/prefix_cache.rs`) may already pass this half of the exit test.
- Then `seq_cp`-based reuse from cached sequences.

### C5. Own Qwen3 dense builder (Sonnet, after C1.5; new ADR)
- Build on the extracted attention and KV module.
- Parity: per op and end to end against libllama on a tiny dense fixture, with `TOL_CPU` and `TOL_GPU` and `libllama_reference_at` (`crates/model/tests/common/mod.rs`), plus a greedy golden sequence.
- It implements the C1 multi-sequence `Backend`.

### C6. KV layout experiment, staged (ADR first; `crates/kvcache`)
1. Chunked contiguous KV on unmodified ggml attention, plus a CPU gather micro-benchmark for paged blocks. The ADR states the threshold before measuring: paged has to save at least X% memory under a mixed-length workload at no more than Y% decode cost.
2. Only if the CPU numbers clear the threshold: a CUDA gather patch in the patch queue (Rule 8: ADR plus backend tests), with GPU numbers on box D2.
3. The winner becomes the KV for C4 and C5, which closes the open decision in plan.md.

### C4. MoE builder multi-sequence KV (Sonnet, after C1.5 and C6 step 1)
Needed so that streamed MoE serves concurrency 4, and so the 30B can count on the own builder.

### Phase 4 deliverables without a step yet (scheduled 2026-11-14)
- Speculation wired into `serve` (ADR 0065: drafts only when one sequence decodes): after C2, one Sonnet task; measured in the D2 spec stage.
- Self-speculative decoding with a low-bit copy, and MTP where a model has it: parked until after the Phase 4 verdict. Neither is in the exit test; ADR 0043 showed drafts lose at batch 1 in the streamed regime until op offload was fixed, so the D2 rerun decides whether more drafter kinds are worth building.

### Speculative rerun
One stage on box D2 (`scripts/spec_run.sh`, the ADR 0044 fix is in). It is not an exit test.

## Track B: decode levers (parallel, off the critical path)
- **B0. Offline replay at the configuration we use (coordinator, local, no box).**
  - Rerun the replay from the saved trace at about 26 VRAM slots, for policies lru, freq, glru and gfreq, plus the hot-set refresh, and compare synchronous miss bytes over the first four 16-token windows.
  - The result decides B1 against the never-run `REFRESH_HOT` arm.
- **B0 result (2026-11-14, ADR 0050 addendum): B1 is not built.** At 26 VRAM slots the best pool across layers reads 0.54 (all tokens) and 0.63 (first 64) of the best per-layer policy, short of ADR 0050's halving threshold; about 2 ms per token of synchronous reads is left. The VRAM hot-set arm (`main-overlap-poll0-hot`) goes on box D1 instead.
- **B1 (not built). Global RAM tier, only if B0 shows at least 10% fewer synchronous misses at 26 slots.**
  - The coordinator writes the ADR 0050 (c) addendum; it is design work. A global pool changes the planner's RAM budget, the plan IR and the "memory never exceeded" property.
  - Then Sonnet ports `GlobalSlots` (`crates/memory/src/slots.rs`) and the gfreq and glru policies (`crates/memory/src/replay/sim.rs`).
  - Tests: greedy parity against LRU; engine disk bytes equal the replay on a synthetic trace; `crates/memory/tests/global_slot_props.rs`.
- **B2. VRAM fill (Sonnet).**
  - Add a largest-fitting tier in `vram_slot_tiers` / `feasible_vram_tiers_kv`, with a property test that the budget is never exceeded.
  - In `scripts/exit_run.sh`, derive `VRAMFILL` from measured free VRAM minus a reserve, instead of the fixed 18 GB, which would OOM a 24 GB card. Make it the KT-pairing default; the llama.cpp pairing stays at 10.97 GB.
- **B3. Polling rule: measure before changing.**
  - Add a stage pair, poll 0 against poll 50 at quota minus 4 threads, to box D1.
  - The ADR 0061 addendum and the rule change follow only if two clean pairs agree.
- **B4. Smaller levers (one Sonnet worker, after B2):**
  - ADR 0064 step 2 (one graph per token, judged by `segment_overhead`);
  - tool-call repair G1 and G2 (`docs/gateway-briefs.md`);
  - planner slot-count campaign script (`scripts/planner_builder_campaign.sh`, m1-open item 6);
  - `serve --plan` replanning part B (ADR 0055).

## Open items from `docs/m1-open.md`, each placed
| Item | Where |
|---|---|
| 1 Speculative rerun | box D2 |
| 2 Read wait under overlap: the `-uf-w4`, `-w8` and `-poll0` arms, then delete `Reader` | box D1 stages; `Reader` deletion in B4 after D1 |
| 6 Planner slot-count candidates | B4 |
| 8 Low-bit KV (b) | done in tranche A; its value returns with C6 (KV room) |
| 8 Miss reads under the CPU hit group (a) | done (ADR 0054 overlap) |
| 8 Frequency tiers (c) | B0, then B1 |
| 8 Tool-call repair | B4 |
| 8 Recheck llama-server and planner baselines with the expert cache | box D2 (llama-server baseline for C0) |
| 8 Strata pairing | parked: needs both engines on one model; revisit when Strata runs Qwen3 |
| 9 FP8-KV llama.cpp baseline | pin already includes #28392 (b12274, `f8_e4m3`, 2026-11-14); still owed: rerun `llamacpp-cache` FP8 against the builder on box D1, which checks whether point 1 holds against the newer baseline |
| 10 imatrix and IQ formats below 4 bits | parked to Phase 9 (breadth); no exit test needs them |
| 10 Per-expert formats (typed slots) | parked until C6 decides the KV/slot layout |
| 10 `serve --plan` part B | B4; its box benchmark on D2 |
| 10 `DRAIN_S` / `RELOAD_S` guesses | measured on D2 with part B |
| 10 Small-RAM fusion-off drift check | CI, one Sonnet task in B4 |

## Track A and E: close-out and housekeeping (coordinator, local)
- **m1-open:**
  - point-3 scope: one host, 48 GB card; a 24 GB card ties;
  - remove the stale "M1 status (2026-11-11)" section and the sentence "Phase 2 does not until point 1";
  - replace "Open after M1" with a pointer to this plan's tracks.
- **plan.md:** Effort status line updated (M1 on 2026-11-14).
- **My earlier VRAM statement:** corrected in the ADR 0045 addendum.
- **Branches:** check diffs of `track-adaptive-vram`, `track-poll-bench-cpu` and `track-vram-policy-replay`, then merge or close each.
- **Worktrees:** remove merged, clean ones, after Step 0.
- **Size cap on fetched otel and samples:** about 30 MB per session today. A taxameter fetch filter or a commit-time cap.
- **ADRs:** of the 21 marked Proposed, mark the measured ones Accepted.

## Order
1. Step 0. At the same time, start the C1 and C0 design (Opus agent) and B0 (coordinator).
2. While C1 is being designed: Track A/E (coordinator); B2 (Sonnet); the FP8 baseline-pin move (Sonnet).
3. After the C1 ADR: C1.5. Then C2, C3 and C5 in parallel; C6 step 1 once C5 builds. B1 if B0 passes; B4.
4. **Box D1** (48 GB AVX-512, about $2), after B2 (and B1 if built):
   - vramfill against KT twice;
   - the poll 0/50 pair;
   - the ADR 0054 arms;
   - FP8 `llamacpp-cache` against the builder;
   - the planner's own fill tier, last (it may OOM): 35 slots (38.4e9 B) by its estimate on a 49,140 MiB (51.5e9 B) card. Measured besides slots: 13.9e9 B at the exit run's ubatch 2048 (40,852 MiB peak with 28.97e9 B of slots), which matches the planner formula at ubatch 2048, so 35 slots fit only at ubatch 256 (estimate 50.5e9 B); candidates are fit-checked per ubatch (`builder.rs`), so serve would pick 35 slots at ubatch 256. Record its peak and prefill rate (the harness's 22.5 GB reserve gives 26 slots at any ubatch);
   - trace.
   - Host 60400 or a second qualifying host, to widen point 3's scope.
5. C4 on the C6 layout.
6. **Box D2** (3090 class, about $3 to $4), after C2, C3, C4 and C5:
   - the C0 verdict pairs on the 8B (own dense builder and libllama) and the 30B (own MoE builder), at concurrency 1 and 4 against tuned llama-server;
   - the TTFT comparison;
   - C6 GPU numbers if step 2 ran;
   - the spec rerun;
   - `serve --plan` part B.

**Budget:** "at most two boxes" probably won't hold. Phase 4 likely needs a D3 confirmation box (about $3). The total is about $8 to $10 of the $19.66.

The coordinator reviews every diff, consult Fable at C1 and before each box, and drivers pick machines by `docs/box-driver.md` without asking the owner.

## Verification
- **Every change:** `cargo build/test/clippy -D warnings/fmt --check --workspace` on Windows and Linux CI; `tests/test_stage_names.py` for stage lists; no em dashes in docs.
- **C1.5:** MoE logits bit-equal before and after.
- **C5:** parity per op and end to end on CPU and GPU, plus the greedy golden sequence.
- **C1 to C3:** scheduler property tests, then a concurrency-4 CI smoke on the tiny fixture.
- **C6:** micro-benchmark JSON committed; thresholds stated in the ADR before measuring.
- **B1 and B2:** greedy parity; engine bytes equal the replay; the planner never exceeds its budget.
- **Boxes:** paired same-box runs in `bench/results/<date>/` (no traces over a few MB). Verdicts go in ADR 0045, ADR 0046 and a new Phase 4 verdict ADR judged by C0's rules.
