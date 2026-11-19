# Tetraeder after M1: plan to M2 (2026-11-14, reviewed twice with Fable)

Measurements live in the session READMEs (`bench/results/<session>/README.md`), not here; dated sections below keep their outcome, decisions and a session link.

## Context
M1 is reached and the owner confirmed it (main `8e6382c`; `docs/m1-open.md`; ADR 0045 addendum 2026-11-14). The 2026-11-08 resume plan in this file is finished. Main CI is green, credit is $19.66, and no boxes are running. (No session directory yet.)

The next milestone is M2, which ends after Phase 5. Its critical path is Phase 4: own KV and scheduler. Its exit test:
- at least llama.cpp server throughput at concurrency 4;
- prefix hits cut time to first token on a Claude Code replay.

Today the gateway reaches well under llama-server's throughput with 4 parallel slots (`-np 4`, ADR 0046, `20261103`). The scheduler and both engines run one sequence at a time.

The owner chose to keep the own dense builder and the KV-layout experiment inside Phase 4. This plan includes every item still open in `docs/m1-open.md`, each either scheduled or parked with a reason.

## Facts from the read-only checks
- **Replay (`bench/results/20261114/replay-235b-ram.json`, the real 235B trace).**
  - At 77 RAM slots with prefetch, the global policies (gfreq:0.5, glru) cut disk bytes per token and synchronous misses compared with per-layer LRU (`20261114`).
  - That replay ran at only 10 VRAM slots per layer. The point-3 setup is now about 26 slots, so this does not yet show what the global tier gains at the configuration we use (Track B, B0).
- **`serve` does not fill free VRAM by itself (stated otherwise on 2026-11-14, corrected).**
  - `vram_slot_tiers` (`crates/planner/src/builder.rs`) offers only 8, 16, 32, 64 and 128 slots on the 235B, at about 1.1 GB each.
  - On a 48 GB card the planner picks at most 16 slots; the measured winner had more, the about 26 slots we now use (`20261114`).
- **Poll 50 evidence conflicts.**
  - ADR 0061's tq4 run (19 threads on a 23 quota) stayed throttled and decoded slower (`20261112e`).
  - In `20261114` and `20261114b`, poll 50 at 26 threads on a 30.7 quota was heavily throttled but still won, on one host. The default stays unchanged until a paired run settles it.
- **20261114b per-window tok/s are sampler artifacts** (`20261114b`). Use only the builder's window JSON.
- **The 55 MB routing trace exists only in a driver worktree:** `.claude/worktrees/agent-a0e200fe4acde8ff3/bench/results/20261114/routing-235b-041805.trace.zst`.

## Step 0: safeguards (coordinator)
1. Copy the trace to `runs/traces/` (git-ignored), before any worktree cleanup.
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

### Review fixes from the C2b and C5 review (2026-11-15), one worker after C3b and C4 merge
1. A per-input `Failed` (e.g. an out-of-vocabulary token id) fails every request in the step (`scheduler.rs` step error path; `qwen3.rs` decode): validate token ids at admission, and on `Failed` retry the step without the offending input.
2. Chunked relocation asks for `need + GROW` contiguous rows (`kvcache/src/chunked.rs`): try `need` first; add a proptest over pool size and load on the dense backend that would have caught it.
3. libllama `seq_restore` cleanup is a no-op on an empty sequence (`engine.rs`, `seq_rm(seq, 0)` returns early): clear cells unconditionally after a failed restore; test with an injected failure if possible.
4. A preempted request with `toks.len() == n_ctx_seq` gets an Error instead of `Length` at re-admission (`scheduler.rs` admit).
5. libllama `decode` names `inputs[0].seq` in `KvFull` instead of the sequence that needed rows (`backend.rs`).
6. The scheduler's `reserved_to` goes stale after a dense restore or truncation (efficiency only).
7. A failed RAM restore drops the cache entry (`scheduler.rs` place): put it back.
Also: document that decode-profile `eval_s` is shared step wall time and `gpu_seconds` a token share (no gate may use per-request `gpu_seconds`).

### c4 gap on CPU (2026-11-15, `track-c4gap`, ADR 0065 addendum)
- Most of the 0.60 to 0.65 was the benchmark: c4 reused c1's prompts (llama-server's prompt cache hit, tetraeder's did not: sequences under `SAVE_MIN_TOKENS` 512), and tetraeder was built native-static against llama-server's DL variants. Both fixed in loop-bench. (No session directory yet.)
- The loop itself costs 0.04 to 0.05 ms between steps; four sequences shared almost every decode step. (No session directory yet.)
- On CPU, mixed decode-plus-prompt steps now run as separate calls (`LlamaBackend`, `DenseBackend`, `MoeBackend`).
- Result on an AMX Xeon runner (run 43668812301): `--engine llama` 1.09 (c1) and 1.01 (c4) of llama-server; `--engine dense` 0.70 and 0.56, a kernel gap of the own builder on that CPU (prompt 4.97 against 1.84 ms per token alone; hypothesis: libllama repacks for AMX, the builder does not, ADR 0039). (No session directory yet.)
- Open, decided on the GPU box: the unified pool (ADR 0065 (a)) costs on CPU against one KV stream per sequence (llama-server's `-np 4` default): 4-sequence decode step 573 against 496 ms on the Xeon, 74 against 61 on another runner. D2 runs `crates/scheduler/tests/step_shapes.rs` on the GPU first; if streams win by more than the band there, ADR 0065's layout decision is revisited before the timed pairs count. (No session directory yet.)
- Later: dense builder CPU prompt kernels (repack or AMX path); `SAVE_MIN_TOKENS` for short prompts.

### Point-1 box result (2026-11-15, `bench/results/20261115p1/`, box 60734395, host 79986, EPYC 7702, RTX 4090, quota 30.6, 24 threads)
Outcome (`20261115p1`):
- Polling: poll 50 lost against the auto default on this host, while it won on the Zen 4 host (`20261114`), so the effect depends on the host class, not on headroom alone. ADR 0061's default is not changed (B3 closed: no rule change).
- C4 on the 235B: no sign of a regression against `20261113b` (same profile, other host).
- FP8 KV baseline: llama-server at b12274 aborts writing f8_e4m3 K and V on CUDA, so the FP8 cache is not usable on CUDA at the baseline pin. Item 9 is parked until a baseline pin where CUDA writes FP8 KV; the builder's `kvq8-slots` ran but has nothing to pair with.
- Most of the box cost was model traffic: drivers must rank offers by box plus traffic (`docs/box-driver.md` rule 2).

### Third review (2026-11-15, `2c2a43f..0cf081e`): no correctness bug; follow-ups, CPU only, not blocking D2
1. MoE whole-step graph cache holds one entry, and the CPU split alternates two shapes (decoders, then a prompt chunk), so mixed CPU steps rebuild the graph each time (`qwen3moe.rs` step reuse): a two-slot cache, or measure the build cost.
2. `hot_pending` fires on the first model call with more than one token; with the split that can be a decoder group, so the CPU hot set can come from decode routing instead of a prefill.
3. `MoeBackend::restore` still shrinks and regrows; use a `Qwen3Moe::rollback` over `KvPool::rollback` as Dense does.
4. Tests: a failed step whose reservation moved the sequence, rolled back, for Dense and Moe; the busy-copy fallback reaching an idle or RAM source and `no_busy` clearing; a prompt chunk before a decoder in input order.
5. Small: a stray `no_busy` can skip busy reuse once for an unrelated request; `KvPool::blocks()` omits rolled-back reserved-empty sequences from the `reserved` metric.

### Housekeeping (2026-11-15, `track-housekeeping`)
- Old tracks: track-adaptive-vram and track-vram-policy-replay are on main in later form (obsolete); track-poll-bench-cpu's `ram_slots` input is ported. The three remote branches can be deleted (owner's call).
- m1-open item 8 tool-call repair: done (ADR 0051, 0057). Item 9 FP8 KV baseline: parked until a baseline pin where CUDA writes FP8 KV.
- ADR index carries status; 12 ADRs moved to Accepted. Still Proposed, each waiting on a box measurement: 0052 (haswell vs auto pair), 0053 (a clean `noiqk` pair), 0056 part F, 0059 async stages, 0060 sampler vs OTel agreement; candidates for a later combined box session.

### Phase 5 progress and follow-ups (2026-11-15)
- On main: E1 (ggml-sys executor bindings, NCCL on Linux), E4 (patch 0004, CUDA `mul_mat_id` skips id -1), E9 (`serve --worker`: the engine in a child process over pipes, restart with backoff, lost-sequence guard, `TetraederWorkerRestarting` alert). Running: E2, E5. Next: E3 after E2; E6, E7, E8 after E5.
- E9 follow-ups: an "Executor" dashboard panel for `tetraeder.executor.ipc.duration` and `tetraeder.executor.worker.restarts` (Rule 6); the scheduler should treat a backend's lost sequences after a worker restart as free (today a fresh request landing on a lost idle slot can fail once); sticky CUDA errors are detected by message text until the backend trait carries a sticky signal.
- E1 note: no backend-to-backend `ggml_backend_tensor_copy_async` binding yet (pipeline peer copies need it; E5 adds it if needed).
- Box sessions waiting on hardware: S1 (one GPU: patch 0004 GPU tests and timing, the E1 and E5 GPU tests), then S2 (two GPUs).

### Engine pin b12426: prepared, adoption deferred until after the Phase 4 verdict (2026-11-16)
- Branch `track-pin-b12426` (d2e8696): patches 0001 to 0003 apply unchanged; 0004 is rebased onto upstream's stream-k `tile_spec` (zeroes `tile_spec`, fills the compact-row maps, zeroes dst rows of -1 slots); no ggml-sys source change; all compile jobs green incl. CUDA.
- Not adopted now: CPU numerics changed upstream (likely #29810, CPU flash attention V in F32): goldens shift (max 0.34 on the sse42 golden, greedy tokens change on the tiny model), the replan test and speculative near-ties follow, and on Windows batched against single-row differs by 0.315 of 31.48 (1%) in `cpu_batch_invariance_*`, real drift beyond tolerance. (No session directory yet.)
- Why defer: the Phase 4 verdict runs on CUDA, where the CPU change does not matter; the baseline is already b12426, so the verdict compares against the newest llama-server either way; the builders' CUDA parity was validated on b11976 and a pin switch would reset it.
- To adopt after the verdict: per-OS goldens re-recorded with review, the Windows batch-invariance drift explained or fixed, one GPU session (CUDA parity on the new pin, 0004 GPU tests incl. MMQ/MMF with -1 above 16 tokens). The rebase-weekly workflow already works (failures were the fixed `apply.sh` permission).

### D2 split into two sessions (2026-11-16)
Time budget: from the record-log timing (5 longest Claude Code tasks, `20261111d`), each reuse-off replay of the 8B is about 13 to 20 min on a 3090 and the TTFT stage about 1 h; with build, gpu-parity, gpu-shapes, prepass, probe, both models' pairs and downloads one session would be about 2 h 45 min, over the 2 h 15 min a 3 h box leaves. So: **D2a** throughput (gates, prepass, probe, pairs for the 8B dense and the 30B MoE), **D2b** TTFT on the 8B (record-log, the reuse on/off pairs, the informative llama-server run). Both on RTX 3090 (baseline built for arch 86; the noise file has 11 sessions), `--min-inet 1000`, RAM checked at SSH, ranked by box plus about 30 GB (D2a) or 10 GB (D2b) of traffic; at SSH `/opt/llama.cpp-baseline` must report fc504f91 or the box is destroyed. Before D2a: the harness gaps (`track-d2-gaps`: rehearsal against the b12426 server, a moe stand-in, batched real-model parity, per-model stop rule). Note for reading the results: the baseline has CUDA MoE kernel work (#32067) the b11976 engine lacks, and gpu-shapes compares against the pinned libllama, not the baseline; if the 30B probe is low while the 8B passes, the deferred pin is the first suspect.

### D2a rerun result and the c4 gap (2026-11-16, `bench/results/20261116d2a2/`)
- Parity closed on GPU for both models; c4 misses on both (8B probe, 30B pairs), c1 level (`20261116d2a2`, ADR 0065 addendum of the same date).
- The dense "slowdown between boxes" was the first box, not the code: on one 3090 dense and libllama are level per step at both 030d1f3 and HEAD (`20261116d2a2`, `dense-regress/table.md`).
- Next: `track-c4-loop` (our loop at c4, CPU) and `track-pin-ab` (pin against baseline on GPU). If the pin explains the gap, adopting b12426 moves onto the critical path ahead of the verdict, reversing the deferral above; the baseline stays at b12426.

### Pin A/B: the c4 gap has two parts (2026-11-16, `bench/results/20261116pinab/`, box 60924954, RTX 3090)
Outcome (`20261116pinab`):
- The pin: llama-server built from b11976 is slower than b12426 at c4 (level at c1), and so is libllama's 4-sequence decode step. **Pin adoption moves onto the critical path ahead of the verdict** (reverses the deferral above; `track-pin-adopt`), then one GPU parity session on the new pin.
- Our loop, against llama-server at the same pin: the 8B still loses at c4, the 30B wins. The 30B step log shows host time before every admitting step (the prefix-cache save of the reused idle sequence) and one prompt admitted per step under the 512 budget, where llama-server admits a wave in one step. Fixes on `track-c4-loop`: no save at admission while KV room exists, step budget 2048.
- Then D2a again on both models.

### CUDA confirmation on b12426 (2026-11-17, `bench/results/20261116pc/`, box 60983677, RTX 3090, driver 580)
Outcome (`20261116pc`):
- Holds: real-model parity (8B and 30B, anchored gaps inside the bounds), patch 0004 GPU tests incl. id -1 above 16 tokens, gpu_dense, executor gpu tests. Step shapes: dense close to libllama.
- Probe against llama-server b12426: 8B level at c1 and c4; 30B level at c1, c4 one round under and one over the pass line 0.925.
- Broken: (1) the dense engine aborts with a CUDA illegal memory access at a fixed point of the c4 workload, right after several admissions in one pass, i.e. the first step carrying several prompts under the 2048 budget (`track-dense-crash`); (2) `gpu_batched_matches_the_cpu_builder` (MoE fixture) fails its bound at the new pin (`track-gpu-multiseq`).
- The remaining c4 loss is the admission save per reused idle sequence, serialized when several are admitted together (`track-fast-save`). Dense pays on decode with resident idle sequences, libllama nothing; a follow-up for the dense attention span.
- Next box after these three: GPU confirmation of the fixes, then D2a on both models (pairs only for a model whose probe c4 reaches 0.925).

### D2a throughput passes, provisional (2026-11-17, `bench/results/20261117d/`, box 61046208, RTX 3090, driver 595)
- Confirmation first (`20261117c`, box 61042290): the dense crash workload is clean after the mask padding; the gpu_multi_seq shift is one router near-tie (ADR 0071). The pinned-staging save alone did not get faster: the save is bound by page faults on the fresh snapshot (ADR 0074).
- D2a at `99988e3` (`track-save-thp`, huge pages on the snapshot; not on main), `PROBE_MIN=0.925`: probes and pairs pass on both models, all pairs clean; `phase4_pass` null (no TTFT in D2a). The libllama control moved with the own builder, so most of the gain is the pin and the step budget (`20261117d`).
- Not the verdict of record: the 8B server probe on the same box stalled admission saves with huge-page advice, while the pairs show no stall (`20261117d`). The verdict of record is a full `phase4_box.sh` session (pairs and TTFT) on the commit that merges, once the save stays off the admission path (`track-save-pool`).

### D2a verdict of record (2026-11-17, `bench/results/20261117k/`, box 61097370)
- Both models pass on main 6b07795 (line 0.925; `20261117k`). The way there: huge-page advice on the save path stalled admissions (`20261117d`, `h`), so the pool prepares plain spares off the admission path; the dense backend now reports its row size; the throttle rule judges BoardLimit and Reliability by a 1 Hz sampler under load (ADR 0065); `20261117e` and `g` stay null.
- D2b (`20261117l`): prefix reuse cuts the 8B's median and p90 TTFT on the Claude Code replay well below reuse off. Phase 4's exit test passes (ADR 0065 addendum); Phase 5 is next.

### Speculative rerun
One stage on box D2 (`scripts/spec_run.sh`, the ADR 0044 fix is in). It is not an exit test.
- Candidate pair (owner, 2026-11-18): `nvidia/Qwen3.5-4B-Qwen3.8-27B-Draft`, a full 4.2B draft model sharing Qwen3.8-27B's tokenizer (248,320 tokens); its card claims 5.58 tokens accepted per target step on SPEED-Bench (block 16) against 3.81 for the target's MTP head. BF16 safetensors only, NVIDIA Open Model Agreement (fine for benchmarks). Both at about 4 bits fit one RTX 3090 (about 16 plus 2.5 GB). (No session directory yet.)
- First check, before any box: whether libllama at the pin (b12426) converts and loads Qwen3.8 and the draft (architecture, likely hybrid attention, which our own Qwen3 builders do not cover), and that the converted GGUFs keep identical tokenizers. Then a libllama-engine stage: draft on and off, same prompts, tokens per step and tg tok/s.

## Phase 5: where it stands (2026-11-19)

Phase 4 is done (exit test passed 2026-11-17, ADR 0065). Phase 5's exit test (ADR 0068): on a 2-GPU and a 4-GPU box, for the dense 32B (cases D2, D4) and the MoE 30B (M2, M4), the planner's analytic pick is in the measured winner set, every candidate's prediction is within 15%, and the pick is at least as fast as the best llama.cpp split mode within the band. Measurements are in the session READMEs linked here.

**Done since 2026-11-15**
- Multi-GPU boxes: taxameter 0.3.3 `--num-gpus`; power-limit, idle and co-tenant gates at SSH; a 1 Hz GPU sampler judges throttle per run (ADR 0065, 0068).
- Executor correctness on CUDA: ranges are graph views so CUDA fuses in them (ADR 0075); GPU parity anchored on libllama's own gaps for tensor and expert parallel (ADR 0071); sharded MoE row copies fixed. All `gpu2_` and `gpu4_` gates pass (`s2c-20261118-step0`, `s3-20261119`).
- Step-0 check (ADR 0069): own tp-2 at or above llama.cpp `-sm tensor` on both models (`s2f-20261118-step0`); no executor redesign.
- Cost model version 2 (ADR 0070 History): separate expert rate, head on output rows, single on the ranks path, all-reduce latency share, pipelined prefill. All seven 2-GPU candidates within 15% out of sample (`s2g-20261118`). Owner decision 2026-11-18: predictions stay analytic, made before the candidates run (ADR 0068 History; ADR 0076 not adopted), with reopen criteria fixed.
- Pipelined layer-N prefill, GPipe micro-batches (ADR 0077), its issue order fixed to llama.cpp's per-ubatch order, and layer splits balanced for the pipelined prefill step (ADR 0070 History, 2026-11-19).
- Also on the way: prefix-cache saves off the admission path and no saves nobody reuses (ADR 0074), the long-prompt KV fragmentation fix with a 400 for real overflows (ADR 0067), ADRs hold decisions only with measurements in session READMEs (AGENTS.md).

**Open, in order**
1. 4-GPU rehearsal `s3c-20261119` (two rounds, recorded routing, tp-4 anchors, pp-overlap stage): both picks right, M4 meets the baseline gate (0.943 against `-sm layer`) with both decode toggles on, which met their keep rule and become the default (`track-decode-toggles-default`). Open:
   - Predictions still over 15%: D4 tp2-pp2 pp (+0.45), M4 ep-4 pp (+0.19), M4 tp2-pp2 pp (+0.28), M4 layer-4 tg (+0.17, new with the toggles). Micro-batched prefill closes only part of the tp2-pp2 gap. Diagnosis on `track-costmodel-s3c`.
   - D4 lacks clean rounds: GPU 0 of host 20210 shows 0x20 on the 32B in every session; verdict sessions avoid that host.
2. Expert-parallel prefill is over-predicted (ep-2 and ep-4); a cost model term to find, then a check.
3. Thermal 0x20 samples void runs (one sample voids a run, ADR 0068); size `RERUNS` and `BUDGET_MIN` from the rehearsals, or one model per session.
4. Verdict sessions V2 (2-GPU, D2 and M2) and V4a, V4b (4-GPU, D4 and M4), on a clean host at default power, two clean rounds per configuration.
5. Before reporting M2: one D2a probe on final main (the admission path changed after the Phase 4 verdict commit), and the list of verdict-rule changes made after failed sessions, for the owner.

**TTFT on the Claude Code replay (ADR 0074 bar: median and p90 at most 1.15x llama-server, saves at most 10% of TTFT):** median now at or below llama-server; p90 and save share still over the bar in `ttft3-20261118`; with the free-slot placement and pinned restores (`ttft4-20261119`) median and save share meet it, p90 does not (1.50x, 2.55x). The worst p90 outliers are avoidable saves and restores on the admission path (diagnosis in that README), fixed by ADR 0074's 2026-11-19 section after ttft4 (`track-ttft-outliers`); by hand that leaves p90 at about 1.27x, held by prefill speed at long context. Next: the replay on that branch, then the long-context prefill gap.

**Parked:** D1 (48 GB box) and Track B (decode levers; first item when it resumes: streaming prefill misses, see below). Speculative rerun with the Qwen3.8-27B draft pair: open.

## Track B: decode levers (parallel, off the critical path)
- Parked, candidate first item when Track B resumes (2026-11-19, `docs/notes/inferssd-study.md`): the 235B 32K prefill of `20261113b` mostly waits on synchronous expert misses read on the main thread far below the disk's measured curve; stream prefill misses through the I/O service and compute experts as they land (ADR 0038's named follow-up, amends 0054), after one timed box run of the prefill buckets. InferSSD as an expert-streaming baseline: its published comparisons use llama.cpp without the expert cache on models 4x to 60x larger than RAM, so they do not compare to our verdict baseline; a reported, not gated, run is possible only if it supports a model both engines run on Linux CUDA.
- **B0. Offline replay at the configuration we use (coordinator, local, no box).**
  - Rerun the replay from the saved trace at about 26 VRAM slots, for policies lru, freq, glru and gfreq, plus the hot-set refresh, and compare synchronous miss bytes over the first four 16-token windows.
  - The result decides B1 against the never-run `REFRESH_HOT` arm.
- **B0 result (2026-11-14, ADR 0050 addendum): B1 is not built.** At 26 VRAM slots the best pool across layers falls short of ADR 0050's halving threshold against the best per-layer policy, and little synchronous read time per token is left (`replay-20261114-v26`). The VRAM hot-set arm (`main-overlap-poll0-hot`) goes on box D1 instead.
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
  - tool-call repair G1 and G2 (`docs/gateway-briefs.md`): built (ADR 0051);
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
   - the planner's own fill tier, last (it may OOM): 35 slots (38.4e9 B) by its estimate on a 49,140 MiB (51.5e9 B) card. The memory measured besides slots at the exit run's ubatch 2048 (`20261114`) matches the planner formula at ubatch 2048, so 35 slots fit only at ubatch 256 (estimate 50.5e9 B); candidates are fit-checked per ubatch (`builder.rs`), so serve would pick 35 slots at ubatch 256. Record its peak and prefill rate (the harness's 22.5 GB reserve gives 26 slots at any ubatch);
   - trace.
   - Host 60400 or a second qualifying host, to widen point 3's scope.
5. C4 on the C6 layout.
6. **Box D2** (3090 class, about $3 to $4), after C2, C3, C4 and C5:
   - the C0 verdict pairs on the 8B (own dense builder and libllama) and the 30B (own MoE builder), at concurrency 1 and 4 against tuned llama-server;
   - the TTFT comparison;
   - C6 GPU numbers if step 2 ran;
   - the spec rerun;
   - `serve --plan` part B.

**Budget:** "at most two boxes" probably won't hold. Phase 4 likely needs a D3 confirmation box (about $3). The total is about $8 to $10 of the $19.66. (No session directory yet.)

The coordinator reviews every diff, consult Fable at C1 and before each box, and drivers pick machines by `docs/box-driver.md` without asking the owner.

## Verification
- **Every change:** `cargo build/test/clippy -D warnings/fmt --check --workspace` on Windows and Linux CI; `tests/test_stage_names.py` for stage lists; no em dashes in docs.
- **C1.5:** MoE logits bit-equal before and after.
- **C5:** parity per op and end to end on CPU and GPU, plus the greedy golden sequence.
- **C1 to C3:** scheduler property tests, then a concurrency-4 CI smoke on the tiny fixture.
- **C6:** micro-benchmark JSON committed; thresholds stated in the ADR before measuring.
- **B1 and B2:** greedy parity; engine bytes equal the replay; the planner never exceeds its budget.
- **Boxes:** paired same-box runs in `bench/results/<date>/` (no traces over a few MB). Verdicts go in ADR 0045, ADR 0046 and a new Phase 4 verdict ADR judged by C0's rules.
