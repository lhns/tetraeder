# 0067. KV layout: chunked contiguous first, paged only if a staged experiment clears thresholds fixed now

Status: Accepted (owner, 2026-11-19). Proposed (2026-11-14). Phase 4. Settles the plan's open decision "KV layout in Phase 4" by the procedure below; the result is added here as an addendum and in `docs/plan.md`. Builds on ADR 0065 (one pool per layer, attention with a mask) and 0011 (patch queue). Briefs: C6-1 in `docs/phase4-briefs.md`; step 2 gets its own brief only if step 1 sends it there. Step 1 decided: chunked, outcome 4, no patch (`20261114`, addendum below).

Measurements live in the session READMEs (`bench/results/<session>/README.md`), not here.

## Context

With ADR 0065 (a), attention is one `flash_attn_ext` over rows `[0, n_kv)` of a per-layer pool and a mask; which rows a sequence owns does not change the op. A layout is therefore only a row allocator, and the question splits in two:

- **Memory.** A contiguous run per sequence must be reserved ahead (or moved when it grows) and leaves holes between runs; fixed blocks allocate on demand with at most one partial block per sequence.
- **Reads.** The CPU kernel skips masked rows (`ggml-cpu/ops.cpp:8944`), so scattered rows cost it almost nothing. CUDA reads every row below n_kv per query tile (`fattn-common.cuh:671` trims only the tail), so blocks scattered across the pool make n_kv follow the pool's high-water mark instead of the step's own sequences. The way around it on unmodified ggml is a per-sequence gather: `ggml_get_rows` returns F32 only (`ggml.c:4015`), so a gathered K and V need a copy back to F16 before CUDA flash attention takes them.

At the pin ggml already has a sparse flash attention path: `ggml_flash_attn_ext_set_n_kv_max` makes CUDA compact the finite mask entries into index lists and read only those rows (`fattn.cu:14-125`). It is compiled only for head sizes 256, 512 and 576 (`ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse`, `fattn-mma-f16.cuh:1788`), NVIDIA Turing or newer, K of at least 4096 rows. Qwen3 has head size 128, so it is not reachable without a patch; the CPU ignores the hint.

## Decision

**Step 1 (build and measure, unmodified ggml).**

- `crates/kvcache` gets the `Layout` trait and two layouts, both 256-row aligned (`KV_PAD`):
  - `Chunked`: one contiguous run per sequence, first fit from the lowest row. Reserved at admission for prompt plus min(max_tokens, GROW), GROW = 1024 (ADR 0065), rounded up to 256. Grows in place when the next rows are free, else moves to a free run of the needed size plus GROW (row copies the builder applies before the step; a 20K-token move of Qwen3-8B F16 is 2.9 GB, milliseconds on a GPU against seconds of re-prefill), else `Full`.
  - `Paged { block }`: blocks of `block` rows (256 default, 64 simulated too), allocated on demand, lowest free block first.
- `KvPool` over a layout plans each step: RoPE positions, `set_rows` destination rows, n_kv and the F16 mask. Chunked is what C4 and C5 use from the start; paged is built for the simulator and the pool tests only.
- **Memory simulation** (deterministic, CI), workload W: the 10 Tetraeder sessions of `bench/results/20261111d/claude-code-proxy/proxy-tetraeder-r1-*.jsonl` (232 requests; prompt = `input_tokens + cache_read_tokens`, output = `output_tokens`), run as 4 concurrent streams (sessions round-robin by file name, each stream's requests back to back); a request continues the open conversation of its session with the longest last prompt not longer than its own, else starts one (the rule of `scripts/claude_code_cache_classes.py`; a session can hold several, such as a main conversation and a subagent); a finished session's sequence stays as an idle prefix and is evicted, oldest first, when rows are needed; `max_tokens` 32000, at most 131072 positions per sequence. The metric is the smallest pool, in 256-row units, at which no live sequence is preempted (idle ones may be evicted), found by bisection, per layout. Sanity workload: the c4 bench (16 requests of 512 plus 128, concurrency 4), where both layouts must need the hand-computed pool of 4 x 768 = 3072 rows. Memory saving s = 1 - pool(paged, 256) / pool(chunked) on W (256-row blocks decide; 64 is reported only).
- **Read cost of paged on CUDA, from the simulation:** per decode step of W the simulator records n_kv under each layout. With row bytes R (K plus V, all layers; 147,456 for Qwen3-8B F16) and weight bytes Wt = 5,021,827,072 (Qwen3-8B Q4_K_M, `model_size` in `bench/results/20261103/gateway/overhead-q8b-llama-bench.json`), y_mask = sum over steps of (n_kv(paged) - n_kv(chunked)) x R / sum over steps of (Wt + n_kv(chunked) x R): extra bytes per byte of a decode step, the bandwidth bound a decode step sits on.
- **CPU gather micro-benchmark** (`crates/model/tests/kv_gather.rs`, ignored, label `cpu`): Qwen3-8B attention shape (32 query heads, 8 KV heads, head 128, F16 K and V), sequence length L in {2048, 8192, 32768}, queries {1, 4}, blocks {64, 256} scattered in a pool of 4L rows, median of 20 at `bench_threads()`: (i) flash attention over a contiguous L-row view; (ii) `get_rows` of the sequence's rows, copy to F16, flash attention; (iii) flash attention over the whole pool with a mask selecting the L rows. From it y_gather = (t(ii) / t(i) - 1) x share(L), share(L) = L x R / (Wt + L x R), averaged over W's decode steps at their lengths (ratios interpolated between the measured L). (iii) reads only the rows the mask leaves open, which is what the sparse path of step 2 does on CUDA, so it is the proxy for the patched cost: y_sparse = (t(iii) / t(i) - 1) x share(L), averaged the same way.
- **Thresholds, fixed before measuring:**
  - **X = 25% memory saving.** KV bytes buy concurrency and context. A quarter is one more sequence at four, the smallest step that changes what a box serves; smaller savings are already beaten by Q8_0 KV (ADR 0050 b, 47% fewer bytes at a measured quality cost); and a patch carries a weekly rebase cost (Rule 8, ADR 0011) that a smaller gain does not repay.
  - **Y = 3% decode cost.** The Phase 4 verdict passes at ratios down to 1 - band, and a measured band is clamped at a floor of 0.02 (ADR 0026 addendum); the A40 and A6000 class files sit at that floor and gate at 0.98 once they hold three sessions. A layout costing more than 3% of decode could by itself turn a pass on those cards into a fail. It is also the size of host costs the decode-gap work removed one by one (ADR 0046 addendum 2026-11-09).
- **Outcome rules:**
  1. s < X: chunked is the layout; the open decision closes; no patch.
  2. s >= X and min(y_mask, y_gather) <= Y: paged is adopted on unmodified ggml with the cheaper read path; no patch.
  3. s >= X, both above Y, and y_sparse <= Y: step 1 is cleared; step 2.
  4. s >= X and y_sparse > Y: even a patched read costs too much; chunked, no patch.

Outputs: `bench/results/<date>/kv-layout/sim.json` (`{workload, pool_rows: {chunked, paged_256, paged_64}, saving, y_mask, relocations, steps}`) and `kv-gather.json` (`{cpu, threads, cases: [{l, n_q, block, t_contig_us, t_gather_us, t_mask_us}], y_gather, y_sparse}`), and `verdict` in `sim.json`: `chunked`, `paged_unmodified` or `step2`.

**Step 2 (only on outcome 3): a CUDA patch in the queue.** First candidate, because it is the smallest change: extend the pinned sparse path to head size 128 (`may_use_sparse` and the kernel instances for (128, 128), with its tile widths), so mask attention reads only the step's own rows for any layout. A block-table attention kernel is the fallback if that path cannot reach the cost. Either way: an ADR of its own, a patch in `patches/series` with a `docs/sources.md` row, backend tests (`test-backend-ops` cases for the new shapes, parity of the builders against the CPU within `TOL_GPU`), never an edit of vendored ggml in place (Rule 8). Gate, paired on a box (ADR 0065 pairing and band rules): paged plus patch against chunked, decode at c1 and throughput at c4 with W's length mix, ratio at least 1 - Y in both pairs. Fail: chunked.

**Adoption.** `KvPool` takes a `Box<dyn Layout>`; C4 and C5 build it from `--kv-layout chunked|paged` (default the winner). Masks, rows and copies come from the pool, so a switch needs no builder change. The libllama backend keeps llama.cpp's own cache. The prefix cache's entries stay per-sequence row snapshots (ADR 0049), layout independent. Shared prefix rows (several sequences reading one row, ADR 0065) are a paged-only follow-up with reference counts and are not part of this decision.

**Metrics (Rule 6).** `tetraeder.kv.blocks` (catalog gauge, 256-row units, `tetraeder.state` = used, reserved, idle, free; device, tier), `tetraeder.kv.relocations` (counter; bytes go to `tetraeder.tier.transfer.bytes` with purpose `kv`, VRAM to VRAM, ADR 0063). Panel "KV pool: blocks by state, relocations" in `overview.json` beside "KV cache". Alert `TetraederKvFragmented`: preemptions while free blocks exceed 25% of the pool, for 10 minutes (holes, not size, are the limit: the production signal for paged).

## Consequences

- C4 and C5 do not wait for the experiment: chunked on unmodified attention is the starting layout either way.
- The memory question is answered in CI from committed request counts, without a box; only outcome 3 needs GPU time.
- The thresholds cannot be moved after the numbers are in; a later change of X or Y needs a new ADR that names the numbers it saw.
- Prefix sharing without copies, the main reason paged caches exist elsewhere, is not priced here; if it is wanted later, it reopens this decision through a new ADR.

## Addendum 2026-11-14: step 1 result, chunked

`bench/results/20261114/kv-layout/` (`20261114`; branch `track-c6-kv`).
- Memory: paged saves at least X on W. Reads: y_gather and y_sparse are both above Y (`20261114`).
- y_mask is read at the common pool, chunked's minimal one. The definition above does not fix the pool, but its concern is that scattered blocks make n_kv follow the pool's high-water mark, which is a comparison at one pool size; at its own smaller pool paged's n_kv is capped by the pool, so the memory saving would count a second time, in Y as well as in X. Read that way it is above Y.
- Outcome 4: **chunked**, no patch. C4 and C5 use chunked (`--kv-layout` default `chunked`); paged stays in `kvcache` for the simulator and tests.
- The timings ran on a slow, loaded dev machine; a faster kernel makes gather overhead relatively larger, so the y values likely understate the cost on real hardware. The verdict does not depend on them being exact (all exceed Y). Revisit only if a later workload needs more than about 1.5x the sequences per VRAM that chunked gives.

## History

- 2026-11-18: a lone request whose contiguous run cannot grow where it lies (rows after it taken, a move cannot overlap the old run) no longer fails although the pool has room: admission frees its own idle prefix (saved to RAM) and places it again from row 0, and a lone growing sequence is requeued instead of ending with `Length`. Decision unchanged; cost one snapshot and restore, or a re-prefill when `--cache-ram` cannot hold the prefix. Compaction (moving a run down over itself) would avoid that and is left open. Found in `20261117l`.
