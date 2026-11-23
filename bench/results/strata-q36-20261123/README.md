# Session strata-q36-20261123

- Status: measurement, no verdict (ADR 0085), all 9 paired runs clean. Medians of three interleaved rounds, one A40, Qwen3.6-35B-A3B UD-Q4_K_XL: Strata v0.2.20 over llama-server is 1.503 at c1 decode and 0.837 at c4 throughput; tetraeder serve --engine auto (libllama) over llama-server is 1.009 and 0.996.
- Cleanliness: errors 0, 128 tokens per request, board-limit share 0.00 and no throttle bit outside 0x1 in any run window.
- Purpose: node `bench-strata-q36`. How far Strata's q36 engine is ahead of llama.cpp on CUDA (its claim: 1.44x decode on HIP, `docs/QWEN36.md`) and where the libllama path of Tetraeder stands, before any own `qwen35moe` builder (ADR 0085 decision 3). Also the ADR 0084 step 1 read-rate probes and an A/A of the own 30B engine (`bench-30b-c4-gap`).
- Code: branch `bench-strata-q36`; box script `scripts/strata_q36_box.sh` (run at 2eb4a320 for the final pass), summary `scripts/strata_q36_summary.py` (run locally on the fetched files, writes `summary.json`), probe `scripts/probes/mapped_read.cu`.
- Box: Vast 62065877, host 682584 (Belgium), 1x NVIDIA A40 (46068 MiB, compute 8.6), driver 595.99.02, power limit 300 W = default (gate ok), AMD EPYC 7662 (64 cores, 256 host CPUs, cgroup quota 61, threads 61), 2003 GB host RAM, 120 GB disk. 0.58 USD/h plus traffic. Gates at SSH: `cuda: ok`, power gate, idle gate 10/10, co-tenant sample 3.60 busy host cores over 20 s with the container idle (limit max(4, 15% of 128 physical cores)). Clocks at start 225 MHz idle of 1740 MHz max (`env.json`, `nvidia-smi-clock-performance.txt`); throttle reasons 0x1 (idle) before the runs, 0x0 before and after every run (`gpu-*.txt`), 1 Hz sampler `gpu-samples.csv`: 19 to 22 window samples per run, board-limit share 0.00 in all.
- Life: created 01:35, destroyed 02:18 (about 43 min), credit 31.16 to 30.69 USD (0.47 USD with traffic of 44 GB), destroyed after a verified fetch (`results-62065877-20261123-021725.tgz`). Times are the local clock; the box clock runs one hour behind.

## Model and engines

- Model: `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`, `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (21795 MB), the file Strata's `docs/QWEN36.md` uses (it carries the MTP block). `general.architecture` = `qwen35moe` (read from the GGUF header on the box). Offline after download, one file for all three engines.
- a, llama-server: `/opt/llama.cpp-baseline/bin/llama-server`, tag b12426, commit fc504f91 (`vendor/PIN` `baseline_commit`), all layers on GPU, `-np 4 -c 32768 -fa on -ctk f16 -ctv f16 -t 61`, plus the variant the pre-pass picked (below): `-kvu -ub 2048 --ctx-checkpoints 0 --cache-ram 0`. The MTP block is not used by llama.cpp.
- b, Tetraeder: `target/release/tetraeder serve -m F --engine auto --ngl -1 --parallel 4 --ctx 32768 -t 61 --flash-attn on --no-prefix-reuse --cache-ram 0 --accept-any-model`, built with CUDA at the commit `2eb4a320` (libllama path; the own builders do not run `qwen35moe`). The last two flags are a deviation, see Findings.
- c, Strata: v0.2.20, installed with the release's own `install.sh` (the release folder `https://github.com/Niko1221/Strata/releases/download/v0.2.20`, archive SHA-256 checked by the installer), engine package `strata-engine-0.1.66-linux-x86_64-cuda-cuda13` (the Terminal chose cuda13 for this driver; BUILD.json: archs 75;86;89;120, CUDA 13.0), fetched by `strata launch --feed <that folder>`; engine binary sha256 `a8b072a9b8befd2510df89d448439f61265487a562bf969a5dbc6fed0b9a2ca5`. Not built from source (an official package exists and ran headless). Launch (one fresh `strata launch` and `strata launch --stop` per run): `strata launch --feed <folder> --model-path F --context 8192 --parallel 4 --vision off --access local --port 18090 --no-api-key --yes --no-tui`; the plan expands this to `--model ud-q4_k_xl --family qwen36 --tune skip` (no tuning: the command line skips it unless asked). The engine command line was `strata --serve --native F --max-context 8192 --prefill auto --mtp 2 --batch 4` (`strata-effective.txt`): 4 effective slots, MTP block loaded with 2 drafts per step (Strata's default for this model), prompt chunk 1024, fp32 KV (4 KiB per token per attention layer), every routed expert in VRAM (18.75 GiB; 27.4 GB of the 46 GB used), 4.40 GiB of dense weights on the GPU (`strata-engine.log`).
- Not like for like, by Strata's defaults: Strata decodes with MTP drafts, llama-server and libllama do not; Strata's context is 8192 per slot (llama-server 32768 shared over 4 slots, 8192 each), its KV is fp32 against f16, pictures are off (no mmproj loaded, as for the others).

## Workload and rule (fixed before the runs)

- `scripts/gateway_bench.py` at each server's OpenAI chat endpoint, temperature 0, `--no-keepalive`: c4 `--prompt-tokens 512 --max-tokens 128 --concurrency 4 --requests 16`, then c1 the same with `--concurrency 1 --requests 8`; seeds per round (c4 round*10+4, c1 round*10+1), the same for the three engines; one warm-up request; a fresh server process per run. Rounds in the order a b c, b c a, c a b (`bench-<a|b|c>-r<n>-<c4|c1>.json`). c4 is `throughput_tps`, c1 is `decode_tps_median`. Ratio per round, then the median of the three. No threshold, no verdict.
- Baseline pre-pass (ADR 0065 style, c4 once per variant, seed 84): the variant with the highest c4 throughput is the baseline. Result (tok/s): `-kvu` 180.7, `-no-kvu` 189.3, `-kvu -ub 2048` 190.6, `-kvu --ctx-checkpoints 0 --cache-ram 0` 210.0, `-kvu -ub 2048 --ctx-checkpoints 0 --cache-ram 0` 217.3 (picked; `prepass-*.json`). The unique prompts of this workload never reuse the llama-server prompt cache or its recurrent-state checkpoints, so switching them off costs nothing and saves their per-request work (TTFT 0.89 s default against 0.33 s picked).

## Results (final pass, baseline variant `kvu-ub2048-nocp`)

Source of every number: `summary.json` (from `bench-*.json`; `scripts/strata_q36_summary.py`).

| round | order | c4 a tok/s | c4 b | c4 c | c4 b/a | c4 c/a | c1 a decode tok/s | c1 b | c1 c | c1 b/a | c1 c/a |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | a b c | 214.1 | 213.2 | 179.2 | 0.996 | 0.837 | 116.1 | 116.7 | 172.1 | 1.005 | 1.481 |
| 2 | b c a | 213.6 | 212.7 | 180.7 | 0.996 | 0.846 | 115.9 | 116.9 | 174.2 | 1.009 | 1.503 |
| 3 | c a b | 216.6 | 215.8 | 179.5 | 0.996 | 0.829 | 115.8 | 118.6 | 177.5 | 1.024 | 1.533 |
| median | | | | | 0.996 | 0.837 | | | | 1.009 | 1.503 |

- Strata decodes a single request 1.48 to 1.53 times faster than llama-server (172 to 178 against 116 tok/s; its claim on HIP is 1.44x), and is 0.83 to 0.85 of llama-server at concurrency 4 (179 to 181 against 214 to 217 tok/s). At c4 its per-request decode is 54 to 55 tok/s against 61 to 62 for llama-server and its median TTFT 0.52 s against 0.33 s: the MTP drafts that win at c1 do not carry over to the 4-slot batch (the engine's batch window has no drafts unless `--batch-mtp`, `docs/QWEN36.md`), and the 8-row window is shared.
- The libllama path of Tetraeder is level with llama-server (b/a 0.996 at c4 in all three rounds, 1.005 to 1.024 at c1), as expected for the same ggml; it adds no speed and no loss.
- First pass, with llama-server at its default cache flags (`-kvu`, checkpoints and prompt cache on): c4 a 180.7, 185.6, 179.1 tok/s, b/a 1.178, 1.150, 1.186 (median 1.178), c/a 0.989, 0.974, 1.003 (median 0.989); c1 a 116.2, 116.8, 114.5, b/a 1.010, 1.004, 1.024 (median 1.010), c/a 1.483, 1.489, 1.552 (median 1.489). Files in `pass1/`. That baseline was not tuned (the pre-pass then had only `-kvu` and `-no-kvu`), so the b/a above 1 there is the baseline's own overhead, not a gain; the final pass supersedes it. Kept because the choice was made after seeing it.
- Run noise: the three b/a c4 ratios agree to 0.001; c1 a varies 0.3% and b 1.6% across rounds; Strata's c1 rises 172.1, 174.2, 177.5 over the rounds (3%; not order: it ran third, second, first).

## Findings

- `tetraeder serve --engine auto` on `qwen35moe` fails on slot reuse with default flags: the second request on a used slot returns `engine error: invalid request: the KV cache cannot drop positions N.. of sequence S` (HTTP 500; as a stream it ends with no tokens). `bench-b0-default-c4.json` has 9 of 16 requests with 128 tokens (7 empty) and `bench-b0-default-c1.json` 4 of 8 (4 empty), errors 0 on the client because the stream is HTTP 200. The cause is the recurrent state of the GatedDeltaNet layers, which cannot drop a suffix (ADR 0051 deferred exactly this; ADR 0085 decision 4 names it the design question of the own builder). `--no-prefix-reuse --cache-ram 0` makes every request prefill from an empty sequence and removes the failure (4 concurrent requests x 3 rounds probe, then all runs here); with unique prompts nothing is lost against llama-server, but any shared-prefix workload on this model is broken on the libllama path today.
- Strata ran headless from its official CUDA package after one attempt, no source build, no tuning step, default launch for the model apart from `--context 8192 --parallel 4 --vision off`.

## Probes for ADR 0084 step 1 (`probes/`)

| Probe | Result | ADR 0084 assumption |
|---|---|---|
| Mapped host read by a CUDA kernel (sum reduction, `cudaHostAllocMapped`, 2 GB, 7 repeats; `mapped-read-2gb.txt`) | median 24.60 GB/s (4 GB, 5 repeats: 24.63) | 25 GB/s over PCIe 4.0 x16 |
| `cudaMemcpy` host to device of the same pinned buffer | median 24.78 GB/s (4 GB: 24.94) | 25 GB/s |
| CPU attention over host KV, `kv_gather` contiguous variant, `n_q` 1, one Qwen3-8B layer (32 heads, 8 KV heads, 128), F16 K and V, ggml CPU tier avx2, 61 threads, EPYC 7662 (`kv-gather.json`, `kv-gather.log`) | L 2048: 94 us (89 GB/s); L 8192: 282 to 292 us (115 to 119 GB/s); L 32768: 2012 to 2051 us (65 to 67 GB/s) of K plus V read | 60 GB/s from host RAM; the earlier 0.33 GB/s was a local 8-thread SSE-tier CPU |

- GB/s is bytes of K plus V (4096 per token and layer) over the measured attention time. The 32K case is 134 MB per layer, and the EPYC's 256 MB of L3 holds it between repeats, so 65 to 67 GB/s is an upper bound for a cold DRAM read; the 8192 case (33 MB) is cache resident. The gather variant (scattered blocks) reads 26 to 27 GB/s at 32K.
- Reading: the PCIe path (mapped read and copy) saturates at 24.6 to 24.9 GB/s, matching the ADR's 25. CPU attention on a 61-thread EPYC reads faster (65 to 67 GB/s at 32K) than PCIe, so the ADR's step-1 skip rule (CPU attention under the PCIe rate) does not fire on this CPU; the 0.33 GB/s figure came from a small SSE-tier CPU and does not carry over to a many-core AVX2 server. A box with fewer cores will sit lower: the rate scales with threads, which this probe did not sweep.

## A/A of the own 30B engine (`bench-30b-c4-gap`)

Same flags as `bench/results/ab30b-20261122/README.md` apart from the machine and threads: `--engine moe --ngl -1 --parallel 4 --ctx 32768 -t 61 --accept-any-model`, Qwen3-30B-A3B Q4_K_M, one binary, two runs (seed 14 and 11 for both), c4 and c1 of the same workload.

| | run 1 | run 2 | run 2 / run 1 |
|---|---|---|---|
| c4 throughput tok/s | 226.0 | 226.2 | 1.001 |
| c1 decode tok/s | 153.1 | 152.5 | 0.996 |

The noise of this box for the own 30B engine is under 0.5% at c4 and c1 (two runs only). Absolute tok/s are not comparable with other machines.

## Deviations

- The box was created with the offer chosen by price among A40 offers (no A6000, L40 or L40S offer was cheaper at `--min-inet 1000`); `.vast/config.json` `disk_gb` 120 (git-ignored).
- Threads 61 for llama-server and Tetraeder is the cgroup quota; all layers are on the GPU, so they do little.
- The llama-server baseline was re-tuned after the first pass (above); the first pass is kept in `pass1/` and reported.
- Tetraeder runs with `--no-prefix-reuse --cache-ram 0` (Findings); a default-flag run is kept as `bench-b0-default-*`, informational.
- Runs are short (6 to 11 s per test, the model is in the page cache); the three rounds are the repetition. `smoke-*` files are a four-request first look per engine, not results.
- No cotenant sampler per run (`engine_sample.py` follows one server pid and Strata runs several processes); the idle co-tenant sample and the GPU sampler stand in.
- ADR 0084 probe: the AVX-512 tier of ADR 0084's wording was not available (EPYC 7662 is Zen 2, AVX2 tier); no thread sweep.
