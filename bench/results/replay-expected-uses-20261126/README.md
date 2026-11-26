# Replay: expected-uses expert ranking (`gexp`), 2026-11-26

Tree node `memory-replay-expected-uses`. Offline replay of the saved 235B routing trace; no box, no engine change.

- Code: commit 77e80d2a on `memory-replay-expected-uses` (policy `gexp:<horizon>`, `decode_sync_windows`).
- Trace: `routing-235b-041805.trace` (fnv1a64 `2941f8e4dcdb3812`, the same as in `bench/results/20261114/replay-235b-ram.json`), found in the main checkout's `runs/traces/` (`routing-235b-20261114-041805.trace` and its `.zst`). The reader takes the uncompressed file. Calibration: `lru`, VRAM 10, RAM 77, prefetch on gives 46.91 MB sync and 265.64 MB disk per token, as in the 20261114 JSON.
- Command (docker host, lane `replay`, trace copied in with `DOCKER_CARGO_EXTRA`):
  `DOCKER_CARGO_EXTRA=bench/traces/routing-235b-20261114-041805.trace bash scripts/docker_cargo.sh -l replay -- cargo run --release -p tetraeder-bench --bin tetraeder-bench -- replay --trace bench/traces/routing-235b-20261114-041805.trace --vram 26,10 --ram 77 --policies lru,glru,gfreq:0.5,gfreq:1,gbelady,gexp:64,gexp:512,gexp:2048 --decode-window 16 --out replay-235b-expected-uses.json`
- Raw: `replay-235b-expected-uses.json` (prefetch on and off; `decode_windows` is disk, `decode_sync_windows` synchronous misses, bytes per token per 16 decode tokens).

## The policy

`gexp:<H>` is `glru` (one byte budget across layers) in prefill. In decode the resident with the fewest expected uses within H decode steps (the tier turnover) is evicted, ties by LRU. Value of (layer, expert):

- history: a count of decode steps that routed it, decayed with time constant H, seeded at the first decode step from the prefill routing counts per token (times H);
- lookahead: +precision if a pending router prediction names it (`predicted` for the next layer, `predicted2` for the layer after; precision is the running share of named experts that were routed, prior 4 of 5).

A prefetch read may displace only a resident of lower value (a refused read is skipped, not issued); a demand read always gets room. H = 512 was fixed before the run as the turnover estimate (7238 resident experts over about 14 admissions per step at VRAM 10); 64 and 2048 are the sweep around it.

Not done: the trace has no next-token routing, so lookahead is the layer predictions only. The VRAM bank is the static hot set of the engine as built (no admissions in decode), so the ranking applies to the RAM tier only; a VRAM admission policy is a separate row (`VramPolicy`) and was not built.

## Numbers

MB per decode token, prefetch on, RAM 77 slots, static VRAM tier, decode of 256 tokens. "first 64" is the mean of the first four 16-token windows.

VRAM 26 (the configuration in use):

| policy | sync, all | disk, all | sync, first 64 | disk, first 64 |
|---|---|---|---|---|
| lru | 17.60 | 100.61 | 36.33 | 228.66 |
| glru | 14.45 | 66.21 | 39.51 | 162.02 |
| gfreq:0.5 | 13.24 | 59.20 | 37.15 | 143.76 |
| gfreq:1 | 11.80 | 48.93 | 34.27 | 117.89 |
| gbelady (no prefetch, bound) | 2.25 | 2.25 | 2.74 | 2.74 |
| gexp:64 | 11.46 | 57.08 | 33.25 | 123.13 |
| gexp:512 (pre-registered) | 11.91 | 53.99 | 33.44 | 123.13 |
| gexp:2048 | 15.55 | 45.12 | 35.09 | 114.45 |

Paired ratios at VRAM 26 (gexp over reference, lower is better; sync all / disk all / sync first 64 / disk first 64):

| gexp | vs gfreq:1 (lowest sync) | vs gfreq:0.5 | vs glru |
|---|---|---|---|
| 512 | 1.009 / 1.103 / 0.976 / 1.045 | 0.899 / 0.912 / 0.900 / 0.857 | 0.824 / 0.815 / 0.846 / 0.760 |
| 64 | 0.971 / 1.167 / 0.970 / 1.044 | 0.865 / 0.964 / 0.895 / 0.857 | 0.793 / 0.862 / 0.842 / 0.760 |

VRAM 10 (comparison with the 2026-11-14 result):

| policy | sync, all | disk, all | sync, first 64 | disk, first 64 |
|---|---|---|---|---|
| lru | 46.91 | 265.64 | 83.81 | 561.48 |
| glru | 27.20 | 171.53 | 67.32 | 408.03 |
| gfreq:0.5 | 27.42 | 164.22 | 66.03 | 363.65 |
| gfreq:1 | 32.79 | 190.27 | 60.71 | 317.26 |
| gbelady | 2.62 | 2.62 | 3.10 | 3.10 |
| gexp:64 | 26.11 | 181.06 | 61.17 | 329.31 |
| gexp:512 | 55.60 | 164.54 | 72.10 | 306.10 |
| gexp:2048 | 74.12 | 189.21 | 82.46 | 308.94 |

At VRAM 10 `gexp:512` and `gexp:2048` are worse than `glru` on sync bytes: with a long horizon the history outweighs the lookahead (value of a lookahead is at most 1), so correct prefetches are refused and become synchronous reads, while the disk total stays level.

## Verdict

Does expected-uses beat the best existing global policy by at least 10% fewer synchronous miss bytes at 26 slots? **No.** The best existing global policy on sync bytes at 26 slots is `gfreq:1` (11.80 MB per token). `gexp:512` reads 11.91 (1.009x, 1% more); the best of the sweep, `gexp:64`, reads 11.46 (0.971x, 2.9% fewer). Against `gfreq:0.5` (13.24) `gexp:512` is 10.1% lower, and against `glru` 17.6% lower, but those are not the best reference. Over the first 64 tokens `gexp` is 2 to 3% below `gfreq:1`. One trace, one session, replay only: no port to the engine on this evidence (rule 3).
