# Session 20261116pc

- Status: confirmation session; not a verdict.
- Code and box: commit on main at the time of box 60983677 (see git log of the results commit).
- Raw files: this directory.

## Findings

From ADR 0074 Context (the measurement that motivated it):

- An admission that reuses an idle sequence first saves it to the host RAM prefix cache (ADR 0049, 0065), on the worker thread, with every live sequence stalled. On one RTX 3090 (`bench/results/20261116pc/pin-confirm/`, `save_ms` in the admission log) that takes 41 to 44 ms for Qwen3-8B and 30 to 33 ms for Qwen3-30B-A3B per sequence of about 533 tokens (78 and 52 MB of F16 KV, so about 1.8 GB/s); three saves in one pass serialize to 91 to 121 ms. It is the largest remaining loss at concurrency 4.
