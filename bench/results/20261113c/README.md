# Session 20261113c

- Status: null; no pairing: llama.cpp dio OOM-killed at load, builder stages fail the co-tenant guard (ADR 0045).
- Code and box: commit on main at the time of box 60449808 (see git log of the results commit); host 130210.
- Raw files: this directory.

## From ADR 0045 (addendum "tuner too slow for the 235B; thin margin at 23 threads", the `20261113c` paragraph, moved verbatim)

`20261113c` (box 60449808, host 130210, new container, same launch, cache stages `llamacpp-cache-dio-poll0` and its repeat): **no pairing.** llama-server with `--load-mode dio` was OOM-killed at load in both stages (exit -9, cgroup `oom_kill 1`, balloon intact): dio reads the CPU-resident experts of the 142 GB model into anonymous memory, which cannot fit under the 91 GB limit, while mmap pages file-backed memory. So in the larger-than-RAM regime the cache runs only in mmap, and mmap at poll 0 (`20261113b`) is its best runnable configuration. Both builder stages fail the co-tenant guard (about 28 cores busy outside the container, limit 14.4; decode 1.39 and 0.97 tok/s, prefill and disk about half of `20261113b`) and do not count. Results `bench/results/20261113c/`.
