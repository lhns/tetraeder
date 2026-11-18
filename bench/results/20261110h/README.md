# Session 20261110h

- Status: null; KTransformers did not start (disk-starved host), builder only, no pairing (ADR 0045).
- Code and box: commit on main at the time of box 60101282 (see git log of the results commit); RTX 3090, Xeon Gold 6138 (AVX-512), 91 GB balloon, 38 threads.
- Raw files: this directory.

## From ADR 0045 (addendum "AVX-512 attempt", moved verbatim)

## Addendum (2026-11-10, box 60101282): AVX-512 attempt

Xeon Gold 6138 (Skylake AVX-512, no VNNI), RTX 3090, cgroup 137 GB ballooned to 91, 38 threads = floor(quota). KTransformers did not start: sglang hung at "Capture cuda graph bs [1]" for 40 min, no OOM, about 40% iowait. Its LLAMAFILE backend reads the GGUF experts in place through mmap (peak RSS 23.8 GB; the captured log tail shows MoE layer setup through layer 84 of 94), so the capture's forward passes page in experts with small faulting reads; on this disk that did not finish in the 40 min stage timeout. Likely cause: the disk, not a KTransformers bug, since the same command ran on the 3 GB/s boxes. A retry needs a fast disk under random reads, not only a good sequential number. The host was disk-starved (decode reads 675 to 863 MB/s, prefill about 410 MB/s; earlier boxes 2.8 to 3.0 GB/s), so absolute numbers do not compare across boxes. Builder only: overlap/prefetch decode 1.27 (1.574 vs 1.241 tok/s), prefill 1.03, greedy tokens equal, hidden share 0.41; overlap and `cpu_experts` GPU parity tests pass. Overlap against KTransformers on AVX-512 remains unmeasured. Results `bench/results/20261110h/`.
