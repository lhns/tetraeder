# Session 20261111c

- Status: clean against the llama.cpp cache at share 0.5 (builder ahead, ADR 0045); the KTransformers runs are an outlier (fault-bound host) and do not count; the tuner failed (OOM).
- Code and box: results commit 22e2952; box 60206988, host 706255, RTX 3090, AMD EPYC 7642, 46 threads, 91 GB balloon.
- Raw files: this directory.

## From ADR 0045 (addendum "llama.cpp cache at share 0.5, tuner OOM", moved verbatim)

## Addendum (2026-11-11, box 60206988): llama.cpp cache at share 0.5, tuner OOM

RTX 3090, AMD EPYC 7642 (AVX2), host 706255, 46 threads, ballooned to 91 GB, disk 7.3 GB/s at depth 16 (gate passed). Stage `moe-tune` failed: llama.cpp's tuner sizes its first probe at 10% of the expert bytes (13048 MiB on the 235B) regardless of `--max-mib`, and with the f16 KV at 32K the CUDA compute buffer did not fit (rc 1, no report). `exit_run.sh` now runs the tuner at `TUNE_CTX` 8192. `llamacpp-cache` ran the session fallback, 10458 MiB at share 0.5: decode 1.440 and 1.562 tok/s against overlap 3.460 and 3.555, ratios 2.40 and 2.28 adjacent, 2.21 and 2.47 cross; prefill 65 vs 28 tok/s. With shares 0 and 0.9 (`20261110f`) the hand sweep now covers 0, 0.5 and 0.9 at the largest cache that fits; the builder leads each by 1.66x or more. KTransformers (kt-kernel 0.7.2, AVX2) decoded 0.523 and 0.529 tok/s with a 1453 s time to first token, against 4.5 to 4.7 on the EPYC 7742 at 30 threads; ratios 6.5 to 6.8. That is an outlier for KTransformers on this host, not a builder result, and does not enter the verdict. Its host block shows why: decode read 69 to 79 MB/s at queue depth 0.37 with 56% CPU and about 2,600 major faults per token, so it waited on page faults served one at a time (about 0.7 ms each), not on disk bandwidth or thread count; on the EPYC 7742 the same fault count cost about 0.08 ms each. Builder runs 2.7% apart, greedy tokens equal, same prompt hash as the cache runs. Results `bench/results/20261111c/`.
