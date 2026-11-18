# Session s2d-20261118

- Status: rehearsal stopped at the gpu-parity gate (no verdict).
- Code and box: 723519f, box 61185926, host 34742.
- Raw files: this directory.

## Findings

From ADR 0074 (addendum "staged against straight is not a gate", moved verbatim):

- On the S2d rehearsal box the 30B save took 10.93 ms staged against 10.81 ms straight (ratio 1.01); the same 52 MB copy took 36 ms straight on other hosts. The ratio depends on the host, so `gpu_kv_save` prints it and asserts only the absolute bounds (25 ms 8B, 18 ms 30B).
