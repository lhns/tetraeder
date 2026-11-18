# Session 20261116gpu5

- Status: clean: every one-GPU parity test of the fixtures passes under ADR 0071.
- Code and box: results commit a6aa634; box 60861226, RTX 3090.
- Raw files: this directory.

## From ADR 0071 (addendum "gpu-smoke5 closes it", moved verbatim)

Every GPU parity test passes: the move is bit-exact after the move (F16 and q8_0 KV); the CPU anchor (builder against libllama with `op_offload` off) is 0 on all 38 rows, confirming the offload cause; the dense and MoE GPU suites stay within the rule (worst builder batched against split 3.35e-3 and 2.40e-3 of the largest logit, libllama's own 5.73e-3 and 3.06e-3). Status stays Proposed until the Phase 4 verdict session runs the parity gate on its own box.
