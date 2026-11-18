# Session 20261116gpu3

- Status: diagnosis: dense GPU parity passes under ADR 0071; the move subcases failed (the reference's fault); S1 patch 0004 timing test measured flat and was replaced (ADR 0070).
- Code and box: results commit e620987; box 60849241, RTX 3090.
- Raw files: this directory.

## From ADR 0070 (section "Revised after S1", moved verbatim)

- At the pin MMQ skips every tile of an expert without tokens and `mm_ids_helper` never matches -1, so -1 slots cost nothing in MMQ even without patch 0004. The expert-parallel saving comes from the experts that get no tokens on a rank: "without it, EP wastes (g-1)/g of expert compute at prefill" holds against remapping remote slots to local experts, not against MMQ.
- What the patch adds: defined output for -1 on MMQ and MMF (zero rows, a few microseconds); no weight reads for -1 on MMVQ and MMVF (decode), where unpatched kernels read expert -1 out of bounds; the generic path no longer asserts on -1; and `mm_ids_helper` entries left unwritten for -1 now point at row 0 (without that, MMQ's gate/up scatter wrote through stale pool indices; found by reading after S1).
- S1's timing test used random -1 ids, which leave nearly every expert with a token, and measured flat (0.61 ms). It is replaced by time against the number of local experts (g = 1 to 8) compared with remapping. K-quant parity uses 2e-2 (q8_1 against q8_K activations, about 5e-3 expected); q8_0 and f16 keep 5e-3.

## From ADR 0071 (addendum "gpu-smoke3", moved verbatim)

- The rule holds on the dense builder: every dense GPU test passes, including the three-sequence one that was open (CUDA flash attention over mixed sequences). On CUDA libllama's own batched against split gap is 0.0895 (5.7e-3 of the largest logit), the builder's 0.0523; builder against libllama at the same shapes 0.0582 batched and 0.0312 split. No growth with position or sequence; no greedy flip.
- MoE on CUDA: the builder batched is bit-equal to libllama batched; batched against split 0.0377 for both.
- Not closed: `gpu_seq_cp_snapshot_and_move` fails in its move subcases (F16 KV jumps to 0.171 at step 19; q8_0 KV is off by 2.4e-2 of the largest logit from step 0, with a greedy flip at step 62) while copy, prefix copy and source continuation pass. This looks like a real fault in the device-side row move, not noise; under diagnosis (`track-gpu-move`). The verdict's gpu-parity gate stays strict until it is fixed.

## From ADR 0071 (addendum "the move failures were the reference's", `track-gpu-move`, moved verbatim)

- `move_mid_decode` compared a run whose sequence starts at the moved row: before the move that reference has another `n_kv` and row offset, and attention sums in an order that depends on both (split-KV CPU flash attention from `n_kv` 512 with several threads and F16 V accumulation, and CUDA's KV split); on near-tied MoE fixtures this reaches 2e-2. The F16 spike at step 19 comes before the move (step 56), and after the move the gap falls to 4e-4 to 1e-3. The test now checks that the moved bytes are equal and compares only steps with the same rows and `n_kv`; the tolerance is unchanged. `copy_rows` was checked by reading (whole rows, correct stride and offset, synchronized before the next step, no overlap, rotation matrices fixed).
- CPU references (`n_gpu_layers` 0) set `op_offload` off: in a CUDA build libllama otherwise runs host ops of 32 or more rows on the GPU, the likely cause of the box-only CPU anchor gap (1.09e-3 relative). gpu-smoke5 confirms it or not.
