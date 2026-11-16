# Pin confirm summary

```
{
  "host": "7928aca21abd",
  "commit": "759ca1d89f70da2d57d92e9d97a0bb694d2a7939",
  "threads": "23",
  "vast": "",
  "bpin": "tag=b12426 commit=fc504f91d8cfdfcd9b0fedcf14600fd2b0e63420",
  "quota": "23",
  "baseline": "version: 1.0.0-dev (build 1, commit fc504f9) built with GNU 13.3.0 for Linux x86_64",
  "gpu": "NVIDIA GeForce RTX 3090, 580.159.03, 255 MHz, 2130 MHz, 5001 MHz, 24576 MiB",
  "pin": "tag=b12426 commit=fc504f91d8cfdfcd9b0fedcf14600fd2b0e63420 ik_commit=b25c3fd2c6d7e407c3a104a413d0211fe2c7a88e ik_date=2026-11-01 baseline_tag=b12426 baseline_commit=fc504f91d8cfdfcd9b0fedcf14600fd2b0e63420 image_tag=b12426-ikb25c3fd2-blfc504f91"
}
```

## Test stages

- smoke6 rc: {"smoke5_rc": 101, "executor_gpu_rc": 0}
- smoke5 build/tests rc: {"build_rc": 0, "tests_rc": 101}
- real-batched rc: {"q8b_rc": 0, "q30b_rc": 0}
- gpu-smoke5/mul_mat_id_skip.log: test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 0.73s
- gpu-smoke5/gpu_multi_seq.log: test result: FAILED. 5 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.02s
- gpu-smoke5/gpu_dense.log: test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 4.79s
- gpu-smoke5/executor_gpu.log: test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 2 filtered out; finished in 0.22s
- real-batched/q8b.log: test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2 filtered out; finished in 6.20s
- real-batched/q30b.log: test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2 filtered out; finished in 16.11s

### real-batched q8b (verbatim lines)

    qwen3-8b libllama batched / libllama split: worst 1.94e-2 of the largest logit
    qwen3-8b libllama batched / libllama shifted by 7 cells: worst 0.00e0 of the largest logit
    qwen3-8b libllama batched / libllama shifted by 256 cells: worst 0.00e0 of the largest logit
    qwen3-8b builder batched / builder split: worst 1.90e-2 of the largest logit
    qwen3-8b builder batched / libllama batched: worst 1.85e-2 of the largest logit
    qwen3-8b builder split / libllama split: worst 0.00e0 of the largest logit
    qwen3-8b builder alone / libllama alone: worst 0.00e0 of the largest logit
    qwen3-8b: bound max(5e-3, 2.91e-2 of the largest logit), same shapes 2.91e-2, alone 5.00e-3

### real-batched q30b (verbatim lines)

    qwen3-30b libllama batched / libllama split: (0, 22) argmax 250 against 245: reference margin 7.704544e-2 (3.44e-3 of largest), other margin 1.3469696e-2
    qwen3-30b libllama batched / libllama split: (0, 24) argmax 222 against 243: reference margin 8.094788e-3 (4.55e-4 of largest), other margin 6.0709e-2
    qwen3-30b libllama batched / libllama split: (1, 23) argmax 217 against 205: reference margin 2.2277832e-3 (1.75e-4 of largest), other margin 3.01857e-2
    qwen3-30b libllama batched / libllama split: (1, 29) argmax 216 against 205: reference margin 1.6649246e-2 (1.28e-3 of largest), other margin 6.147957e-2
    qwen3-30b builder batched / builder split: (0, 21) argmax 232 against 223: reference margin 1.8974304e-2 (9.62e-4 of largest), other margin 3.4503937e-2
    qwen3-30b builder batched / builder split: (0, 22) argmax 250 against 245: reference margin 7.704544e-2 (3.44e-3 of largest), other margin 2.7456284e-2
    qwen3-30b builder batched / builder split: (0, 24) argmax 222 against 243: reference margin 8.094788e-3 (4.55e-4 of largest), other margin 1.6978073e-1
    qwen3-30b builder batched / builder split: (1, 27) argmax 204 against 205: reference margin 2.119255e-2 (1.83e-3 of largest), other margin 1.1397362e-1
    qwen3-30b builder batched / builder split: (1, 29) argmax 216 against 205: reference margin 1.6649246e-2 (1.28e-3 of largest), other margin 1.13931656e-1
    qwen3-30b builder batched / libllama batched: (0, 21) argmax 232 against 223: reference margin 2.3582458e-2 (1.20e-3 of largest), other margin 3.4503937e-2
    qwen3-30b builder batched / libllama batched: (1, 23) argmax 205 against 217: reference margin 3.01857e-2 (2.38e-3 of largest), other margin 3.3964157e-2
    qwen3-30b builder batched / libllama batched: (1, 27) argmax 204 against 205: reference margin 3.4210205e-2 (2.94e-3 of largest), other margin 1.1397362e-1
    qwen3-30b libllama batched / libllama split: worst 6.87e-2 of the largest logit
    qwen3-30b libllama batched / libllama shifted by 7 cells: worst 0.00e0 of the largest logit
    qwen3-30b libllama batched / libllama shifted by 256 cells: worst 0.00e0 of the largest logit
    qwen3-30b builder batched / builder split: worst 6.99e-2 of the largest logit
    qwen3-30b builder batched / libllama batched: worst 5.78e-2 of the largest logit
    qwen3-30b builder split / libllama split: worst 0.00e0 of the largest logit
    qwen3-30b builder alone / libllama alone: worst 0.00e0 of the largest logit
    qwen3-30b: bound max(5e-3, 1.03e-1 of the largest logit), same shapes 1.03e-1, alone 5.00e-3

## step_shapes on CUDA (Qwen3-8B Q4_K_M, all layers on GPU), ms per step shape

shape | engine | per round | min | median
--- | --- | --- | --- | ---
alone | dense | 83.3, 84.1, 84.1 | 83.3 | 84.1
alone | llama | 84.0, 86.3, 87.2 | 84.0 | 86.3
decode1x8_alone | dense | 60.7, 59.0, 59.1 | 59.0 | 59.1
decode1x8_alone | llama | 59.8, 59.5, 60.2 | 59.5 | 59.8
decode1x8_resident | dense | 63.7, 63.3, 63.3 | 63.3 | 63.3
decode1x8_resident | llama | 59.7, 59.5, 59.9 | 59.5 | 59.7
decode4x8 | dense | 73.2, 72.8, 72.9 | 72.8 | 72.9
decode4x8 | llama | 72.0, 71.3, 72.0 | 71.3 | 72.0
mixed1 | dense | 94.3, 94.7, 94.9 | 94.3 | 94.7
mixed1 | llama | 87.6, 88.9, 89.8 | 87.6 | 88.9
mixed1_prompt_first | dense | 93.8, 95.2, 94.7 | 93.8 | 94.7
mixed1_prompt_first | llama | 88.3, 89.0, 90.0 | 88.3 | 89.0
mixed2 | dense | 93.7, 94.5, 95.7 | 93.7 | 94.5
mixed2 | llama | 89.8, 90.3, 91.7 | 89.8 | 90.3
mixed2_prompt_first | dense | 93.9, 94.5, 94.7 | 93.9 | 94.5
mixed2_prompt_first | llama | 89.5, 90.4, 91.3 | 89.5 | 90.4
mixed3 | dense | 94.4, 95.0, 95.7 | 94.4 | 95.0
mixed3 | llama | 91.7, 91.8, 92.7 | 91.7 | 91.8
mixed3_prompt_first | dense | 94.3, 95.3, 95.2 | 94.3 | 95.2
mixed3_prompt_first | llama | 91.4, 92.3, 93.4 | 91.4 | 92.3
resident | dense | 93.2, 94.1, 94.1 | 93.2 | 94.1
resident | llama | 85.4, 86.3, 86.8 | 85.4 | 86.3
split1 | dense | 102.2, 103.6, 104.4 | 102.2 | 103.6
split1 | llama | 94.2, 95.0, 96.5 | 94.2 | 95.0
split2 | dense | 102.7, 103.4, 103.6 | 102.7 | 103.4
split2 | llama | 95.4, 95.7, 96.7 | 95.4 | 95.7
split3 | dense | 104.3, 107.4, 105.1 | 104.3 | 105.1
split3 | llama | 96.0, 96.8, 98.1 | 96.0 | 96.8

## Probe (D2a workload: 512 prompt tokens, 128 max tokens, 8 requests; seeds equal across engines per round)

### q30b c1

round | engine | throughput tps | ttft median s | ttft p90 s | decode tps median | errors | throttle before/after
--- | --- | --- | --- | --- | --- | --- | ---
r1 | llama-server | 170.0 | 0.119 | 0.121 | 200.7 | 0 | 0x0000000000000000/0x0000000000000004
r1 | tetraeder | 167.6 | 0.130 | 0.135 | 200.5 | 0 | 0x0000000000000000/0x0000000000000004
r2 | llama-server | 169.0 | 0.120 | 0.123 | 199.8 | 0 | 0x0000000000000000/0x0000000000000004
r2 | tetraeder | 167.9 | 0.130 | 0.133 | 200.5 | 0 | 0x0000000000000000/0x0000000000000004

round | tetraeder / llama-server throughput
--- | ---
r1 | 0.986
r2 | 0.993
mean | 0.990

### q30b c4

round | engine | throughput tps | ttft median s | ttft p90 s | decode tps median | errors | throttle before/after
--- | --- | --- | --- | --- | --- | --- | ---
r1 | llama-server | 312.9 | 0.403 | 0.429 | 103.1 | 0 | 0x0000000000000000/0x0000000000000004
r1 | tetraeder | 282.9 | 0.484 | 0.532 | 96.4 | 0 | 0x0000000000000000/0x0000000000000004
r2 | llama-server | 313.1 | 0.407 | 0.435 | 103.5 | 0 | 0x0000000000000000/0x0000000000000004
r2 | tetraeder | 290.1 | 0.482 | 0.530 | 99.6 | 0 | 0x0000000000000000/0x0000000000000004

round | tetraeder / llama-server throughput
--- | ---
r1 | 0.904
r2 | 0.927
mean | 0.915

### q8b c1

round | engine | throughput tps | ttft median s | ttft p90 s | decode tps median | errors | throttle before/after
--- | --- | --- | --- | --- | --- | --- | ---
r1 | llama-server | 120.8 | 0.137 | 0.141 | 137.9 | 0 | 0x0000000000000000/0x0000000000000004
r1 | tetraeder | 120.4 | 0.146 | 0.149 | 138.3 | 0 | 0x0000000000000000/0x0000000000000004
r2 | llama-server | 119.6 | 0.140 | 0.141 | 136.7 | 0 | 0x0000000000000000/0x0000000000000004
r2 | tetraeder | 119.9 | 0.148 | 0.180 | 138.2 | 0 | 0x0000000000000000/0x0000000000000004

round | tetraeder / llama-server throughput
--- | ---
r1 | 0.997
r2 | 1.002
mean | 0.999

### q8b c4

round | engine | throughput tps | ttft median s | ttft p90 s | decode tps median | errors | throttle before/after
--- | --- | --- | --- | --- | --- | --- | ---
r1 | llama-server | 316.2 | 0.448 | 0.486 | 108.6 | 0 | 0x0000000000000000/0x0000000000000004
r1 | tetraeder | 251.3 | 0.400 | 0.405 | 107.7 | 4 | 0x0000000000000000/0x0000000000000004
r2 | llama-server | 312.8 | 0.443 | 0.481 | 106.4 | 0 | 0x0000000000000000/0x0000000000000004
r2 | tetraeder | 308.4 | 0.460 | 0.520 | 106.4 | 0 | 0x0000000000000000/0x0000000000000004

round | tetraeder / llama-server throughput
--- | ---
r1 | 0.795
r2 | 0.986
mean | 0.890

## tetraeder probe logs (RUST_LOG=info,tetraeder::step=debug)

### q30b-tetraeder-r1: 1298 steps, 17 admits, 13 admitting steps, 13 prefill steps

Per admitting step (admits logged since the previous step):

step | admits | pre_seqs | pre_toks | save_ms (sum) | place_ms (sum) | queued_ms (max) | gap_ms | decode_ms
--- | --- | --- | --- | --- | --- | --- | --- | ---
0 | 1 | 1 | 112 | 0.00 | 0.05 | 0.08 | -1.00 | 167.12
16 | 1 | 1 | 404 | 0.00 | 0.02 | 0.02 | 146.76 | 81.74
144 | 1 | 1 | 402 | 30.82 | 30.84 | 0.02 | 33.45 | 80.15
272 | 1 | 1 | 402 | 31.11 | 31.13 | 0.02 | 33.95 | 79.78
400 | 1 | 1 | 404 | 31.00 | 31.02 | 0.01 | 33.39 | 81.09
528 | 1 | 1 | 401 | 31.52 | 31.54 | 0.02 | 33.88 | 79.58
656 | 1 | 1 | 403 | 31.14 | 31.16 | 0.02 | 33.48 | 80.30
784 | 1 | 1 | 403 | 30.96 | 30.98 | 0.02 | 33.39 | 80.97
912 | 1 | 1 | 401 | 30.79 | 30.80 | 0.01 | 33.23 | 79.74
1040 | 1 | 1 | 403 | 33.16 | 33.19 | 0.04 | 181.36 | 80.43
1041 | 3 | 3 | 1216 | 0.00 | 0.01 | 112.99 | 0.06 | 281.37
1169 | 1 | 1 | 402 | 29.85 | 29.87 | 9.52 | 30.32 | 97.80
1170 | 3 | 3 | 1206 | 91.53 | 91.59 | 184.50 | 92.10 | 284.66

gap_ms distribution (gap -1, no previous step, excluded):

steps | n | p50 | p90 | p99 | max
--- | --- | --- | --- | --- | ---
all | 1297 | 0.01 | 0.01 | 0.04 | 181.36
admitting | 12 | 33.48 | 146.76 | 181.36 | 181.36
other | 1285 | 0.01 | 0.01 | 0.03 | 0.05

Prompts per prefill step (pre_seqs):

prompts | prefill steps
--- | ---
1 | 11
3 | 2

### q30b-tetraeder-r2: 1298 steps, 17 admits, 13 admitting steps, 13 prefill steps

Per admitting step (admits logged since the previous step):

step | admits | pre_seqs | pre_toks | save_ms (sum) | place_ms (sum) | queued_ms (max) | gap_ms | decode_ms
--- | --- | --- | --- | --- | --- | --- | --- | ---
0 | 1 | 1 | 112 | 0.00 | 0.03 | 0.04 | -1.00 | 170.34
16 | 1 | 1 | 403 | 0.00 | 0.02 | 0.02 | 144.27 | 79.81
144 | 1 | 1 | 402 | 31.28 | 31.30 | 0.01 | 33.88 | 79.22
272 | 1 | 1 | 401 | 30.93 | 30.96 | 0.01 | 33.37 | 80.55
400 | 1 | 1 | 400 | 30.38 | 30.40 | 0.01 | 33.19 | 79.38
528 | 1 | 1 | 401 | 31.49 | 31.52 | 0.01 | 33.97 | 80.19
656 | 1 | 1 | 402 | 30.71 | 30.73 | 0.02 | 33.15 | 80.54
784 | 1 | 1 | 403 | 30.62 | 30.65 | 0.02 | 33.55 | 79.54
912 | 1 | 1 | 404 | 31.41 | 31.43 | 0.01 | 33.79 | 81.55
1040 | 1 | 1 | 401 | 30.91 | 30.94 | 0.01 | 180.30 | 78.08
1041 | 3 | 3 | 1215 | 0.00 | 0.01 | 108.66 | 0.06 | 282.63
1169 | 1 | 1 | 402 | 30.12 | 30.13 | 9.10 | 30.17 | 93.16
1170 | 3 | 3 | 1204 | 90.02 | 90.06 | 178.19 | 90.52 | 291.26

gap_ms distribution (gap -1, no previous step, excluded):

steps | n | p50 | p90 | p99 | max
--- | --- | --- | --- | --- | ---
all | 1297 | 0.01 | 0.01 | 0.04 | 180.30
admitting | 12 | 33.79 | 144.27 | 180.30 | 180.30
other | 1285 | 0.01 | 0.01 | 0.03 | 0.09

Prompts per prefill step (pre_seqs):

prompts | prefill steps
--- | ---
1 | 11
3 | 2

### q8b-tetraeder-r1: 1170 steps, 14 admits, 12 admitting steps, 12 prefill steps

Per admitting step (admits logged since the previous step):

step | admits | pre_seqs | pre_toks | save_ms (sum) | place_ms (sum) | queued_ms (max) | gap_ms | decode_ms
--- | --- | --- | --- | --- | --- | --- | --- | ---
0 | 1 | 1 | 115 | 0.00 | 0.05 | 0.03 | -1.00 | 72.70
16 | 1 | 1 | 404 | 0.00 | 0.02 | 0.03 | 142.48 | 84.61
144 | 1 | 1 | 402 | 44.50 | 44.53 | 0.02 | 47.71 | 83.32
272 | 1 | 1 | 402 | 42.19 | 42.22 | 0.02 | 45.01 | 83.79
400 | 1 | 1 | 404 | 41.79 | 41.81 | 0.02 | 44.33 | 84.13
528 | 1 | 1 | 401 | 42.29 | 42.30 | 0.02 | 45.20 | 83.69
656 | 1 | 1 | 403 | 42.29 | 42.30 | 0.02 | 44.94 | 83.46
784 | 1 | 1 | 403 | 41.52 | 41.53 | 0.02 | 44.22 | 84.03
912 | 1 | 1 | 401 | 42.61 | 42.62 | 0.02 | 44.87 | 83.73
1040 | 1 | 1 | 403 | 41.92 | 41.95 | 0.02 | 191.51 | 83.96
1041 | 3 | 3 | 1216 | 0.00 | 0.01 | 125.53 | 0.05 | 244.14
1169 | 1 | 1 | 402 | 42.04 | 42.06 | 7.79 | 42.56 | 99.12

gap_ms distribution (gap -1, no previous step, excluded):

steps | n | p50 | p90 | p99 | max
--- | --- | --- | --- | --- | ---
all | 1169 | 0.01 | 0.01 | 0.05 | 191.51
admitting | 11 | 44.94 | 142.48 | 191.51 | 191.51
other | 1158 | 0.01 | 0.01 | 0.02 | 0.06

Prompts per prefill step (pre_seqs):

prompts | prefill steps
--- | ---
1 | 11
3 | 1

### q8b-tetraeder-r2: 1298 steps, 17 admits, 13 admitting steps, 13 prefill steps

Per admitting step (admits logged since the previous step):

step | admits | pre_seqs | pre_toks | save_ms (sum) | place_ms (sum) | queued_ms (max) | gap_ms | decode_ms
--- | --- | --- | --- | --- | --- | --- | --- | ---
0 | 1 | 1 | 114 | 0.00 | 0.03 | 0.02 | -1.00 | 73.91
16 | 1 | 1 | 402 | 0.00 | 0.03 | 0.04 | 140.68 | 84.42
144 | 1 | 1 | 402 | 43.48 | 43.50 | 0.02 | 47.12 | 83.45
272 | 1 | 1 | 401 | 42.56 | 42.58 | 0.02 | 45.52 | 84.29
400 | 1 | 1 | 400 | 41.42 | 41.44 | 0.02 | 43.77 | 83.04
528 | 1 | 1 | 401 | 42.37 | 42.40 | 0.02 | 45.23 | 83.40
656 | 1 | 1 | 402 | 41.72 | 41.74 | 0.02 | 44.03 | 84.09
784 | 1 | 1 | 403 | 42.70 | 42.73 | 0.02 | 45.67 | 84.73
912 | 1 | 1 | 404 | 74.11 | 74.14 | 0.02 | 76.45 | 84.76
1040 | 1 | 1 | 401 | 41.00 | 41.03 | 0.02 | 185.74 | 83.09
1041 | 3 | 3 | 1215 | 0.00 | 0.00 | 123.77 | 0.05 | 242.55
1169 | 1 | 1 | 402 | 40.50 | 40.52 | 7.47 | 40.58 | 96.06
1170 | 3 | 3 | 1204 | 120.96 | 121.01 | 213.66 | 121.59 | 238.68

gap_ms distribution (gap -1, no previous step, excluded):

steps | n | p50 | p90 | p99 | max
--- | --- | --- | --- | --- | ---
all | 1297 | 0.01 | 0.01 | 0.05 | 185.74
admitting | 12 | 45.67 | 140.68 | 185.74 | 185.74
other | 1285 | 0.01 | 0.01 | 0.02 | 0.05

Prompts per prefill step (pre_seqs):

prompts | prefill steps
--- | ---
1 | 11
3 | 2

