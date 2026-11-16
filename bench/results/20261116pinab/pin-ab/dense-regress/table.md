Qwen3-8B Q4_K_M, all layers on GPU, min of 3 rounds (ms per step shape); old = d2e8696

shape | engine | old | head
--- | --- | --- | ---
alone | dense | 83.9 | 83.7
alone | llama | - | 88.1
decode1x8_alone | dense | 60.4 | 61.4
decode1x8_alone | llama | 59.1 | 58.4
decode1x8_resident | dense | 64.7 | 63.8
decode1x8_resident | llama | 59.3 | 60.7
decode4x8 | dense | 75.3 | 85.8
decode4x8 | llama | 71.8 | 83.4
mixed1 | dense | 93.7 | 92.5
mixed1 | llama | 88.5 | 92.1
mixed1_prompt_first | dense | 92.9 | 92.7
mixed1_prompt_first | llama | 88.1 | 91.6
mixed2 | dense | 94.2 | 92.7
mixed2 | llama | 90.1 | 91.9
mixed2_prompt_first | dense | 92.8 | 92.4
mixed2_prompt_first | llama | 89.9 | 91.5
mixed3 | dense | 93.7 | 93.3
mixed3 | llama | 91.6 | 92.2
mixed3_prompt_first | dense | 93.6 | 93.2
mixed3_prompt_first | llama | 91.5 | 92.1
resident | dense | 92.9 | 91.8
resident | llama | 86.0 | 91.2
split1 | dense | 101.6 | 100.6
split1 | llama | 94.4 | 100.3
split2 | dense | 102.6 | 102.1
split2 | llama | 95.2 | 100.6
split3 | dense | 104.6 | 102.8
split3 | llama | 96.4 | 101.9

dense / llama ratio (min of rounds)

shape | old | head | head dense_norepack
--- | --- | --- | ---
alone | - | 0.950 | -
decode1x8_alone | 1.022 | 1.051 | -
decode1x8_resident | 1.091 | 1.051 | -
decode4x8 | 1.049 | 1.029 | -
mixed1 | 1.059 | 1.004 | -
mixed1_prompt_first | 1.054 | 1.012 | -
mixed2 | 1.046 | 1.009 | -
mixed2_prompt_first | 1.032 | 1.010 | -
mixed3 | 1.023 | 1.012 | -
mixed3_prompt_first | 1.023 | 1.012 | -
resident | 1.080 | 1.007 | -
split1 | 1.076 | 1.003 | -
split2 | 1.078 | 1.015 | -
split3 | 1.085 | 1.009 | -
