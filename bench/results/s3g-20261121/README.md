# Session s3g-20261121

- Status: pending (criteria written before the box ran).
- Rehearsal of `phase5_box.sh` on 4 GPUs (`SESSION=s3g-20261121 NGPUS=4 ROUNDS=1`, default MODELS, layer anchor on). First box run of cost model 5 (ADR 0082) and its `layer-g@L/2` anchor. Base: main f204ed4, branch box-s3g-20261121.

## Criteria (written before measuring)

- **R1 (plumbing):** the layer anchor ran (two rounds); `clocks-*.csv` and `window-*.txt` exist for every timed run of both engines; calibrate logged one `decode_clocks <model>: ...` line per model. Report k and the reason when k = 1.
- **R2 (held-out):** the layer anchor's measured tg is within 0.05 of its v5 prediction.
- **R3:** 30B layer-4 `e_tg` (pred/meas - 1) is within 0.07.
- **R4:** the 30B tp2-pp2 decode tail clock (from its clocks file, cut to the timed tail with `scripts/decode_clocks.py`) is in the busy state, above the low clock by more than 15 MHz.
- **Report only:** every candidate's pp and tg ratios against prediction, next to s3f2's.
- **Clock-limited host:** if busy_mhz is within 15 MHz of low_mhz (as s3f2 at 1590), then k = 1. R2 to R4 then test only the plumbing, and only R1 is judged.

## Results

(to be filled in)
