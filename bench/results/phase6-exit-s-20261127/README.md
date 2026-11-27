# Session phase6-exit-s-20261127 (Case S rerun): power gate failed on box a, nothing built or downloaded

- Status: null, no verdict. Box a's RTX 3090 runs at a 360 W power limit against its 370 W default, so the power gate (ADR 0091 answer 8, `phase6_box.sh gates`) fails; the `gates` stage stopped the session and both boxes were destroyed before the bulk screen, the build and the 142 GB download (0 of 2 download attempts used).
- Rule: ADR 0091 Case S, ADR 0092 rerun rules, `docs/notes/phase6-rule-changes.md`; scripts at main ba76e33d (`STAGES="gates bulk setup dryrun budget plan linkgate fetch calib predict" bash scripts/phase6_exit_s.sh`).
- Pair (`pairs.json`, the only pair the finder listed at 00:54, 38 covering candidates, bulk 1487 Mbit/s expected; created by `scripts/phase6_pair_create.sh`, both creates at once, disk_gb 230, `provision --models ,`, both `cuda: ok`):
  - a: box 62771670, host 392144 (NL), RTX 3090, Threadripper 3960X (1 socket, 1 NUMA node, 48 threads, quota 23 cores), memory.max 91.6 GB, 0.365 USD/h plus 0.326 USD/h traffic; created 00:54:47, destroyed 01:11:31 (local time, UTC+1).
  - b: box 62771679, host 79466 (Quebec), RTX 3090, Xeon W-2133 (1 socket, 1 NUMA node, 12 threads), memory.max 129.3 GB, 0.409 USD/h plus 0.195 USD/h traffic; created 00:54:47, destroyed 01:11:33.
  - a stayed `loading` until 01:02:37 (8 min), b was ready at 00:56:11.
- Gates (`gates/<n>/`): one NUMA node, at least 64 GB RAM (`first-ssh.txt`), `cuda: ok`, idle and co-tenant (a 0.55 outside cores, b 0.25, limit 4.0) pass on both. Power: b 350 W at default 350 W, ok; a "gpu 0: limit 360.00 W, default 370.00 W, max 390.00 W LIMITED" (`gates/a/power-gate.txt`, `nvidia-smi` enforced limit 360 W), so "GATE FAILED a: power or idle" and the driver's "FATAL: a gate failed on a box".
- Clock at the gates: 15 min old, 160 min padded to the end, 175 of 180 min.
- Cost: rate times box time about 0.25 USD (a 16.7 min at 0.432, b 16.8 min at 0.443 USD/h, rent only; no traffic billed). The Vast credit read 15.87 USD at 01:05 and 15.77 USD at 01:11, it may lag.
- Consequence: host 392144 goes on the finder's skip list (power capped). Host 79466 passed every gate again (as in pair 1 of 20261126).
