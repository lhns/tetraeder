# Session phase6-exit-s-20261127f (Case S rerun, attempt f): power gate failed on box a, nothing built or downloaded

- Status: null, no verdict. Box a's RTX 3090 runs at a 300 W power limit against its 420 W default, so the power gate (ADR 0091 answer 8, `phase6_box.sh gates`) failed and the `gates` stage stopped the session; both boxes were destroyed before the bulk screen, the build and the 142 GB download (0 of 2 download attempts used).
- Rule: ADR 0091 Case S, ADR 0092 rerun rules, `docs/notes/phase6-rule-changes.md`; scripts at 09a3216a (main 293cf6a6 plus the finder's gated-host ranking), `STAGES="gates" bash scripts/phase6_exit_s.sh`.
- Search: `scripts/phase6_pair_create.sh` in the foreground from 14:39 (local time, UTC+1). Four earlier tries rented nothing usable: two listed no pair, two lost one side because the finder's offer was gone at create (taxameter found no offer on the machine in three searches), and the other side's box (62863428 host 306949, 62864025 host 195919) was destroyed within a minute.
- Pair (`pairs.json` of try 5, ranked first by the gated-host preference: host 79466 passed the gates twice before; both creates at 15:02, disk_gb 230, `--ports 7000`, `provision --models ,`, both `cuda: ok`):
  - a: box 62865644, host 195919 (Alberta, CA), RTX 3090, Ryzen 9 5950X (1 socket, 1 NUMA node), offer RAM 64.4 GB, 0.360 USD/h all-in; created 15:02:48, destroyed 15:11:27.
  - b: box 62865634, host 79466 (Quebec, CA), RTX 3090, Xeon W-2133 (1 socket, 1 NUMA node), 0.443 USD/h all-in; created 15:02:45, destroyed 15:11:30.
- Gates (`gates/<n>/`): one NUMA node, idle and co-tenant (a 0.02 outside cores, b 0.07, limit 4.0) pass on both; power b 350 W at default 350 W ok, a "gpu 0: limit 300.00 W, default 420.00 W, max 450.00 W LIMITED" (`gates/a/power-gate.txt`), so "GATE FAILED a: power or idle".
- Destroy needed `--force` (taxameter's run list holds the committed `bench/results` dirs; nothing of this session was on either box beyond the gate files fetched above).
- Cost: about 0.12 USD for this pair (about 9 min at 0.80 USD/h, rent only) plus about 0.02 USD for the two lone boxes; the Vast credit read 15.52 USD at 14:33 and 15.37 USD at 15:11, it may lag.
- Consequence: host 195919 goes on the finder's skip list (power capped is a host property, docs/box-driver.md). Host 79466 passed every gate a third time.
