# Session phase6-exit-s-20261126c (Case S rerun): one-NUMA gate failed at first SSH, nothing built or downloaded

- Status: null, no verdict. The A40 box has two sockets and two NUMA nodes, so the one-NUMA-node gate (ADR 0091 answer 8) fails; both boxes were destroyed at first SSH, before the gates stage, the build and the 142 GB download (0 of 2 download attempts used).
- Rule: ADR 0091 Case S, ADR 0092 rerun rules, `docs/notes/phase6-rule-changes.md` (judged boxes at least 64 GB RAM), scripts at 4531e7f2 plus `scripts/phase6_pair_create.sh`.
- Pair (`pairs.json`, finder at 19:40 UTC with the real planner, 38 covering candidates; created by `scripts/phase6_pair_create.sh`, both creates at once):
  - a: box 62743509, host 45726 (Japan), RTX 3090 (390 W limit at default), Ryzen 9 5950X (1 socket, 1 NUMA node, 32 threads), memory.max 91.6 GB, disk 230 GB, 0.334 USD/h; created 20:40:28, destroyed 20:51:11 (local time, UTC+1).
  - b: box 62743504, host 682584 (Belgium), A40 (300 W at default), EPYC 7662, **2 sockets, 2 NUMA nodes** (255 threads), memory.max 1033 GB, disk 230 GB, 0.561 USD/h; created 20:40:25, destroyed 20:51:09.
  - Both took about 10 min from create to SSH (a: status `loading` until 20:50:28).
- Evidence: `gates/<n>/first-ssh.txt` (`lscpu`, `memory.max`, `df`, `nvidia-smi` power limits). The gate's own check on them, `python3 -I scripts/sc_gates.py numa gates/<n>/first-ssh.txt`: a "gate numa ok"; b "GATE FAILED numa: the host has 2 NUMA nodes, a judged session needs exactly one". The `gates` stage itself never ran (decided at first SSH).
- Earlier tries of the same evening: an A40 alone (box 62741097, host 682584, 20:20 to 20:24, never reached SSH) was destroyed when its 3090 partner offer vanished during the create; two finder pairs vanished before any create (offer ids rotate within seconds, so `phase6_pair_create.sh` takes the id the finder's machine has at create time).
- Cost: rate times box time about 0.21 USD for all three boxes (62741097 4.5 min at 0.636, 62743504 10.7 min at 0.561, 62743509 10.7 min at 0.334 USD/h, rent only); the Vast credit moved 16.13 to 16.04 USD (0.09) by 21:00, it may lag.
- The 4 h one-off (user, 2026-11-26) was for this pair: `max_box_hours` is back to 3 and the AGENTS.md line is marked used.
- Consequence: host 682584 is the only A40 host the finder has listed on 2026-11-26, so it goes on the finder's skip list (two sockets); every other listed pair that evening had a box under 64 GB RAM.
