# Session phase6-exit-s-20261127h (Case S rerun, attempt h): one-NUMA gate failed on box a, nothing built or downloaded

- Status: null, no verdict. Box a's host has two sockets and two NUMA nodes, so the one-NUMA gate (ADR 0091 answer 8, `phase6_box.sh gates`) failed and the `gates` stage stopped the session; both boxes were destroyed before the bulk screen, the build and the 142 GB download (0 of 2 download attempts used).
- Rule: ADR 0091 Case S, ADR 0092 rerun rules, `docs/notes/phase6-rule-changes.md`; scripts at 34a0b350, `STAGES="gates" bash scripts/phase6_exit_s.sh`.
- Pair (`scripts/phase6_pair_create.sh`, both creates at 15:28 local time, UTC+1, disk_gb 230, `--ports 7000`, `provision --models ,`, both `cuda: ok`):
  - a: box 62869102, host 560104 (UK), RTX 3090, Xeon E5-2690 v4 (2 sockets, 2 NUMA nodes, 56 threads), 0.331 USD/h all-in; created 15:28:03, ready 15:35:25 (7 min loading), destroyed 15:42:11.
  - b: box 62869108, host 79466 (Quebec, CA), RTX 3090, Xeon W-2133, 0.443 USD/h all-in; created 15:28:05, ready 15:28:55, destroyed 15:42:14.
- Gates (`gates/<n>/`): power at default on both (a 350 of 350 W, b 350 of 350 W), idle and co-tenant pass (a 2.99 outside cores, b 0.03, limit 4.0); NUMA b one node ok, a "GATE FAILED numa: the host has 2 NUMA nodes" (`gates/a/lscpu.txt`).
- Cost: about 0.18 USD (both boxes about 14 min at 0.77 USD/h, rent only); the Vast credit read 15.26 USD at 15:26 and 15.12 USD at 15:42, it may lag.
- Consequence: host 560104 goes on the finder's skip list (two sockets is a host property). The offer has no socket or NUMA field; its CPU name (E5-2690 v4, 14 cores) against the host.s 56 threads is the only hint, and the raw offer.s host thread count (`cpu_cores`, 32 on host 195919 for 16 rented) could flag such hosts as a ranking hint.
