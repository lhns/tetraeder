# Session phase6-exit-s-20261127j (Case S rerun, attempt j): scout probes, no pair, no verdict

- Status: null, no verdict. No pair of two pass-list machines was listed during the search, so no pair was created and `scripts/phase6_exit_s.sh` never ran (0 of 2 download attempts used).
- Rule: ADR 0091 Case S, ADR 0092 rerun rules, `docs/notes/phase6-rule-changes.md` (rows of 2026-11-27: partners learned apart from pairing; measured memory.max of a scouted machine); scripts on branch `phase6-exit-s-20261127j`.
- Approach (advisor, delegated by the owner 2026-11-27): learn each partner's host properties with a one-box scout (`scripts/phase6_scout.py probe`: create, one ssh call for sockets, NUMA nodes, power limit against default and memory.max, `provision --models ,` for `cuda: ok` only when those pass, destroy), and pair only machines on the pass list (`phase6_pair_finder.py --pass-only`). Read-only market reads between probes; every rent from the driver's own call. Times are local (UTC+1).
- Step 1: host 45726 (machine 7543, passed the gates on 20261126c) was in no RTX 3090 search at 16:5x (1, 2 and 4 GPUs, disk 50 and 230 GB, reliability down to 0.5; 24 to 33 rows, under the limit): not listed, so no static rule applies to it.

## Probes

| Time | Host / machine | GPU, CPU | Box | Up | Result | List |
|---|---|---|---|---|---|---|
| 17:07 | 79466 / 54637 | 3090, Ryzen 9 3900X | none | | offer not found by create (3 searches) | |
| 17:15 | 440607 / 139376 | 3090, EPYC 7742 | none | | offer not found by create | |
| 17:16 | 440607 / 139376 | 3090, EPYC 7742 | 62882398 | 15.2 min | stall: still loading when create gave up (one stall, no host property) | |
| 17:36, 17:39 | 79466 / 54637 | 3090, Ryzen 9 3900X | none | | offer not found by create | |
| 17:43, 17:47 | 543999, 440607 | | none | | offer not found by create | |
| 17:49 | 543999 / 155645 | 3090, Core i9-14900KS (32 threads) | 62885713 | 5.8 min | PASS: 1 socket, 1 NUMA node, 420 of 420 W, memory.max 64.40 GB, `cuda: ok` | pass (scout) |
| 17:57 | 440607 / 139376 | 3090, EPYC 7742 (128 threads) | 62886511 | 0.5 min | power 270 W against 420 W default | skip |
| 18:04 | 88516 / 104634 | 3090, Core i9-13900KF | 62887406 | 5.7 min | power 220 W against 350 W default; memory.max 63.45 GB | skip |
| 18:12 | 79466 / 54637 | 3090, Ryzen 9 3900X | 62888226 | 10.2 min | stall: no SSH within the 10 min rule (one stall; host 79466's machine 41880 passed every gate) | |
| 18:29 | 710540 / 155008 | L40S, Xeon SapphireRapids (48 threads, two-socket hint) | 62890242 | 0.8 min | 2 sockets, 2 NUMA nodes (power 350 of 350 W) | skip |
| 18:33 | 22965 / 43388 | 3090, 48 threads (two-socket hint) | 62890751 | 10.0 min | stall: no SSH within 10 min, its second (attempt i) | skip |

| 18:48 | 351328 / 48881 | 3090, EPYC 7B12 | none | | offer not found by create, also not by machine id (gone) | |

Result files: `scout-<host>-<machine>.json` in this directory. Search ended 19:04, before the 3 h bound: Claude Code stopped the read-only market watch because the local machine was critically low on memory, and a judged session needs a driver that is not reaped (docs/box-driver.md "Stage by stage"). The last read (19:04) listed no pair of two pass-list machines.

## Findings

- Offer ids the finder lists were often not found by `create --offer`: one Vast search returns a varying subset of the market (the create's and the finder's searches seconds apart listed 22 and 24 machines with 10 in common), and a 50 GB search missed machine 54637 three times where 230 GB searches found it. Fixed: `sc_offers.sh` searches at the box's 230 GB disk (`--min-disk`), and the scout and `phase6_pair_create.sh` record a machine the offers listing missed in taxameter's offer map, so `create` follows it with a machine-id search (same filters and caps). After the fix six of seven creates rented; the seventh machine (351328) was not found by a machine-id search either, so it had left the market.
- The offer's RAM under-reads memory.max by about 0.65% (79466 machine 41880: 128.491 against 129.3 GB; 543999: 63.977 against 64.40). Machine 155645's offer (63.977 GB) carried the "RAM under 64 GB" flag that `phase6_pair_create.sh` drops; for a scouted machine the measured memory.max now decides (advisor, delegated by the owner 2026-11-27; `MEASURED_MEMORY` in the finder); `sc_balloon.sh start` still refuses a box whose memory.max is not above 64 GB before any download.
- Of the seven rented boxes, three never opened SSH within 10 min (440607, 79466 machine 54637, 22965), three failed a host property (power cap twice, two sockets once) and one passed. The two-socket hint flagged the one two-socket host (710540) and 22965, which never came up.

## Spend and lists

- Spend: about 0.33 USD of rent (seven boxes, 48 box-minutes at 0.33 to 0.52 USD/h, no traffic billed), within the 10 probes or 0.60 USD bound; the Vast credit read 14.86 USD at 16:51 and 14.68 USD at 19:04 (it lags). `taxameter status` at 19:04: no instances.
- Pass list: + host 543999 machine 155645 (scout); the list is now 79466/41880 and 45726/7543 (gates) and 543999/155645 (scout). Skip list: + 440607, 88516, 710540, 22965.
- Open: a pair needs two pass-list machines on offer at once; 45726 was not on offer, and 543999 and 79466 machine 41880 were never listed together during the search.
