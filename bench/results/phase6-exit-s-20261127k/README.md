# Session phase6-exit-s-20261127k (Case S rerun, attempt k): no pair, no box rented, no verdict

- Status: null, no verdict. No pair of two pass-list machines was listed from 21:59 to 22:44, no scout box was rented (the one scout create found no offer), and the search stopped at 22:45 when Claude Code reaped the driver's background timer shell for low local memory (brief: a reaped shell is not restarted). `scripts/phase6_exit_s.sh` never ran (0 of 2 download attempts used).
- Rule: ADR 0091 Case S, ADR 0092 rerun rules, `docs/notes/phase6-rule-changes.md`; scripts at main e92ba090.
- Approach: as attempt j. Poll `PAIR_FINDER_ARGS=--pass-only bash scripts/phase6_pair_create.sh` (or the read-only finder while a scout box is up) every 3 to 5 min; scout one static-passing machine at a time. Times are local (UTC+1). Credit 14.68 USD at 21:59 on 2026-11-27, no instances.

## Polls and probes

| Date, time | Action | Result |
|---|---|---|
| 11-27 21:59 | finder `--pass-only` | 32 offers, 1 static pass, no pair; 0 scout candidates |
| 11-27 22:02 | `phase6_pair_create.sh` (`--pass-only`) | no pair, nothing created |
| 11-27 22:06 | finder `--pass-only --scout`, three reads | no pair; scouts 2, 1 (623333 machine 151579, Xeon E5-2673 v4, 80 threads, two-socket hint), 0; no pass-list machine (79466/41880, 45726/7543, 543999/155645) in a 29-offer read |
| 11-27 22:11 | finder, three reads | no pair, 0 scouts, no pass-list machine on offer (33 offers) |
| 11-27 22:16 | finder | 79466/41880 on offer (the one static pass); no partner, 0 scouts |
| 11-27 22:21 to 22:26 | finder, two polls | 79466/41880 the only static pass; no pair, 0 scouts |
| 11-27 22:30 | finder | 2 static passes (79466/41880, 623333/151579), not both on the pass list; scout 623333 (two-socket hint) |
| 11-27 22:31 | scout probe 623333/151579 | finder pick found 0 candidates; by offer id: create found no offer on its machine, nothing rented |
| 11-27 22:35 to 22:44 | finder, three polls | 79466/41880 the only static pass (once also 623333); no pair, 0 scouts |

Result files: `scout-623333-151579.json` (no box rented), `create-623333.log`, `scouts.json` (the scout pick's finder read).

## Findings

- The pass list rarely meets the market at night: 79466 machine 41880 was on offer in every poll from 22:16, 543999 machine 155645 and 45726 machine 7543 in none (reads of 29 to 37 offers). At most two offers passed the static rules in any read.
- The only scout candidate of the window, 623333 machine 151579 (Xeon E5-2673 v4, 80 host threads, two-socket hint), appeared in two of about nine reads; `create --offer` did not find it in three searches, also not by machine id, so nothing was rented.
- The finder's `--scout` mode prints only the scout list, not the pairs; the polls ran the pair read (`--pass-only`) and the scout read separately.
- The local memory reaper stopped a 4 min `sleep` plus read-only poll at 22:45, idle session, no box up. A judged Case S session needs local tool calls of up to an hour (`fetch`, `judged`) that would be reaped the same way: the driver machine's memory, not the market alone, blocks the session.

## Spend and lists

- Spend: 0 USD (no box rented). Credit 14.68 USD at 21:59 and 22:45; `taxameter status` at 22:45: no instances.
- Pass list and skip list unchanged.
- Open: a pair needs 543999/155645 or 45726/7543 (or a new scouted machine) on offer together with 79466/41880; and a driver machine with enough free memory that long stage calls are not reaped (or `CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1`, the user's call).
