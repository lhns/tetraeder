# Session phase6-exit-f-20261125

- Status: pass: every rule holds in every mode (`verdict-f.json`, `scripts/phase6_exit_f_summary.py`; ADR 0091 Case F, judged). Nine valid trials, one invalid attempt repeated (worker attempt 3, no request in flight on B), no manual action.
- Rule: ADR 0091 Case F as the summary applies it, with the 2026-11-25 fault timing and invalid-attempt rules (`docs/notes/phase6-rule-changes.md`). No speed is judged; no paired ratio (`docs/features.md` unchanged).
- Code: branch `bench-phase6-exit-f` on main cbed00ef (scripts unchanged, no Rust change); the boxes ran that commit.
- Boxes (one RTX 3090 each; power gate at default, idle gate ok, one NUMA node, co-tenant gate ok, `cuda: ok`, both AMD Ryzen 5 5600X, 12 threads, cgroup quota 11.52):
  - A (head, node `a`): box 62561949, host 264868 (Missouri offer, 0.205 USD/h), driver 595.99.02, 350 W = default, memory limit 32.2 GB. Created 19:32:58, destroyed 20:52:36 (1 h 19 min).
  - B (node `b`): box 62560750, host 306949 (Illinois offer, 0.260 USD/h), driver 615.71.09, 420 W = default, memory limit 64.6 GB. Created about 19:22, destroyed 20:52:44 (1 h 30 min).
  - Pair: no pair under 5 ms was found (the two hosts connect in 24.8 to 27.3 ms, ten TCP connects from A to B's mapped port 7001), so a WAN-class pair. Clock of B against A: offset 0.004 to 0.010 s, worst uncertainty 0.0248 s over the 10 measurements (direct exchange with the trigger, 20 samples each; rule limit 0.5 s). Fault time uncertainty 0.033 to 0.037 s per attempt.
  - Model: Qwen3-30B-A3B Q4_K_M, sha256 0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5 on both boxes.
  - Cost: account credit 19.56 to 18.00 USD for the whole session including the aborted boxes and traffic (1.56 USD). The two session boxes: about 0.27 and 0.39 USD of rent (price x age) plus about 19 GB of download each; `taxameter destroy` printed no `inet_down_billed`. Both within 0.90 USD/h all-in.
- Files: `gates/`, `certs/`, `cluster/`, `trials/<mode>-<attempt>/`, `head-requests.jsonl.gz`, `a-logs.tgz`, `b-logs.tgz`, `session.json`, `verdict-f.json`.

## Result

| mode | trial (attempt) | in flight | errors | worst error delay (s) | rejoin after restore (s) | rules |
|---|---|---|---|---|---|---|
| 1 kill of the replica engine | 1 (1) | 8 | 4 | 0.487 | 2.67 | all ok |
| 1 | 2 (2) | 8 | 4 | 0.668 | 2.67 | all ok |
| 1 | 3 (4) | 8 | 4 | 0.614 | 2.78 | all ok |
| 2 kill of the daemon and children | 1 | 8 | 4 | 0.043 | 3.58 | all ok |
| 2 | 2 | 8 | 4 | 0.045 | 3.74 | all ok |
| 2 | 3 | 8 | 4 | 0.048 | 3.82 | all ok |
| 3 SIGSTOP, SIGCONT after 60 s | 1 | 8 | 4 | 1.728 | 0.38 | all ok |
| 3 | 2 | 8 | 4 | 2.093 | 0.32 | all ok |
| 3 | 3 | 8 | 4 | 1.787 | 0.31 | all ok |

Rules per trial, each from its source in `verdict-f.json`: `fault_evidence` (B's own log, 3 requests in flight on B at every valid fault), `clean_error_in_bound` (every request in flight on R_B ended with one error within the 3 s bound: worst 2.093 s, mode 3 trial 2), `no_failure_on_r_a`, `ok_after_bound`, `queued_move`, `exactly_once` (client and gateway counts equal in every trial), `rejoin` (within 60 s of the restore: worst 3.82 s, mode 2 trial 3), `b_served_before`, `trial1_no_manual` (no operator action in the session). Session rules gates_a, gates_b, same_model, certificates_pinned, cluster_toml_after_create, load_shape (concurrency 8, 180 s, 512-token prompt, 256 output tokens, streaming), clock_offset, repeats_within_limit: all ok.

## Deviations and notes

- Invalid attempt: worker attempt 3 (no request in flight on B at the fault, `live_b` 0), repeated once as the rules allow (`MAX_REPEATS` 2); kept in `trials/worker-3/` and listed in `verdict-f.json` `attempts`. Judged worker trials are attempts 1, 2 and 4.
- Boxes that did not qualify (destroyed; the gates run before any download, so nothing was downloaded; about 0.35 USD of rent): host 130779 (co-tenant 5.2 cores against a limit of 4.8) with host 3497 (two NUMA nodes, co-tenant 25 cores); host 8725 (two NUMA nodes) with host 594304 (power limit below default); host 519702 (power limit below default). Host 135723 stalled in image pull twice (over 10 minutes each) and was destroyed per the stall rule. Two offers had gone at `create` (no cost).
- Pair is WAN-class (about 25 ms), not the preferred under 5 ms pair; no clean low-RTT pair was on offer in the time.
- `setsid` is absent on the Windows driver, so the script was started with `nohup` only (the script needs it only for the loopback).
- No script change; no manual action recorded (`manual.jsonl` absent).
