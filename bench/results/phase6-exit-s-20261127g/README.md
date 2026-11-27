# Session phase6-exit-s-20261127g (Case S rerun, attempt g): box b never left loading, nothing gated, built or downloaded

- Status: null, no verdict. Box b (L40S) stayed `loading` (image pull) for more than 10 min after create (docs/box-driver.md: destroy), so both boxes were destroyed before the gates, the bulk screen, the build and the 142 GB download (0 of 2 download attempts used). `scripts/phase6_exit_s.sh` was never started.
- Rule: ADR 0091 Case S, ADR 0092 rerun rules, `docs/notes/phase6-rule-changes.md`; scripts at f750a585.
- Pair (`scripts/phase6_pair_create.sh`, both creates at 15:13 local time, UTC+1, disk_gb 230, `--ports 7000`; ranked first by the gated-host preference):
  - a: box 62866939, host 79466 (Quebec, CA), RTX 3090, 0.443 USD/h all-in; ready, `provision --models ,` ended `PROVISION_DONE rc=0`, `cuda: ok` at 15:18:39; no gate run; destroyed 15:24:25.
  - b: box 62866936, host 710540 (New York, US), L40S, 0.517 USD/h all-in; `loading` from 15:13:30 to 15:23:28 and still at the deadline; destroyed 15:24:21.
- Cost: about 0.16 USD (both boxes about 11 min, rent only); the Vast credit read 15.34 USD at 15:17 and 15.28 USD at 15:24, it may lag.
- Consequence: no host list changes; one pull stall is not a host property (docs/box-driver.md "On surprises").
