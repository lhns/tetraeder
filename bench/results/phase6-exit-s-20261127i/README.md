# Session phase6-exit-s-20261127i (Case S rerun, attempt i): box a never opened SSH, nothing gated, built or downloaded

- Status: null, no verdict. Box a was `running` from about 15:50 but its direct and proxy SSH ports stayed down past 10 min after create (docs/box-driver.md: destroy), so both boxes were destroyed before the gates, the bulk screen, the build and the 142 GB download (0 of 2 download attempts used). `scripts/phase6_exit_s.sh` was never started.
- Rule: ADR 0091 Case S, ADR 0092 rerun rules, `docs/notes/phase6-rule-changes.md`; scripts at the attempt h commit.
- Pair (`scripts/phase6_pair_create.sh`, both creates at 15:44 local time, UTC+1, disk_gb 230, `--ports 7000`; listed first by the gated-host preference, expected bulk rate 212 Mbit/s):
  - a: box 62870946, host 22965, RTX 3090, 0.304 USD/h before traffic; created 15:44:01, SSH down at 15:54:09; destroyed 15:54:39.
  - b: box 62870950, host 79466 (Quebec, CA), RTX 3090, 0.443 USD/h all-in; ready 15:44:39, `provision --models ,` with `cuda: ok` at 15:47:46; no gate run; destroyed 15:54:45.
- Cost: about 0.12 USD (both boxes about 10.5 min, rent only); the Vast credit read 15.11 USD at 15:43 and 15.00 USD at 15:54, it may lag.
- Consequence: no host list changes; one SSH stall is not a host property (docs/box-driver.md "On surprises").
