# Session phase6-exit-s-20261127e (Case S rerun, attempt e): box a never came up, nothing gated, built or downloaded

- Status: null, no verdict. Box a stayed `loading` (image pull; the ssh proxy refused every connection) for 15 min after create, past the driver's 01:37 deadline (docs/box-driver.md: SSH not up after 10 minutes, destroy), so both boxes were destroyed before the gates, the bulk screen, the build and the 142 GB download (0 of 2 download attempts used).
- Rule: ADR 0091 Case S, ADR 0092 rerun rules, `docs/notes/phase6-rule-changes.md`; scripts at main 146a809e (`max_box_hours` 4, user one-off 2026-11-27). `scripts/phase6_exit_s.sh` was never started.
- Pair (created by `scripts/phase6_pair_create.sh` at about 01:22 local, UTC+1; that script was killed by the local memory reaper before it wrote `pair.env`, so the driver adopted both boxes):
  - a: box 62774511, host 306949, RTX 3090, 0.426 USD/h all-in; `loading` from create to destroy; two `provision --models ,` runs (01:24, 01:27) each ended "proxy ssh5.vast.ai:15742: Connection refused" after three retries; status still `loading` at 01:37:01; destroyed 01:37:22.
  - b: box 62774518, host 79466, RTX 3090, 0.443 USD/h all-in; ready, `provision --models ,` ended `PROVISION_DONE rc=0`, `cuda: ok` at 01:25:41; no gate run; destroyed 01:37:23.
- Destroy needed `--force`: taxameter's run list held the committed `bench/results` dirs; nothing of this session was on either box.
- Cost: about 0.22 USD (both boxes about 15 min at their rates, rent only, no traffic billed); the Vast credit read 15.72 USD at 01:24 and 15.57 USD at 01:37, it may lag.
- Consequence: no host list changes. One pull or SSH stall is not a host property (docs/box-driver.md "On surprises"); host 306949 ran Case F node b on 2026-11-25. Host 79466 was ready with `cuda: ok` again. The 4 h one-off carries to the next pair (AGENTS.md: it covers the next pair if this one stops before the download).
