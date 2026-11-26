#!/usr/bin/env bash
# Report-only runs of Case S on node a (ADR 0091 answer 9, History 2026-11-26), driven by scripts/phase6_exit_s.sh through
# `phase6_box.sh launch`. They never feed the verdict. Each is one whole-model streamed greedy run of the exit_regime test (sc_box.sh stage
# `trace`, the same 64 GB budget by balloon, the 235B, no head), with a results name and a run dir of its own so the frozen trace and the
# frozen prompt copy are never touched:
#   second-trace   the routing trace of a second prompt (bench/prompts/second_prompt.txt), recorded after the freeze and before the judged
#                  runs; the driver prices it after the summary
#   reference      the single-node streamed reference of the judged prompt; ROUND_FILE gets its result (exit_regime shape)
# Env: RN (results name stem, required), P6 (work dir), HIDDEN_AT (the cut, for the trace's hidden rows), EMULATE_RAM_GB (64), ROUND_FILE (reference).
# Ends with `REPORT_DONE rc=N stage=...` in the launch log.
cd "$(dirname "$0")/.." || exit 1
stage="${1:?second-trace|reference}"
rn="${RN:?set RN}"
P6="${P6:-/root/p6}"
run="$P6/rep-$stage"
mkdir -p "$run"
case "$stage" in
  second-trace) extra=(PROMPT_FILE=bench/prompts/second_prompt.txt) ;;
  reference) extra=() ;; # the judged prompt: sc_box's default
  *) echo "unknown stage $stage" >&2; echo "REPORT_DONE rc=2 stage=$stage"; exit 2 ;;
esac
env "${extra[@]}" RESULTS_NAME="$rn-$stage" STAGES=trace SC_RUN_DIR="$run" SC_LOG="$run/sc_box.log" SC_LOCK="$run/sc_box.lock" \
  SC_PROMPT_COPY="$run/prompt.txt" SPLIT=0 END=94 HIDDEN_AT="${HIDDEN_AT:-47}" EMULATE_RAM_GB="${EMULATE_RAM_GB:-64}" bash scripts/sc_box.sh
rc=0
res="bench/results/$rn-$stage/trace/whole-model.json"
[ -s "$res" ] || rc=1
if [ "$rc" = 0 ] && [ "$stage" = reference ] && [ -n "${ROUND_FILE:-}" ]; then cp "$res" "$ROUND_FILE" || rc=1; fi
echo "REPORT_DONE rc=$rc stage=$stage"
