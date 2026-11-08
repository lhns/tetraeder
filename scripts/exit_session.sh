#!/usr/bin/env bash
# Track A box session of 2026-11-03 (ADR 0045 follow-ups), in order of value: GPU logit checks on the synthetic
# fixtures, the 235B logit drift at 2K, the prefetch levers paired at 32K, a second KTransformers 32K pair, Track E's
# speculative pair (spec_run.sh main-builder, with ADR 0044 in place), ik_llama.cpp last. Needs fetch_exit_model.sh
# and fetch_spec_models.sh started first. Started detached by `taxameter run scripts/exit_session.sh`; ends with
# EXIT_SESSION_DONE in runs/vast/exit_session.log. OT_FIT and VRAM_SLOT_BYTES come from the launch env (the box class
# was fitted before: 7 layers, 10966007808 B).
exec 8> /root/tetraeder_exit_session.lock; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs/vast
log=runs/vast/exit_session.log
step() {
  echo "[$(date +%T)] start $1" >> "$log"
  shift
  "$@" 8>&-
  echo "[$(date +%T)] done rc=$?" >> "$log"
}
# prefetch on explicitly (the 2026-11-03 run measured these with it off); cpu_poll 0 cost 90 ms/token, dropped
levers="d2:TETRAEDER_EXIT_PREFETCH=1,TETRAEDER_EXIT_PREFETCH_DEPTH=2"
# one compute thread fewer while a prefetch read is in flight
fewer=$(($(bash scripts/cpu_quota.sh) - 1))
levers="$levers d2fewer:TETRAEDER_EXIT_PREFETCH=1,TETRAEDER_EXIT_PREFETCH_DEPTH=2,TETRAEDER_EXIT_PREFETCH_THREADS=$fewer"
step exit_run env STAGES="gpu-tests smoke drift-ref drift-off drift-on drift-small drift-nooffload drift-nooffload-on drift-nofuse drift-ref-ub512 drift-ref-nofuse main-prefetch d2 d2fewer llamacpp-cache ktransformers main-repeat llamacpp-cache-repeat" \
  VARIANTS="$levers" bash scripts/exit_run.sh
step spec_run env STAGES=main-builder bash scripts/spec_run.sh
# exit_run.sh takes a new stamp and keeps the earlier results
step ik env STAGES=exit-ik-ot-fit-d32k bash scripts/exit_run.sh
echo "EXIT_SESSION_DONE" >> "$log"
