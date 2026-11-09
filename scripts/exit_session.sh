#!/usr/bin/env bash
# The full stage list, far beyond one 3 h box: for box runs use the profiles of scripts/session.sh (b1-exit,
# b1b-levers; docs/sessions.md), which keep each pair on one box and fit the cap. This list stays as the reference.
# The kv-* stages (ADR 0050 b, written, not yet run) pair with main-prefetch; kv-quality needs the 30B GGUF too and is
# started on its own: STAGES=kv-quality TETRAEDER_KV_QUALITY_GGUF=... bash scripts/exit_run.sh.
# Track A box session of 2026-11-03 (ADR 0045 follow-ups), in order of value: GPU logit checks on the synthetic
# fixtures, the 235B logit drift at 2K, the prefetch levers paired at 32K, a second KTransformers 32K pair, Track E's
# speculative pair (spec_run.sh main-builder, with ADR 0044 in place), ik_llama.cpp last. Needs fetch_exit_model.sh
# and fetch_spec_models.sh started first. Started detached by `taxameter run scripts/exit_session.sh`; ends with
# EXIT_SESSION_DONE in runs/vast/exit_session.log. OT_FIT and VRAM_SLOT_BYTES come from the launch env (the box class
# was fitted before: 7 layers, 10966007808 B). main / main-native and main-repeat / main-native-repeat are the
# AVX2 against native-ISA pairs, ktransformers-avx2 the KTransformers run on AVX2 kernels (ADR 0045, "Diagnosing the
# KTransformers decode gap").
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
step exit_run env STAGES="gpu-tests smoke drift-ref drift-off drift-on drift-small drift-nooffload drift-nooffload-on drift-nofuse drift-ref-ub512 drift-ref-nofuse main main-native main-prefetch kv-q8 kv-q8-slots kv-q8q4 kv-q8q4-slots d2 d2fewer llamacpp-cache ktransformers ktransformers-avx2 main-repeat main-native-repeat llamacpp-cache-repeat" \
  VARIANTS="$levers" bash scripts/exit_run.sh
# routing trace and replay for ADR 0050 (c) (written, not yet run: box runs on hold)
step exit_trace env STAGES=trace bash scripts/exit_run.sh
step spec_run env STAGES=main-builder bash scripts/spec_run.sh
# CPU-only Q4_K kernel comparison (mainline, ik, baseline), minutes
step kernels_cpu bash scripts/kernels_cpu.sh
# exit_run.sh takes a new stamp and keeps the earlier results
step ik env STAGES=exit-ik-ot-fit-d32k bash scripts/exit_run.sh
echo "EXIT_SESSION_DONE" >> "$log"
