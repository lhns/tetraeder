# Sourced by exit_run.sh. vramfill_bytes FLOOR TOTAL_MIB USED_MIB: the expert VRAM slot bytes that fill the card.
#
# Free VRAM (total - used by others) minus a reserve for dense weights, KV and prefill compute buffers
# (VRAMFILL_RESERVE, default 22.5 GB: on a 49140 MiB card that is the 29 GB of slots that peaked at 39.6 GB, ADR 0045
# 2026-11-14). Never below FLOOR (the matched slots). VRAMFILL_EXTRA bytes, if set, override: FLOOR + EXTRA.
vramfill_bytes() {
  local floor=$1 total_mib=${2:-0} used_mib=${3:-0}
  if [ -n "${VRAMFILL_EXTRA:-}" ]; then
    echo $((floor + VRAMFILL_EXTRA))
    return
  fi
  local fill=$(((total_mib - used_mib) * 1048576 - ${VRAMFILL_RESERVE:-22500000000}))
  if [ "$fill" -lt "$floor" ]; then fill=$floor; fi
  echo "$fill"
}

# vramfill_from_smi FLOOR: the same, read from nvidia-smi (GPU 0); the floor when it is unavailable.
vramfill_from_smi() {
  local t u
  read -r t u < <(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i 0 2> /dev/null | tr -d ',') || true
  vramfill_bytes "$1" "${t:-0}" "${u:-0}"
}
