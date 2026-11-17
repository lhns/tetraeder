#!/usr/bin/env bash
# Idle gate at SSH (ADR 0065, addendum): 10 samples of the clock event reasons over 30 s; exit 0 only if at least 9
# read within 0x0-0x3 (every GPU of a sample). Prints the samples. INTERVAL (s, default 3) and SAMPLES (default 10)
# exist for tests.
n="${SAMPLES:-10}" step="${INTERVAL:-3}" ok=0
for i in $(seq 1 "$n"); do
  v=$(nvidia-smi --query-gpu=clocks_event_reasons.active --format=csv,noheader 2> /dev/null) \
    || v=$(nvidia-smi --query-gpu=clocks_throttle_reasons.active --format=csv,noheader 2> /dev/null)
  good=1
  [ -n "$v" ] || good=0
  while read -r x; do
    [ -z "$x" ] && continue
    if ! [[ $x =~ ^0[xX][0-9a-fA-F]+$ ]] || [ $((x & ~0x3)) -ne 0 ]; then good=0; fi
  done <<< "$v"
  [ "$good" = 1 ] && ok=$((ok + 1))
  echo "sample $i: ${v//$'\n'/ } $([ "$good" = 1 ] && echo ok || echo BAD)"
  [ "$i" -lt "$n" ] && sleep "$step"
done
need=$((n * 9 / 10))
echo "idle gate: $ok/$n within 0x0-0x3 (need $need)"
[ "$ok" -ge "$need" ]
