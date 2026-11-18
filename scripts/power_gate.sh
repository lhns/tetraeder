#!/usr/bin/env bash
# Power-limit gate at SSH and in phase5_box.sh's gpus stage (ADR 0068, "Power limit"): exit 0 only if every GPU's
# power.limit equals its power.default_limit within 1 W. Read-only. Prints one line per GPU. NVIDIA_SMI overrides the
# binary (tests).
smi="${NVIDIA_SMI:-nvidia-smi}"
v=$("$smi" --query-gpu=index,power.limit,power.default_limit,power.max_limit --format=csv,noheader,nounits 2> /dev/null) || v=""
printf '%s\n' "$v" | awk -F, '
  function num(s) { gsub(/[ W]/, "", s); return s }
  NF >= 4 {
    n++; l = num($2); d = num($3); m = num($4)
    if (l !~ /^[0-9.]+$/ || d !~ /^[0-9.]+$/) { bad++; st = "BAD"; }
    else if (l - d > 1 || d - l > 1) { bad++; st = "LIMITED" } else st = "ok"
    printf "gpu %s: limit %s W, default %s W, max %s W %s\n", num($1), l, d, m, st
  }
  END {
    if (n == 0) { print "power gate: no GPU values from nvidia-smi"; exit 1 }
    if (bad) { printf "power gate: %d of %d GPUs not at default power limit\n", bad, n; exit 1 }
    printf "power gate: %d GPUs at default power limit\n", n
  }'
