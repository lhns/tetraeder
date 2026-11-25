#!/usr/bin/env bash
# Driver side of the segment confirmation (AGENTS.md "Money", owner 2026-11-24): traffic counts toward the per-box cap, so
# `offers` and `create` always carry the real download size. Never type --download-gb by hand for the 235B.
#   bash scripts/sc_offers.sh offers [taxameter args]     rank offers with the traffic priced in
#   bash scripts/sc_offers.sh create --offer ID [args]    create with the same flags
# Env: DOWNLOAD_GB (150: the 142 GB model plus the build), MAX_PRICE (0.90 USD/h all-in), DISK_GB (230: the box disk; taxameter
# has no create option for it, it reads `disk_gb` from .vast/config.json, which this script writes; the default 80 GB of
# taxameter.toml filled up in segment-confirm-20261125b), DRY_RUN=1 prints the command and writes nothing.
# The first line the driver prints for a box is `date +%T`; `provision` and the session follow docs/box-driver.md "Flow".
set -u
cmd="${1:-}"
case "$cmd" in
  offers | create) shift ;;
  *) echo "usage: sc_offers.sh offers|create [taxameter args]" >&2; exit 2 ;;
esac
gb="${DOWNLOAD_GB:-150}"
case "$gb" in '' | *[!0-9]*) echo "DOWNLOAD_GB must be a whole number of GB, got '$gb'" >&2; exit 2 ;; esac
[ "$gb" -ge 100 ] || { echo "DOWNLOAD_GB $gb is below the 142 GB model: the cap would not price the traffic" >&2; exit 2; }
for a in "$@"; do
  case "$a" in --download-gb | --download-gb=*) echo "do not pass --download-gb, DOWNLOAD_GB sets it" >&2; exit 2 ;; esac
done
disk="${DISK_GB:-230}"
case "$disk" in '' | *[!0-9]*) echo "DISK_GB must be a whole number of GB, got '$disk'" >&2; exit 2 ;; esac
[ "$disk" -ge 230 ] || { echo "DISK_GB $disk is below the 230 GB the 235B needs" >&2; exit 2; }
set -- uv run taxameter "$cmd" --download-gb "$gb" --max-price "${MAX_PRICE:-0.90}" "$@"
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "$*"
  echo "would set disk_gb $disk in ${VAST_STATE_DIR:-.vast}/config.json" >&2
  exit 0
fi
mkdir -p "${VAST_STATE_DIR:-.vast}"
python3 -I -c '
import json, os, sys
p = sys.argv[1]
try:
    c = json.load(open(p, encoding="utf-8"))
except (OSError, ValueError):
    c = {}
c["disk_gb"] = int(sys.argv[2])
json.dump(c, open(p, "w", encoding="utf-8"), indent=2)
print("disk_gb", c["disk_gb"], "in", p, file=sys.stderr)
' "${VAST_STATE_DIR:-.vast}/config.json" "$disk" || exit 1
exec "$@"
