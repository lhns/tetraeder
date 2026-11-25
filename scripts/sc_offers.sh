#!/usr/bin/env bash
# Driver side of the segment confirmation (AGENTS.md "Money", owner 2026-11-24): traffic counts toward the per-box cap, so
# `offers` and `create` always carry the real download size. Never type --download-gb by hand for the 235B.
#   bash scripts/sc_offers.sh offers [taxameter args]     rank offers with the traffic priced in
#   bash scripts/sc_offers.sh create --offer ID [args]    create with the same flags
# Env: DOWNLOAD_GB (150: the 142 GB model plus the build), MAX_PRICE (0.90 USD/h all-in), DRY_RUN=1 prints the command.
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
set -- uv run taxameter "$cmd" --download-gb "$gb" --max-price "${MAX_PRICE:-0.90}" "$@"
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "$*"
  exit 0
fi
exec "$@"
