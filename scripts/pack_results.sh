#!/usr/bin/env bash
# Pack only the small result files of the given directories into one archive (run on the GPU box), so a flaky SSH
# connection only has to stay up for one short copy.
#   bash scripts/pack_results.sh bench/results runs/vast bench/golden   -> /root/results.tgz
# Kept: JSON/JSONL results, DONE markers, run logs; everything under bench/golden (greedy golden sequences).
# Dropped: *.gguf and anything under target/.
set -euo pipefail
cd "$(dirname "$0")/.."
out="${OUT:-/root/results.tgz}"
list="$(mktemp)"
trap 'rm -f "$list"' EXIT
for d in "$@"; do
  case "$d" in
    bench/golden*) find "$d" -type f ;;
    *) find "$d" -type f \( -name '*.json' -o -name '*.jsonl' -o -name 'DONE' -o -name '*.log' \) ;;
  esac
done | { grep -v -e '\.gguf$' -e '/target/' || true; } | sort -u > "$list"
tar -czf "$out" -T "$list"
echo "$(wc -l < "$list") files, $(du -h "$out" | cut -f1) -> $out"
