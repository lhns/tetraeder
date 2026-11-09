#!/usr/bin/env bash
# Applies the ggml patch queue unless vendor/llama.cpp already has every patch of patches/series
# (rewriting the patched files would rebuild ggml on every box script). Run before cargo build/test.
set -eu
root="$(cd "$(dirname "$0")/.." && pwd)"
sub="$root/vendor/llama.cpp"
ok=0
if [ -f "$sub/CMakeLists.txt" ]; then
  ok=1
  while IFS= read -r line || [ -n "$line" ]; do
    p="${line%%#*}"; p="$(echo "$p" | xargs)"
    [ -z "$p" ] && continue
    git -C "$sub" apply --reverse --check "$root/patches/$p" 2>/dev/null || { ok=0; break; }
  done < "$root/patches/series"
fi
if [ "$ok" = 1 ]; then
  echo "patches: already applied"
else
  bash "$root/patches/apply.sh"
fi
