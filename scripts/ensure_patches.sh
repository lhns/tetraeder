#!/usr/bin/env bash
# Applies the ggml patch queue unless vendor/llama.cpp already has every patch of patches/series
# (rewriting the patched files would rebuild ggml on every box script). Run before cargo build/test.
set -eu
root="$(cd "$(dirname "$0")/.." && pwd)"
sub="$root/vendor/llama.cpp"
ok=0
# Only a vendor/llama.cpp that is its own git repo can be checked (a box checkout has none, or a pushed copy with a
# dangling .git and maybe CRLF files); otherwise patches/apply.sh builds a fresh repo at the pin from upstream.
if [ -f "$sub/CMakeLists.txt" ] && pre="$(git -C "$sub" rev-parse --show-prefix 2>/dev/null)" && [ -z "$pre" ]; then
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
