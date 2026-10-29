#!/usr/bin/env bash
# Reset vendor/llama.cpp to the pin (or $PIN_REF) and apply patches/series in order. Idempotent.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
sub="$root/vendor/llama.cpp"
ref="${PIN_REF:-$(sed -n 's/^commit=//p' "$root/vendor/PIN")}"

git -C "$root" submodule update --init --depth 1 vendor/llama.cpp
if ! git -C "$sub" cat-file -e "$ref^{commit}" 2>/dev/null; then
  git -C "$sub" fetch --depth 1 origin "$ref"
  ref="FETCH_HEAD"
fi
git -C "$sub" checkout -q --detach "$ref"
git -C "$sub" reset -q --hard
git -C "$sub" clean -qfdx

while IFS= read -r line || [ -n "$line" ]; do
  p="${line%%#*}"; p="$(echo "$p" | xargs)"
  [ -z "$p" ] && continue
  echo "apply $p"
  git -C "$sub" apply --index "$root/patches/$p"
done < "$root/patches/series"
echo "ok: $(git -C "$sub" rev-parse --short HEAD) + $(grep -cvE '^\s*(#|$)' "$root/patches/series" || true) patches"
