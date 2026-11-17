#!/usr/bin/env bash
# Reset vendor/llama.cpp to the pin (or $PIN_REF) and apply patches/series in order. Idempotent.
set -euo pipefail
# No background auto gc or maintenance: after a shallow fetch it rewrites .git/shallow while the next command reads it
# ("shallow file has changed since we read it", CI 2026-11-17).
export GIT_CONFIG_COUNT=2 GIT_CONFIG_KEY_0=gc.auto GIT_CONFIG_VALUE_0=0 GIT_CONFIG_KEY_1=maintenance.auto GIT_CONFIG_VALUE_1=false
root="$(cd "$(dirname "$0")/.." && pwd)"
sub="$root/vendor/llama.cpp"
ref="${PIN_REF:-$(sed -n 's/^commit=//p' "$root/vendor/PIN")}"

# Is $sub its own git repo? A box checkout has no .git at the root (taxameter pushes files), and a pushed submodule
# has a dangling .git file and possibly CRLF files: then build a fresh repo at the pin from the upstream URL.
own_repo() { local pre; pre="$(git -C "$sub" rev-parse --show-prefix 2>/dev/null)" && [ -z "$pre" ]; }
if ! own_repo && [ -e "$root/.git" ]; then
  git -C "$root" submodule update --init --depth 1 vendor/llama.cpp || true
fi
if ! own_repo; then
  # taxameter does not push .gitmodules: fall back to upstream
  url="${PATCH_REMOTE:-$(git config -f "$root/.gitmodules" submodule.vendor/llama.cpp.url 2>/dev/null || true)}"
  url="${url:-https://github.com/ggml-org/llama.cpp}"
  echo "vendor/llama.cpp is not a git repo here: fresh repo from $url"
  mkdir -p "$sub"
  find "$sub" -mindepth 1 -delete
  git -C "$sub" init -q
  git -C "$sub" remote add origin "$url"
fi
# A detached auto gc left by the checkout's shallow submodule fetch may still be running: wait for it (at most 120 s).
gdir="$(git -C "$sub" rev-parse --absolute-git-dir 2>/dev/null || true)"
for _ in $(seq 1 120); do
  [ -n "$gdir" ] && [ -e "$gdir/gc.pid" ] || break
  sleep 1
done
if ! git -C "$sub" cat-file -e "$ref^{commit}" 2>/dev/null; then
  git -C "$sub" fetch --depth 1 origin "$ref"
  ref="FETCH_HEAD"
fi
# Patches are byte exact (ADR 0053 copies a CRLF file): no line-ending conversion in the submodule.
crlf_reset=0
if [ "$(git -C "$sub" config --get core.autocrlf || true)" != "false" ]; then
  git -C "$sub" config core.autocrlf false
  crlf_reset=1
fi
# -f: a cached tree with converted line endings or a changed pin must not block the checkout; reset and clean follow.
git -C "$sub" checkout -q -f --detach "$ref"
# Empty index: the reset below rewrites every file without conversion.
[ "$crlf_reset" = 1 ] && git -C "$sub" rm -rqf --cached .
git -C "$sub" reset -q --hard
git -C "$sub" clean -qfdx

while IFS= read -r line || [ -n "$line" ]; do
  p="${line%%#*}"; p="$(echo "$p" | xargs)"
  [ -z "$p" ] && continue
  echo "apply $p"
  git -C "$sub" apply --whitespace=nowarn --index "$root/patches/$p"
done < "$root/patches/series"
echo "ok: $(git -C "$sub" rev-parse --short HEAD) + $(grep -cvE '^\s*(#|$)' "$root/patches/series" || true) patches"
