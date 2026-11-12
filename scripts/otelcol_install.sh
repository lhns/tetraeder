#!/usr/bin/env bash
# Installs the OpenTelemetry Collector (contrib distribution) for the box runs and CI (ADR 0060; docs/sources.md).
#   scripts/otelcol_install.sh [dir]   (default /opt/otelcol; the binary is <dir>/otelcol-contrib)
# Fetches the pinned release tarball for linux amd64 and checks its sha256 before unpacking. Does nothing when
# <dir>/PIN names that tarball (the 0.116.1 binary reports itself as 0.116.0). Same version as the compose image
# (deploy/docker-compose.yml).
set -euo pipefail
OTELCOL_VERSION=0.116.1
OTELCOL_SHA256=7dbe6063bb158910beecb4699688401dd42241202e27a0a8c7a111852f3ed78b
dir="${1:-/opt/otelcol}"
bin="$dir/otelcol-contrib"
if [ -x "$bin" ] && grep -qx "$OTELCOL_VERSION $OTELCOL_SHA256" "$dir/PIN" 2> /dev/null; then
  echo "otelcol-contrib $OTELCOL_VERSION already in $dir"
  exit 0
fi
[ "$(uname -m)" = x86_64 ] || { echo "otelcol_install.sh: only linux amd64 is pinned, not $(uname -m)" >&2; exit 1; }
name="otelcol-contrib_${OTELCOL_VERSION}_linux_amd64.tar.gz"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fsSL --retry 3 -o "$tmp/$name" \
  "https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v${OTELCOL_VERSION}/$name"
echo "$OTELCOL_SHA256  $tmp/$name" | sha256sum -c --quiet -
mkdir -p "$dir"
tar -xzf "$tmp/$name" -C "$tmp" otelcol-contrib
install -m 0755 "$tmp/otelcol-contrib" "$bin"
echo "$OTELCOL_VERSION $OTELCOL_SHA256" > "$dir/PIN"
echo "otelcol-contrib $OTELCOL_VERSION installed in $dir"
