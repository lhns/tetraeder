#!/usr/bin/env bash
# Run cargo (or any command) for a checkout on the docker host instead of this machine (owner, 2026-11-26: the docker
# command reaches a remote engine with far more cores and RAM). The engine is remote, so nothing is bind-mounted: the
# checkout's tracked files (submodules included, plus uncommitted edits) are streamed in as a tar; target/ and the
# cargo registry live in named volumes per lane, so builds stay incremental. Linux only (the Windows half stays in CI).
# Usage: scripts/docker_cargo.sh [-l LANE] -- <command...>     e.g. scripts/docker_cargo.sh -l kv -- cargo test -p tetraeder-control
# Env: DOCKER_CARGO_JOBS (default 8), DOCKER_CARGO_MEM (default 24g). Containers and volumes are named tetraeder-*.
set -euo pipefail
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*" # Git Bash on Windows must not rewrite /src, /target into Windows paths
lane=main
while [ $# -gt 0 ]; do
  case $1 in
    -l) lane=$2; shift 2 ;;
    --) shift; break ;;
    *) break ;;
  esac
done
[ $# -gt 0 ] || { echo "usage: $0 [-l LANE] -- <command...>" >&2; exit 2; }
image=tetraeder-dev:1.95
root=$(git rev-parse --show-toplevel)
if ! docker image inspect "$image" > /dev/null 2>&1; then
  docker build -q -t "$image" - << 'EOF' > /dev/null
FROM rust:1.95.0-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libclang-dev cmake ninja-build python3 python3-pip git && rm -rf /var/lib/apt/lists/*
RUN rustup component add rustfmt clippy
EOF
fi
name="tetraeder-$lane-$$"
trap 'docker rm -f "$name" > /dev/null 2>&1 || true' EXIT
docker create --name "$name" -w /src --cpus 16 --memory "${DOCKER_CARGO_MEM:-24g}" \
  -e CARGO_TARGET_DIR=/target -e CARGO_BUILD_JOBS="${DOCKER_CARGO_JOBS:-8}" -e CARGO_HOME=/cargo \
  -v "tetraeder-target-$lane:/target" -v "tetraeder-vendor-$lane:/src/vendor/llama.cpp" -v tetraeder-cargo:/cargo "$image" sleep infinity > /dev/null
docker start "$name" > /dev/null
# tracked files of the checkout as they are on disk (uncommitted edits included, deleted files skipped); vendor/llama.cpp is not
# copied: it lives in a volume, where patches/apply.sh fetches the pinned commit once and applies the patch queue
(cd "$root" && git ls-files -z | while IFS= read -r -d '' f; do [ -f "$f" ] && printf '%s\0' "$f"; done \
  | tar --null -T - -cf -) | docker cp - "$name:/src"
docker exec "$name" sh -c 'cd /src && bash patches/apply.sh > /dev/null'
docker exec -w /src "$name" "$@"
