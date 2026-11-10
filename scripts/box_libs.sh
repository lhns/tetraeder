# shellcheck shell=bash
# Sourced by box scripts after `cd` to the repo root and the PATH export (shellcheck source=scripts/box_libs.sh).
# 1. Boxes build the static native ggml (ADR 0052 addendum): set TETRAEDER_GGML_NATIVE=ON explicitly, because the
#    Docker ENV and .cargo/config.toml [env] did not reliably reach a `taxameter run` shell (b3-detector, 2026-11-10,
#    got a DL build). A caller's value wins.
# 2. Binaries run directly (test binaries from --no-run, target/release/*) need the shared ggml libs when the build is
#    DL: build.rs copies them to target/<profile>/ and the rpath is $ORIGIN:$ORIGIN/.., but that is lost whenever
#    RUSTFLAGS is set. LD_LIBRARY_PATH is the safety net. Call box_libs_path again after a build: the ggml-sys out dir
#    only exists then.
export TETRAEDER_GGML_NATIVE="${TETRAEDER_GGML_NATIVE:-ON}"
box_libs_path() {
  local root d p=""
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  for d in "$root"/target/release "$root"/target/debug "$root"/target/*/build/tetraeder-ggml-sys-*/out/lib \
    "$root"/target/*/build/tetraeder-ggml-sys-*/out/lib64 "$root"/target/*/build/tetraeder-ggml-sys-*/out/bin; do
    [ -d "$d" ] && p="$p:$d"
  done
  export LD_LIBRARY_PATH="${p#:}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}
# box_ggml_mode: "native" (static) or "dl" (shared libllama next to the binaries), for the logs
box_ggml_mode() {
  local root
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  if compgen -G "$root/target/release/libllama.so*" > /dev/null; then echo dl; else echo native; fi
}
box_libs_path
