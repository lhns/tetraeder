#!/usr/bin/env bash
# The box collector of an exit run (ADR 0060), config deploy/otelcol/config-file.yaml.
#   scripts/otelcol_box.sh start <dir>   OTLP JSON-lines files and the collector log go to <dir>; returns once the
#                                        OTLP HTTP port answers (exit 1 if it does not: the run goes on without it)
#   scripts/otelcol_box.sh stop          SIGTERM to the collector's process group, waits until it has flushed and exited
# stop also compresses the finished files with zstd (ADR 0060 addendum); readers take plain and .zst.
# One collector per box: a running one is reused, and its own lock keeps a relaunch from starting a second one.
# OTELCOL_BIN overrides the binary (default /opt/otelcol/otelcol-contrib, scripts/otelcol_install.sh), OTELCOL_STATE
# the directory of its pidfile and lock (default /root; the CI check uses a temp dir).
cd "$(dirname "$0")/.." || exit 1
bin="${OTELCOL_BIN:-/opt/otelcol/otelcol-contrib}"
pidfile="${OTELCOL_STATE:-/root}/tetraeder_otelcol.pid"
lock="${OTELCOL_STATE:-/root}/tetraeder_otelcol.lock"
dirfile="${OTELCOL_STATE:-/root}/tetraeder_otelcol.dir"
running() { [ -s "$pidfile" ] && kill -0 -- "-$(cat "$pidfile")" 2> /dev/null; }
case "${1:-}" in
  start)
    dir=${2:?usage: otelcol_box.sh start <dir>}
    if running; then
      echo "otelcol: already running (group $(cat "$pidfile"))"
      exit 0
    fi
    [ -x "$bin" ] || { echo "otelcol: $bin missing (scripts/otelcol_install.sh)"; exit 1; }
    mkdir -p "$dir"
    echo "$dir" > "$dirfile"
    # its own session, so stop can signal the collector and flock together; the session leader writes the pidfile
    # (its pid is the process group). fd 8 (the caller's lock) closed. GOMAXPROCS and GOMEMLIMIT bound it beside an
    # engine at its CPU quota. An empty provenance value would stop the resource processor from building, so empty
    # and unset become "unknown".
    rm -f "$pidfile"
    TETRAEDER_OTEL_DIR="$dir" TETRAEDER_HOST_ID="${CONTAINER_ID:-$(hostname)}" GOMAXPROCS=1 GOMEMLIMIT=200MiB \
      TETRAEDER_SESSION="${TETRAEDER_SESSION:-unknown}" TETRAEDER_GIT_COMMIT="${TETRAEDER_GIT_COMMIT:-unknown}" \
      setsid bash -c 'echo $$ > "$1"; shift; exec flock -n "$@"' _ "$pidfile" "$lock" "$bin" \
      --config=deploy/otelcol/config-file.yaml < /dev/null > "$dir/otelcol.log" 2>&1 8>&- &
    for _ in $(seq 1 30); do
      sleep 1
      if curl -fs -o /dev/null -X POST -H 'Content-Type: application/json' -d '{}' http://127.0.0.1:4318/v1/metrics; then
        echo "otelcol: up, files in $dir"
        exit 0
      fi
      running || break
    done
    echo "otelcol: did not start, see $dir/otelcol.log"
    tail -5 "$dir/otelcol.log"
    running && kill -TERM -- "-$(cat "$pidfile")"
    rm -f "$pidfile"
    exit 1
    ;;
  stop)
    running || { rm -f "$pidfile"; exit 0; }
    pg=$(cat "$pidfile")
    kill -TERM -- "-$pg" 2> /dev/null
    for _ in $(seq 1 30); do
      kill -0 -- "-$pg" 2> /dev/null || break
      sleep 1
    done
    kill -0 -- "-$pg" 2> /dev/null && kill -KILL -- "-$pg"
    rm -f "$pidfile"
    echo "otelcol: stopped"
    ;;
  compress)
    # the collector is down, so every file is final; 2 threads (the box has a CPU quota). A failure leaves the plain file.
    dir=${2:-$(cat "$dirfile" 2> /dev/null)}
    [ -d "$dir" ] || exit 0
    command -v zstd > /dev/null 2>&1 || { echo "otelcol: zstd missing, files stay plain"; exit 0; }
    before=$(du -sk "$dir" | cut -f1)
    find "$dir" -name '*.jsonl' -size +0 -exec zstd -q -T2 -10 --rm {} + || echo "otelcol: zstd failed on some file"
    echo "otelcol: compressed $dir ${before} kB -> $(du -sk "$dir" | cut -f1) kB"
    ;;
  *)
    echo "usage: otelcol_box.sh start <dir> | stop | compress [dir]" >&2
    exit 2
    ;;
esac
