#!/usr/bin/env bash
# Shared by scripts/phase6_exit_f.sh and scripts/phase6_exit_s.sh (sourced): how the driver reaches a box and the session file.
# The caller sets LOOPBACK (1: both boxes are directories and process sets on this machine), repo, out, lb (the loopback work dir),
# port_a, port_b, head_port, poll_s, wait_step, and defines say and die.
# shellcheck shell=bash disable=SC2154
p6dir() { if [ "$LOOPBACK" = 1 ]; then echo "$lb/$1"; else echo /root/p6; fi; }
box_id() { if [ "$1" = a ]; then echo "${ID_A:?set ID_A}"; else echo "${ID_B:?set ID_B}"; fi; }
bx() { # bx BOX TIMEOUT CMD...: CMD runs in the box's repo with its P6 work dir
  local box=$1 t=$2
  shift 2
  if [ "$LOOPBACK" = 1 ]; then
    mkdir -p "$lb/$box"
    (cd "$repo" && P6="$lb/$box" P6_LOOPBACK=1 P6_LB_PORT="$([ "$box" = a ] && echo "$port_a" || echo "$port_b")" HEAD_HOST=127.0.0.1 HEAD_PORT="$head_port" \
      TETRAEDER_BIN="$lb/tetraeder" POLL_S="$poll_s" bash -c "$*")
  else
    # taxameter prints its own "box running ..." status line (CRLF) on stdout before the command's output: drop it, keep the exit status
    local rc
    uv run taxameter ssh --id "$(box_id "$box")" --timeout "$t" -- "cd /root/tetraeder && export P6=/root/p6 && $*" \
      | tr -d '\r' | sed -u '/^\[[0-9:]*\] box running /d'
    rc=${PIPESTATUS[0]}
    return "$rc"
  fi
}
put() { # put BOX LOCAL_FILE REL_PATH_UNDER_P6
  local box=$1 f=$2 rel=$3
  if [ "$LOOPBACK" = 1 ]; then
    mkdir -p "$(dirname "$lb/$box/$rel")" && cp "$f" "$lb/$box/$rel"
  else
    local bin=()
    case "$f" in *.gz) bin=(--binary) ;; esac # text mode turns CRLF into LF, which breaks a gzip file
    uv run taxameter put --id "$(box_id "$box")" "${bin[@]}" "$f" "/root/p6/$rel" > /dev/null
  fi
}
get() { # get BOX REL_PATH_UNDER_P6 LOCAL_FILE: through ssh (base64), so no fetch pack is needed for files of a few MB
  local box=$1 rel=$2 f=$3
  mkdir -p "$(dirname "$f")"
  if [ "$LOOPBACK" = 1 ]; then
    cp "$lb/$box/$rel" "$f"
  else
    bx "$box" 300 "gzip -c /root/p6/$rel | base64 -w0" | base64 -d | gzip -dc > "$f"
  fi
}
wait_line() { # wait_line BOX REL_LOG REGEX SECONDS: waits on the box (ssh loops of 500 s) for REGEX on the log's LAST line
  local box=$1 rel=$2 re=$3 secs=$4 t0=$SECONDS line
  while [ $((SECONDS - t0)) -lt "$secs" ]; do
    line=$(bx "$box" 560 "for i in \$(seq 1 100); do tail -n 1 $(p6dir "$box")/$rel 2> /dev/null | grep -E '$re' && exit 0; sleep $wait_step; done; exit 1") && { echo "$line"; return 0; }
  done
  return 1
}
epoch_of() { bx "$1" 60 "date +%s.%N" | tail -n 1; }
sess() { # sess KEY JSON_VALUE: sets session.json[KEY] (dotted keys nest)
  python3 -I -c '
import json, os, sys
p, key, val = sys.argv[1:4]
d = json.load(open(p)) if os.path.exists(p) else {}
cur = d
parts = key.split(".")
for k in parts[:-1]:
    cur = cur.setdefault(k, {})
cur[parts[-1]] = json.loads(val)
json.dump(d, open(p, "w"), indent=2)
' "$out/session.json" "$1" "$2"
}

