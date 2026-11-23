#!/usr/bin/env bash
# Transport benchmark on rented boxes (ADR 0023, tree node cluster-transport-wan). Runs ON the box from the repo dir.
# The file name ends in bench.sh so taxameter's [box] busy pattern keeps the idle watchdog quiet.
#
#   MODE=prep   build transport-bench (release, this crate only), start the servers, write meta.txt and ports.env
#   MODE=loop   client against the local servers (127.0.0.1)
#   MODE=wan    client against a peer: PEER_IP=<ip> PEER_PORTS=<aes_tcp,aes_tls,aes_quic,cha_tcp,cha_tls,cha_quic>
#               (the peer's external ports, from its ports.env); SCENARIO names the result subdirectory
#
# Env: LABEL (box name in the results path), SCENARIO (default = MODE), REPS (3), ITERS (2000), WARMUP (200),
#      TBYTES (268435456), RESULTS_DATE (20261123), PEER_LABEL (name of the peer, for meta only).
# Servers: AES set on 7000 (tcp 7000, tls 7001, quic udp 7002), ChaCha set on 7100 (7100, 7101, 7102/udp).
# Each client call is one (cipher, transport) pair, so repetitions interleave by protocol and cipher:
# for rep: for cipher: tcp, tls, quic. Each call writes one JSON.
set -uo pipefail
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH"
cd "$(dirname "$0")/.." || exit 1
MODE=${MODE:-prep}
LABEL=${LABEL:?LABEL}
DATE=${RESULTS_DATE:-20261123}
OUT=bench/results/transport-wan-$DATE/$LABEL
BIN=target/release/transport-bench
mkdir -p "$OUT" runs/vast
exec > >(tee -a "runs/vast/transport_wan_$MODE.log") 2>&1
echo "== transport_wan_bench $MODE label=$LABEL $(date +%T)"

case "$MODE" in
prep)
  exec 9> /root/transport_prep.lock
  flock -n 9 || { echo "prep already running"; exit 0; }
  # Vast puts the port mappings in the container's PID 1 environment, not in ssh sessions
  eval "$(xargs -0 -n1 < /proc/1/environ | grep -E '^(VAST_(TCP|UDP)_PORT_[0-9]+|PUBLIC_IPADDR)=' | sed 's/^/export /')"
  CARGO_BUILD_JOBS=4 cargo build --release -p tetraeder-transport --bin transport-bench || { echo "BUILD_FAILED"; exit 1; }
  {
    echo "label=$LABEL"
    echo "commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "date=$(date -u +%FT%TZ)"
    echo "kernel=$(uname -r)"
    echo "cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
    echo "cores_online=$(nproc)"
    echo "cpu_quota=$(cat /sys/fs/cgroup/cpu.max 2> /dev/null || echo unknown)"
    echo "aes_flag=$(grep -m1 -o -w aes /proc/cpuinfo || echo none)"
    echo "public_ip=${PUBLIC_IPADDR:-unknown}"
    echo "gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2> /dev/null | head -1)"
    for i in $(ls /sys/class/net 2> /dev/null); do
      echo "nic_$i=$(cat /sys/class/net/$i/speed 2> /dev/null || echo ?)"
    done
    env | grep -E '^VAST_(TCP|UDP)_PORT_' | sort | sed 's/^/ports_/'
  } > "$OUT/meta.txt"
  {
    echo "AES_TCP=${VAST_TCP_PORT_7000:-}"
    echo "AES_TLS=${VAST_TCP_PORT_7001:-}"
    echo "AES_QUIC=${VAST_UDP_PORT_7002:-}"
    echo "CHA_TCP=${VAST_TCP_PORT_7100:-}"
    echo "CHA_TLS=${VAST_TCP_PORT_7101:-}"
    echo "CHA_QUIC=${VAST_UDP_PORT_7102:-}"
    echo "PUBLIC_IP=${PUBLIC_IPADDR:-}"
  } > "$OUT/ports.env"
  # servers outlive this script (setsid); a second prep does not start a second pair
  start_server() { # name base cipher
    ( exec 8> "/root/ts_$1.lock"; flock -n 8 || exit 0
      setsid nohup "$BIN" --cipher "$3" server --addr "0.0.0.0:$2" --transport all > "runs/vast/ts_$1.log" 2>&1 9>&- < /dev/null & )
  }
  start_server aes 7000 aes
  start_server cha 7100 chacha
  sleep 3
  tail -n3 runs/vast/ts_aes.log runs/vast/ts_cha.log
  cat "$OUT/ports.env"
  echo PREP_DONE
  ;;
loop | wan)
  SCEN=${SCENARIO:-$MODE}
  REPS=${REPS:-3}
  ITERS=${ITERS:-2000}
  WARMUP=${WARMUP:-200}
  TBYTES=${TBYTES:-268435456}
  exec 9> "/root/transport_$SCEN.lock"
  flock -n 9 || { echo "$SCEN already running"; exit 0; }
  mkdir -p "$OUT/$SCEN"
  if [ "$MODE" = loop ]; then
    HOST=127.0.0.1
    P_AES=(7000 7001 7002) P_CHA=(7100 7101 7102)
  else
    HOST=${PEER_IP:?PEER_IP}
    IFS=, read -r a1 a2 a3 c1 c2 c3 <<< "${PEER_PORTS:?PEER_PORTS}"
    P_AES=("$a1" "$a2" "$a3") P_CHA=("$c1" "$c2" "$c3")
    {
      echo "peer=${PEER_LABEL:-?} ip=$HOST ports=$PEER_PORTS"
      echo "-- ping $HOST"
      ping -c 20 -i 0.2 -q "$HOST" 2>&1 || echo "ping unavailable or blocked"
    } | tee "$OUT/$SCEN/ping.txt"
  fi
  echo "params reps=$REPS iters=$ITERS warmup=$WARMUP throughput_bytes=$TBYTES host=$HOST" | tee "$OUT/$SCEN/params.txt"
  TR=(tcp tls quic)
  for rep in $(seq 1 "$REPS"); do
    for cipher in aes chacha; do
      if [ "$cipher" = aes ]; then P=("${P_AES[@]}"); else P=("${P_CHA[@]}"); fi
      for i in 0 1 2; do
        t=${TR[$i]}
        f="$OUT/$SCEN/${t}-${cipher}-r${rep}.json"
        [ -s "$f" ] && continue # resume
        echo "$(date +%T) rep $rep $cipher $t -> $HOST:${P[$i]}"
        timeout 900 "$BIN" --cipher "$cipher" client --addr "$HOST:${P[$i]}" --transport "$t" \
          --iters "$ITERS" --warmup "$WARMUP" --throughput-bytes "$TBYTES" --json "$f" || echo "CALL_FAILED $t $cipher r$rep"
      done
    done
  done
  echo "$SCEN" > "$OUT/$SCEN/DONE"
  echo "TRANSPORT_${SCEN}_DONE"
  ;;
*)
  echo "unknown MODE $MODE"
  exit 2
  ;;
esac
