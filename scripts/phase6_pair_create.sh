#!/usr/bin/env bash
# Case S: find a pair and create both boxes in one step (pairs left the market within minutes on 2026-11-26).
#   bash scripts/phase6_pair_create.sh OUT_DIR
# Runs scripts/phase6_pair_finder.py (TETRAEDER_PLAN_BIN, default C:/tt/debug/tetraeder-plan.exe), takes the first plannable pair whose
# boxes carry no "RAM under 64 GB" flag and starts both creates at once through scripts/sc_offers.sh (--download-gb 150, disk 230, the
# 0.90 cap), each with --gpu of its class and --ports 7000, the second with --allow-second. Logs: OUT_DIR/create-{a,b}.log.
# Exit 0: both up, prints ID_A, ID_B, T_CREATED_A, T_CREATED_B (also OUT_DIR/pair.env). Exit 3: no pair. Exit 4: one box came up and was
# destroyed at once. Exit 5: none came up. The API key never passes through here: taxameter reads it from its file.
set -u
cd "$(dirname "$0")/.." || exit 1
outd="${1:?usage: phase6_pair_create.sh OUT_DIR}"
mkdir -p "$outd"
py="${PYTHON:-.venv/Scripts/python.exe}"
[ -x "$py" ] || py=python3
export TETRAEDER_PLAN_BIN="${TETRAEDER_PLAN_BIN:-C:/tt/debug/tetraeder-plan.exe}"
echo "[$(date +%T)] pair finder"
# shellcheck disable=SC2086 # PAIR_FINDER_ARGS: extra finder flags, word split (e.g. --skip-hosts H for an offer taxameter cannot find this session)
"$py" scripts/phase6_pair_finder.py --out "$outd/pairs.json" ${PAIR_FINDER_ARGS:-} | tail -n 4
pick=$("$py" -I -c '
import json, sys
d = json.load(open(sys.argv[1]))
for p in d.get("pairs", []):
    if p.get("plannable") and not any("RAM under 64 GB" in f for f in p.get("flags", [])):
        print(" ".join(b["id"] + "|" + b["gpu_name"].replace(" ", "_") + "|" + str(b.get("machine_id") or "") for b in p["boxes"]))
        break
' "$outd/pairs.json") || exit 3
[ -n "$pick" ] || { echo "[$(date +%T)] no pair without a RAM flag: nothing created"; exit 3; }
read -r pa pb <<< "$pick"
IFS='|' read -r oa ga ma <<< "$pa"
IFS='|' read -r ob gb mb <<< "$pb"
ga=${ga//_/ } gb=${gb//_/ }
# the finder's offer ids rotate within seconds (2026-11-26: both ids gone 13 s later); one `offers` listing of both classes records
# each current id's machine in .vast/offers.json (taxameter ADR 0013), the creates take the id the finder's machine has now, and
# `create --offer` follows that machine if it rotates again. Same machine, same filters and caps: no rule changes
bash scripts/sc_offers.sh offers --gpu "$ga,$gb" --top 500 > "$outd/offers.log" 2>&1
cur() { # cur MACHINE FALLBACK_ID: the listed offer id of that machine now, else the finder's
  "$py" -I -c '
import json, sys
m, fb = sys.argv[2:4]
d = json.load(open(sys.argv[1])) if m else {}
print(next((k for k, v in d.items() if str(v.get("machine_id")) == m), fb))
' "${VAST_STATE_DIR:-.vast}/offers.json" "$1" "$2" 2> /dev/null || echo "$2"
}
oa=$(cur "$ma" "$oa") ob=$(cur "$mb" "$ob")
echo "[$(date +%T)] pair: a $oa ($ga), b $ob ($gb); both creates now"
ta=$(date +%s)
bash scripts/sc_offers.sh create --offer "$oa" --gpu "$ga" --ports 7000 > "$outd/create-a.log" 2>&1 &
pid_a=$!
tb=$(date +%s)
bash scripts/sc_offers.sh create --offer "$ob" --gpu "$gb" --ports 7000 --allow-second > "$outd/create-b.log" 2>&1 &
pid_b=$!
wait "$pid_a"
wait "$pid_b"
# a box is rented once taxameter logs "created instance", even when its create failed later (waiting for ssh)
id_of() { grep -oE '\[[0-9]+\] created instance' "$1" | head -n 1 | tr -dc '0-9'; }
ia=$(id_of "$outd/create-a.log") ib=$(id_of "$outd/create-b.log")
tail -n 3 "$outd/create-a.log" "$outd/create-b.log"
if [ -n "$ia" ] && [ -n "$ib" ]; then
  printf 'ID_A=%s\nID_B=%s\nT_CREATED_A=%s\nT_CREATED_B=%s\nOFFER_A=%s\nOFFER_B=%s\n' "$ia" "$ib" "$ta" "$tb" "$oa" "$ob" | tee "$outd/pair.env"
  exit 0
fi
for i in $ia $ib; do
  echo "[$(date +%T)] only box $i came up: destroyed at once (never hold one for a partner)"
  uv run taxameter destroy --id "$i" --yes | tail -n 2
  exit 4
done
echo "[$(date +%T)] neither create rented a box"
exit 5
