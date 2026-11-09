#!/usr/bin/env bash
# Weekly GPU smoke, run LOCALLY (docs/sessions.md, "GPU smoke"): one cheap RTX 3090-class box, the whole taxameter cycle.
#   create (cheapest all-in offer under the budget) -> provision -> check cuda: ok -> run scripts/session.sh
#   SESSION=gpu-smoke detached -> wait on the box -> fetch -> verify -> destroy.
#
#   scripts/gpu_smoke.sh             dry run (default): prints every command, touches nothing, rents nothing
#   scripts/gpu_smoke.sh --run       the real cycle. Needs the user's go while the box hold stands (AGENTS.md).
#
# Env: GPU_SMOKE_BUDGET_USD (0.30, box plus traffic), GPU_SMOKE_HOURS (0.75, billed hours assumed for the price cap),
# GPU_SMOKE_NOISE (1: also fetch the 8B and run the A/A pair), GPU_SMOKE_TRAFFIC_USD_GB (0.026, the worst
# inet_down_cost seen), GPU_SMOKE_MAX_MIN (45: wait limit for the session on the box), GPU_SMOKE_MIN_CORES (6), GPU_SMOKE_SESSION_MIN (30).
# Exit: 0 smoke ok, 1 smoke failed (results fetched), 2 usage or precondition, 3 no box (cap or no offer), 4 box id
# not found in create's output (destroy it by hand), 5 CUDA not ok, 6 session did not start or finish, 7 fetch failed.
# Every exit after create destroys the box (fetch first, --force only when the fetch failed).
set -u
mode=dry
case "${1:-}" in
  --run) mode=run ;;
  --dry-run | "") ;;
  *) echo "usage: $0 [--dry-run|--run]" >&2; exit 2 ;;
esac
cd "$(dirname "$0")/.." || exit 2

budget="${GPU_SMOKE_BUDGET_USD:-0.30}"
hours="${GPU_SMOKE_HOURS:-0.75}"
noise="${GPU_SMOKE_NOISE:-1}"
traffic="${GPU_SMOKE_TRAFFIC_USD_GB:-0.026}"
max_min="${GPU_SMOKE_MAX_MIN:-45}"
cores="${GPU_SMOKE_MIN_CORES:-6}"
session_min="${GPU_SMOKE_SESSION_MIN:-30}" # script minutes on the box; a cold CUDA build on a new image may need 40
models="Qwen/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q8_0.gguf"
dl_gb=0.7
if [ "$noise" = 1 ]; then
  models="$models,Qwen/Qwen3-8B-GGUF:Qwen3-8B-Q4_K_M.gguf"
  dl_gb=6
fi
# price cap: what is left of the budget after the worst-case traffic, over the billed hours, never above the policy cap
max_price=$(awk -v b="$budget" -v g="$dl_gb" -v t="$traffic" -v h="$hours" \
  'BEGIN {p = (b - g * t) / h; if (p > 0.45) p = 0.45; printf "%.3f", p}')
awk -v p="$max_price" 'BEGIN {exit !(p >= 0.05)}' || { echo "budget $budget leaves no room for a box ($max_price USD/h)" >&2; exit 2; }
T="uv run taxameter"
log=/root/tetraeder/runs/vast/session-gpu-smoke.log
stamp=$(date +%Y%m%d-%H%M%S)
wlog="runs/vast/gpu-smoke-$stamp.local.log"
mkdir -p runs/vast

id=""
destroyed=0
fetched=0

# t <args...>: a taxameter call; the dry run only prints it. The id is passed explicitly once the box exists.
t() {
  if [ "$mode" = dry ]; then echo "+ $T $*"; return 0; fi
  # shellcheck disable=SC2086
  $T "$@"
}

# shellcheck disable=SC2329 # runs from the trap
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  if [ "$mode" = run ] && [ -n "$id" ] && [ "$destroyed" = 0 ]; then
    echo "[$(date +%T)] cleanup: box $id still up"
    [ "$fetched" = 1 ] || $T fetch --id "$id" --minutes 5 || true
    if ! $T destroy --id "$id" --yes; then
      echo "destroy refused (unfetched results?): forcing, the box must not keep billing"
      $T destroy --id "$id" --yes --force || echo "DESTROY FAILED: box $id is still billing, destroy it by hand"
    fi
  fi
  exit "$rc"
}

die() { echo "gpu_smoke: $2" >&2; exit "$1"; }

echo "[$(date +%T)] gpu-smoke ($mode) budget $budget USD = $max_price USD/h x $hours h + $dl_gb GB x $traffic; models $models"
if [ "$mode" = run ]; then
  exec > >(tee -a "$wlog") 2>&1
  trap cleanup EXIT INT TERM
  [ -z "$(git status --porcelain --untracked-files=no)" ] || die 2 "uncommitted changes: provision and run push committed HEAD only"
  command -v uv > /dev/null || die 2 "uv not found"
  uv sync --quiet || die 2 "uv sync failed"
  echo "HEAD $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"
fi

# 1. slot and offer: status first (AGENTS.md), create refuses past max_instances and past the price cap
t status
create_out=$(t --json create --auto --max-price "$max_price" --download-gb "$dl_gb" --min-cpu-cores "$cores") \
  || die 3 "create failed (instance cap, no offer under $max_price USD/h, or an error)"
if [ "$mode" = dry ]; then
  echo "$create_out"
  id="<id>"
else
  id=$(printf '%s' "$create_out" | python -I -c '
import json, re, sys
text = sys.stdin.read()
def find(o):
    if isinstance(o, dict):
        for k in ("id", "instance_id", "contract_id"):
            if isinstance(o.get(k), (int, str)) and re.fullmatch(r"\d{5,}", str(o[k])):
                return str(o[k])
        for v in o.values():
            r = find(v)
            if r:
                return r
    elif isinstance(o, list):
        for v in o:
            r = find(v)
            if r:
                return r
try:
    r = find(json.loads(text))
except ValueError:
    r = None
if not r:
    m = re.findall(r"\d{6,}", text)
    r = m[-1] if m else ""
print(r)')
  if [ -z "$id" ]; then
    $T status
    die 4 "no box id in create's output: a box may be running, check taxameter status and destroy it by hand"
  fi
  echo "[$(date +%T)] box $id created"
  echo "$id" > .vast/gpu-smoke.id 2> /dev/null || true
fi
start=$(date +%s)

# 2. provision (pushes committed HEAD; only the smoke's models) and the CUDA check
t provision --id "$id" --models "$models" || die 5 "provision failed"
# shellcheck disable=SC2016 # expands on the box
cuda=$(t ssh --id "$id" --timeout 60 -- '. /root/.taxameter/cuda_env.sh; echo "cuda: $CUDA_ENV_STATUS"')
echo "$cuda"
if [ "$mode" = run ]; then
  case "$cuda" in *"cuda: ok"*) ;; *) die 5 "cuda is not ok on box $id: $cuda" ;; esac
fi

# 3. launch detached (session.sh holds the flock); check that its log advances
echo "[$(date +%T)] launch"
t run --id "$id" scripts/session.sh --env SESSION=gpu-smoke --env "NOISE_STAGE=$noise" --env "SESSION_MIN=$session_min"
first=$(t ssh --id "$id" --timeout 120 -- "sleep 45; tail -n 2 $log 2>/dev/null")
echo "$first"
if [ "$mode" = run ] && [ -z "$first" ]; then
  echo "session log is empty after 45 s: starting by hand (the flock makes a duplicate a no-op)"
  t ssh --id "$id" --timeout 120 -- "cd /root/tetraeder && (SESSION=gpu-smoke NOISE_STAGE=$noise SESSION_MIN=$session_min nohup setsid bash scripts/session.sh > /root/session.out 2>&1 < /dev/null &); sleep 30; tail -n 2 $log"
fi

# 4. wait on the box (loops run there, not here); each call returns after at most 540 s
deadline=$((start + max_min * 60))
fails=0
done_seen=0
while [ "$mode" = run ]; do
  out=$($T ssh --id "$id" --timeout 540 -- "for i in \$(seq 100); do grep -q SESSION_DONE $log && break; sleep 5; done; tail -n 3 $log") && fails=0 || fails=$((fails + 1))
  echo "[$(date +%T)] $out"
  case "$out" in *SESSION_DONE*) done_seen=1; break ;; esac
  [ "$fails" -lt 5 ] || break
  [ "$(date +%s)" -lt "$deadline" ] || { echo "OVERRUN: more than $max_min min"; break; }
done
if [ "$mode" = dry ]; then
  t ssh --id "$id" --timeout 540 -- "for i in \$(seq 100); do grep -q SESSION_DONE $log && break; sleep 5; done; tail -n 3 $log"
fi

# 5. fetch, verify, destroy (a session that did not finish is still fetched: its logs say why)
if t fetch --id "$id"; then fetched=1; else echo "fetch failed"; fi
result=1
if [ "$mode" = run ] && [ "$fetched" = 1 ]; then
  summary=$(find bench/results -path '*/gpu-smoke/summary.json' -newer "$wlog" 2> /dev/null | sort | tail -n 1)
  if [ -n "$summary" ] && grep -q '"ok": true' "$summary" && [ -f "$(dirname "$(dirname "$summary")")/DONE" ]; then
    result=0
    if [ "$noise" = 1 ] && ! ls "$(dirname "$summary")"/phase0-*.json > /dev/null 2>&1; then
      echo "no A/A results next to $summary"
      result=1
    fi
  fi
  echo "[$(date +%T)] verify: summary ${summary:-missing}, ok=$([ "$result" = 0 ] && echo yes || echo no)"
fi
end=$(date +%s)
dest=$(t destroy --id "$id" --yes) && destroyed=1
echo "$dest"
if [ "$mode" = run ]; then
  if [ "$destroyed" = 0 ]; then
    dest=$($T destroy --id "$id" --yes --force) && destroyed=1
    echo "$dest"
  fi
  # cost record next to the results: upper bound from the price cap, plus taxameter's destroy output (it reports the
  # traffic fields inet_down_billed and inet_down_cost)
  if [ -n "${summary:-}" ]; then
    DEST="$dest" ID="$id" START="$start" END="$end" MAXP="$max_price" BUDGET="$budget" python -I -c '
import json, os
e = os.environ
mins = (int(e["END"]) - int(e["START"])) / 60
print(json.dumps({"box_id": e["ID"], "minutes_since_create": round(mins, 1), "max_price_per_hour": float(e["MAXP"]),
                  "budget_usd": float(e["BUDGET"]), "box_cost_upper_bound_usd": round(mins / 60 * float(e["MAXP"]), 3),
                  "destroy_output": e["DEST"]}, indent=2))' > "$(dirname "$summary")/cost.json"
  fi
  [ "$fetched" = 1 ] || exit 7
  [ "$done_seen" = 1 ] || exit 6
  [ "$result" = 0 ] || exit 1
  echo "[$(date +%T)] gpu-smoke ok. Next: cargo run -p tetraeder-bench -- noise update; commit bench/results and bench/noise"
fi
exit 0
