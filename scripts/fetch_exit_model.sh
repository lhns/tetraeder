#!/usr/bin/env bash
# Downloads the Phase 2 exit model (Qwen3-235B-A22B Q4_K_M, five shards, about 142 GB) with taxameter's box-side
# downloader (parallel ranged requests). Started detached so GPU work can run meanwhile; ends with
# FETCH_DONE rc=N in runs/vast/fetch_exit.log.
# This wrapper adds what the downloader lacks (20261112c):
#  - a shard whose size equals the recorded size is never handed to the downloader (it re-fetched and truncated
#    complete shards);
#  - the mirror is probed here: a probe of 0 KB/s or an error means Hugging Face, ModelScope only when it is not
#    slower by 1.5x (both probes at 0 picked ModelScope, which then ran at 1 MB/s);
#  - stall guard: a download under FETCH_MIN_MBPS (20) for FETCH_STALL_S (180) seconds is killed and restarted on
#    the other mirror; the last round (FETCH_ROUNDS, 4) runs unguarded.
exec 8> "${FETCH_LOCK:-/root/tetraeder_fetch_exit.lock}"; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs/vast
log=runs/vast/fetch_exit.log
echo "=== RUN $(date +%FT%T) fetch_exit_model.sh ===" >> "$log"
model=Qwen/Qwen3-235B-A22B-GGUF
q=Q4_K_M/Qwen3-235B-A22B-Q4_K_M
dir="$PWD/${TETRAEDER_MODELS_DIR:-.hf_cache}/Qwen__Qwen3-235B-A22B-GGUF"
box="${TAXAMETER_HOME:-/root/.taxameter}/box.sh"
# expected bytes per shard (from the 20261112c fetch log); FETCH_SIZES ("file bytes;file bytes") overrides for tests
shards=(
  "$q-00001-of-00005.gguf 29742283872"
  "$q-00002-of-00005.gguf 29974106496"
  "$q-00003-of-00005.gguf 29933980640"
  "$q-00004-of-00005.gguf 29936094304"
  "$q-00005-of-00005.gguf 22567608864"
)
[ -n "${FETCH_SIZES:-}" ] && IFS=";" read -r -a shards <<< "$FETCH_SIZES"
min_bps=$(( ${FETCH_MIN_MBPS:-20} * 1048576 ))
stall_s=${FETCH_STALL_S:-180}
sample_s=${FETCH_SAMPLE_S:-10}
rounds=${FETCH_ROUNDS:-4}

missing() {  # prints the shard files whose size differs from the recorded one
  local e f want
  for e in "${shards[@]}"; do
    f=${e% *}; want=${e#* }
    [ "$(stat -c %s "$dir/$f" 2>/dev/null || echo 0)" = "$want" ] || echo "$f"
  done
}

probe() {  # probe URL: best of two 8 MB ranged reads in KB/s; 0 on any error
  local s best=0 i
  for i in 1 2; do
    s=$(curl -sL -r 0-8000000 -o /dev/null -w "%{speed_download}" --max-time 10 "$1" 2>/dev/null || echo 0)
    s=${s%.*}; [[ "$s" =~ ^[0-9]+$ ]] || s=0
    [ "$s" -gt "$best" ] && best=$s
  done
  echo $(( best / 1024 ))
}

pick_mirror() {  # prints huggingface or modelscope
  local first=$1 ms hf
  ms=${FETCH_PROBE_MS_KBPS:-$(probe "https://modelscope.cn/models/$model/resolve/master/$first")}
  hf=${FETCH_PROBE_HF_KBPS:-$(probe "https://huggingface.co/$model/resolve/main/$first")}
  echo "probe: modelscope $ms KB/s, huggingface $hf KB/s" >&2
  if [ "${ms:-0}" -le 0 ] || [ $(( hf * 2 )) -gt $(( ms * 3 )) ]; then echo huggingface; else echo modelscope; fi
}

tree_bytes() { du -sb "$dir" 2>/dev/null | cut -f1; }  # shards and their .part files

# fetch_round MIRROR GUARDED SPEC: the downloader's status, or 99 when the stall guard killed it
fetch_round() {
  local src=$1 guarded=$2 spec=$3 pid prev cur win=0 slow=0
  TAXAMETER_MODEL_SOURCE=$src TAXAMETER_DIR="$PWD" setsid bash "$box" models "$spec" 8>&- &
  pid=$!
  prev=$(tree_bytes)
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$sample_s"
    cur=$(tree_bytes)
    # only growth counts: moving parts into the shard keeps the size, a restart that deletes parts is not progress
    [ "${cur:-0}" -gt "${prev:-0}" ] && win=$(( win + cur - prev ))
    prev=$cur
    slow=$(( slow + sample_s ))
    if [ "$slow" -ge "$stall_s" ]; then
      if [ "$guarded" = 1 ] && [ "$win" -lt $(( min_bps * slow )) ]; then
        echo "[$(date +%T)] stall on $src: $(( win / slow / 1048576 )) MB/s over $slow s, switching mirror"
        kill -- "-$pid" 2>/dev/null; pkill -P "$pid" 2>/dev/null; kill "$pid" 2>/dev/null
        wait "$pid" 2>/dev/null
        return 99
      fi
      win=0; slow=0
    fi
  done
  wait "$pid"
}

{
  echo "[$(date +%T)] start fetch, disk free $(df -h --output=avail . | tail -1)"
  mkdir -p "$dir"
  src=""
  for ((round = 1; round <= rounds; round++)); do
    todo=$(missing)
    [ -n "$todo" ] || break
    echo "[$(date +%T)] round $round: missing $(echo $todo | wc -w) of ${#shards[@]} shards"
    [ -n "$src" ] || src=$(pick_mirror "$(echo "$todo" | head -1)")
    spec="$model:$(echo $todo | tr ' ' '+')"
    guarded=1; [ "$round" = "$rounds" ] && guarded=0
    fetch_round "$src" "$guarded" "$spec"
    if [ $? = 99 ]; then
      if [ "$src" = huggingface ]; then src=modelscope; else src=huggingface; fi
    fi
  done
  rc=0; [ -z "$(missing)" ] || rc=1
  echo "[$(date +%T)] FETCH_DONE rc=$rc"
} >> "$log" 2>&1
