#!/usr/bin/env bash
# On-box helper for scripts/vastctl.py. Uploaded to /root/vast_box.sh (temp name + mv, so a running copy is never
# overwritten in place) and called as `bash /root/vast_box.sh <command> ...`. All pgrep patterns live here, in a
# file, so they can never match the ssh command line that invoked them.
#   health                     key=value lines: GPU, build/bench processes, job log progress, watchdog, disk, RAM, DNS
#   ensure-watchdog            start scripts/idle_autostop.sh unless running (supervisor if available, else nohup)
#                              and register it to come back after a container restart (supervisor + onstart.sh)
#   install-push               move a pushed tree from /root/.tetraeder_push into /root/tetraeder file by file
#                              (rename = new inode, so a script bash is executing is never rewritten in place)
#   deps                       make sure rustup + the pinned toolchain, cmake, ninja, sccache and a C toolchain exist
#                              (only installs what is missing: the GHCR image has them); no Python packages
#   provision <models...>      deps + cuda check + models + watchdog + inventory; run detached, logs PROVISION_DONE
#   verify <models...>         GPU visible, cargo works, models present, CUDA initialises (replacement acceptance)
#   models <org/name:file>...  download model files from ModelScope or Hugging Face (the faster) into
#                              .hf_cache/<org>__<name>/, verify sizes; file lists are joined with + (a.gguf+b.gguf)
#   run <script> [args...]     start a repo script detached (it guards itself with flock); log runs/vast/<name>.log
#   pack <dirs...>             pack small result files into /root/results.tgz (scripts/pack_results.sh)
#   inventory                  models and result directories on the box (for re-provisioning a replacement)
#   logs [n]                   tail the nightly log and the watchdog log
set -uo pipefail
TETRAEDER="${TETRAEDER_DIR:-/root/tetraeder}"
RUNS="$TETRAEDER/runs/vast"
cmd="${1:-health}"; shift || true

disk_probe() {  # sequential read bandwidth of /root in MB/s; measured once, cached (health runs every few minutes)
  local cache=/root/.disk_probe f=/root/.disk_probe.dat mbs=""
  [ -s "$cache" ] && { cat "$cache"; return; }
  if command -v fio > /dev/null 2>&1; then
    # terse v3 field 7 = read bandwidth in KiB/s
    mbs=$(timeout 40 fio --name=probe --filename="$f" --rw=read --bs=1M --size=512M --direct=1 --minimal 2>/dev/null \
          | awk -F';' 'NR==1 {printf "%d", $7 / 1024}')
  fi
  if [ -z "$mbs" ] || [ "$mbs" = 0 ]; then  # dd fallback: write 512 MB, read it back without the page cache
    dd if=/dev/zero of="$f" bs=1M count=512 oflag=direct status=none 2> /dev/null
    mbs=$(timeout 40 dd if="$f" of=/dev/null bs=1M iflag=direct 2>&1 | tail -1 \
          | awk '{for (i = 1; i < NF; i++) if ($(i + 1) == "MB/s") printf "%d", $i; else if ($(i + 1) == "GB/s") printf "%d", $i * 1000}')
  fi
  rm -f "$f"
  echo "${mbs:-0}" | tee "$cache"
}

health() {
  echo "now=$(date +%s)"
  local q
  q=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null \
      | head -1 | tr -d ' ')
  echo "gpu_util=$(echo "$q" | cut -d, -f1)"
  echo "gpu_mem_used=$(echo "$q" | cut -d, -f2)"
  echo "gpu_mem_total=$(echo "$q" | cut -d, -f3)"
  # busy = a cargo/rustc/nvcc/cmake/ninja process, or an engine/bench binary under target/ (anchored: a bare
  # "tetraeder" would match this very script's path /root/tetraeder/scripts/...)
  echo "exp_procs=$(pgrep -f '(^|/)(cargo|rustc|nvcc|cmake|ninja)( |$)|target/(release|debug)/tetraede[r]|bench[.]s[h]|(^|/)llama-' | wc -l)"
  echo "batch_alive=$(pgrep -f 'nightl[y][.]sh' | wc -l)"
  local log="$RUNS/nightly.log"
  if [ -f "$log" ]; then
    echo "batch_log_age_s=$(( $(date +%s) - $(stat -c %Y "$log") ))"
    echo "batch_started=$(grep -c '] start' "$log")"
    echo "batch_done=$(grep -c '] done' "$log")"
    echo "batch_failed=$(grep '] done' "$log" | grep -vc 'rc=0$')"
    echo "batch_all_done=$(grep -c '^NIGHTLY_DONE' "$log")"
    echo "batch_last=$(tail -1 "$log" | tr -d '\r' | cut -c1-160)"
  else
    echo "batch_log_age_s="
  fi
  # progress = newest write to any log under runs/vast (the nightly log only changes per stage; cargo output goes there too)
  local newest
  newest=$(find "$RUNS" -maxdepth 2 -name '*.log' -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1)
  [ -n "$newest" ] && echo "progress_age_s=$(( $(date +%s) - newest ))"
  echo "watchdog=$(pgrep -f 'idle_autost[o]p[.]sh' | wc -l)"
  echo "disk_free_gb=$(df -BG --output=avail /root 2>/dev/null | tail -1 | tr -dc '0-9')"
  echo "ram_gb=$(awk '/MemTotal/ {printf "%d", $2 / 1048576}' /proc/meminfo 2>/dev/null)"
  echo "disk_read_mbs=$(disk_probe)"
  if getent hosts console.vast.ai > /dev/null 2>&1; then echo "dns_ok=1"; else echo "dns_ok=0"; fi
}

ensure_watchdog() {
  local wrapper=/root/autostop_run.sh
  # supervisord may not pass the Vast-injected variables through; fall back to /etc/environment
  cat > "$wrapper.tmp" <<'EOS'
#!/usr/bin/env bash
if [ -z "${CONTAINER_API_KEY:-}" ] && [ -f /etc/environment ]; then set -a; . /etc/environment; set +a; fi
exec bash /root/tetraeder/scripts/idle_autostop.sh
EOS
  mv "$wrapper.tmp" "$wrapper"
  local mode=nohup
  if [ -d /etc/supervisor/conf.d ] && command -v supervisorctl > /dev/null 2>&1; then
    cat > /etc/supervisor/conf.d/tetraeder-autostop.conf <<EOS
[program:tetraeder-autostop]
command=/bin/bash $wrapper
autostart=true
autorestart=true
startsecs=5
stdout_logfile=/root/autostop.log
redirect_stderr=true
EOS
    supervisorctl reread > /dev/null 2>&1; supervisorctl update > /dev/null 2>&1
    mode=supervisor
  fi
  # also relaunch from the Vast on-start hook (runs on every container start), idempotently
  local line='bash /root/vast_box.sh ensure-watchdog >> /root/autostop.log 2>&1 || true'
  touch /root/onstart.sh
  grep -qF "$line" /root/onstart.sh || echo "$line" >> /root/onstart.sh
  sleep 3
  if [ "$(pgrep -f 'idle_autost[o]p[.]sh' | wc -l)" -eq 0 ]; then
    [ "$mode" = supervisor ] && supervisorctl start tetraeder-autostop > /dev/null 2>&1 && sleep 3
  fi
  if [ "$(pgrep -f 'idle_autost[o]p[.]sh' | wc -l)" -eq 0 ]; then
    nohup setsid bash "$wrapper" >> /root/autostop.log 2>&1 < /dev/null &
    mode="$mode+nohup"
    sleep 2
  fi
  echo "watchdog=$(pgrep -f 'idle_autost[o]p[.]sh' | wc -l) mode=$mode"
}

install_push() {
  local src=/root/.tetraeder_push n=0
  [ -d "$src" ] || { echo "no push dir"; return 1; }
  # normalise line endings in the staging copy, before anything is moved next to running scripts
  find "$src" -type f \( -name '*.sh' -o -name '*.py' -o -name '*.toml' -o -name '*.yaml' -o -name '*.md' \) \
      -not -path "$src/vendor/*" -print0 | xargs -0 -r sed -i 's/\r$//'
  cd "$src" || return 1
  while IFS= read -r -d '' f; do
    mkdir -p "$TETRAEDER/$(dirname "$f")"
    mv -f "$f" "$TETRAEDER/$f"; n=$((n + 1))
  done < <(find . -type f -print0)
  cd / && rm -rf "$src"
  echo "installed $n files into $TETRAEDER"
}

deps() {
  export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
  local sudo="" need=""
  [ "$(id -u)" -eq 0 ] || sudo="sudo"
  command -v cc > /dev/null 2>&1 || need="$need build-essential"
  command -v curl > /dev/null 2>&1 || need="$need curl ca-certificates"
  command -v git > /dev/null 2>&1 || need="$need git"
  command -v cmake > /dev/null 2>&1 || need="$need cmake"
  command -v ninja > /dev/null 2>&1 || need="$need ninja-build"
  command -v sccache > /dev/null 2>&1 || need="$need sccache"
  command -v python3 > /dev/null 2>&1 || need="$need python3"  # cuda_env.sh probes libcuda through ctypes
  if [ -n "$need" ]; then
    echo "installing missing packages:$need"
    # sccache is optional: retry without it if apt does not know it
    { DEBIAN_FRONTEND=noninteractive $sudo apt-get update -qq && \
      DEBIAN_FRONTEND=noninteractive $sudo apt-get install -y -qq --no-install-recommends $need; } \
      || { need=${need// sccache/}; DEBIAN_FRONTEND=noninteractive $sudo apt-get install -y -qq --no-install-recommends $need; } \
      || return 1
  fi
  if ! command -v rustup > /dev/null 2>&1; then
    curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none --profile minimal || return 1
  fi
  # the channel comes from the repo's rust-toolchain.toml; rustup does nothing when it is already installed
  local ch
  ch=$(sed -n 's/^channel *= *"\(.*\)"/\1/p' "$TETRAEDER/rust-toolchain.toml" 2>/dev/null | head -1)
  rustup toolchain install "${ch:-stable}" --profile minimal -c rustfmt -c clippy || return 1
  command -v sccache > /dev/null 2>&1 || echo "note: sccache missing, builds will not be cached"
  echo "deps ok: $(cargo --version) $(cmake --version | head -1) ninja $(ninja --version) sccache $(command -v sccache > /dev/null 2>&1 && echo yes || echo no)"
}

provision() {
  local rc=0
  echo "$(date +%T) provision start"
  deps || { echo "deps FAILED"; rc=1; }
  # shellcheck disable=SC1091
  . "$TETRAEDER/scripts/cuda_env.sh"
  echo "cuda: $CUDA_ENV_STATUS${LD_PRELOAD:+ (LD_PRELOAD=$LD_PRELOAD)}"
  [ "$CUDA_ENV_STATUS" = fail ] && { echo "CUDA FAILED: libcuda cannot initialise the GPU"; rc=1; }
  if [ "$#" -gt 0 ]; then models "$@" || rc=1; fi
  disk_probe > /dev/null  # cache the disk bandwidth now so `health` never pays for it
  ensure_watchdog
  inventory
  echo "$(date +%T) PROVISION_DONE rc=$rc"
}

verify() {  # replacement acceptance test: GPU visible, cargo works, models present, CUDA initialises
  local ok=1
  export PATH="$HOME/.cargo/bin:$PATH"
  nvidia-smi -L > /dev/null 2>&1 && echo "verify_gpu=1" || { echo "verify_gpu=0"; ok=0; }
  if (cd "$TETRAEDER" && cargo --version > /dev/null 2>&1 && command -v cmake > /dev/null 2>&1); then
    echo "verify_tools=1"
  else echo "verify_tools=0"; ok=0; fi
  # shellcheck disable=SC1091
  . "$TETRAEDER/scripts/cuda_env.sh"
  [ "$CUDA_ENV_STATUS" = fail ] && { echo "verify_cuda=0"; ok=0; } || echo "verify_cuda=1"
  local m dir file
  for m in "$@"; do
    dir="$TETRAEDER/.hf_cache/$(model_dir "${m%%:*}")"
    for file in $(echo "${m#*:}" | tr '+' ' '); do
      [ -s "$dir/$file" ] || { echo "verify_missing=$m/$file"; ok=0; }
    done
  done
  echo "verify_ok=$ok"
}

model_dir() { echo "${1//\//__}"; }

remote_size() {  # content length of the final response after redirects
  curl -sIL --max-time 60 "$1" | tr -d '\r' | grep -i '^content-length:' | tail -1 | awk '{print $2}'
}

probe_speed() {  # bytes/s for the first 8 MB of a URL: best of 2 tries of 10 s (single probes were noisy)
  local s best=0
  for _ in 1 2; do
    s=$(curl -sL -r 0-8000000 -o /dev/null -w "%{speed_download}" --max-time 10 "$1" 2>/dev/null || echo 0)
    s=${s%.*}; [ "${s:-0}" -gt "$best" ] && best=$s
  done
  echo "$best"
}

pfetch() {  # pfetch URL OUT SIZE [N]: N parallel ranged requests (hosts throttle per connection: 8 x ~0.6 MB/s)
  local url="$1" out="$2" size="$3" n="${4:-16}" chunk i start end pids=()
  chunk=$(( (size + n - 1) / n ))
  for ((i = 0; i < n; i++)); do
    start=$(( i * chunk )); end=$(( start + chunk - 1 )); [ "$end" -ge "$size" ] && end=$(( size - 1 ))
    curl -sSL --retry 5 --max-time 3600 -r "$start-$end" -o "$out.part$i" "$url" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  : > "$out"
  for ((i = 0; i < n; i++)); do cat "$out.part$i" >> "$out" && rm -f "$out.part$i"; done
}

models() {  # models org/name:file.gguf [org/name:a.gguf+b.gguf ...]
  local ok=1 spec
  for spec in "$@"; do
    local model="${spec%%:*}" files="" file
    if [ "$model" = "$spec" ]; then echo "FAIL  $spec: expected org/name:file.gguf"; ok=0; continue; fi
    files=$(echo "${spec#*:}" | tr '+' ' ')
    local dir
    dir="$TETRAEDER/.hf_cache/$(model_dir "$model")"
    mkdir -p "$dir"
    # one downloader per model directory: a second `models`/provision run waits instead of writing the same files
    # (two curls once appended to one shard and corrupted it)
    exec 9> "$dir/.download.lock"
    flock 9
    # ModelScope is the default (Hugging Face was throttled on earlier hosts), but some hosts get ~10-50 KB/s from it:
    # probe both on the first file and take Hugging Face when it is 1.5x faster
    local base_ms="https://modelscope.cn/models/$model/resolve/master" base_hf="https://huggingface.co/$model/resolve/main"
    local big="${files%% *}" sp_ms sp_hf base="$base_ms"
    sp_ms=$(probe_speed "$base_ms/$big"); sp_hf=$(probe_speed "$base_hf/$big")
    if [ $(( ${sp_hf:-0} * 2 )) -gt $(( ${sp_ms:-0} * 3 )) ]; then base="$base_hf"; fi
    echo "source $model: $base (modelscope $(( ${sp_ms:-0} / 1024 )) KB/s, huggingface $(( ${sp_hf:-0} / 1024 )) KB/s)"
    for file in $files; do
      local url="$base/$file" want have
      want=$(remote_size "$url")
      have=$(stat -c %s "$dir/$file" 2>/dev/null || echo 0)
      if [ -n "$want" ] && [ "$have" = "$want" ]; then echo "ok    $model/$file ($have bytes)"; continue; fi
      for attempt in 1 2 3; do
        if [ -n "$want" ] && [ "$want" -gt 50000000 ]; then
          pfetch "$url" "$dir/$file" "$want"
        else
          # resume only when the size is known (servers without byte ranges fail with curl error 33 on -C -)
          if [ -n "$want" ]; then
            curl -sSL --retry 3 --max-time 3600 -C - -o "$dir/$file" "$url" || curl -sSL --max-time 3600 -o "$dir/$file" "$url"
          else
            curl -sSL --retry 3 --max-time 3600 -o "$dir/$file" "$url"
          fi
        fi
        have=$(stat -c %s "$dir/$file" 2>/dev/null || echo 0)
        { [ -z "$want" ] && [ "$have" -gt 0 ]; } || [ "$have" = "$want" ] && break
        echo "retry $model/$file attempt $attempt (have $have want $want)"; rm -f "$dir/$file"
      done
      if [ -n "$want" ] && [ "$have" != "$want" ]; then echo "FAIL  $model/$file ($have != $want)"; ok=0
      else echo "got   $model/$file ($have bytes)"; fi
    done
    flock -u 9
  done
  [ "$ok" = 1 ] && echo "models_ok=1" || { echo "models_ok=0"; return 1; }
}

run_job() {  # run <script> [args...]: detached; the script itself takes a flock, so a repeated launch is harmless
  local script="$1"; shift
  cd "$TETRAEDER" || exit 1
  [ -f "$script" ] || { echo "no such script: $script"; exit 2; }
  mkdir -p runs/vast
  export PATH="$HOME/.cargo/bin:$PATH"
  nohup setsid bash "$script" "$@" >> "runs/vast/$(basename "$script" .sh).log" 2>&1 < /dev/null &
  sleep 3
  echo "launched $script at $(date +%T)"
  tail -n 5 "runs/vast/$(basename "$script" .sh).log" 2>/dev/null
}

pack() {
  cd "$TETRAEDER" || exit 1
  local dirs=() d
  for d in "$@" runs/vast bench/golden; do [ -e "$d" ] && dirs+=("$d"); done
  [ "${#dirs[@]}" -gt 0 ] || { echo "nothing to pack"; exit 2; }
  bash scripts/pack_results.sh "${dirs[@]}" && echo "md5=$(md5sum /root/results.tgz | cut -d' ' -f1)"
}

inventory() {
  cd "$TETRAEDER" 2>/dev/null || { echo "no_repo=1"; return; }
  local d
  for d in .hf_cache/*/; do [ -d "$d" ] && echo "model=$(basename "$d") $(du -sm "$d" | cut -f1)MB"; done
  for d in bench/results/*/; do [ -d "$d" ] && echo "run=${d%/} done=$(find "$d" -name DONE | wc -l)"; done
}

logs() {
  local n="${1:-30}"
  echo "== $RUNS/nightly.log"; tail -n "$n" "$RUNS/nightly.log" 2>/dev/null
  echo "== /root/autostop.log"; tail -n "$n" /root/autostop.log 2>/dev/null
}

case "$cmd" in
  health) health ;;
  ensure-watchdog) ensure_watchdog ;;
  install-push) install_push ;;
  deps) deps ;;
  provision) provision "$@" ;;
  verify) verify "$@" ;;
  models) models "$@" ;;
  run) run_job "$@" ;;
  pack) pack "$@" ;;
  inventory) inventory ;;
  logs) logs "$@" ;;
  *) echo "unknown command: $cmd" >&2; exit 2 ;;
esac
