# Measured stage budgets for the deadline guard (sourced by exit_run.sh). A stage's default minutes come from the
# 91 GB / 3090 class; on slower hosts (EPYC 7642 at a 23-core quota the builder 32K stages took 19.5 to 26 min) the
# last measured wall time of the same class replaces them when larger.
LAST_BUILDER_SEC=0
LAST_CACHE_SEC=0

# budget_min DEFAULT_MIN LAST_SEC: max(DEFAULT_MIN, ceil(LAST_SEC * 1.1 / 60))
budget_min() {
  local m=$(((${2:-0} * 11 + 599) / 600))
  if [ "$m" -gt "$1" ]; then echo "$m"; else echo "$1"; fi
}

# stage_class NAME: builder, cache or empty
stage_class() {
  case "$1" in
    main-overlap* | main-prefetch* | main-repeat) echo builder ;;
    llamacpp-cache*) echo cache ;;
  esac
}

# record_stage_sec NAME SEC: remember the wall time of a finished stage under its class
record_stage_sec() {
  case "$(stage_class "$1")" in
    builder) LAST_BUILDER_SEC=$2 ;;
    cache) LAST_CACHE_SEC=$2 ;;
  esac
}
