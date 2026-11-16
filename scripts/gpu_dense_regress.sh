#!/usr/bin/env bash
# Dense regression check on CUDA, after D2a: step_shapes (Qwen3-8B, all layers offloaded) built at the first smoke box
# commit (OLD, dense was 1.53x libllama there) and at this checkout (HEAD), run interleaved for ROUNDS rounds.
# Engines: dense,llama at OLD (its test may lack dense_norepack), dense,dense_norepack,llama at HEAD.
# The box checkout has no .git (taxameter pushes files), so OLD is fetched from the public repo into its own
# directory; HEAD is built from this checkout. Each has its own CARGO_TARGET_DIR and keeps its shared libraries
# beside its binary (LD_LIBRARY_PATH per binary).
# Started detached by `taxameter run scripts/gpu_dense_regress.sh`; log runs/vast/dense_regress.log ends with
# DENSE_REGRESS_DONE rc=<n>. Results: bench/results/<session>/dense-regress/{shapes.txt,table.md,DONE}.
# A rerun resumes: built binaries and finished runs are kept. Env: SESSION, OLD (030d1f3), ROUNDS (3), REPO_URL.
exec 8> /root/tetraeder_dense_regress.lock
flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
root=$PWD
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH" CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
# shellcheck source=scripts/box_libs.sh
. scripts/box_libs.sh
command -v sccache > /dev/null 2>&1 && export RUSTC_WRAPPER=sccache
# shellcheck disable=SC1091
. "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
if [ -z "${TETRAEDER_CUDA_ARCHS:-}" ]; then
  archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
  [ -n "$archs" ] && export TETRAEDER_CUDA_ARCHS="$archs"
fi
session="${SESSION:-$(date +%Y%m%d)}"
out="$root/bench/results/$session/dense-regress"
log="$root/runs/vast/dense_regress.log"
old="${OLD:-030d1f3}"
rounds="${ROUNDS:-3}"
repo_url="${REPO_URL:-https://github.com/lhns/tetraeder}"
gguf="$root/.hf_cache/Qwen__Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"
work=/root/dense-regress
threads=$(bash scripts/cpu_quota.sh 8>&-) || threads=8
mkdir -p "$out" "$work/bins" "$(dirname "$log")"
exec >> "$log" 2>&1
say() { echo "[$(date +%T)] $*"; }
die() {
  say "FATAL: $*"
  touch "$out/DONE"
  say "DENSE_REGRESS_DONE rc=1"
  exit 1
}
say "dense-regress start: old=$old session=$session threads=$threads"
[ -s "$gguf" ] || die "no model $gguf"
if [ -f "$out/DONE" ] && [ -s "$out/table.md" ]; then
  say "already done"
  say "DENSE_REGRESS_DONE rc=0"
  exit 0
fi
rm -f "$out/DONE"

# build NAME SRCDIR: step_shapes test binary and its shared libraries into $work/bins/NAME{,.lib}
build() {
  local name=$1 src=$2 exe
  [ -x "$work/bins/$name" ] && { say "build $name: done before"; return 0; }
  say "build $name start ($src)"
  (
    cd "$src" || exit 1
    CARGO_TARGET_DIR="$work/target-$name" timeout -k 30 50m cargo test --release -p tetraeder-scheduler --features cuda \
      --test step_shapes --no-run --message-format=json 8>&- > "$work/build-$name.json" 2> "$work/build-$name.err"
  ) || { tail -n 20 "$work/build-$name.err"; die "build $name failed"; }
  exe=$(grep -o '"executable":"[^"]*step_shapes-[0-9a-f]*"' "$work/build-$name.json" | tail -n1 | cut -d'"' -f4)
  [ -x "$exe" ] || die "build $name: no step_shapes executable in the cargo output"
  cp "$exe" "$work/bins/$name.tmp"
  mkdir -p "$work/bins/$name.lib"
  find "$work/target-$name/release" -maxdepth 2 \( -name '*.so' -o -name '*.so.*' \) -exec cp -a {} "$work/bins/$name.lib/" \;
  find "$work/target-$name/release/build" -maxdepth 4 -path '*tetraeder-ggml-sys-*/out/*' \( -name '*.so' -o -name '*.so.*' \) \
    -exec cp -a {} "$work/bins/$name.lib/" \; 2> /dev/null
  mv "$work/bins/$name.tmp" "$work/bins/$name"
  LD_LIBRARY_PATH="$work/bins/$name.lib" "$work/bins/$name" --list 2>&1 | tail -n 3
  if LD_LIBRARY_PATH="$work/bins/$name.lib" ldd "$work/bins/$name" | grep 'not found'; then
    rm -f "$work/bins/$name"
    die "build $name: missing shared libraries"
  fi
  say "build $name done"
}

# old commit in its own directory, patch queue applied at its own pin
if [ ! -x "$work/bins/old" ]; then
  if [ ! -d "$work/src-old/.git" ]; then
    rm -rf "$work/src-old"
    mkdir -p "$work/src-old"
    git -C "$work/src-old" init -q
    git -C "$work/src-old" fetch -q --depth 1 "$repo_url" "$old" || die "cannot fetch $old from $repo_url"
    git -C "$work/src-old" checkout -q FETCH_HEAD
  fi
  say "old tree: $(git -C "$work/src-old" rev-parse --short HEAD)"
  bash "$work/src-old/patches/apply.sh" 8>&- || die "patch queue of $old failed"
fi
build old "$work/src-old"
build head "$root"

# run_one ROUND NAME ENGINES: one interleaved run; raw output kept, a SHAPE line required
run_one() {
  local round=$1 name=$2 engines=$3 raw="$out/raw-$1-$2.txt"
  if [ -s "$raw" ] && grep -q '^SHAPE' "$raw"; then
    say "round $round $name: done before"
    return 0
  fi
  say "round $round $name engines=$engines"
  TETRAEDER_REQUIRE_GPU=1 TETRAEDER_SHAPES_GGUF="$gguf" TETRAEDER_SHAPES_NGL=99 TETRAEDER_SHAPES_THREADS="$threads" \
    TETRAEDER_SHAPES_ENGINES="$engines" LD_LIBRARY_PATH="$work/bins/$name.lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    timeout -k 30 25m "$work/bins/$name" --nocapture --test-threads 1 > "$raw.tmp" 2>&1 8>&- \
    || { tail -n 30 "$raw.tmp"; die "run $name round $round failed"; }
  grep -q '^SHAPE' "$raw.tmp" || { tail -n 30 "$raw.tmp"; die "run $name round $round printed no SHAPE line"; }
  mv "$raw.tmp" "$raw"
}
for r in $(seq 1 "$rounds"); do
  run_one "$r" old dense,llama
  run_one "$r" head dense,dense_norepack,llama
done

: > "$out/shapes.txt"
for r in $(seq 1 "$rounds"); do
  for n in old head; do grep '^SHAPE' "$out/raw-$r-$n.txt" | sed "s/^/$r $n /" >> "$out/shapes.txt"; done
done
python3 - "$out" "$old" "$rounds" << 'PY' || die "table failed"
import re, sys, collections
out, old, rounds = sys.argv[1], sys.argv[2], sys.argv[3]
d = collections.defaultdict(list)
for l in open(out + '/shapes.txt'):
    m = re.match(r'(\d+) (\S+) SHAPE engine=(\S+) shape=(\S+) ms=([\d.]+)', l)
    if m: d[(m[4], m[3], m[2])].append(float(m[5]))
shapes = sorted({k[0] for k in d})
engines = ['dense', 'dense_norepack', 'llama']
w = [f'Qwen3-8B Q4_K_M, all layers on GPU, min of {rounds} rounds (ms per step shape); old = {old}', '']
w.append('shape | engine | old | head')
w.append('--- | --- | --- | ---')
for s in shapes:
    for e in engines:
        row = [f'{min(d[(s, e, c)]):.1f}' if d[(s, e, c)] else '-' for c in ('old', 'head')]
        if row != ['-', '-']: w.append(f'{s} | {e} | ' + ' | '.join(row))
w += ['', 'dense / llama ratio (min of rounds)', '', 'shape | old | head | head dense_norepack', '--- | --- | --- | ---']
for s in shapes:
    def ratio(e, c):
        a, b = d[(s, e, c)], d[(s, 'llama', c)]
        return f'{min(a) / min(b):.3f}' if a and b else '-'
    w.append(f'{s} | {ratio("dense", "old")} | {ratio("dense", "head")} | {ratio("dense_norepack", "head")}')
open(out + '/table.md', 'w').write('\n'.join(w) + '\n')
print('\n'.join(w))
PY
touch "$out/DONE"
say "DENSE_REGRESS_DONE rc=0"
