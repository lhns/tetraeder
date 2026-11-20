#!/usr/bin/env bash
# Worktree cleanup (AGENTS.md, "Local files"). Lists every linked worktree of this repo and removes the ones whose
# HEAD is on a remote branch and that have no local changes. Others are kept unless --archive: then their unpushed
# commits go into a bundle and their local changes into a tarball under runs/archive/worktrees-<date>/ (git-ignored),
# and they are removed too.
#   bash scripts/wt_gc.sh            dry run: print what would happen
#   bash scripts/wt_gc.sh --apply    remove pushed, clean worktrees
#   bash scripts/wt_gc.sh --apply --archive   also archive and remove the rest
# A worktree an agent is still using must not be touched: run it when no worker is active.
set -u
apply=0 archive=0
for a in "$@"; do
  case $a in --apply) apply=1 ;; --archive) archive=1 ;; *) echo "unknown arg $a" >&2; exit 2 ;; esac
done
root=$(git rev-parse --show-toplevel) || exit 1
cd "$root" || exit 1
git fetch -q origin
dest="runs/archive/worktrees-$(date +%Y%m%d)"
main=$(git worktree list --porcelain | awk '/^worktree /{print $2; exit}')
removed=0 kept=0
while read -r w; do
  [ "$w" = "$main" ] && continue
  [ -d "$w" ] || continue
  head=$(git -C "$w" rev-parse HEAD 2>/dev/null) || continue
  # the vendored ggml gitlink shows as modified in every patched checkout; that is not a local change
  changes=$(git -C "$w" status --porcelain | grep -v ' vendor/llama.cpp$\|stackdump$')
  pushed=$([ -n "$(git branch -r --contains "$head" 2>/dev/null | head -1)" ] && echo 1 || echo 0)
  if [ "$pushed" = 1 ] && [ -z "$changes" ]; then
    echo "remove  $w"
  elif [ "$archive" = 1 ]; then
    echo "archive $w (pushed=$pushed, $(echo "$changes" | grep -c .) local changes)"
    if [ "$apply" = 1 ]; then
      mkdir -p "$dest"
      n=$(basename "$w")
      [ "$pushed" = 1 ] || git update-ref "refs/archive/$n" "$head"
      [ -z "$changes" ] || (cd "$w" && echo "$changes" | cut -c4- | tar -czf - -T -) > "$dest/$n-local.tgz"
      echo "$n $head pushed=$pushed" >> "$dest/INDEX.txt"
    fi
  else
    echo "keep    $w (pushed=$pushed, $(echo "$changes" | grep -c .) local changes)"
    kept=$((kept + 1))
    continue
  fi
  if [ "$apply" = 1 ]; then
    if [ "$archive" = 1 ] && [ -n "$(git for-each-ref refs/archive)" ]; then
      git bundle create "$dest/unpushed.bundle" $(git for-each-ref --format='%(refname)' refs/archive) --not --remotes -q 2> /dev/null
    fi
    git worktree remove --force --force "$w" 2> /dev/null || rm -rf "$w"
    removed=$((removed + 1))
  fi
done < <(git worktree list --porcelain | awk '/^worktree /{print $2}')
[ "$apply" = 1 ] && git worktree prune
echo "removed $removed, kept $kept$([ "$apply" = 1 ] || echo ' (dry run)')"
