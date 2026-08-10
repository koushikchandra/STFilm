#!/bin/bash
# Mirror paper/ -> the Overleaf project's git remote and push (Overleaf Git bridge).
# Credential: .overleaf/token holds an Overleaf "Git authentication token" (gitignored, never
# committed). Project id in .overleaf/project_id. Usage: ./sync_overleaf.sh ["commit msg"]
set -euo pipefail
cd "$(dirname "$0")"

PID="$(cat .overleaf/project_id 2>/dev/null || true)"
TOKEN="$(cat .overleaf/token 2>/dev/null || true)"
[ -z "$PID" ]   && { echo "ERROR: no .overleaf/project_id"; exit 1; }
[ -z "$TOKEN" ] && { echo "ERROR: no .overleaf/token — create it with your Overleaf git token (see setup notes)"; exit 1; }

REMOTE="https://git:${TOKEN}@git.overleaf.com/${PID}"
MIRROR=.overleaf/mirror

if [ ! -d "$MIRROR/.git" ]; then
  echo "[overleaf] cloning project $PID ..."
  git clone "$REMOTE" "$MIRROR"
else
  git -C "$MIRROR" remote set-url origin "$REMOTE"
  git -C "$MIRROR" pull --no-rebase --no-edit || true
fi

# Non-destructive sync: add/update our sources in the mirror but NEVER delete Overleaf-side files
# (e.g. wacv.cls / ieeenat_fullname.bst uploaded there to compile). Only the local article-class
# build/ copy is explicitly removed. Trade-off: files deleted/renamed in paper/ won't auto-remove
# from Overleaf; delete those by hand if needed.
rm -rf "$MIRROR/build"
rsync -a \
  --exclude='build/' \
  --include='*/' \
  --include='*.tex' --include='*.bib' --include='*.cls' --include='*.sty' --include='*.bst' \
  --include='figs/***' \
  --exclude='*' \
  paper/ "$MIRROR"/

cd "$MIRROR"
git add -A
if git diff --cached --quiet; then
  echo "[overleaf] no changes to sync"; exit 0
fi
git -c user.name="STFiLM sync" -c user.email="noreply@local" \
    commit -q -m "${1:-sync from paper/ $(date -u +%FT%TZ)}"
git push -q origin HEAD 2>&1 | grep -viE "token|https://git:" || true
echo "[overleaf] pushed paper/ -> project $PID"
