#!/bin/bash
# setup_smoke_repo.sh — create a fresh smoke test repo from the fixture.
#
# Usage:
#   ./setup_smoke_repo.sh [destination]
#
# Default destination: /tmp/fmc-smoke-repo
# Wipes any existing repo at the destination, copies fixture files,
# initializes git with a single "scaffold" commit.

set -euo pipefail

DEST="${1:-/tmp/fmc-smoke-repo}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIXTURE="$SCRIPT_DIR/fixture"

if [ ! -d "$FIXTURE" ]; then
    echo "ERROR: fixture not found at $FIXTURE" >&2
    exit 1
fi

if [ -d "$DEST" ]; then
    echo "Removing existing $DEST"
    rm -rf "$DEST"
fi

echo "Creating fresh smoke repo at $DEST"
mkdir -p "$DEST"
cp -r "$FIXTURE/." "$DEST/"
mv "$DEST/REPO_README.md" "$DEST/README.md"

cd "$DEST"
git init -q -b main

# Set git identity at the REPO level (lives in .git/config, shared by all worktrees).
# This fixes the Sev-2 bug found in the first smoke run where walker worktrees
# had no committer configured and had to use -c user.email/user.name inline.
git config user.email "fmc-smoke@local"
git config user.name "FMC Smoke"

git add .
git commit -q -m "initial scaffold: empty fizzbuzz + 5 acceptance tests"

INIT_HEAD=$(git rev-parse HEAD)

cat <<EOF

✓ Smoke repo ready at $DEST

  initial commit: $INIT_HEAD
  files staged:
$(git ls-files | sed 's/^/    /')

Next:
  1. cd $DEST
  2. (in a NEW Claude Code session, with the fractal-coding-loop plugin installed)
  3. paste the prompt from $SCRIPT_DIR/PROMPT.txt into Claude Code
  4. while it runs, monitor via the checklist in $SCRIPT_DIR/OBSERVATION_CHECKLIST.md
  5. after it completes, run: pytest -v
  6. fill in $SCRIPT_DIR/RUN_LOG.md with what you observed
EOF
