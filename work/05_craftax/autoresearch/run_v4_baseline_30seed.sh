#!/usr/bin/env bash
# run_v4_baseline_30seed.sh — atomically run v4 baseline 30-seed validation.
#
# Required for Gap 2 (PAPER_HANDOFF): paired Wilcoxon test needs per-seed
# data for v4 with the SAME seed bank (42-71) as exp17 30-seed.
#
# Procedure:
#   1. Stash current fmc_mutable.py (= exp17 final state from commit 00b7f71)
#   2. Write v4 baseline version (from commit 28f33a4) to fmc_mutable.py
#   3. Run evaluate_30seed.py with output → results/v4_30seed.json
#   4. Restore fmc_mutable.py to the exp17 state (git checkout from 00b7f71)
#
# Idempotency: safe to re-run; restores fmc_mutable.py unconditionally on exit.

set -euo pipefail

cd "$(dirname "$0")"
HERE="$(pwd)"
PY="${PY:-/Users/vladvrinceanu/.pyenv/versions/3.11.7/bin/python}"
V4_COMMIT="28f33a4"
EXP17_COMMIT="00b7f71"
STASH_FILE="/tmp/exp17_fmc_mutable.$$.py"

cleanup() {
    echo "[v4-baseline] restoring exp17 fmc_mutable.py..."
    if [[ -f "$STASH_FILE" ]]; then
        cp "$STASH_FILE" "$HERE/fmc_mutable.py"
        rm -f "$STASH_FILE"
    else
        # Fallback: restore from git (commit 00b7f71)
        cd "$HERE/../../.."
        git checkout "$EXP17_COMMIT" -- work/05_craftax/autoresearch/fmc_mutable.py
        cd "$HERE"
    fi
    # Verify restored content has exp17 marker
    if grep -q "MAKE_IRON_PICKAXE \*\*\* BLOCKER (exp17 final)" "$HERE/fmc_mutable.py"; then
        echo "[v4-baseline] exp17 state confirmed restored."
    else
        echo "[v4-baseline] WARNING: exp17 marker NOT found after restore!" >&2
        exit 99
    fi
}
trap cleanup EXIT INT TERM

echo "[v4-baseline] step 1: stash current (exp17) fmc_mutable.py"
cp "$HERE/fmc_mutable.py" "$STASH_FILE"

echo "[v4-baseline] step 2: load v4 baseline from commit $V4_COMMIT"
cd "$HERE/../../.."
git show "${V4_COMMIT}:work/05_craftax/autoresearch/fmc_mutable.py" \
    > "$HERE/fmc_mutable.py"
cd "$HERE"

# Sanity: v4 should NOT contain exp17 markers
if grep -q "MAKE_IRON_PICKAXE \*\*\* BLOCKER (exp17 final)" "$HERE/fmc_mutable.py"; then
    echo "[v4-baseline] ERROR: v4 file still contains exp17 marker; aborting." >&2
    exit 1
fi
echo "[v4-baseline] v4 baseline file loaded ($(wc -l < "$HERE/fmc_mutable.py") lines)"

echo "[v4-baseline] step 3: run 30-seed evaluation"
JAX_PLATFORMS=cpu "$PY" evaluate_30seed.py \
    --out_json results/v4_30seed.json \
    --n_seeds 30 \
    --seed_start 42 \
    --wall_budget_s 7200 \
    --description "v4_baseline_30seed_for_gap2"

echo "[v4-baseline] step 4: cleanup will restore exp17 state via trap"
echo "[v4-baseline] DONE — output at results/v4_30seed.json"
