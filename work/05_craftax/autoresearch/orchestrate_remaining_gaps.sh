#!/usr/bin/env bash
# orchestrate_remaining_gaps.sh — runs after Gap 1 (exp17 30-seed) finishes.
#
# Sequence (each step is CPU-bound, run in series; estimated total ~5 hours):
#   1. v4 baseline 30-seed      ~45 min  → results/v4_30seed.json
#   2. Gap 3 ablations L1..L5  ~5 × 45 min = ~3.75 h → results/gap3_*.json
#   3. Gap 2 statistical test   ~30 s   → results/statistical_validation.json
#
# Usage (after Gap 1 completes):
#   nohup bash orchestrate_remaining_gaps.sh > orchestrate.log 2>&1 &

set -uo pipefail   # NOT -e: continue on individual failures

cd "$(dirname "$0")"
HERE="$(pwd)"
PY="${PY:-/Users/vladvrinceanu/.pyenv/versions/3.11.7/bin/python}"
LOG_DIR="$HERE"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[orchestrate] === starting at $(ts) ==="

# Wait for Gap 1 to finish if still running
wait_for_gap1() {
    local pid_file="/tmp/gap1_exp17.pid"
    while pgrep -f "evaluate_30seed.py.*exp17_30seed.json" > /dev/null; do
        echo "[orchestrate] Gap 1 still running, waiting 60 s..."
        sleep 60
    done
    if [[ -f "$HERE/results/exp17_30seed.json" ]]; then
        echo "[orchestrate] Gap 1 output found; proceeding."
        return 0
    else
        echo "[orchestrate] ERROR: Gap 1 finished but exp17_30seed.json missing." >&2
        return 1
    fi
}

wait_for_gap1 || exit 1

# ----- Step 1: v4 baseline 30-seed -----
echo "[orchestrate] === step 1: v4 baseline 30-seed @ $(ts) ==="
bash "$HERE/run_v4_baseline_30seed.sh" \
    > "$LOG_DIR/v4_baseline_30seed.log" 2>&1
v4_status=$?
if [[ $v4_status -ne 0 ]]; then
    echo "[orchestrate] WARNING: v4 baseline returned exit $v4_status" >&2
fi

# ----- Step 2: Gap 3 ablations -----
echo "[orchestrate] === step 2: Gap 3 ablations @ $(ts) ==="
JAX_PLATFORMS=cpu "$PY" gap3_ablations.py \
    --n_seeds 30 \
    --wall_budget_s 7200 \
    > "$LOG_DIR/gap3_ablations.log" 2>&1
gap3_status=$?
if [[ $gap3_status -ne 0 ]]; then
    echo "[orchestrate] WARNING: Gap 3 returned exit $gap3_status" >&2
fi

# ----- Step 3: Gap 2 statistical test -----
echo "[orchestrate] === step 3: Gap 2 statistical test @ $(ts) ==="
if [[ -f "$HERE/results/v4_30seed.json" && -f "$HERE/results/exp17_30seed.json" ]]; then
    JAX_PLATFORMS=cpu "$PY" gap2_statistical_test.py \
        --exp17_json results/exp17_30seed.json \
        --v4_json    results/v4_30seed.json \
        --out_json   results/statistical_validation.json \
        > "$LOG_DIR/gap2_test.log" 2>&1
    gap2_status=$?
    if [[ $gap2_status -ne 0 ]]; then
        echo "[orchestrate] WARNING: Gap 2 returned exit $gap2_status" >&2
    fi
else
    echo "[orchestrate] SKIP: Gap 2 (missing input JSONs)" >&2
fi

echo "[orchestrate] === done at $(ts) ==="
echo "[orchestrate] outputs:"
ls -la "$HERE/results/"
