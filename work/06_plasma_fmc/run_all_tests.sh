#!/usr/bin/env bash
# run_all_tests.sh — runs the full test suite (M2-M13) and reports pass/fail.
# Usage: bash run_all_tests.sh

set -u
cd "$(dirname "$0")"

TESTS=(
    "tests/test_simulator.py"
    "tests/test_fmc.py"
    "tests/test_policy.py"
    "tests/test_dagger.py"
    "tests/test_fmc_jax.py"
    "tests/test_dagger_jax.py"
    "tests/test_freegs_truth.py"
    "tests/test_calibrated.py"
    "tests/test_shape_surrogate.py"
    "tests/test_nn_sim.py"
    "tests/test_oracle.py"
)

total_pass=0
total_fail=0
failures=()

echo "================================================================="
echo "FMC TCV Plasma Control — full test suite (M2-M13)"
echo "================================================================="

for t in "${TESTS[@]}"; do
    echo ""
    echo "--- $t ---"
    out=$(python "$t" 2>&1 | tail -3)
    echo "$out"
    # Parse "N passed, M failed"
    line=$(echo "$out" | grep -E "passed.*failed" | tail -1)
    if [[ -n "$line" ]]; then
        n_pass=$(echo "$line" | grep -oE "^[0-9]+" | head -1)
        n_fail=$(echo "$line" | grep -oE "[0-9]+ failed" | grep -oE "^[0-9]+")
        total_pass=$((total_pass + n_pass))
        total_fail=$((total_fail + n_fail))
        if [[ "$n_fail" != "0" ]]; then
            failures+=("$t: $n_fail failures")
        fi
    fi
done

echo ""
echo "================================================================="
echo "TOTAL: $total_pass passed, $total_fail failed"
if [[ ${#failures[@]} -gt 0 ]]; then
    echo ""
    echo "Failures by file:"
    for f in "${failures[@]}"; do
        echo "  - $f"
    done
fi
echo "================================================================="

[[ "$total_fail" == "0" ]] && exit 0 || exit 1
