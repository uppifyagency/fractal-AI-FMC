#!/usr/bin/env bash
# run_all.sh — orchestrazione 5 giochi × 5 seed = 25 episodi
# Tempo stimato: 4-6h CPU, 1-2h GPU
set -e
cd "$(dirname "$0")"

GAMES=(boxing ms_pacman asteroids centipede montezuma_revenge)
SEEDS=(42 137 271 314 1729)

mkdir -p ../results

for game in "${GAMES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    out="../results/${game}_seed${seed}.json"
    if [[ -f "$out" ]]; then
      echo "[SKIP] $out già esistente"
      continue
    fi
    echo "[RUN ] $game seed=$seed"
    python run_single.py \
      --config "../configs/${game}.yaml" \
      --seed "$seed" \
      --output "$out" \
      || echo "[FAIL] $game seed=$seed"
  done
done

echo ""
echo "Done. Aggregazione → python aggregate_results.py"
