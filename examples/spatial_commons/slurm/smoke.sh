#!/bin/bash
# End-to-end smoke test on CPU (a few minutes; fine on a login node):
# 4 starts, 300-round evaluation episodes, PPO trained for 8,192 steps with
# seeds 0 and 1 (exercises the seed-averaged CIs).
#   bash examples/spatial_commons/slurm/smoke.sh [output_dir]
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "${REPO}"
OUT=${1:-examples/results/spatial_commons/baselines_smoke}
STEPS=300
rm -rf "${OUT}"
mkdir -p "${OUT}/episodes"
"${PY}" -m examples.spatial_commons.baselines starts --count 4 \
    --seed "${START_SEED}" --min-energy "${MIN_ENERGY}" \
    --output "${OUT}/starts.json"
for EXP in "${PPO_EXPERIMENTS[@]}"; do
  "${PY}" -m examples.spatial_commons.train --experiments "${EXP}" --seeds 0 1 \
      --total-steps 8192 --num-envs 8 --rollout-steps 64 --starts random \
      --start-min-energy "${MIN_ENERGY}" --episode-length "${TRAIN_EPISODE_LENGTH}" \
      --output "${OUT}/ppo" --no-manifest | tail -1
done
for METHOD in "${FIXED_METHODS[@]}"; do
  "${PY}" -m examples.spatial_commons.baselines evaluate --method "${METHOD}" \
      --starts "${OUT}/starts.json" --steps "${STEPS}" \
      --output "${OUT}/episodes/episodes_${METHOD}.csv"
done
for i in "${!PPO_METHODS[@]}"; do
  for SEED in 0 1; do
    "${PY}" -m examples.spatial_commons.baselines evaluate \
        --method "${PPO_METHODS[$i]}" \
        --checkpoint "${OUT}/ppo/${PPO_EXPERIMENTS[$i]}/seed_${SEED}/final.pt" \
        --starts "${OUT}/starts.json" --steps "${STEPS}" \
        --output "${OUT}/episodes/episodes_${PPO_METHODS[$i]}_s${SEED}.csv"
  done
done
"${PY}" -m examples.spatial_commons.baselines summarize \
    --input "${OUT}/episodes" --output "${OUT}"
