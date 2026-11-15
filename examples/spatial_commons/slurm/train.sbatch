#!/bin/bash
#SBATCH --job-name=sb-train
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00
# Array: one task per (PPO experiment, seed); 5 x 5 = 25 tasks (0-24).
# 1000-round training episodes. Expected ~10-55 min per task at 2M steps
# (centralized fastest, the two 4-network experiments slowest).
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/examples/spatial_commons/slurm/common.sh"
cd "${REPO}"
N=${#SEEDS[@]}
EXP=${PPO_EXPERIMENTS[$(( SLURM_ARRAY_TASK_ID / N ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % N ))]}
"${PY}" -m examples.spatial_commons.train --experiments "${EXP}" \
    --seeds "${SEED}" --total-steps "${TOTAL_STEPS}" --starts "${TRAIN_STARTS}" \
    --start-min-energy "${MIN_ENERGY}" --episode-length "${TRAIN_EPISODE_LENGTH}" \
    ${TRAIN_SHIELD_FLAG} --output "${OUT}/ppo" --no-manifest
