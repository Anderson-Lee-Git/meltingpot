#!/bin/bash
#SBATCH --job-name=sb-eval-ppo
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
# Array: one task per (PPO method, seed, start chunk); 5 x 5 x 4 = 100 tasks
# (0-99). ~5-25 min each. Frozen checkpoints, stochastic actions, filter off and on.
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/examples/spatial_commons/slurm/common.sh"
cd "${REPO}"
N=${#SEEDS[@]}
PER_METHOD=$(( N * CHUNKS ))
M=$(( SLURM_ARRAY_TASK_ID / PER_METHOD ))
REST=$(( SLURM_ARRAY_TASK_ID % PER_METHOD ))
SEED=${SEEDS[$(( REST / CHUNKS ))]}
CHUNK=$(( REST % CHUNKS ))
METHOD=${PPO_METHODS[$M]}
EXP=${PPO_EXPERIMENTS[$M]}
read -r FIRST LAST <<< "$(chunk_range "${CHUNK}")"
"${PY}" -m examples.spatial_commons.baselines evaluate --method "${METHOD}" \
    --checkpoint "${OUT}/ppo/${EXP}/seed_${SEED}/final.pt" \
    --starts "${OUT}/starts.json" --first-start "${FIRST}" --last-start "${LAST}" \
    --steps "${STEPS}" \
    --output "${OUT}/episodes/episodes_${METHOD}_s${SEED}_c${CHUNK}.csv"
