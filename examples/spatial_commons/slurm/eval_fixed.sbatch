#!/bin/bash
#SBATCH --job-name=sb-eval-fixed
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
# Array: one task per (fixed method, start chunk); 2 x 4 = 8 tasks (0-7).
# Greedy with the filter on is the slow case: ~25-50 ms per round, i.e.
# up to ~25 min for a chunk of 25 starts x 1000 rounds. Random: ~5 min.
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/examples/spatial_commons/slurm/common.sh"
cd "${REPO}"
METHOD=${FIXED_METHODS[$(( SLURM_ARRAY_TASK_ID / CHUNKS ))]}
CHUNK=$(( SLURM_ARRAY_TASK_ID % CHUNKS ))
read -r FIRST LAST <<< "$(chunk_range "${CHUNK}")"
"${PY}" -m examples.spatial_commons.baselines evaluate --method "${METHOD}" \
    --starts "${OUT}/starts.json" --first-start "${FIRST}" --last-start "${LAST}" \
    --steps "${STEPS}" --output "${OUT}/episodes/episodes_${METHOD}_c${CHUNK}.csv"
