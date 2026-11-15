#!/bin/bash
#SBATCH --job-name=sb-summarize
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/examples/spatial_commons/slurm/common.sh"
cd "${REPO}"
"${PY}" -m examples.spatial_commons.baselines summarize \
    --input "${OUT}/episodes" --output "${OUT}"
