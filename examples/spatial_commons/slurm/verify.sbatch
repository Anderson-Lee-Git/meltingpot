#!/bin/bash
#SBATCH --job-name=sb-verify
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=runs/2026-11-06-spatial-verify/%x-%j.out
# Full-scale (x5) numerical safety checks of Section 3. By default only the
# closed-loop filter check, which timed out on the login node; override with
# CHECKS="core rollout backup filter".
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/examples/spatial_commons/slurm/common.sh"
cd "${REPO}"
CHECKS=${CHECKS:-filter}
"${PY}" -m examples.spatial_commons.verify_safety --seed 0 --only ${CHECKS}
