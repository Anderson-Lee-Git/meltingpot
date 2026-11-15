#!/bin/bash
#SBATCH --job-name=sb-eval-packed
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:05:00
# Runs eval_ppo.sbatch for array indices FIRST_ID..LAST_ID (inclusive), up to
# SLURM_CPUS_PER_TASK at a time, inside one allocation. Use when the cluster
# only starts a few single-CPU tasks at a time.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
LOGDIR="${OUT}/logs/packed_${SLURM_JOB_ID}"
mkdir -p "${LOGDIR}"
seq "${FIRST_ID}" "${LAST_ID}" | xargs -P "${SLURM_CPUS_PER_TASK}" -I{} \
    bash -c 'SLURM_ARRAY_TASK_ID={} bash examples/spatial_commons/slurm/eval_ppo.sbatch > "$0/task_{}.out" 2>&1' "${LOGDIR}"
