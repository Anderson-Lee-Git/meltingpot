#!/bin/bash
# Full baseline comparison. Run from anywhere on a login node:
#   bash examples/spatial_commons/slurm/submit.sh
# Samples the shared starts here (seconds, no network needed), then submits
# training, the two evaluation arrays and the summary with dependencies.
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "${REPO}"
mkdir -p "${OUT}/logs" "${OUT}/episodes"
"${PY}" -m examples.spatial_commons.baselines starts --count "${NUM_STARTS}" \
    --seed "${START_SEED}" --min-energy "${MIN_ENERGY}" \
    --output "${OUT}/starts.json"
LOG="${OUT}/logs/%x_%A_%a.out"
NT=$(( ${#PPO_EXPERIMENTS[@]} * ${#SEEDS[@]} ))
NF=$(( ${#FIXED_METHODS[@]} * CHUNKS ))
NP=$(( ${#PPO_METHODS[@]} * ${#SEEDS[@]} * CHUNKS ))
TRAIN=$(sbatch --parsable --array=0-$(( NT - 1 )) --output="${LOG}" \
    examples/spatial_commons/slurm/train.sbatch)
FIXED=$(sbatch --parsable --array=0-$(( NF - 1 )) --output="${LOG}" \
    examples/spatial_commons/slurm/eval_fixed.sbatch)
PPO=$(sbatch --parsable --array=0-$(( NP - 1 )) --output="${LOG}" \
    --dependency=afterok:${TRAIN} examples/spatial_commons/slurm/eval_ppo.sbatch)
SUM=$(sbatch --parsable --output="${OUT}/logs/%x_%j.out" \
    --dependency=afterok:${FIXED}:${PPO} \
    examples/spatial_commons/slurm/summarize.sbatch)
echo "train=${TRAIN} eval_fixed=${FIXED} eval_ppo=${PPO} summarize=${SUM}"
echo "Results: ${OUT}/table.md, table.tex, summary.{csv,json}, episodes.csv"
