# Shared settings for the baseline comparison. Sourced by every script here.
# Override any of these in the environment before calling submit.sh.
REPO=${REPO:-/scratch/gpfs/KOROLOVA/cl6486/meltingpot}
OUT=${OUT:-runs/2026-11-06-spatial-baselines}
PY="${REPO}/.venv/bin/python"
export PYTHONPATH="${REPO}"
NUM_STARTS=${NUM_STARTS:-100}        # Shared evaluation starts.
START_SEED=${START_SEED:-532}
CHUNKS=${CHUNKS:-4}                   # Start chunks per evaluated policy.
STEPS=${STEPS:-1000}                  # Evaluation horizon (rounds).
SEEDS=(${SEEDS:-0 1 2 3 4})           # PPO training seeds.
TOTAL_STEPS=${TOTAL_STEPS:-2000896}   # PPO environment steps per run.
TRAIN_STARTS=${TRAIN_STARTS:-random}  # default | random
TRAIN_SHIELD_FLAG=${TRAIN_SHIELD_FLAG:-}  # set to --train-shield to enable
TRAIN_EPISODE_LENGTH=${TRAIN_EPISODE_LENGTH:-1000}  # = evaluation horizon
MIN_ENERGY=${MIN_ENERGY:-15}          # Starts: E_i uniform in {MIN_ENERGY..E_max}
PPO_METHODS=(ippo_shared ippo_shared_team ippo_hetero ippo_hetero_team central_ppo)
PPO_EXPERIMENTS=(selfish_shared cooperative selfish_independent cooperative_independent centralized)
FIXED_METHODS=(random greedy)

chunk_range() {  # chunk index -> "first last"
  local size=$(( (NUM_STARTS + CHUNKS - 1) / CHUNKS ))
  local first=$(( $1 * size ))
  local last=$(( first + size ))
  (( last > NUM_STARTS )) && last=${NUM_STARTS}
  echo "${first} ${last}"
}
