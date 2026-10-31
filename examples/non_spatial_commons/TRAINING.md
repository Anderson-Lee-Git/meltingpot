# Unshielded PPO training and frozen-policy evaluation

All training in this experiment uses `CommonsEnvironment(shield=False)`.
There is no shielded-training flag. Requests that exceed the resource available
are still physically rationed by Euclidean projection, exactly as in the fixed
baselines. Metabolic floors and the ecological safety cap are not enforced.

## Environment managed by uv

The repository's `.venv` was recreated with `uv`, using Python 3.13.3. It is
isolated (`include-system-site-packages = false`); the full Lab2D/TensorFlow
stack is not needed. The exact installed training/test/plot dependencies are
recorded in `requirements-training.lock`. The installation uses PyTorch 2.9.1.
For a fresh environment, from the repository root:

```sh
uv venv --python 3.13 .venv
uv pip sync --python .venv/bin/python \
  examples/non_spatial_commons/requirements-training.lock
```

Use the existing `.venv` if it is already installed. A less restrictive install
from the input requirements is also available:

```sh
uv pip install --python .venv/bin/python \
  -r examples/non_spatial_commons/requirements-training.txt \
  -r examples/non_spatial_commons/requirements-dev.txt
```

## Three experiments

1. `individual_maintenance`: agent 0 learns; agents 1–3 always request maintenance.
2. `individual_msy`: agent 0 learns; agents 1–3 always request their MSY share.
3. `cooperative`: one centralized policy outputs all four harvest requests and
   optimizes their total executed harvest. This is a cooperative joint-controller
   baseline, not independently learning actors or MAPPO.

The individual reward is agent 0's executed harvest. The cooperative reward is
total executed harvest. A positive constant rescales each objective for PPO:
individual reward is divided by MSY/N; cooperative reward is divided by MSY.
There is no death penalty, intervention penalty, ecological shaping reward,
fairness bonus, or learned safety critic. Starvation ends the episode and
removes all future returns.

The initialization distribution is R uniform on [20,100] and each E_i uniform
on [2,8], independently, at the default model parameters. This randomization
does not certify subsequent actions. Some starts with fixed MSY partners are
inherently difficult or unrecoverable under those partners: the learner cannot
force their requests to decrease. The depleted state R=13.3, E_i=0 is held out
for post-training evaluation.

## PPO details

- Separate actor and critic networks: two tanh layers of 64 units each.
- Individual input: R/K, all E_i/E_max, and player ID (9 features).
- Cooperative input: R/K and all E_i/E_max (5 features).
- A Gaussian latent action z is transformed to a bounded request K*sigmoid(z).
- Initial mean requests are near maintenance; initial latent standard deviation
  is 0.3. This is an initialization, not a hard-coded harvesting controller.
- PPO's likelihood ratio uses the sampled latent action. The fixed sigmoid
  transformation's Jacobian cancels between old and new policy densities.
- PPO clip 0.2; gamma 0.99; GAE lambda 0.95; Adam learning rate 3e-4,
  linearly annealed; maximum six epochs; minibatches of 256; KL stop at 0.03.
- No entropy bonus. The stochastic policy's standard deviation is learned,
  with log standard deviation restricted to [-3,0.8].
- 16 environments, 128-step rollouts, and 256-round training episode limits.
- Time-limit truncations bootstrap the critic; starvation does not. GAE does
  not cross a reset into the next episode.
- CPU execution with two PyTorch threads and deterministic algorithms enabled.

Each run uses 200,704 transitions (98 rollouts). Three seeds (0,1,2) give nine
runs and 1,806,336 training transitions in total. The final checkpoint is the
network after that fixed budget; there is no shielded validation, best-on-test
selection, or post-evaluation policy update.

```sh
.venv/bin/python -u -m examples.non_spatial_commons.train \
  --total-steps 200704 --seeds 0 1 2 \
  --output examples/results/non_spatial_commons/ppo
```

A smoke run can use `--total-steps 4096 --experiments individual_maintenance
--seeds 0 --output examples/results/non_spatial_commons/ppo_smoke`.
The requested budget is rounded up to a complete rollout batch.

For every run, the trainer saves `initial.pt`, `final.pt`, `metadata.json`, and
`training.csv`. Checkpoints contain network weights and provenance for
inference, not optimizer state for exact resumption. `training_manifest.json`
is written only after every requested training run completes.

## Evaluate only after all training completes

```sh
.venv/bin/python -u -m examples.non_spatial_commons.evaluate_trained \
  --input examples/results/non_spatial_commons/ppo \
  --steps 1000 --random-starts 10
```

This loads and freezes all final policies, then evaluates both deterministic
requests (sigmoid of the latent mean) and stochastic requests. These need not
have the same performance. For each action-selection mode, the policy is run
with and without shielding from identical initial states and matching random
streams. Joint stochastic actions are sampled once per round, then shared
consistently across the four agent calls.

The evaluation includes the default state, the depleted state, and ten held-out
random starts in the certified core, generated with seed 532. Held-out energies
are uniform on [0,E_max]; resource is uniform between the protected core boundary
plus 0.01 and K. These are broader than the training initialization distribution.

There are 9 checkpoints x 12 starts x 2 action-selection modes x 2 shielding
modes = 432 post-training evaluation episodes. Surviving episodes run for 1,000
rounds; starvation terminates earlier. The same positive harvest rewards are
used in both modes. No weights change during evaluation.

Each run has an `evaluation/summary.json` and `evaluation/trajectories.csv`.
The sweep has `evaluation_summary.json`. The per-round trace records proposals,
executed harvests, all energies, resource, corrections, and the core margin.
As in the fixed baselines, horizon-normalized harvest counts the remaining
rounds after failure as zero; it is not a survivor-only average.

```sh
.venv/bin/python -m examples.non_spatial_commons.plot_training \
  --input examples/results/non_spatial_commons/ppo
```

The training chart shows the mean and range over seeds, not confidence intervals.
Episode statistics are the most recent 100 completed episodes; return magnitudes
are not directly comparable between individual and cooperative objectives.

## Tests

```sh
.venv/bin/python -m pytest -n 0 examples/non_spatial_commons -q
.venv/bin/python -m pyink --check examples/non_spatial_commons
```

Additional tests cover latent-action probabilities, time-limit versus failure
bootstrapping, unshielded-only training pools, checkpoint round trips, frozen
weights during evaluation, shared joint stochastic actions, and requiring the
completed training manifest before sweep evaluation.

## Longer Cooperative Episodes

The trainer exposes `--episode-length` (default 256). To isolate the effect of
matching training episodes to the 1,000-round evaluation horizon, train fresh
cooperative policies with the original transition budget and seeds:

```sh
uv run --no-project --python .venv/bin/python python -u -m examples.non_spatial_commons.train \
  --experiments cooperative --seeds 0 1 2 --total-steps 200704 \
  --episode-length 1000 \
  --output examples/results/non_spatial_commons/ppo_cooperative_h1000
uv run --no-project --python .venv/bin/python python -u -m examples.non_spatial_commons.evaluate_trained \
  --input examples/results/non_spatial_commons/ppo_cooperative_h1000 \
  --steps 1000 --random-starts 10
```

These runs remain entirely unshielded during training. Evaluation begins after
all three runs finish, uses the frozen final checkpoints, and applies the same
starts and action-noise seeds as the 256-round baseline. The initialization,
reward, PPO hyperparameters, and total transitions per seed are unchanged.
Longer episodes produce fewer resets and completed episodes at the same budget.
Recent-100-episode training statistics and raw episode returns therefore do not
have the same interpretation across horizons; use matched frozen-policy
evaluation to compare performance. Existing baseline artifacts remain under
`examples/results/non_spatial_commons/ppo`.
